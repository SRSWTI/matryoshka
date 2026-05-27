from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from cradle.embeddings import DEFAULT_QUERY_TASK, TextEmbedder, build_text_embedder, format_query_text
from cradle.graph_models import HierarchicalSearchResult, TraversalCandidate, TraversalStep
from cradle.result_loader import SQLiteResultLoader
from cradle.retrieval import build_query_plan
from cradle.semantic_index import SemanticIndexStore


@dataclass(slots=True)
class HierarchySearchConfig:
    index_dir: Path | None = None
    branch_width: int = 3
    file_limit: int = 5
    symbol_limit: int = 5
    max_depth: int = 6
    default_query_task: str = DEFAULT_QUERY_TASK


class AxeHierarchySearcher:
    def __init__(
        self,
        db_path: str | Path,
        *,
        index_dir: str | Path | None = None,
        embedder: TextEmbedder | None = None,
        config: HierarchySearchConfig | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._config = config or HierarchySearchConfig(index_dir=Path(index_dir) if index_dir else None)
        self._index = SemanticIndexStore(self._db_path, index_dir=self._config.index_dir)
        self._loader = SQLiteResultLoader(self._db_path)
        self._embedder = embedder or build_text_embedder(self._index.model_name, truncate_dim=self._index.dimension)
        if self._embedder.dimension != self._index.dimension:
            raise ValueError(
                f"Hierarchy embedder dimension {self._embedder.dimension} does not match index dimension {self._index.dimension}"
            )

    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        branch_width: int | None = None,
        symbol_limit: int | None = None,
        task: str | None = None,
    ) -> HierarchicalSearchResult:
        plan = build_query_plan(query)
        node_limit = limit or self._config.file_limit
        resolved_branch_width = branch_width or self._config.branch_width
        resolved_symbol_limit = symbol_limit or self._config.symbol_limit
        task_name = task or self._config.default_query_task
        query_vector = self._embedder.encode([format_query_text(query, task=task_name)])[0]

        with self._loader.connect() as conn:
            node_rows = {
                row["node_id"]: row
                for row in conn.execute(
                    "SELECT node_id, parent_id, kind, path, name, summary, description, primary_category, symbol_count FROM nodes ORDER BY path"
                ).fetchall()
            }
            children_by_parent = _children_by_parent(node_rows)
            symbol_ids_by_node = _symbol_ids_by_node(conn)

            root_ids = sorted(node_id for node_id, row in node_rows.items() if row["parent_id"] is None)
            steps = self._traverse_steps(conn, node_rows, children_by_parent, query_vector, plan, root_ids, resolved_branch_width)
            file_candidate_ids = self._collect_file_candidates(node_rows, children_by_parent, steps)
            if not file_candidate_ids:
                file_candidate_ids = [node_id for node_id, row in node_rows.items() if row["kind"] == "file"]

            node_hits = self._load_ranked_node_hits(conn, node_rows, query_vector, plan, file_candidate_ids, node_limit)
            symbol_candidate_ids = [symbol_id for hit in node_hits for symbol_id in symbol_ids_by_node.get(hit.node.node_id, [])]
            symbol_hits = self._load_ranked_symbol_hits(conn, query_vector, plan, symbol_candidate_ids, resolved_symbol_limit)

        return HierarchicalSearchResult(query=query, steps=steps, node_hits=node_hits, symbol_hits=symbol_hits)

    def _traverse_steps(
        self,
        conn: sqlite3.Connection,
        node_rows: dict[str, sqlite3.Row],
        children_by_parent: dict[str, list[str]],
        query_vector,
        plan,
        root_ids: list[str],
        branch_width: int,
    ) -> list[TraversalStep]:
        steps: list[TraversalStep] = []
        frontier = root_ids
        depth = 0
        while frontier and depth < self._config.max_depth:
            child_ids = []
            for parent_id in frontier:
                child_ids.extend(children_by_parent.get(parent_id, []))
            if not child_ids:
                break

            structural_child_ids = [node_id for node_id in child_ids if node_rows[node_id]["kind"] in {"repo", "folder"}]
            file_child_ids = [node_id for node_id in child_ids if node_rows[node_id]["kind"] == "file"]
            if structural_child_ids and file_child_ids:
                structural_scores = self._score_node_subset(node_rows, query_vector, plan, structural_child_ids, prefer_structure=True)
                file_scores = self._score_node_subset(node_rows, query_vector, plan, file_child_ids, prefer_structure=False)
                best_structural = max(structural_scores.values(), default=0.0)
                best_file = max(file_scores.values(), default=0.0)
                best_structural_token_hits = max((_path_name_token_hits(plan, node_rows[node_id]) for node_id in structural_child_ids), default=0)
                best_file_token_hits = max((_path_name_token_hits(plan, node_rows[node_id]) for node_id in file_child_ids), default=0)
                allow_file_children = best_file >= best_structural + 12.0 or best_file_token_hits > best_structural_token_hits
                scored = structural_scores if not allow_file_children else {**structural_scores, **file_scores}
            elif structural_child_ids:
                scored = self._score_node_subset(node_rows, query_vector, plan, structural_child_ids, prefer_structure=True)
            else:
                scored = self._score_node_subset(node_rows, query_vector, plan, child_ids, prefer_structure=False)
            selected = _top_pairs(scored, branch_width)
            if structural_child_ids and file_child_ids and allow_file_children:
                best_structural_pair = _top_pairs(structural_scores, 1)
                if best_structural_pair and all(node_rows[node_id]["kind"] == "file" for node_id, _ in selected):
                    selected = [best_structural_pair[0], *selected[: max(0, branch_width - 1)]]
                    deduped: list[tuple[str, float]] = []
                    seen_selected: set[str] = set()
                    for node_id, score in selected:
                        if node_id in seen_selected:
                            continue
                        seen_selected.add(node_id)
                        deduped.append((node_id, score))
                    selected = deduped[:branch_width]
            if not selected:
                break

            candidates = [TraversalCandidate(score=score, node=self._loader.load_node(conn, node_id)) for node_id, score in selected]
            level = "file" if all(node_rows[node_id]["kind"] == "file" for node_id, _ in selected) else "branch"
            steps.append(TraversalStep(level=level, parent_node_ids=list(frontier), candidates=candidates))

            next_frontier = [node_id for node_id, _ in selected if node_rows[node_id]["kind"] in {"repo", "folder"}]
            if not next_frontier:
                break
            frontier = next_frontier
            depth += 1
        return steps

    def _collect_file_candidates(
        self,
        node_rows: dict[str, sqlite3.Row],
        children_by_parent: dict[str, list[str]],
        steps: list[TraversalStep],
    ) -> list[str]:
        if not steps:
            return []

        file_ids: list[str] = []
        for step in steps:
            for candidate in step.candidates:
                if candidate.node.kind == "file":
                    file_ids.append(candidate.node.node_id)

        for candidate in steps[-1].candidates:
            if candidate.node.kind != "file":
                file_ids.extend(_descendant_file_ids(candidate.node.node_id, children_by_parent, node_rows))
        return list(dict.fromkeys(file_ids))

    def _load_ranked_node_hits(self, conn: sqlite3.Connection, node_rows, query_vector, plan, node_ids: list[str], limit: int):
        scores = self._score_node_subset(node_rows, query_vector, plan, node_ids, prefer_structure=False)
        return [self._loader.load_node_hit(conn, node_id, score) for node_id, score in _top_pairs(scores, limit)]

    def _load_ranked_symbol_hits(self, conn: sqlite3.Connection, query_vector, plan, symbol_ids: list[str], limit: int):
        if not symbol_ids:
            return []

        rows = {
            row["symbol_id"]: row
            for row in conn.execute(
                "SELECT symbol_id, name, qualified_name, normalized_name, path, signature FROM symbols ORDER BY qualified_name"
            ).fetchall()
            if row["symbol_id"] in set(symbol_ids)
        }
        semantic_scores = {symbol_id: score * 100.0 for symbol_id, score in self._index.search_symbol_subset(query_vector, symbol_ids, top_k=max(limit * 6, len(rows)))}
        for symbol_id, row in rows.items():
            semantic_scores[symbol_id] = semantic_scores.get(symbol_id, 0.0) + _symbol_bonus(plan, row)
        return [self._loader.load_symbol_hit(conn, symbol_id, score) for symbol_id, score in _top_pairs(semantic_scores, limit)]

    def _score_node_subset(self, node_rows, query_vector, plan, node_ids: list[str], *, prefer_structure: bool) -> dict[str, float]:
        semantic_scores = {
            node_id: score * 100.0
            for node_id, score in self._index.search_node_subset(query_vector, node_ids, top_k=max(len(node_ids), self._config.branch_width * 4))
        }
        for node_id in node_ids:
            row = node_rows[node_id]
            semantic_scores[node_id] = semantic_scores.get(node_id, 0.0) + _hierarchy_node_bonus(plan, row, prefer_structure=prefer_structure)
        return semantic_scores


def axe_hierarchy_search(
    db_path: str | Path,
    query: str,
    *,
    index_dir: str | Path | None = None,
    limit: int = 5,
    branch_width: int = 3,
    symbol_limit: int = 5,
    task: str | None = None,
    embedder: TextEmbedder | None = None,
) -> HierarchicalSearchResult:
    return AxeHierarchySearcher(db_path, index_dir=index_dir, embedder=embedder).search(
        query,
        limit=limit,
        branch_width=branch_width,
        symbol_limit=symbol_limit,
        task=task,
    )


def _children_by_parent(node_rows: dict[str, sqlite3.Row]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for node_id, row in node_rows.items():
        parent_id = row["parent_id"]
        if parent_id is None:
            continue
        children[parent_id].append(node_id)
    for parent_id in children:
        children[parent_id].sort()
    return children


def _symbol_ids_by_node(conn: sqlite3.Connection) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for row in conn.execute("SELECT node_id, symbol_id FROM symbols ORDER BY path, start_line, name").fetchall():
        mapping[row["node_id"]].append(row["symbol_id"])
    return mapping


def _descendant_file_ids(node_id: str, children_by_parent: dict[str, list[str]], node_rows: dict[str, sqlite3.Row]) -> list[str]:
    results: list[str] = []
    stack = list(children_by_parent.get(node_id, []))
    while stack:
        child_id = stack.pop()
        row = node_rows[child_id]
        if row["kind"] == "file":
            results.append(child_id)
            continue
        stack.extend(children_by_parent.get(child_id, []))
    return list(dict.fromkeys(results))


def _hierarchy_node_bonus(plan, row: sqlite3.Row, *, prefer_structure: bool) -> float:
    path_text = f"{row['path'].lower()} {row['name'].lower()}"
    descriptive_text = " ".join(
        part.lower() for part in (row["summary"] or "", row["description"] or "", row["primary_category"] or "") if part
    )
    path_matches = sum(1 for token in plan.tokens if token in path_text)
    summary_matches = sum(1 for token in plan.tokens if token in descriptive_text)
    score = path_matches * 8.0 + summary_matches * 6.0
    if prefer_structure and row["kind"] in {"repo", "folder"}:
        score += 4.0
    if not prefer_structure and row["kind"] == "file":
        score += 4.0
    if plan.wants_implementation and row["kind"] == "file":
        score += 5.0
    if plan.wants_implementation and row["symbol_count"] == 0:
        score -= 4.0
    if plan.wants_implementation and row["kind"] == "file":
        lowered_path = row["path"].lower()
        if lowered_path.endswith(".d.ts"):
            score -= 12.0
        if any(marker in lowered_path for marker in ("register-builtins", "_provider.py", "-provider.ts", "provider-module")):
            score -= 14.0
        if "provider module" in descriptive_text or "registers and initializes" in descriptive_text:
            score -= 14.0
    return score


def _path_name_token_hits(plan, row: sqlite3.Row) -> int:
    text = f"{row['path'].lower()} {row['name'].lower()}"
    return sum(1 for token in plan.tokens if token in text)


def _symbol_bonus(plan, row: sqlite3.Row) -> float:
    identifier_text = f"{row['name'].lower()} {row['qualified_name'].lower()} {row['path'].lower()}"
    signature = (row["signature"] or "").lower()
    identifier_matches = sum(1 for token in plan.tokens if token in identifier_text)
    signature_matches = sum(1 for token in plan.tokens if token in signature)
    score = identifier_matches * 7.0 + signature_matches * 1.5
    if plan.normalized and row["normalized_name"] == plan.normalized:
        score += 48.0
    if plan.lowered == row["name"].lower():
        score += 24.0
    return score


def _top_pairs(scores: dict[str, float], limit: int) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]