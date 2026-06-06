from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from matryoshka.embeddings import (
    DEFAULT_QUERY_TASK,
    TextEmbedder,
    build_text_embedder,
    format_query_text,
)
from matryoshka.graph_models import RetrievalResult
from matryoshka.result_loader import SQLiteResultLoader
from matryoshka.retrieval import build_query_plan
from matryoshka.semantic_index import SemanticIndexStore, load_semantic_manifest

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SemanticSearchConfig:
    index_dir: Path | None = None
    default_query_task: str = DEFAULT_QUERY_TASK
    search_k: int | None = None


class AxeSemanticSearcher:
    def __init__(
        self,
        db_path: str | Path,
        *,
        index_dir: str | Path | None = None,
        embedder: TextEmbedder | None = None,
        config: SemanticSearchConfig | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._config = config or SemanticSearchConfig(
            index_dir=Path(index_dir) if index_dir else None
        )
        self._index = SemanticIndexStore(
            self._db_path, index_dir=self._config.index_dir
        )
        self._loader = SQLiteResultLoader(self._db_path)
        self._embedder = embedder or build_text_embedder(
            self._index.model_name, truncate_dim=self._index.dimension
        )
        if self._embedder.dimension != self._index.dimension:
            raise ValueError(
                f"Semantic search embedder dimension {self._embedder.dimension} does not match index dimension {self._index.dimension}"
            )

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        search_k: int | None = None,
        task: str | None = None,
    ) -> RetrievalResult:
        plan = build_query_plan(query)
        task_name = (
            task or self._index.default_query_task or self._config.default_query_task
        )
        candidate_count = search_k or self._config.search_k or max(limit * 8, 25)
        query_vector = self._embedder.encode(
            [format_query_text(query, task=task_name)]
        )[0]

        node_scores = {
            entity_id: score * 100.0
            for entity_id, score in self._index.search_nodes(
                query_vector, top_k=candidate_count
            )
        }
        symbol_scores = {
            entity_id: score * 100.0
            for entity_id, score in self._index.search_symbols(
                query_vector, top_k=candidate_count
            )
        }

        with self._loader.connect() as conn:
            self._warn_if_index_stale(conn)
            symbol_scores = self._rerank_symbols(conn, plan, symbol_scores)
            node_scores = self._rerank_nodes(conn, plan, node_scores, symbol_scores)
            node_hits = [
                self._loader.load_node_hit(conn, node_id, score)
                for node_id, score in _top_hits(node_scores, limit)
            ]
            symbol_hits = [
                self._loader.load_symbol_hit(conn, symbol_id, score)
                for symbol_id, score in _top_hits(symbol_scores, limit)
            ]

        logger.info(
            "semantic search retrieved %s node hits and %s symbol hits for query %r",
            len(node_hits),
            len(symbol_hits),
            query,
        )
        return RetrievalResult(
            query=query, node_hits=node_hits, symbol_hits=symbol_hits
        )

    def _warn_if_index_stale(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("SELECT value FROM meta WHERE key = 'updated_at'").fetchone()
        db_updated_at = None if row is None else row[0]
        manifest = load_semantic_manifest(
            self._db_path, index_dir=self._config.index_dir
        )
        if (
            db_updated_at
            and manifest.get("db_updated_at")
            and db_updated_at != manifest["db_updated_at"]
        ):
            logger.warning(
                "semantic index appears stale relative to the SQLite graph; rebuild with `matryoshka semantic-index`"
            )

    def _rerank_symbols(
        self, conn: sqlite3.Connection, plan, scores: dict[str, float]
    ) -> dict[str, float]:
        rows = _load_rows_by_ids(
            conn,
            "symbols",
            "symbol_id",
            list(scores),
            "symbol_id, name, qualified_name, normalized_name, path, kind, signature",
        )
        for symbol_id, row in rows.items():
            scores[symbol_id] += _symbol_bonus(row, plan)
        return scores

    def _rerank_nodes(
        self,
        conn: sqlite3.Connection,
        plan,
        node_scores: dict[str, float],
        symbol_scores: dict[str, float],
    ) -> dict[str, float]:
        top_symbols = _top_hits(symbol_scores, max(8, len(symbol_scores)))
        anchor_symbols = _anchor_symbols(conn, top_symbols, plan, limit=max(3, 5))
        symbol_node_rows = _load_rows_by_ids(
            conn,
            "symbols",
            "symbol_id",
            [symbol_id for symbol_id, _ in anchor_symbols],
            "symbol_id, node_id",
        )

        for symbol_id, score in anchor_symbols:
            row = symbol_node_rows.get(symbol_id)
            if row is None:
                continue
            owner_node_id = row["node_id"]
            if plan.wants_implementation:
                node_scores[owner_node_id] = (
                    node_scores.get(owner_node_id, 0.0) + 16.0 + min(18.0, score * 0.2)
                )
            elif plan.wants_callers:
                node_scores[owner_node_id] = node_scores.get(owner_node_id, 0.0) - 80.0
            else:
                node_scores[owner_node_id] = (
                    node_scores.get(owner_node_id, 0.0) + 4.0 + min(8.0, score * 0.15)
                )

            if plan.wants_callers:
                for call_row in conn.execute(
                    "SELECT caller_node_id FROM call_sites WHERE target_symbol_id = ?",
                    (symbol_id,),
                ).fetchall():
                    node_scores[call_row["caller_node_id"]] = (
                        node_scores.get(call_row["caller_node_id"], 0.0) + 48.0
                    )
                for reference_row in conn.execute(
                    "SELECT source_node_id, reference_kind FROM symbol_references WHERE target_symbol_id = ?",
                    (symbol_id,),
                ).fetchall():
                    bonus = 18.0 if reference_row["reference_kind"] == "call" else 9.0
                    node_scores[reference_row["source_node_id"]] = (
                        node_scores.get(reference_row["source_node_id"], 0.0) + bonus
                    )

            if plan.wants_callees:
                for call_row in conn.execute(
                    "SELECT target_node_id FROM call_sites WHERE caller_symbol_id = ? AND target_node_id IS NOT NULL",
                    (symbol_id,),
                ).fetchall():
                    node_scores[call_row["target_node_id"]] = (
                        node_scores.get(call_row["target_node_id"], 0.0) + 30.0
                    )

        rows = _load_rows_by_ids(
            conn,
            "nodes",
            "node_id",
            list(node_scores),
            "node_id, path, name, kind, summary, description, primary_category, symbol_count",
        )
        for node_id, row in rows.items():
            node_scores[node_id] += _node_bonus(row, plan)
        return node_scores


def axe_semantic_search(
    db_path: str | Path,
    query: str,
    *,
    index_dir: str | Path | None = None,
    limit: int = 5,
    search_k: int | None = None,
    task: str | None = None,
    embedder: TextEmbedder | None = None,
) -> RetrievalResult:
    return AxeSemanticSearcher(db_path, index_dir=index_dir, embedder=embedder).search(
        query,
        limit=limit,
        search_k=search_k,
        task=task,
    )


def _anchor_symbols(
    conn: sqlite3.Connection, symbol_hits: list[tuple[str, float]], plan, *, limit: int
) -> list[tuple[str, float]]:
    if not symbol_hits:
        return []
    rows = _load_rows_by_ids(
        conn,
        "symbols",
        "symbol_id",
        [symbol_id for symbol_id, _ in symbol_hits],
        "symbol_id, name, qualified_name, normalized_name",
    )
    token_set = set(plan.tokens)
    exact_matches: list[tuple[str, float]] = []
    for symbol_id, score in symbol_hits:
        row = rows.get(symbol_id)
        if row is None:
            continue
        if (
            row["normalized_name"] == plan.normalized
            or row["normalized_name"] in token_set
            or row["name"].lower() in token_set
            or row["qualified_name"].lower() in token_set
        ):
            exact_matches.append((symbol_id, score))
    return exact_matches[:limit] if exact_matches else symbol_hits[:limit]


def _symbol_bonus(row: sqlite3.Row, plan) -> float:
    name = row["name"].lower()
    qualified_name = row["qualified_name"].lower()
    normalized_name = row["normalized_name"]
    path = row["path"].lower()
    signature = (row["signature"] or "").lower()
    identifier_text = f"{name} {qualified_name}"
    path_text = f"{path} {name}"

    identifier_match_count = sum(1 for token in plan.tokens if token in identifier_text)
    signature_match_count = sum(1 for token in plan.tokens if token in signature)
    path_match_count = sum(1 for token in plan.tokens if token in path_text)

    bonus = identifier_match_count * 5.0
    bonus += signature_match_count * 1.1
    bonus += path_match_count * 3.0

    if plan.is_identifier_query and normalized_name == plan.normalized:
        bonus += 48.0
    if plan.lowered == name:
        bonus += 20.0
    if plan.lowered == qualified_name:
        bonus += 14.0
    if len(plan.tokens) > 1 and row["kind"] in {"function", "method", "class"}:
        bonus += 3.0
    if (
        len(plan.tokens) > 1
        and row["kind"] == "variable"
        and not _looks_callable(signature)
    ):
        bonus -= 4.0
    if len(plan.tokens) > 1 and not identifier_match_count and path_match_count:
        bonus -= 6.0
    if len(plan.tokens) > 1 and name.startswith("_"):
        bonus -= 3.0
    if plan.wants_implementation and path.endswith(".d.ts"):
        bonus -= 8.0
    if plan.wants_implementation and _looks_indirect_symbol(name, qualified_name):
        bonus -= 10.0
    return bonus


def _node_bonus(row: sqlite3.Row, plan) -> float:
    summary = (row["summary"] or "").lower()
    description = (row["description"] or "").lower()
    primary_category = (row["primary_category"] or "").lower()
    path = row["path"].lower()
    name = row["name"].lower()
    path_name_text = f"{path} {name}"
    descriptive_text = f"{summary} {description} {primary_category}"

    path_name_matches = sum(1 for token in plan.tokens if token in path_name_text)
    descriptive_matches = sum(1 for token in plan.tokens if token in descriptive_text)

    bonus = path_name_matches * 8.0
    bonus += descriptive_matches * 1.5
    if plan.prefers_files and row["kind"] == "file":
        bonus += 3.0
    if plan.wants_implementation and row["kind"] == "file":
        bonus += 4.0
    if plan.wants_callers and row["kind"] == "file":
        bonus += 2.0
    if plan.wants_implementation and path.endswith(".d.ts"):
        bonus -= 10.0
    if plan.wants_implementation and row["symbol_count"] == 0:
        bonus -= 4.0
    if plan.wants_implementation and row["kind"] == "repo":
        bonus -= 6.0
    if plan.wants_implementation and _looks_indirect_node(
        path, name, summary, description
    ):
        bonus -= 18.0
    return bonus


def _looks_callable(signature: str) -> bool:
    compact = " ".join(signature.split())
    return "function " in compact or "=>" in compact or compact.endswith(")")


def _looks_indirect_symbol(name: str, qualified_name: str) -> bool:
    text = f"{name} {qualified_name}"
    return any(marker in text for marker in ("lazy", "register", "providermodule"))


def _looks_indirect_node(path: str, name: str, summary: str, description: str) -> bool:
    combined = f"{path} {name} {summary} {description}"
    path_markers = ("register", "registry", "builtins")
    summary_markers = ("registers", "lazy", "exporting", "provider module")
    return any(marker in path for marker in path_markers) or any(
        marker in combined for marker in summary_markers
    )


def _load_rows_by_ids(
    conn: sqlite3.Connection,
    table_name: str,
    id_column: str,
    ids: list[str],
    columns: str,
) -> dict[str, sqlite3.Row]:
    if not ids:
        return {}
    placeholders = ", ".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT {columns} FROM {table_name} WHERE {id_column} IN ({placeholders})", ids
    ).fetchall()
    return {row[id_column]: row for row in rows}


def _top_hits(scores: dict[str, float], limit: int) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
