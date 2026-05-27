from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from cradle.exact_search import axe_call_search, axe_import_search, axe_reference_search, axe_symbol_search
from cradle.graph_models import CodeExcerpt, QuestionResult
from cradle.hierarchical_search import AxeHierarchySearcher, HierarchySearchConfig
from cradle.retrieval import build_query_plan


@dataclass(slots=True)
class QuestionConfig:
    index_dir: Path | None = None
    branch_width: int = 3
    file_limit: int = 4
    symbol_limit: int = 4
    excerpt_line_budget: int = 24


class AxeQuestionAnswerer:
    def __init__(self, db_path: str | Path, *, index_dir: str | Path | None = None, embedder=None, config: QuestionConfig | None = None) -> None:
        self._db_path = Path(db_path)
        self._config = config or QuestionConfig(index_dir=Path(index_dir) if index_dir else None)
        self._hierarchy = AxeHierarchySearcher(
            self._db_path,
            index_dir=self._config.index_dir,
            embedder=embedder,
            config=HierarchySearchConfig(
                index_dir=self._config.index_dir,
                branch_width=self._config.branch_width,
                file_limit=self._config.file_limit,
                symbol_limit=self._config.symbol_limit,
            ),
        )

    def answer(self, query: str) -> QuestionResult:
        plan = build_query_plan(query)
        hierarchy_result = self._hierarchy.search(
            query,
            limit=self._config.file_limit,
            branch_width=self._config.branch_width,
            symbol_limit=self._config.symbol_limit,
        )

        symbol_result = axe_symbol_search(self._db_path, query, limit=self._config.symbol_limit)
        import_result = axe_import_search(self._db_path, query, limit=self._config.file_limit)
        call_result = axe_call_search(self._db_path, query, limit=self._config.file_limit)
        reference_result = axe_reference_search(self._db_path, query, limit=self._config.file_limit)

        merged_node_hits = _merge_hits(hierarchy_result.node_hits, [])
        merged_symbol_hits = _merge_hits(hierarchy_result.symbol_hits, symbol_result.symbol_hits)
        preferred_symbol_hit = _preferred_symbol_hit(plan, merged_symbol_hits)
        if preferred_symbol_hit is not None:
            merged_symbol_hits = [preferred_symbol_hit, *[hit for hit in merged_symbol_hits if hit.symbol.symbol_id != preferred_symbol_hit.symbol.symbol_id]]
        excerpts = _load_excerpts(self._db_path, merged_node_hits, merged_symbol_hits, self._config.excerpt_line_budget)
        answer = _build_answer(
            query,
            plan,
            hierarchy_result,
            merged_node_hits,
            merged_symbol_hits,
            import_result.import_hits,
            call_result.call_hits,
            reference_result.reference_hits,
            excerpts,
        )
        return QuestionResult(
            query=query,
            answer=answer,
            traversal_steps=hierarchy_result.steps,
            node_hits=merged_node_hits,
            symbol_hits=merged_symbol_hits,
            import_hits=import_result.import_hits,
            call_hits=call_result.call_hits,
            reference_hits=reference_result.reference_hits,
            excerpts=excerpts,
        )


def axe_question(db_path: str | Path, query: str, *, index_dir: str | Path | None = None, embedder=None) -> QuestionResult:
    return AxeQuestionAnswerer(db_path, index_dir=index_dir, embedder=embedder).answer(query)


def _merge_hits(primary_hits, secondary_hits):
    results = []
    seen: set[str] = set()
    for hit in [*primary_hits, *secondary_hits]:
        if hasattr(hit, "node"):
            key = f"node:{hit.node.node_id}"
        else:
            key = f"symbol:{hit.symbol.symbol_id}"
        if key in seen:
            continue
        seen.add(key)
        results.append(hit)
    return results


def _load_excerpts(db_path: Path, node_hits, symbol_hits, line_budget: int) -> list[CodeExcerpt]:
    repo_root = _repo_root(db_path)
    excerpts: list[CodeExcerpt] = []
    seen: set[tuple[str, int, int]] = set()

    for hit in symbol_hits[:3]:
        symbol = hit.symbol
        start_line = max(1, symbol.start_line or 1)
        end_line = start_line if symbol.end_line is None else min(symbol.end_line, start_line + line_budget - 1)
        excerpt = _read_excerpt(repo_root, symbol.path, start_line, end_line)
        if excerpt is None:
            continue
        key = (excerpt.path, excerpt.start_line, excerpt.end_line)
        if key in seen:
            continue
        seen.add(key)
        excerpts.append(excerpt)

    if excerpts:
        return excerpts

    for hit in node_hits[:2]:
        if hit.node.kind != "file":
            continue
        excerpt = _read_excerpt(repo_root, hit.node.path, 1, line_budget)
        if excerpt is None:
            continue
        key = (excerpt.path, excerpt.start_line, excerpt.end_line)
        if key in seen:
            continue
        seen.add(key)
        excerpts.append(excerpt)
    return excerpts


def _repo_root(db_path: Path) -> Path:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT root_path FROM repos LIMIT 1").fetchone()
    finally:
        conn.close()
    if row is None:
        return db_path.parent
    return Path(row[0])


def _read_excerpt(repo_root: Path, relative_path: str, start_line: int, end_line: int) -> CodeExcerpt | None:
    path = Path(relative_path)
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.exists():
        return None
    lines = resolved.read_text(encoding="utf-8").splitlines()
    start_index = max(0, start_line - 1)
    end_index = min(len(lines), end_line)
    excerpt_lines = lines[start_index:end_index]
    if not excerpt_lines:
        return None
    return CodeExcerpt(path=relative_path, start_line=start_line, end_line=end_index, text="\n".join(excerpt_lines))


def _build_answer(query, plan, hierarchy_result, node_hits, symbol_hits, import_hits, call_hits, reference_hits, excerpts):
    lines = [f"Question: {query}"]
    preferred_symbol_hit = _preferred_symbol_hit(plan, symbol_hits)

    traversal_labels = [candidate.node.path for step in hierarchy_result.steps for candidate in step.candidates[:1]]
    if traversal_labels:
        lines.append(f"Traversal: {' -> '.join(traversal_labels)}")

    if plan.wants_callers and call_hits:
        target = call_hits[0].target_symbol.qualified_name if call_hits[0].target_symbol is not None else call_hits[0].call_record.callee_name
        caller_labels = []
        for hit in call_hits[:4]:
            if hit.caller_symbol is not None:
                caller_labels.append(f"{hit.caller_symbol.qualified_name} in {hit.caller_node.path if hit.caller_node is not None else hit.call_record.caller_node_id}")
            elif hit.caller_node is not None:
                caller_labels.append(hit.caller_node.path)
        lines.append(f"Best answer: {target} is called from {', '.join(caller_labels)}.")
    elif plan.wants_callees and call_hits:
        source = call_hits[0].caller_symbol.qualified_name if call_hits[0].caller_symbol is not None else call_hits[0].call_record.caller_node_id
        callee_labels = [hit.target_symbol.qualified_name if hit.target_symbol is not None else hit.call_record.callee_name for hit in call_hits[:4]]
        lines.append(f"Best answer: {source} calls {', '.join(callee_labels)}.")
    elif preferred_symbol_hit is not None:
        symbol = preferred_symbol_hit.symbol
        lines.append(f"Best answer: the strongest match is {symbol.qualified_name} in {symbol.path}.")
    elif node_hits:
        node = node_hits[0].node
        lines.append(f"Best answer: the strongest file match is {node.path}.")

    top_node = _answer_node_hit(node_hits, preferred_symbol_hit)
    if top_node is not None and top_node.node.summary:
        lines.append(f"Why this branch: {top_node.node.summary}")

    preferred_symbol_name = None if preferred_symbol_hit is None else preferred_symbol_hit.symbol.name.lower()

    if import_hits and any(token in plan.lowered for token in ("import", "module")):
        top_import = import_hits[0]
        lines.append(
            f"Supporting import evidence: {top_import.importer_node.path} imports {top_import.import_record.imported_module}."
        )
    if reference_hits and (plan.is_identifier_query or plan.wants_callers or plan.wants_callees or preferred_symbol_name is not None):
        aligned_references = [
            hit for hit in reference_hits if preferred_symbol_name is None or hit.reference_record.target_name.lower() == preferred_symbol_name
        ]
        if aligned_references or plan.is_identifier_query or plan.wants_callers or plan.wants_callees:
            top_reference = aligned_references[0] if aligned_references else reference_hits[0]
            lines.append(
                f"Supporting reference evidence: {top_reference.reference_record.source_node_id} references {top_reference.reference_record.target_name} as a {top_reference.reference_record.reference_kind}."
            )

    if excerpts:
        excerpt = excerpts[0]
        lines.append(f"Code evidence ({excerpt.path}:{excerpt.start_line}-{excerpt.end_line}):")
        lines.append(excerpt.text)

    return "\n".join(lines)


def _preferred_symbol_hit(plan, symbol_hits):
    if not symbol_hits:
        return None
    if plan.wants_callers or plan.wants_callees:
        return symbol_hits[0]

    preferred_kinds = {"function", "method", "class"}
    if any(word in plan.lowered for word in ("how", "where", "implement", "defined", "definition")):
        preferred_kinds.update({"variable"})
        for hit in symbol_hits:
            if hit.symbol.kind in preferred_kinds:
                return hit

    return symbol_hits[0]


def _answer_node_hit(node_hits, preferred_symbol_hit):
    if preferred_symbol_hit is None:
        return node_hits[0] if node_hits else None
    for hit in node_hits:
        if hit.node.path == preferred_symbol_hit.symbol.path:
            return hit
    return node_hits[0] if node_hits else None