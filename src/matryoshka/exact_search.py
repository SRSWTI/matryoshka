from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from matryoshka.graph_models import CallRecord, CodeNode, CodeSymbol, ExactCallHit, ExactImportHit, ExactReferenceHit, ExactSearchResult, ImportRecord, RetrievalNodeHit, RetrievalSymbolHit, SymbolReferenceRecord
from matryoshka.result_loader import SQLiteResultLoader
from matryoshka.retrieval import QueryPlan, build_query_plan

_WORD_SPLIT = re.compile(r"[^a-z0-9_]+")


class AxeExactSearcher:
    def __init__(self, db_path: str | Path) -> None:
        self._loader = SQLiteResultLoader(db_path)

    def search_files(self, query: str, *, limit: int = 5) -> ExactSearchResult:
        plan = build_query_plan(query)
        with self._loader.connect() as conn:
            rows = conn.execute(
                "SELECT node_id, path, name, normalized_name FROM nodes WHERE kind = 'file' ORDER BY path"
            ).fetchall()
            scores = {
                row["node_id"]: score
                for row in rows
                if (score := _score_node_like_match(plan, row["path"], row["name"], row["normalized_name"])) > 0
            }
            hits = [self._loader.load_node_hit(conn, node_id, score) for node_id, score in _top_hits(scores, limit)]
        return ExactSearchResult(query=query, search_type="file", node_hits=hits)

    def search_symbols(self, query: str, *, limit: int = 5) -> ExactSearchResult:
        plan = build_query_plan(query)
        with self._loader.connect() as conn:
            rows = conn.execute(
                "SELECT symbol_id, name, qualified_name, normalized_name, path, signature FROM symbols ORDER BY qualified_name"
            ).fetchall()
            scores = {
                row["symbol_id"]: score
                for row in rows
                if (score := _score_symbol_like_match(plan, row["name"], row["qualified_name"], row["normalized_name"], row["path"], row["signature"])) > 0
            }
            hits = [self._loader.load_symbol_hit(conn, symbol_id, score) for symbol_id, score in _top_hits(scores, limit)]
        return ExactSearchResult(query=query, search_type="symbol", symbol_hits=hits)

    def search_imports(self, query: str, *, limit: int = 5) -> ExactSearchResult:
        plan = build_query_plan(query)
        with self._loader.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM imports ORDER BY importer_node_id, start_line, imported_module"
            ).fetchall()
            hits = self._import_hits(conn, rows, plan, limit=limit)
        return ExactSearchResult(query=query, search_type="import", import_hits=hits)

    def search_modules(self, query: str, *, limit: int = 5) -> ExactSearchResult:
        plan = build_query_plan(query)
        with self._loader.connect() as conn:
            node_rows = conn.execute(
                "SELECT node_id, path, name, normalized_name FROM nodes WHERE kind IN ('repo', 'folder', 'file') ORDER BY kind, path"
            ).fetchall()
            node_scores = {
                row["node_id"]: score
                for row in node_rows
                if (score := _score_node_like_match(plan, row["path"], row["name"], row["normalized_name"], include_module_aliases=True)) > 0
            }
            import_rows = conn.execute("SELECT * FROM imports ORDER BY importer_node_id, start_line, imported_module").fetchall()
            import_hits = self._import_hits(conn, import_rows, plan, limit=limit)
            node_hits = [self._loader.load_node_hit(conn, node_id, score) for node_id, score in _top_hits(node_scores, limit)]
        return ExactSearchResult(query=query, search_type="module", node_hits=node_hits, import_hits=import_hits)

    def search_calls(self, query: str, *, limit: int = 5) -> ExactSearchResult:
        plan = build_query_plan(query)
        with self._loader.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    call_sites.*,
                    caller.name AS caller_name,
                    caller.qualified_name AS caller_qualified_name,
                    target.name AS target_name,
                    target.qualified_name AS target_qualified_name
                FROM call_sites
                LEFT JOIN symbols AS caller ON caller.symbol_id = call_sites.caller_symbol_id
                LEFT JOIN symbols AS target ON target.symbol_id = call_sites.target_symbol_id
                ORDER BY call_sites.caller_node_id, call_sites.start_line, call_sites.callee_name
                """
            ).fetchall()
            hits = self._call_hits(conn, rows, plan, limit=limit)
        return ExactSearchResult(query=query, search_type="call", call_hits=hits)

    def search_references(self, query: str, *, limit: int = 5) -> ExactSearchResult:
        plan = build_query_plan(query)
        with self._loader.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    symbol_references.*,
                    source.name AS source_name,
                    source.qualified_name AS source_qualified_name,
                    target.name AS target_symbol_name,
                    target.qualified_name AS target_qualified_name
                FROM symbol_references
                LEFT JOIN symbols AS source ON source.symbol_id = symbol_references.source_symbol_id
                LEFT JOIN symbols AS target ON target.symbol_id = symbol_references.target_symbol_id
                ORDER BY symbol_references.source_node_id, symbol_references.start_line, symbol_references.reference_kind
                """
            ).fetchall()
            hits = self._reference_hits(conn, rows, plan, limit=limit)
        return ExactSearchResult(query=query, search_type="reference", reference_hits=hits)

    def _import_hits(self, conn: sqlite3.Connection, rows: list[sqlite3.Row], plan: QueryPlan, *, limit: int) -> list[ExactImportHit]:
        scored_rows: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            names = _json_loads(row["names_json"])
            score = _score_module_like_match(plan, row["imported_module"], names)
            if score <= 0:
                continue
            scored_rows.append((score, row))

        return self._materialize_import_hits(conn, scored_rows, limit=limit)

    def _call_hits(self, conn: sqlite3.Connection, rows: list[sqlite3.Row], plan: QueryPlan, *, limit: int) -> list[ExactCallHit]:
        scored_rows: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            caller_text = " ".join(filter(None, [row["caller_name"], row["caller_qualified_name"]]))
            target_text = " ".join(filter(None, [row["callee_name"], row["target_name"], row["target_qualified_name"]]))
            if plan.wants_callees and not plan.wants_callers:
                score = _score_identifier_match(plan, caller_text) * 1.4 + _score_identifier_match(plan, target_text) * 0.25
            elif plan.wants_callers and not plan.wants_callees:
                score = _score_identifier_match(plan, target_text) * 1.5 + _score_identifier_match(plan, caller_text) * 0.25
            else:
                score = _score_identifier_match(plan, caller_text) + _score_identifier_match(plan, target_text)
            if score <= 0:
                continue
            if plan.wants_callers:
                score += 8.0
            if plan.wants_callees:
                score += 8.0
            scored_rows.append((score, row))

        return self._materialize_call_hits(conn, scored_rows, limit=limit)

    def _reference_hits(self, conn: sqlite3.Connection, rows: list[sqlite3.Row], plan: QueryPlan, *, limit: int) -> list[ExactReferenceHit]:
        scored_rows: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            target_text = " ".join(filter(None, [row["target_name"], row["target_symbol_name"], row["target_qualified_name"]]))
            score = _score_identifier_match(plan, target_text)
            if score <= 0:
                continue
            if row["reference_kind"] == "call":
                score += 6.0
            elif row["reference_kind"] == "import":
                score += 4.0
            scored_rows.append((score, row))

        return self._materialize_reference_hits(conn, scored_rows, limit=limit)

    def _materialize_import_hits(self, conn: sqlite3.Connection, scored_rows: list[tuple[float, sqlite3.Row]], *, limit: int) -> list[ExactImportHit]:
        hits: list[ExactImportHit] = []
        seen: set[tuple[object, ...]] = set()
        for score, row in sorted(scored_rows, key=lambda item: (-item[0], item[1]["importer_node_id"], item[1]["imported_module"])):
            key = (row["importer_node_id"], row["imported_module"], row["target_node_id"], row["start_line"], row["start_column"])
            if key in seen:
                continue
            seen.add(key)
            importer_node = self._loader.load_node(conn, row["importer_node_id"])
            target_node = self._safe_load_node(conn, row["target_node_id"])
            hits.append(
                ExactImportHit(
                    score=score,
                    import_record=_row_to_import(row),
                    importer_node=importer_node,
                    target_node=target_node,
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def _materialize_call_hits(self, conn: sqlite3.Connection, scored_rows: list[tuple[float, sqlite3.Row]], *, limit: int) -> list[ExactCallHit]:
        hits: list[ExactCallHit] = []
        seen: set[tuple[object, ...]] = set()
        for score, row in sorted(scored_rows, key=lambda item: (-item[0], item[1]["caller_node_id"], item[1]["start_line"], item[1]["callee_name"])):
            key = (row["caller_symbol_id"], row["caller_node_id"], row["callee_name"], row["target_symbol_id"], row["start_line"], row["start_column"])
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                ExactCallHit(
                    score=score,
                    call_record=_row_to_call(row),
                    caller_node=self._safe_load_node(conn, row["caller_node_id"]),
                    caller_symbol=self._safe_load_symbol(conn, row["caller_symbol_id"]),
                    target_node=self._safe_load_node(conn, row["target_node_id"]),
                    target_symbol=self._safe_load_symbol(conn, row["target_symbol_id"]),
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def _materialize_reference_hits(self, conn: sqlite3.Connection, scored_rows: list[tuple[float, sqlite3.Row]], *, limit: int) -> list[ExactReferenceHit]:
        hits: list[ExactReferenceHit] = []
        seen: set[tuple[object, ...]] = set()
        for score, row in sorted(scored_rows, key=lambda item: (-item[0], item[1]["source_node_id"], item[1]["start_line"], item[1]["reference_kind"])):
            key = (
                row["target_symbol_id"],
                row["target_name"],
                row["source_node_id"],
                row["source_symbol_id"],
                row["reference_kind"],
                row["start_line"],
                row["start_column"],
            )
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                ExactReferenceHit(
                    score=score,
                    reference_record=_row_to_reference(row),
                    source_node=self._safe_load_node(conn, row["source_node_id"]),
                    source_symbol=self._safe_load_symbol(conn, row["source_symbol_id"]),
                    target_node=self._safe_load_node(conn, row["target_node_id"]),
                    target_symbol=self._safe_load_symbol(conn, row["target_symbol_id"]),
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def _safe_load_node(self, conn: sqlite3.Connection, node_id: str | None) -> CodeNode | None:
        if not node_id:
            return None
        try:
            return self._loader.load_node(conn, node_id)
        except KeyError:
            return None

    def _safe_load_symbol(self, conn: sqlite3.Connection, symbol_id: str | None) -> CodeSymbol | None:
        if not symbol_id:
            return None
        try:
            return self._loader.load_symbol(conn, symbol_id)
        except KeyError:
            return None


def axe_file_search(db_path: str | Path, query: str, *, limit: int = 5) -> ExactSearchResult:
    return AxeExactSearcher(db_path).search_files(query, limit=limit)


def axe_symbol_search(db_path: str | Path, query: str, *, limit: int = 5) -> ExactSearchResult:
    return AxeExactSearcher(db_path).search_symbols(query, limit=limit)


def axe_import_search(db_path: str | Path, query: str, *, limit: int = 5) -> ExactSearchResult:
    return AxeExactSearcher(db_path).search_imports(query, limit=limit)


def axe_module_search(db_path: str | Path, query: str, *, limit: int = 5) -> ExactSearchResult:
    return AxeExactSearcher(db_path).search_modules(query, limit=limit)


def axe_call_search(db_path: str | Path, query: str, *, limit: int = 5) -> ExactSearchResult:
    return AxeExactSearcher(db_path).search_calls(query, limit=limit)


def axe_reference_search(db_path: str | Path, query: str, *, limit: int = 5) -> ExactSearchResult:
    return AxeExactSearcher(db_path).search_references(query, limit=limit)


def _score_node_like_match(
    plan: QueryPlan,
    path: str | None,
    name: str | None,
    normalized_name: str | None,
    *,
    include_module_aliases: bool = False,
) -> float:
    score = 0.0
    score += _score_identifier_match(plan, path, exact_weight=46.0, contains_weight=9.0, segment_weight=12.0, include_aliases=include_module_aliases)
    score += _score_identifier_match(plan, name, exact_weight=36.0, contains_weight=8.0, segment_weight=10.0)
    if normalized_name and plan.normalized and normalized_name == plan.normalized:
        score += 48.0
    return score


def _score_symbol_like_match(
    plan: QueryPlan,
    name: str | None,
    qualified_name: str | None,
    normalized_name: str | None,
    path: str | None,
    signature: str | None,
) -> float:
    score = 0.0
    score += _score_identifier_match(plan, name, exact_weight=44.0, contains_weight=10.0, segment_weight=12.0)
    score += _score_identifier_match(plan, qualified_name, exact_weight=40.0, contains_weight=8.0, segment_weight=10.0, include_aliases=True)
    score += _score_identifier_match(plan, path, exact_weight=20.0, contains_weight=3.5, segment_weight=4.0, include_aliases=True)
    score += _score_identifier_match(plan, signature, exact_weight=12.0, contains_weight=2.5, segment_weight=0.0)
    if normalized_name and plan.normalized and normalized_name == plan.normalized:
        score += 54.0
    return score


def _score_module_like_match(plan: QueryPlan, module_name: str | None, names: list[str]) -> float:
    score = _score_identifier_match(plan, module_name, exact_weight=48.0, contains_weight=9.0, segment_weight=12.0, include_aliases=True)
    for name in names:
        score += _score_identifier_match(plan, name, exact_weight=18.0, contains_weight=3.0, segment_weight=4.0)
    return score


def _score_identifier_match(
    plan: QueryPlan,
    value: str | None,
    *,
    exact_weight: float = 40.0,
    contains_weight: float = 8.0,
    segment_weight: float = 10.0,
    include_aliases: bool = False,
) -> float:
    if not value:
        return 0.0
    lowered = value.lower()
    normalized = _normalize_identifier(value)
    candidates = {lowered}
    if include_aliases:
        candidates.update(_string_aliases(lowered))

    score = 0.0
    if plan.lowered and plan.lowered in candidates:
        score += exact_weight
    if plan.normalized and normalized == plan.normalized:
        score += exact_weight * 0.95

    segments = {segment for candidate in candidates for segment in _word_segments(candidate)}
    for token in plan.tokens:
        if token in segments:
            score += segment_weight
        elif any(token in candidate for candidate in candidates):
            score += contains_weight
    return score


def _string_aliases(value: str) -> set[str]:
    aliases = {value}
    aliases.add(value.replace("/", "."))
    aliases.add(value.replace(".", "/"))
    aliases.add(value.replace("-", "_"))
    aliases.add(value.replace("_", "-"))
    return {alias for alias in aliases if alias}


def _word_segments(value: str) -> set[str]:
    return {segment for segment in _WORD_SPLIT.split(value) if segment}


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", value).lower()


def _json_loads(value: str | None) -> list[str]:
    if not value:
        return []
    return [str(item) for item in json.loads(value)]


def _top_hits(scores: dict[str, float], limit: int) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _row_to_import(row: sqlite3.Row) -> ImportRecord:
    return ImportRecord(
        importer_node_id=row["importer_node_id"],
        imported_module=row["imported_module"],
        target_node_id=row["target_node_id"],
        is_internal=bool(row["is_internal"]),
        strength_label=row["strength_label"],
        strength_weight=row["strength_weight"],
        names=_json_loads(row["names_json"]),
        start_line=row["start_line"],
        start_column=row["start_column"],
        end_line=row["end_line"],
        end_column=row["end_column"],
    )


def _row_to_call(row: sqlite3.Row) -> CallRecord:
    return CallRecord(
        caller_symbol_id=row["caller_symbol_id"],
        caller_node_id=row["caller_node_id"],
        callee_name=row["callee_name"],
        target_symbol_id=row["target_symbol_id"],
        target_node_id=row["target_node_id"],
        start_line=row["start_line"],
        start_column=row["start_column"],
        end_line=row["end_line"],
        end_column=row["end_column"],
    )


def _row_to_reference(row: sqlite3.Row) -> SymbolReferenceRecord:
    return SymbolReferenceRecord(
        target_symbol_id=row["target_symbol_id"],
        target_node_id=row["target_node_id"],
        target_name=row["target_name"],
        source_node_id=row["source_node_id"],
        source_symbol_id=row["source_symbol_id"],
        reference_kind=row["reference_kind"],
        start_line=row["start_line"],
        start_column=row["start_column"],
        end_line=row["end_line"],
        end_column=row["end_column"],
    )