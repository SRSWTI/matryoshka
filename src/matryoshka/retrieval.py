from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from matryoshka.graph_models import RetrievalResult
from matryoshka.result_loader import SQLiteResultLoader

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
IDENTIFIER_QUERY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_./:-]*$")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "do",
    "does",
    "file",
    "files",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "module",
    "modules",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "where",
    "which",
    "who",
}

TOKEN_ALIASES = {
    "authentication": ("auth",),
    "environment": ("env",),
    "implemented": ("implement",),
    "implementation": ("implement",),
    "keys": ("key",),
    "providers": ("provider",),
    "registered": ("register",),
    "registration": ("register",),
    "responses": ("response",),
    "streaming": ("stream",),
    "streams": ("stream",),
}

CALLABLE_KINDS = {"function", "method", "class"}


@dataclass(frozen=True, slots=True)
class QueryPlan:
    raw: str
    lowered: str
    normalized: str
    tokens: list[str]
    is_identifier_query: bool
    wants_callers: bool
    wants_callees: bool
    wants_implementation: bool
    prefers_files: bool


class AxeRetriever:
    def __init__(self, db_path: str | Path) -> None:
        self._loader = SQLiteResultLoader(db_path)

    def retrieve(self, query: str, *, limit: int = 5) -> RetrievalResult:
        plan = build_query_plan(query)
        with self._loader.connect() as conn:
            symbol_scores = self._score_symbols(conn, plan, limit)
            node_scores = self._score_nodes(conn, plan, limit, symbol_scores)
            node_hits = [self._loader.load_node_hit(conn, node_id, score) for node_id, score in self._top_hits(node_scores, limit)]
            symbol_hits = [self._loader.load_symbol_hit(conn, symbol_id, score) for symbol_id, score in self._top_hits(symbol_scores, limit)]
        logger.info("retrieved %s node hits and %s symbol hits for query %r", len(node_hits), len(symbol_hits), query)
        return RetrievalResult(query=query, node_hits=node_hits, symbol_hits=symbol_hits)

    def _score_nodes(self, conn: sqlite3.Connection, plan: QueryPlan, limit: int, symbol_scores: dict[str, float]) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        if not plan.wants_callers:
            for token in plan.tokens:
                like_value = f"%{token}%"
                for row in conn.execute(
                    "SELECT node_id FROM nodes WHERE lower(name) = ? OR lower(path) = ?",
                    (token, token),
                ).fetchall():
                    scores[row["node_id"]] += 16.0
                for row in conn.execute(
                    "SELECT node_id FROM nodes WHERE lower(path) LIKE ? OR lower(name) LIKE ?",
                    (like_value, like_value),
                ).fetchall():
                    scores[row["node_id"]] += 7.0
                for row in conn.execute(
                    "SELECT node_id FROM nodes WHERE lower(summary) LIKE ? OR lower(description) LIKE ?",
                    (like_value, like_value),
                ).fetchall():
                    scores[row["node_id"]] += 6.0
                for row in conn.execute("SELECT node_id FROM node_tags WHERE lower(tag) = ?", (token,)).fetchall():
                    scores[row["node_id"]] += 5.0
                for row in conn.execute("SELECT node_id FROM node_categories WHERE lower(category) = ?", (token,)).fetchall():
                    scores[row["node_id"]] += 5.0

            if _table_exists(conn, "node_search") and plan.tokens:
                match_query = " OR ".join(f'"{token}"*' for token in plan.tokens)
                for row in conn.execute(
                    "SELECT node_id, bm25(node_search) AS rank FROM node_search WHERE node_search MATCH ? LIMIT ?",
                    (match_query, limit * 3),
                ).fetchall():
                    scores[row["node_id"]] += max(0.25, 5.0 - float(row["rank"]))

        top_symbols = self._top_hits(symbol_scores, max(limit * 4, 8))
        anchor_symbols = self._anchor_symbols(conn, top_symbols, plan, limit=max(3, limit))
        symbol_node_rows = _load_rows_by_ids(conn, "symbols", "symbol_id", [symbol_id for symbol_id, _ in anchor_symbols], "symbol_id, node_id")
        for symbol_id, score in anchor_symbols:
            row = symbol_node_rows.get(symbol_id)
            if row is None:
                continue
            if plan.wants_implementation:
                scores[row["node_id"]] += 18.0 + min(20.0, score * 0.25)
            elif plan.wants_callers:
                scores[row["node_id"]] += 2.0
            else:
                scores[row["node_id"]] += 4.0 + min(10.0, score * 0.18)

            if plan.wants_callers:
                scores[row["node_id"]] -= 200.0
                for call_row in conn.execute(
                    "SELECT caller_node_id FROM call_sites WHERE target_symbol_id = ?",
                    (symbol_id,),
                ).fetchall():
                    scores[call_row["caller_node_id"]] += 120.0
                for reference_row in conn.execute(
                    "SELECT source_node_id, reference_kind FROM symbol_references WHERE target_symbol_id = ?",
                    (symbol_id,),
                ).fetchall():
                    bonus = 24.0 if reference_row["reference_kind"] == "call" else 12.0
                    scores[reference_row["source_node_id"]] += bonus

            if plan.wants_callees:
                for call_row in conn.execute(
                    "SELECT target_node_id FROM call_sites WHERE caller_symbol_id = ? AND target_node_id IS NOT NULL",
                    (symbol_id,),
                ).fetchall():
                    scores[call_row["target_node_id"]] += 18.0

        node_rows = _load_rows_by_ids(
            conn,
            "nodes",
            "node_id",
            list(scores),
            "node_id, path, name, kind, summary, description, primary_category, symbol_count",
        )
        for node_id, row in node_rows.items():
            scores[node_id] += _node_query_bonus(row, plan)
        return scores

    def _anchor_symbols(
        self,
        conn: sqlite3.Connection,
        symbol_hits: list[tuple[str, float]],
        plan: QueryPlan,
        *,
        limit: int,
    ) -> list[tuple[str, float]]:
        if not symbol_hits:
            return []

        symbol_rows = _load_rows_by_ids(
            conn,
            "symbols",
            "symbol_id",
            [symbol_id for symbol_id, _ in symbol_hits],
            "symbol_id, name, qualified_name, normalized_name",
        )
        token_set = set(plan.tokens)
        exact_matches: list[tuple[str, float]] = []
        for symbol_id, score in symbol_hits:
            row = symbol_rows.get(symbol_id)
            if row is None:
                continue
            if (
                row["normalized_name"] == plan.normalized
                or row["normalized_name"] in token_set
                or row["name"].lower() in token_set
                or row["qualified_name"].lower() in token_set
            ):
                exact_matches.append((symbol_id, score))

        if exact_matches:
            return exact_matches[:limit]
        return symbol_hits[:limit]

    def _score_symbols(self, conn: sqlite3.Connection, plan: QueryPlan, limit: int) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for token in plan.tokens:
            like_value = f"%{token}%"
            for row in conn.execute(
                "SELECT symbol_id FROM symbols WHERE lower(name) = ? OR lower(qualified_name) = ? OR normalized_name = ?",
                (token, token, token),
            ).fetchall():
                scores[row["symbol_id"]] += 26.0
            for row in conn.execute(
                "SELECT symbol_id FROM symbols WHERE lower(name) LIKE ? OR lower(qualified_name) LIKE ?",
                (like_value, like_value),
            ).fetchall():
                scores[row["symbol_id"]] += 9.0
            for row in conn.execute("SELECT symbol_id FROM symbols WHERE lower(path) LIKE ?", (like_value,)).fetchall():
                scores[row["symbol_id"]] += 2.0
            for row in conn.execute(
                "SELECT symbol_id FROM symbols WHERE lower(signature) LIKE ? OR lower(COALESCE(docstring, '')) LIKE ?",
                (like_value, like_value),
            ).fetchall():
                scores[row["symbol_id"]] += 3.0

        if plan.is_identifier_query and plan.normalized:
            for row in conn.execute(
                "SELECT symbol_id FROM symbols WHERE normalized_name = ? OR lower(name) = ? OR lower(qualified_name) = ?",
                (plan.normalized, plan.lowered, plan.lowered),
            ).fetchall():
                scores[row["symbol_id"]] += 64.0

        if _table_exists(conn, "symbol_search") and plan.tokens:
            match_query = " OR ".join(f'"{token}"*' for token in plan.tokens)
            for row in conn.execute(
                "SELECT symbol_id, bm25(symbol_search) AS rank FROM symbol_search WHERE symbol_search MATCH ? LIMIT ?",
                (match_query, limit * 5),
            ).fetchall():
                scores[row["symbol_id"]] += max(0.25, 5.5 - float(row["rank"]))

        symbol_rows = _load_rows_by_ids(
            conn,
            "symbols",
            "symbol_id",
            list(scores),
            "symbol_id, name, qualified_name, normalized_name, path, kind, signature",
        )
        for symbol_id, row in symbol_rows.items():
            scores[symbol_id] += _symbol_query_bonus(row, plan)
        return scores

    def _top_hits(self, scores: dict[str, float], limit: int) -> list[tuple[str, float]]:
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def axe_retrieval(db_path: str | Path, query: str, *, limit: int = 5) -> RetrievalResult:
    return AxeRetriever(db_path).retrieve(query, limit=limit)


def build_query_plan(query: str) -> QueryPlan:
    lowered = query.strip().lower()
    normalized = _normalize_identifier(query)
    tokens = tokenize_query(query)
    wants_callers = any(phrase in lowered for phrase in ("who calls", "called by", "who uses", "used by"))
    wants_callees = any(phrase in lowered for phrase in ("what does", "calls from", "callees of"))
    wants_implementation = any(word in lowered for word in ("implement", "implementation", "defined", "definition", "where is", "how does", "how are"))
    prefers_files = any(word in lowered for word in ("file", "folder", "module", "repo", "where"))
    is_identifier_query = bool(IDENTIFIER_QUERY_PATTERN.fullmatch(query.strip())) and " " not in query.strip()
    return QueryPlan(
        raw=query,
        lowered=lowered,
        normalized=normalized,
        tokens=tokens,
        is_identifier_query=is_identifier_query,
        wants_callers=wants_callers,
        wants_callees=wants_callees,
        wants_implementation=wants_implementation,
        prefers_files=prefers_files,
    )


def tokenize_query(query: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in TOKEN_PATTERN.findall(query):
        expanded = _expand_query_token(raw_token)
        tokens.extend(token for token in expanded if token and token not in STOPWORDS)

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _json_loads(value: str | None) -> list[str]:
    if not value:
        return []
    return json.loads(value)


def _expand_query_token(token: str) -> list[str]:
    lowered = token.lower()
    pieces = [piece for piece in CAMEL_CASE_BOUNDARY.sub(" ", token).replace("-", " ").split() if piece]
    results: list[str] = []
    for candidate in [lowered, *[piece.lower() for piece in pieces]]:
        results.extend(_normalize_token_variants(candidate))
    normalized = _normalize_identifier(token)
    if normalized:
        results.append(normalized)
    return results


def _normalize_token_variants(token: str) -> list[str]:
    variants = {token}
    if len(token) > 3 and token.endswith("ies"):
        variants.add(token[:-3] + "y")
    elif len(token) > 3 and token.endswith("s"):
        variants.add(token[:-1])
    if len(token) > 4 and token.endswith("ing"):
        variants.add(token[:-3])
    if len(token) > 3 and token.endswith("ed"):
        variants.add(token[:-2])

    expanded = set(variants)
    for variant in tuple(variants):
        expanded.update(TOKEN_ALIASES.get(variant, ()))
    return [variant for variant in expanded if variant and variant not in STOPWORDS]


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", value).lower()


def _symbol_query_bonus(row: sqlite3.Row, plan: QueryPlan) -> float:
    name = row["name"].lower()
    qualified_name = row["qualified_name"].lower()
    normalized_name = row["normalized_name"]
    path = row["path"].lower()
    signature = (row["signature"] or "").lower()
    identifier_text = f"{name} {qualified_name}"

    identifier_match_count = sum(1 for token in plan.tokens if token in identifier_text)
    signature_match_count = sum(1 for token in plan.tokens if token in signature)
    path_match_count = sum(1 for token in plan.tokens if token in path)

    bonus = identifier_match_count * 3.5
    bonus += signature_match_count * 1.25
    bonus += path_match_count * 0.35

    if plan.is_identifier_query and normalized_name == plan.normalized:
        bonus += 80.0
    if plan.lowered == name:
        bonus += 32.0
    if plan.lowered == qualified_name:
        bonus += 24.0
    if len(plan.tokens) > 1 and row["kind"] in CALLABLE_KINDS:
        bonus += 4.0
    if len(plan.tokens) > 1 and row["kind"] == "variable" and not _looks_callable(signature):
        bonus -= 6.0
    if len(plan.tokens) > 1 and not identifier_match_count and path_match_count:
        bonus -= 8.0
    if len(plan.tokens) > 1 and name.startswith("_"):
        bonus -= 4.0
    if plan.wants_implementation and path.endswith(".d.ts"):
        bonus -= 10.0
    return bonus


def _node_query_bonus(row: sqlite3.Row, plan: QueryPlan) -> float:
    summary = (row["summary"] or "").lower()
    description = (row["description"] or "").lower()
    primary_category = (row["primary_category"] or "").lower()
    text = f"{row['path'].lower()} {row['name'].lower()} {summary} {description} {primary_category}"

    bonus = sum(1.75 for token in plan.tokens if token in text)
    if plan.prefers_files and row["kind"] == "file":
        bonus += 3.0
    if plan.wants_implementation and row["kind"] == "file":
        bonus += 4.0
    if plan.wants_callers and row["kind"] == "file":
        bonus += 2.0
    if plan.wants_implementation and row["path"].endswith(".d.ts"):
        bonus -= 10.0
    if plan.wants_implementation and row["symbol_count"] == 0:
        bonus -= 4.0
    if plan.wants_implementation and row["kind"] == "repo":
        bonus -= 6.0
    return bonus


def _looks_callable(signature: str) -> bool:
    compact = " ".join(signature.split())
    return "function " in compact or "=>" in compact or compact.endswith(")")


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
    rows = conn.execute(f"SELECT {columns} FROM {table_name} WHERE {id_column} IN ({placeholders})", ids).fetchall()
    return {row[id_column]: row for row in rows}


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", (table_name,)).fetchone()
    return row is not None