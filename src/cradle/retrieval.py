from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from cradle.graph_models import CallRecord, CodeNode, CodeSymbol, ImportRecord, NodeContextRecord, RetrievalNodeHit, RetrievalResult, RetrievalSymbolHit, SymbolReferenceRecord

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


class AxeRetriever:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)

    def retrieve(self, query: str, *, limit: int = 5) -> RetrievalResult:
        with self._connect() as conn:
            node_scores = self._score_nodes(conn, query, limit)
            symbol_scores = self._score_symbols(conn, query, limit)
            node_hits = [self._load_node_hit(conn, node_id, score) for node_id, score in self._top_hits(node_scores, limit)]
            symbol_hits = [self._load_symbol_hit(conn, symbol_id, score) for symbol_id, score in self._top_hits(symbol_scores, limit)]
        logger.info("retrieved %s node hits and %s symbol hits for query %r", len(node_hits), len(symbol_hits), query)
        return RetrievalResult(query=query, node_hits=node_hits, symbol_hits=symbol_hits)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _score_nodes(self, conn: sqlite3.Connection, query: str, limit: int) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        tokens = tokenize_query(query)
        for token in tokens:
            like_value = f"%{token}%"
            for row in conn.execute(
                "SELECT node_id FROM nodes WHERE lower(path) LIKE ? OR lower(name) LIKE ?",
                (like_value, like_value),
            ).fetchall():
                scores[row["node_id"]] += 8.0
            for row in conn.execute("SELECT node_id FROM node_tags WHERE lower(tag) = ?", (token,)).fetchall():
                scores[row["node_id"]] += 4.0
            for row in conn.execute("SELECT node_id FROM node_categories WHERE lower(category) = ?", (token,)).fetchall():
                scores[row["node_id"]] += 4.0

        if _table_exists(conn, "node_search") and tokens:
            match_query = " OR ".join(f'"{token}"*' for token in tokens)
            for row in conn.execute(
                "SELECT node_id, bm25(node_search) AS rank FROM node_search WHERE node_search MATCH ? LIMIT ?",
                (match_query, limit * 3),
            ).fetchall():
                scores[row["node_id"]] += max(0.5, 6.0 - float(row["rank"]))
        return scores

    def _score_symbols(self, conn: sqlite3.Connection, query: str, limit: int) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        tokens = tokenize_query(query)
        for token in tokens:
            like_value = f"%{token}%"
            for row in conn.execute(
                "SELECT symbol_id FROM symbols WHERE lower(name) LIKE ? OR lower(qualified_name) LIKE ? OR lower(path) LIKE ?",
                (like_value, like_value, like_value),
            ).fetchall():
                scores[row["symbol_id"]] += 10.0

        if _table_exists(conn, "symbol_search") and tokens:
            match_query = " OR ".join(f'"{token}"*' for token in tokens)
            for row in conn.execute(
                "SELECT symbol_id, bm25(symbol_search) AS rank FROM symbol_search WHERE symbol_search MATCH ? LIMIT ?",
                (match_query, limit * 5),
            ).fetchall():
                scores[row["symbol_id"]] += max(0.5, 8.0 - float(row["rank"]))
        return scores

    def _load_node_hit(self, conn: sqlite3.Connection, node_id: str, score: float) -> RetrievalNodeHit:
        node = _load_node(conn, node_id)
        contexts = [
            NodeContextRecord(
                node_id=row["node_id"],
                source_node_id=row["source_node_id"],
                strength_label=row["strength_label"],
                strength_weight=row["strength_weight"],
                inherited_summary=row["inherited_summary"],
                inherited_category=row["inherited_category"],
                inherited_tags=_json_loads(row["inherited_tags_json"]),
            )
            for row in conn.execute(
                "SELECT * FROM node_context WHERE node_id = ? ORDER BY strength_weight DESC, source_node_id",
                (node_id,),
            ).fetchall()
        ]
        imports = [
            ImportRecord(
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
            for row in conn.execute(
                "SELECT * FROM imports WHERE importer_node_id = ? OR target_node_id = ? ORDER BY start_line, imported_module",
                (node_id, node_id),
            ).fetchall()
        ]
        return RetrievalNodeHit(score=score, node=node, contexts=contexts, imports=imports)

    def _load_symbol_hit(self, conn: sqlite3.Connection, symbol_id: str, score: float) -> RetrievalSymbolHit:
        symbol = _load_symbol(conn, symbol_id)
        references = [
            SymbolReferenceRecord(
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
            for row in conn.execute(
                "SELECT * FROM symbol_references WHERE target_symbol_id = ? OR target_name = ? ORDER BY start_line, source_node_id",
                (symbol_id, symbol.name),
            ).fetchall()
        ]
        callees = [
            _row_to_call(row)
            for row in conn.execute("SELECT * FROM call_sites WHERE caller_symbol_id = ? ORDER BY start_line, callee_name", (symbol_id,)).fetchall()
        ]
        called_by = [
            _row_to_call(row)
            for row in conn.execute("SELECT * FROM call_sites WHERE target_symbol_id = ? ORDER BY start_line, caller_node_id", (symbol_id,)).fetchall()
        ]
        return RetrievalSymbolHit(score=score, symbol=symbol, references=references, callees=callees, called_by=called_by)

    def _top_hits(self, scores: dict[str, float], limit: int) -> list[tuple[str, float]]:
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def axe_retrieval(db_path: str | Path, query: str, *, limit: int = 5) -> RetrievalResult:
    return AxeRetriever(db_path).retrieve(query, limit=limit)


def tokenize_query(query: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(query) if token.strip()]


def _load_node(conn: sqlite3.Connection, node_id: str) -> CodeNode:
    row = conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
    if row is None:
        raise KeyError(f"Unknown node_id: {node_id}")
    categories = [item["category"] for item in conn.execute("SELECT category FROM node_categories WHERE node_id = ? ORDER BY rank", (node_id,)).fetchall()]
    tags = [item["tag"] for item in conn.execute("SELECT tag FROM node_tags WHERE node_id = ? ORDER BY rank", (node_id,)).fetchall()]
    return CodeNode(
        node_id=row["node_id"],
        path=row["path"],
        name=row["name"],
        kind=row["kind"],
        parent_id=row["parent_id"],
        language=row["language"],
        summary=row["summary"],
        description=row["description"],
        primary_category=row["primary_category"],
        categories=categories,
        tags=tags,
        confidence=row["confidence"],
        start_line=row["start_line"],
        start_column=row["start_column"],
        end_line=row["end_line"],
        end_column=row["end_column"],
        symbol_count=row["symbol_count"],
        import_count=row["import_count"],
        file_count=row["file_count"],
        folder_count=row["folder_count"],
        content_hash=row["content_hash"],
    )


def _load_symbol(conn: sqlite3.Connection, symbol_id: str) -> CodeSymbol:
    row = conn.execute("SELECT * FROM symbols WHERE symbol_id = ?", (symbol_id,)).fetchone()
    if row is None:
        raise KeyError(f"Unknown symbol_id: {symbol_id}")
    return CodeSymbol(
        symbol_id=row["symbol_id"],
        node_id=row["node_id"],
        path=row["path"],
        name=row["name"],
        qualified_name=row["qualified_name"],
        normalized_name=row["normalized_name"],
        kind=row["kind"],
        signature=row["signature"],
        parent_name=row["parent_name"],
        return_type=row["return_type"],
        docstring=row["docstring"],
        parameters=_json_loads(row["parameters_json"]),
        decorators=_json_loads(row["decorators_json"]),
        base_classes=_json_loads(row["base_classes_json"]),
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


def _json_loads(value: str | None) -> list[str]:
    if not value:
        return []
    return json.loads(value)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", (table_name,)).fetchone()
    return row is not None