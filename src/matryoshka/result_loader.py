from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from matryoshka.graph_models import CallRecord, CodeNode, CodeSymbol, ImportRecord, NodeContextRecord, RetrievalNodeHit, RetrievalSymbolHit, SymbolReferenceRecord


class SQLiteResultLoader:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def load_node_hit(self, conn: sqlite3.Connection, node_id: str, score: float) -> RetrievalNodeHit:
        node = self.load_node(conn, node_id)
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

    def load_symbol_hit(self, conn: sqlite3.Connection, symbol_id: str, score: float) -> RetrievalSymbolHit:
        symbol = self.load_symbol(conn, symbol_id)
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
            for row in conn.execute(
                "SELECT * FROM call_sites WHERE caller_symbol_id = ? ORDER BY start_line, callee_name",
                (symbol_id,),
            ).fetchall()
        ]
        called_by = [
            _row_to_call(row)
            for row in conn.execute(
                "SELECT * FROM call_sites WHERE target_symbol_id = ? ORDER BY start_line, caller_node_id",
                (symbol_id,),
            ).fetchall()
        ]
        return RetrievalSymbolHit(score=score, symbol=symbol, references=references, callees=callees, called_by=called_by)

    def load_node(self, conn: sqlite3.Connection, node_id: str) -> CodeNode:
        row = conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown node_id: {node_id}")
        categories = [
            item["category"]
            for item in conn.execute(
                "SELECT category FROM node_categories WHERE node_id = ? ORDER BY rank",
                (node_id,),
            ).fetchall()
        ]
        tags = [
            item["tag"]
            for item in conn.execute(
                "SELECT tag FROM node_tags WHERE node_id = ? ORDER BY rank",
                (node_id,),
            ).fetchall()
        ]
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

    def load_symbol(self, conn: sqlite3.Connection, symbol_id: str) -> CodeSymbol:
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