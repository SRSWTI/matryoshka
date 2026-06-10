from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from matryoshka.graph_models import (
    CallRecord,
    CodeNode,
    CodeSymbol,
    ImportRecord,
    NodeContextRecord,
    SymbolReferenceRecord,
)


@dataclass(slots=True)
class FileReadResult:
    """Rich read result for a single file from the Matryoshka DB."""

    node: CodeNode
    symbols: list[CodeSymbol] = field(default_factory=list)
    imports: list[ImportRecord] = field(default_factory=list)
    exports: list[CodeSymbol] = field(default_factory=list)
    called_by: list[CallRecord] = field(default_factory=list)
    callees: list[CallRecord] = field(default_factory=list)
    references: list[SymbolReferenceRecord] = field(default_factory=list)
    reverse_references: list[SymbolReferenceRecord] = field(default_factory=list)
    contexts: list[NodeContextRecord] = field(default_factory=list)
    repo_summary: str = ""
    repo_categories: list[str] = field(default_factory=list)
    source_lines: list[str] = field(default_factory=list)
    symbol_blocks: list[str] = field(default_factory=list)
    import_lines: list[str] = field(default_factory=list)


def _json_loads(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


class FileReader:
    """Queries the Matryoshka SQLite DB for rich file-level detail.

    Use ``read`` for a structured summary (symbols, imports, calls, references)
    and ``read_more`` to also get collapsed source-code blocks for every
    top-level function and class with line numbers.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def read(self, file_path: str) -> FileReadResult:
        """Return a rich summary for *file_path* from the DB.

        Includes: node metadata, symbols, imports, calls, references,
        reverse references, and related context.
        """
        with self._connect() as conn:
            return self._read_impl(conn, file_path, include_source=False)

    def read_more(self, file_path: str) -> FileReadResult:
        """Like ``read`` but also collapses top-level functions/classes
        into formatted source-code blocks with line numbers and includes
        the raw source lines for the file."""
        with self._connect() as conn:
            return self._read_impl(conn, file_path, include_source=True)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _read_impl(
        self, conn: sqlite3.Connection, file_path: str, *, include_source: bool
    ) -> FileReadResult:
        node = self._load_node(conn, file_path)
        symbols = self._load_symbols(conn, node.node_id)
        imports = self._load_imports(conn, node.node_id)
        called_by = self._load_called_by(conn, node.node_id)
        callees = self._load_callees(conn, node.node_id)
        references = self._load_references(conn, node.node_id)
        reverse_refs = self._load_reverse_references(conn, node.node_id)
        contexts = self._load_contexts(conn, node.node_id)
        repo = self._load_repo_summary(conn)
        exports = self._load_exports(conn, node.node_id)

        result = FileReadResult(
            node=node,
            symbols=symbols,
            imports=imports,
            exports=exports,
            called_by=called_by,
            callees=callees,
            references=references,
            reverse_references=reverse_refs,
            contexts=contexts,
            repo_summary=repo.get("summary", "") if repo else "",
            repo_categories=_json_loads(repo.get("tags_json", "[]")) if repo else [],
        )

        if include_source:
            result.source_lines = self._load_source_lines(node)
            result.symbol_blocks = self._build_symbol_blocks(
                symbols, result.source_lines
            )
            result.import_lines = self._build_import_lines(imports, result.source_lines)

        return result

    # ---- node ----

    def _load_node(self, conn: sqlite3.Connection, file_path: str) -> CodeNode:
        row = conn.execute(
            "SELECT * FROM nodes WHERE path = ? OR node_id = ? LIMIT 1",
            (file_path, file_path),
        ).fetchone()
        if row is None:
            raise FileNotFoundError(f"No node found for path: {file_path}")
        categories = [
            r["category"]
            for r in conn.execute(
                "SELECT category FROM node_categories WHERE node_id = ? ORDER BY rank",
                (row["node_id"],),
            ).fetchall()
        ]
        tags = [
            r["tag"]
            for r in conn.execute(
                "SELECT tag FROM node_tags WHERE node_id = ? ORDER BY rank",
                (row["node_id"],),
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

    # ---- symbols ----

    def _load_symbols(self, conn: sqlite3.Connection, node_id: str) -> list[CodeSymbol]:
        rows = conn.execute(
            "SELECT * FROM symbols WHERE node_id = ? ORDER BY start_line, name",
            (node_id,),
        ).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    def _load_exports(self, conn: sqlite3.Connection, node_id: str) -> list[CodeSymbol]:
        """Return symbols that are referenced by other files (effectively 'exported')."""
        target_ids = {
            r["target_symbol_id"]
            for r in conn.execute(
                "SELECT DISTINCT target_symbol_id FROM symbol_references "
                "WHERE target_symbol_id IS NOT NULL AND source_node_id != ?",
                (node_id,),
            ).fetchall()
            if r["target_symbol_id"]
        }
        if not target_ids:
            return []
        placeholders = ",".join("?" for _ in target_ids)
        rows = conn.execute(
            f"SELECT * FROM symbols WHERE symbol_id IN ({placeholders}) ORDER BY start_line",
            list(target_ids),
        ).fetchall()
        return [self._row_to_symbol(r) for r in rows]

    def _row_to_symbol(self, row: sqlite3.Row) -> CodeSymbol:
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

    # ---- imports ----

    def _load_imports(
        self, conn: sqlite3.Connection, node_id: str
    ) -> list[ImportRecord]:
        rows = conn.execute(
            "SELECT * FROM imports WHERE importer_node_id = ? ORDER BY start_line",
            (node_id,),
        ).fetchall()
        return [self._row_to_import(r) for r in rows]

    def _row_to_import(self, row: sqlite3.Row) -> ImportRecord:
        return ImportRecord(
            importer_node_id=row["importer_node_id"],
            imported_module=row["imported_module"],
            target_node_id=row["target_node_id"],
            is_internal=bool(row["is_internal"]),
            strength_label=row["strength_label"],
            strength_weight=row["strength_weight"],
            is_out_of_scope=bool(row["is_out_of_scope"]),
            names=_json_loads(row["names_json"]),
            start_line=row["start_line"],
            start_column=row["start_column"],
            end_line=row["end_line"],
            end_column=row["end_column"],
        )

    # ---- calls ----

    def _load_called_by(
        self, conn: sqlite3.Connection, node_id: str
    ) -> list[CallRecord]:
        """Calls INTO this file (other files calling symbols in this file)."""
        rows = conn.execute(
            "SELECT * FROM call_sites WHERE target_node_id = ? ORDER BY start_line",
            (node_id,),
        ).fetchall()
        return [self._row_to_call(r) for r in rows]

    def _load_callees(self, conn: sqlite3.Connection, node_id: str) -> list[CallRecord]:
        """Calls OUT of this file (symbols in this file calling others)."""
        rows = conn.execute(
            "SELECT * FROM call_sites WHERE caller_node_id = ? ORDER BY start_line",
            (node_id,),
        ).fetchall()
        return [self._row_to_call(r) for r in rows]

    def _row_to_call(self, row: sqlite3.Row) -> CallRecord:
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

    # ---- references ----

    def _load_references(
        self, conn: sqlite3.Connection, node_id: str
    ) -> list[SymbolReferenceRecord]:
        """References FROM this node to others."""
        rows = conn.execute(
            "SELECT * FROM symbol_references WHERE source_node_id = ? ORDER BY start_line",
            (node_id,),
        ).fetchall()
        return [self._row_to_ref(r) for r in rows]

    def _load_reverse_references(
        self, conn: sqlite3.Connection, node_id: str
    ) -> list[SymbolReferenceRecord]:
        """References INTO this node from others."""
        rows = conn.execute(
            "SELECT * FROM symbol_references WHERE target_node_id = ? ORDER BY start_line",
            (node_id,),
        ).fetchall()
        return [self._row_to_ref(r) for r in rows]

    def _row_to_ref(self, row: sqlite3.Row) -> SymbolReferenceRecord:
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

    # ---- context ----

    def _load_contexts(
        self, conn: sqlite3.Connection, node_id: str
    ) -> list[NodeContextRecord]:
        rows = conn.execute(
            "SELECT * FROM node_context WHERE node_id = ? ORDER BY strength_weight DESC",
            (node_id,),
        ).fetchall()
        return [
            NodeContextRecord(
                node_id=r["node_id"],
                source_node_id=r["source_node_id"],
                strength_label=r["strength_label"],
                strength_weight=r["strength_weight"],
                inherited_summary=r["inherited_summary"],
                inherited_category=r["inherited_category"],
                inherited_tags=_json_loads(r["inherited_tags_json"]),
            )
            for r in rows
        ]

    # ---- repo ----

    def _load_repo_summary(self, conn: sqlite3.Connection) -> dict | None:
        row = conn.execute(
            "SELECT summary, tags_json FROM repos ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {"summary": row["summary"], "tags_json": row["tags_json"]}

    # ---- source code helpers (read_more only) ----

    def _load_source_lines(self, node: CodeNode) -> list[str]:
        """Read the actual source file and return 1-indexed lines."""
        repo_root = self._resolve_repo_root(node)
        if repo_root is None:
            return []
        file_path = repo_root / node.path
        if not file_path.is_file():
            return []
        try:
            return file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []

    def _resolve_repo_root(self, node: CodeNode) -> Path | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT root_path FROM repos ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return Path(row["root_path"])

    def _build_symbol_blocks(
        self, symbols: list[CodeSymbol], source_lines: list[str]
    ) -> list[str]:
        """Build collapsed source-code blocks for top-level symbols
        (functions, classes, methods, structs, enums, etc.)."""
        if not source_lines:
            return []

        blocks: list[str] = []
        # Only include top-level symbols (no parent_name) + struct fields / enum variants
        top_level = [
            s
            for s in symbols
            if not s.parent_name or s.kind in ("field", "enum_variant")
        ]

        for sym in top_level:
            start = (sym.start_line or 1) - 1  # 0-indexed
            end = (sym.end_line or len(source_lines)) - 1
            if start < 0:
                start = 0
            if end >= len(source_lines):
                end = len(source_lines) - 1

            chunk = source_lines[start : end + 1]
            if not chunk:
                continue

            # Build collapsed representation
            lines = [f"-- {sym.qualified_name} (L{sym.start_line}-{sym.end_line})"]
            lines.append(f"-- kind={sym.kind}")
            if sym.signature:
                lines.append(f"-- signature: {sym.signature}")
            if sym.docstring:
                first_line = sym.docstring.strip().split("\n")[0][:120]
                lines.append(f"-- doc: {first_line}")
            lines.append("")

            # Truncate to first 30 lines of the body, then show "... truncated"
            max_display = 30
            if len(chunk) <= max_display:
                lines.extend(chunk)
            else:
                lines.extend(chunk[:max_display])
                lines.append(
                    f"-- ... ({len(chunk) - max_display} more lines, end L{sym.end_line})"
                )

            lines.append("")
            blocks.append("\n".join(lines))

        return blocks

    def _build_import_lines(
        self, imports: list[ImportRecord], source_lines: list[str]
    ) -> list[str]:
        """Extract actual import lines from source for verification."""
        if not source_lines or not imports:
            return []

        lines: list[str] = []
        for imp in imports:
            if imp.start_line is None:
                continue
            idx = imp.start_line - 1
            if idx < 0 or idx >= len(source_lines):
                continue
            raw = source_lines[idx].rstrip()
            internal = "internal" if imp.is_internal else "external"
            lines.append(f"  L{imp.start_line}: {raw}  [{internal}]")
        return lines
