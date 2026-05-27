from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from cradle.graph_models import AnalysisSummary, RepositoryGraph

logger = logging.getLogger(__name__)


class CradleDatabase:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=OFF")
            self._create_schema(conn)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def replace_graph(self, graph: RepositoryGraph) -> AnalysisSummary:
        self.initialize()
        with self.connect() as conn:
            self._clear_graph(conn)
            self._insert_graph(conn, graph)
            self._rebuild_search_indexes(conn)
            conn.commit()
        summary = summarize_graph(graph)
        logger.info("persisted SQLite graph to %s", self._path)
        return summary

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            DROP TABLE IF EXISTS edges;
            DROP TABLE IF EXISTS symbol_references;
            DROP TABLE IF EXISTS call_sites;
            DROP TABLE IF EXISTS imports;
            DROP TABLE IF EXISTS node_context;
            DROP TABLE IF EXISTS symbols;
            DROP TABLE IF EXISTS node_tags;
            DROP TABLE IF EXISTS node_categories;
            DROP TABLE IF EXISTS nodes;
            DROP TABLE IF EXISTS repos;
            DROP TABLE IF EXISTS meta;
            DROP TABLE IF EXISTS node_search;
            DROP TABLE IF EXISTS symbol_search;
            DROP TABLE IF EXISTS "references";

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS repos (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                root_path TEXT NOT NULL,
                language_json TEXT NOT NULL,
                summary TEXT NOT NULL,
                category TEXT,
                tags_json TEXT NOT NULL,
                content_hash TEXT,
                indexed_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                repo_id TEXT NOT NULL,
                path TEXT NOT NULL,
                name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                kind TEXT NOT NULL,
                parent_id TEXT,
                language TEXT,
                summary TEXT NOT NULL,
                description TEXT NOT NULL,
                primary_category TEXT,
                top_child_categories_json TEXT NOT NULL,
                top_dependency_tags_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                start_line INTEGER,
                start_column INTEGER,
                end_line INTEGER,
                end_column INTEGER,
                symbol_count INTEGER NOT NULL,
                import_count INTEGER NOT NULL,
                file_count INTEGER NOT NULL,
                folder_count INTEGER NOT NULL,
                content_hash TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS node_categories (
                node_id TEXT NOT NULL,
                category TEXT NOT NULL,
                rank INTEGER NOT NULL,
                PRIMARY KEY (node_id, category)
            );

            CREATE TABLE IF NOT EXISTS node_tags (
                node_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                rank INTEGER NOT NULL,
                PRIMARY KEY (node_id, tag)
            );

            CREATE TABLE IF NOT EXISTS symbols (
                symbol_id TEXT PRIMARY KEY,
                repo_id TEXT NOT NULL,
                symbol_key TEXT NOT NULL,
                node_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                path TEXT NOT NULL,
                name TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                kind TEXT NOT NULL,
                signature TEXT NOT NULL,
                category TEXT,
                summary TEXT,
                tags_json TEXT NOT NULL,
                parent_name TEXT,
                return_type TEXT,
                docstring TEXT,
                parameters_json TEXT NOT NULL,
                decorators_json TEXT NOT NULL,
                base_classes_json TEXT NOT NULL,
                start_line INTEGER,
                start_column INTEGER,
                end_line INTEGER,
                end_column INTEGER,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS imports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT NOT NULL,
                importer_node_id TEXT NOT NULL,
                imported_module TEXT NOT NULL,
                target_node_id TEXT,
                is_internal INTEGER NOT NULL,
                strength_label TEXT NOT NULL,
                strength_weight REAL NOT NULL,
                names_json TEXT NOT NULL,
                start_line INTEGER,
                start_column INTEGER,
                end_line INTEGER,
                end_column INTEGER,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS call_sites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT NOT NULL,
                caller_symbol_id TEXT NOT NULL,
                caller_node_id TEXT NOT NULL,
                callee_name TEXT NOT NULL,
                target_symbol_id TEXT,
                target_node_id TEXT,
                start_line INTEGER,
                start_column INTEGER,
                end_line INTEGER,
                end_column INTEGER,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS symbol_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT NOT NULL,
                target_symbol_id TEXT,
                target_node_id TEXT,
                target_name TEXT NOT NULL,
                source_node_id TEXT NOT NULL,
                source_symbol_id TEXT,
                reference_kind TEXT NOT NULL,
                start_line INTEGER,
                start_column INTEGER,
                end_line INTEGER,
                end_column INTEGER,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS "references" (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT NOT NULL,
                symbol_key TEXT,
                ref_path TEXT NOT NULL,
                ref_line INTEGER,
                ref_col INTEGER,
                ref_kind TEXT NOT NULL,
                context TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS node_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                source_node_id TEXT NOT NULL,
                node_path TEXT NOT NULL,
                source_path TEXT NOT NULL,
                relation TEXT NOT NULL,
                strength_label TEXT NOT NULL,
                strength_weight REAL NOT NULL,
                weight REAL NOT NULL,
                inherited_summary TEXT NOT NULL,
                inherited_category TEXT,
                inherited_tags_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT NOT NULL,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                strength TEXT NOT NULL,
                from_line INTEGER,
                to_line INTEGER,
                detail TEXT,
                updated_at TEXT NOT NULL
            );
            """
        )

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);
            CREATE INDEX IF NOT EXISTS idx_nodes_category ON nodes(primary_category);
            CREATE INDEX IF NOT EXISTS idx_nodes_repo ON nodes(repo_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_normalized_name ON nodes(normalized_name);
            CREATE INDEX IF NOT EXISTS idx_node_categories_category ON node_categories(category);
            CREATE INDEX IF NOT EXISTS idx_node_tags_tag ON node_tags(tag);
            CREATE INDEX IF NOT EXISTS idx_symbols_node_id ON symbols(node_id);
            CREATE INDEX IF NOT EXISTS idx_symbols_repo ON symbols(repo_id);
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(normalized_name);
            CREATE INDEX IF NOT EXISTS idx_symbols_symbol_key ON symbols(symbol_key);
            CREATE INDEX IF NOT EXISTS idx_imports_importer ON imports(importer_node_id);
            CREATE INDEX IF NOT EXISTS idx_imports_target ON imports(target_node_id);
            CREATE INDEX IF NOT EXISTS idx_calls_caller ON call_sites(caller_symbol_id);
            CREATE INDEX IF NOT EXISTS idx_calls_target ON call_sites(target_symbol_id);
            CREATE INDEX IF NOT EXISTS idx_refs_target ON symbol_references(target_symbol_id, target_name);
            CREATE INDEX IF NOT EXISTS idx_refs_source ON symbol_references(source_node_id, source_symbol_id);
            CREATE INDEX IF NOT EXISTS idx_public_refs_symbol ON "references"(symbol_key);
            CREATE INDEX IF NOT EXISTS idx_public_refs_path ON "references"(ref_path);
            CREATE INDEX IF NOT EXISTS idx_context_node ON node_context(node_id);
            CREATE INDEX IF NOT EXISTS idx_context_source ON node_context(source_node_id);
            CREATE INDEX IF NOT EXISTS idx_context_repo ON node_context(repo_id);
            CREATE INDEX IF NOT EXISTS idx_edges_repo ON edges(repo_id);
            CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id, edge_type);
            CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id, edge_type);
            """
        )

        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS node_search USING fts5(
                    node_id UNINDEXED,
                    path,
                    name,
                    summary,
                    description,
                    primary_category,
                    categories,
                    tags,
                    contexts
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS symbol_search USING fts5(
                    symbol_id UNINDEXED,
                    node_id UNINDEXED,
                    path,
                    name,
                    qualified_name,
                    signature,
                    docstring
                )
                """
            )
        except sqlite3.OperationalError:
            logger.warning("SQLite FTS5 is unavailable; retrieval will use indexed fallback queries")

    def _clear_graph(self, conn: sqlite3.Connection) -> None:
        statements = [
            "DELETE FROM meta",
            "DELETE FROM repos",
            "DELETE FROM edges",
            "DELETE FROM \"references\"",
            "DELETE FROM node_context",
            "DELETE FROM symbol_references",
            "DELETE FROM call_sites",
            "DELETE FROM imports",
            "DELETE FROM node_tags",
            "DELETE FROM node_categories",
            "DELETE FROM symbols",
            "DELETE FROM nodes",
        ]
        if _table_exists(conn, "node_search"):
            statements.insert(1, "DELETE FROM node_search")
        if _table_exists(conn, "symbol_search"):
            statements.insert(2, "DELETE FROM symbol_search")
        conn.executescript(";\n".join(statements) + ";")

    def _insert_graph(self, conn: sqlite3.Connection, graph: RepositoryGraph) -> None:
        timestamp = _utc_now()
        repo_id = graph.repo_root
        repo_node = next((node for node in graph.nodes if node.kind == "repo"), None)
        languages = sorted({node.language for node in graph.nodes if node.language})
        child_category_map = _top_child_categories(graph)
        dependency_tag_map = _top_dependency_tags(graph)
        symbol_start_lines = {symbol.symbol_id: symbol.start_line for symbol in graph.symbols}

        conn.executemany(
            "INSERT INTO meta(key, value) VALUES(?, ?)",
            [("repo_root", graph.repo_root), ("updated_at", timestamp)],
        )

        conn.execute(
            """
            INSERT INTO repos(id, name, root_path, language_json, summary, category, tags_json, content_hash, indexed_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                repo_id,
                Path(graph.repo_root).name,
                graph.repo_root,
                _json(languages),
                repo_node.summary if repo_node is not None else "",
                repo_node.primary_category if repo_node is not None else None,
                _json(repo_node.tags if repo_node is not None else []),
                _repo_content_hash(graph),
                timestamp,
                timestamp,
            ),
        )

        conn.executemany(
            """
            INSERT INTO nodes(
                node_id, repo_id, path, name, normalized_name, kind, parent_id, language, summary, description,
                primary_category, top_child_categories_json, top_dependency_tags_json, confidence,
                start_line, start_column, end_line, end_column, symbol_count, import_count,
                file_count, folder_count, content_hash, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    node.node_id,
                    repo_id,
                    node.path,
                    node.name,
                    node.name.lower(),
                    node.kind,
                    node.parent_id,
                    node.language,
                    node.summary,
                    node.description,
                    node.primary_category,
                    _json(child_category_map.get(node.node_id, [])),
                    _json(dependency_tag_map.get(node.node_id, [])),
                    node.confidence,
                    node.start_line,
                    node.start_column,
                    node.end_line,
                    node.end_column,
                    node.symbol_count,
                    node.import_count,
                    node.file_count,
                    node.folder_count,
                    node.content_hash,
                    timestamp,
                )
                for node in graph.nodes
            ],
        )

        conn.executemany(
            "INSERT INTO node_categories(node_id, category, rank) VALUES(?, ?, ?)",
            [
                (node.node_id, category, rank)
                for node in graph.nodes
                for rank, category in enumerate(node.categories)
            ],
        )
        conn.executemany(
            "INSERT INTO node_tags(node_id, tag, rank) VALUES(?, ?, ?)",
            [
                (node.node_id, tag, rank)
                for node in graph.nodes
                for rank, tag in enumerate(node.tags)
            ],
        )

        conn.executemany(
            """
            INSERT INTO symbols(
                symbol_id, repo_id, symbol_key, node_id, file_path, path, name, qualified_name, normalized_name, kind,
                signature, category, summary, tags_json, parent_name, return_type, docstring, parameters_json,
                decorators_json, base_classes_json, start_line, start_column, end_line, end_column, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    symbol.symbol_id,
                    repo_id,
                    symbol.symbol_id,
                    symbol.node_id,
                    symbol.path,
                    symbol.path,
                    symbol.name,
                    symbol.qualified_name,
                    symbol.normalized_name,
                    symbol.kind,
                    symbol.signature,
                    None,
                    None,
                    _json([]),
                    symbol.parent_name,
                    symbol.return_type,
                    symbol.docstring,
                    _json(symbol.parameters),
                    _json(symbol.decorators),
                    _json(symbol.base_classes),
                    symbol.start_line,
                    symbol.start_column,
                    symbol.end_line,
                    symbol.end_column,
                    timestamp,
                )
                for symbol in graph.symbols
            ],
        )

        conn.executemany(
            """
            INSERT INTO imports(
                repo_id, importer_node_id, imported_module, target_node_id, is_internal, strength_label,
                strength_weight, names_json, start_line, start_column, end_line, end_column, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    repo_id,
                    record.importer_node_id,
                    record.imported_module,
                    record.target_node_id,
                    1 if record.is_internal else 0,
                    record.strength_label,
                    record.strength_weight,
                    _json(record.names),
                    record.start_line,
                    record.start_column,
                    record.end_line,
                    record.end_column,
                    timestamp,
                )
                for record in graph.imports
            ],
        )

        conn.executemany(
            """
            INSERT INTO call_sites(
                repo_id, caller_symbol_id, caller_node_id, callee_name, target_symbol_id, target_node_id,
                start_line, start_column, end_line, end_column, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    repo_id,
                    record.caller_symbol_id,
                    record.caller_node_id,
                    record.callee_name,
                    record.target_symbol_id,
                    record.target_node_id,
                    record.start_line,
                    record.start_column,
                    record.end_line,
                    record.end_column,
                    timestamp,
                )
                for record in graph.calls
            ],
        )

        conn.executemany(
            """
            INSERT INTO symbol_references(
                repo_id, target_symbol_id, target_node_id, target_name, source_node_id, source_symbol_id,
                reference_kind, start_line, start_column, end_line, end_column, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    repo_id,
                    record.target_symbol_id,
                    record.target_node_id,
                    record.target_name,
                    record.source_node_id,
                    record.source_symbol_id,
                    record.reference_kind,
                    record.start_line,
                    record.start_column,
                    record.end_line,
                    record.end_column,
                    timestamp,
                )
                for record in graph.references
            ],
        )

        conn.executemany(
            """
            INSERT INTO "references"(
                repo_id, symbol_key, ref_path, ref_line, ref_col, ref_kind, context, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    repo_id,
                    record.target_symbol_id,
                    record.source_node_id,
                    record.start_line,
                    record.start_column,
                    record.reference_kind,
                    _json(
                        {
                            "target_name": record.target_name,
                            "target_node_id": record.target_node_id,
                            "source_symbol_id": record.source_symbol_id,
                            "end_line": record.end_line,
                            "end_column": record.end_column,
                        }
                    ),
                    timestamp,
                )
                for record in graph.references
            ],
        )

        conn.executemany(
            """
            INSERT INTO node_context(
                repo_id, node_id, source_node_id, node_path, source_path, relation, strength_label, strength_weight,
                weight, inherited_summary, inherited_category, inherited_tags_json, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    repo_id,
                    record.node_id,
                    record.source_node_id,
                    record.node_id,
                    record.source_node_id,
                    "internal_import",
                    record.strength_label,
                    record.strength_weight,
                    record.strength_weight,
                    record.inherited_summary,
                    record.inherited_category,
                    _json(record.inherited_tags),
                    timestamp,
                )
                for record in graph.node_context
            ],
        )

        edge_rows: list[tuple[str, str, str, str, str, int | None, int | None, str | None, str]] = []
        for node in graph.nodes:
            if node.parent_id is None:
                continue
            edge_rows.append((repo_id, node.parent_id, node.node_id, "child", "structural", None, node.start_line, None, timestamp))

        for record in graph.imports:
            edge_rows.append(
                (
                    repo_id,
                    record.importer_node_id,
                    record.target_node_id or record.imported_module,
                    "import",
                    record.strength_label,
                    record.start_line,
                    None,
                    _json({"module": record.imported_module, "names": record.names, "is_internal": record.is_internal}),
                    timestamp,
                )
            )

        for record in graph.calls:
            edge_rows.append(
                (
                    repo_id,
                    record.caller_symbol_id,
                    record.target_symbol_id or record.callee_name,
                    "call",
                    "strong",
                    record.start_line,
                    symbol_start_lines.get(record.target_symbol_id),
                    _json({"callee_name": record.callee_name, "target_node_id": record.target_node_id, "caller_node_id": record.caller_node_id}),
                    timestamp,
                )
            )

        conn.executemany(
            "INSERT INTO edges(repo_id, from_id, to_id, edge_type, strength, from_line, to_line, detail, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            edge_rows,
        )

    def _rebuild_search_indexes(self, conn: sqlite3.Connection) -> None:
        if not _table_exists(conn, "node_search") or not _table_exists(conn, "symbol_search"):
            return

        conn.execute("DELETE FROM node_search")
        conn.execute("DELETE FROM symbol_search")

        node_rows = conn.execute(
            """
            SELECT
                nodes.node_id,
                nodes.path,
                nodes.name,
                nodes.summary,
                nodes.description,
                COALESCE(nodes.primary_category, '') AS primary_category,
                COALESCE(GROUP_CONCAT(DISTINCT node_categories.category), '') AS categories,
                COALESCE(GROUP_CONCAT(DISTINCT node_tags.tag), '') AS tags
            FROM nodes
            LEFT JOIN node_categories ON node_categories.node_id = nodes.node_id
            LEFT JOIN node_tags ON node_tags.node_id = nodes.node_id
            GROUP BY nodes.node_id
            """
        ).fetchall()
        context_map = {
            row["node_id"]: row["contexts"]
            for row in conn.execute(
                """
                SELECT node_id, GROUP_CONCAT(inherited_summary, ' ') AS contexts
                FROM node_context
                GROUP BY node_id
                """
            ).fetchall()
        }
        conn.executemany(
            "INSERT INTO node_search(node_id, path, name, summary, description, primary_category, categories, tags, contexts) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["node_id"],
                    row["path"],
                    row["name"],
                    row["summary"],
                    row["description"],
                    row["primary_category"],
                    row["categories"],
                    row["tags"],
                    context_map.get(row["node_id"], ""),
                )
                for row in node_rows
            ],
        )

        symbol_rows = conn.execute(
            "SELECT symbol_id, node_id, path, name, qualified_name, signature, COALESCE(docstring, '') AS docstring FROM symbols"
        ).fetchall()
        conn.executemany(
            "INSERT INTO symbol_search(symbol_id, node_id, path, name, qualified_name, signature, docstring) VALUES(?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["symbol_id"],
                    row["node_id"],
                    row["path"],
                    row["name"],
                    row["qualified_name"],
                    row["signature"],
                    row["docstring"],
                )
                for row in symbol_rows
            ],
        )


def summarize_graph(graph: RepositoryGraph) -> AnalysisSummary:
    repo_node = next((node for node in graph.nodes if node.kind == "repo"), None)
    repo_categories = repo_node.categories if repo_node is not None else []
    repo_summary = repo_node.summary if repo_node is not None else ""
    return AnalysisSummary(
        repo_root=graph.repo_root,
        file_count=sum(1 for node in graph.nodes if node.kind == "file"),
        folder_count=sum(1 for node in graph.nodes if node.kind == "folder"),
        symbol_count=len(graph.symbols),
        import_count=len(graph.imports),
        call_count=len(graph.calls),
        reference_count=len(graph.references),
        repo_summary=repo_summary,
        repo_categories=repo_categories,
    )


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", (table_name,)).fetchone()
    return row is not None


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _repo_content_hash(graph: RepositoryGraph) -> str:
    digest = sha256()
    file_nodes = sorted((node for node in graph.nodes if node.kind == "file"), key=lambda item: item.node_id)
    for node in file_nodes:
        digest.update(node.node_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update((node.content_hash or "").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _top_child_categories(graph: RepositoryGraph) -> dict[str, list[str]]:
    children_by_parent: dict[str, Counter[str]] = defaultdict(Counter)
    for node in graph.nodes:
        if node.parent_id is None or not node.primary_category:
            continue
        children_by_parent[node.parent_id][node.primary_category] += 1
    return {node_id: [category for category, _ in counter.most_common(5)] for node_id, counter in children_by_parent.items()}


def _top_dependency_tags(graph: RepositoryGraph) -> dict[str, list[str]]:
    tags_by_node: dict[str, Counter[str]] = defaultdict(Counter)
    for record in graph.node_context:
        for tag in record.inherited_tags:
            tags_by_node[record.node_id][tag] += 1
    return {node_id: [tag for tag, _ in counter.most_common(8)] for node_id, counter in tags_by_node.items()}