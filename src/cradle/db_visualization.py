from __future__ import annotations

import json
import sqlite3
from pathlib import Path

TABLE_NAMES = [
    "repos",
    "nodes",
    "node_categories",
    "node_tags",
    "symbols",
    "imports",
    "call_sites",
    "symbol_references",
    "references",
    "node_context",
    "community_members",
    "theme_members",
    "edges",
]


def build_db_visualization(db_path: str | Path, *, sample_limit: int = 10) -> str:
    path = Path(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        repo_row = conn.execute("SELECT * FROM repos LIMIT 1").fetchone()
        if repo_row is None:
            raise ValueError(f"No repository graph found in {path}")

        table_counts = {table_name: conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0] for table_name in TABLE_NAMES}
        node_kind_counts = conn.execute(
            "SELECT kind, COUNT(*) AS count FROM nodes GROUP BY kind ORDER BY count DESC, kind"
        ).fetchall()
        edge_type_counts = conn.execute(
            "SELECT edge_type, COUNT(*) AS count FROM edges GROUP BY edge_type ORDER BY count DESC, edge_type"
        ).fetchall()
        import_strength_counts = conn.execute(
            "SELECT strength_label, COUNT(*) AS count FROM imports GROUP BY strength_label ORDER BY count DESC, strength_label"
        ).fetchall()
        top_files = conn.execute(
            '''
            SELECT path, COALESCE(primary_category, 'none') AS category, symbol_count, import_count, summary
            FROM nodes
            WHERE kind = 'file'
            ORDER BY symbol_count DESC, import_count DESC, path ASC
            LIMIT ?
            ''',
            (sample_limit,),
        ).fetchall()
        top_folders = conn.execute(
            '''
            SELECT path, COALESCE(primary_category, 'none') AS category, file_count, folder_count, summary
            FROM nodes
            WHERE kind = 'folder'
            ORDER BY file_count DESC, folder_count DESC, path ASC
            LIMIT ?
            ''',
            (sample_limit,),
        ).fetchall()
        top_symbols = conn.execute(
            '''
            SELECT
                symbols.qualified_name,
                symbols.path,
                symbols.kind,
                COUNT(symbol_references.id) AS reference_count
            FROM symbols
            LEFT JOIN symbol_references ON symbol_references.target_symbol_id = symbols.symbol_id
            GROUP BY symbols.symbol_id
            ORDER BY reference_count DESC, symbols.qualified_name ASC
            LIMIT ?
            ''',
            (sample_limit,),
        ).fetchall()
        top_import_edges = conn.execute(
            '''
            SELECT importer_node_id AS source_path, target_node_id AS target_path, strength_label, COUNT(*) AS weight
            FROM imports
            WHERE target_node_id IS NOT NULL
            GROUP BY importer_node_id, target_node_id, strength_label
            ORDER BY weight DESC, importer_node_id ASC, target_node_id ASC
            LIMIT ?
            ''',
            (sample_limit,),
        ).fetchall()
        # Out-of-scope: internal imports whose target lies outside the analyzed root.
        # These are real dependencies Cradle could not resolve to a file in the graph.
        # Guard against older DBs that predate the is_out_of_scope column.
        import_columns = {row[1] for row in conn.execute("PRAGMA table_info(imports)").fetchall()}
        _has_scope_col = "is_out_of_scope" in import_columns
        if _has_scope_col:
            out_of_scope_counts = conn.execute(
                '''
                SELECT imported_module, COUNT(*) AS count
                FROM imports
                WHERE is_out_of_scope = 1
                GROUP BY imported_module
                ORDER BY count DESC, imported_module ASC
                LIMIT ?
                ''',
                (sample_limit,),
            ).fetchall()
            import_scope_summary = conn.execute(
                '''
                SELECT
                    SUM(CASE WHEN is_internal = 1 AND target_node_id IS NOT NULL THEN 1 ELSE 0 END) AS resolved_internal,
                    SUM(CASE WHEN is_out_of_scope = 1 THEN 1 ELSE 0 END) AS out_of_scope,
                    SUM(CASE WHEN is_internal = 0 THEN 1 ELSE 0 END) AS external
                FROM imports
                '''
            ).fetchone()
        else:
            out_of_scope_counts = []
            import_scope_summary = None
        top_call_edges = conn.execute(
            '''
            SELECT caller_node_id AS source_path, target_node_id AS target_path, COUNT(*) AS weight
            FROM call_sites
            WHERE target_node_id IS NOT NULL
            GROUP BY caller_node_id, target_node_id
            ORDER BY weight DESC, caller_node_id ASC, target_node_id ASC
            LIMIT ?
            ''',
            (sample_limit,),
        ).fetchall()
        table_schemas = {table_name: _load_table_schema(conn, table_name) for table_name in TABLE_NAMES}
        table_samples = {table_name: _load_table_samples(conn, table_name, sample_limit=min(sample_limit, 3)) for table_name in TABLE_NAMES}
    finally:
        conn.close()

    repo_tags = json.loads(repo_row["tags_json"])
    report_lines = [
        "# Cradle DB Visualization",
        "",
        "## Repository",
        "",
        f"- Name: {repo_row['name']}",
        f"- Root: {repo_row['root_path']}",
        f"- Category: {repo_row['category'] or 'none'}",
        f"- Tags: {', '.join(repo_tags) if repo_tags else 'none'}",
        f"- Summary: {repo_row['summary']}",
        "",
        "## Table Counts",
        "",
        "| Table | Rows |",
        "| --- | ---: |",
        *[f"| {table_name} | {table_counts[table_name]} |" for table_name in TABLE_NAMES],
        "",
        "## Node Kinds",
        "",
        "| Kind | Rows |",
        "| --- | ---: |",
        *[f"| {row['kind']} | {row['count']} |" for row in node_kind_counts],
        "",
        "## Edge Types",
        "",
        "| Type | Rows |",
        "| --- | ---: |",
        *[f"| {row['edge_type']} | {row['count']} |" for row in edge_type_counts],
        "",
        "## Import Strengths",
        "",
        "| Strength | Rows |",
        "| --- | ---: |",
        *[f"| {row['strength_label']} | {row['count']} |" for row in import_strength_counts],
        "",
        "## Import Scope",
        "",
        "Resolved internal imports point to a file inside the analyzed root.  "
        "Out-of-scope imports are real internal dependencies whose target lies **outside** "
        "the portion of the repository that Cradle analysed.  "
        "External imports are third-party or stdlib packages.",
        "",
        *(
            [
                "| Category | Count |",
                "| --- | ---: |",
                f"| resolved internal | {import_scope_summary['resolved_internal'] or 0} |",
                f"| **out of scope** | {import_scope_summary['out_of_scope'] or 0} |",
                f"| external (stdlib / third-party) | {import_scope_summary['external'] or 0} |",
                "",
            ]
            if import_scope_summary is not None
            else ["_Import scope data not available (older database schema)._", ""]
        ),
        *(
            [
                "### Out-of-Scope Modules",
                "",
                "These modules were detected as internal (same package / relative path) "
                "but could not be resolved to any file in the analyzed root.",
                "",
                "| Module | Importers |",
                "| --- | ---: |",
                *[f"| `{row['imported_module']}` | {row['count']} |" for row in out_of_scope_counts],
                "",
            ]
            if out_of_scope_counts
            else []
        ),
        "## Top Files",
        "",
        "| Path | Category | Symbols | Imports | Summary |",
        "| --- | --- | ---: | ---: | --- |",
        *[
            f"| {row['path']} | {row['category']} | {row['symbol_count']} | {row['import_count']} | {_markdown_cell(row['summary'])} |"
            for row in top_files
        ],
        "",
        "## Top Folders",
        "",
        "| Path | Category | Files | Folders | Summary |",
        "| --- | --- | ---: | ---: | --- |",
        *[
            f"| {row['path']} | {row['category']} | {row['file_count']} | {row['folder_count']} | {_markdown_cell(row['summary'])} |"
            for row in top_folders
        ],
        "",
        "## Most Referenced Symbols",
        "",
        "| Symbol | Path | Kind | References |",
        "| --- | --- | --- | ---: |",
        *[
            f"| {_markdown_cell(row['qualified_name'])} | {row['path']} | {row['kind']} | {row['reference_count']} |"
            for row in top_symbols
        ],
        "",
        "## SQL Schema",
        "",
        *[
            _render_table_schema(table_name, table_schemas[table_name])
            for table_name in TABLE_NAMES
        ],
        "",
        "## Sample Stored Rows",
        "",
        "These are real rows from the SQLite DB, trimmed for readability.",
        "",
        *[
            _render_table_samples(table_name, table_samples[table_name])
            for table_name in TABLE_NAMES
        ],
        "",
        "## Schema View",
        "",
        "```mermaid",
        "erDiagram",
        "  REPOS ||--o{ NODES : contains",
        "  NODES ||--o{ NODE_CATEGORIES : classifies",
        "  NODES ||--o{ NODE_TAGS : tags",
        "  NODES ||--o{ SYMBOLS : defines",
        "  NODES ||--o{ IMPORTS : imports_from",
        "  NODES ||--o{ NODE_CONTEXT : inherits_from",
        "  SYMBOLS ||--o{ CALL_SITES : calls",
        "  SYMBOLS ||--o{ SYMBOL_REFERENCES : referenced_by",
        "  REPOS ||--o{ REFERENCES : stores_public_refs",
        "  REPOS ||--o{ EDGES : materializes_graph",
        "```",
        "",
        "## Stored Graph Sample",
        "",
        _render_graph_mermaid(repo_row["name"], top_files, top_import_edges, top_call_edges),
    ]
    return "\n".join(report_lines)


def _render_graph_mermaid(repo_name: str, top_files: list[sqlite3.Row], top_import_edges: list[sqlite3.Row], top_call_edges: list[sqlite3.Row]) -> str:
    selected_paths = []
    for row in top_files:
        selected_paths.append(row["path"])
    for row in top_import_edges:
        selected_paths.extend([row["source_path"], row["target_path"]])
    for row in top_call_edges:
        selected_paths.extend([row["source_path"], row["target_path"]])

    ordered_paths: list[str] = []
    seen: set[str] = set()
    for path in selected_paths:
        if not path or path in seen:
            continue
        seen.add(path)
        ordered_paths.append(path)

    node_ids = {path: f"n{index}" for index, path in enumerate(ordered_paths, start=1)}
    lines = ["```mermaid", "graph TD", f'  repo["repo: {_mermaid_label(repo_name)}"]']
    for path in ordered_paths:
        lines.append(f'  {node_ids[path]}["{_mermaid_label(path)}"]')
        lines.append(f"  repo --> {node_ids[path]}")

    for row in top_import_edges:
        source_id = node_ids.get(row["source_path"])
        target_id = node_ids.get(row["target_path"])
        if source_id is None or target_id is None:
            continue
        lines.append(f'  {source_id} -. "import {row["strength_label"]} x{row["weight"]}" .-> {target_id}')

    for row in top_call_edges:
        source_id = node_ids.get(row["source_path"])
        target_id = node_ids.get(row["target_path"])
        if source_id is None or target_id is None:
            continue
        lines.append(f'  {source_id} ==>|"call x{row["weight"]}"| {target_id}')

    lines.append("```")
    return "\n".join(lines)


def _markdown_cell(value: str) -> str:
    return " ".join((value or "").split()).replace("|", "\\|")


def _mermaid_label(value: str) -> str:
    return value.replace('"', "'")


def _load_table_schema(conn: sqlite3.Connection, table_name: str) -> list[sqlite3.Row]:
    return conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()


def _load_table_samples(conn: sqlite3.Connection, table_name: str, *, sample_limit: int) -> list[sqlite3.Row]:
    return conn.execute(f'SELECT * FROM "{table_name}" LIMIT ?', (sample_limit,)).fetchall()


def _render_table_schema(table_name: str, schema_rows: list[sqlite3.Row]) -> str:
    lines = [f"### `{table_name}`", "", "| Column | Type | Not Null | Default | PK |", "| --- | --- | --- | --- | ---: |"]
    for row in schema_rows:
        default_value = "" if row["dflt_value"] is None else str(row["dflt_value"])
        lines.append(
            f"| {row['name']} | {row['type'] or ''} | {'yes' if row['notnull'] else 'no'} | {_markdown_cell(default_value)} | {row['pk']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _render_table_samples(table_name: str, sample_rows: list[sqlite3.Row]) -> str:
    lines = [f"### `{table_name}`", ""]
    if not sample_rows:
        lines.append("_No rows_")
        lines.append("")
        return "\n".join(lines)

    for index, row in enumerate(sample_rows, start=1):
        lines.append(f"Row {index}:")
        lines.append("```json")
        lines.append(json.dumps(_trim_row(dict(row)), indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _trim_row(row: dict[str, object]) -> dict[str, object]:
    return {key: _trim_value(value) for key, value in row.items()}


def _trim_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    compact = " ".join(value.split())
    if len(compact) <= 220:
        return compact
    return compact[:217] + "..."