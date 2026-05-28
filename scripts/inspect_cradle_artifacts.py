from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a Cradle .cradle artifact directory")
    parser.add_argument("cradle_dir", help="Path to the .cradle directory")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--projection-target",
        choices=["nodes", "symbols", "centroids"],
        default=None,
        help="Also export a 2D PCA projection for the selected semantic vector set.",
    )
    parser.add_argument(
        "--projection-out",
        default=None,
        help="Path to write the projection JSON. Required when --projection-target is used.",
    )
    args = parser.parse_args()

    summary = inspect_cradle_dir(Path(args.cradle_dir))
    if args.projection_target:
        if not args.projection_out:
            raise SystemExit("--projection-out is required when --projection-target is used")
        projection = export_projection(Path(args.cradle_dir), args.projection_target)
        output_path = Path(args.projection_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(projection, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_summary(summary))
    return 0


def inspect_cradle_dir(cradle_dir: Path) -> dict[str, Any]:
    if not cradle_dir.exists() or not cradle_dir.is_dir():
        raise SystemExit(f"Cradle directory not found: {cradle_dir}")

    db_paths = sorted(cradle_dir.glob("*.db"))
    semantic_dirs = sorted(path for path in cradle_dir.iterdir() if path.is_dir() and (path / "manifest.json").exists())
    if not db_paths:
        raise SystemExit(f"No SQLite DB found in {cradle_dir}")

    db_path = db_paths[0]
    semantic_dir = semantic_dirs[0] if semantic_dirs else None

    db_summary = inspect_db(db_path)
    semantic_summary = inspect_semantic_dir(semantic_dir) if semantic_dir is not None else None

    capabilities = {
        "has_repo_tree": db_summary["table_counts"].get("nodes", 0) > 0,
        "has_symbol_detail": db_summary["table_counts"].get("symbols", 0) > 0,
        "has_import_graph": db_summary["table_counts"].get("imports", 0) > 0,
        "has_out_of_scope_imports": bool(
            db_summary.get("import_scope") and db_summary["import_scope"].get("out_of_scope", 0) > 0
        ),
        "has_call_graph": db_summary["table_counts"].get("call_sites", 0) > 0,
        "has_reference_graph": db_summary["table_counts"].get("symbol_references", 0) > 0,
        "has_node_context": db_summary["table_counts"].get("node_context", 0) > 0,
        "has_communities": db_summary["table_counts"].get("community_members", 0) > 0,
        "has_themes": db_summary["table_counts"].get("theme_members", 0) > 0,
        "has_semantic_nodes": bool(semantic_summary and semantic_summary["manifest"].get("node_count", 0) > 0),
        "has_semantic_symbols": bool(semantic_summary and semantic_summary["manifest"].get("symbol_count", 0) > 0),
        "has_centroids": bool(semantic_summary and semantic_summary["manifest"].get("centroid_count", 0) > 0),
    }

    return {
        "cradle_dir": str(cradle_dir),
        "db_path": str(db_path),
        "semantic_dir": None if semantic_dir is None else str(semantic_dir),
        "db": db_summary,
        "semantic": semantic_summary,
        "capabilities": capabilities,
        "ui_surfaces": suggest_ui_surfaces(capabilities, db_summary, semantic_summary),
    }


def inspect_db(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        table_counts = {
            table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in tables
            if not table.startswith("sqlite_")
        }

        node_kind_counts = _count_rows(conn, "SELECT kind, COUNT(*) FROM nodes GROUP BY kind ORDER BY COUNT(*) DESC, kind")
        edge_type_counts = _count_rows(conn, "SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type ORDER BY COUNT(*) DESC, edge_type")
        import_strength_counts = _count_rows(conn, "SELECT strength_label, COUNT(*) FROM imports GROUP BY strength_label ORDER BY COUNT(*) DESC, strength_label")
        # Check whether the is_out_of_scope column exists (older DBs may not have it).
        import_columns = {row[1] for row in conn.execute("PRAGMA table_info(imports)").fetchall()}
        if "is_out_of_scope" in import_columns:
            import_scope_summary_row = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN is_internal = 1 AND target_node_id IS NOT NULL THEN 1 ELSE 0 END) AS resolved_internal,
                    SUM(CASE WHEN is_out_of_scope = 1 THEN 1 ELSE 0 END)                           AS out_of_scope,
                    SUM(CASE WHEN is_internal = 0 THEN 1 ELSE 0 END)                               AS external
                FROM imports
                """
            ).fetchone()
            out_of_scope_modules = [
                {"module": row[0], "importer_count": row[1]}
                for row in conn.execute(
                    """
                    SELECT imported_module, COUNT(*) AS cnt
                    FROM imports
                    WHERE is_out_of_scope = 1
                    GROUP BY imported_module
                    ORDER BY cnt DESC, imported_module ASC
                    LIMIT 20
                    """
                ).fetchall()
            ]
            import_scope = {
                "resolved_internal": import_scope_summary_row["resolved_internal"] or 0,
                "out_of_scope": import_scope_summary_row["out_of_scope"] or 0,
                "external": import_scope_summary_row["external"] or 0,
                "out_of_scope_modules": out_of_scope_modules,
            }
        else:
            import_scope = None
        file_paths = [row[0] for row in conn.execute("SELECT path FROM nodes WHERE kind = 'file' ORDER BY path").fetchall()]
        extension_counter = Counter(_path_extension(path) for path in file_paths)
        file_extension_counts = [{"name": name, "count": count} for name, count in extension_counter.most_common()]
        top_categories = _count_rows(
            conn,
            "SELECT COALESCE(primary_category, '[none]'), COUNT(*) FROM nodes GROUP BY primary_category ORDER BY COUNT(*) DESC, primary_category",
        )
        top_tag_rows = conn.execute(
            "SELECT tag, COUNT(*) AS count FROM node_tags GROUP BY tag ORDER BY count DESC, tag LIMIT 20"
        ).fetchall()
        top_tags = [{"name": row["tag"], "count": row["count"]} for row in top_tag_rows]
        top_folders = conn.execute(
            "SELECT path, file_count, folder_count, summary FROM nodes WHERE kind = 'folder' ORDER BY file_count DESC, path LIMIT 12"
        ).fetchall()
        top_files = conn.execute(
            "SELECT path, symbol_count, import_count, summary FROM nodes WHERE kind = 'file' ORDER BY symbol_count DESC, import_count DESC, path LIMIT 12"
        ).fetchall()
    finally:
        conn.close()

    return {
        "size_bytes": db_path.stat().st_size,
        "table_counts": table_counts,
        "node_kind_counts": node_kind_counts,
        "edge_type_counts": edge_type_counts,
        "import_strength_counts": import_strength_counts,
        "import_scope": import_scope,
        "file_extension_counts": file_extension_counts,
        "top_categories": top_categories,
        "top_tags": top_tags,
        "top_folders": [_row_preview(row, ["path", "file_count", "folder_count", "summary"]) for row in top_folders],
        "top_files": [_row_preview(row, ["path", "symbol_count", "import_count", "summary"]) for row in top_files],
    }


def inspect_semantic_dir(semantic_dir: Path) -> dict[str, Any]:
    manifest = json.loads((semantic_dir / "manifest.json").read_text(encoding="utf-8"))
    node_records = json.loads((semantic_dir / "nodes.records.json").read_text(encoding="utf-8"))
    symbol_records = json.loads((semantic_dir / "symbols.records.json").read_text(encoding="utf-8"))
    centroid_records = json.loads((semantic_dir / "node_centroids.records.json").read_text(encoding="utf-8"))

    node_kind_counts = Counter(record.get("kind") or "[none]" for record in node_records)
    centroid_parent_counts = Counter(record.get("parent_id") or "[none]" for record in centroid_records)
    centroid_member_sizes = [len(record.get("member_ids") or []) for record in centroid_records]
    representative_parents = []
    for parent_id, count in centroid_parent_counts.most_common(12):
        matching = [record for record in centroid_records if record.get("parent_id") == parent_id]
        representative_parents.append(
            {
                "parent_id": parent_id,
                "centroid_count": count,
                "member_count_total": sum(len(record.get("member_ids") or []) for record in matching),
                "example_representatives": [
                    item
                    for record in matching[:2]
                    for item in (record.get("representative_ids") or [])[:3]
                ][:6],
            }
        )

    sample_nodes = [_semantic_record_preview(record) for record in node_records[:8]]
    sample_symbols = [_semantic_record_preview(record) for record in symbol_records[:8]]
    sample_centroids = [_centroid_preview(record) for record in centroid_records[:8]]

    artifact_sizes = {
        path.name: path.stat().st_size
        for path in sorted(semantic_dir.iterdir())
        if path.is_file()
    }

    return {
        "manifest": manifest,
        "artifact_sizes": artifact_sizes,
        "node_record_kind_counts": [{"name": name, "count": count} for name, count in node_kind_counts.most_common()],
        "centroid_parent_counts": representative_parents,
        "centroid_member_size_stats": _size_stats(centroid_member_sizes),
        "sample_nodes": sample_nodes,
        "sample_symbols": sample_symbols,
        "sample_centroids": sample_centroids,
    }


def export_projection(cradle_dir: Path, target: str) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("numpy is required to export embedding projections") from exc

    semantic_dirs = sorted(path for path in cradle_dir.iterdir() if path.is_dir() and (path / "manifest.json").exists())
    if not semantic_dirs:
        raise SystemExit(f"No semantic sidecar found in {cradle_dir}")
    semantic_dir = semantic_dirs[0]

    if target == "nodes":
        records = json.loads((semantic_dir / "nodes.records.json").read_text(encoding="utf-8"))
        vectors = np.load(semantic_dir / "nodes.vectors.npy").astype(np.float32)
        points = [
            {
                "id": record.get("entity_id"),
                "label": record.get("title"),
                "path": record.get("path"),
                "kind": record.get("kind"),
            }
            for record in records
        ]
    elif target == "symbols":
        records = json.loads((semantic_dir / "symbols.records.json").read_text(encoding="utf-8"))
        vectors = np.load(semantic_dir / "symbols.vectors.npy").astype(np.float32)
        points = [
            {
                "id": record.get("entity_id"),
                "label": record.get("title"),
                "path": record.get("path"),
                "kind": record.get("kind"),
            }
            for record in records
        ]
    else:
        records = json.loads((semantic_dir / "node_centroids.records.json").read_text(encoding="utf-8"))
        vectors = np.load(semantic_dir / "node_centroids.vectors.npy").astype(np.float32)
        points = [
            {
                "id": record.get("centroid_id"),
                "label": record.get("title"),
                "path": record.get("parent_id"),
                "kind": "centroid",
                "parent_id": record.get("parent_id"),
                "member_count": len(record.get("member_ids") or []),
            }
            for record in records
        ]

    if len(points) != len(vectors):
        raise SystemExit(
            f"Projection source mismatch for {target}: {len(points)} records vs {len(vectors)} vectors"
        )

    projected = _project_pca_2d(vectors)
    for point, coords in zip(points, projected, strict=False):
        point["x"] = round(float(coords[0]), 6)
        point["y"] = round(float(coords[1]), 6)

    return {
        "target": target,
        "method": "pca",
        "count": len(points),
        "source_dir": str(semantic_dir),
        "points": points,
    }


def suggest_ui_surfaces(capabilities: dict[str, bool], db_summary: dict[str, Any], semantic_summary: dict[str, Any] | None) -> list[dict[str, str]]:
    surfaces: list[dict[str, str]] = []
    if capabilities["has_repo_tree"]:
        surfaces.append(
            {
                "name": "Repository Tree",
                "why": "Nodes already encode repo, folder, and file hierarchy.",
                "view": "Collapsible tree and zoomable canvas grouped by repo -> folder -> file.",
            }
        )
    if capabilities["has_import_graph"]:
        surfaces.append(
            {
                "name": "Import Graph",
                "why": "The DB stores importer -> target links with strength labels.",
                "view": "Directed edges between files or folders with weak/medium/strong styling.",
            }
        )
        import_scope = db_summary.get("import_scope")
        if import_scope and import_scope.get("out_of_scope", 0) > 0:
            surfaces.append(
                {
                    "name": "Out-of-Scope Import Boundary",
                    "why": (
                        f"{import_scope['out_of_scope']} import(s) were detected as internal "
                        "but point outside the analyzed root.  These are real dependencies that "
                        "Cradle cannot follow because the target files were excluded or live in "
                        "a parent package that was not analysed."
                    ),
                    "view": (
                        "Dashed boundary edges from the importer file to a ghost node labelled "
                        "'[out of Cradle scope: <module>]'.  Toggled separately from resolved "
                        "import edges so the main graph stays clean."
                    ),
                }
            )
    if capabilities["has_themes"]:
        surfaces.append(
            {
                "name": "Theme Domains",
                "why": "Theme membership rows group files into semantic domains such as auth, payments, or ML.",
                "view": "Domain chips or expandable branch nodes that reveal semantically related files even across folders.",
            }
        )
    if capabilities["has_call_graph"]:
        surfaces.append(
            {
                "name": "Symbol Call Graph",
                "why": "Call sites resolve caller and often target symbol IDs.",
                "view": "Per-file or per-symbol neighborhoods with expandable caller/callee chains.",
            }
        )
    if capabilities["has_symbol_detail"]:
        surfaces.append(
            {
                "name": "File and Symbol Inspector",
                "why": "Files and symbols already carry summary, category, signature, docstring, and path metadata.",
                "view": "Click a node to open a side panel with summary, tags, symbols, imports, and source excerpt.",
            }
        )
    if capabilities["has_centroids"]:
        surfaces.append(
            {
                "name": "Centroid Cluster Layer",
                "why": "Semantic sidecars already group children into centroid clusters under structural parents.",
                "view": "React Flow cluster nodes that sit above files and can be expanded into representative children.",
            }
        )
    if capabilities["has_communities"]:
        surfaces.append(
            {
                "name": "Community Overlay",
                "why": "Community memberships would provide cross-tree virtual clusters.",
                "view": "Toggle overlay that adds non-tree semantic/import communities across the repo.",
            }
        )
    if not capabilities["has_communities"]:
        surfaces.append(
            {
                "name": "Community Overlay",
                "why": "This dataset does not currently store community memberships, so the UI should hide or disable this layer.",
                "view": "Future layer once `community_members` is populated for the analyzed repo.",
            }
        )
    if not capabilities["has_node_context"]:
        surfaces.append(
            {
                "name": "Context Inheritance",
                "why": "This dataset has no `node_context` rows, so imported-summary inheritance panels would be empty.",
                "view": "Keep the panel optional and render only when context rows exist.",
            }
        )
    if semantic_summary is not None:
        surfaces.append(
            {
                "name": "Semantic Search Debug",
                "why": "Node, symbol, and centroid sidecars can show why a query matched a branch.",
                "view": "Display query hit list with semantic score, centroid parent, and representative members.",
            }
        )
    return surfaces


def _count_rows(conn: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
    rows = conn.execute(query).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        results.append({"name": row[0], "count": row[1]})
    return results


def _row_preview(row: sqlite3.Row, columns: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for column in columns:
        value = row[column]
        if isinstance(value, str):
            payload[column] = _trim_text(value, 220)
        else:
            payload[column] = value
    return payload


def _semantic_record_preview(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": record.get("entity_id"),
        "title": record.get("title"),
        "path": record.get("path"),
        "kind": record.get("kind"),
        "content_preview": _trim_text(str(record.get("content") or ""), 280),
    }


def _centroid_preview(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "centroid_id": record.get("centroid_id"),
        "parent_id": record.get("parent_id"),
        "member_count": len(record.get("member_ids") or []),
        "representative_ids": list(record.get("representative_ids") or []),
    }


def _size_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "avg": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": round(sum(values) / len(values), 2),
    }


def _trim_text(value: str, max_chars: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _path_extension(path: str) -> str:
    suffix = Path(path).suffix
    return suffix if suffix else "[none]"


def _project_pca_2d(vectors):
    import numpy as np

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise SystemExit(f"Expected a 2D vector matrix, got shape {matrix.shape}")
    if len(matrix) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(matrix) == 1:
        return np.zeros((1, 2), dtype=np.float32)

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    if components.shape[1] == 1:
        components = np.concatenate([components, np.zeros((components.shape[0], 1), dtype=np.float32)], axis=1)
    return centered @ components[:, :2]


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Cradle dir: {summary['cradle_dir']}",
        f"DB: {summary['db_path']}",
        f"Semantic dir: {summary['semantic_dir'] or '[none]'}",
        "",
        "Capabilities:",
    ]
    for name, enabled in summary["capabilities"].items():
        lines.append(f"  - {name}: {'yes' if enabled else 'no'}")

    db = summary["db"]
    lines.extend(
        [
            "",
            f"DB size: {db['size_bytes']} bytes",
            "Table counts:",
        ]
    )
    for name, count in sorted(db["table_counts"].items()):
        lines.append(f"  - {name}: {count}")

    lines.append("")
    lines.append("Node kinds:")
    for item in db["node_kind_counts"]:
        lines.append(f"  - {item['name']}: {item['count']}")

    lines.append("")
    lines.append("Edge types:")
    for item in db["edge_type_counts"]:
        lines.append(f"  - {item['name']}: {item['count']}")

    import_scope = db.get("import_scope")
    if import_scope is not None:
        lines.extend(
            [
                "",
                "Import scope breakdown:",
                f"  - resolved internal (in graph):           {import_scope['resolved_internal']}",
                f"  - out of scope (internal, target missing): {import_scope['out_of_scope']}",
                f"  - external (stdlib / third-party):        {import_scope['external']}",
            ]
        )
        if import_scope["out_of_scope_modules"]:
            lines.append("")
            lines.append("  Out-of-scope modules (imported as internal, not in analyzed root):")
            for item in import_scope["out_of_scope_modules"]:
                lines.append(f"    - {item['module']}  ({item['importer_count']} importer(s))")

    semantic = summary.get("semantic")
    if semantic is not None:
        manifest = semantic["manifest"]
        lines.extend(
            [
                "",
                "Semantic manifest:",
                f"  - model: {manifest['model_name']}",
                f"  - dimension: {manifest['dimension']}",
                f"  - engine: {manifest['engine']}",
                f"  - node_count: {manifest['node_count']}",
                f"  - symbol_count: {manifest['symbol_count']}",
                f"  - centroid_count: {manifest['centroid_count']}",
                "",
                "Top centroid parents:",
            ]
        )
        for item in semantic["centroid_parent_counts"][:10]:
            lines.append(
                f"  - {item['parent_id']}: {item['centroid_count']} centroids, {item['member_count_total']} members"
            )

    lines.extend(["", "Suggested UI surfaces:"])
    for item in summary["ui_surfaces"]:
        lines.append(f"  - {item['name']}: {item['view']} ({item['why']})")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())