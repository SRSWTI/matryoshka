"""Local web dashboard for Cradle analysis artifacts.

Serves a single-page application at http://127.0.0.1:<port>/ backed by
Python's built-in HTTP server (no extra dependencies beyond numpy which is
already required).  All data is loaded from the SQLite DB and the optional
semantic sidecar directory.

Endpoints
---------
GET /                  → the dashboard SPA (single HTML page)
GET /api/graph         → all nodes + import edges as JSON
GET /api/themes        → theme nodes + member lists
GET /api/communities   → community nodes + member lists
GET /api/embeddings    → PCA-projected 2D coords from semantic sidecar
GET /api/node?id=...   → full detail for a single node
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project an N×D matrix to N×2 via PCA (no external dependency)."""
    if matrix.shape[0] <= 2:
        return matrix[:, :2]
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


# ---------------------------------------------------------------------------
# Dashboard data layer
# ---------------------------------------------------------------------------

class CradleDashboard:
    def __init__(
        self,
        db_path: str | Path,
        index_dir: str | Path | None = None,
        port: int = 8765,
    ) -> None:
        self.db_path = Path(db_path)
        if index_dir:
            self.index_dir = Path(index_dir)
        else:
            # Conventional sidecar location: <db_stem>-semantic next to the DB
            self.index_dir = self.db_path.parent / (self.db_path.stem + "-semantic")
        self.port = port
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # DB connection
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # /api/graph
    # ------------------------------------------------------------------

    def get_graph_data(self) -> dict[str, Any]:
        if "graph" in self._cache:
            return self._cache["graph"]

        with self._connect() as conn:
            # Nodes with all membership metadata
            nodes_rows = conn.execute(
                """
                SELECT node_id, kind, path, name, summary, primary_category,
                       symbol_count, import_count, file_count, folder_count, confidence
                FROM nodes
                ORDER BY kind, path
                """
            ).fetchall()

            tags_map: dict[str, list[str]] = {}
            for row in conn.execute(
                "SELECT node_id, tag FROM node_tags ORDER BY rank"
            ):
                tags_map.setdefault(row["node_id"], []).append(row["tag"])

            cats_map: dict[str, list[str]] = {}
            for row in conn.execute(
                "SELECT node_id, category FROM node_categories ORDER BY rank"
            ):
                cats_map.setdefault(row["node_id"], []).append(row["category"])

            file_themes: dict[str, list[str]] = {}
            for row in conn.execute(
                "SELECT theme_node_id, member_node_id FROM theme_members"
            ):
                file_themes.setdefault(row["member_node_id"], []).append(
                    row["theme_node_id"]
                )

            file_communities: dict[str, list[str]] = {}
            for row in conn.execute(
                "SELECT community_node_id, member_node_id FROM community_members"
            ):
                file_communities.setdefault(row["member_node_id"], []).append(
                    row["community_node_id"]
                )

            nodes: list[dict[str, Any]] = []
            for row in nodes_rows:
                nid = row["node_id"]
                nodes.append(
                    {
                        "id": nid,
                        "kind": row["kind"],
                        "path": row["path"],
                        "name": row["name"],
                        "summary": row["summary"] or "",
                        "category": row["primary_category"],
                        "symbol_count": row["symbol_count"],
                        "import_count": row["import_count"],
                        "file_count": row["file_count"],
                        "folder_count": row["folder_count"],
                        "confidence": row["confidence"],
                        "tags": tags_map.get(nid, []),
                        "categories": cats_map.get(nid, []),
                        "themes": file_themes.get(nid, []),
                        "communities": file_communities.get(nid, []),
                    }
                )

            # Build module-name → node_id map so we can resolve import edge targets.
            # The `edges` table stores module names (e.g. "omlx.scheduler") as to_id
            # for import edges, NOT file paths.
            module_map = self._build_module_node_map(conn)

            # Read all import + OOS edges from the edges table
            edge_rows = conn.execute(
                """
                SELECT from_id, to_id, edge_type, strength
                FROM edges
                WHERE edge_type IN ('import', 'out_of_scope_import')
                """
            ).fetchall()

            # File-to-file call edges (resolved via detail JSON)
            call_rows = conn.execute(
                """
                SELECT
                    json_extract(detail,'$.caller_node_id') AS src,
                    json_extract(detail,'$.target_node_id') AS tgt,
                    COUNT(*) AS weight
                FROM edges
                WHERE edge_type = 'call'
                  AND json_extract(detail,'$.target_node_id') IS NOT NULL
                  AND json_extract(detail,'$.caller_node_id') IS NOT NULL
                  AND json_extract(detail,'$.caller_node_id') !=
                      json_extract(detail,'$.target_node_id')
                GROUP BY src, tgt
                """
            ).fetchall()

        edges: list[dict[str, Any]] = []
        oos_modules: set[str] = set()

        for row in edge_rows:
            if row["edge_type"] == "out_of_scope_import":
                mod = row["to_id"]
                oos_modules.add(mod)
                edges.append(
                    {
                        "source": row["from_id"],
                        "target": f"__oos__{mod}",
                        "type": "out_of_scope_import",
                        "strength": row["strength"] or "weak",
                        "is_oos": True,
                    }
                )
            else:
                # Resolve module name → file node_id
                target_nid = module_map.get(row["to_id"])
                if target_nid is None:
                    continue  # external/unresolved — skip from graph
                edges.append(
                    {
                        "source": row["from_id"],
                        "target": target_nid,
                        "type": "import",
                        "strength": row["strength"] or "weak",
                        "is_oos": False,
                    }
                )

        # Synthetic ghost nodes for OOS modules
        for mod in oos_modules:
            nodes.append(
                {
                    "id": f"__oos__{mod}",
                    "kind": "out_of_scope",
                    "path": mod,
                    "name": mod,
                    "summary": f"Out-of-scope import: '{mod}' was not found in the analyzed root.",
                    "category": None,
                    "symbol_count": 0,
                    "import_count": 0,
                    "file_count": 0,
                    "folder_count": 0,
                    "confidence": 0.0,
                    "tags": [],
                    "categories": [],
                    "themes": [],
                    "communities": [],
                }
            )

        call_file_edges: list[dict[str, Any]] = [
            {"source": row["src"], "target": row["tgt"], "weight": row["weight"]}
            for row in call_rows
            if row["src"] and row["tgt"]
        ]

        result: dict[str, Any] = {"nodes": nodes, "edges": edges, "call_edges": call_file_edges}
        self._cache["graph"] = result
        return result

    @staticmethod
    def _build_module_node_map(conn: sqlite3.Connection) -> dict[str, str]:
        """Map Python module names (both with and without package prefix) to node_ids.

        For a file at ``cache/prefix_cache.py`` in a package named ``omlx`` this
        produces both ``cache.prefix_cache`` and ``omlx.cache.prefix_cache``
        pointing to the same node_id.  ``__init__.py`` files are mapped to their
        parent package name (e.g. ``adapter/__init__.py`` → ``adapter`` and
        ``omlx.adapter``).
        """
        repo_row = conn.execute("SELECT name FROM repos LIMIT 1").fetchone()
        pkg_name: str = repo_row["name"] if repo_row else ""

        module_map: dict[str, str] = {}
        for row in conn.execute("SELECT node_id FROM nodes WHERE kind = 'file'"):
            path: str = row["node_id"]
            if not path.endswith(".py"):
                continue
            stem = path[:-3]  # strip .py
            parts = stem.replace("/", ".").split(".")
            if parts[-1] == "__init__":
                parts = parts[:-1]
            rel_module = ".".join(parts)
            if rel_module:
                module_map[rel_module] = path
                if pkg_name:
                    module_map[f"{pkg_name}.{rel_module}"] = path
            elif pkg_name:
                # root __init__.py → maps to package name itself
                module_map[pkg_name] = path

        return module_map

    # ------------------------------------------------------------------
    # /api/themes
    # ------------------------------------------------------------------

    def get_themes(self) -> dict[str, Any]:
        if "themes" in self._cache:
            return self._cache["themes"]

        with self._connect() as conn:
            theme_rows = conn.execute(
                "SELECT node_id, name, summary FROM nodes WHERE kind = 'theme' ORDER BY name"
            ).fetchall()
            member_rows = conn.execute(
                "SELECT theme_node_id, member_node_id, membership_weight "
                "FROM theme_members ORDER BY theme_node_id, membership_rank"
            ).fetchall()

        members_map: dict[str, list[str]] = {}
        for row in member_rows:
            members_map.setdefault(row["theme_node_id"], []).append(row["member_node_id"])

        themes = [
            {
                "id": row["node_id"],
                "name": row["name"],
                "summary": row["summary"] or "",
                "members": members_map.get(row["node_id"], []),
            }
            for row in theme_rows
        ]
        result: dict[str, Any] = {"themes": themes}
        self._cache["themes"] = result
        return result

    # ------------------------------------------------------------------
    # /api/communities
    # ------------------------------------------------------------------

    def get_communities(self) -> dict[str, Any]:
        if "communities" in self._cache:
            return self._cache["communities"]

        with self._connect() as conn:
            comm_rows = conn.execute(
                "SELECT node_id, name, summary, primary_category "
                "FROM nodes WHERE kind = 'community' ORDER BY name"
            ).fetchall()
            member_rows = conn.execute(
                "SELECT community_node_id, member_node_id "
                "FROM community_members ORDER BY community_node_id, membership_rank"
            ).fetchall()

        members_map: dict[str, list[str]] = {}
        for row in member_rows:
            members_map.setdefault(row["community_node_id"], []).append(row["member_node_id"])

        communities = [
            {
                "id": row["node_id"],
                "name": row["name"],
                "summary": row["summary"] or "",
                "category": row["primary_category"],
                "members": members_map.get(row["node_id"], []),
            }
            for row in comm_rows
        ]
        result: dict[str, Any] = {"communities": communities}
        self._cache["communities"] = result
        return result

    # ------------------------------------------------------------------
    # /api/embeddings
    # ------------------------------------------------------------------

    def get_embeddings(self) -> dict[str, Any]:
        if "embeddings" in self._cache:
            return self._cache["embeddings"]

        vectors_path = self.index_dir / "nodes.vectors.npy"
        records_path = self.index_dir / "nodes.records.json"

        if not vectors_path.exists() or not records_path.exists():
            result: dict[str, Any] = {
                "error": (
                    "No semantic index found. "
                    f"Run: cradle semantic-index {self.db_path}"
                ),
                "points": [],
            }
            self._cache["embeddings"] = result
            return result

        matrix = np.load(str(vectors_path))
        records: list[dict[str, Any]] = json.loads(
            records_path.read_text(encoding="utf-8")
        )

        coords = _pca_2d(matrix)

        # Enrich with DB metadata (category, full summary)
        with self._connect() as conn:
            node_meta: dict[str, dict[str, Any]] = {
                row["node_id"]: {
                    "kind": row["kind"],
                    "category": row["primary_category"],
                    "name": row["name"],
                    "summary": row["summary"] or "",
                }
                for row in conn.execute(
                    "SELECT node_id, kind, primary_category, name, summary FROM nodes"
                )
            }

        points: list[dict[str, Any]] = []
        for i, rec in enumerate(records):
            eid = rec["entity_id"]
            meta = node_meta.get(
                eid,
                {
                    "kind": rec.get("kind", "unknown"),
                    "category": None,
                    "name": rec.get("title", eid),
                    "summary": "",
                },
            )
            points.append(
                {
                    "id": eid,
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                    "kind": meta["kind"],
                    "category": meta["category"],
                    "name": meta["name"],
                    "summary": meta["summary"],
                }
            )

        result = {"points": points}
        self._cache["embeddings"] = result
        return result

    # ------------------------------------------------------------------
    # /api/node?id=...
    # ------------------------------------------------------------------

    def get_node_detail(self, node_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            if row is None:
                return {"error": f"Node '{node_id}' not found"}

            symbols = conn.execute(
                """
                SELECT name, qualified_name, kind, signature, summary, return_type
                FROM symbols
                WHERE node_id = ?
                ORDER BY kind, name
                LIMIT 50
                """,
                (node_id,),
            ).fetchall()

            imports_out = conn.execute(
                """
                SELECT imported_module, target_node_id, strength_label,
                       COALESCE(is_out_of_scope, 0) AS is_oos
                FROM imports
                WHERE importer_node_id = ?
                ORDER BY strength_label DESC, imported_module
                """,
                (node_id,),
            ).fetchall()

            imports_in = conn.execute(
                """
                SELECT importer_node_id, strength_label
                FROM imports
                WHERE target_node_id = ?
                ORDER BY importer_node_id
                LIMIT 30
                """,
                (node_id,),
            ).fetchall()

            tags = conn.execute(
                "SELECT tag FROM node_tags WHERE node_id = ? ORDER BY rank",
                (node_id,),
            ).fetchall()

        return {
            "node_id": row["node_id"],
            "kind": row["kind"],
            "path": row["path"],
            "name": row["name"],
            "summary": row["summary"] or "",
            "description": row["description"] or "",
            "primary_category": row["primary_category"],
            "symbol_count": row["symbol_count"],
            "import_count": row["import_count"],
            "file_count": row["file_count"],
            "folder_count": row["folder_count"],
            "confidence": row["confidence"],
            "tags": [r["tag"] for r in tags],
            "symbols": [
                {
                    "name": r["name"],
                    "qualified_name": r["qualified_name"],
                    "kind": r["kind"],
                    "signature": r["signature"] or "",
                    "summary": r["summary"] or "",
                    "return_type": r["return_type"],
                }
                for r in symbols
            ],
            "imports_out": [
                {
                    "module": r["imported_module"],
                    "target": r["target_node_id"],
                    "strength": r["strength_label"],
                    "oos": bool(r["is_oos"]),
                }
                for r in imports_out
            ],
            "imports_in": [
                {"source": r["importer_node_id"], "strength": r["strength_label"]}
                for r in imports_in
            ],
        }

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------

    def serve(self, open_browser: bool = True) -> None:
        dashboard = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:  # noqa: A002
                pass  # suppress default access log

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                qs = parse_qs(parsed.query)

                try:
                    if path in ("/", "/index.html"):
                        self._html(_DASHBOARD_HTML)
                    elif path == "/api/graph":
                        self._json(dashboard.get_graph_data())
                    elif path == "/api/themes":
                        self._json(dashboard.get_themes())
                    elif path == "/api/communities":
                        self._json(dashboard.get_communities())
                    elif path == "/api/embeddings":
                        self._json(dashboard.get_embeddings())
                    elif path == "/api/node":
                        node_id = qs.get("id", [""])[0]
                        if not node_id:
                            self._json({"error": "missing id parameter"})
                        else:
                            self._json(dashboard.get_node_detail(node_id))
                    else:
                        self.send_response(404)
                        self.end_headers()
                except Exception:
                    logger.exception("Error handling %s", self.path)
                    self.send_response(500)
                    self.end_headers()

            def _json(self, data: dict[str, Any]) -> None:
                body = json.dumps(data, default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            def _html(self, html: str) -> None:
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(("127.0.0.1", self.port), _Handler)
        url = f"http://127.0.0.1:{self.port}"
        print(f"\n  Cradle Dashboard  →  {url}\n  DB: {self.db_path}\n  Press Ctrl-C to stop.\n")

        if open_browser:
            threading.Timer(0.6, webbrowser.open, args=[url]).start()

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
        finally:
            server.server_close()


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_dashboard(
    db_path: str | Path,
    index_dir: str | Path | None = None,
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    CradleDashboard(db_path, index_dir=index_dir, port=port).serve(open_browser=open_browser)


# ---------------------------------------------------------------------------
# Dashboard HTML — single-page application
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cradle · Dashboard</title>
<script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:       #0f1117;
  --surface:  #1a1d27;
  --surface2: #252836;
  --border:   #2e3347;
  --text:     #e8eaf0;
  --muted:    #7b82a0;
  --accent:   #4f8ef7;
  --radius:   8px;
  --font:     system-ui, -apple-system, 'Segoe UI', sans-serif;
}
html, body { height: 100%; overflow: hidden; background: var(--bg); color: var(--text); font-family: var(--font); font-size: 13px; }
#app { display: flex; flex-direction: column; height: 100vh; }

/* header */
#hdr { display: flex; align-items: center; gap: 20px; padding: 0 18px; height: 46px; background: var(--surface); border-bottom: 1px solid var(--border); flex-shrink: 0; overflow: hidden; }
#app-title { font-size: 15px; font-weight: 700; color: var(--accent); white-space: nowrap; letter-spacing: -0.3px; }
#db-name   { font-size: 11px; color: var(--muted); font-family: monospace; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 340px; }
#stats-bar { display: flex; gap: 18px; margin-left: auto; flex-shrink: 0; }
.stat      { display: flex; flex-direction: column; align-items: center; }
.stat-val  { font-size: 17px; font-weight: 700; color: var(--accent); line-height: 1.1; }
.stat-val.oos { color: #ef4444; }
.stat-lbl  { font-size: 9px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }

/* layout */
#layout { display: flex; flex: 1; overflow: hidden; }

/* sidebar */
#sidebar { width: 210px; background: var(--surface); border-right: 1px solid var(--border); overflow-y: auto; flex-shrink: 0; padding: 12px 10px; }
.sb-section { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); margin: 14px 0 6px; }
.sb-section:first-child { margin-top: 0; }
select, .sb-input { width: 100%; background: var(--bg); border: 1px solid var(--border); color: var(--text); padding: 5px 8px; border-radius: 5px; font-size: 12px; margin-bottom: 5px; outline: none; font-family: var(--font); }
select:focus { border-color: var(--accent); }
.tog { display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--muted); margin-bottom: 5px; cursor: pointer; }
.tog input[type=checkbox] { width: auto; margin: 0; accent-color: var(--accent); cursor: pointer; }
.tog:hover { color: var(--text); }
#legend { display: flex; flex-direction: column; gap: 4px; }
.legend-row { display: flex; align-items: center; gap: 7px; font-size: 11px; color: var(--muted); }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.btn-sm { width: 100%; background: var(--surface2); border: 1px solid var(--border); color: var(--muted); padding: 5px 0; border-radius: 5px; font-size: 11px; cursor: pointer; margin-bottom: 5px; font-family: var(--font); transition: all .15s; }
.btn-sm:hover { background: var(--accent); color: #fff; border-color: var(--accent); }
#sb-detail-filters { display: none; }

/* main content */
#content { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }
#tab-bar { display: flex; background: var(--surface); border-bottom: 1px solid var(--border); flex-shrink: 0; }
.tab { padding: 10px 16px; border: none; background: none; color: var(--muted); cursor: pointer; font-size: 12px; font-weight: 500; border-bottom: 2px solid transparent; font-family: var(--font); transition: all .15s; white-space: nowrap; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-panel { display: none; flex: 1; overflow: hidden; flex-direction: column; }
.tab-panel.active { display: flex; }

/* graph toolbar */
#graph-toolbar { display: flex; align-items: center; gap: 10px; padding: 6px 12px; background: var(--surface2); border-bottom: 1px solid var(--border); font-size: 12px; flex-shrink: 0; min-height: 38px; }
.view-btns { display: flex; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; flex-shrink: 0; }
.vbtn { padding: 4px 13px; border: none; background: transparent; color: var(--muted); cursor: pointer; font-size: 12px; font-family: var(--font); transition: all .15s; white-space: nowrap; }
.vbtn.active { background: var(--accent); color: #fff; }
.vbtn:hover:not(.active) { color: var(--text); }
#btn-back { display: none; background: var(--surface2); border: 1px solid var(--border); color: var(--muted); padding: 4px 10px; border-radius: 5px; font-size: 11px; cursor: pointer; white-space: nowrap; font-family: var(--font); }
#btn-back:hover { color: var(--text); border-color: var(--muted); }
#edge-mode-wrap { display: none; align-items: center; gap: 6px; font-size: 11px; color: var(--muted); }
#edge-mode-wrap select { width: auto; margin-bottom: 0; font-size: 11px; }
#graph-info { font-size: 11px; color: var(--muted); }
#cy { flex: 1; background: var(--bg); }

/* card grids */
.card-grid { flex: 1; overflow-y: auto; padding: 14px; display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 10px; align-content: start; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; transition: border-color .15s; }
.card:hover { border-color: var(--accent); }
.card-head { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; margin-bottom: 8px; }
.card-summary { font-size: 12px; color: var(--muted); line-height: 1.55; margin-bottom: 10px; }
.card details summary { font-size: 12px; color: var(--accent); cursor: pointer; margin-bottom: 6px; user-select: none; }
.member-list { list-style: none; max-height: 130px; overflow-y: auto; margin-bottom: 8px; }
.member-item { font-size: 11px; color: var(--muted); padding: 2px 5px; border-radius: 3px; font-family: monospace; cursor: pointer; }
.member-item:hover { background: var(--surface2); color: var(--text); }
.member-more { font-size: 11px; color: var(--muted); padding: 2px 5px; font-style: italic; }
.btn-focus { background: var(--surface2); border: 1px solid var(--border); color: var(--muted); padding: 4px 10px; border-radius: 4px; font-size: 11px; cursor: pointer; font-family: var(--font); transition: all .15s; }
.btn-focus:hover { background: var(--accent); color: #fff; border-color: var(--accent); }

/* embeddings */
#emb-toolbar { display: flex; align-items: center; gap: 10px; padding: 6px 12px; background: var(--surface2); border-bottom: 1px solid var(--border); font-size: 12px; flex-shrink: 0; }
#embedding-plot { flex: 1; min-height: 0; }

/* search */
#search-wrap { flex: 1; overflow-y: auto; padding: 16px; }
#search-input { width: 100%; max-width: 520px; padding: 9px 13px; font-size: 14px; border-radius: 6px; margin-bottom: 14px; display: block; background: var(--surface); border: 1px solid var(--border); color: var(--text); font-family: var(--font); outline: none; }
#search-input:focus { border-color: var(--accent); }
#search-results { display: flex; flex-direction: column; gap: 7px; }
.sr { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 10px 14px; cursor: pointer; transition: border-color .15s; }
.sr:hover { border-color: var(--accent); }
.sr-head { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; margin-bottom: 3px; }
.sr-path { font-size: 12px; font-family: monospace; }
.sr-summary { font-size: 12px; color: var(--muted); line-height: 1.5; }

/* detail panel */
#detail-panel { width: 330px; background: var(--surface); border-left: 1px solid var(--border); overflow-y: auto; flex-shrink: 0; padding: 14px 13px; }
#detail-panel.closed { display: none; }
#close-detail { float: right; background: none; border: none; color: var(--muted); cursor: pointer; font-size: 15px; padding: 0 4px; line-height: 1; }
#close-detail:hover { color: var(--text); }
.det-head { margin-bottom: 10px; }
.det-name { font-size: 15px; font-weight: 600; margin: 5px 0 3px; word-break: break-all; }
.det-path { font-size: 11px; color: var(--muted); font-family: monospace; word-break: break-all; }
.det-sec { margin-bottom: 11px; }
.det-sec-title { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); cursor: pointer; user-select: none; margin-bottom: 6px; display: block; }
.det-summary { font-size: 12px; color: var(--muted); line-height: 1.6; }
.det-stats { display: flex; gap: 12px; flex-wrap: wrap; font-size: 11px; color: var(--muted); margin-bottom: 11px; }
.tag-row { display: flex; flex-wrap: wrap; gap: 4px; }
.tag { background: var(--surface2); color: var(--muted); font-size: 10px; padding: 2px 7px; border-radius: 20px; }
.sym-list { list-style: none; display: flex; flex-direction: column; gap: 5px; }
.sym-item { background: var(--bg); border-radius: 4px; padding: 6px 8px; }
.sym-kind { font-size: 10px; color: var(--accent); font-family: monospace; margin-right: 4px; }
.sym-name { font-size: 12px; font-weight: 500; }
.sym-sig  { display: block; font-size: 10px; color: var(--muted); font-family: monospace; margin-top: 2px; word-break: break-all; }
.sym-sum  { font-size: 11px; color: var(--muted); margin-top: 3px; line-height: 1.45; }
.imp-list { list-style: none; display: flex; flex-direction: column; gap: 2px; }
.imp-item { display: flex; align-items: center; gap: 6px; padding: 3px 6px; border-radius: 3px; font-size: 11px; }
.imp-item.clickable:hover { background: var(--surface2); cursor: pointer; }
.imp-item.oos { border-left: 2px solid #ef4444; padding-left: 4px; }
.imp-mod  { font-family: monospace; color: var(--muted); flex: 1; word-break: break-all; }
.imp-str  { font-size: 10px; color: var(--muted); flex-shrink: 0; }
.oos-pill { font-size: 9px; background: #ef444418; color: #ef4444; border: 1px solid #ef444430; border-radius: 20px; padding: 1px 5px; flex-shrink: 0; }

/* badges */
.badge { font-size: 10px; padding: 2px 7px; border-radius: 20px; font-weight: 500; border: 1px solid transparent; white-space: nowrap; }
.bk-file         { background: #3b82f618; color: #3b82f6; border-color: #3b82f630; }
.bk-folder       { background: #f59e0b18; color: #f59e0b; border-color: #f59e0b30; }
.bk-community    { background: #10b98118; color: #10b981; border-color: #10b98130; }
.bk-theme        { background: #8b5cf618; color: #8b5cf6; border-color: #8b5cf630; }
.bk-repo         { background: #ef444418; color: #ef4444; border-color: #ef444430; }
.bk-out_of_scope { background: #6b728018; color: #9ca3af; border-color: #6b728030; }
.bc-cat   { background: #10b98118; color: #10b981; border-color: #10b98130; }
.bc-count { background: #37415118; color: #9ca3af; border-color: #37415130; }
.bc-theme { background: #8b5cf618; color: #8b5cf6; border-color: #8b5cf630; }

/* loading */
#loading-overlay { position: fixed; inset: 0; background: rgba(15,17,23,.92); display: flex; align-items: center; justify-content: center; z-index: 9999; flex-direction: column; gap: 12px; }
.spinner { width: 30px; height: 30px; border: 3px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin .75s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
#loading-msg { font-size: 13px; color: var(--muted); }
.empty-msg { color: var(--muted); font-size: 13px; padding: 30px; text-align: center; }
.err-msg   { color: #ef4444; font-size: 12px; padding: 16px; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
</head>
<body>

<div id="loading-overlay">
  <div class="spinner"></div>
  <span id="loading-msg">Loading…</span>
</div>

<div id="app">
  <header id="hdr">
    <span id="app-title">⬡ Cradle</span>
    <span id="db-name"></span>
    <div id="stats-bar">
      <div class="stat"><span class="stat-val" id="sv-files">—</span><span class="stat-lbl">files</span></div>
      <div class="stat"><span class="stat-val" id="sv-themes">—</span><span class="stat-lbl">themes</span></div>
      <div class="stat"><span class="stat-val" id="sv-comms">—</span><span class="stat-lbl">communities</span></div>
      <div class="stat"><span class="stat-val" id="sv-imports">—</span><span class="stat-lbl">imports</span></div>
      <div class="stat"><span class="stat-val oos" id="sv-oos">—</span><span class="stat-lbl">out-of-scope</span></div>
    </div>
  </header>

  <div id="layout">
    <aside id="sidebar">
      <div class="sb-section">Color by</div>
      <select id="color-by">
        <option value="community">Community</option>
        <option value="category">Category</option>
        <option value="theme">Theme</option>
        <option value="kind">Kind</option>
      </select>

      <div id="sb-detail-filters">
        <div class="sb-section">Filter files</div>
        <select id="f-cat"><option value="">All categories</option></select>
        <select id="f-comm"><option value="">All communities</option></select>
        <select id="f-theme"><option value="">All themes</option></select>
        <div class="sb-section">Edges</div>
        <label class="tog"><input type="checkbox" id="tog-oos" checked> Out-of-scope edges</label>
      </div>

      <div class="sb-section">Legend</div>
      <div id="legend"></div>

      <div class="sb-section">Actions</div>
      <button class="btn-sm" id="btn-fit">Fit view</button>
      <button class="btn-sm" id="btn-relayout">Re-layout</button>
      <button class="btn-sm" id="btn-reset-filters">Reset filters</button>
    </aside>

    <main id="content">
      <nav id="tab-bar">
        <button class="tab active" data-tab="graph">Graph</button>
        <button class="tab" data-tab="themes">Themes</button>
        <button class="tab" data-tab="communities">Communities</button>
        <button class="tab" data-tab="embeddings">Embeddings</button>
        <button class="tab" data-tab="search">Search</button>
      </nav>

      <div id="tab-graph" class="tab-panel active">
        <div id="graph-toolbar">
          <div class="view-btns">
            <button class="vbtn active" id="btn-view-overview">⬡ Overview</button>
            <button class="vbtn" id="btn-view-detail">◉ Files</button>
          </div>
          <button id="btn-back">← All communities</button>
          <span id="graph-info"></span>
          <span style="flex:1"></span>
          <span id="edge-mode-wrap">
            Edges:&nbsp;
            <select id="edge-mode">
              <option value="call-cross">Cross-community calls</option>
              <option value="call-all">All call edges</option>
              <option value="import">Import edges only</option>
            </select>
          </span>
        </div>
        <div id="cy"></div>
      </div>

      <div id="tab-themes" class="tab-panel">
        <div class="card-grid" id="themes-grid"></div>
      </div>

      <div id="tab-communities" class="tab-panel">
        <div class="card-grid" id="communities-grid"></div>
      </div>

      <div id="tab-embeddings" class="tab-panel">
        <div id="emb-toolbar">
          <span style="color:var(--muted)">Color by:</span>
          <select id="emb-color-by" style="width:auto;margin-bottom:0">
            <option value="community">Community</option>
            <option value="category">Category</option>
            <option value="theme">Theme</option>
            <option value="kind">Kind</option>
          </select>
          <span style="color:var(--muted);font-size:11px;margin-left:8px">Click a point to inspect · PCA projection of semantic embeddings</span>
        </div>
        <div id="embedding-plot"></div>
      </div>

      <div id="tab-search" class="tab-panel">
        <div id="search-wrap">
          <input id="search-input" type="text" placeholder="Search files, categories, tags, summaries…" autocomplete="off" spellcheck="false">
          <div id="search-results"></div>
        </div>
      </div>
    </main>

    <aside id="detail-panel" class="closed">
      <button id="close-detail" onclick="hideDetail()">&#x2715;</button>
      <div id="detail-content"></div>
    </aside>
  </div>
</div>

<script>
'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let G = { nodes: [], edges: [], call_edges: [] };
let T = { themes: [] };
let C = { communities: [] };
let E = { points: [], error: null };

let cy          = null;
let embReady    = false;
let colorBy     = 'community';
let fCat        = '';
let fComm       = '';
let fTheme      = '';
let showOOS     = true;
let viewMode    = 'overview';   // 'overview' | 'detail'
let focusedComm = null;
let edgeMode    = 'call-cross'; // 'call-cross' | 'call-all' | 'import'

// Stable community → colour (initialised in loadAll)
const COMM_COLORS = {};

// ── Palette ────────────────────────────────────────────────────────────────
const PALETTE = [
  '#4f8ef7','#10b981','#f59e0b','#ef4444','#8b5cf6',
  '#ec4899','#06b6d4','#84cc16','#f97316','#6366f1',
  '#14b8a6','#a855f7','#f43f5e','#22c55e','#eab308',
  '#0ea5e9','#d946ef','#fb923c','#4ade80','#facc15',
];
const KIND_COLORS = {
  file: '#4f8ef7', folder: '#f59e0b', community: '#10b981',
  theme: '#8b5cf6', repo: '#ef4444', out_of_scope: '#6b7280',
};
const _palCache = {};
function _palColor(key, offset = 0) {
  if (!_palCache[key]) _palCache[key] = PALETTE[(Object.keys(_palCache).length + offset) % PALETTE.length];
  return _palCache[key];
}
function commColor(cid) { return COMM_COLORS[cid] || '#6b7280'; }
function nodeColor(n) {
  if (colorBy === 'kind')      return KIND_COLORS[n.kind] || '#6b7280';
  if (colorBy === 'category')  return _palColor(n.category || '__none__');
  if (colorBy === 'community') return (n.communities && n.communities[0]) ? commColor(n.communities[0]) : '#3d4259';
  if (colorBy === 'theme')     return (n.themes && n.themes[0]) ? _palColor(n.themes[0], 5) : '#3d4259';
  return '#6b7280';
}

// ── Bootstrap ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupSidebar();
  setupSearch();
  setupDetailEvents();
  loadAll();
});

async function loadAll() {
  setLoading('Loading data…');
  const [gd, td, cd] = await Promise.all([
    fetch('/api/graph').then(r => r.json()),
    fetch('/api/themes').then(r => r.json()),
    fetch('/api/communities').then(r => r.json()),
  ]);
  G = gd; T = td; C = cd;

  // Stable community colours
  C.communities.forEach((c, i) => { COMM_COLORS[c.id] = PALETTE[i % PALETTE.length]; });

  document.getElementById('db-name').textContent = 'DB loaded · ' + G.nodes.length + ' nodes';
  populateFilters();
  updateStats();
  initOrUpdateGraph();
  renderThemes();
  renderCommunities();
  renderLegend();
  setLoading(null);
}

function setLoading(msg) {
  const el = document.getElementById('loading-overlay');
  if (msg) { document.getElementById('loading-msg').textContent = msg; el.style.display = 'flex'; }
  else      { el.style.display = 'none'; }
}

// ── Stats ──────────────────────────────────────────────────────────────────
function updateStats() {
  document.getElementById('sv-files').textContent   = G.nodes.filter(n => n.kind === 'file').length;
  document.getElementById('sv-themes').textContent  = T.themes.length;
  document.getElementById('sv-comms').textContent   = C.communities.length;
  document.getElementById('sv-imports').textContent = G.edges.filter(e => e.type === 'import').length;
  document.getElementById('sv-oos').textContent     = G.edges.filter(e => e.type === 'out_of_scope_import').length;
}

// ── Filters ────────────────────────────────────────────────────────────────
function populateFilters() {
  const cats = [...new Set(G.nodes.map(n => n.category).filter(Boolean))].sort();
  const catSel = document.getElementById('f-cat');
  catSel.innerHTML = '<option value="">All categories</option>';
  cats.forEach(c => { const o = document.createElement('option'); o.value = o.textContent = c; catSel.appendChild(o); });

  const commSel = document.getElementById('f-comm');
  commSel.innerHTML = '<option value="">All communities</option>';
  C.communities.forEach(c => { const o = document.createElement('option'); o.value = c.id; o.textContent = c.name; commSel.appendChild(o); });

  const thSel = document.getElementById('f-theme');
  thSel.innerHTML = '<option value="">All themes</option>';
  T.themes.forEach(t => { const o = document.createElement('option'); o.value = t.id; o.textContent = t.name; thSel.appendChild(o); });
}

function setupSidebar() {
  document.getElementById('color-by').addEventListener('change', e => {
    colorBy = e.target.value;
    if (viewMode === 'detail') updateDetailColors();
    renderLegend();
    if (embReady) renderEmbeddings();
  });
  document.getElementById('f-cat').addEventListener('change',  e => { fCat   = e.target.value; if (viewMode === 'detail') initOrUpdateGraph(); });
  document.getElementById('f-comm').addEventListener('change', e => { fComm  = e.target.value; focusedComm = e.target.value || null; if (viewMode === 'detail') initOrUpdateGraph(); });
  document.getElementById('f-theme').addEventListener('change',e => { fTheme = e.target.value; if (viewMode === 'detail') initOrUpdateGraph(); });
  document.getElementById('tog-oos').addEventListener('change', e => { showOOS = e.target.checked; if (viewMode === 'detail') initOrUpdateGraph(); });
  document.getElementById('btn-fit').addEventListener('click',  () => cy && cy.fit(undefined, 40));
  document.getElementById('btn-relayout').addEventListener('click', runLayout);
  document.getElementById('btn-reset-filters').addEventListener('click', resetFilters);
  document.getElementById('btn-view-overview').addEventListener('click', () => setViewMode('overview'));
  document.getElementById('btn-view-detail').addEventListener('click',   () => setViewMode('detail'));
  document.getElementById('btn-back').addEventListener('click', () => {
    focusedComm = null; fComm = '';
    document.getElementById('f-comm').value = '';
    setViewMode('detail');
  });
  document.getElementById('edge-mode').addEventListener('change', e => {
    edgeMode = e.target.value;
    if (viewMode === 'detail') initOrUpdateGraph();
  });
}

function resetFilters() {
  fCat = fComm = fTheme = ''; focusedComm = null;
  document.getElementById('f-cat').value  = '';
  document.getElementById('f-comm').value = '';
  document.getElementById('f-theme').value = '';
  if (viewMode === 'detail') initOrUpdateGraph();
}

// ── View mode ──────────────────────────────────────────────────────────────
function setViewMode(mode, commId) {
  viewMode = mode;
  if (commId !== undefined) {
    focusedComm = commId || null;
    fComm = commId || '';
    document.getElementById('f-comm').value = commId || '';
  }
  document.getElementById('btn-view-overview').classList.toggle('active', mode === 'overview');
  document.getElementById('btn-view-detail').classList.toggle('active',   mode === 'detail');
  document.getElementById('sb-detail-filters').style.display = mode === 'detail' ? 'block' : 'none';
  document.getElementById('edge-mode-wrap').style.display    = mode === 'detail' ? 'flex'  : 'none';
  document.getElementById('btn-back').style.display = (mode === 'detail' && focusedComm) ? 'inline-block' : 'none';
  renderLegend();
  initOrUpdateGraph();
}

// ── Legend ─────────────────────────────────────────────────────────────────
function renderLegend() {
  const container = document.getElementById('legend');
  const items = [];
  if (viewMode === 'overview' || colorBy === 'community') {
    C.communities.forEach(c => items.push([c.name, commColor(c.id)]));
  } else if (colorBy === 'kind') {
    Object.entries(KIND_COLORS).filter(([k]) => k !== 'community' && k !== 'theme' && k !== 'repo').forEach(([k, c]) => items.push([k, c]));
  } else if (colorBy === 'category') {
    [...new Set(G.nodes.map(n => n.category).filter(Boolean))].slice(0, 10).forEach(c => items.push([c, _palColor(c)]));
  } else if (colorBy === 'theme') {
    T.themes.forEach(t => items.push([t.name, _palColor(t.id, 5)]));
  }
  container.innerHTML = items.map(([label, color]) =>
    `<div class="legend-row"><div class="legend-dot" style="background:${color}"></div><span>${esc(label)}</span></div>`
  ).join('');
}

// ── Overview elements ──────────────────────────────────────────────────────
function buildOverviewElements() {
  const fileCommMap = {};
  G.nodes.forEach(n => { if (n.kind === 'file' && n.communities[0]) fileCommMap[n.id] = n.communities[0]; });

  const cyNodes = C.communities.map(c => {
    const col  = commColor(c.id);
    const size = Math.max(60, Math.min(110, 38 + c.members.length * 2.8));
    return { data: { id: c.id, label: c.name, color: col, size, memberCount: c.members.length, bdStyle: 'solid' } };
  });

  const weights = {};
  G.call_edges.forEach(e => {
    const sc = fileCommMap[e.source], tc = fileCommMap[e.target];
    if (sc && tc && sc !== tc) {
      const key = [sc, tc].sort().join('\x01');
      weights[key] = (weights[key] || 0) + e.weight;
    }
  });
  const cyEdges = Object.entries(weights).map(([key, w], i) => {
    const [s, t] = key.split('\x01');
    return { data: { id: `oe${i}`, source: s, target: t, edgeWidth: Math.max(1, Math.min(7, Math.log(w + 1) * 0.85)), edgeColor: '#4b5563', label: String(w), lineStyle: 'solid' } };
  });

  return { cyNodes, cyEdges, nodeCount: cyNodes.length, edgeCount: cyEdges.length };
}

// ── Detail elements (compound graph) ──────────────────────────────────────
function buildDetailElements() {
  let fileNodes = G.nodes.filter(n => n.kind === 'file');
  if (fCat)   fileNodes = fileNodes.filter(n => n.category === fCat);
  if (fComm)  fileNodes = fileNodes.filter(n => n.communities && n.communities.includes(fComm));
  if (fTheme) fileNodes = fileNodes.filter(n => n.themes && n.themes.includes(fTheme));

  const nodeIds = new Set(fileNodes.map(n => n.id));

  // Community compound parents (only for communities with visible files)
  const visComms = new Set(fileNodes.map(n => n.communities && n.communities[0]).filter(Boolean));
  const parentNodes = C.communities
    .filter(c => visComms.has(c.id))
    .map(c => ({ data: { id: `cp__${c.id}`, label: c.name, color: commColor(c.id), isParent: true } }));

  // File child nodes — sized by symbol count, parented to community
  const cyFileNodes = fileNodes.map(n => {
    const cid = n.communities && n.communities[0];
    return { data: {
      id: n.id, label: n.name, kind: n.kind,
      color: nodeColor(n),
      size: Math.max(14, Math.min(44, 10 + Math.sqrt(n.symbol_count || 1) * 2.4)),
      parent: cid ? `cp__${cid}` : undefined,
      bdStyle: 'solid',
    }};
  });

  // OOS ghost nodes + edges
  const oosGhostList = [], oosEdgeList = [];
  if (showOOS) {
    const oosEs = G.edges.filter(e => e.type === 'out_of_scope_import' && nodeIds.has(e.source));
    oosEdgeList.push(...oosEs);
    const ghostIds = new Set(oosEs.map(e => e.target));
    G.nodes.filter(n => n.kind === 'out_of_scope' && ghostIds.has(n.id)).forEach(n => oosGhostList.push(n));
  }

  // Build edges
  const cyEdges = [];
  if (edgeMode !== 'import') {
    const fileCommMapLocal = Object.fromEntries(G.nodes.map(n => [n.id, (n.communities && n.communities[0]) || '']));
    let callEdgesToUse = G.call_edges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target));
    if (edgeMode === 'call-cross') callEdgesToUse = callEdgesToUse.filter(e => fileCommMapLocal[e.source] !== fileCommMapLocal[e.target]);
    callEdgesToUse.slice(0, 700).forEach((e, i) => cyEdges.push({ data: {
      id: `ce${i}`, source: e.source, target: e.target,
      edgeColor: '#3d4259',
      edgeWidth: Math.max(0.5, Math.min(3, Math.log(e.weight + 1) * 0.55)),
      lineStyle: 'solid',
    }}));
  }
  if (edgeMode === 'import' || edgeMode === 'call-all') {
    G.edges
      .filter(e => e.type === 'import' && nodeIds.has(e.source) && nodeIds.has(e.target))
      .forEach((e, i) => cyEdges.push({ data: { id: `ie${i}`, source: e.source, target: e.target, edgeColor: '#4f8ef755', edgeWidth: 0.8, lineStyle: 'dashed' }}));
  }
  oosEdgeList.forEach((e, i) => cyEdges.push({ data: { id: `oose${i}`, source: e.source, target: e.target, edgeColor: '#ef4444', edgeWidth: 1, lineStyle: 'dashed' }}));

  const cyOosNodes = oosGhostList.map(n => ({ data: { id: n.id, label: n.name, kind: 'out_of_scope', color: '#6b7280', size: 16, bdStyle: 'dashed' } }));

  return {
    cyNodes: [...parentNodes, ...cyFileNodes, ...cyOosNodes],
    cyEdges,
    nodeCount: cyFileNodes.length,
    edgeCount: cyEdges.length,
  };
}

// ── Cytoscape style ────────────────────────────────────────────────────────
function getCyStyle() {
  const s = [
    { selector: 'node', style: {
      'background-color': 'data(color)', 'label': 'data(label)',
      'color': '#e8eaf0', 'font-size': '9px',
      'text-valign': 'bottom', 'text-margin-y': 4,
      'text-outline-width': 2, 'text-outline-color': '#0f1117',
      'width': 'data(size)', 'height': 'data(size)',
      'border-width': 1.5, 'border-color': '#2e3347', 'border-style': 'data(bdStyle)',
    }},
    { selector: ':parent', style: {
      'background-color': 'data(color)', 'background-opacity': 0.10,
      'border-color': 'data(color)', 'border-width': 2, 'border-opacity': 0.55,
      'label': 'data(label)', 'text-valign': 'top', 'text-halign': 'center',
      'text-margin-y': 2, 'font-size': '11px', 'font-weight': '600',
      'color': 'data(color)', 'text-outline-width': 0,
      'padding': '24px', 'shape': 'roundrectangle',
    }},
    { selector: 'node.dim',    style: { 'opacity': 0.12 } },
    { selector: ':parent.dim', style: { 'opacity': 0.07 } },
    { selector: 'node:selected', style: { 'border-color': '#f9fafb', 'border-width': 3 } },
    { selector: 'edge', style: {
      'line-color': 'data(edgeColor)', 'target-arrow-color': 'data(edgeColor)',
      'target-arrow-shape': 'triangle', 'curve-style': 'bezier',
      'width': 'data(edgeWidth)', 'opacity': 0.55,
      'line-style': 'data(lineStyle)', 'arrow-scale': 0.6,
    }},
    { selector: 'edge:selected', style: { 'opacity': 1, 'width': 2.5 } },
  ];

  if (viewMode === 'overview') {
    s.push({ selector: 'node', style: {
      'font-size': '11px', 'text-valign': 'center', 'text-halign': 'center',
      'color': '#ffffff', 'text-outline-width': 0,
      'text-wrap': 'wrap', 'text-max-width': '80px',
    }});
    s.push({ selector: 'edge', style: {
      'curve-style': 'haystack', 'target-arrow-shape': 'none',
      'label': 'data(label)', 'font-size': '9px', 'color': '#9ca3af',
      'text-outline-width': 1.5, 'text-outline-color': '#0f1117',
    }});
  }
  return s;
}

// ── Layout ─────────────────────────────────────────────────────────────────
function getLayout() {
  if (viewMode === 'overview') {
    return { name: 'circle', padding: 80, animate: true, animationDuration: 500, startAngle: -Math.PI / 2 };
  }
  return {
    name: 'cose',
    animate: true, animationDuration: 900,
    nodeRepulsion: 500000,
    idealEdgeLength: 70,
    edgeElasticity: 150,
    nestingFactor: 1.2,
    gravity: 80,
    numIter: 1000,
    padding: 28,
    componentSpacing: 80,
    nodeOverlap: 8,
    fit: true,
  };
}

// ── Graph init/update ──────────────────────────────────────────────────────
function initOrUpdateGraph() {
  const { cyNodes, cyEdges, nodeCount, edgeCount } =
    viewMode === 'overview' ? buildOverviewElements() : buildDetailElements();

  const infoTxt = viewMode === 'overview'
    ? `${nodeCount} communities · ${edgeCount} cross-community call flows`
    : `${nodeCount} files · ${edgeCount} edges`;
  document.getElementById('graph-info').textContent = infoTxt;

  if (!cy) {
    cy = cytoscape({
      container: document.getElementById('cy'),
      elements: [...cyNodes, ...cyEdges],
      style: getCyStyle(),
      layout: getLayout(),
      minZoom: 0.04, maxZoom: 10,
    });
    cy.on('tap', 'node', handleNodeTap);
    cy.on('tap', evt => { if (evt.target === cy) clearHighlight(); });
  } else {
    cy.elements().remove();
    cy.setStyle(getCyStyle());
    cy.add([...cyNodes, ...cyEdges]);
    cy.layout(getLayout()).run();
  }
}

function handleNodeTap(evt) {
  const nid = evt.target.id();
  if (viewMode === 'overview') { setViewMode('detail', nid); return; }
  if (nid.startsWith('cp__'))  { cy.fit(cy.nodes(`[parent = "${nid}"]`), 40); return; }
  if (nid.startsWith('__oos__')) { showOOSDetail(nid, evt.target.data('label')); return; }
  showNodeDetail(nid);
}

function clearHighlight() {
  hideDetail();
  if (cy) cy.elements().removeClass('dim');
}

function runLayout() {
  if (cy) cy.layout(getLayout()).run();
}

function updateDetailColors() {
  if (!cy || viewMode === 'overview') return;
  const nodeMap = Object.fromEntries(G.nodes.map(n => [n.id, n]));
  cy.nodes().forEach(cn => { const n = nodeMap[cn.id()]; if (n) cn.data('color', nodeColor(n)); });
}

// ── Themes tab ─────────────────────────────────────────────────────────────
function renderThemes() {
  const el = document.getElementById('themes-grid');
  if (!T.themes.length) { el.innerHTML = '<p class="empty-msg">No themes found.</p>'; return; }
  el.innerHTML = T.themes.map(t => `
    <div class="card">
      <div class="card-head">
        <span class="badge bc-theme">${esc(t.name)}</span>
        <span class="badge bc-count">${t.members.length} files</span>
      </div>
      <p class="card-summary">${esc(t.summary)}</p>
      <details>
        <summary>View files (${t.members.length})</summary>
        <ul class="member-list">
          ${t.members.slice(0, 25).map(m => `<li class="member-item" data-node-id="${esc(m)}">${esc(m)}</li>`).join('')}
          ${t.members.length > 25 ? `<li class="member-more">\u2026and ${t.members.length - 25} more</li>` : ''}
        </ul>
      </details>
      <button class="btn-focus" data-focus-theme="${esc(t.id)}">Focus in graph \u2197</button>
    </div>`).join('');

  el.addEventListener('click', async e => {
    const ni = e.target.closest('[data-node-id]');  if (ni) { await showNodeDetail(ni.dataset.nodeId); return; }
    const fi = e.target.closest('[data-focus-theme]'); if (fi) focusTheme(fi.dataset.focusTheme);
  });
}

function focusTheme(tid) {
  fTheme = tid; fComm = ''; focusedComm = null;
  document.getElementById('f-theme').value = tid;
  document.getElementById('f-comm').value  = '';
  switchTab('graph');
  setViewMode('detail');
}

// ── Communities tab ────────────────────────────────────────────────────────
function renderCommunities() {
  const el = document.getElementById('communities-grid');
  if (!C.communities.length) { el.innerHTML = '<p class="empty-msg">No communities found.</p>'; return; }
  el.innerHTML = C.communities.map(c => {
    const col = commColor(c.id);
    return `
    <div class="card">
      <div class="card-head">
        <span class="badge" style="background:${col}18;color:${col};border-color:${col}30">${esc(c.name)}</span>
        <span class="badge bc-count">${c.members.length} files</span>
        ${c.category ? `<span class="badge bc-cat">${esc(c.category)}</span>` : ''}
      </div>
      <p class="card-summary">${esc(c.summary)}</p>
      <details>
        <summary>View files (${c.members.length})</summary>
        <ul class="member-list">
          ${c.members.slice(0, 25).map(m => `<li class="member-item" data-node-id="${esc(m)}">${esc(m)}</li>`).join('')}
          ${c.members.length > 25 ? `<li class="member-more">\u2026and ${c.members.length - 25} more</li>` : ''}
        </ul>
      </details>
      <button class="btn-focus" data-focus-comm="${esc(c.id)}">Focus in graph \u2197</button>
    </div>`;
  }).join('');

  el.addEventListener('click', async e => {
    const ni = e.target.closest('[data-node-id]');  if (ni) { await showNodeDetail(ni.dataset.nodeId); return; }
    const fi = e.target.closest('[data-focus-comm]'); if (fi) focusCommunity(fi.dataset.focusComm);
  });
}

function focusCommunity(cid) {
  switchTab('graph');
  setViewMode('detail', cid);
}

// ── Embeddings tab ─────────────────────────────────────────────────────────
async function loadEmbeddings() {
  if (embReady) { renderEmbeddings(); return; }
  setLoading('Computing embedding projection\u2026');
  try {
    E = await fetch('/api/embeddings').then(r => r.json());
    embReady = true;
    renderEmbeddings();
  } catch (err) {
    document.getElementById('embedding-plot').innerHTML = `<p class="err-msg">Failed: ${esc(err.message)}</p>`;
  } finally { setLoading(null); }
}

function renderEmbeddings() {
  const plotEl = document.getElementById('embedding-plot');
  if (E.error) { plotEl.innerHTML = `<p class="empty-msg">${esc(E.error)}</p>`; return; }
  const cby = document.getElementById('emb-color-by').value;
  const groups = {};
  const nodeMap = Object.fromEntries(G.nodes.map(n => [n.id, n]));
  E.points.forEach(p => {
    let key;
    if      (cby === 'kind')      key = p.kind;
    else if (cby === 'category')  key = p.category || 'unknown';
    else if (cby === 'community') { const gn = nodeMap[p.id]; key = (gn && gn.communities && gn.communities[0]) || 'none'; }
    else                          { const gn = nodeMap[p.id]; key = (gn && gn.themes && gn.themes[0]) || 'none'; }
    if (!groups[key]) groups[key] = { x: [], y: [], text: [], ids: [] };
    groups[key].x.push(p.x); groups[key].y.push(p.y);
    groups[key].text.push(`<b>${esc(p.name)}</b><br>${p.kind}${p.category ? ' \u00b7 ' + p.category : ''}<br><i>${esc((p.summary || '').substring(0, 100))}\u2026</i>`);
    groups[key].ids.push(p.id);
  });
  const commNameMap  = Object.fromEntries(C.communities.map(c => [c.id, c.name]));
  const themeNameMap = Object.fromEntries(T.themes.map(t  => [t.id, t.name]));
  const traces = Object.entries(groups).map(([key, g]) => {
    const displayName = commNameMap[key] || themeNameMap[key] || key;
    const color = cby === 'community' ? commColor(key) : cby === 'kind' ? (KIND_COLORS[key] || '#6b7280') : _palColor(key, cby === 'theme' ? 5 : 0);
    return { type: 'scatter', mode: 'markers', name: displayName, x: g.x, y: g.y, text: g.text, hoverinfo: 'text', customdata: g.ids, marker: { size: 7, color, opacity: 0.85 } };
  });
  Plotly.newPlot(plotEl, traces, {
    paper_bgcolor: '#0f1117', plot_bgcolor: '#1a1d27',
    font: { color: '#e8eaf0', family: 'system-ui,sans-serif', size: 11 },
    xaxis: { showgrid: false, zeroline: false, showticklabels: false, title: 'PC 1' },
    yaxis: { showgrid: false, zeroline: false, showticklabels: false, title: 'PC 2' },
    legend: { bgcolor: '#1a1d27', bordercolor: '#2e3347', borderwidth: 1 },
    margin: { t: 16, b: 40, l: 40, r: 16 }, hovermode: 'closest',
  }, { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['select2d', 'lasso2d'] });
  plotEl.on('plotly_click', async data => { await showNodeDetail(data.points[0].customdata); });
}

// ── Search tab ─────────────────────────────────────────────────────────────
function setupSearch() {
  const inp = document.getElementById('search-input');
  inp.addEventListener('input', () => {
    const q = inp.value.trim().toLowerCase();
    const res = document.getElementById('search-results');
    if (!q) { res.innerHTML = ''; return; }
    const hits = G.nodes
      .filter(n => ['file', 'folder', 'community', 'theme'].includes(n.kind))
      .filter(n => n.name.toLowerCase().includes(q) || n.path.toLowerCase().includes(q) ||
        (n.summary || '').toLowerCase().includes(q) || (n.category || '').toLowerCase().includes(q) ||
        n.tags.some(t => t.toLowerCase().includes(q)) || n.categories.some(c => c.toLowerCase().includes(q)))
      .slice(0, 40);
    res.innerHTML = hits.length
      ? hits.map(n => `<div class="sr" data-node-id="${esc(n.id)}">
          <div class="sr-head"><span class="badge bk-${n.kind}">${n.kind}</span><span class="sr-path">${esc(n.path)}</span>${n.category ? `<span class="badge bc-cat">${esc(n.category)}</span>` : ''}</div>
          <p class="sr-summary">${esc((n.summary || '').substring(0, 140))}${n.summary && n.summary.length > 140 ? '\u2026' : ''}</p>
        </div>`).join('')
      : '<p class="empty-msg">No results.</p>';
  });
  document.getElementById('search-results').addEventListener('click', async e => {
    const el = e.target.closest('[data-node-id]'); if (el) await showNodeDetail(el.dataset.nodeId);
  });
}

// ── Detail panel ───────────────────────────────────────────────────────────
function setupDetailEvents() {
  document.getElementById('detail-content').addEventListener('click', async e => {
    const el = e.target.closest('[data-node-id]'); if (el) await showNodeDetail(el.dataset.nodeId);
  });
}

async function showNodeDetail(nodeId) {
  const panel   = document.getElementById('detail-panel');
  const content = document.getElementById('detail-content');
  panel.classList.remove('closed');
  content.innerHTML = '<p class="det-summary" style="padding:8px">Loading\u2026</p>';
  try {
    const d = await fetch('/api/node?id=' + encodeURIComponent(nodeId)).then(r => r.json());
    if (d.error) { content.innerHTML = `<p class="err-msg">${esc(d.error)}</p>`; return; }
    content.innerHTML = `
      <div class="det-head">
        <span class="badge bk-${d.kind}">${d.kind}</span>
        <div class="det-name">${esc(d.name)}</div>
        <div class="det-path">${esc(d.path)}</div>
      </div>
      ${d.primary_category ? `<div style="margin-bottom:10px"><span class="badge bc-cat">${esc(d.primary_category)}</span></div>` : ''}
      <div class="det-stats">
        ${d.symbol_count  ? `<span>symbols: <b>${d.symbol_count}</b></span>` : ''}
        ${d.import_count  ? `<span>imports: <b>${d.import_count}</b></span>` : ''}
        ${d.file_count    ? `<span>files: <b>${d.file_count}</b></span>` : ''}
        ${d.folder_count  ? `<span>folders: <b>${d.folder_count}</b></span>` : ''}
        <span>confidence: <b>${(d.confidence * 100).toFixed(0)}%</b></span>
      </div>
      ${d.summary ? `<div class="det-sec"><p class="det-summary">${esc(d.summary)}</p></div>` : ''}
      ${d.tags.length ? `<div class="det-sec"><div class="tag-row">${d.tags.map(t => `<span class="tag">${esc(t)}</span>`).join('')}</div></div>` : ''}
      ${d.symbols.length ? `
        <details class="det-sec" open>
          <summary class="det-sec-title">Symbols (${d.symbols.length}${d.symbol_count > d.symbols.length ? ', showing first ' + d.symbols.length : ''})</summary>
          <ul class="sym-list">
            ${d.symbols.map(s => `<li class="sym-item">
              <span><span class="sym-kind">${esc(s.kind)}</span><span class="sym-name">${esc(s.name)}</span></span>
              ${s.signature ? `<code class="sym-sig">${esc(s.signature)}</code>` : ''}
              ${s.summary ? `<p class="sym-sum">${esc(s.summary)}</p>` : ''}
            </li>`).join('')}
          </ul>
        </details>` : ''}
      ${d.imports_out.length ? `
        <details class="det-sec">
          <summary class="det-sec-title">Imports out (${d.imports_out.length})</summary>
          <ul class="imp-list">
            ${d.imports_out.map(i => `<li class="imp-item${i.oos ? ' oos' : ''}">
              ${i.oos ? '<span class="oos-pill">out-of-scope</span>' : ''}
              <span class="imp-mod">${esc(i.module)}</span>
              <span class="imp-str">${esc(i.strength)}</span>
            </li>`).join('')}
          </ul>
        </details>` : ''}
      ${d.imports_in.length ? `
        <details class="det-sec">
          <summary class="det-sec-title">Imported by (${d.imports_in.length}${d.import_count > d.imports_in.length ? '+' : ''})</summary>
          <ul class="imp-list">
            ${d.imports_in.map(i => `<li class="imp-item clickable" data-node-id="${esc(i.source)}">
              <span class="imp-mod">${esc(i.source)}</span>
              <span class="imp-str">${esc(i.strength)}</span>
            </li>`).join('')}
          </ul>
        </details>` : ''}
    `;
    if (cy && viewMode === 'detail') {
      cy.elements().removeClass('dim');
      const cyN = cy.getElementById(nodeId);
      if (cyN.length) {
        cy.elements().not(cyN.closedNeighborhood()).addClass('dim');
        cy.animate({ fit: { eles: cyN.closedNeighborhood(), padding: 60 }, duration: 350 });
      }
    }
  } catch (err) {
    content.innerHTML = `<p class="err-msg">Error: ${esc(err.message)}</p>`;
  }
}

function showOOSDetail(nodeId, name) {
  const panel   = document.getElementById('detail-panel');
  const content = document.getElementById('detail-content');
  panel.classList.remove('closed');
  content.innerHTML = `
    <div class="det-head">
      <span class="badge bk-out_of_scope">out-of-scope</span>
      <div class="det-name">${esc(name)}</div>
    </div>
    <div class="det-sec"><p class="det-summary">
      This module was imported as an internal dependency but its target file was
      not found in the analysed root. It may live in a parent package or sibling
      project that Cradle did not analyse.
    </p></div>`;
}

function hideDetail() {
  document.getElementById('detail-panel').classList.add('closed');
  if (cy) cy.elements().removeClass('dim');
}

// ── Tab switching ──────────────────────────────────────────────────────────
function setupTabs() {
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
  document.getElementById('emb-color-by').addEventListener('change', () => { if (embReady) renderEmbeddings(); });
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
  if (name === 'embeddings') loadEmbeddings();
  if (name === 'graph' && cy) cy.resize();
}

// ── Utility ────────────────────────────────────────────────────────────────
function esc(s) {
  if (!s) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
</script>
</body>
</html>
"""
