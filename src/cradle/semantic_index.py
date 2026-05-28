from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from math import isqrt
from pathlib import Path

import numpy as np

from cradle.embeddings import DEFAULT_QUERY_TASK, TextEmbedder, format_document_text

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 2


@dataclass(slots=True)
class SemanticRecord:
    entity_id: str
    entity_type: str
    title: str
    content: str
    path: str
    kind: str


@dataclass(slots=True)
class NodeCentroidRecord:
    centroid_id: str
    parent_id: str
    member_ids: list[str]
    representative_ids: list[str]
    title: str
    content: str
    summary: str = ""
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SemanticIndexConfig:
    output_dir: Path | None = None
    default_query_task: str = DEFAULT_QUERY_TASK
    max_child_names: int = 12
    max_symbol_names: int = 24
    max_imports: int = 12
    max_contexts: int = 6
    max_callers: int = 8
    max_callees: int = 8
    max_references: int = 8
    min_centroid_children: int = 4
    max_centroids_per_parent: int = 4
    centroid_iterations: int = 10
    centroid_preview_members: int = 6


@dataclass(slots=True)
class SemanticIndexSummary:
    index_dir: Path
    model_name: str
    dimension: int
    node_count: int
    symbol_count: int
    centroid_count: int
    engine: str


@dataclass(slots=True)
class SemanticIndexManifest:
    version: int
    db_path: str
    db_updated_at: str | None
    model_name: str
    dimension: int
    engine: str
    default_query_task: str
    built_at: str
    node_count: int
    symbol_count: int
    centroid_count: int


class SemanticIndexPaths:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.manifest_path = self.root_dir / "manifest.json"
        self.node_records_path = self.root_dir / "nodes.records.json"
        self.node_vectors_path = self.root_dir / "nodes.vectors.npy"
        self.node_index_path = self.root_dir / "nodes.faiss"
        self.symbol_records_path = self.root_dir / "symbols.records.json"
        self.symbol_vectors_path = self.root_dir / "symbols.vectors.npy"
        self.symbol_index_path = self.root_dir / "symbols.faiss"
        self.centroid_records_path = self.root_dir / "node_centroids.records.json"
        self.centroid_vectors_path = self.root_dir / "node_centroids.vectors.npy"


class SemanticIndexBuilder:
    def __init__(self, db_path: str | Path, *, embedder: TextEmbedder, config: SemanticIndexConfig | None = None) -> None:
        self._db_path = Path(db_path)
        self._embedder = embedder
        self._config = config or SemanticIndexConfig()
        self._paths = SemanticIndexPaths(default_semantic_index_dir(self._db_path, self._config.output_dir))

    @property
    def index_dir(self) -> Path:
        return self._paths.root_dir

    def build(self) -> SemanticIndexSummary:
        with _connect(self._db_path) as conn:
            db_updated_at = _db_updated_at(conn)
            node_records = _load_node_records(conn, self._config)
            symbol_records = _load_symbol_records(conn, self._config)
            child_map = _load_rollup_child_map(conn)

        self._paths.root_dir.mkdir(parents=True, exist_ok=True)

        node_vectors = self._embed_records(node_records)
        symbol_vectors = self._embed_records(symbol_records)
        centroid_records, centroid_vectors = _build_node_centroid_records(node_records, node_vectors, child_map, self._config)
        engine = _write_vector_artifacts(
            self._paths,
            node_records,
            node_vectors,
            symbol_records,
            symbol_vectors,
            centroid_records,
            centroid_vectors,
        )

        manifest = SemanticIndexManifest(
            version=MANIFEST_VERSION,
            db_path=str(self._db_path),
            db_updated_at=db_updated_at,
            model_name=self._embedder.model_name,
            dimension=self._embedder.dimension,
            engine=engine,
            default_query_task=self._config.default_query_task,
            built_at=_utc_now(),
            node_count=len(node_records),
            symbol_count=len(symbol_records),
            centroid_count=len(centroid_records),
        )
        self._paths.manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
        logger.info(
            "built semantic index at %s: %s nodes, %s symbols, %s centroids, %s-dim via %s",
            self._paths.root_dir,
            len(node_records),
            len(symbol_records),
            len(centroid_records),
            self._embedder.dimension,
            engine,
        )
        return SemanticIndexSummary(
            index_dir=self._paths.root_dir,
            model_name=self._embedder.model_name,
            dimension=self._embedder.dimension,
            node_count=len(node_records),
            symbol_count=len(symbol_records),
            centroid_count=len(centroid_records),
            engine=engine,
        )

    def _embed_records(self, records: list[SemanticRecord]) -> np.ndarray:
        prompts = [format_document_text(record.content, title=record.title) for record in records]
        return self._embedder.encode(prompts, show_progress_bar=True)


class SemanticIndexStore:
    def __init__(self, db_path: str | Path, *, index_dir: str | Path | None = None) -> None:
        self._db_path = Path(db_path)
        self._paths = SemanticIndexPaths(default_semantic_index_dir(self._db_path, index_dir))
        if not self._paths.manifest_path.exists():
            raise FileNotFoundError(
                f"Semantic index manifest not found at {self._paths.manifest_path}. Run `cradle semantic-index` first."
            )
        self.manifest = load_semantic_manifest(self._db_path, index_dir=index_dir)
        self.node_records = _load_records(self._paths.node_records_path)
        self.symbol_records = _load_records(self._paths.symbol_records_path)
        self.centroid_records = _load_centroid_records(self._paths.centroid_records_path)
        self._node_positions = {record.entity_id: index for index, record in enumerate(self.node_records)}
        self._symbol_positions = {record.entity_id: index for index, record in enumerate(self.symbol_records)}
        self._centroid_positions = {record.centroid_id: index for index, record in enumerate(self.centroid_records)}
        self._centroid_ids_by_parent: dict[str, list[str]] = defaultdict(list)
        for record in self.centroid_records:
            self._centroid_ids_by_parent[record.parent_id].append(record.centroid_id)
        self._node_vectors = np.load(self._paths.node_vectors_path).astype(np.float32)
        self._symbol_vectors = np.load(self._paths.symbol_vectors_path).astype(np.float32)
        self._centroid_vectors = (
            np.load(self._paths.centroid_vectors_path).astype(np.float32)
            if self._paths.centroid_vectors_path.exists()
            else np.zeros((0, self.dimension), dtype=np.float32)
        )
        self._node_index = _load_faiss_index(self._paths.node_index_path)
        self._symbol_index = _load_faiss_index(self._paths.symbol_index_path)

    @property
    def default_query_task(self) -> str:
        return str(self.manifest["default_query_task"])

    @property
    def dimension(self) -> int:
        return int(self.manifest["dimension"])

    @property
    def model_name(self) -> str:
        return str(self.manifest["model_name"])

    def search_nodes(self, query_vector: np.ndarray, *, top_k: int) -> list[tuple[str, float]]:
        return self._search(self._node_index, self._node_vectors, self.node_records, query_vector, top_k)

    def search_symbols(self, query_vector: np.ndarray, *, top_k: int) -> list[tuple[str, float]]:
        return self._search(self._symbol_index, self._symbol_vectors, self.symbol_records, query_vector, top_k)

    def search_node_subset(self, query_vector: np.ndarray, node_ids: list[str], *, top_k: int) -> list[tuple[str, float]]:
        return self._search_subset(self._node_vectors, self.node_records, self._node_positions, query_vector, node_ids, top_k)

    def search_symbol_subset(self, query_vector: np.ndarray, symbol_ids: list[str], *, top_k: int) -> list[tuple[str, float]]:
        return self._search_subset(self._symbol_vectors, self.symbol_records, self._symbol_positions, query_vector, symbol_ids, top_k)

    def search_node_centroids(self, query_vector: np.ndarray, parent_ids: list[str], *, top_k: int) -> list[tuple[str, float]]:
        centroid_ids: list[str] = []
        for parent_id in dict.fromkeys(parent_ids):
            centroid_ids.extend(self._centroid_ids_by_parent.get(parent_id, []))
        return self._search_centroid_subset(query_vector, centroid_ids, top_k)

    def centroid_member_ids(self, centroid_id: str) -> list[str]:
        position = self._centroid_positions.get(centroid_id)
        if position is None:
            return []
        return list(self.centroid_records[position].member_ids)

    def _search(
        self,
        faiss_index,
        vectors: np.ndarray,
        records: list[SemanticRecord],
        query_vector: np.ndarray,
        top_k: int,
    ) -> list[tuple[str, float]]:
        if not records or top_k <= 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        limit = min(top_k, len(records))
        if faiss_index is not None:
            scores, indices = faiss_index.search(query, limit)
            return [
                (records[index].entity_id, float(score))
                for score, index in zip(scores[0], indices[0], strict=False)
                if index >= 0
            ]

        scores = vectors @ query[0]
        ranking = np.argsort(-scores)[:limit]
        return [(records[index].entity_id, float(scores[index])) for index in ranking]

    def _search_subset(
        self,
        vectors: np.ndarray,
        records: list[SemanticRecord],
        positions: dict[str, int],
        query_vector: np.ndarray,
        ids: list[str],
        top_k: int,
    ) -> list[tuple[str, float]]:
        if not ids or top_k <= 0:
            return []

        unique_positions = [positions[item_id] for item_id in dict.fromkeys(ids) if item_id in positions]
        if not unique_positions:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        subset_vectors = vectors[unique_positions]
        scores = subset_vectors @ query[0]
        ranking = np.argsort(-scores)[: min(top_k, len(unique_positions))]
        return [(records[unique_positions[index]].entity_id, float(scores[index])) for index in ranking]

    def _search_centroid_subset(self, query_vector: np.ndarray, centroid_ids: list[str], top_k: int) -> list[tuple[str, float]]:
        if not centroid_ids or top_k <= 0 or self._centroid_vectors.size == 0:
            return []

        unique_positions = [self._centroid_positions[item_id] for item_id in dict.fromkeys(centroid_ids) if item_id in self._centroid_positions]
        if not unique_positions:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        subset_vectors = self._centroid_vectors[unique_positions]
        scores = subset_vectors @ query[0]
        ranking = np.argsort(-scores)[: min(top_k, len(unique_positions))]
        return [(self.centroid_records[unique_positions[index]].centroid_id, float(scores[index])) for index in ranking]


def default_semantic_index_dir(db_path: str | Path, output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    path = Path(db_path)
    return path.parent / f"{path.stem}.semantic"


def load_semantic_manifest(db_path: str | Path, *, index_dir: str | Path | None = None) -> dict[str, object]:
    paths = SemanticIndexPaths(default_semantic_index_dir(db_path, index_dir))
    return json.loads(paths.manifest_path.read_text(encoding="utf-8"))


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _write_vector_artifacts(
    paths: SemanticIndexPaths,
    node_records: list[SemanticRecord],
    node_vectors: np.ndarray,
    symbol_records: list[SemanticRecord],
    symbol_vectors: np.ndarray,
    centroid_records: list[NodeCentroidRecord],
    centroid_vectors: np.ndarray,
) -> str:
    paths.node_records_path.write_text(json.dumps([asdict(record) for record in node_records], indent=2), encoding="utf-8")
    paths.symbol_records_path.write_text(json.dumps([asdict(record) for record in symbol_records], indent=2), encoding="utf-8")
    paths.centroid_records_path.write_text(json.dumps([asdict(record) for record in centroid_records], indent=2), encoding="utf-8")
    np.save(paths.node_vectors_path, node_vectors)
    np.save(paths.symbol_vectors_path, symbol_vectors)
    np.save(paths.centroid_vectors_path, centroid_vectors)

    faiss = _load_faiss()
    if faiss is None:
        return "numpy"

    node_index = faiss.IndexFlatIP(int(node_vectors.shape[1]))
    node_index.add(node_vectors)
    symbol_index = faiss.IndexFlatIP(int(symbol_vectors.shape[1]))
    symbol_index.add(symbol_vectors)
    faiss.write_index(node_index, str(paths.node_index_path))
    faiss.write_index(symbol_index, str(paths.symbol_index_path))
    return "faiss"


def _load_records(path: Path) -> list[SemanticRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [SemanticRecord(**item) for item in payload]


def _load_centroid_records(path: Path) -> list[NodeCentroidRecord]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [NodeCentroidRecord(**item) for item in payload]


def _load_faiss():
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError:
        return None
    return faiss


def _load_faiss_index(path: Path):
    if not path.exists():
        return None
    faiss = _load_faiss()
    if faiss is None:
        return None
    return faiss.read_index(str(path))


def _db_updated_at(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'updated_at'").fetchone()
    return None if row is None else row[0]


def _load_node_records(conn: sqlite3.Connection, config: SemanticIndexConfig) -> list[SemanticRecord]:
    node_rows = conn.execute("SELECT * FROM nodes ORDER BY node_id").fetchall()
    category_map = _group_single_values(conn, "SELECT node_id, category FROM node_categories ORDER BY rank")
    tag_map = _group_single_values(conn, "SELECT node_id, tag FROM node_tags ORDER BY rank")
    symbol_map = _group_single_values(conn, "SELECT node_id, name FROM symbols ORDER BY start_line, name")
    import_map = _group_single_values(conn, "SELECT importer_node_id, imported_module FROM imports ORDER BY start_line, imported_module")
    context_map = _group_pairs(conn, "SELECT node_id, source_path, inherited_summary FROM node_context ORDER BY weight DESC, id")
    child_map = _group_single_values(conn, "SELECT parent_id, name FROM nodes WHERE parent_id IS NOT NULL ORDER BY kind, name")
    community_member_map = _group_single_values(conn, "SELECT community_node_id, member_node_id FROM community_members ORDER BY membership_rank, member_node_id")
    theme_member_map = _group_single_values(conn, "SELECT theme_node_id, member_node_id FROM theme_members ORDER BY membership_rank, member_node_id")

    records: list[SemanticRecord] = []
    for row in node_rows:
        categories = category_map.get(row["node_id"], [])
        tags = tag_map.get(row["node_id"], [])
        symbols = symbol_map.get(row["node_id"], [])
        imports = import_map.get(row["node_id"], [])
        contexts = context_map.get(row["node_id"], [])
        children = child_map.get(row["node_id"], [])
        lines = [
            f"type: {row['kind']}",
            f"path: {row['path']}",
            f"name: {row['name']}",
        ]
        if row["language"]:
            lines.append(f"language: {row['language']}")
        if row["primary_category"]:
            lines.append(f"category: {row['primary_category']}")
        if categories:
            lines.append(f"categories: {', '.join(categories[:6])}")
        if tags:
            lines.append(f"tags: {', '.join(tags[:8])}")
        if row["summary"]:
            lines.append(f"summary: {row['summary']}")
        if row["description"]:
            lines.append(f"description: {row['description']}")
        child_categories = _json_list(row["top_child_categories_json"])
        if child_categories:
            lines.append(f"child_categories: {', '.join(child_categories[:6])}")
        dependency_tags = _json_list(row["top_dependency_tags_json"])
        if dependency_tags:
            lines.append(f"dependency_tags: {', '.join(dependency_tags[:8])}")
        if row["kind"] == "file":
            if symbols:
                lines.append(f"symbols: {', '.join(symbols[: config.max_symbol_names])}")
            if imports:
                lines.append(f"imports: {', '.join(imports[: config.max_imports])}")
            if contexts:
                lines.append(f"context: {' | '.join(contexts[: config.max_contexts])}")
        elif row["kind"] == "community":
            members = community_member_map.get(row["node_id"], [])
            if members:
                lines.append(f"members: {', '.join(members[: config.max_child_names])}")
            lines.append(f"counts: files={row['file_count']} folders={row['folder_count']} symbols={row['symbol_count']}")
        elif row["kind"] == "theme":
            members = theme_member_map.get(row["node_id"], [])
            if members:
                lines.append(f"theme_members: {', '.join(members[: config.max_child_names])}")
            lines.append(f"counts: files={row['file_count']} folders={row['folder_count']} symbols={row['symbol_count']}")
        else:
            if children:
                lines.append(f"children: {', '.join(children[: config.max_child_names])}")
            lines.append(f"counts: files={row['file_count']} folders={row['folder_count']} symbols={row['symbol_count']}")

        title = row["path"] if row["path"] else row["name"]
        records.append(
            SemanticRecord(
                entity_id=row["node_id"],
                entity_type="node",
                title=title,
                content="\n".join(lines),
                path=row["path"],
                kind=row["kind"],
            )
        )
    return records


def _load_symbol_records(conn: sqlite3.Connection, config: SemanticIndexConfig) -> list[SemanticRecord]:
    symbol_rows = conn.execute(
        """
        SELECT symbols.*, nodes.summary AS node_summary, nodes.primary_category AS node_category
        FROM symbols
        JOIN nodes ON nodes.node_id = symbols.node_id
        ORDER BY symbols.symbol_id
        """
    ).fetchall()
    callers_map = _group_single_values(conn, "SELECT target_symbol_id, caller_node_id FROM call_sites WHERE target_symbol_id IS NOT NULL ORDER BY start_line, caller_node_id")
    callees_map = _group_single_values(conn, "SELECT caller_symbol_id, callee_name FROM call_sites ORDER BY start_line, callee_name")
    references_map = _group_single_values(conn, "SELECT target_symbol_id, source_node_id FROM symbol_references WHERE target_symbol_id IS NOT NULL ORDER BY start_line, source_node_id")

    records: list[SemanticRecord] = []
    for row in symbol_rows:
        lines = [
            "type: symbol",
            f"path: {row['path']}",
            f"name: {row['name']}",
            f"qualified_name: {row['qualified_name']}",
            f"kind: {row['kind']}",
            f"signature: {row['signature']}",
        ]
        if row["parent_name"]:
            lines.append(f"parent: {row['parent_name']}")
        if row["return_type"]:
            lines.append(f"return_type: {row['return_type']}")
        if row["node_category"]:
            lines.append(f"file_category: {row['node_category']}")
        if row["node_summary"]:
            lines.append(f"file_summary: {row['node_summary']}")
        if row["docstring"]:
            lines.append(f"docstring: {row['docstring']}")
        parameters = _json_list(row["parameters_json"])
        if parameters:
            lines.append(f"parameters: {', '.join(parameters[:8])}")
        decorators = _json_list(row["decorators_json"])
        if decorators:
            lines.append(f"decorators: {', '.join(decorators[:6])}")
        base_classes = _json_list(row["base_classes_json"])
        if base_classes:
            lines.append(f"base_classes: {', '.join(base_classes[:6])}")
        callers = callers_map.get(row["symbol_id"], [])
        if callers:
            lines.append(f"called_by: {', '.join(callers[: config.max_callers])}")
        callees = callees_map.get(row["symbol_id"], [])
        if callees:
            lines.append(f"callees: {', '.join(callees[: config.max_callees])}")
        references = references_map.get(row["symbol_id"], [])
        if references:
            lines.append(f"references: {', '.join(references[: config.max_references])}")

        records.append(
            SemanticRecord(
                entity_id=row["symbol_id"],
                entity_type="symbol",
                title=row["qualified_name"],
                content="\n".join(lines),
                path=row["path"],
                kind=row["kind"],
            )
        )
    return records


def _group_single_values(conn: sqlite3.Connection, query: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for key, value in conn.execute(query).fetchall():
        if key is None or value is None:
            continue
        grouped.setdefault(str(key), []).append(str(value))
    return grouped


def _group_pairs(conn: sqlite3.Connection, query: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for key, source_path, summary in conn.execute(query).fetchall():
        if key is None or summary is None:
            continue
        label = f"{source_path}: {summary}" if source_path else str(summary)
        grouped.setdefault(str(key), []).append(label)
    return grouped


def _json_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [str(item) for item in json.loads(value)]


def _load_rollup_child_map(conn: sqlite3.Connection) -> dict[str, list[str]]:
    mapping = _group_single_values(conn, "SELECT parent_id, node_id FROM nodes WHERE parent_id IS NOT NULL ORDER BY kind, path")
    community_mapping = _group_single_values(conn, "SELECT community_node_id, member_node_id FROM community_members ORDER BY membership_rank, member_node_id")
    theme_mapping = _group_single_values(conn, "SELECT theme_node_id, member_node_id FROM theme_members ORDER BY membership_rank, member_node_id")
    for parent_id, member_ids in community_mapping.items():
        mapping.setdefault(parent_id, []).extend(member_ids)
        mapping[parent_id] = list(dict.fromkeys(mapping[parent_id]))
    for parent_id, member_ids in theme_mapping.items():
        mapping.setdefault(parent_id, []).extend(member_ids)
        mapping[parent_id] = list(dict.fromkeys(mapping[parent_id]))
    return mapping


def _build_node_centroid_records(
    node_records: list[SemanticRecord],
    node_vectors: np.ndarray,
    child_map: dict[str, list[str]],
    config: SemanticIndexConfig,
) -> tuple[list[NodeCentroidRecord], np.ndarray]:
    positions = {record.entity_id: index for index, record in enumerate(node_records)}
    records_by_id = {record.entity_id: record for record in node_records}
    centroid_records: list[NodeCentroidRecord] = []
    centroid_vectors: list[np.ndarray] = []

    for parent_id, child_ids in sorted(child_map.items()):
        valid_child_ids = [child_id for child_id in dict.fromkeys(child_ids) if child_id in positions]
        if len(valid_child_ids) < config.min_centroid_children:
            continue
        child_positions = [positions[child_id] for child_id in valid_child_ids]
        child_vectors = node_vectors[child_positions]
        centroid_count = min(config.max_centroids_per_parent, max(2, isqrt(len(valid_child_ids))))
        assignments, centroids = _run_kmeans(child_vectors, centroid_count, config.centroid_iterations)
        parent_title = records_by_id.get(parent_id).title if parent_id in records_by_id else parent_id
        for centroid_index in range(len(centroids)):
            member_indexes = [index for index, assignment in enumerate(assignments) if assignment == centroid_index]
            if not member_indexes:
                continue
            member_ids = [valid_child_ids[index] for index in member_indexes]
            member_vectors = child_vectors[member_indexes]
            representative_ids = _representative_member_ids(member_vectors, centroids[centroid_index], member_ids, limit=3)
            representative_titles = [records_by_id[member_id].title for member_id in representative_ids if member_id in records_by_id]
            preview_titles = [records_by_id[member_id].title for member_id in member_ids[: config.centroid_preview_members] if member_id in records_by_id]
            categories = _top_record_fields(member_ids, records_by_id, field_name="categories", limit=4)
            tags = _top_record_fields(member_ids, records_by_id, field_name="tags", limit=6)
            summary = _centroid_summary(parent_title, representative_ids, records_by_id, categories, tags)
            centroid_records.append(
                NodeCentroidRecord(
                    centroid_id=f"{parent_id}::centroid::{centroid_index + 1}",
                    parent_id=parent_id,
                    member_ids=member_ids,
                    representative_ids=representative_ids,
                    title=f"{parent_title} cluster {centroid_index + 1}",
                    content="\n".join(
                        [
                            f"parent: {parent_title}",
                            f"summary: {summary}",
                            f"member_count: {len(member_ids)}",
                            f"categories: {', '.join(categories)}",
                            f"tags: {', '.join(tags)}",
                            f"representatives: {', '.join(representative_titles)}",
                            f"members: {', '.join(preview_titles)}",
                        ]
                    ),
                    summary=summary,
                    categories=categories,
                    tags=tags,
                )
            )
            centroid_vectors.append(centroids[centroid_index])

    if not centroid_vectors:
        empty = np.zeros((0, node_vectors.shape[1]), dtype=np.float32)
        return centroid_records, empty
    return centroid_records, np.asarray(centroid_vectors, dtype=np.float32)


def _run_kmeans(vectors: np.ndarray, centroid_count: int, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    if centroid_count >= len(vectors):
        assignments = np.arange(len(vectors), dtype=np.int32)
        return assignments, vectors.copy()

    centroids = _initial_centroids(vectors, centroid_count)
    assignments = np.full(len(vectors), -1, dtype=np.int32)
    for _ in range(max(iterations, 1)):
        scores = vectors @ centroids.T
        next_assignments = np.argmax(scores, axis=1).astype(np.int32)
        if np.array_equal(assignments, next_assignments):
            assignments = next_assignments
            break
        assignments = next_assignments
        next_centroids = centroids.copy()
        for centroid_index in range(len(centroids)):
            mask = assignments == centroid_index
            if not np.any(mask):
                continue
            centroid = vectors[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm == 0:
                continue
            next_centroids[centroid_index] = centroid / norm
        centroids = next_centroids
    return assignments, centroids


def _initial_centroids(vectors: np.ndarray, centroid_count: int) -> np.ndarray:
    selected = [0]
    while len(selected) < centroid_count:
        best_index = None
        best_distance = None
        for index in range(len(vectors)):
            if index in selected:
                continue
            similarity = max(float(vectors[index] @ vectors[selected_index]) for selected_index in selected)
            distance = 1.0 - similarity
            if best_distance is None or distance > best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            break
        selected.append(best_index)
    return vectors[selected].copy()


def _representative_member_ids(member_vectors: np.ndarray, centroid: np.ndarray, member_ids: list[str], *, limit: int) -> list[str]:
    scores = member_vectors @ centroid
    ranking = np.argsort(-scores)[: min(limit, len(member_ids))]
    return [member_ids[index] for index in ranking]


def _top_record_fields(
    member_ids: list[str],
    records_by_id: dict[str, SemanticRecord],
    *,
    field_name: str,
    limit: int,
) -> list[str]:
    counter: defaultdict[str, int] = defaultdict(int)
    for member_id in member_ids:
        record = records_by_id.get(member_id)
        if record is None:
            continue
        values = _record_field_values(record.content, field_name)
        for value in values:
            counter[value] += 1
    return [name for name, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _centroid_summary(
    parent_title: str,
    representative_ids: list[str],
    records_by_id: dict[str, SemanticRecord],
    categories: list[str],
    tags: list[str],
) -> str:
    representative_titles = [records_by_id[member_id].title for member_id in representative_ids if member_id in records_by_id]
    representative_summaries = []
    for member_id in representative_ids:
        record = records_by_id.get(member_id)
        if record is None:
            continue
        summary_values = _record_field_values(record.content, "summary")
        if summary_values:
            representative_summaries.append(summary_values[0])
    parts: list[str] = [f"Semantic routing summary for {parent_title}"]
    if categories:
        parts.append(f"categories: {', '.join(categories[:3])}")
    if tags:
        parts.append(f"tags: {', '.join(tags[:4])}")
    if representative_titles:
        parts.append(f"representatives: {', '.join(representative_titles[:3])}")
    if representative_summaries:
        parts.append(f"focus: {' | '.join(representative_summaries[:2])}")
    return "; ".join(parts)


def _record_field_values(content: str, field_name: str) -> list[str]:
    prefix = f"{field_name}:"
    for line in content.splitlines():
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        if not value:
            return []
        if field_name in {"categories", "tags"}:
            return [item.strip() for item in value.split(",") if item.strip()]
        return [value]
    return []


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()