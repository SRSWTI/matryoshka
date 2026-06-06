from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fnmatch import fnmatch
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterable

from pathspec import GitIgnoreSpec

from matryoshka.ast_extractor import FileExtraction, SymbolRecord, extract_file
from matryoshka.community_detection import build_louvain_communities, build_theme_domains
from matryoshka.graph_builder import RepositoryGraphBuilder
from matryoshka.graph_models import RepositoryGraph
from matryoshka.labeling import LabelingEngine
from matryoshka.models import AnalyzedFile, FilePacket, LabelResult, NodePacket


@dataclass(slots=True)
class PipelineConfig:
    include_suffixes: tuple[str, ...] = (".py", ".ts", ".tsx")
    excluded_suffixes: tuple[str, ...] = ()
    excluded_paths: tuple[str, ...] = ()
    honor_gitignore: bool = True
    ignored_directories: tuple[str, ...] = (
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        "test",
        "tests",
        "__pycache__",
        ".pytest_cache",
    )
    max_symbols_per_file: int = 5
    max_docstrings_per_file: int = 3
    max_call_hints_per_file: int = 8
    max_snippets_per_file: int = 3
    snippet_line_limit: int = 12
    max_node_child_summaries: int = 8
    max_node_tags: int = 8
    max_node_packages: int = 8
    max_node_symbols: int = 8
    max_node_files: int = 6
    max_node_snippets: int = 4
    max_files: int | None = None


class RepositoryWalker:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def collect_source_files(self, repo_root: str | Path) -> list[Path]:
        root = Path(repo_root)
        gitignore_spec = _load_gitignore_spec(root) if self._config.honor_gitignore else None
        files: list[Path] = []
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            relative_path = path.relative_to(root).as_posix()
            if path.suffix.lower() not in self._config.include_suffixes:
                continue
            if path.suffix.lower() in self._config.excluded_suffixes:
                continue
            if any(part in self._config.ignored_directories for part in path.parts):
                continue
            if gitignore_spec is not None and gitignore_spec.match_file(relative_path):
                continue
            if _is_excluded_path(path, root, self._config.excluded_paths):
                continue
            files.append(path)
        ordered = sorted(files)
        if self._config.max_files is not None:
            return ordered[: self._config.max_files]
        return ordered


def _load_gitignore_spec(repo_root: Path) -> GitIgnoreSpec | None:
    gitignore_path = repo_root / ".gitignore"
    if not gitignore_path.exists():
        return None
    lines = gitignore_path.read_text(encoding="utf-8").splitlines()
    return GitIgnoreSpec.from_lines(lines)


def _is_excluded_path(path: Path, repo_root: Path, excluded_paths: tuple[str, ...]) -> bool:
    if not excluded_paths:
        return False
    relative_path = path.relative_to(repo_root).as_posix()
    parts = Path(relative_path).parts
    for raw_pattern in excluded_paths:
        pattern = raw_pattern.strip().strip("/")
        if not pattern:
            continue
        if any(token in pattern for token in "*?[]"):
            if fnmatch(relative_path, pattern) or fnmatch(path.name, pattern):
                return True
            continue
        normalized = Path(pattern).as_posix().strip("/")
        if relative_path == normalized or relative_path.startswith(f"{normalized}/"):
            return True
        if normalized in parts:
            return True
    return False


class FilePacketBuilder:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    def build(self, repo_root: str | Path, file_path: str | Path) -> AnalyzedFile:
        root = Path(repo_root)
        path = Path(file_path)
        extraction = extract_file(path, repo_root=root)
        source_text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(root).as_posix()
        ranked_symbols = _rank_symbols(extraction.symbols)
        top_symbols = [symbol.signature for symbol in ranked_symbols[: self._config.max_symbols_per_file]]
        docstrings = [symbol.docstring for symbol in ranked_symbols if symbol.docstring][: self._config.max_docstrings_per_file]
        call_hints = _dedupe(
            hint
            for symbol in ranked_symbols
            for hint in [*symbol.callees, *symbol.callers]
        )[: self._config.max_call_hints_per_file]
        code_snippets = _select_snippets(
            source_text,
            ranked_symbols[: self._config.max_snippets_per_file],
            self._config.snippet_line_limit,
        )
        imports_external = sorted({edge.imported_module for edge in extraction.import_edges if not edge.is_internal})
        imports_internal = sorted({edge.imported_module for edge in extraction.import_edges if edge.is_internal})
        imported_symbols = _dedupe(name for edge in extraction.import_edges for name in edge.names)
        import_signature = Counter(_external_signature_key(edge.imported_module) for edge in extraction.import_edges if not edge.is_internal)
        internal_signature = Counter(edge.imported_module for edge in extraction.import_edges if edge.is_internal)

        packet = FilePacket(
            path=relative_path,
            language=extraction.language,
            summary_input=_build_summary_input(relative_path, extraction, top_symbols, imports_external, imports_internal),
            imports_external=imports_external,
            imports_internal=imports_internal,
            imported_symbols=imported_symbols,
            top_symbols=top_symbols,
            docstrings=docstrings,
            call_hints=call_hints,
            code_snippets=code_snippets,
            import_signature=dict(import_signature),
            internal_signature=dict(internal_signature),
            metadata={
                "symbol_count": len(extraction.symbols),
                "import_count": len(extraction.import_edges),
                "external_packages": extraction.external_packages,
                "absolute_path": str(path),
            },
        )
        line_count = len(source_text.splitlines())
        return AnalyzedFile(
            packet=packet,
            extraction=extraction,
            absolute_path=str(path),
            content_hash=sha256(source_text.encode("utf-8")).hexdigest(),
            line_count=line_count,
        )


class MatryoshkaPipeline:
    def __init__(
        self,
        *,
        config: PipelineConfig | None = None,
        walker: RepositoryWalker | None = None,
        file_packet_builder: FilePacketBuilder | None = None,
        labeling_engine: LabelingEngine | None = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._walker = walker or RepositoryWalker(self._config)
        self._file_packet_builder = file_packet_builder or FilePacketBuilder(self._config)
        self._labeling_engine = labeling_engine
        self._graph_builder = RepositoryGraphBuilder()

    def analyze(self, repo_root: str | Path, progress: Callable[[str], None] | None = None) -> RepositoryGraph:
        root = Path(repo_root)
        files = self._walker.collect_source_files(root)
        _emit_progress(progress, f"collected {len(files)} source files")
        analyzed_files = {analysis.packet.path: analysis for analysis in (self._file_packet_builder.build(root, path) for path in files)}
        file_packets = {path: analyzed.packet for path, analyzed in analyzed_files.items()}
        _emit_progress(progress, f"built {len(file_packets)} file packets")

        file_labels: dict[str, LabelResult] = {}
        if self._labeling_engine is not None and file_packets:
            file_labels = self._labeling_engine.label_files(list(file_packets.values()), progress=progress)
            _emit_progress(progress, f"labeled {len(file_labels)} files")

        folder_packets, node_labels = self._build_folder_packets(root, file_packets, file_labels, progress=progress)
        _emit_progress(progress, f"built {len(folder_packets)} folder nodes")

        repo_packet = self._build_repo_packet(root, file_packets, file_labels, folder_packets, node_labels)
        repo_label: LabelResult | None = None
        if self._labeling_engine is not None:
            repo_label = self._labeling_engine.label_node(repo_packet)
            _emit_progress(progress, "labeled repo node")

        graph = self._graph_builder.build(
            root,
            analyzed_files,
            file_labels,
            folder_packets,
            node_labels,
            repo_packet,
            repo_label,
        )
        community_nodes, community_members = build_louvain_communities(graph.nodes, graph.imports, graph.calls)
        if community_nodes:
            graph.nodes.extend(community_nodes)
            graph.community_members.extend(community_members)
        theme_nodes, theme_members = build_theme_domains(graph.nodes)
        if theme_nodes:
            graph.nodes.extend(theme_nodes)
            graph.theme_members.extend(theme_members)
        _emit_progress(progress, f"built graph with {len(graph.nodes)} nodes and {len(graph.symbols)} symbols")
        return graph

    def _build_folder_packets(
        self,
        repo_root: Path,
        file_packets: dict[str, FilePacket],
        file_labels: dict[str, LabelResult],
        progress: Callable[[str], None] | None = None,
    ) -> tuple[dict[str, NodePacket], dict[str, LabelResult]]:
        directory_to_files = _directory_file_map(file_packets)
        folder_packets: dict[str, NodePacket] = {}
        folder_labels: dict[str, LabelResult] = {}
        directories_by_depth: dict[int, list[str]] = defaultdict(list)
        for directory in directory_to_files:
            directories_by_depth[_directory_depth(directory)].append(directory)

        for depth in sorted(directories_by_depth, reverse=True):
            level_packets: list[NodePacket] = []
            for directory in sorted(directories_by_depth[depth]):
                packet = _build_folder_packet(
                    directory=directory,
                    repo_root=repo_root,
                    file_packets=file_packets,
                    file_labels=file_labels,
                    folder_packets=folder_packets,
                    folder_labels=folder_labels,
                    config=self._config,
                )
                folder_packets[packet.node_id] = packet
                level_packets.append(packet)
            if self._labeling_engine is not None and level_packets:
                folder_labels.update(self._labeling_engine.label_nodes(level_packets, progress=progress))

        return folder_packets, folder_labels

    def _build_repo_packet(
        self,
        repo_root: Path,
        file_packets: dict[str, FilePacket],
        file_labels: dict[str, LabelResult],
        folder_packets: dict[str, NodePacket],
        node_labels: dict[str, LabelResult],
    ) -> NodePacket:
        top_level_dirs = sorted(node_id for node_id in folder_packets if "/" not in node_id)
        root_files = sorted(path for path in file_packets if "/" not in path)
        child_ids = [*root_files, *top_level_dirs]
        child_summaries = [
            *[_child_summary_for_file(path, file_packets, file_labels) for path in root_files],
            *[_child_summary_for_folder(node_id, folder_packets, node_labels) for node_id in top_level_dirs],
        ][: self._config.max_node_child_summaries]

        file_iterable = list(file_packets.values())
        top_tags = _top_counter_keys(Counter(tag for label in [*file_labels.values(), *node_labels.values()] for tag in label.tags), self._config.max_node_tags)
        top_external = _top_counter_keys(Counter(name for packet in file_iterable for name in packet.imports_external), self._config.max_node_packages)
        top_internal = _top_counter_keys(Counter(name for packet in file_iterable for name in packet.imports_internal), self._config.max_node_packages)
        representative_symbols = _top_counter_keys(Counter(symbol for packet in file_iterable for symbol in packet.top_symbols), self._config.max_node_symbols)
        representative_files = _representative_files(file_iterable, self._config.max_node_files)
        representative_snippets = _representative_snippets(file_iterable, self._config.max_node_snippets)

        return NodePacket(
            node_id="repo",
            path=str(repo_root),
            level="repo",
            child_ids=child_ids,
            child_summaries=child_summaries,
            top_tags=top_tags,
            top_external_packages=top_external,
            top_internal_modules=top_internal,
            representative_symbols=representative_symbols,
            representative_files=representative_files,
            representative_snippets=representative_snippets,
            metadata={"repo_name": repo_root.name, "file_count": len(file_iterable), "folder_count": len(folder_packets)},
        )


def _build_folder_packet(
    *,
    directory: str,
    repo_root: Path,
    file_packets: dict[str, FilePacket],
    file_labels: dict[str, LabelResult],
    folder_packets: dict[str, NodePacket],
    folder_labels: dict[str, LabelResult],
    config: PipelineConfig,
) -> NodePacket:
    child_files = sorted(path for path in file_packets if _parent_directory(path) == directory)
    child_folders = sorted(node_id for node_id in folder_packets if _parent_directory(node_id) == directory)
    descendant_files = [packet for path, packet in file_packets.items() if _is_descendant_or_self(directory, _parent_directory(path))]
    child_ids = [*child_files, *child_folders]
    child_summaries = [
        *[_child_summary_for_file(path, file_packets, file_labels) for path in child_files],
        *[_child_summary_for_folder(node_id, folder_packets, folder_labels) for node_id in child_folders],
    ][: config.max_node_child_summaries]
    top_tags = _top_counter_keys(
        Counter(
            tag
            for label in [*[file_labels[path] for path in child_files if path in file_labels], *[folder_labels[node_id] for node_id in child_folders if node_id in folder_labels]]
            for tag in label.tags
        ),
        config.max_node_tags,
    )
    top_external = _top_counter_keys(Counter(name for packet in descendant_files for name in packet.imports_external), config.max_node_packages)
    top_internal = _top_counter_keys(Counter(name for packet in descendant_files for name in packet.imports_internal), config.max_node_packages)
    representative_symbols = _top_counter_keys(Counter(symbol for packet in descendant_files for symbol in packet.top_symbols), config.max_node_symbols)
    representative_files = _representative_files(descendant_files, config.max_node_files)
    representative_snippets = _representative_snippets(descendant_files, config.max_node_snippets)

    return NodePacket(
        node_id=directory,
        path=directory,
        level="folder",
        child_ids=child_ids,
        child_summaries=child_summaries,
        top_tags=top_tags,
        top_external_packages=top_external,
        top_internal_modules=top_internal,
        representative_symbols=representative_symbols,
        representative_files=representative_files,
        representative_snippets=representative_snippets,
        metadata={
            "absolute_path": str(repo_root / directory),
            "file_count": len(descendant_files),
            "direct_file_count": len(child_files),
            "direct_folder_count": len(child_folders),
        },
    )


def _build_summary_input(
    relative_path: str,
    extraction: FileExtraction,
    top_symbols: list[str],
    imports_external: list[str],
    imports_internal: list[str],
) -> str:
    summary_lines = [
        f"path: {relative_path}",
        f"language: {extraction.language}",
        f"top_symbols: {', '.join(top_symbols) if top_symbols else 'none'}",
        f"external_imports: {', '.join(imports_external) if imports_external else 'none'}",
        f"internal_imports: {', '.join(imports_internal) if imports_internal else 'none'}",
    ]
    return "\n".join(summary_lines)


def _rank_symbols(symbols: Iterable[SymbolRecord]) -> list[SymbolRecord]:
    kind_weight = {
        "function": 4,
        "method": 4,
        "class": 3,
        "interface": 3,
        "enum": 2,
        "type_alias": 2,
        "constant": 1,
        "variable": 1,
        "field": 1,
    }
    return sorted(
        symbols,
        key=lambda symbol: (
            kind_weight.get(symbol.kind, 0),
            len(symbol.callees) + len(symbol.callers),
            len(symbol.parameters),
            1 if symbol.docstring else 0,
            symbol.name,
        ),
        reverse=True,
    )


def _select_snippets(source_text: str, symbols: list[SymbolRecord], line_limit: int) -> list[str]:
    lines = source_text.splitlines()
    snippets: list[str] = []
    for symbol in symbols:
        start = max(symbol.line_range.start_line - 1, 0)
        end = min(start + line_limit, len(lines))
        snippet = "\n".join(lines[start:end]).strip()
        if snippet:
            snippets.append(snippet)
    return snippets


def _external_signature_key(module_name: str) -> str:
    if "/" in module_name:
        return module_name.split("/", 1)[0]
    return module_name.split(".", 1)[0]


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _directory_file_map(file_packets: dict[str, FilePacket]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for path in file_packets:
        directory = _parent_directory(path)
        if directory:
            mapping[directory].append(path)
            for ancestor in _ancestor_directories(directory):
                mapping.setdefault(ancestor, [])
    return mapping


def _ancestor_directories(directory: str) -> list[str]:
    parts = directory.split("/")
    return ["/".join(parts[:index]) for index in range(1, len(parts))]


def _directory_depth(directory: str) -> int:
    return directory.count("/") + 1 if directory else 0


def _parent_directory(path: str) -> str:
    parent = Path(path).parent.as_posix()
    return "" if parent == "." else parent


def _is_descendant_or_self(directory: str, candidate: str) -> bool:
    return candidate == directory or candidate.startswith(f"{directory}/")


def _sort_folder_node_ids(nodes: dict[str, NodePacket]) -> list[str]:
    return sorted(nodes, key=lambda node_id: (_directory_depth(node_id), node_id))


def _child_summary_for_file(path: str, file_packets: dict[str, FilePacket], file_labels: dict[str, LabelResult]) -> str:
    if path in file_labels and file_labels[path].summary:
        return file_labels[path].summary
    return file_packets[path].summary_input


def _child_summary_for_folder(node_id: str, folder_packets: dict[str, NodePacket], folder_labels: dict[str, LabelResult]) -> str:
    if node_id in folder_labels and folder_labels[node_id].summary:
        return folder_labels[node_id].summary
    return f"folder {folder_packets[node_id].path}"


def _top_counter_keys(counter: Counter[str], limit: int) -> list[str]:
    return [key for key, _ in counter.most_common(limit)]


def _representative_files(file_packets: list[FilePacket], limit: int) -> list[str]:
    scored = sorted(file_packets, key=lambda packet: (len(packet.top_symbols), len(packet.imports_external) + len(packet.imports_internal), packet.path), reverse=True)
    return [packet.path for packet in scored[:limit]]


def _representative_snippets(file_packets: list[FilePacket], limit: int) -> list[str]:
    snippets: list[str] = []
    for packet in _sorted_packets_for_snippets(file_packets):
        for snippet in packet.code_snippets:
            if len(snippets) >= limit:
                return snippets
            snippets.append(snippet)
    return snippets


def _sorted_packets_for_snippets(file_packets: list[FilePacket]) -> list[FilePacket]:
    return sorted(file_packets, key=lambda packet: (len(packet.code_snippets), len(packet.top_symbols), packet.path), reverse=True)


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)