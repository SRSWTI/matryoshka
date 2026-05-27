from __future__ import annotations

import logging
import posixpath
from collections import defaultdict
from pathlib import Path

from cradle.graph_models import CallRecord, CodeNode, CodeSymbol, ImportRecord, NodeContextRecord, RepositoryGraph, SymbolReferenceRecord
from cradle.models import AnalyzedFile, LabelResult, NodePacket

logger = logging.getLogger(__name__)


class RepositoryGraphBuilder:
    def build(
        self,
        repo_root: Path,
        analyzed_files: dict[str, AnalyzedFile],
        file_labels: dict[str, LabelResult],
        folder_packets: dict[str, NodePacket],
        node_labels: dict[str, LabelResult],
        repo_packet: NodePacket,
        repo_label: LabelResult | None,
    ) -> RepositoryGraph:
        nodes = self._build_nodes(repo_root, analyzed_files, file_labels, folder_packets, node_labels, repo_packet, repo_label)
        symbols = self._build_symbols(analyzed_files)
        imports = self._build_imports(repo_root, analyzed_files, folder_packets)
        node_context = self._build_node_context(imports, file_labels, node_labels, repo_label)
        calls = self._build_calls(analyzed_files, symbols, imports)
        references = self._build_references(imports, calls, symbols)
        logger.info(
            "built repository graph: %s nodes, %s symbols, %s imports, %s calls, %s references",
            len(nodes),
            len(symbols),
            len(imports),
            len(calls),
            len(references),
        )
        return RepositoryGraph(
            repo_root=str(repo_root),
            nodes=nodes,
            symbols=symbols,
            imports=imports,
            calls=calls,
            references=references,
            node_context=node_context,
        )

    def _build_nodes(
        self,
        repo_root: Path,
        analyzed_files: dict[str, AnalyzedFile],
        file_labels: dict[str, LabelResult],
        folder_packets: dict[str, NodePacket],
        node_labels: dict[str, LabelResult],
        repo_packet: NodePacket,
        repo_label: LabelResult | None,
    ) -> list[CodeNode]:
        nodes: list[CodeNode] = []

        for relative_path in sorted(analyzed_files):
            analyzed = analyzed_files[relative_path]
            label = file_labels.get(relative_path)
            metadata = analyzed.packet.metadata
            start_line = 1 if analyzed.line_count else None
            end_line = analyzed.line_count if analyzed.line_count else None
            nodes.append(
                CodeNode(
                    node_id=relative_path,
                    path=relative_path,
                    name=Path(relative_path).name,
                    kind="file",
                    parent_id=_parent_node_id(relative_path),
                    language=analyzed.packet.language,
                    summary=label.summary if label is not None else analyzed.packet.summary_input,
                    description=label.description if label is not None else analyzed.packet.summary_input,
                    primary_category=label.primary_category if label is not None else None,
                    categories=list(label.categories) if label is not None else [],
                    tags=list(label.tags) if label is not None else [],
                    confidence=label.confidence if label is not None else 0.0,
                    start_line=start_line,
                    start_column=1 if start_line is not None else None,
                    end_line=end_line,
                    end_column=None,
                    symbol_count=int(metadata.get("symbol_count", 0)),
                    import_count=int(metadata.get("import_count", 0)),
                    content_hash=analyzed.content_hash,
                )
            )

        for node_id in sorted(folder_packets):
            packet = folder_packets[node_id]
            label = node_labels.get(node_id)
            metadata = packet.metadata
            nodes.append(
                CodeNode(
                    node_id=node_id,
                    path=packet.path,
                    name=Path(packet.path).name,
                    kind="folder",
                    parent_id=_parent_node_id(node_id),
                    summary=label.summary if label is not None else f"Folder {packet.path}",
                    description=label.description if label is not None else "",
                    primary_category=label.primary_category if label is not None else None,
                    categories=list(label.categories) if label is not None else [],
                    tags=list(label.tags) if label is not None else list(packet.top_tags),
                    confidence=label.confidence if label is not None else 0.0,
                    file_count=int(metadata.get("file_count", 0)),
                    folder_count=int(metadata.get("direct_folder_count", 0)),
                )
            )

        nodes.append(
            CodeNode(
                node_id=repo_packet.node_id,
                path=str(repo_root),
                name=repo_root.name,
                kind="repo",
                parent_id=None,
                summary=repo_label.summary if repo_label is not None else f"Repository {repo_root.name}",
                description=repo_label.description if repo_label is not None else "",
                primary_category=repo_label.primary_category if repo_label is not None else None,
                categories=list(repo_label.categories) if repo_label is not None else [],
                tags=list(repo_label.tags) if repo_label is not None else list(repo_packet.top_tags),
                confidence=repo_label.confidence if repo_label is not None else 0.0,
                file_count=int(repo_packet.metadata.get("file_count", 0)),
                folder_count=int(repo_packet.metadata.get("folder_count", 0)),
            )
        )
        return nodes

    def _build_symbols(self, analyzed_files: dict[str, AnalyzedFile]) -> list[CodeSymbol]:
        symbols: list[CodeSymbol] = []
        for relative_path in sorted(analyzed_files):
            extraction = analyzed_files[relative_path].extraction
            for symbol in extraction.symbols:
                qualified_name = f"{symbol.parent}.{symbol.name}" if symbol.parent else symbol.name
                symbol_id = build_symbol_id(relative_path, qualified_name, symbol.line_range.start_line)
                symbols.append(
                    CodeSymbol(
                        symbol_id=symbol_id,
                        node_id=relative_path,
                        path=relative_path,
                        name=symbol.name,
                        qualified_name=qualified_name,
                        normalized_name=qualified_name.lower(),
                        kind=symbol.kind,
                        signature=symbol.signature,
                        parent_name=symbol.parent,
                        return_type=symbol.return_type,
                        docstring=symbol.docstring,
                        parameters=list(symbol.parameters),
                        decorators=list(symbol.decorators),
                        base_classes=list(symbol.base_classes),
                        start_line=symbol.line_range.start_line,
                        start_column=symbol.line_range.start_column,
                        end_line=symbol.line_range.end_line,
                        end_column=symbol.line_range.end_column,
                    )
                )
        return symbols

    def _build_imports(
        self,
        repo_root: Path,
        analyzed_files: dict[str, AnalyzedFile],
        folder_packets: dict[str, NodePacket],
    ) -> list[ImportRecord]:
        folder_node_ids = set(folder_packets)
        file_node_ids = set(analyzed_files)
        imports: list[ImportRecord] = []
        for relative_path in sorted(analyzed_files):
            analyzed = analyzed_files[relative_path]
            language = analyzed.extraction.language
            for edge in analyzed.extraction.import_edges:
                target_node_id = None
                if edge.is_internal:
                    target_node_id = resolve_internal_import(
                        repo_root=repo_root,
                        importer_path=relative_path,
                        language=language,
                        imported_module=edge.imported_module,
                        file_node_ids=file_node_ids,
                        folder_node_ids=folder_node_ids,
                    )
                strength_label, strength_weight = classify_import_strength(relative_path, target_node_id, edge.is_internal)
                imports.append(
                    ImportRecord(
                        importer_node_id=relative_path,
                        imported_module=edge.imported_module,
                        target_node_id=target_node_id,
                        is_internal=edge.is_internal,
                        strength_label=strength_label,
                        strength_weight=strength_weight,
                        names=list(edge.names),
                        start_line=edge.line_range.start_line if edge.line_range is not None else None,
                        start_column=edge.line_range.start_column if edge.line_range is not None else None,
                        end_line=edge.line_range.end_line if edge.line_range is not None else None,
                        end_column=edge.line_range.end_column if edge.line_range is not None else None,
                    )
                )
        return imports

    def _build_node_context(
        self,
        imports: list[ImportRecord],
        file_labels: dict[str, LabelResult],
        node_labels: dict[str, LabelResult],
        repo_label: LabelResult | None,
    ) -> list[NodeContextRecord]:
        label_lookup: dict[str, LabelResult] = {**file_labels, **node_labels}
        if repo_label is not None:
            label_lookup[repo_label.target_id] = repo_label

        contexts: list[NodeContextRecord] = []
        seen: set[tuple[str, str]] = set()
        for record in imports:
            if not record.is_internal or record.target_node_id is None:
                continue
            key = (record.importer_node_id, record.target_node_id)
            if key in seen:
                continue
            seen.add(key)
            label = label_lookup.get(record.target_node_id)
            if label is None:
                continue
            contexts.append(
                NodeContextRecord(
                    node_id=record.importer_node_id,
                    source_node_id=record.target_node_id,
                    strength_label=record.strength_label,
                    strength_weight=record.strength_weight,
                    inherited_summary=label.summary,
                    inherited_category=label.primary_category,
                    inherited_tags=list(label.tags),
                )
            )
        return contexts

    def _build_calls(
        self,
        analyzed_files: dict[str, AnalyzedFile],
        symbols: list[CodeSymbol],
        imports: list[ImportRecord],
    ) -> list[CallRecord]:
        symbols_by_node_name: dict[tuple[str, str], list[CodeSymbol]] = defaultdict(list)
        symbols_by_name: dict[str, list[CodeSymbol]] = defaultdict(list)
        for symbol in symbols:
            symbols_by_node_name[(symbol.node_id, symbol.name)].append(symbol)
            symbols_by_name[symbol.name].append(symbol)

        imported_targets = _imported_symbol_targets(imports, symbols_by_node_name)
        calls: list[CallRecord] = []
        for relative_path in sorted(analyzed_files):
            extraction = analyzed_files[relative_path].extraction
            for call_site in extraction.call_sites:
                caller_candidates = symbols_by_node_name.get((relative_path, call_site.caller_name), [])
                if not caller_candidates:
                    continue
                caller_symbol = sorted(caller_candidates, key=lambda item: (item.start_line or 0, item.symbol_id))[0]
                target_symbol = resolve_call_target(relative_path, call_site.callee_name, symbols_by_node_name, imported_targets, symbols_by_name)
                calls.append(
                    CallRecord(
                        caller_symbol_id=caller_symbol.symbol_id,
                        caller_node_id=relative_path,
                        callee_name=call_site.callee_name,
                        target_symbol_id=target_symbol.symbol_id if target_symbol is not None else None,
                        target_node_id=target_symbol.node_id if target_symbol is not None else None,
                        start_line=call_site.line_range.start_line,
                        start_column=call_site.line_range.start_column,
                        end_line=call_site.line_range.end_line,
                        end_column=call_site.line_range.end_column,
                    )
                )
        return calls

    def _build_references(
        self,
        imports: list[ImportRecord],
        calls: list[CallRecord],
        symbols: list[CodeSymbol],
    ) -> list[SymbolReferenceRecord]:
        references: list[SymbolReferenceRecord] = []
        symbols_by_node_name: dict[tuple[str, str], list[CodeSymbol]] = defaultdict(list)
        symbols_by_id = {symbol.symbol_id: symbol for symbol in symbols}
        for symbol in symbols:
            symbols_by_node_name[(symbol.node_id, symbol.name)].append(symbol)

        for call in calls:
            references.append(
                SymbolReferenceRecord(
                    target_symbol_id=call.target_symbol_id,
                    target_node_id=call.target_node_id,
                    target_name=call.callee_name,
                    source_node_id=call.caller_node_id,
                    source_symbol_id=call.caller_symbol_id,
                    reference_kind="call",
                    start_line=call.start_line,
                    start_column=call.start_column,
                    end_line=call.end_line,
                    end_column=call.end_column,
                )
            )

        for record in imports:
            if not record.is_internal or record.target_node_id is None:
                continue
            names = record.names or [record.imported_module]
            for imported_name in names:
                clean_name = normalize_imported_name(imported_name)
                target_symbol = None
                candidates = symbols_by_node_name.get((record.target_node_id, clean_name), [])
                if len(candidates) == 1:
                    target_symbol = candidates[0]
                references.append(
                    SymbolReferenceRecord(
                        target_symbol_id=target_symbol.symbol_id if target_symbol is not None else None,
                        target_node_id=record.target_node_id,
                        target_name=clean_name,
                        source_node_id=record.importer_node_id,
                        source_symbol_id=None,
                        reference_kind="import",
                        start_line=record.start_line,
                        start_column=record.start_column,
                        end_line=record.end_line,
                        end_column=record.end_column,
                    )
                )

        unresolved_targets = sum(1 for reference in references if reference.target_symbol_id is None and reference.target_node_id is None)
        if unresolved_targets:
            logger.debug("recorded %s unresolved references", unresolved_targets)
        else:
            logger.debug("recorded %s symbol references", len(references))
        return references


def build_symbol_id(path: str, qualified_name: str, start_line: int | None) -> str:
    line_suffix = start_line if start_line is not None else 0
    return f"{path}::{qualified_name}::L{line_suffix}"


def classify_import_strength(importer_path: str, target_node_id: str | None, is_internal: bool) -> tuple[str, float]:
    if not is_internal or target_node_id is None:
        return "weak", 0.2

    importer_parent = _parent_node_id(importer_path)
    target_parent = _parent_node_id(target_node_id)
    if importer_parent == target_parent:
        return "strong", 0.8
    return "medium", 0.5


def resolve_call_target(
    caller_node_id: str,
    callee_name: str,
    symbols_by_node_name: dict[tuple[str, str], list[CodeSymbol]],
    imported_targets: dict[str, dict[str, list[CodeSymbol]]],
    symbols_by_name: dict[str, list[CodeSymbol]],
) -> CodeSymbol | None:
    local_candidates = symbols_by_node_name.get((caller_node_id, callee_name), [])
    if len(local_candidates) == 1:
        return local_candidates[0]

    imported_candidates = imported_targets.get(caller_node_id, {}).get(callee_name, [])
    if len(imported_candidates) == 1:
        return imported_candidates[0]

    global_candidates = symbols_by_name.get(callee_name, [])
    if len(global_candidates) == 1:
        return global_candidates[0]
    return None


def resolve_internal_import(
    *,
    repo_root: Path,
    importer_path: str,
    language: str,
    imported_module: str,
    file_node_ids: set[str],
    folder_node_ids: set[str],
) -> str | None:
    candidates: list[str] = []
    if language == "python":
        parts = imported_module.split(".")
        module_path = Path(*parts).as_posix()
        candidates.extend([
            f"{module_path}.py",
            Path(module_path, "__init__.py").as_posix(),
            module_path,
        ])
    else:
        if imported_module.startswith("@/"):
            base = imported_module[2:]
        elif imported_module.startswith("/"):
            base = imported_module.lstrip("/")
        else:
            base = posixpath.normpath(posixpath.join(posixpath.dirname(importer_path), imported_module))
        base = base.rstrip("/")
        candidates.extend(
            [
                base,
                f"{base}.ts",
                f"{base}.tsx",
                f"{base}.d.ts",
                Path(base, "index.ts").as_posix(),
                Path(base, "index.tsx").as_posix(),
                Path(base, "index.d.ts").as_posix(),
            ]
        )

    for candidate in candidates:
        normalized = posixpath.normpath(candidate)
        if normalized in file_node_ids:
            return normalized
        if normalized in folder_node_ids:
            return normalized
        candidate_path = repo_root / normalized
        if candidate_path.is_dir() and normalized in folder_node_ids:
            return normalized
    return None


def normalize_imported_name(value: str) -> str:
    candidate = value.strip()
    if " as " in candidate:
        candidate = candidate.split(" as ", 1)[0].strip()
    return candidate.strip("{} ")


def _imported_symbol_targets(
    imports: list[ImportRecord],
    symbols_by_node_name: dict[tuple[str, str], list[CodeSymbol]],
) -> dict[str, dict[str, list[CodeSymbol]]]:
    imported_targets: dict[str, dict[str, list[CodeSymbol]]] = defaultdict(lambda: defaultdict(list))
    for record in imports:
        if not record.is_internal or record.target_node_id is None:
            continue
        names = record.names or [record.imported_module]
        if names == ["*"]:
            for (node_id, symbol_name), candidates in symbols_by_node_name.items():
                if node_id != record.target_node_id:
                    continue
                imported_targets[record.importer_node_id][symbol_name].extend(candidates)
            continue
        for imported_name in names:
            clean_name = normalize_imported_name(imported_name)
            imported_targets[record.importer_node_id][clean_name].extend(symbols_by_node_name.get((record.target_node_id, clean_name), []))
    return imported_targets


def _parent_node_id(node_id: str) -> str | None:
    if node_id == "repo":
        return None
    parent = Path(node_id).parent.as_posix()
    return "repo" if parent == "." else parent