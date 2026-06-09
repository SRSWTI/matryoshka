from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from matryoshka.cache import LabelCache
from matryoshka.dashboard import run_dashboard
from matryoshka.db_visualization import build_db_visualization
from matryoshka.embeddings import DEFAULT_EMBEDDING_MODEL, build_text_embedder
from matryoshka.exact_search import (
    axe_call_search,
    axe_file_search,
    axe_import_search,
    axe_module_search,
    axe_reference_search,
    axe_symbol_search,
)
from matryoshka.file_reader import FileReader
from matryoshka.focus_visualization import build_focus_visualization
from matryoshka.hierarchical_search import axe_hierarchy_search
from matryoshka.labeling import LabelingConfig, LabelingEngine
from matryoshka.llm_client import LLMClientConfig, OpenAICompatibleClient
from matryoshka.pipeline import MatryoshkaPipeline, PipelineConfig
from matryoshka.question_answering import axe_question
from matryoshka.retrieval import axe_retrieval
from matryoshka.semantic_index import (
    SemanticIndexBuilder,
    SemanticIndexConfig,
    load_semantic_manifest,
)
from matryoshka.semantic_search import axe_semantic_search
from matryoshka.storage import MatryoshkaDatabase

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(prog="matryoshka")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("repo_root")
    analyze_parser.add_argument("--base-url", default="http://127.0.0.1:44445")
    analyze_parser.add_argument("--model", required=True)
    analyze_parser.add_argument("--api-key", default=None)
    analyze_parser.add_argument("--cache-path", default=None)
    analyze_parser.add_argument("--output", default=None)
    analyze_parser.add_argument("--log-level", default="INFO")
    analyze_parser.add_argument("--max-parallel-requests", type=int, default=8)
    analyze_parser.add_argument("--max-tokens", type=int, default=600)
    analyze_parser.add_argument("--temperature", type=float, default=0.0)
    analyze_parser.add_argument("--thinking-budget", type=int, default=0)
    analyze_parser.add_argument("--max-files", type=int, default=None)
    analyze_parser.add_argument("--exclude-path", action="append", default=None)
    analyze_parser.add_argument("--exclude-extension", action="append", default=None)

    retrieve_parser = subparsers.add_parser("retrieve")
    retrieve_parser.add_argument("db_path")
    retrieve_parser.add_argument("query")
    retrieve_parser.add_argument("--limit", type=int, default=5)
    retrieve_parser.add_argument("--log-level", default="INFO")

    file_search_parser = subparsers.add_parser("file-search")
    file_search_parser.add_argument("db_path")
    file_search_parser.add_argument("query")
    file_search_parser.add_argument("--limit", type=int, default=5)
    file_search_parser.add_argument("--log-level", default="INFO")

    symbol_search_parser = subparsers.add_parser("symbol-search")
    symbol_search_parser.add_argument("db_path")
    symbol_search_parser.add_argument("query")
    symbol_search_parser.add_argument("--limit", type=int, default=5)
    symbol_search_parser.add_argument("--log-level", default="INFO")

    import_search_parser = subparsers.add_parser("import-search")
    import_search_parser.add_argument("db_path")
    import_search_parser.add_argument("query")
    import_search_parser.add_argument("--limit", type=int, default=5)
    import_search_parser.add_argument("--log-level", default="INFO")

    module_search_parser = subparsers.add_parser("module-search")
    module_search_parser.add_argument("db_path")
    module_search_parser.add_argument("query")
    module_search_parser.add_argument("--limit", type=int, default=5)
    module_search_parser.add_argument("--log-level", default="INFO")

    call_search_parser = subparsers.add_parser("call-search")
    call_search_parser.add_argument("db_path")
    call_search_parser.add_argument("query")
    call_search_parser.add_argument("--limit", type=int, default=5)
    call_search_parser.add_argument("--log-level", default="INFO")

    reference_search_parser = subparsers.add_parser("reference-search")
    reference_search_parser.add_argument("db_path")
    reference_search_parser.add_argument("query")
    reference_search_parser.add_argument("--limit", type=int, default=5)
    reference_search_parser.add_argument("--log-level", default="INFO")

    focus_parser = subparsers.add_parser("visualize-focus")
    focus_parser.add_argument("db_path")
    focus_parser.add_argument("query")
    focus_parser.add_argument(
        "--kind", default="auto", choices=["auto", "file", "symbol"]
    )
    focus_parser.add_argument("--limit", type=int, default=8)
    focus_parser.add_argument("--output", default=None)
    focus_parser.add_argument("--log-level", default="INFO")

    visualize_parser = subparsers.add_parser("visualize-db")
    visualize_parser.add_argument("db_path")
    visualize_parser.add_argument("--output", default=None)
    visualize_parser.add_argument("--sample-limit", type=int, default=10)
    visualize_parser.add_argument("--log-level", default="INFO")

    semantic_index_parser = subparsers.add_parser("semantic-index")
    semantic_index_parser.add_argument("db_path")
    semantic_index_parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    semantic_index_parser.add_argument("--output-dir", default=None)
    semantic_index_parser.add_argument("--batch-size", type=int, default=32)
    semantic_index_parser.add_argument("--truncate-dim", type=int, default=None)
    semantic_index_parser.add_argument(
        "--backend", default="auto", choices=["auto", "mlx", "sentence-transformers"]
    )
    semantic_index_parser.add_argument("--log-level", default="INFO")

    semantic_search_parser = subparsers.add_parser("semantic-search")
    semantic_search_parser.add_argument("db_path")
    semantic_search_parser.add_argument("query")
    semantic_search_parser.add_argument("--index-dir", default=None)
    semantic_search_parser.add_argument("--limit", type=int, default=5)
    semantic_search_parser.add_argument("--search-k", type=int, default=None)
    semantic_search_parser.add_argument("--task", default=None)
    semantic_search_parser.add_argument(
        "--backend", default="auto", choices=["auto", "mlx", "sentence-transformers"]
    )
    semantic_search_parser.add_argument("--log-level", default="INFO")

    hierarchy_search_parser = subparsers.add_parser("hierarchy-search")
    hierarchy_search_parser.add_argument("db_path")
    hierarchy_search_parser.add_argument("query")
    hierarchy_search_parser.add_argument("--index-dir", default=None)
    hierarchy_search_parser.add_argument("--limit", type=int, default=5)
    hierarchy_search_parser.add_argument("--branch-width", type=int, default=3)
    hierarchy_search_parser.add_argument("--symbol-limit", type=int, default=5)
    hierarchy_search_parser.add_argument(
        "--backend", default="auto", choices=["auto", "mlx", "sentence-transformers"]
    )
    hierarchy_search_parser.add_argument("--log-level", default="INFO")

    question_parser = subparsers.add_parser("question")
    question_parser.add_argument("db_path")
    question_parser.add_argument("query")
    question_parser.add_argument("--index-dir", default=None)
    question_parser.add_argument(
        "--backend", default="auto", choices=["auto", "mlx", "sentence-transformers"]
    )
    question_parser.add_argument("--log-level", default="INFO")

    serve_parser = subparsers.add_parser(
        "serve", help="Launch the Matryoshka web dashboard"
    )
    serve_parser.add_argument("db_path")
    serve_parser.add_argument(
        "--index-dir",
        default=None,
        help="Semantic sidecar directory (default: auto-detect)",
    )
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab automatically",
    )
    serve_parser.add_argument("--log-level", default="INFO")

    read_parser = subparsers.add_parser(
        "read", help="Read rich file summary from the analysis DB"
    )
    read_parser.add_argument("db_path")
    read_parser.add_argument("file_path")
    read_parser.add_argument("--log-level", default="INFO")

    read_more_parser = subparsers.add_parser(
        "read-more", help="Read rich file detail with collapsed source blocks"
    )
    read_more_parser.add_argument("db_path")
    read_more_parser.add_argument("file_path")
    read_more_parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    if args.command == "analyze":
        return _run_analyze(args)
    if args.command == "retrieve":
        return _run_retrieve(args)
    if args.command == "file-search":
        return _run_exact_search(args, axe_file_search)
    if args.command == "symbol-search":
        return _run_exact_search(args, axe_symbol_search)
    if args.command == "import-search":
        return _run_exact_search(args, axe_import_search)
    if args.command == "module-search":
        return _run_exact_search(args, axe_module_search)
    if args.command == "call-search":
        return _run_exact_search(args, axe_call_search)
    if args.command == "reference-search":
        return _run_exact_search(args, axe_reference_search)
    if args.command == "visualize-focus":
        return _run_visualize_focus(args)
    if args.command == "visualize-db":
        return _run_visualize_db(args)
    if args.command == "semantic-index":
        return _run_semantic_index(args)
    if args.command == "semantic-search":
        return _run_semantic_search(args)
    if args.command == "hierarchy-search":
        return _run_hierarchy_search(args)
    if args.command == "question":
        return _run_question(args)
    if args.command == "serve":
        return _run_serve(args)
    if args.command == "read":
        return _run_read(args)
    if args.command == "read-more":
        return _run_read_more(args)
    return 1


def _run_analyze(args: argparse.Namespace) -> int:
    _configure_logging(args.log_level)
    repo_root = Path(args.repo_root)
    output_path = (
        Path(args.output) if args.output else _default_analysis_db_path(repo_root)
    )
    cache_path = Path(args.cache_path) if args.cache_path else output_path
    cache = LabelCache(cache_path)
    client = OpenAICompatibleClient(
        LLMClientConfig(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            max_parallel_requests=args.max_parallel_requests,
            extra_body={"thinking_budget": args.thinking_budget},
        )
    )
    engine = LabelingEngine(
        client,
        LabelingConfig(temperature=args.temperature, max_tokens=args.max_tokens),
        cache=cache,
    )
    pipeline = MatryoshkaPipeline(
        config=PipelineConfig(
            max_files=args.max_files,
            excluded_paths=tuple(args.exclude_path or ()),
            excluded_suffixes=_normalize_excluded_suffixes(
                args.exclude_extension or ()
            ),
        ),
        labeling_engine=engine,
    )
    graph = pipeline.analyze(
        repo_root, progress=lambda message: print(f"progress: {message}", flush=True)
    )
    database = MatryoshkaDatabase(output_path)
    summary = database.replace_graph(graph)
    engine.flush_cache()

    logger.info("analysis completed for %s", repo_root)
    print(_summarize_result(summary, output_path))
    return 0


def _run_retrieve(args: argparse.Namespace) -> int:
    _configure_logging(args.log_level)
    result = axe_retrieval(args.db_path, args.query, limit=args.limit)
    print(_format_retrieval_result(result))
    return 0


def _run_exact_search(args: argparse.Namespace, search_fn) -> int:
    _configure_logging(args.log_level)
    result = search_fn(args.db_path, args.query, limit=args.limit)
    print(_format_exact_search_result(result))
    return 0


def _run_visualize_db(args: argparse.Namespace) -> int:
    _configure_logging(args.log_level)
    report = build_db_visualization(args.db_path, sample_limit=args.sample_limit)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"visualization: {output_path}")
    else:
        print(report)
    return 0


def _run_visualize_focus(args: argparse.Namespace) -> int:
    _configure_logging(args.log_level)
    report = build_focus_visualization(
        args.db_path, args.query, kind=args.kind, limit=args.limit
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"focus_visualization: {output_path}")
    else:
        print(report)
    return 0


def _run_semantic_index(args: argparse.Namespace) -> int:
    _configure_logging(args.log_level)
    embedder = build_text_embedder(
        args.model,
        batch_size=args.batch_size,
        truncate_dim=args.truncate_dim,
        backend=args.backend,
    )
    builder = SemanticIndexBuilder(
        args.db_path,
        embedder=embedder,
        config=SemanticIndexConfig(
            output_dir=Path(args.output_dir) if args.output_dir else None
        ),
    )
    summary = builder.build()
    print(_summarize_semantic_index(summary))
    return 0


def _run_semantic_search(args: argparse.Namespace) -> int:
    _configure_logging(args.log_level)
    manifest = load_semantic_manifest(args.db_path, index_dir=args.index_dir)
    embedder = build_text_embedder(
        str(manifest["model_name"]),
        truncate_dim=int(manifest["dimension"]),
        backend=args.backend,
    )
    result = axe_semantic_search(
        args.db_path,
        args.query,
        index_dir=args.index_dir,
        limit=args.limit,
        search_k=args.search_k,
        task=args.task,
        embedder=embedder,
    )
    print(_format_retrieval_result(result))
    return 0


def _run_hierarchy_search(args: argparse.Namespace) -> int:
    _configure_logging(args.log_level)
    manifest = load_semantic_manifest(args.db_path, index_dir=args.index_dir)
    embedder = build_text_embedder(
        str(manifest["model_name"]),
        truncate_dim=int(manifest["dimension"]),
        backend=args.backend,
    )
    result = axe_hierarchy_search(
        args.db_path,
        args.query,
        index_dir=args.index_dir,
        limit=args.limit,
        branch_width=args.branch_width,
        symbol_limit=args.symbol_limit,
        embedder=embedder,
    )
    print(_format_hierarchical_search_result(result))
    return 0


def _run_question(args: argparse.Namespace) -> int:
    _configure_logging(args.log_level)
    manifest = load_semantic_manifest(args.db_path, index_dir=args.index_dir)
    embedder = build_text_embedder(
        str(manifest["model_name"]),
        truncate_dim=int(manifest["dimension"]),
        backend=args.backend,
    )
    result = axe_question(
        args.db_path, args.query, index_dir=args.index_dir, embedder=embedder
    )
    print(_format_question_result(result))
    return 0


def _run_serve(args: argparse.Namespace) -> int:
    _configure_logging(args.log_level)
    run_dashboard(
        db_path=args.db_path,
        index_dir=args.index_dir,
        port=args.port,
        open_browser=not args.no_browser,
    )
    return 0


def _summarize_result(summary, output_path: Path) -> str:
    repo_categories = ", ".join(summary.repo_categories)
    return "\n".join(
        [
            f"database: {output_path}",
            f"files: {summary.file_count}",
            f"folders: {summary.folder_count}",
            f"symbols: {summary.symbol_count}",
            f"imports: {summary.import_count}",
            f"calls: {summary.call_count}",
            f"references: {summary.reference_count}",
            f"repo_summary: {summary.repo_summary}",
            f"repo_categories: {repo_categories}",
        ]
    )


def _normalize_excluded_suffixes(
    values: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if not value:
            continue
        suffix = value if value.startswith(".") else f".{value}"
        normalized.append(suffix.lower())
    return tuple(normalized)


def _default_analysis_db_path(repo_root: Path) -> Path:
    return repo_root / ".matryoshka" / f"{repo_root.name}.db"


def _summarize_semantic_index(summary) -> str:
    return "\n".join(
        [
            f"semantic_index: {summary.index_dir}",
            f"model: {summary.model_name}",
            f"dimension: {summary.dimension}",
            f"engine: {summary.engine}",
            f"nodes: {summary.node_count}",
            f"symbols: {summary.symbol_count}",
            f"centroids: {summary.centroid_count}",
        ]
    )


def _format_retrieval_result(result) -> str:
    lines = [f"query: {result.query}"]
    if result.node_hits:
        lines.append("nodes:")
        for hit in result.node_hits:
            lines.append(
                f"  - {hit.node.path} [{hit.node.kind}] score={hit.score:.2f} category={hit.node.primary_category or 'none'}"
            )
            if hit.node.summary:
                lines.append(f"    summary: {hit.node.summary}")
            for context in hit.contexts[:3]:
                lines.append(
                    f"    context: {context.source_node_id} ({context.strength_label}) -> {context.inherited_summary}"
                )
    if result.symbol_hits:
        lines.append("symbols:")
        for hit in result.symbol_hits:
            symbol = hit.symbol
            location = (
                f"{symbol.path}:{symbol.start_line or 0}:{symbol.start_column or 0}"
            )
            lines.append(
                f"  - {symbol.qualified_name} [{symbol.kind}] score={hit.score:.2f} at {location}"
            )
            lines.append(f"    signature: {_compact_signature(symbol.signature)}")
            if hit.called_by:
                lines.append(
                    f"    called_by: {', '.join(_call_source_label(call) for call in hit.called_by[:5])}"
                )
            if hit.callees:
                lines.append(
                    f"    callees: {', '.join(call.callee_name for call in hit.callees[:5])}"
                )
            if hit.references:
                lines.append(
                    f"    references: {', '.join(_reference_label(reference) for reference in hit.references[:5])}"
                )
    return "\n".join(lines)


def _format_exact_search_result(result) -> str:
    lines = [f"query: {result.query}", f"search_type: {result.search_type}"]
    if result.node_hits:
        lines.append("nodes:")
        for hit in result.node_hits:
            lines.append(
                f"  - {hit.node.path} [{hit.node.kind}] score={hit.score:.2f} category={hit.node.primary_category or 'none'}"
            )
            if hit.node.summary:
                lines.append(f"    summary: {hit.node.summary}")
    if result.symbol_hits:
        lines.append("symbols:")
        for hit in result.symbol_hits:
            symbol = hit.symbol
            location = (
                f"{symbol.path}:{symbol.start_line or 0}:{symbol.start_column or 0}"
            )
            lines.append(
                f"  - {symbol.qualified_name} [{symbol.kind}] score={hit.score:.2f} at {location}"
            )
            lines.append(f"    signature: {_compact_signature(symbol.signature)}")
    if result.import_hits:
        lines.append("imports:")
        for hit in result.import_hits:
            target_path = (
                hit.target_node.path if hit.target_node is not None else "unresolved"
            )
            lines.append(
                f"  - {hit.import_record.imported_module} score={hit.score:.2f} importer={hit.importer_node.path} target={target_path}"
            )
    if result.call_hits:
        lines.append("calls:")
        for hit in result.call_hits:
            caller = (
                hit.caller_symbol.qualified_name
                if hit.caller_symbol is not None
                else hit.call_record.caller_node_id
            )
            target = (
                hit.target_symbol.qualified_name
                if hit.target_symbol is not None
                else hit.call_record.callee_name
            )
            lines.append(
                f"  - {caller} -> {target} score={hit.score:.2f} at {hit.call_record.caller_node_id}:{hit.call_record.start_line or 0}"
            )
    if result.reference_hits:
        lines.append("references:")
        for hit in result.reference_hits:
            source = (
                hit.source_symbol.qualified_name
                if hit.source_symbol is not None
                else hit.reference_record.source_node_id
            )
            target = (
                hit.target_symbol.qualified_name
                if hit.target_symbol is not None
                else hit.reference_record.target_name
            )
            lines.append(
                f"  - {source} -> {target} [{hit.reference_record.reference_kind}] score={hit.score:.2f} at {hit.reference_record.source_node_id}:{hit.reference_record.start_line or 0}"
            )
    return "\n".join(lines)


def _format_hierarchical_search_result(result) -> str:
    lines = [f"query: {result.query}"]
    if result.steps:
        lines.append("traversal:")
        for step in result.steps:
            labels = ", ".join(
                f"{candidate.node.path} ({candidate.score:.2f})"
                for candidate in step.candidates
            )
            lines.append(f"  - {step.level}: {labels}")
    if result.node_hits:
        lines.append("nodes:")
        for hit in result.node_hits:
            lines.append(f"  - {hit.node.path} [{hit.node.kind}] score={hit.score:.2f}")
            if hit.node.summary:
                lines.append(f"    summary: {hit.node.summary}")
    if result.symbol_hits:
        lines.append("symbols:")
        for hit in result.symbol_hits:
            lines.append(
                f"  - {hit.symbol.qualified_name} [{hit.symbol.kind}] score={hit.score:.2f} at {hit.symbol.path}:{hit.symbol.start_line or 0}:{hit.symbol.start_column or 0}"
            )
    return "\n".join(lines)


def _format_question_result(result) -> str:
    lines = [result.answer]
    if result.excerpts:
        lines.append("")
        lines.append("Excerpts:")
        for excerpt in result.excerpts[:2]:
            lines.append(f"  - {excerpt.path}:{excerpt.start_line}-{excerpt.end_line}")
    return "\n".join(lines)


def _compact_signature(signature: str, *, max_length: int = 220) -> str:
    compact = " ".join(signature.split())
    if "=>" in compact and "{" in compact:
        compact = compact.split("{", 1)[0].rstrip() + " { ... }"
    elif "{" in compact and compact.startswith("function "):
        compact = compact.split("{", 1)[0].rstrip() + " { ... }"
    if len(compact) > max_length:
        return compact[: max_length - 3].rstrip() + "..."
    return compact


def _call_source_label(call) -> str:
    return f"{call.caller_node_id}:{call.start_line or 0}"


def _reference_label(reference) -> str:
    return f"{reference.source_node_id}:{reference.start_line or 0} ({reference.reference_kind})"


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


# ------------------------------------------------------------------
# read / read-more
# ------------------------------------------------------------------


def _run_read(args: argparse.Namespace) -> int:
    _configure_logging(args.log_level)
    try:
        reader = FileReader(args.db_path)
        result = reader.read(args.file_path)
        print(_format_read_result(result, include_source=False))
        return 0
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run_read_more(args: argparse.Namespace) -> int:
    _configure_logging(args.log_level)
    try:
        reader = FileReader(args.db_path)
        result = reader.read_more(args.file_path)
        print(_format_read_result(result, include_source=True))
        return 0
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _format_read_result(result, *, include_source: bool = False) -> str:
    """Format a FileReadResult for terminal output."""
    from matryoshka.file_reader import FileReadResult

    node = result.node
    lines: list[str] = []

    # ---- header ----
    lines.append(f"file: {node.path}")
    lines.append(f"kind: {node.kind}")
    lines.append(f"language: {node.language or 'unknown'}")
    lines.append(f"symbols: {node.symbol_count}")
    lines.append(f"imports: {node.import_count}")
    if node.primary_category:
        lines.append(f"category: {node.primary_category}")
    if node.categories:
        lines.append(f"categories: {', '.join(node.categories)}")
    if node.tags:
        lines.append(f"tags: {', '.join(node.tags)}")
    if node.summary:
        lines.append(f"summary: {node.summary}")
    if node.description:
        lines.append(f"description: {node.description}")

    # ---- repo context ----
    if result.repo_summary:
        lines.append("")
        lines.append(f"repo_summary: {result.repo_summary}")
    if result.repo_categories:
        lines.append(f"repo_tags: {', '.join(result.repo_categories)}")

    # ---- imports ----
    if result.imports:
        lines.append("")
        lines.append("imports:")
        for imp in result.imports:
            internal = "internal" if imp.is_internal else "external"
            strength = imp.strength_label
            names = ", ".join(imp.names) if imp.names else imp.imported_module
            line_info = f"L{imp.start_line}" if imp.start_line else ""
            lines.append(
                f"  - {imp.imported_module} [{internal}] ({strength}) {names} {line_info}"
            )

    # ---- symbols ----
    if result.symbols:
        lines.append("")
        lines.append("symbols:")
        for sym in result.symbols:
            loc = f"L{sym.start_line or 0}-{sym.end_line or 0}"
            sig = _compact_signature(sym.signature or "")
            parent = f" (parent: {sym.parent_name})" if sym.parent_name else ""
            lines.append(f"  - {sym.qualified_name} [{sym.kind}] {loc}{parent}")
            lines.append(f"    signature: {sig}")
            if sym.docstring:
                first_line = sym.docstring.strip().split("\n")[0][:120]
                lines.append(f"    doc: {first_line}")
            if sym.parameters:
                lines.append(f"    params: {', '.join(sym.parameters)}")
            if sym.return_type:
                lines.append(f"    returns: {sym.return_type}")
            if sym.base_classes:
                lines.append(f"    extends: {', '.join(sym.base_classes)}")
            if sym.decorators:
                lines.append(f"    decorators: {', '.join(sym.decorators)}")

    # ---- exports ----
    if result.exports:
        lines.append("")
        lines.append("exports (referenced by other files):")
        for sym in result.exports:
            loc = f"L{sym.start_line or 0}"
            lines.append(f"  - {sym.qualified_name} [{sym.kind}] {loc}")

    # ---- calls ----
    if result.called_by:
        lines.append("")
        lines.append("called_by (other files calling this file):")
        for call in result.called_by[:10]:
            lines.append(
                f"  - {call.caller_node_id}:{call.start_line or 0} -> {call.callee_name}"
            )
    if result.callees:
        lines.append("")
        lines.append("callees (this file calling others):")
        for call in result.callees[:10]:
            target = call.target_node_id or call.callee_name
            lines.append(
                f"  - {call.callee_name} -> {target} (caller L{call.start_line or 0})"
            )

    # ---- references ----
    if result.references:
        lines.append("")
        lines.append("references (from this file):")
        for ref in result.references[:10]:
            lines.append(
                f"  - {ref.target_name} [{ref.reference_kind}] -> {ref.target_node_id or 'unresolved'} (L{ref.start_line or 0})"
            )
    if result.reverse_references:
        lines.append("")
        lines.append("reverse_references (into this file):")
        for ref in result.reverse_references[:10]:
            lines.append(
                f"  - {ref.source_node_id} -> {ref.target_name} [{ref.reference_kind}] (L{ref.start_line or 0})"
            )

    # ---- related context ----
    if result.contexts:
        lines.append("")
        lines.append("related_context:")
        for ctx in result.contexts:
            lines.append(
                f"  - {ctx.source_node_id} ({ctx.strength_label}): {ctx.inherited_summary}"
            )

    # ---- source blocks (read_more only) ----
    if include_source and result.symbol_blocks:
        lines.append("")
        lines.append("===== COLLAPSED SOURCE BLOCKS =====")
        for block in result.symbol_blocks:
            lines.append(block)

    if include_source and result.import_lines:
        lines.append("")
        lines.append("===== IMPORT LINES (from source) =====")
        for line in result.import_lines:
            lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
