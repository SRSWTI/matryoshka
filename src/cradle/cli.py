from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cradle.cache import LabelCache
from cradle.db_visualization import build_db_visualization
from cradle.embeddings import DEFAULT_EMBEDDING_MODEL, build_text_embedder
from cradle.exact_search import axe_call_search, axe_file_search, axe_import_search, axe_module_search, axe_reference_search, axe_symbol_search
from cradle.focus_visualization import build_focus_visualization
from cradle.hierarchical_search import axe_hierarchy_search
from cradle.labeling import LabelingConfig, LabelingEngine
from cradle.llm_client import LLMClientConfig, OpenAICompatibleClient
from cradle.pipeline import CradlePipeline, PipelineConfig
from cradle.question_answering import axe_question
from cradle.retrieval import axe_retrieval
from cradle.semantic_index import SemanticIndexBuilder, SemanticIndexConfig, load_semantic_manifest
from cradle.semantic_search import axe_semantic_search
from cradle.storage import CradleDatabase

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(prog="cradle")
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
    focus_parser.add_argument("--kind", default="auto", choices=["auto", "file", "symbol"])
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
    semantic_index_parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "sentence-transformers"])
    semantic_index_parser.add_argument("--log-level", default="INFO")

    semantic_search_parser = subparsers.add_parser("semantic-search")
    semantic_search_parser.add_argument("db_path")
    semantic_search_parser.add_argument("query")
    semantic_search_parser.add_argument("--index-dir", default=None)
    semantic_search_parser.add_argument("--limit", type=int, default=5)
    semantic_search_parser.add_argument("--search-k", type=int, default=None)
    semantic_search_parser.add_argument("--task", default=None)
    semantic_search_parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "sentence-transformers"])
    semantic_search_parser.add_argument("--log-level", default="INFO")

    hierarchy_search_parser = subparsers.add_parser("hierarchy-search")
    hierarchy_search_parser.add_argument("db_path")
    hierarchy_search_parser.add_argument("query")
    hierarchy_search_parser.add_argument("--index-dir", default=None)
    hierarchy_search_parser.add_argument("--limit", type=int, default=5)
    hierarchy_search_parser.add_argument("--branch-width", type=int, default=3)
    hierarchy_search_parser.add_argument("--symbol-limit", type=int, default=5)
    hierarchy_search_parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "sentence-transformers"])
    hierarchy_search_parser.add_argument("--log-level", default="INFO")

    question_parser = subparsers.add_parser("question")
    question_parser.add_argument("db_path")
    question_parser.add_argument("query")
    question_parser.add_argument("--index-dir", default=None)
    question_parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "sentence-transformers"])
    question_parser.add_argument("--log-level", default="INFO")

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
    return 1


def _run_analyze(args: argparse.Namespace) -> int:
    _configure_logging(args.log_level)
    repo_root = Path(args.repo_root)
    output_path = Path(args.output) if args.output else repo_root / ".cradle" / "index.db"
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
    engine = LabelingEngine(client, LabelingConfig(temperature=args.temperature, max_tokens=args.max_tokens), cache=cache)
    pipeline = CradlePipeline(
        config=PipelineConfig(
            max_files=args.max_files,
            excluded_paths=tuple(args.exclude_path or ()),
            excluded_suffixes=_normalize_excluded_suffixes(args.exclude_extension or ()),
        ),
        labeling_engine=engine,
    )
    graph = pipeline.analyze(repo_root, progress=lambda message: print(f"progress: {message}", flush=True))
    database = CradleDatabase(output_path)
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
    report = build_focus_visualization(args.db_path, args.query, kind=args.kind, limit=args.limit)
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
        config=SemanticIndexConfig(output_dir=Path(args.output_dir) if args.output_dir else None),
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
    result = axe_question(args.db_path, args.query, index_dir=args.index_dir, embedder=embedder)
    print(_format_question_result(result))
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


def _normalize_excluded_suffixes(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if not value:
            continue
        suffix = value if value.startswith(".") else f".{value}"
        normalized.append(suffix.lower())
    return tuple(normalized)


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
            lines.append(f"  - {hit.node.path} [{hit.node.kind}] score={hit.score:.2f} category={hit.node.primary_category or 'none'}")
            if hit.node.summary:
                lines.append(f"    summary: {hit.node.summary}")
            for context in hit.contexts[:3]:
                lines.append(f"    context: {context.source_node_id} ({context.strength_label}) -> {context.inherited_summary}")
    if result.symbol_hits:
        lines.append("symbols:")
        for hit in result.symbol_hits:
            symbol = hit.symbol
            location = f"{symbol.path}:{symbol.start_line or 0}:{symbol.start_column or 0}"
            lines.append(f"  - {symbol.qualified_name} [{symbol.kind}] score={hit.score:.2f} at {location}")
            lines.append(f"    signature: {_compact_signature(symbol.signature)}")
            if hit.called_by:
                lines.append(f"    called_by: {', '.join(_call_source_label(call) for call in hit.called_by[:5])}")
            if hit.callees:
                lines.append(f"    callees: {', '.join(call.callee_name for call in hit.callees[:5])}")
            if hit.references:
                lines.append(f"    references: {', '.join(_reference_label(reference) for reference in hit.references[:5])}")
    return "\n".join(lines)


def _format_exact_search_result(result) -> str:
    lines = [f"query: {result.query}", f"search_type: {result.search_type}"]
    if result.node_hits:
        lines.append("nodes:")
        for hit in result.node_hits:
            lines.append(f"  - {hit.node.path} [{hit.node.kind}] score={hit.score:.2f} category={hit.node.primary_category or 'none'}")
            if hit.node.summary:
                lines.append(f"    summary: {hit.node.summary}")
    if result.symbol_hits:
        lines.append("symbols:")
        for hit in result.symbol_hits:
            symbol = hit.symbol
            location = f"{symbol.path}:{symbol.start_line or 0}:{symbol.start_column or 0}"
            lines.append(f"  - {symbol.qualified_name} [{symbol.kind}] score={hit.score:.2f} at {location}")
            lines.append(f"    signature: {_compact_signature(symbol.signature)}")
    if result.import_hits:
        lines.append("imports:")
        for hit in result.import_hits:
            target_path = hit.target_node.path if hit.target_node is not None else "unresolved"
            lines.append(
                f"  - {hit.import_record.imported_module} score={hit.score:.2f} importer={hit.importer_node.path} target={target_path}"
            )
    if result.call_hits:
        lines.append("calls:")
        for hit in result.call_hits:
            caller = hit.caller_symbol.qualified_name if hit.caller_symbol is not None else hit.call_record.caller_node_id
            target = hit.target_symbol.qualified_name if hit.target_symbol is not None else hit.call_record.callee_name
            lines.append(f"  - {caller} -> {target} score={hit.score:.2f} at {hit.call_record.caller_node_id}:{hit.call_record.start_line or 0}")
    if result.reference_hits:
        lines.append("references:")
        for hit in result.reference_hits:
            source = hit.source_symbol.qualified_name if hit.source_symbol is not None else hit.reference_record.source_node_id
            target = hit.target_symbol.qualified_name if hit.target_symbol is not None else hit.reference_record.target_name
            lines.append(
                f"  - {source} -> {target} [{hit.reference_record.reference_kind}] score={hit.score:.2f} at {hit.reference_record.source_node_id}:{hit.reference_record.start_line or 0}"
            )
    return "\n".join(lines)


def _format_hierarchical_search_result(result) -> str:
    lines = [f"query: {result.query}"]
    if result.steps:
        lines.append("traversal:")
        for step in result.steps:
            labels = ", ".join(f"{candidate.node.path} ({candidate.score:.2f})" for candidate in step.candidates)
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
            lines.append(f"  - {hit.symbol.qualified_name} [{hit.symbol.kind}] score={hit.score:.2f} at {hit.symbol.path}:{hit.symbol.start_line or 0}:{hit.symbol.start_column or 0}")
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
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    raise SystemExit(main())