from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cradle.cache import LabelCache
from cradle.db_visualization import build_db_visualization
from cradle.labeling import LabelingConfig, LabelingEngine
from cradle.llm_client import LLMClientConfig, OpenAICompatibleClient
from cradle.pipeline import CradlePipeline, PipelineConfig
from cradle.retrieval import axe_retrieval
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

    retrieve_parser = subparsers.add_parser("retrieve")
    retrieve_parser.add_argument("db_path")
    retrieve_parser.add_argument("query")
    retrieve_parser.add_argument("--limit", type=int, default=5)
    retrieve_parser.add_argument("--log-level", default="INFO")

    visualize_parser = subparsers.add_parser("visualize-db")
    visualize_parser.add_argument("db_path")
    visualize_parser.add_argument("--output", default=None)
    visualize_parser.add_argument("--sample-limit", type=int, default=10)
    visualize_parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    if args.command == "analyze":
        return _run_analyze(args)
    if args.command == "retrieve":
        return _run_retrieve(args)
    if args.command == "visualize-db":
        return _run_visualize_db(args)
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
    pipeline = CradlePipeline(config=PipelineConfig(max_files=args.max_files), labeling_engine=engine)
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