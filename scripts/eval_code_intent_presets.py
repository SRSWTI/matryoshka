#!/usr/bin/env python3
"""Run code intent search across chunk presets and sample queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mlx_embeddings import load

from code_intent_search import (
    DEFAULT_MODEL,
    build_index,
    embed_texts,
    load_index,
    snippet,
)


PRESETS = [
    ("precise", 40, 10),
    ("balanced", 100, 25),
    ("broad", 180, 45),
]

QUERIES = [
    "how prepare chooses between index update repair rebuild_search and prepare_results",
    "where cancellation is checked before indexing repair search rebuild and prewarm",
    "how progress.json gets written for prepare status phase percent and current file",
    "where changed files refresh file cards folder cards semantic records and late vectors",
    "how semantic records are built for files snippets symbols folders and repo cards",
    "where late interaction token vectors are generated with camelCase and underscore splitting",
    "how search combines FTS dense embeddings exact token matches graph boosts cards and reranking",
    "where query text is classified into find symbol edit target architecture test lookup trace dependency",
    "how multiple symbol and snippet hits collapse into one file level search result",
    "where read-bundle picks a primary file and packs related files in brief edit or flow mode",
    "how watcher debounces filesystem changes and merges added changed removed paths",
    "where SQLite stores semantic records FTS rows late vectors file cards and orphan pruning",
    "tests for real oMLX prepare search read lifecycle and progress embedding batches",
    "how heuristic enrichment summarizes file roles behaviors side effects blast radius and retrieval tags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate code intent search over preset chunk sizes."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=Path("/private/tmp/code-intent-eval"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-file-bytes", type=int, default=512_000)
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=["test_repo"],
        help="Additional directory name to exclude. Defaults to test_repo for this fixture-heavy repo.",
    )
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def rank(embeddings: np.ndarray, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    scores = embeddings @ query_vec
    return [(int(idx), float(scores[int(idx)])) for idx in np.argsort(-scores)[:top_k]]


def main() -> None:
    args = parse_args()
    args.root = args.root.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading query model {args.model}")
    query_model, query_tokenizer = load(args.model)

    summary = []
    for preset_name, chunk_lines, overlap_lines in PRESETS:
        print(f"\n=== preset={preset_name} chunk_lines={chunk_lines} overlap_lines={overlap_lines} ===")
        index_path = args.out_dir / f"{preset_name}.npz"

        build_args = argparse.Namespace(
            root=args.root,
            model=args.model,
            batch_size=args.batch_size,
            chunk_lines=chunk_lines,
            overlap_lines=overlap_lines,
            max_file_bytes=args.max_file_bytes,
            exclude_dir=args.exclude_dir,
        )
        if args.rebuild or not index_path.exists():
            embeddings, chunks = build_index(build_args, index_path)
        else:
            print(f"loading cached index {index_path}")
            embeddings, chunks = load_index(index_path)

        preset_results = {
            "preset": preset_name,
            "chunk_lines": chunk_lines,
            "overlap_lines": overlap_lines,
            "chunk_count": len(chunks),
            "queries": [],
        }

        for query in QUERIES:
            query_vec = embed_texts(query_model, query_tokenizer, [query], args.batch_size)[0]
            matches = []
            print(f"\nquery: {query}")
            for rank_num, (chunk_idx, score) in enumerate(
                rank(embeddings, query_vec, args.top_k), start=1
            ):
                chunk = chunks[chunk_idx]
                match = {
                    "rank": rank_num,
                    "score": score,
                    "path": chunk["path"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "snippet": chunk["text"],
                }
                matches.append(match)
                print(
                    f"{rank_num}. score={score:.4f} "
                    f"{chunk['path']}:{chunk['start']}-{chunk['end']}"
                )
                print(snippet(chunk["text"], max_lines=5))
            preset_results["queries"].append({"query": query, "matches": matches})

        raw_path = args.out_dir / f"{preset_name}.json"
        raw_path.write_text(json.dumps(preset_results, indent=2), encoding="utf-8")
        print(f"wrote {raw_path}")
        summary.append(preset_results)

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
