#!/usr/bin/env python3
"""Embed a codebase with MLX embeddings and rank code chunks for a query."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from mlx_embeddings import generate, load


DEFAULT_MODEL = "/Users/rohit/.omlx/models/naver--splade-code-06B"
DEFAULT_INDEX = ".code_intent_index.npz"

CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".lua",
    ".m",
    ".md",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".vue",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".cache",
    ".git",
    ".hg",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "target",
    "vendor",
}

SKIP_GLOBS = {
    "*.lock",
    "*.min.js",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.pdf",
    "*.safetensors",
    "*.sqlite",
    "*.db",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a code embedding index and print the top ranked chunks for a query."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Codebase root to embed. Defaults to the current directory.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="MLX embedding model path or HF repo id.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help=f"Index cache path. Defaults to <root>/{DEFAULT_INDEX}.",
    )
    parser.add_argument("--query", help="Query to rank against the codebase.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches to print.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size. Lower this if memory gets tight.",
    )
    parser.add_argument(
        "--chunk-lines",
        type=int,
        default=80,
        help="Approximate number of source lines per chunk.",
    )
    parser.add_argument(
        "--overlap-lines",
        type=int,
        default=20,
        help="Line overlap between adjacent chunks.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=512_000,
        help="Skip files larger than this many bytes.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Additional directory name to exclude. Can be passed more than once.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the embedding index even if a cache exists.",
    )
    return parser.parse_args()


def should_skip(path: Path, root: Path, max_file_bytes: int, skip_dirs: set[str]) -> bool:
    rel = path.relative_to(root)
    if any(part in skip_dirs for part in rel.parts):
        return True
    if path.suffix.lower() not in CODE_EXTENSIONS:
        return True
    if any(fnmatch.fnmatch(path.name, pattern) for pattern in SKIP_GLOBS):
        return True
    try:
        return path.stat().st_size > max_file_bytes
    except OSError:
        return True


def iter_code_files(root: Path, max_file_bytes: int, skip_dirs: set[str]) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        base = Path(dirpath)
        for filename in filenames:
            path = base / filename
            if not should_skip(path, root, max_file_bytes, skip_dirs):
                yield path


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except UnicodeDecodeError:
            return None
    except OSError:
        return None


def make_chunks(
    root: Path,
    chunk_lines: int,
    overlap_lines: int,
    max_file_bytes: int,
    exclude_dirs: Iterable[str] = (),
) -> list[dict]:
    chunks: list[dict] = []
    step = max(1, chunk_lines - overlap_lines)
    skip_dirs = SKIP_DIRS | set(exclude_dirs)
    for path in sorted(iter_code_files(root, max_file_bytes, skip_dirs)):
        text = read_text(path)
        if not text:
            continue
        lines = text.splitlines()
        if not lines:
            continue
        rel = str(path.relative_to(root))
        for start in range(0, len(lines), step):
            end = min(start + chunk_lines, len(lines))
            body = "\n".join(lines[start:end]).strip()
            if not body:
                continue
            chunks.append(
                {
                    "path": rel,
                    "start": start + 1,
                    "end": end,
                    "text": body,
                }
            )
            if end == len(lines):
                break
    return chunks


def as_numpy(array) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def embed_texts(model, tokenizer, texts: list[str], batch_size: int) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        output = generate(model, tokenizer, batch)
        if output.text_embeds is None:
            raise RuntimeError("model did not return text_embeds")
        vectors.append(as_numpy(output.text_embeds))
        print(f"embedded {min(offset + len(batch), len(texts))}/{len(texts)} chunks", flush=True)
    embeddings = np.vstack(vectors)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-9)


def build_index(args: argparse.Namespace, index_path: Path) -> tuple[np.ndarray, list[dict]]:
    print(f"scanning {args.root}")
    chunks = make_chunks(
        args.root,
        args.chunk_lines,
        args.overlap_lines,
        args.max_file_bytes,
        args.exclude_dir,
    )
    if not chunks:
        raise RuntimeError(f"no code chunks found under {args.root}")

    print(f"loading model {args.model}")
    model, tokenizer = load(args.model)
    texts = [format_chunk_for_embedding(chunk) for chunk in chunks]
    embeddings = embed_texts(model, tokenizer, texts, args.batch_size)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        index_path,
        embeddings=embeddings,
        chunks=json.dumps(chunks),
        model=args.model,
        root=str(args.root),
    )
    print(f"saved {len(chunks)} chunks to {index_path}")
    return embeddings, chunks


def load_index(index_path: Path) -> tuple[np.ndarray, list[dict]]:
    data = np.load(index_path, allow_pickle=False)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    chunks = json.loads(str(data["chunks"]))
    return embeddings, chunks


def format_chunk_for_embedding(chunk: dict) -> str:
    return f"path: {chunk['path']}\nlines: {chunk['start']}-{chunk['end']}\n\n{chunk['text']}"


def snippet(text: str, max_lines: int = 12) -> str:
    lines = text.strip().splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["..."]
    return "\n".join(f"    {line}" for line in lines)


def rank_query(
    model_path: str,
    query: str,
    embeddings: np.ndarray,
    chunks: list[dict],
    batch_size: int,
    top_k: int,
) -> None:
    model, tokenizer = load(model_path)
    query_vec = embed_texts(model, tokenizer, [query], batch_size=batch_size)[0]
    scores = embeddings @ query_vec
    top_indices = np.argsort(-scores)[:top_k]

    print(f"\nquery: {query}\n")
    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[int(idx)]
        score = float(scores[int(idx)])
        print(f"{rank}. score={score:.4f} {chunk['path']}:{chunk['start']}-{chunk['end']}")
        print(snippet(chunk["text"]))
        print()


def main() -> None:
    args = parse_args()
    args.root = args.root.expanduser().resolve()
    index_path = (
        args.index.expanduser().resolve()
        if args.index is not None
        else args.root / DEFAULT_INDEX
    )

    if args.rebuild or not index_path.exists():
        embeddings, chunks = build_index(args, index_path)
    else:
        print(f"loading cached index {index_path}")
        embeddings, chunks = load_index(index_path)

    query = args.query or input("query> ").strip()
    if not query:
        raise SystemExit("empty query")
    rank_query(args.model, query, embeddings, chunks, args.batch_size, args.top_k)


if __name__ == "__main__":
    main()
