# Matryoshka

Matryoshka is a Rust-first code-intelligence layer for coding agents.

It prewarms a repository into a SQLite-backed map of:

- files
- folders
- symbols
- import and dependency edges
- rich file, folder, and repo cards
- semantic search records for files, snippets, symbols, folders, and the repo

The goal is simple: let an agent search for behavior and read rich summaries
before it falls back to full-file reads.

## What It Does

Matryoshka currently ships these core workflows:

1. `index`
   Parse a repository, resolve structural relationships, generate rich cards,
   build semantic records, and persist everything into SQLite.

2. `update`
   Re-run the pipeline incrementally against a changed repository and refresh
   affected facts, cards, and semantic records.

3. `watch`
   Poll a repository, debounce changes, and trigger `update` automatically.

4. `search`
   Run hybrid retrieval over persisted semantic records using embeddings plus
   lexical, ownership, intent, and structural boosts.

5. `read`
   Return a rich file card with folder context, interpreted imports,
   dependents, blast radius, and selected snippets.

6. `read-more`
   Extend `read` with symbol blocks, import lines, and larger source excerpts.

7. `rebuild-semantic`
   Rebuild the semantic search layer from already-persisted facts and cards
   without reparsing or re-enriching the whole repository.

## Architecture

The workspace is organized around focused Rust crates:

- `core-ir`
  Shared facts, cards, semantic records, provenance, and API types.

- `parser`
  Source walking and symbol/import/snippet extraction.

- `resolver`
  Folder graph construction, import resolution, and dependency edge creation.

- `store-sqlite`
  Canonical persistence for facts, cards, semantic records, and invalidation.

- `enricher`
  Rich file, folder, and repo card generation through MLX chat or heuristic fallback.

- `embed-client`
  OpenAI-compatible embeddings client plus deterministic offline embedder.

- `indexer`
  Full prewarm, incremental refresh, and semantic repair orchestration.

- `search`
  Hybrid semantic search and reranking.

- `read-api`
  `read` and `read-more` assembly.

- `watcher`
  Polling and debounce-based repo change detection.

- `cli`
  The `matryoshka-rs` command surface.

## Storage Model

SQLite is the source of truth.

Persisted tables include:

- structural facts
  - files
  - folders
  - symbols
  - edges

- enriched artifacts
  - file cards
  - folder cards
  - repo card

- retrieval layer
  - semantic records

This means semantic search can be rebuilt independently when embeddings or late
pipeline stages fail.

## Local MLX Defaults

The non-offline path expects a local OpenAI-compatible MLX server:

- base URL: `http://127.0.0.1:44445`
- API key: `2508`
- embeddings model: `mlx-community--embeddinggemma-300m-bf16`
- chat model: `MercuriusDream--Qwen3.5-4B-MLX-mxfp8`

Chat enrichment disables thinking by default.

You can override them per command with:

- `--model <chat-model>`
- `--embedding-model <embedding-model>`

## Quick Start

Index a repo:

```bash
cargo run -p matryoshka-cli -- index /path/to/repo --db /path/to/repo/.matryoshka/index.db
```

Search it:

```bash
cargo run -p matryoshka-cli -- search "authentication flow" --db /path/to/repo/.matryoshka/index.db
```

Read a file card:

```bash
cargo run -p matryoshka-cli -- read \
  --db /path/to/repo/.matryoshka/index.db \
  --repo-root /path/to/repo \
  path/to/file.py
```

Repair only the semantic layer:

```bash
cargo run -p matryoshka-cli -- rebuild-semantic \
  /path/to/repo \
  --db /path/to/repo/.matryoshka/index.db
```

## Docs

- See [usage.md](/Users/rohit/cradle-embed/usage.md) for commands, flags, and examples.
- See [crates/README.md](/Users/rohit/cradle-embed/crates/README.md) for crate-local notes.
