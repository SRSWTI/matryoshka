# Matryoshka

Matryoshka is a Rust-first code-intelligence layer for coding agents.

It prewarms a repository into a SQLite-backed map of:

- files
- folders
- symbols
- import and dependency edges
- rich file, folder, and repo cards
- semantic search records for files, snippets, symbols, folders, and the repo
- SQLite FTS records and late-interaction token vectors for hybrid retrieval

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
   Poll a repository, run a startup freshness update, debounce changes, and
   trigger `update` automatically. It can run in the foreground or as a daemon.

4. `prewarm`
   Rebuild FTS, warm retrieval paths, optionally ensure freshness, and optionally
   start the watcher.

5. `search`
   Run hybrid retrieval over persisted semantic records using embeddings plus
   FTS, lexical, symbol/path, late-interaction, ownership, intent, graph, and
   structural boosts.

6. `op`
   Run task-shaped search for agent operations such as `find-symbol`,
   `edit-target`, `trace-dependency`, `architecture`, and `tests-for`.

7. `read`
   Return a rich file card with folder context, interpreted imports,
   dependents, blast radius, and selected snippets.

8. `read-bundle`
   Search, pick a primary file, select related files, and return packed read
   context in `brief`, `edit`, or `flow` mode.

9. `rebuild-semantic`
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
  `read` and `read-bundle` assembly.

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
  - FTS records
  - late-interaction vectors

By default, Matryoshka stores operational state under:

```text
<repo>/.matryoshka/
```

The default SQLite DB is:

```text
<repo>/.matryoshka/matryoshka.db
```

Watcher daemon files live next to it:

```text
<repo>/.matryoshka/watch.pid
<repo>/.matryoshka/logs/watch.jsonl
<repo>/.matryoshka/logs/watch.stdout.jsonl
```

This means semantic search can be rebuilt independently when embeddings or late
pipeline stages fail.

## Local MLX Defaults

The non-offline path expects a local OpenAI-compatible MLX server:

- base URL: `http://127.0.0.1:44445`
- API key: `2508`
- embeddings model: `mlx-community--embeddinggemma-300m-bf16`
- chat model: `MercuriusDream--Qwen3.5-4B-MLX-mxfp8`
- oMLX reranker model: `mlx-community--Qwen3-Reranker-0.6B-mxfp8`

Chat enrichment disables thinking by default.

You can override them per command with:

- `--model <chat-model>`
- `--embedding-model <embedding-model>`

## Quick Start

Examples below assume `matryoshka-rs` is on your `PATH` or exported from
`target/debug`. From this repo, prefix commands with:

```bash
cargo run -p matryoshka-cli --
```

Index a repo:

```bash
matryoshka-rs index /path/to/repo --watch-daemon
```

Prewarm retrieval:

```bash
matryoshka-rs prewarm \
  --repo-root /path/to/repo \
  --ensure-fresh \
  --query "auth flow token refresh" \
  --query "where policy enforcement happens"
```

Search it:

```bash
cd /path/to/repo
matryoshka-rs search "authentication flow" --omlx-rerank
```

Read a file card:

```bash
matryoshka-rs read \
  --repo-root /path/to/repo \
  path/to/file.py
```

Read a packed context bundle:

```bash
matryoshka-rs read-bundle \
  --repo-root /path/to/repo \
  --mode edit \
  "where should I edit retry behavior"
```

Repair only the semantic layer:

```bash
matryoshka-rs rebuild-semantic \
  /path/to/repo
```

## Docs

- See [usage.md](/Users/rohit/cradle-embed/usage.md) for commands, flags, and examples.
- See [crates/README.md](/Users/rohit/cradle-embed/crates/README.md) for crate-local notes.
