# Matryoshka Usage

This document describes the `matryoshka-rs` CLI surface as it exists in this
repository today.

## Command Summary

```bash
matryoshka-rs <command> [args] [flags]
```

Available commands:

- `index`
- `update`
- `watch`
- `rebuild-semantic`
- `search`
- `read`
- `read-more`

## Shared MLX Defaults

These defaults are used by the non-offline commands unless you override them:

- `--base-url`
  Default: `http://127.0.0.1:44445`
  The OpenAI-compatible MLX HTTP endpoint.

- `--api-key`
  Default: `2508`
  Bearer token used for both chat and embeddings requests.

- `--embedding-model`
  Default: `mlx-community--embeddinggemma-300m-bf16`
  Embedding model used for semantic indexing and search queries.

- `--model`
  Default: `MercuriusDream--Qwen3.5-4B-MLX-mxfp8`
  Chat model used for file, folder, and repo enrichment.

The older flag names still work as aliases:

- `--embed-model`
- `--chat-model`

- `--offline`
  Uses deterministic local embeddings and heuristic enrichment instead of MLX.
  Useful for tests, smoke checks, and environments where the MLX server is unavailable.

## `index`

Build a full Matryoshka database for a repository.

```bash
cargo run -p matryoshka-cli -- index <repo_root> --db <db_path> [flags]
```

Arguments:

- `<repo_root>`
  Absolute or relative path to the repository you want to index.

Required flags:

- `--db <db_path>`
  SQLite destination path. A common pattern is:
  `/path/to/repo/.matryoshka/index.db`

Optional flags:

- `--offline`
- `--base-url <url>`
- `--api-key <key>`
- `--embedding-model <model>`
- `--model <model>`

What it does:

- parses the repo
- resolves folders and import/dependency edges
- writes structural facts
- generates file/folder/repo cards
- creates semantic search records
- embeds search records

Example:

```bash
cargo run -p matryoshka-cli -- index /Users/rohit/octane-1 \
  --db /Users/rohit/octane-1/.matryoshka/octane-1.db
```

## `update`

Refresh an existing DB after code changes.

```bash
cargo run -p matryoshka-cli -- update <repo_root> --db <db_path> [flags]
```

Arguments:

- `<repo_root>`
  Repository root that was previously indexed.

Required flags:

- `--db <db_path>`
  Existing Matryoshka SQLite DB.

Optional flags:

- `--offline`
- `--base-url <url>`
- `--api-key <key>`
- `--embedding-model <model>`
- `--model <model>`

What it does:

- reparses the repo
- computes the delta
- refreshes affected files/folders/cards
- updates semantic records and embeddings for affected entities

## `watch`

Continuously poll a repo and trigger debounced updates.

```bash
cargo run -p matryoshka-cli -- watch <repo_root> --db <db_path> [flags]
```

Arguments:

- `<repo_root>`
  Repository root to monitor.

Required flags:

- `--db <db_path>`
  Existing Matryoshka SQLite DB.

Optional flags:

- `--offline`
- `--base-url <url>`
- `--api-key <key>`
- `--embedding-model <model>`
- `--model <model>`
- `--interval-ms <milliseconds>`
  Default: `2000`
  Polling interval for repo scanning.

- `--debounce-ms <milliseconds>`
  Default: `3000`
  Debounce window used to group rapid changes into a single update.

Use this when you want Matryoshka to stay warm during an active coding session.

## `rebuild-semantic`

Repair or rebuild the semantic search layer from the stored facts and cards
without reparsing the repo or rerunning enrichment.

```bash
cargo run -p matryoshka-cli -- rebuild-semantic <repo_root> --db <db_path> [flags]
```

Arguments:

- `<repo_root>`
  Repository root associated with the existing DB.

Required flags:

- `--db <db_path>`
  Existing Matryoshka SQLite DB.

Optional flags:

- `--offline`
- `--base-url <url>`
- `--api-key <key>`
- `--embedding-model <model>`

What it does:

- loads persisted files, folders, symbols, and cards from SQLite
- rebuilds raw semantic records
- rebuilds file/folder/repo card semantic records
- embeds them in batches
- atomically replaces `semantic_records`

Use this when:

- search returns poor or empty results
- the DB has cards but is missing searchable semantic card records
- embedding requests timed out during a prior run

## `search`

Run hybrid retrieval over the semantic index.

```bash
cargo run -p matryoshka-cli -- search "<query>" --db <db_path> [flags]
```

Arguments:

- `<query>`
  Natural-language or symbol-oriented query string.

Required flags:

- `--db <db_path>`
  Existing Matryoshka SQLite DB.

Optional flags:

- `--limit <n>`
  Default: `8`
  Maximum number of hits returned.

- `--offline`
- `--base-url <url>`
- `--api-key <key>`
- `--embedding-model <model>`

Search ranking combines:

- semantic similarity
- lexical/path/symbol overlap
- behavior intents
- edit intents
- retrieval tags
- ownership/facade signals
- folder/repo boosts

Example:

```bash
cargo run -p matryoshka-cli -- search "planner fallback when llm fails" \
  --db /Users/rohit/octane-1/.matryoshka/octane-1.db
```

## `read`

Read a rich file card without opening the full file manually first.

```bash
cargo run -p matryoshka-cli -- read --db <db_path> --repo-root <repo_root> <file>
```

Arguments:

- `<file>`
  Repository-relative path to the file you want to inspect.

Required flags:

- `--db <db_path>`
  Existing Matryoshka SQLite DB.

- `--repo-root <repo_root>`
  Repository root used for resolving the file path.

What it returns:

- file fact
- file card
- folder card
- imports
- incoming and outgoing edges
- selected snippets

Use this for agent-facing “file card” inspection.

## `read-more`

Read a richer expanded view of a file.

```bash
cargo run -p matryoshka-cli -- read-more --db <db_path> --repo-root <repo_root> <file>
```

Arguments:

- `<file>`
  Repository-relative path to the file you want to inspect.

Required flags:

- `--db <db_path>`
- `--repo-root <repo_root>`

What it adds on top of `read`:

- symbol blocks
- import lines
- larger source excerpts

Use this when the file card is not enough and the agent needs more grounded source context.

## Useful Environment Overrides

- `MATRYOSHKA_ENRICH_CONCURRENCY`
  Controls parallel MLX chat enrichment concurrency.
  Default: `6`

- `MATRYOSHKA_EMBED_BATCH`
  Controls embedding batch size for semantic-record embedding requests.
  Default: `64`

This is especially useful for tuning large repos against your local MLX server.
