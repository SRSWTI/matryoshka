# Matryoshka Usage

This document describes the `matryoshka-rs` CLI on the `matryoshka-boost`
branch.

Matryoshka is designed for coding agents: index once, keep the repo warm, then
use search/read operations to get precise context in fewer steps and fewer
tokens than raw file browsing.

## Command Summary

```bash
matryoshka-rs <command> [args] [flags]
```

Commands:

- `index`
- `update`
- `watch`
- `prewarm`
- `search`
- `op`
- `read`
- `read-bundle`
- `rebuild-semantic`

## Default Storage

Most commands accept `--db`, but it is no longer required when Matryoshka can
infer the repo root.

Default DB:

```bash
<repo_root>/.matryoshka/matryoshka.db
```

Watcher/daemon files:

```bash
<repo_root>/.matryoshka/watch.pid
<repo_root>/.matryoshka/logs/watch.jsonl
<repo_root>/.matryoshka/logs/watch.stdout.jsonl
```

Use explicit `--db` when you want multiple indexes for one repo:

```bash
matryoshka-rs index /path/to/repo --db /path/to/repo/.matryoshka/experiment.db
```

## MLX Defaults

Non-offline commands expect a local OpenAI-compatible MLX/oMLX server.

Defaults:

- `--base-url http://127.0.0.1:44445`
- `--api-key 2508`
- `--embedding-model mlx-community--embeddinggemma-300m-bf16`
- `--model MercuriusDream--Qwen3.5-4B-MLX-mxfp8`
- `--omlx-rerank-model mlx-community--Qwen3-Reranker-0.6B-mxfp8`

Aliases:

- `--embed-model` for `--embedding-model`
- `--chat-model` for `--model`

Use `--offline` for deterministic local embeddings and heuristic enrichment.
That is good for tests and smoke checks, but production-quality cards and
semantic matching should use MLX embeddings/enrichment.

## Recommended Production Flow

Start MLX/oMLX first:

```bash
cd /Users/rohit/cradle-mlx/helpers/omlx
source .venv/bin/activate
jesco-apple serve --host 127.0.0.1 --port 44445 --api-key 2508 --max-concurrent-requests 6
```

Index and start the watcher daemon:

```bash
matryoshka-rs index /path/to/repo \
  --model srswti--bodega-raptor-90m \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --ignore target \
  --ignore node_modules \
  --watch-daemon
```

Prewarm common retrieval paths:

```bash
matryoshka-rs prewarm \
  --repo-root /path/to/repo \
  --ensure-fresh \
  --query "auth flow token refresh" \
  --query "where policy enforcement happens" \
  --query "tests for parser behavior"
```

Then use `search`, `op`, `read`, and `read-bundle` during agent work.

## `index`

Build a full Matryoshka database.

```bash
matryoshka-rs index <repo_root> [flags]
```

Common flags:

- `--db <path>`
- `--offline`
- `--base-url <url>`
- `--api-key <key>`
- `--embedding-model <model>`
- `--model <chat-model>`
- `--progress-jsonl`
- `--ignore <path>`
- `--watch`
- `--watch-daemon`

What it does:

- parses source with Tree-sitter-backed extraction where supported
- falls back to the line parser for unsupported files
- extracts files, folders, symbols, imports, snippets, and structural metadata
- resolves import/dependency edges
- generates file, folder, and repo cards
- creates semantic records for files, symbols, snippets, cards, folders, and repo
- embeds records
- builds SQLite FTS records
- stores late-interaction token vectors for MaxSim-style matching

Output:

```text
files: <n>
folders: <n>
symbols: <n>
semantic_records: <n>
embedding_model: <model>
```

Use `--watch-daemon` when you want the repo to stay fresh immediately after the
first index.

## `update`

Refresh an existing DB after code changes.

```bash
matryoshka-rs update <repo_root> [flags]
```

Common flags:

- `--db <path>`
- `--offline`
- `--base-url <url>`
- `--api-key <key>`
- `--embedding-model <model>`
- `--model <chat-model>`
- `--progress-jsonl`
- `--ignore <path>`

What it does:

- reparses the repo
- computes changed, added, and removed files
- refreshes affected structural facts
- refreshes affected file/folder/repo cards
- deletes semantic records for removed paths
- updates FTS and late-interaction vectors
- repairs missing artifacts if a prior run was interrupted

Output includes:

```text
changed_files: <n>
removed_files: <n>
changed_folders: <n>
repo_card_updated: true|false
```

## `watch`

Keep the DB fresh during active coding.

```bash
matryoshka-rs watch <repo_root> [flags]
```

Common flags:

- `--db <path>`
- `--offline`
- `--interval-ms <n>` default `2000`
- `--debounce-ms <n>` default `3000`
- `--daemon`
- `--skip-startup-update`
- `--ignore <path>`

Important behavior:

- By default, `watch` runs one `update` before polling.
- That startup update prevents stale DBs when files changed after `index` but
  before `watch` started.
- The watcher then polls, debounces changes, and runs `update` per change batch.

Foreground:

```bash
matryoshka-rs watch /path/to/repo
```

Daemon:

```bash
matryoshka-rs watch /path/to/repo --daemon
```

Daemon state:

```bash
cat /path/to/repo/.matryoshka/watch.pid
tail -f /path/to/repo/.matryoshka/logs/watch.jsonl
tail -f /path/to/repo/.matryoshka/logs/watch.stdout.jsonl
```

`watch.jsonl` contains structured events such as:

- `watch_started`
- `update_started`
- `update_completed`
- `change_batch`

## `prewarm`

Warm retrieval paths and rebuild FTS.

```bash
matryoshka-rs prewarm [flags]
```

Common flags:

- `--repo-root <path>`
- `--db <path>`
- `--offline`
- `--embedding-model <model>`
- `--query <query>` repeatable
- `--limit <n>` default `6`
- `--ensure-fresh`
- `--watch`
- `--watch-daemon`
- `--no-late-interaction`

What it does:

- rebuilds `semantic_records_fts`
- runs each prewarm query through search
- warms embedding/ranking paths
- optionally runs `update` first with `--ensure-fresh`
- optionally starts watcher after prewarm with `--watch` or `--watch-daemon`

Output:

```text
fts_records: <n>
queries: <n>
warmed_hits: <n>
```

`--limit` means hits per prewarm query, not total files or total embeddings.

## `search`

Run hybrid retrieval over the repo intelligence index.

```bash
matryoshka-rs search "<query>" [flags]
```

Common flags:

- `--db <path>`
- `--limit <n>` default `8`
- `--offline`
- `--embedding-model <model>`
- `--rerank`
- `--rerank-model <chat-model>`
- `--omlx-rerank`
- `--omlx-rerank-model <reranker-model>`
- `--omlx-rerank-candidates <n>` default `20`
- `--no-late-interaction`

Search uses:

- query planning
- SQLite FTS candidates
- exact symbol/path candidates
- dense embedding similarity
- late-interaction MaxSim over indexed code-token vectors
- behavior/edit/retrieval tags
- file/folder/repo cards
- facade vs implementation ownership signals
- optional chat or oMLX reranking

Example:

```bash
matryoshka-rs search "where is MCP bearer authentication enforced" \
  --omlx-rerank
```

Search returns JSON hits with:

- `path`
- `summary`
- `description`
- `matched_terms`
- `matched_symbols`
- `score`
- `why_matched`

Use search when you know what behavior, symbol, or subsystem you need.

## `op`

Search with an explicit agent task.

```bash
matryoshka-rs op <task> "<query>" [flags]
```

Tasks:

- `find-symbol`
- `find-behavior`
- `edit-target`
- `trace-dependency`
- `architecture`
- `tests-for`
- `read-next`

Examples:

```bash
matryoshka-rs op find-symbol "resolve_import"
matryoshka-rs op edit-target "retry behavior in provider routing"
matryoshka-rs op trace-dependency "token refresh credentials"
matryoshka-rs op tests-for "parser tree sitter extraction"
```

Use `op` when the agent already knows its intent. It biases retrieval toward the
right evidence: symbols, implementation owners, dependency context, architecture
cards, or tests.

## `read`

Read one rich file card.

```bash
matryoshka-rs read <file> [flags]
```

Common flags:

- `--repo-root <path>`
- `--db <path>`

What it returns:

- file overview
- file card summary/description
- folder context
- symbols
- imports
- dependents
- dependencies
- counts for the file context

Use `read` after a search result points to a concrete file.

Example:

```bash
matryoshka-rs read --repo-root /path/to/repo crates/foo/src/lib.rs
```

## `read-bundle`

Search and return a packed read context.

```bash
matryoshka-rs read-bundle "<query>" [flags]
```

Common flags:

- `--repo-root <path>`
- `--db <path>`
- `--limit <n>` default `4`
- `--related <n>` default `3`
- `--mode brief|edit|flow`
- search flags such as `--offline`, `--omlx-rerank`, `--no-late-interaction`

Modes:

- `brief`
  Smallest context. Good for orientation.

- `edit`
  More symbols and dependency context. Good before making changes.

- `flow`
  Wider dependency/import context. Good for tracing behavior.

What it does:

- runs a `read-next` planned search
- picks the top file-level hit as primary
- selects related files from nearby/top hits
- returns packed read cards for the primary and related files

Use `read-bundle` when you want one command to give an agent the next files it
should inspect.

Example:

```bash
matryoshka-rs read-bundle \
  --repo-root /path/to/repo \
  --mode edit \
  --related 4 \
  "where should I edit spend cap policy enforcement"
```

## `rebuild-semantic`

Repair or rebuild the semantic layer from persisted facts and cards.

```bash
matryoshka-rs rebuild-semantic <repo_root> [flags]
```

Common flags:

- `--db <path>`
- `--offline`
- `--embedding-model <model>`
- `--progress-jsonl`

What it does:

- loads persisted files, folders, symbols, and cards from SQLite
- rebuilds semantic records
- re-embeds records
- rebuilds FTS records
- rebuilds late-interaction vectors

Use this when:

- search returns empty/poor results but cards/facts exist
- embedding timed out during a prior run
- you changed retrieval/schema logic and want to avoid full enrichment

## Good Query Patterns

Symbol lookup:

```bash
matryoshka-rs op find-symbol "CredentialStore"
matryoshka-rs search "where is resolve_import defined"
```

Behavior lookup:

```bash
matryoshka-rs op find-behavior "OAuth device authorization flow"
matryoshka-rs search "how does provider import from codex and claude code work"
```

Edit target:

```bash
matryoshka-rs op edit-target "rate limit policy enforcement"
```

Dependency/blast radius:

```bash
matryoshka-rs op trace-dependency "token refresh credentials"
```

Architecture:

```bash
matryoshka-rs op architecture "repository overview"
```

Tests:

```bash
matryoshka-rs op tests-for "parser rust impl methods"
```

## Useful Environment Overrides

- `MATRYOSHKA_ENRICH_CONCURRENCY`
  Controls parallel MLX chat enrichment concurrency. Default: `6`.

- `MATRYOSHKA_EMBED_BATCH`
  Controls embedding batch size for semantic-record embedding requests.
  Default: `64`.

Tune these for your local MLX server and repo size.
