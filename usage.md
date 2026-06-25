# Matryoshka Usage

This is the operational guide for the current Matryoshka CLI in `/Users/rohit/cradle-embed`.

The main architecture is now split into two lanes:

```text
prepare
  parse AST
  extract files, folders, symbols, imports, chunks
  store canonical facts
  build lexical/search records and embeddings
  mark the repo searchable

enrich
  slowly summarize files, chunks, folders, and repo
  checkpoint progress
  update derived records
  rebuild affected search records and embeddings
  report readiness and staleness
```

The important rule:

```text
Canonical DB facts are truth.
Indexes, summaries, cards, and vectors are derived.
Derived assets can be missing, stale, or building without breaking core search.
```

Use this guide when you want to know what each command takes as input, what it does, why it exists, and what output to expect.

## Local defaults used in examples

```bash
export REPO=/Users/rohit/cradle-embed
export DB=/Users/rohit/cradle-embed/.matryoshka/matryoshka.db
export BIN=/Users/rohit/cradle-embed/target/debug/matryoshka-rs
export BASE_URL=http://127.0.0.1:44449
export API_KEY=2508
export CHAT_MODEL=MercuriusDream--Qwen3.5-4B-MLX-mxfp8
export EMBED_MODEL=mlx-community--embeddinggemma-300m-bf16
export RERANK_MODEL=mlx-community--Qwen3-Reranker-0.6B-mxfp8
export CHUNK_MODEL=srswti--bodega-raptor-90m
```

Build the CLI:

```bash
cd /Users/rohit/cradle-embed
cargo build --workspace
```

Default database path if `--db` is omitted:

```text
<repo>/.matryoshka/matryoshka.db
```

Useful files written beside the DB:

```text
.matryoshka/matryoshka.db
.matryoshka/.jesco-prewarm-complete
.matryoshka/state/progress.json
.matryoshka/logs/*.jsonl
.matryoshka/watch.pid
```

## Mental model

`prepare` should be the first command.

It should make the repo searchable quickly. It should not require full LLM summarization unless you explicitly pass `--enrich-now`.

`enrich` should be the slow/background command.

It works in bounded batches. It can be run every few minutes, every hour, or manually. It improves cards, summaries, chunk descriptions, folder descriptions, repo overview, and derived semantic records.

`search`, `read`, and `read-bundle` should work after core prepare even if enrichment is partial.

If enrichment is partial, expect some summaries to be missing, doc-derived, raw fallback, or stale. That is not automatically a failure.

## Common retrieval flags

These flags appear on many commands.

| Flag | Input | Why it exists | Expected effect |
|---|---|---|---|
| `--retrieval-primary hybrid` | `fts`, `splade`, `dense`, or `hybrid` | Chooses main retrieval strategy | Hybrid balances lexical and semantic search |
| `--enable-dense` | boolean flag | Builds/uses embedding vectors | Better semantic matches, needs embedder endpoint |
| `--disable-dense` | boolean flag | Avoids embedding calls | Faster/offline lexical search |
| `--dense-fallback` | boolean flag | Allows dense fallback when primary misses | More recall |
| `--no-dense-fallback` | boolean flag | Avoids fallback embedding work | More deterministic lexical behavior |
| `--no-late-interaction` | boolean flag | Disables late vector scoring | Faster search, less ranking quality |
| `--embedding-model` | model id | Chooses embedding model | Used for semantic records and query vectors |
| `--base-url` | URL | oMLX/OpenAI-compatible endpoint | Used by embedding, chat, and rerank clients |
| `--api-key` | string | API auth token | Sent to the endpoint |

Conflict rules:

```text
--enable-dense conflicts with --disable-dense
--dense-fallback conflicts with --no-dense-fallback
--retrieval-primary dense requires dense embeddings
--dense-fallback requires dense embeddings
```

## Common model flags

| Flag | Used by | Meaning |
|---|---|---|
| `--model` | prepare with `--enrich-now`, raw index/update, enrich | Chat model for file/folder/repo summaries |
| `--chat-model` | alias for `--model` | Same as above |
| `--chunk-summary-model` | prepare with `--enrich-now`, raw index/update, enrich | Smaller model for function/class/method chunk summaries |
| `--chunk-summary-concurrency` | chunk summarizer | Number of concurrent chunk summary requests |
| `--no-chunk-summaries` | prepare/index/update/enrich | Disables generated chunk summaries |

## Recommended first run

Use `prepare` first.

```bash
"$BIN" prepare "$REPO" \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --model "$CHAT_MODEL" \
  --chunk-summary-model "$CHUNK_MODEL" \
  --enable-dense \
  --ignore target \
  --ignore .git \
  --ignore .matryoshka \
  --json
```

What it takes as input:

| Input | Required | Meaning |
|---|---|---|
| `repo_root` | yes | Repository to parse and index |
| `--db` | no | SQLite DB path |
| `--base-url` | no | Endpoint for embeddings/prewarm queries |
| `--api-key` | no | Endpoint key |
| `--embedding-model` | no | Embedding model |
| `--ignore` | repeatable | Paths to exclude |
| `--enable-dense` | no | Enables vector embeddings |

What it does by default:

```text
1. Creates .matryoshka layout.
2. Parses the repo.
3. Extracts files, folders, symbols, imports, and chunks.
4. Stores canonical facts in SQLite.
5. Builds semantic records.
6. Builds FTS search records.
7. Builds dense embeddings if dense is enabled.
8. Writes progress state.
9. Runs small search prewarm queries.
10. Writes readiness marker when core search is usable.
11. Reports enrichment readiness separately.
```

What it does not do by default:

```text
It does not run full LLM file/folder/repo enrichment.
It does not require all file cards to have summaries.
It does not require all chunks to have LLM summaries.
```

Why this command exists:

```text
prepare is the safe default for IDEs and agents.
It makes search/read usable without forcing expensive full LLM summarization up front.
```

Expected JSON output shape:

```json
{
  "status": "ready",
  "repo_root": "/Users/rohit/cradle-embed",
  "db": "/Users/rohit/cradle-embed/.matryoshka/matryoshka.db",
  "ready_marker": "/Users/rohit/cradle-embed/.matryoshka/.jesco-prewarm-complete",
  "logs": "/Users/rohit/cradle-embed/.matryoshka/logs",
  "actions_taken": ["index", "prewarm"],
  "project_map": {
    "status": "needs_attention",
    "files": 34,
    "folders": 34,
    "symbols": 900,
    "cards": {
      "file": 0,
      "folder": 0,
      "repo": 0,
      "missing_text": 68
    }
  },
  "enrichment": {
    "status": "partial",
    "files_total": 34,
    "file_cards_ready": 0,
    "file_cards_pending": 34,
    "chunks_total": 973,
    "chunks_ready": 14,
    "chunks_pending": 959,
    "derived_semantic_records_ready": 14,
    "derived_semantic_records_pending": 0,
    "derived_semantic_records_stale": 0,
    "repo_card_ready": false
  },
  "search": {
    "status": "ready",
    "semantic_records": 1200,
    "embedded_records": 1200,
    "fts_records": 1200,
    "late_vector_rows": 5000,
    "records_with_late_vectors": 1200
  },
  "changes": {
    "changed_files": 34,
    "removed_files": 0,
    "changed_folders": 34,
    "repo_card_updated": false
  },
  "prepare_results": {
    "fts_records": 1200,
    "query_count": 4,
    "warmed_hits": 20
  },
  "embedding_model": "mlx-community--embeddinggemma-300m-bf16"
}
```

Important interpretation:

```text
search.status == ready means search/read can run.
enrichment.status == partial means summaries are still building.
project_map.status may still say needs_attention during this transition because old card-gap reporting is still present.
Do not treat missing file/folder/repo summaries as core prepare failure.
```

## Prepare with progress JSONL

```bash
"$BIN" prepare "$REPO" \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --enable-dense \
  --progress-jsonl
```

What it takes as input:

```text
Same as prepare, plus --progress-jsonl.
```

What it does:

```text
Runs prepare and streams each progress event as one JSON object per line.
```

Why it exists:

```text
Use it for UI progress, logs, and debugging long runs.
```

Expected output:

```jsonl
{"event":"progress_state","state":{"operation":"prepare","status":"running","phase":"discovering_files","message":"Looking through the project"}}
{"event":"progress_state","state":{"operation":"prepare","status":"running","phase":"reading_files","message":"Reading code structure"}}
{"event":"progress_state","state":{"operation":"prepare","status":"completed","phase":"complete","message":"Ready"}}
```

The latest state is also written to:

```text
.matryoshka/state/progress.json
```

## Prepare with immediate enrichment

```bash
"$BIN" prepare "$REPO" \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --model "$CHAT_MODEL" \
  --chunk-summary-model "$CHUNK_MODEL" \
  --chunk-summary-concurrency 6 \
  --enable-dense \
  --enrich-now \
  --json
```

What it takes as input:

```text
Same as prepare, plus --enrich-now and chat/chunk model options.
```

What it does:

```text
Runs core prepare and enables LLM enrichment during that run.
```

Why it exists:

```text
Use it only when you intentionally want old-style blocking enrichment during prepare.
For normal IDE use, do not pass --enrich-now.
```

Expected output:

```text
Slower prepare.
More file cards, folder cards, repo card, and chunk summaries ready by the time the command exits.
```

## Enrichment status

```bash
"$BIN" enrich "$REPO" \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --model "$CHAT_MODEL" \
  --chunk-summary-model "$CHUNK_MODEL" \
  --status \
  --json
```

What it takes as input:

| Input | Required | Meaning |
|---|---|---|
| `repo_root` | yes | Repo whose enrichment readiness should be checked |
| `--db` | no | DB to inspect |
| `--status` | yes for status mode | Do not enrich, only report readiness |
| `--json` | no | Emit machine-readable report |

What it does:

```text
Reads enrichment readiness from the DB.
Does not call the LLM.
Does not modify records.
```

Why it exists:

```text
Use it to decide what is ready, pending, or stale before scheduling background enrichment.
```

Expected output:

```json
{
  "status": "partial",
  "files_total": 34,
  "file_cards_ready": 0,
  "file_cards_pending": 34,
  "file_cards_stale": 0,
  "folders_total": 34,
  "folder_cards_ready": 0,
  "folder_cards_pending": 34,
  "chunks_total": 973,
  "chunks_ready": 14,
  "chunks_pending": 959,
  "derived_semantic_records_ready": 14,
  "derived_semantic_records_pending": 0,
  "derived_semantic_records_stale": 0,
  "repo_card_ready": false,
  "repo_card_pending": true,
  "pending_files_sample": ["crates/api/src/lib.rs"],
  "pending_folders_sample": ["crates/api"],
  "pending_chunks_sample": ["crates/api/src/lib.rs::Matryoshka::prepare:100"]
}
```

How to interpret it:

```text
status=ready means enrichment is complete.
status=partial means some derived summaries/cards are missing.
status=stale means source facts changed and derived summaries need refresh.
status=pending means enrichment has not been built yet.
Search can still be ready while enrichment is partial.
```

## Enrich one batch

```bash
"$BIN" enrich "$REPO" \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --model "$CHAT_MODEL" \
  --chunk-summary-model "$CHUNK_MODEL" \
  --chunk-summary-concurrency 6 \
  --max-files 1 \
  --json
```

What it takes as input:

| Input | Required | Meaning |
|---|---|---|
| `repo_root` | yes | Repo to enrich |
| `--db` | no | DB to update |
| `--max-files` | no | Maximum file-card units to enrich in this batch |
| `--model` | no | Chat model for file/folder/repo summaries |
| `--chunk-summary-model` | no | Model for chunk summaries |
| `--chunk-summary-concurrency` | no | Chunk request concurrency |
| `--no-chunk-summaries` | no | Skip generated chunk summaries |

What it does:

```text
1. Checks current enrichment readiness.
2. Selects a bounded amount of pending/stale work.
3. Summarizes selected files/chunks/folders/repo as needed.
4. Writes file cards, folder cards, repo card, and chunk summaries.
5. Rebuilds affected semantic records.
6. Rebuilds affected FTS/embedding records.
7. Writes progress state.
8. Returns before/after readiness.
```

Why it exists:

```text
This is the background command.
Run it repeatedly instead of doing expensive LLM enrichment inside prepare.
```

Expected JSON output shape:

```json
{
  "selected_files": 1,
  "selected_folders": 4,
  "repo_card_updated": false,
  "before": {
    "status": "partial",
    "file_cards_ready": 0,
    "file_cards_pending": 34
  },
  "after": {
    "status": "partial",
    "file_cards_ready": 1,
    "file_cards_pending": 33
  },
  "semantic_record_count": 1215,
  "artifact_quality": {
    "file_cards_with_summary": 1,
    "file_cards": 34
  },
  "retrieval_index": {
    "semantic_records": 1215,
    "embedded_records": 1215,
    "fts_records": 1215
  },
  "embedding_model": "mlx-community--embeddinggemma-300m-bf16"
}
```

Recommended cadence:

```bash
# Manual one-file batch
"$BIN" enrich "$REPO" --db "$DB" --base-url "$BASE_URL" --api-key "$API_KEY" --embedding-model "$EMBED_MODEL" --model "$CHAT_MODEL" --max-files 1 --json

# Bigger background batch
"$BIN" enrich "$REPO" --db "$DB" --base-url "$BASE_URL" --api-key "$API_KEY" --embedding-model "$EMBED_MODEL" --model "$CHAT_MODEL" --max-files 5 --json
```

## Enrich with progress JSONL

```bash
"$BIN" enrich "$REPO" \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --model "$CHAT_MODEL" \
  --max-files 1 \
  --progress-jsonl
```

What it does:

```text
Runs one enrichment batch and prints progress events as JSONL.
```

Expected output:

```jsonl
{"event":"progress_state","state":{"operation":"enrich","status":"running","phase":"enriching_files"}}
{"event":"progress_state","state":{"operation":"enrich","status":"running","phase":"enriching_chunks"}}
{"event":"progress_state","state":{"operation":"enrich","status":"completed","phase":"complete"}}
```

## Search

```bash
"$BIN" search \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --enable-dense \
  --limit 8 \
  "where is progress state written"
```

What it takes as input:

| Input | Required | Meaning |
|---|---|---|
| `query` | yes | Natural-language or symbol query |
| `--db` | usually | DB to search |
| `--limit` | no | Number of hits |
| `--result-granularity` | no | `file`, `record`, `symbol`, or `chunk` |
| `--compact` | no | Remove verbose match diagnostics |
| `--rerank` | no | Use chat reranker |
| `--omlx-rerank` | no | Use oMLX rerank endpoint |

What it does:

```text
1. Confirms prepare marker exists.
2. Confirms retrieval index is usable.
3. Embeds the query if dense retrieval is enabled.
4. Searches semantic records.
5. Applies late interaction unless disabled.
6. Optionally reranks.
7. Prints JSON hits.
```

Why it exists:

```text
Use it to find relevant files, symbols, chunks, or records before reading source.
```

Expected output:

```json
[
  {
    "path": "crates/api/src/lib.rs",
    "score": 12.34,
    "entity_type": "file_card",
    "summary": "API facade for prepare, search, read, and enrichment.",
    "matched_symbols": ["Matryoshka::prepare_with_progress"]
  }
]
```

If prepare has not run:

```text
Matryoshka prepare is not ready for <db>; run prepare first
```

## Search result granularity

File-level search:

```bash
"$BIN" search --db "$DB" --base-url "$BASE_URL" --api-key "$API_KEY" --embedding-model "$EMBED_MODEL" --enable-dense --result-granularity file "background enrichment status"
```

What to expect:

```text
One collapsed result per file.
Best when deciding which file to inspect.
```

Record-level search:

```bash
"$BIN" search --db "$DB" --base-url "$BASE_URL" --api-key "$API_KEY" --embedding-model "$EMBED_MODEL" --enable-dense --result-granularity record "background enrichment status"
```

What to expect:

```text
Raw semantic records, not collapsed by file.
Best for debugging ranking.
```

Symbol-level search:

```bash
"$BIN" search --db "$DB" --base-url "$BASE_URL" --api-key "$API_KEY" --embedding-model "$EMBED_MODEL" --enable-dense --result-granularity symbol "where is EnrichmentReadinessReport defined" --compact
```

What to expect:

```text
Records whose entity_type is symbol.
Best when looking for definitions and APIs.
```

Chunk-level search:

```bash
"$BIN" search --db "$DB" --base-url "$BASE_URL" --api-key "$API_KEY" --embedding-model "$EMBED_MODEL" --enable-dense --result-granularity chunk "how does enrich_once select files" --compact
```

What to expect:

```text
Function/class/method-level code_chunk results.
Best when asking behavior questions.
If enrichment is partial, some chunk summaries may be doc-derived or missing.
```

No-collapse alias:

```bash
"$BIN" search --db "$DB" --no-collapse "query"
```

Meaning:

```text
Equivalent to --result-granularity record.
Do not combine --no-collapse with --result-granularity chunk or symbol.
```

## Search reranking

Chat reranker:

```bash
"$BIN" search \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --enable-dense \
  --rerank \
  --rerank-model "$CHAT_MODEL" \
  "where should I edit background enrichment scheduling"
```

oMLX reranker:

```bash
"$BIN" search \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --enable-dense \
  --omlx-rerank \
  --omlx-rerank-model "$RERANK_MODEL" \
  --omlx-rerank-candidates 20 \
  "where should I edit background enrichment scheduling"
```

Rules:

```text
Use either --rerank or --omlx-rerank, not both.
Reranking improves precision but adds endpoint latency.
If /v1/rerank is unavailable, --omlx-rerank fails.
```

## Agent task search: op

```bash
"$BIN" op find-symbol \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --enable-dense \
  --compact \
  "EnrichmentReadinessReport"
```

What it takes as input:

| Input | Required | Meaning |
|---|---|---|
| `task` | yes | One of the agent task types |
| `query` | yes | User goal/query |
| search flags | no | Same as `search` |

Available tasks:

| Task | Query rewrite purpose |
|---|---|
| `find-symbol` | Find definitions and usages |
| `find-behavior` | Find logic/responsibility |
| `edit-target` | Find where to change code |
| `trace-dependency` | Find upstream/downstream impact |
| `architecture` | Find subsystem overview |
| `tests-for` | Find test coverage/fixtures |
| `read-next` | Find what to read before editing |

What it does:

```text
Rewrites your query with task-specific intent, then runs search.
```

Why it exists:

```text
Use it for agent workflows where the same user words need different retrieval intent.
```

Expected output:

```text
Same JSON hit shape as search.
```

Example edit-target query:

```bash
"$BIN" op edit-target --db "$DB" --base-url "$BASE_URL" --api-key "$API_KEY" --embedding-model "$EMBED_MODEL" --enable-dense --compact "add background enrichment scheduler"
```

## Read one file

```bash
"$BIN" read crates/api/src/lib.rs \
  --db "$DB" \
  --repo-root "$REPO"
```

What it takes as input:

| Input | Required | Meaning |
|---|---|---|
| `file` | yes | Repo-relative file path |
| `--db` | no | DB to read from |
| `--repo-root` | no | Repo root for path resolution |
| `--chunks` | no | Include chunk details instead of top-level symbol outline |
| `--json` | no | Emit legacy full symbol objects |

What it does:

```text
Reads stored DB facts and cards for one file.
Does not call MLX.
Does not read live source from disk as the source of truth.
Requires core prepare readiness.
```

Why it exists:

```text
Use it after search identifies a relevant file.
It gives compact context without loading the entire source file.
```

Expected default output:

```json
{
  "file": {
    "file_id": "crates/api/src/lib.rs",
    "path": "crates/api/src/lib.rs",
    "language": "rust",
    "line_count": 2000
  },
  "summary": "API facade for Matryoshka.",
  "description": "...",
  "folder": {
    "path": "crates/api/src",
    "summary": "..."
  },
  "symbols": [
    "100-130 method Matryoshka::prepare_with_progress :: pub fn prepare_with_progress(...)"
  ],
  "imports": {
    "external": "anyhow, serde, std.path"
  }
}
```

If enrichment is partial:

```text
summary may be empty, doc-derived, or stale.
symbols/imports/chunks still come from canonical parser facts.
```

## Read with full legacy symbols

```bash
"$BIN" read crates/api/src/lib.rs \
  --db "$DB" \
  --repo-root "$REPO" \
  --json
```

What it does:

```text
Returns full symbol objects instead of compact symbol strings.
```

Expected symbol shape:

```json
{
  "symbols": [
    {
      "name": "prepare_with_progress",
      "qualified_name": "Matryoshka::prepare_with_progress",
      "kind": "method",
      "signature": "pub fn prepare_with_progress(...) -> Result<PrepareSummary>",
      "lines": "100-130"
    }
  ]
}
```

Why it exists:

```text
Use it when a program needs structured symbol fields.
For LLM context, default compact read is usually lower-token.
```

## Read with chunks

```bash
"$BIN" read crates/api/src/lib.rs \
  --db "$DB" \
  --repo-root "$REPO" \
  --chunks
```

Alias:

```bash
--include-chunks
```

What it does:

```text
Returns function/class/method chunks for the file.
Suppresses the top-level symbols array because chunks carry symbol-level detail.
```

Expected chunk shape:

```json
{
  "chunks": [
    {
      "chunk_id": "crates/api/src/lib.rs::Matryoshka::prepare_with_progress:100",
      "symbol": "prepare_with_progress",
      "qualified_name": "Matryoshka::prepare_with_progress",
      "kind": "Method",
      "signature": "pub fn prepare_with_progress(...) -> Result<PrepareSummary>",
      "lines": "100-130",
      "summary_source": "llm",
      "summary": "Runs prepare and emits progress events."
    }
  ]
}
```

Summary source meanings:

| Source | Meaning | Trust level |
|---|---|---|
| `llm` | Generated by enrichment | Helpful, still verify for critical edits |
| `doc_comment` | From Rust/TS/etc docs | Usually reliable but may be stale with code |
| `docstring` | From Python-style docstring | Usually reliable but may be stale with code |
| `empty` | No summary yet | Read code or wait for enrichment |

## Read bundle

```bash
"$BIN" read-bundle \
  "where is background enrichment status calculated" \
  --db "$DB" \
  --repo-root "$REPO" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --enable-dense \
  --mode brief \
  --limit 5 \
  --related 3
```

What it takes as input:

| Input | Required | Meaning |
|---|---|---|
| `query` | yes | Search query |
| `--limit` | no | Search hits considered |
| `--related` | no | Related files to include |
| `--mode` | no | `brief`, `edit`, or `flow` |
| search flags | no | Same as `search` |

What it does:

```text
1. Rewrites the query as read-next intent.
2. Searches for relevant files.
3. Chooses the primary file.
4. Selects nearby/related files.
5. Packs read cards into one JSON bundle.
```

Why it exists:

```text
Use it when an agent needs a small context pack before editing.
```

Modes:

| Mode | Use for | Expected shape |
|---|---|---|
| `brief` | quick orientation | Smaller cards |
| `edit` | preparing to patch code | More symbols and descriptions |
| `flow` | dependency tracing | More relationship context |

Expected output:

```json
{
  "primary": {
    "file": {
      "file_id": "crates/api/src/lib.rs"
    },
    "summary": "..."
  },
  "related": [
    {
      "file": {
        "file_id": "crates/indexer/src/lib.rs"
      },
      "summary": "..."
    }
  ]
}
```

Known behavior:

```text
read-bundle chooses good files, but packed symbols may be source-order limited.
If you need exact function-level context, follow with search --result-granularity chunk or read --chunks.
```

## Cards

```bash
"$BIN" cards --db "$DB"
```

What it takes as input:

| Input | Required | Meaning |
|---|---|---|
| `--db` | no | DB to inspect |
| `--empty` | no | Only show cards with empty summaries |
| `--json` | no | Emit JSON rows |

What it does:

```text
Lists stored file/folder/repo card summaries.
Does not call MLX.
Does not repair missing summaries.
```

Why it exists:

```text
Use it to inspect enrichment coverage and find missing card text.
```

Expected Markdown output:

```markdown
# Matryoshka Card Summaries

- Database: `/Users/rohit/cradle-embed/.matryoshka/matryoshka.db`
- Cards returned: 12
- File cards: 10
- Folder cards: 1
- Repo cards: 1
- Empty summaries: 0
```

JSON mode:

```bash
"$BIN" cards --db "$DB" --json
```

Expected JSON row:

```json
{
  "card_type": "file",
  "id": "crates/api/src/lib.rs",
  "summary": "API facade for Matryoshka.",
  "is_empty": false
}
```

Empty cards only:

```bash
"$BIN" cards --db "$DB" --empty
```

Use this to see what background enrichment still needs to improve.

## Chunks parser inspection

```bash
"$BIN" chunks "$REPO" \
  --ignore target \
  --ignore .git \
  --ignore .matryoshka
```

What it takes as input:

| Input | Required | Meaning |
|---|---|---|
| `repo_root` | yes | Repo to parse |
| `--ignore` | no | Excluded paths |
| `--source` | no | `all`, `doc_comment`, `docstring`, or `empty` |
| `--json` | no | Emit JSON instead of table |
| `--with-code` | no | Include full chunk source code |

What it does:

```text
Runs parser only.
Does not open or modify the DB.
Does not call embeddings.
Does not call LLMs.
```

Why it exists:

```text
Use it to debug AST chunk extraction and docstring/doc-comment detection in isolation.
```

Expected table output:

```text
chunks: 973
path                                                         symbol                                  kind       lines      source         summary
crates/api/src/lib.rs                                        Matryoshka::prepare                     Method     100-130    DocComment     Runs prepare...
```

JSON mode:

```bash
"$BIN" chunks "$REPO" --source empty --json
```

Expected JSON object:

```json
{
  "chunk_id": "crates/api/src/lib.rs::prepare:100",
  "path": "crates/api/src/lib.rs",
  "symbol": "prepare",
  "qualified_name": "Matryoshka::prepare",
  "kind": "Function",
  "signature": "fn prepare(...) -> Result<...>",
  "start_line": 100,
  "end_line": 130,
  "summary_source": "Empty",
  "summary": "",
  "doc_summary": null
}
```

With code:

```bash
"$BIN" chunks "$REPO" --source empty --json --with-code
```

Expected effect:

```text
Adds a code field containing full chunk source.
This can be large.
```

## Raw index

```bash
"$BIN" index "$REPO" \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --model "$CHAT_MODEL" \
  --chunk-summary-model "$CHUNK_MODEL" \
  --chunk-summary-concurrency 6 \
  --enable-dense \
  --ignore target \
  --ignore .git \
  --ignore .matryoshka \
  --progress-jsonl
```

What it takes as input:

```text
Repo root, DB, endpoint/model options, ignore paths, retrieval options.
```

What it does:

```text
Runs the lower-level FullIndexer full index path.
Parses repo.
Generates cards/summaries according to indexer behavior.
Builds semantic records, FTS, embeddings, and late vectors.
Can optionally start watch afterward with --watch or --watch-daemon.
```

Why it exists:

```text
Compatibility and low-level testing.
Not the preferred new architecture path for IDE startup.
```

Expected output:

```text
files: 34
folders: 34
symbols: 900
semantic_records: 1200
file_card_summaries: 34/34
folder_card_summaries: 34/34
repo_card_has_summary: true
embedded_records: 1200
fts_records: 1200
late_vector_rows: 5000
records_with_late_vectors: 1200
embedding_model: mlx-community--embeddinggemma-300m-bf16
```

Important:

```text
For the new architecture, prefer prepare plus background enrich.
Raw index can still be expensive because it uses direct FullIndexer behavior.
```

## Raw update

```bash
"$BIN" update "$REPO" \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --model "$CHAT_MODEL" \
  --chunk-summary-model "$CHUNK_MODEL" \
  --chunk-summary-concurrency 6 \
  --enable-dense \
  --ignore target \
  --ignore .git \
  --ignore .matryoshka \
  --progress-jsonl
```

What it does:

```text
Runs lower-level FullIndexer incremental update.
Detects changed, added, and removed files.
Refreshes DB facts and derived artifacts according to direct indexer behavior.
Rebuilds affected semantic records and retrieval records.
```

Why it exists:

```text
Compatibility, watcher support, and lower-level incremental indexing tests.
For new architecture IDE flow, prefer prepare for core readiness and enrich for summaries.
```

Expected output:

```text
files: 34
folders: 34
symbols: 900
semantic_records: 1200
file_card_summaries: 10/34
folder_card_summaries: 3/34
repo_card_has_summary: false
embedded_records: 1200
fts_records: 1200
late_vector_rows: 5000
records_with_late_vectors: 1200
changed_files: 2
removed_files: 0
changed_folders: 2
repo_card_updated: false
embedding_model: mlx-community--embeddinggemma-300m-bf16
```

## Watch

```bash
"$BIN" watch "$REPO" \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --model "$CHAT_MODEL" \
  --interval-ms 2000 \
  --debounce-ms 3000 \
  --enable-dense \
  --ignore target \
  --ignore .git \
  --ignore .matryoshka
```

What it takes as input:

| Input | Required | Meaning |
|---|---|---|
| `repo_root` | yes | Repo to watch |
| `--interval-ms` | no | Poll interval |
| `--debounce-ms` | no | Debounce window after changes |
| `--daemon` | no | Spawn background watcher |
| `--skip-startup-update` | no | Do not run update immediately on start |

What it does:

```text
Polls for changed, added, and removed files.
Runs raw update when a debounced change batch is detected.
Writes watch logs.
```

Why it exists:

```text
Use it for long-running local refresh loops.
In the new architecture, this should eventually become core-update plus queued enrichment, but current implementation still uses raw update.
```

Expected output:

```text
watching /Users/rohit/cradle-embed every 2000ms with 3000ms debounce
watch_log: /Users/rohit/cradle-embed/.matryoshka/logs/watch.jsonl
change batch detected: changed=1 added=0 removed=0
files: 34
changed_files: 1
```

Daemon mode:

```bash
"$BIN" watch "$REPO" --db "$DB" --base-url "$BASE_URL" --api-key "$API_KEY" --embedding-model "$EMBED_MODEL" --daemon
```

Expected output:

```text
watch_daemon_pid: 12345
watch_pid_file: /Users/rohit/cradle-embed/.matryoshka/watch.pid
watch_log: /Users/rohit/cradle-embed/.matryoshka/logs/watch.jsonl
watch_stdout_log: /Users/rohit/cradle-embed/.matryoshka/logs/watch.stdout.jsonl
```

## Prewarm search

```bash
"$BIN" prewarm \
  --repo-root "$REPO" \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --enable-dense \
  --limit 6
```

What it takes as input:

| Input | Required | Meaning |
|---|---|---|
| `--repo-root` | no | Repo root, defaults to current directory |
| `--db` | no | DB to prewarm |
| `--query` | repeatable | Custom prewarm queries |
| `--limit` | no | Hits per prewarm query |
| `--ensure-fresh` | no | Run raw update before prewarming |
| `--watch` | no | Start watch after prewarm |
| `--watch-daemon` | no | Start daemon watch after prewarm |

What it does:

```text
Runs common search queries to warm query paths/caches.
Prints retrieval index stats.
Optionally runs raw update first with --ensure-fresh.
```

Why it exists:

```text
Use it to warm search after prepare or before an agent session.
```

Expected output:

```text
fts_records: 1200
queries: 4
warmed_hits: 24
embedded_records: 1200
late_vector_rows: 5000
records_with_late_vectors: 1200
```

Custom prewarm queries:

```bash
"$BIN" prewarm --repo-root "$REPO" --db "$DB" --base-url "$BASE_URL" --api-key "$API_KEY" --embedding-model "$EMBED_MODEL" --query "prepare enrichment status" --query "read bundle search"
```

## Rebuild semantic index

```bash
"$BIN" rebuild-semantic "$REPO" \
  --db "$DB" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --enable-dense \
  --progress-jsonl
```

What it takes as input:

```text
Repo root, DB, endpoint/model options, retrieval options.
```

What it does:

```text
Rebuilds semantic_records from existing stored facts/cards/chunks.
Rebuilds FTS records.
Rebuilds dense embeddings if dense is enabled.
Does not parse changed source as the main operation.
Does not generate new file/folder/repo summaries.
```

Why it exists:

```text
Use it when search records, FTS, dense vectors, or late vectors are missing/stale, but canonical facts/cards are already present.
```

Expected output:

```text
semantic_records: 1200
file_card_records: 34
folder_card_records: 34
repo_card_records: 1
file_card_summaries: 10/34
folder_card_summaries: 3/34
repo_card_has_summary: false
embedded_records: 1200
fts_records: 1200
late_vector_rows: 5000
records_with_late_vectors: 1200
embedding_model: mlx-community--embeddinggemma-300m-bf16
```

## DB inspection commands

These are not Matryoshka CLI commands, but they are useful to understand output.

Count semantic record types:

```bash
sqlite3 "$DB" "SELECT entity_type, COUNT(*) FROM semantic_records GROUP BY entity_type ORDER BY entity_type;"
```

Expected output:

```text
code_chunk|973
file_card|34
folder_card|34
repo_card|1
symbol|900
```

Check retrieval coverage:

```bash
sqlite3 "$DB" "SELECT 'semantic_records', COUNT(*) FROM semantic_records UNION ALL SELECT 'fts_records', COUNT(*) FROM semantic_records_fts UNION ALL SELECT 'late_vector_rows', COUNT(*) FROM semantic_late_vectors UNION ALL SELECT 'records_with_late_vectors', COUNT(DISTINCT record_id) FROM semantic_late_vectors;"
```

Expected output:

```text
semantic_records|1200
fts_records|1200
late_vector_rows|5000
records_with_late_vectors|1200
```

Check chunk summary sources:

```bash
sqlite3 "$DB" "SELECT json_extract(payload_json, '$.summary_source') AS source, COUNT(*) FROM code_chunks GROUP BY source ORDER BY source;"
```

Expected output:

```text
doc_comment|14
empty|959
llm|0
```

After enrichment, expect `llm` to increase and `empty` to decrease.

## Recommended workflows

First-time searchable index:

```bash
"$BIN" prepare "$REPO" --db "$DB" --base-url "$BASE_URL" --api-key "$API_KEY" --embedding-model "$EMBED_MODEL" --enable-dense --json
```

Check enrichment readiness:

```bash
"$BIN" enrich "$REPO" --db "$DB" --base-url "$BASE_URL" --api-key "$API_KEY" --embedding-model "$EMBED_MODEL" --model "$CHAT_MODEL" --status --json
```

Run one slow background enrichment unit:

```bash
"$BIN" enrich "$REPO" --db "$DB" --base-url "$BASE_URL" --api-key "$API_KEY" --embedding-model "$EMBED_MODEL" --model "$CHAT_MODEL" --max-files 1 --json
```

Search for where to edit:

```bash
"$BIN" op edit-target --db "$DB" --base-url "$BASE_URL" --api-key "$API_KEY" --embedding-model "$EMBED_MODEL" --enable-dense --compact "add enrichment scheduler"
```

Read the target file compactly:

```bash
"$BIN" read crates/api/src/lib.rs --db "$DB" --repo-root "$REPO"
```

Read function-level summaries:

```bash
"$BIN" read crates/api/src/lib.rs --db "$DB" --repo-root "$REPO" --chunks
```

Pack context for an agent:

```bash
"$BIN" read-bundle "background enrichment readiness status" --db "$DB" --repo-root "$REPO" --base-url "$BASE_URL" --api-key "$API_KEY" --embedding-model "$EMBED_MODEL" --enable-dense --mode edit --limit 5 --related 3
```

## What to trust when output disagrees

Trust this order:

```text
1. Canonical facts: files, symbols, imports, chunks, source hashes.
2. Retrieval readiness: search.status, semantic_records, fts_records, embedded_records.
3. Enrichment readiness: file/chunk/folder/repo ready/pending/stale counts.
4. Human-readable project_map/card gap labels.
```

During the architecture transition, old `project_map.status` and `map_gaps` can make partial enrichment look like a warning.

The correct interpretation is:

```text
Search ready + enrichment partial = healthy searchable repo with background summaries still building.
Search not ready = run prepare again.
Enrichment stale = search is usable, but summaries for changed files may lag.
```

## Command selection table

| Need | Use | Reason |
|---|---|---|
| Make repo searchable | `prepare` | Fast canonical indexing and retrieval readiness |
| Force old blocking enrichment | `prepare --enrich-now` | Only when you intentionally want summaries immediately |
| Check summary/card progress | `enrich --status` | No LLM calls, read-only status |
| Slowly improve summaries | `enrich --max-files N` | Bounded background LLM work |
| Find files/symbols/chunks | `search` | Direct retrieval query |
| Search with agent intent | `op` | Rewrites query for task intent |
| Inspect one file from DB | `read` | Compact file card, symbols, imports, context |
| Inspect function/method summaries | `read --chunks` | Chunk-level view |
| Build context pack | `read-bundle` | Search plus related read cards |
| Inspect cards | `cards` | See stored summaries and empty cards |
| Inspect parser chunks only | `chunks` | No DB, no MLX, parser debugging |
| Rebuild search records | `rebuild-semantic` | Repair derived retrieval index |
| Legacy full indexing | `index` | Direct FullIndexer compatibility path |
| Legacy incremental indexing | `update` | Direct FullIndexer update path |
| Continuous legacy updates | `watch` | Poll changes and raw update |
| Warm common searches | `prewarm` | Query/cache warmup |

## Validation commands

Only run these when you want to validate code changes:

```bash
cd /Users/rohit/cradle-embed
cargo fmt --all
cargo check --workspace
cargo test --workspace
cargo build --workspace
```
