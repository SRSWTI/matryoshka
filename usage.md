# Matryoshka Usage

This is the practical command guide for the current `cradle-embed` retrieval/indexing flow.

It covers:

- first-time setup / indexing
- normal update flow after edits, additions, or deletes
- when to use `prepare`, `update`, `index`, and `rebuild-semantic`
- live progress through `--progress-jsonl` and `.matryoshka/state/progress.json`
- search modes and expected result shapes
- read commands, including `read --chunks` and `read-bundle`

---

## Local Assumptions Used Below

```text
repo:      /Users/rohit/cradle-embed
db:        /Users/rohit/cradle-embed/.matryoshka/matryoshka.db
binary:    /Users/rohit/cradle-embed/target/debug/matryoshka-rs
oMLX URL:  http://127.0.0.1:44449
api key:   2508
chat LLM:  MercuriusDream--Qwen3.5-4B-MLX-mxfp8
chunk LLM: srswti--bodega-raptor-90m
embedder:  mlx-community--embeddinggemma-300m-bf16
```

Build the local CLI after code changes:

```bash
cd /Users/rohit/cradle-embed
cargo build --workspace
```

---

## Start oMLX

```bash
cd /Users/rohit/cradle-mlx/helpers/omlx
source .venv/bin/activate

jesco-apple serve \
  --host 127.0.0.1 \
  --port 44449 \
  --api-key 2508 \
  --max-concurrent-requests 6
```

oMLX is used for:

- file/folder/repo summaries through the chat model (`--model`)
- function/class/method chunk summaries through `--chunk-summary-model`
- dense embeddings through `--embedding-model`

Important model distinction:

```text
--model                 file/folder/repo summaries/cards
--chunk-summary-model   function/class/method chunk summaries
--embedding-model       dense retrieval vectors
```

So seeing oMLX load the 4B chat model is expected if `prepare` or `index` needs file/folder/repo summaries, even when chunk summaries use the 90M model.

---

## Recommended High-Level Flow

Use `prepare` as the default command for IDE/API integration and normal use.

```text
prepare
  -> decides whether to index, update, repair missing summaries, rebuild search, and prewarm
```

Use `update` when you explicitly want an incremental refresh and do not need the extra `prepare` readiness decisions/prewarm.

Use `index` only when you intentionally want a full parser/indexer pass outside the `prepare` readiness flow.

Use `rebuild-semantic` when stored artifacts/cards/chunks already exist but search records/FTS/dense embeddings need to be rebuilt.

---

## First-Time Setup / First-Time Indexing

For first-time use, run `prepare`, not `index` directly. If the DB has no indexed files, `prepare` chooses `action: "index"` automatically.

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs prepare \
  /Users/rohit/cradle-embed \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --model MercuriusDream--Qwen3.5-4B-MLX-mxfp8 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --chunk-summary-model srswti--bodega-raptor-90m \
  --chunk-summary-concurrency 6 \
  --enable-dense \
  --ignore target \
  --ignore .git \
  --ignore .matryoshka \
  --progress-jsonl
```

What this does on a fresh DB:

1. discovers files
2. parses files, symbols, imports, and code chunks
3. extracts docstrings/doc-comments for chunks when available
4. sends undocumented chunks to the chunk-summary model
5. creates file/folder/repo cards
6. builds semantic records for files, folders, repo, symbols, snippets, and code chunks
7. builds SQLite FTS records
8. writes dense embeddings and late-interaction vectors when dense is enabled
9. prewarms initial search results
10. writes readiness state/marker if everything is ready

Expected `prepare --progress-jsonl` events include raw events and canonical UX events. For UI, prefer the canonical events:

```jsonl
{"event":"progress_state","state":{"operation":"prepare","action":"index","phase":"discovering_files","message":"Looking through the project"}}
{"event":"progress_state","state":{"operation":"prepare","action":"index","phase":"enriching_chunks","message":"Understanding code","items_done":12,"items_total":28,"item_label":"batches"}}
```

The latest canonical progress state is also written to:

```text
/Users/rohit/cradle-embed/.matryoshka/state/progress.json
```

---

## Rare: Explicit Full Index Command

You usually do **not** need this. Use raw `index` only when you intentionally want a full parser/indexer pass instead of `prepare`'s readiness decisions, repair checks, and prewarm step.

For first-time setup, `prepare` is still preferred because it automatically chooses `action: "index"` on an empty DB and then verifies readiness.

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs index \
  /Users/rohit/cradle-embed \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --model MercuriusDream--Qwen3.5-4B-MLX-mxfp8 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --chunk-summary-model srswti--bodega-raptor-90m \
  --chunk-summary-concurrency 6 \
  --enable-dense \
  --ignore target \
  --ignore .git \
  --ignore .matryoshka \
  --progress-jsonl
```

Use `prepare` for normal IDE/API readiness; use `update` or `prepare` for normal source edits.

---

## Normal Edits: Modified, Added, or Deleted Files

After editing, adding, or deleting files, run `prepare` again:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs prepare \
  /Users/rohit/cradle-embed \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --model MercuriusDream--Qwen3.5-4B-MLX-mxfp8 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --chunk-summary-model srswti--bodega-raptor-90m \
  --chunk-summary-concurrency 6 \
  --enable-dense \
  --ignore target \
  --ignore .git \
  --ignore .matryoshka \
  --progress-jsonl
```

If the DB already exists and is mostly healthy, `prepare` usually chooses:

```json
{
  "operation": "prepare",
  "action": "update"
}
```

### What update handles automatically

`update` / `prepare(action=update)` handles:

- changed files, detected by `source_hash`
- newly added files
- deleted files
- changed/added/deleted folders
- stale file/folder/repo cards caused by those changes
- stale raw semantic records for changed files
- stale card semantic records for changed files/folders
- stale code chunks for changed files
- dense embeddings/FTS for refreshed semantic records
- import/context neighbors when relationships change

### Deleted files

If a source file is removed, update removes stale data for that file from the index, including relevant cards/chunks/search records. `prepare` also prunes orphaned artifacts before/after the readiness flow.

### Modified files and chunk summaries

Yes: changed-file chunk summaries are handled by `update` already.

More specifically:

- the parser re-extracts chunks for changed files
- chunks with useful docstrings/doc-comments use those docs directly
- undocumented changed chunks are sent to the chunk-summary model
- unchanged chunks preserve their existing generated summaries
- files whose `source_hash` is unchanged are skipped for chunk summarization
- related/import-neighbor files may be refreshed when context changed

So for normal code edits, do **not** manually rebuild all summaries. Run `prepare` or `update`.

---

## Explicit Incremental Update Command

Use this if you want just the incremental index refresh, without the full `prepare` decision/prewarm layer:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs update \
  /Users/rohit/cradle-embed \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --model MercuriusDream--Qwen3.5-4B-MLX-mxfp8 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --chunk-summary-model srswti--bodega-raptor-90m \
  --chunk-summary-concurrency 6 \
  --enable-dense \
  --ignore target \
  --ignore .git \
  --ignore .matryoshka \
  --progress-jsonl
```

Expected output in normal successful update summary:

```text
changed_files: <number of changed/added files>
removed_files: <number of deleted files>
changed_folders: <number of affected folders>
repo_card_updated: true|false
```

Use `prepare` instead of raw `update` when building an IDE workflow, because `prepare` also repairs gaps, rebuilds search if needed, warms search, and writes readiness state.

---

## Repairing Missing Summaries / Gaps

If cards/chunks/search artifacts are missing or incomplete, run `prepare`.

`prepare` detects gaps and may choose:

```json
{
  "operation": "prepare",
  "action": "repair"
}
```

Important UX semantics:

- top-level `operation` stays `prepare`
- internal sub-step is `action: "repair"`
- `progress.json` should not show top-level `operation: "repair"`

Example progress state:

```json
{
  "operation": "prepare",
  "action": "repair",
  "status": "running",
  "phase": "enriching_chunks",
  "message": "Understanding code",
  "files_done": null,
  "files_total": null,
  "items_done": 16,
  "items_total": 24,
  "item_label": "batches"
}
```

---

## Rebuilding Search Records / Embeddings

Use `rebuild-semantic` when existing cards/chunks are good but search data is missing/stale.

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs rebuild-semantic \
  /Users/rohit/cradle-embed \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --progress-jsonl
```

What it does:

- rebuilds semantic records from stored files/cards/chunks
- rebuilds SQLite FTS
- rebuilds dense embeddings if dense is enabled

What it does **not** do:

- it does not parse changed source files as the main operation
- it does not regenerate file/folder/repo cards
- it does not perform normal changed-file chunk-summary refresh

For changed source files, prefer `prepare` or `update`.

---

## Progress UX Contract

There are two progress streams:

1. raw indexer events, useful for debugging
2. canonical `ProgressState`, useful for UI/IDE progress

For UI, prefer:

```jsonl
{"event":"progress_state","state":{...}}
```

The same latest state is written to:

```text
.matryoshka/state/progress.json
```

Canonical fields:

```json
{
  "operation": "prepare",
  "action": "update",
  "status": "running",
  "phase": "enriching_chunks",
  "message": "Understanding code",
  "percent": 0.72,
  "current_file": null,
  "files_done": null,
  "files_total": null,
  "items_done": 12,
  "items_total": 28,
  "item_label": "batches"
}
```

Meanings:

| Field | Meaning |
|---|---|
| `operation` | user-facing command, e.g. `prepare`, `update`, `index`, `rebuild-semantic` |
| `action` | prepare sub-step, e.g. `index`, `update`, `repair`, `rebuild_search`, `prepare_results` |
| `phase` | user-facing current phase |
| `message` | short UI copy |
| `files_done/files_total` | only file progress |
| `items_done/items_total/item_label` | non-file progress, e.g. `chunks`, `batches`, `records`, `queries` |

Common phases/messages:

| Phase | Message |
|---|---|
| `starting` | `Getting ready` |
| `discovering_files` | `Looking through the project` |
| `reading_files` | `Reading code structure` |
| `enriching_files` | `Understanding files` |
| `enriching_chunks` | `Understanding code` |
| `saving` | `Saving updates` |
| `embedding` | `Preparing search` |
| `embedding_skipped` | `Preparing text search` |
| `checking` | `Checking everything` |
| `warming_search` | `Warming search` |
| `complete` | `Ready` |
| `failed` | `Needs attention` |
| `cancelled` | `Cancelled` |

---

## Search Commands

### Default file-collapsed search

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "where is progress state written" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --limit 8
```

Expect file-level results. This is best for “what file owns this behavior?”

### Compact output

Add:

```bash
--compact
```

`--compact` hides noisy debug fields like `matched_terms`, `why_matched`, and `total_matched_symbols`, while keeping useful fields like `matched_symbols`.

### Chunk/function-level search

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "which functions are sent to the LLM for chunk summaries and which are skipped because docs exist" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --result-granularity chunk \
  --compact \
  --limit 5
```

Expect `entity_type: "code_chunk"` records with:

```json
{
  "path": "crates/indexer/src/indexer.rs",
  "matched_symbols": ["FullIndexer::refresh_chunk_summaries"],
  "summary": "Summarize code chunks that have no useful docstring/doc comment..."
}
```

Use this when you want function/class/method-level answers.

### Symbol-only search

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "where is ProgressState defined" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --result-granularity symbol \
  --compact \
  --limit 5
```

Expect `entity_type: "symbol"` records.

### Raw record / no-collapse search

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "where is resolve_search_result_granularity defined" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --no-collapse \
  --compact \
  --limit 5
```

`--no-collapse` is equivalent to:

```bash
--result-granularity record
```

Do not combine `--no-collapse` with `--result-granularity chunk` or `symbol`; choose one result granularity selector.

### Dense disabled search

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "where is ProgressState defined" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --disable-dense \
  --no-dense-fallback \
  --result-granularity symbol \
  --compact \
  --limit 5
```

This uses exact/FTS-style retrieval and avoids query embedding calls.

---

## Read Commands

### Direct file read

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs read crates/api/src/lib.rs \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --repo-root /Users/rohit/cradle-embed
```

Expect:

- file metadata
- file summary
- description
- folder overview
- symbols
- imports
- dependency summaries when available

This is best after search tells you which file matters.

### Direct file read with function/class/method chunk summaries

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs read crates/api/src/lib.rs \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --repo-root /Users/rohit/cradle-embed \
  --chunks
```

Alias:

```bash
--include-chunks
```

Expect an extra `chunks` array:

```json
{
  "chunk_id": "crates/api/src/lib.rs::task_query:2051",
  "symbol": "task_query",
  "qualified_name": "task_query",
  "kind": "function",
  "signature": "fn task_query(task: AgentTask, query: &str) -> String",
  "lines": "2051-2065",
  "summary_source": "llm",
  "summary": "The function processes agent tasks..."
}
```

`generated_summary` is intentionally not included because it duplicates `summary` for LLM-generated chunks. `summary_source` keeps provenance.

Useful compact view:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs read crates/api/src/lib.rs \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --repo-root /Users/rohit/cradle-embed \
  --chunks \
  | jq '{file: .file.file_id, chunk_count: (.chunks | length), chunks: [.chunks[:8][] | {name: (.qualified_name // .symbol), kind, lines, summary_source, summary}]}'
```

### Read bundle: search first, then pack context

`read-bundle` searches for a query, picks a primary file hit, selects related files, then returns packed file cards.

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs read-bundle \
  "progress state operation action item_label batches chunks prepare progress json" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --repo-root /Users/rohit/cradle-embed \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --mode brief \
  --limit 5 \
  --related 3
```

Expected result:

```text
primary: crates/api/src/lib.rs
related: crates/cli/src/main.rs
```

Modes:

| Mode | Use for | Shape |
|---|---|---|
| `brief` | quick agent context | fewer symbols/dependencies, no description |
| `edit` | code editing | more symbols/dependencies, includes description |
| `flow` | tracing relationships | more dependency context, includes description |

Example chunk-summary bundle:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs read-bundle \
  "which functions are sent to the LLM for chunk summaries and which are skipped because docs exist" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --repo-root /Users/rohit/cradle-embed \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --mode edit \
  --limit 5 \
  --related 3
```

Expected primary area:

```text
crates/indexer/src/indexer.rs
FullIndexer::refresh_chunk_summaries
```

Caveat: current `read-bundle` chooses good primary files, but packed symbol lists are still source-order limited. For large files, the most query-relevant symbol/chunk may be omitted from the visible packed symbol list. If you need function-level summaries, use search `--result-granularity chunk` or direct `read --chunks` after identifying the file.

---

## Useful Stress-Test Queries

### Prepare lifecycle

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "prepare lifecycle add changed deleted files repair missing summaries rebuild search data ready marker" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --result-granularity chunk \
  --compact \
  --limit 5
```

Expected areas:

```text
crates/api/src/lib.rs::Matryoshka::prepare_with_progress_and_cancel
crates/api/tests/facade.rs::prepare_search_read_and_repair_lifecycle_work_through_rust_api
```

### Progress UX

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "progress state operation action phase message items_done item_label batches chunks prepare repair progress json" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --result-granularity chunk \
  --compact \
  --limit 5
```

Expected areas:

```text
ProgressState
indexer_progress_state
CliProgressStateWriter::state_with_counters
assert_progress_events_are_consistent
```

### Parser/chunks/docstrings

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "how does parser extract functions classes methods code chunks and attach doc comments or docstrings" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --result-granularity chunk \
  --compact \
  --limit 5
```

Expected areas:

```text
build_code_chunks
doc_summary_source
extract_python_docstring
extract_typescript_doc
```

### Deleted files / stale artifacts

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "deleted files remove stale semantic records orphaned file cards prune artifacts incremental update" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --result-granularity chunk \
  --compact \
  --limit 5
```

Expected areas:

```text
FullIndexer::refresh_artifacts
FullIndexer::update_repo_with_progress
MatryoshkaStore::prune_orphaned_artifacts
apply_structural_delta
```

---

## Inspecting the DB

Check semantic record types:

```bash
sqlite3 /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  "SELECT entity_type, COUNT(*) FROM semantic_records GROUP BY entity_type ORDER BY entity_type;"
```

Check dense/FTS/late-vector coverage:

```bash
sqlite3 /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  "SELECT 'semantic_records', COUNT(*) FROM semantic_records UNION ALL SELECT 'embedded_records', SUM(CASE WHEN json_extract(payload_json, '$.embedding') IS NOT NULL THEN 1 ELSE 0 END) FROM semantic_records UNION ALL SELECT 'late_vector_rows', COUNT(*) FROM semantic_late_vectors UNION ALL SELECT 'records_with_late_vectors', COUNT(DISTINCT record_id) FROM semantic_late_vectors UNION ALL SELECT 'fts_records', COUNT(*) FROM semantic_records_fts;"
```

Check chunk summary sources:

```bash
sqlite3 /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  "SELECT json_extract(payload_json, '$.summary_source') AS source, COUNT(*) FROM code_chunks GROUP BY source ORDER BY source;"
```

---

## Notes and Caveats

1. `prepare` is the safest default command for IDE integration.
2. `update` handles changed/added/deleted source files and changed-file chunk summaries incrementally.
3. `rebuild-semantic` rebuilds search records/FTS/embeddings from existing artifacts; use it for search-data repair, not normal source edits.
4. `read --chunks` exposes function/class/method summaries for a file; it does not include full chunk source code.
5. `read-bundle` currently packs symbols by file/source order, not by query-relevance inside the chosen file.
6. `--compact` only changes output shape. It does not change retrieval scoring.
7. `--result-granularity chunk` returns existing code chunk semantic records; run `prepare` or `update` first if code changed.
8. `--omlx-rerank` requires an oMLX `/v1/rerank` endpoint. If your server returns 404 for `/v1/rerank`, use non-reranked search or the chat reranker path.

---

## Validation Commands

```bash
cd /Users/rohit/cradle-embed
cargo fmt --all
cargo check --workspace
cargo test --workspace
cargo build --workspace
```
