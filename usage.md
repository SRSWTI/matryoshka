# Matryoshka Usage

Matryoshka prepares a repository so Jesco can find, read, and change code with fewer steps.

The simple rule is:

```text
Call prepare.
```

`prepare` decides whether the repo needs a full build, a refresh, a repair, a search rebuild, or just a quick warm-up. The IDE should not try to choose those operations itself.

## Preferred Integration

Use the Rust API from the `matryoshka` crate.

This avoids:

- spawning a CLI process
- passing binary paths through settings
- parsing command output as the only progress source
- losing typed cancellation and progress hooks

The CLI still exists and is useful for terminal/debug work, but IDE integrations should prefer the Rust API.

## Cargo Dependency

Inside this workspace:

```toml
matryoshka = { path = "crates/api" }
```

After publishing:

```toml
matryoshka = "0.1.1"
```

The `matryoshka` crate pulls in the internal parser, resolver, store, indexer, search, read, and enrichment crates it needs.

## Model Profile

Recommended local oMLX profile:

```text
base_url: http://127.0.0.1:44447
api_key: 2508
chat_model: srswti--bodega-raptor-90m
embedding_model: mlx-community--embeddinggemma-300m-bf16
omlx_reranker: mlx-community--Qwen3-Reranker-0.6B-mxfp8
```

Start oMLX:

```bash
cd /Users/rohit/cradle-mlx/helpers/omlx
source .venv/bin/activate

jesco-apple serve \
  --host 127.0.0.1 \
  --port 44447 \
  --api-key 2508 \
  --max-concurrent-requests 6
```

## Rust API: Prepare

Use one API call for setup, startup, file changes, and manual refresh.

```rust
use matryoshka::{
    Matryoshka, MatryoshkaConfig, MatryoshkaEvent, PrepareOptions,
};

let api = Matryoshka::new(
    MatryoshkaConfig::new("/path/to/repo")
        .with_db("/path/to/repo/.matryoshka/matryoshka.db")
        .with_endpoint("http://127.0.0.1:44447", "2508")
        .with_models(
            "srswti--bodega-raptor-90m",
            "mlx-community--embeddinggemma-300m-bf16",
        )
        .with_ignored_paths([".matryoshka", "target"]),
);

let summary = api.prepare_with_progress(
    PrepareOptions {
        limit: 8,
        queries: vec![
            "main routing flow".into(),
            "request response conversion".into(),
            "tests for provider behavior".into(),
        ],
        write_progress_state: true,
    },
    |event: MatryoshkaEvent| {
        // Update the IDE UI from typed events.
    },
)?;

if summary.is_ready() {
    // Jesco is ready to work with this project.
}
```

### Minimal Prepare

```rust
let summary = api.prepare(PrepareOptions::default())?;
```

Use `prepare_with_progress` for UI integration. Use `prepare` for tests or background jobs where progress does not matter.

## What Prepare Does

`prepare` checks the current state and runs the smallest useful operation.

| State | Internal action | Result |
|---|---|---|
| No database | `index` | Builds files, folders, symbols, cards, retrieval data, FTS, late vectors, then warms results. |
| Database exists, ready marker missing | `update` | Treats the repo as partial, checks freshness, fills missing pieces, writes the ready marker. |
| Files added | `update` | Adds facts, cards, semantic records, FTS rows, and late vectors for new files. |
| Files changed | `update` | Refreshes affected facts, cards, semantic records, FTS rows, and late vectors. |
| Files deleted | `update` | Removes stale file rows, cards, semantic records, FTS rows, and late vectors. |
| Card text has gaps | `repair` | Rebuilds missing card text and refreshes affected retrieval data. |
| Search data missing | `rebuild_search` | Rebuilds semantic records, FTS, embeddings, and late vectors from existing project data. |
| Everything healthy | `update` | Performs a quick freshness pass and warms results. |

Every successful run ends with:

```text
prepare_results
```

That step warms first searches so Jesco starts fast.

## Ready Marker

When the repo is ready, Matryoshka writes:

```text
<repo>/.matryoshka/.jesco-prewarm-complete
```

The marker means:

```text
Jesco is ready to work with you on this project.
```

If the marker is missing but the database exists, call `prepare` again. Do not show a complicated repair choice to the user.

## Prepare Summary

`prepare` returns `PrepareSummary`.

Important fields:

```rust
summary.status
summary.actions_taken
summary.file_count
summary.folder_count
summary.symbol_count
summary.semantic_record_count
summary.changed_files
summary.removed_files
summary.changed_folders
summary.artifact_quality
summary.retrieval_index
summary.prewarm
summary.ready_marker
summary.logs_dir
```

Use these for UI:

| Field | Meaning |
|---|---|
| `status` | `Ready` means Jesco can use the repo. `NeedsAttention` means show logs. |
| `actions_taken` | The internal path Matryoshka chose, such as `index`, `update`, `repair`, `rebuild_search`, `prepare_results`. |
| `changed_files` | Added or changed files handled by this run. |
| `removed_files` | Deleted files cleaned from the database and retrieval data. |
| `artifact_quality` | Health report for card text and project-map artifacts. |
| `retrieval_index` | Health report for semantic records, FTS, embeddings, and late vectors. |
| `prewarm.warmed_hit_count` | Number of hits touched while warming first results. |
| `logs_dir` | Folder to open when the user wants details. |
| `ready_marker` | Marker file written after a ready run. |

## Progress Events

Use `prepare_with_progress` to receive typed events.

Current event variants:

```rust
MatryoshkaEvent::PrepareStarted { .. }
MatryoshkaEvent::PrepareDecision { action, reason }
MatryoshkaEvent::IndexerProgress { operation, progress }
MatryoshkaEvent::PrewarmStarted { query_count, limit }
MatryoshkaEvent::PrewarmCompleted { summary }
MatryoshkaEvent::PrepareCompleted { summary }
```

`IndexerProgress` wraps lower-level progress such as:

```text
ParsingFile
ParsedFile
EnrichingFile
EnrichedFile
EmbeddingBatch
EmbeddedBatch
WritingDatabase
ArtifactQuality
RetrievalIndex
Completed
Failed
```

The API can also write a UI-friendly progress file:

```text
<repo>/.matryoshka/state/progress.json
```

Shape:

```json
{
  "operation": "prepare",
  "status": "running",
  "phase": "enriching",
  "message": "Enriching src/lib.rs",
  "percent": 0.42,
  "current_file": "src/lib.rs",
  "files_done": 14,
  "files_total": 30,
  "updated_at_unix_ms": 1781800000000
}
```

For native IDE UI, prefer typed events. Use `progress.json` as a fallback or for cross-process status.

## Logs

Prepare writes:

```text
<repo>/.matryoshka/logs/prepare.jsonl
```

Useful events:

```json
{"event":"prepare_started","fields":{"existing_file_count":30,"existing_missing_text":0,"existing_search_missing":false,"ready_marker_exists":true}}
{"event":"prepare_decision","fields":{"action":"update","reason":"refresh current project map"}}
{"event":"update_started","fields":{}}
{"event":"update_completed","fields":{}}
{"event":"prewarm_started","fields":{}}
{"event":"prewarm_completed","fields":{}}
{"event":"prepare_completed","fields":{"status":"ready"}}
```

Open logs only when the user asks, or when `status` is `NeedsAttention`.

## Search API

Search returns ranked `SearchHit` values from Matryoshka's hybrid retrieval stack:

- exact symbol/path candidates
- SQLite FTS candidates
- embedding similarity
- late-interaction MaxSim
- graph/card signals
- optional reranker

No reranker:

```rust
use matryoshka::SearchOptions;

let hits = api.search(
    "where is debounce handled",
    SearchOptions::default(),
)?;
```

With oMLX reranker:

```rust
use matryoshka::{RerankerOptions, SearchOptions};

let hits = api.search(
    "where is debounce handled",
    SearchOptions {
        limit: 8,
        reranker: RerankerOptions::Omlx {
            model: "mlx-community--Qwen3-Reranker-0.6B-mxfp8".into(),
            candidates: 20,
        },
    },
)?;
```

Use reranking for ambiguous natural-language queries. Skip it for obvious exact symbol/path lookups.

## Read API

Read one known file:

```rust
let card = api.read("src/watcher.rs")?;
```

`read` returns a `ReadCard` with:

- file identity and path
- file summary
- folder context
- symbols with ranges and signatures where available
- imports
- dependencies
- dependents
- snippets and card details

This is the cheap focused read after search has found the right file.

## Read Bundle API

Read bundle starts from a query, searches, chooses a primary file, then adds nearby/related files.

```rust
use matryoshka::ReadBundleOptions;

let bundle = api.read_bundle(
    ReadBundleOptions::new("how does watcher debounce flow work")
)?;
```

Use `read_bundle` when Jesco asks:

- what should I read next?
- what files explain this flow?
- what files should I inspect before editing this?

With reranker:

```rust
use matryoshka::{ReadBundleOptions, RerankerOptions};

let mut options = ReadBundleOptions::new("what should I read before editing debounce behavior");
options.search.reranker = RerankerOptions::Omlx {
    model: "mlx-community--Qwen3-Reranker-0.6B-mxfp8".into(),
    candidates: 20,
};

let bundle = api.read_bundle(options)?;
```

## Cards API

Cards expose the file/folder/repo summaries Matryoshka built.

All cards:

```rust
let cards = api.cards(matryoshka::CardsOptions { empty_only: false })?;
```

Only cards that still need repair:

```rust
let gaps = api.cards(matryoshka::CardsOptions { empty_only: true })?;
if !gaps.is_empty() {
    let summary = api.prepare(PrepareOptions::default())?;
}
```

The IDE should not show "repair summaries" as a normal user action. If gaps exist, call `prepare`.

## CLI Fallback

The CLI remains useful for shell usage and debugging.

Environment:

```bash
export MATRYOSHKA=/Users/rohit/cradle-embed/target/debug/matryoshka-rs
export REPO=/path/to/repo
export DB=$REPO/.matryoshka/matryoshka.db
export MLX_URL=http://127.0.0.1:44447
export MLX_KEY=2508
export CHAT_MODEL=srswti--bodega-raptor-90m
export EMBED_MODEL=mlx-community--embeddinggemma-300m-bf16
export RERANK_MODEL=mlx-community--Qwen3-Reranker-0.6B-mxfp8
```

Prepare:

```bash
$MATRYOSHKA prepare "$REPO" \
  --db "$DB" \
  --base-url "$MLX_URL" \
  --api-key "$MLX_KEY" \
  --model "$CHAT_MODEL" \
  --embedding-model "$EMBED_MODEL" \
  --ignore .matryoshka \
  --ignore target \
  --json
```

Search:

```bash
$MATRYOSHKA search \
  --db "$DB" \
  --base-url "$MLX_URL" \
  --api-key "$MLX_KEY" \
  --embedding-model "$EMBED_MODEL" \
  "where is debounce handled"
```

Search with oMLX reranker:

```bash
$MATRYOSHKA search \
  --db "$DB" \
  --base-url "$MLX_URL" \
  --api-key "$MLX_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --omlx-rerank \
  --omlx-rerank-model "$RERANK_MODEL" \
  "where is debounce handled"
```

Read:

```bash
$MATRYOSHKA read \
  --db "$DB" \
  --repo-root "$REPO" \
  src/watcher.rs
```

Read bundle:

```bash
$MATRYOSHKA read-bundle \
  --db "$DB" \
  --repo-root "$REPO" \
  --base-url "$MLX_URL" \
  --api-key "$MLX_KEY" \
  --embedding-model "$EMBED_MODEL" \
  "what should I read before editing debounce behavior"
```

Cards:

```bash
$MATRYOSHKA cards --db "$DB" --summaries
$MATRYOSHKA cards --db "$DB" --empty
$MATRYOSHKA cards --db "$DB" --json
```

## IDE Wording

Keep the user-facing surface small.

Primary button:

```text
Prepare
```

Good states:

| State | Copy |
|---|---|
| Not ready | Jesco needs a moment to get this project ready. |
| Running | Jesco is preparing this project. |
| Ready | Jesco is ready to work with you on this project. |
| Needs attention | Jesco needs a quick check before using this project. |

Avoid exposing internal terms in the main UI:

- semantic index
- late vectors
- FTS
- repair summaries
- rebuild retrieval

Those are log/details concepts, not first-run UI concepts.

## Verified Tests

Default test lane:

```bash
cargo test -p matryoshka
cargo test
```

Live oMLX lifecycle test:

```bash
MATRYOSHKA_REAL_OMLX=1 \
MATRYOSHKA_MLX_BASE_URL=http://127.0.0.1:44447 \
MATRYOSHKA_MLX_API_KEY=2508 \
MATRYOSHKA_MLX_CHAT_MODEL=srswti--bodega-raptor-90m \
MATRYOSHKA_MLX_EMBED_MODEL=mlx-community--embeddinggemma-300m-bf16 \
MATRYOSHKA_OMLX_RERANK_MODEL=mlx-community--Qwen3-Reranker-0.6B-mxfp8 \
cargo test -p matryoshka --test real_omlx -- --ignored --nocapture
```

This live test covers:

- no DB
- ready marker missing
- file added
- file changed
- file deleted
- card gaps
- search data missing
- healthy repo
- search
- search with oMLX reranker
- read
- read bundle
- cards
- typed progress events

Last verified live result:

```text
test real_omlx_prepare_search_read_lifecycle_work_through_rust_api ... ok
finished in 37.70s
```

Live test against the copied-crates repo:

```bash
MATRYOSHKA_TEST_REPO=/Users/rohit/cradle-embed/test_repo \
MATRYOSHKA_MLX_BASE_URL=http://127.0.0.1:44447 \
MATRYOSHKA_MLX_API_KEY=2508 \
MATRYOSHKA_MLX_CHAT_MODEL=srswti--bodega-raptor-90m \
MATRYOSHKA_MLX_EMBED_MODEL=mlx-community--embeddinggemma-300m-bf16 \
MATRYOSHKA_OMLX_RERANK_MODEL=mlx-community--Qwen3-Reranker-0.6B-mxfp8 \
cargo test -p matryoshka --test test_repo_omlx -- --ignored --nocapture
```

This test uses:

```text
/Users/rohit/cradle-embed/test_repo/.matryoshka/api-test-repo-live/matryoshka.db
```

It verifies the same prepare lifecycle on the real copied-crates repo, then checks search, oMLX-reranked search, read, read-bundle, cards, logs, and progress events.

Last verified result:

```text
no_db actions=["index", "prepare_results"] files=30 symbols=552 records=749 warmed=12
search top=[("watcher/src/poller.rs", ...), ("watcher/src/matryoshka_update_probe.rs", ...)]
reranked search top=[("watcher/src/poller.rs", ...), ("watcher/src/invalidation.rs", ...)]
read file=watcher/src/poller.rs symbols=18 deps=1 dependents=0
read_bundle primary=watcher/src/matryoshka_update_probe.rs related=3
cards total=60 empty=0
marker_missing actions=["update", "prepare_results"]
file_added actions=["update", "prepare_results"] changed_files=1
file_changed actions=["update", "prepare_results"] changed_files=1
file_deleted actions=["update", "prepare_results"] removed_files=1
card_gaps actions=["repair", "prepare_results"]
search_missing actions=["rebuild_search", "prepare_results"] records=719 fts=719 late_records=719
healthy actions=["update", "prepare_results"] changed_files=0 removed_files=0
test test_repo_live_prepare_search_read_and_progress_work_through_rust_api ... ok
finished in 422.76s
```
