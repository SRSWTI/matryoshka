# Matryoshka Usage

Matryoshka prepares a repository so Jesco can work with it in fewer steps: find the right code, read the right files, and avoid wasting tokens on raw browsing.

The normal lifecycle has one command:

```bash
matryoshka-rs prepare <repo-root>
```

`prepare` is safe to run again and again. The IDE does not need to decide whether to index, update, repair, rebuild retrieval, or pre-warm. It calls `prepare`; Matryoshka checks the repository and does the right work.

## The IDE Rule

For setup, startup, file changes, or a user pressing "Prepare", run the same command:

```bash
matryoshka-rs prepare "$REPO" \
  --db "$DB" \
  --base-url "$MLX_URL" \
  --api-key "$MLX_KEY" \
  --model "$CHAT_MODEL" \
  --embedding-model "$EMBED_MODEL" \
  --ignore .matryoshka \
  --ignore target \
  --json
```

Recommended environment:

```bash
export MATRYOSHKA=/Users/rohit/cradle-embed/target/debug/matryoshka-rs
export REPO=/path/to/repo
export DB=$REPO/.matryoshka/matryoshka.db
export MLX_URL=http://127.0.0.1:44447
export MLX_KEY=2508
export CHAT_MODEL=srswti--bodega-raptor-90m
export EMBED_MODEL=mlx-community--embeddinggemma-300m-bf16
```

Then:

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

## What Prepare Does

`prepare` checks the current state, chooses the needed work, writes logs, and emits JSON for the IDE.

It handles:

- first setup
- partial database state
- missing ready marker
- added files
- changed files
- deleted files
- missing card text
- missing search artifacts
- retrieval pre-warming

When it finishes cleanly, it writes:

```text
<repo>/.matryoshka/.jesco-prewarm-complete
```

That marker means Jesco is ready to work with this project.

## Scenario Behavior

The IDE should not branch into different commands for these scenarios. It should always call `prepare`.

| Scenario | What prepare detects | JSON `actions_taken` | Result |
|---|---|---|---|
| No DB | No indexed files | `["index", "prepare_results"]` | Builds the project map, writes cards, builds retrieval data, pre-warms, writes the ready marker. |
| DB exists, marker missing | Database exists but ready marker is absent | `["update", "prepare_results"]` | Treats the state as partial, checks freshness, fills gaps, pre-warms, writes the ready marker. |
| Added file | Filesystem has a new indexed file | `["update", "prepare_results"]` | Adds facts, symbols, cards, retrieval records, FTS rows, and late vectors for the new file. |
| Changed file | File hash changed | `["update", "prepare_results"]` | Refreshes affected facts, cards, retrieval records, FTS rows, and late vectors. |
| Deleted file | Indexed path no longer exists | `["update", "prepare_results"]` | Removes stale file records, cards, retrieval records, FTS rows, and late vectors. |
| Cards have gaps | Existing card text is empty | `["repair", "prepare_results"]` | Re-runs enrichment for gaps, refreshes related retrieval data, pre-warms. |
| Search data missing | Semantic records, FTS, or late vectors are missing/incomplete | `["rebuild_search", "prepare_results"]` | Rebuilds retrieval data from existing cards and facts, then pre-warms. |
| Everything healthy | No gaps and no file changes | `["update", "prepare_results"]` | Performs a quick freshness pass, pre-warms, returns ready. |

## JSON Output Contract

Use `--json` for IDE integration. The important fields are:

```json
{
  "status": "ready",
  "repo_root": "/path/to/repo",
  "db": "/path/to/repo/.matryoshka/matryoshka.db",
  "ready_marker": "/path/to/repo/.matryoshka/.jesco-prewarm-complete",
  "logs": "/path/to/repo/.matryoshka/logs",
  "actions_taken": ["update", "prepare_results"],
  "project_map": {
    "status": "ready",
    "files": 30,
    "folders": 29,
    "symbols": 552,
    "cards": {
      "file": 30,
      "folder": 29,
      "repo": 1,
      "missing_text": 0,
      "empty_file_samples": [],
      "empty_folder_samples": []
    }
  },
  "search": {
    "status": "ready",
    "semantic_records": 749,
    "embedded_records": 719,
    "fts_records": 749,
    "late_vector_rows": 15538,
    "records_with_late_vectors": 719,
    "late_interaction_enabled": true
  },
  "changes": {
    "changed_files": 0,
    "removed_files": 0,
    "changed_folders": 0,
    "repo_card_updated": false
  },
  "prepare_results": {
    "fts_records": 749,
    "query_count": 6,
    "warmed_hits": 24
  },
  "embedding_model": "mlx-community--embeddinggemma-300m-bf16"
}
```

### Fields The IDE Should Read

| Field | Meaning |
|---|---|
| `status` | Overall result. `ready` means Jesco can use the repo. `needs_attention` means show attention state and open logs. |
| `actions_taken` | What Matryoshka decided to do internally. Good for progress text and diagnostics. |
| `project_map.status` | Whether file/folder/repo cards are healthy. |
| `project_map.cards.missing_text` | Number of cards that still need repair. Healthy is `0`. |
| `search.status` | Whether search/read retrieval data is ready. Healthy is `ready`. |
| `changes.changed_files` | Files added or changed in this run. |
| `changes.removed_files` | Files deleted and cleaned up in this run. |
| `changes.changed_folders` | Folder cards refreshed in this run. |
| `prepare_results.warmed_hits` | Number of results touched by pre-warming. Nonzero means first searches should be faster. |
| `logs` | Directory containing `prepare.jsonl` and other command logs. |
| `ready_marker` | File written when the repo is fully ready. |

## IDE State Mapping

Recommended UI language:

| JSON state | IDE state | Suggested copy |
|---|---|---|
| Command running | Getting Ready | Jesco is getting this project ready. |
| `actions_taken` contains `index` | Preparing Project | Reading the project and building Jesco's map. |
| `actions_taken` contains `repair` | Repairing | Filling in missing project details. |
| `actions_taken` contains `rebuild_search` | Refreshing Results | Rebuilding the fast lookup layer. |
| `actions_taken` contains `prepare_results` | Warming Results | Making first searches faster. |
| `status == "ready"` | Ready | Jesco is ready to work with you on this project. |
| `status == "needs_attention"` | Needs Attention | Jesco needs a quick check before it can use this project. |

Keep the primary button simple:

```text
Prepare
```

Advanced/debug buttons can exist elsewhere, but normal setup and freshness should use `prepare`.

## Logs

`prepare` writes JSONL logs here:

```text
<repo>/.matryoshka/logs/prepare.jsonl
```

Useful events:

```json
{"event":"prepare_started","fields":{"existing_file_count":30,"existing_missing_text":0,"existing_search_missing":true,"ready_marker_exists":true}}
{"event":"prepare_decision","fields":{"action":"rebuild_search","reason":"search data is missing or incomplete"}}
{"event":"update_started","fields":{}}
{"event":"update_completed","fields":{}}
{"event":"prewarm_started","fields":{}}
{"event":"prewarm_completed","fields":{}}
{"event":"prepare_completed","fields":{"status":"ready"}}
```

For the IDE:

- stream the command output for progress
- read final JSON for state
- open the `logs` directory when the user asks for details
- use `prepare.jsonl` to explain what happened when a run needs attention

## Tested Prepare Flow

Implemented and tested against:

```text
/Users/rohit/cradle-embed/test_repo
```

Using online oMLX on port `44447`. No offline index was used for the real simulations.

Command shape tested:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs prepare /Users/rohit/cradle-embed/test_repo \
  --db <scenario-db> \
  --base-url http://127.0.0.1:44447 \
  --api-key 2508 \
  --model srswti--bodega-raptor-90m \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --ignore .matryoshka \
  --ignore target \
  --limit 4 \
  --json
```

Results:

| Scenario | JSON `actions_taken` | Result |
|---|---|---|
| No DB | `["index", "prepare_results"]` | Passed. Full index, cards healthy, search ready. |
| DB exists, marker missing | `["update", "prepare_results"]` | Passed. Detected missing marker in logs, wrote marker, ready. |
| Added file | `["update", "prepare_results"]` | Passed. `changed_files: 1`, file count rose to 31, ready. |
| Changed file | `["update", "prepare_results"]` | Passed. `changed_files: 1`, ready. |
| Deleted file | `["update", "prepare_results"]` | Passed. `removed_files: 1`, file count returned to 30, stale records removed. |
| Cards have gaps | `["repair", "prepare_results"]` | Passed. Two blanked cards were repaired; `missing_text: 0`. |
| Search data missing | `["rebuild_search", "prepare_results"]` | Passed. Deleted retrieval rows were rebuilt; status ready. |
| Everything healthy | `["update", "prepare_results"]` | Passed. No changed files, no gaps, search ready. |

Key exact outputs:

```json
{
  "status": "ready",
  "actions_taken": ["index", "prepare_results"],
  "changes": {
    "changed_files": 30,
    "changed_folders": 29,
    "removed_files": 0
  },
  "project_map": {
    "status": "ready"
  },
  "search": {
    "status": "ready"
  },
  "prepare_results": {
    "warmed_hits": 24
  }
}
```

```json
{
  "status": "ready",
  "actions_taken": ["repair", "prepare_results"],
  "project_map": {
    "cards": {
      "missing_text": 0
    }
  },
  "search": {
    "status": "ready"
  }
}
```

```json
{
  "status": "ready",
  "actions_taken": ["rebuild_search", "prepare_results"],
  "search": {
    "semantic_records": 719,
    "fts_records": 719,
    "late_vector_rows": 15538,
    "records_with_late_vectors": 719,
    "status": "ready"
  }
}
```

Deleted file cleanup was verified directly:

```text
file_cards|0
semantic_records|0
```

for the deleted probe path.

Verification:

```bash
cargo build -p matryoshka-cli
cargo test -p matryoshka-cli
```

Both passed.

## Starting oMLX

Start the local server before production `prepare` runs:

```bash
cd /Users/rohit/cradle-mlx/helpers/omlx
source .venv/bin/activate

jesco-apple serve \
  --host 127.0.0.1 \
  --port 44447 \
  --api-key 2508 \
  --max-concurrent-requests 6
```

Then run `prepare` with:

```bash
--base-url http://127.0.0.1:44447
--api-key 2508
--model srswti--bodega-raptor-90m
--embedding-model mlx-community--embeddinggemma-300m-bf16
```

## Cards Health Command

`prepare` already checks card health internally. Use `cards` only for inspection or debugging.

Show only cards with missing text:

```bash
matryoshka-rs cards --db "$DB" --json --empty
```

Healthy output:

```json
[]
```

Non-empty output:

```json
[
  {
    "card_type": "file",
    "id": "watcher/src/poller.rs",
    "summary": "",
    "is_empty": true
  }
]
```

Show all card summaries:

```bash
matryoshka-rs cards --db "$DB" --json --summaries
```

Markdown output for humans or LLM tools:

```bash
matryoshka-rs cards --db "$DB" --summaries
matryoshka-rs cards --db "$DB" --empty
```

## Search

After `prepare` returns `ready`, search is available.

Basic search:

```bash
matryoshka-rs search \
  --db "$DB" \
  --base-url "$MLX_URL" \
  --api-key "$MLX_KEY" \
  --embedding-model "$EMBED_MODEL" \
  "where is watcher debounce handled"
```

Search with oMLX reranker:

```bash
matryoshka-rs search \
  --db "$DB" \
  --base-url "$MLX_URL" \
  --api-key "$MLX_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --omlx-rerank \
  --omlx-rerank-model mlx-community--Qwen3-Reranker-0.6B-mxfp8 \
  "where is watcher debounce handled"
```

Use reranking when the first page needs to be sharper. Skip it when speed matters more.

## Read

Read one file from the prepared project map:

```bash
matryoshka-rs read \
  --db "$DB" \
  --repo-root "$REPO" \
  watcher/src/poller.rs
```

Read a focused bundle for an edit or investigation:

```bash
matryoshka-rs read-bundle \
  --db "$DB" \
  --repo-root "$REPO" \
  --mode edit \
  --base-url "$MLX_URL" \
  --api-key "$MLX_KEY" \
  --embedding-model "$EMBED_MODEL" \
  "change watcher debounce behavior"
```

With oMLX reranking:

```bash
matryoshka-rs read-bundle \
  --db "$DB" \
  --repo-root "$REPO" \
  --mode edit \
  --base-url "$MLX_URL" \
  --api-key "$MLX_KEY" \
  --embedding-model "$EMBED_MODEL" \
  --omlx-rerank \
  --omlx-rerank-model mlx-community--Qwen3-Reranker-0.6B-mxfp8 \
  "change watcher debounce behavior"
```

## Op Commands

`op` commands are task-shaped search/read helpers. Use them after `prepare`.

Examples:

```bash
matryoshka-rs op find-symbol \
  --db "$DB" \
  --base-url "$MLX_URL" \
  --api-key "$MLX_KEY" \
  --embedding-model "$EMBED_MODEL" \
  "RepoWatcher"
```

```bash
matryoshka-rs op edit-target \
  --db "$DB" \
  --base-url "$MLX_URL" \
  --api-key "$MLX_KEY" \
  --embedding-model "$EMBED_MODEL" \
  "add a new search ranking boost"
```

```bash
matryoshka-rs op read-next \
  --db "$DB" \
  --base-url "$MLX_URL" \
  --api-key "$MLX_KEY" \
  --embedding-model "$EMBED_MODEL" \
  "after reading watcher/src/poller.rs, what should I read next"
```

Use `--omlx-rerank` on these when top-result precision matters.

## Advanced Commands

These commands still exist, but normal IDE lifecycle should not call them directly.

| Command | Use |
|---|---|
| `index` | Force a clean full build. Mostly debugging now. |
| `update` | Force incremental refresh. `prepare` calls this when needed. |
| `prewarm` | Warm retrieval only. `prepare` calls this automatically. |
| `rebuild-semantic` | Rebuild retrieval data. `prepare` calls this when search data is missing. |
| `watch` | Long-running watcher mode. IDEs can instead call `prepare` after file changes. |
| `cards` | Inspect card health and text. |

For the product path, prefer:

```text
Prepare -> Search / Read / Op
```

## Offline Mode

`--offline` is for deterministic smoke tests. It avoids the LLM and can produce thinner or missing card text.

Do not use offline mode for production setup.

Production setup should use:

```bash
--base-url "$MLX_URL"
--api-key "$MLX_KEY"
--model "$CHAT_MODEL"
--embedding-model "$EMBED_MODEL"
```

## Practical IDE Flow

On project open:

```text
run prepare
read final JSON
if status == ready: show "Jesco is ready to work with you on this project."
else: show attention state and offer Open Logs
```

On file save, add, delete, branch change, or dependency changes:

```text
run prepare again
use changes.* to show what changed
use project_map and search status to decide ready vs attention
```

Before search/read:

```text
if ready marker exists and latest prepare status is ready: run search/read/op
else: run prepare first
```

That is the simple contract: the IDE asks Matryoshka to prepare; Matryoshka decides the rest.
