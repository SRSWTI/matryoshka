# Matryoshka Usage

This file is the practical command guide for the current retrieval pipeline through **M3.1**.

It focuses on:

- indexing with oMLX
- dense on/off behavior
- function/class/method chunk search
- symbol search
- no-collapse raw record search
- compact output
- validation commands and representative outputs

---

## Current Retrieval Flow

```text
repo files
  -> parser extracts files, symbols, and code chunks
  -> docstrings/doc-comments become chunk summaries when present
  -> undocumented chunks are summarized by oMLX/Raptor
  -> semantic_records are built from files, folders, repo, symbols, snippets, and code chunks
  -> SQLite FTS is built
  -> dense vectors are written only when dense is enabled
  -> search can return file, record, symbol, or chunk-level results
```

Important distinction:

```text
Index flag controls what is stored.
Search flag controls what is used.
```

For example:

| Indexed with | Searched with | Behavior |
|---|---|---|
| `--enable-dense` | `--enable-dense` | exact + FTS + dense + late interaction |
| `--enable-dense` | `--disable-dense` | exact + FTS only; existing dense vectors ignored |
| `--disable-dense` | `--disable-dense` | exact + FTS only |
| `--disable-dense` | `--enable-dense` | exact + FTS; dense has no stored vectors to score against |

---

## Local Assumptions Used Below

```text
repo:      /Users/rohit/cradle-embed
db:        /Users/rohit/cradle-embed/.matryoshka/matryoshka.db
binary:    /Users/rohit/cradle-embed/target/debug/matryoshka-rs
oMLX URL:  http://127.0.0.1:44449
api key:   2508
embedder:  mlx-community--embeddinggemma-300m-bf16
chunk LLM: srswti--bodega-raptor-90m
chat LLM:  MercuriusDream--Qwen3.5-4B-MLX-mxfp8
```

If you have not installed the CLI globally, always use:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs
```

or after release build:

```bash
/Users/rohit/cradle-embed/target/release/matryoshka-rs
```

---

## Start oMLX

Command:

```bash
cd /Users/rohit/cradle-mlx/helpers/omlx
source .venv/bin/activate

jesco-apple serve \
  --host 127.0.0.1 \
  --port 44449 \
  --api-key 2508 \
  --max-concurrent-requests 6
```

Description:

Starts the local oMLX server used for:

- chunk summaries through `srswti--bodega-raptor-90m`
- dense embeddings through `mlx-community--embeddinggemma-300m-bf16`
- later SPLADE work through `/Users/rohit/.omlx/models/naver--splade-code-06B`

Expected server output:

```text
oMLX - LLM inference, optimized for your Mac
Binding server at http://127.0.0.1:44449
INFO:     Uvicorn running on http://127.0.0.1:44449
```

---

## Build the Local CLI

Command:

```bash
cd /Users/rohit/cradle-embed
cargo build -p matryoshka-cli
```

Description:

Builds the current local CLI binary. Use this before testing new flags if you have
not installed the crate globally.

Expected output:

```text
Finished `dev` profile [unoptimized + debuginfo] target(s) in ...
```

---

## Check Search CLI Flags

Command:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search --help
```

Description:

Verifies the current search flags.

Relevant output:

```text
--retrieval-primary <RETRIEVAL_PRIMARY>
    [default: hybrid] [possible values: fts, splade, dense, hybrid]

--enable-dense
--disable-dense
    [aliases: --no-dense-embeddings]
--dense-fallback
--no-dense-fallback

--result-granularity <RESULT_GRANULARITY>
    [default: file] [possible values: file, record, symbol, chunk]

--no-collapse

--compact
    [aliases: --hide-match-details, --no-match-details]
```

---

## Result Granularity Modes

Search can now expose the level of result you want.

| Mode | Command flag | What you get |
|---|---|---|
| File collapsed | `--result-granularity file` | Default. Groups file/symbol/snippet/chunk matches into one file result. |
| Raw record | `--result-granularity record` | No collapse. Returns raw matching semantic records. |
| Raw record shortcut | `--no-collapse` | Same as `--result-granularity record`. |
| Symbol only | `--result-granularity symbol` | Only symbols/functions/classes/method definitions. |
| Chunk only | `--result-granularity chunk` | Function/class/method code chunks with chunk summaries. |

---

## Compact Output

Use:

```bash
--compact
```

Aliases:

```bash
--hide-match-details
--no-match-details
```

Compact output removes:

```text
matched_terms
total_matched_symbols
why_matched
```

Compact output keeps:

```text
matched_symbols
```

Use compact output when you want cleaner responses for agents or UI display.

---

## Check Whether the Current DB Has Dense Embeddings

Command:

```bash
cd /Users/rohit/cradle-embed

sqlite3 .matryoshka/matryoshka.db "SELECT 'semantic_records', COUNT(*) FROM semantic_records UNION ALL SELECT 'embedded_records', SUM(CASE WHEN json_extract(payload_json, '$.embedding') IS NOT NULL THEN 1 ELSE 0 END) FROM semantic_records UNION ALL SELECT 'late_vector_rows', COUNT(*) FROM semantic_late_vectors UNION ALL SELECT 'records_with_late_vectors', COUNT(DISTINCT record_id) FROM semantic_late_vectors UNION ALL SELECT 'fts_records', COUNT(*) FROM semantic_records_fts;"
```

Description:

Checks whether the DB contains FTS rows, dense embeddings, and late-interaction vectors.

Observed output:

```text
semantic_records|1334
embedded_records|1305
late_vector_rows|34487
records_with_late_vectors|1305
fts_records|1334
```

Interpretation:

```text
FTS exists ✅
Dense embeddings exist ✅
Late-interaction vectors exist ✅
```

---

## Count Semantic Record Types

Command:

```bash
cd /Users/rohit/cradle-embed

sqlite3 .matryoshka/matryoshka.db "SELECT entity_type, COUNT(*) FROM semantic_records GROUP BY entity_type ORDER BY entity_type;"
```

Description:

Shows which kinds of records are available for retrieval.

Observed output:

```text
CodeChunk|371
File|58
Folder|27
Repo|1
Snippet|113
Symbol|764
```

Interpretation:

```text
Function/class/method code chunks are indexed ✅
Symbols are indexed ✅
File/folder/repo cards are indexed ✅
```

---

## Fresh Index With Dense Enabled

Command:

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

Description:

Indexes a repo with:

- parser extraction
- docstring/doc-comment chunk summaries
- oMLX fallback summaries for undocumented chunks
- semantic records
- FTS
- dense embeddings
- late-interaction vectors

Expected progress shape:

```jsonl
{"type":"started","total_steps":null}
{"type":"discovering_files"}
{"type":"files_discovered","total_files":...}
{"type":"parsing_file","path":"...","index":...,"total_files":...}
{"type":"parsed_file","path":"...","index":...,"total_files":...}
{"type":"enriching_chunk_batch","batch_index":...,"total_batches":...,"chunks_in_batch":...}
{"type":"enriched_chunk_batch","batch_index":...,"total_batches":...,"chunks_in_batch":...}
{"type":"embedding_batch","batch_index":...,"total_batches":...,"records_in_batch":...}
{"type":"embedded_batch","batch_index":...,"total_batches":...,"records_in_batch":...}
{"type":"retrieval_index_health","report":{"semantic_records":...,"embedded_records":...,"fts_records":...,"late_vector_rows":...,"records_with_late_vectors":...,"dense_enabled":true,"late_interaction_enabled":true}}
{"type":"completed","file_count":...,"folder_count":...,"symbol_count":...,"semantic_record_count":...,"embedding_model":"mlx-community--embeddinggemma-300m-bf16"}
```

Notes:

- `records_in_batch` means semantic records being embedded.
- Records can be file cards, folder cards, repo cards, symbols, snippets, and code chunks.

---

## Fresh Index With Dense Disabled

Command:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs index \
  /Users/rohit/cradle-embed \
  --db /tmp/cradle_embed_dense_off.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --model MercuriusDream--Qwen3.5-4B-MLX-mxfp8 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --chunk-summary-model srswti--bodega-raptor-90m \
  --chunk-summary-concurrency 6 \
  --disable-dense \
  --ignore target \
  --ignore .git \
  --ignore .matryoshka \
  --progress-jsonl
```

Description:

Indexes a repo while skipping dense embeddings. This still writes:

```text
semantic_records
semantic_records_fts
code_chunks
file/folder/repo cards
symbols/snippets
```

It does not write:

```text
dense record embeddings
semantic_late_vectors
```

Observed M3 validation output for a dense-disabled `cradle-embed` temp DB:

```text
files_discovered: 35
symbol_count: 856
code_chunks: 856
chunk summaries requested: 842
chunk summary batches: 27
semantic_records: 1956
fts_records: 1956
embedded_records: 0
late_vectors: 0
retrieval_primary: hybrid
dense_enabled: false
late_interaction_enabled: false
```

DB summary after index + no-change update:

```text
code_chunks: 856
chunk_sources:
  doc_comment: 14
  llm:         842
  empty:       0
semantic_records: 1956
fts_records: 1956
embedded_records: 0
late_vectors: 0
```

---

## Update Existing Repo

Command:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs update \
  /Users/rohit/cradle-embed \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --chunk-summary-model srswti--bodega-raptor-90m \
  --chunk-summary-concurrency 6 \
  --enable-dense \
  --ignore target \
  --ignore .git \
  --ignore .matryoshka \
  --progress-jsonl
```

Description:

Refreshes only changed/deleted/added files. It preserves generated summaries for
unchanged chunks and only calls the LLM for changed undocumented chunks.

Expected progress shape:

```jsonl
{"type":"started","total_steps":null}
{"type":"discovering_files"}
{"type":"files_discovered","total_files":...}
{"type":"parsing_file","path":"...","index":...,"total_files":...}
{"type":"parsed_file","path":"...","index":...,"total_files":...}
{"type":"enriching_chunk_batch","batch_index":...,"total_batches":...,"chunks_in_batch":...}
{"type":"enriched_chunk_batch","batch_index":...,"total_batches":...,"chunks_in_batch":...}
{"type":"writing_database","records_written":...}
{"type":"embedding_batch","batch_index":...,"total_batches":...,"records_in_batch":...}
{"type":"embedded_batch","batch_index":...,"total_batches":...,"records_in_batch":...}
{"type":"retrieval_index_health","report":{...}}
{"type":"completed","file_count":...,"folder_count":...,"symbol_count":...,"semantic_record_count":...,"embedding_model":"mlx-community--embeddinggemma-300m-bf16"}
```

Use this before searching for newly-added symbols such as:

```text
resolve_search_result_granularity
```

---

## Search: Default File-Collapsed Dense Search

Command:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "where is dense embedding disabled during indexing" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --limit 8
```

Description:

Runs normal dense-enabled search. This is default file-collapsed output.

Uses:

```text
exact candidates
SQLite FTS
dense vector similarity
late-interaction MaxSim
query-plan boosts
```

Representative top output:

```json
[
  {
    "entity_id": "crates/indexer/src/indexer.rs",
    "record_id": "semantic:file_card:crates/indexer/src/indexer.rs",
    "path": "crates/indexer/src/indexer.rs",
    "title": "File crates/indexer/src/indexer.rs",
    "entity_type": "file",
    "matched_terms": [
      "dense",
      "disabled",
      "embedding",
      "indexing"
    ],
    "matched_symbols": [
      "ArtifactRepairSet",
      "ArtifactRepairSet::is_empty",
      "EmbeddingProgress",
      "EmbeddingProgress::new",
      "FullIndexer",
      "FullIndexer::collect_file_cards_with_progress",
      "FullIndexer::index_repo",
      "FullIndexer::index_repo_with_progress",
      "FullIndexer::new",
      "FullIndexer::rebuild_semantic_index",
      "FullIndexer::rebuild_semantic_index_with_progress",
      "FullIndexer::refresh_artifacts"
    ],
    "total_matched_symbols": 62,
    "score": 1.5974306,
    "why_matched": [
      "Exact token, symbol, or path candidate matched the query",
      "Late-interaction MaxSim matched indexed code-token vectors",
      "SQLite FTS matched exact query terms in path, title, content, or metadata",
      "Summary/content is semantically close to the query"
    ]
  }
]
```

Interpretation:

```text
Top file is correct: indexing dense gating lives in crates/indexer/src/indexer.rs ✅
Late-interaction evidence is present, so dense search path is active ✅
```

---

## Search: Function/Class/Method Chunk Results

Command:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "how does update preserve unchanged generated chunk summaries" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --result-granularity chunk \
  --compact \
  --limit 3
```

Description:

Returns only `code_chunk` records. This exposes function/class/method-level answers.

`--compact` hides noisy fields while keeping `matched_symbols`.

Observed output:

```json
[
  {
    "description": "Symbol: FullIndexer::refresh_chunk_summaries\nKind: method\nSignature: fn refresh_chunk_summaries(\nLines: 1859-1965\nSummary source: doccomment",
    "entity_id": "crates/indexer/src/indexer.rs::FullIndexer::refresh_chunk_summaries:1859",
    "entity_type": "code_chunk",
    "matched_symbols": [
      "FullIndexer::refresh_chunk_summaries"
    ],
    "path": "crates/indexer/src/indexer.rs",
    "record_id": "semantic:code_chunk:crates/indexer/src/indexer.rs::FullIndexer::refresh_chunk_summaries:1859",
    "score": 1.6419492959976196,
    "summary": "Summarize code chunks that have no useful docstring/doc comment, persist the updated chunks to the store, and build `code_chunk` semantic records in the target template for retrieval.  Only chunks with `summary_source == Empty` (or generic/short docs) are sent to the LLM. Chunks with useful docs are used directly. Chunks in files whose `source_hash` is unchanged are skipped entirely.",
    "title": "CodeChunk FullIndexer::refresh_chunk_summaries in crates/indexer/src/indexer.rs"
  },
  {
    "description": "Symbol: main\nKind: function\nSignature: fn main() -> Result<()>\nLines: 535-1362\nSummary source: llm",
    "entity_id": "crates/cli/src/main.rs::main:535",
    "entity_type": "code_chunk",
    "matched_symbols": [
      "main"
    ],
    "path": "crates/cli/src/main.rs",
    "record_id": "semantic:code_chunk:crates/cli/src/main.rs::main:535",
    "score": 1.3671839237213135,
    "summary": "The code handles command-line arguments for a crate's main function, which processes different commands like preparing, indexing, and updating data. It initializes configuration parameters, resolves database paths, ensures layout, and performs operations based on the command's type, returning results or logging details accordingly.",
    "title": "CodeChunk main in crates/cli/src/main.rs"
  },
  {
    "description": "Symbol: FullIndexer::refresh_artifacts\nKind: method\nSignature: fn refresh_artifacts(\nLines: 1375-1616\nSummary source: llm",
    "entity_id": "crates/indexer/src/indexer.rs::FullIndexer::refresh_artifacts:1375",
    "entity_type": "code_chunk",
    "matched_symbols": [
      "FullIndexer::refresh_artifacts"
    ],
    "path": "crates/indexer/src/indexer.rs",
    "record_id": "semantic:code_chunk:crates/indexer/src/indexer.rs::FullIndexer::refresh_artifacts:1375",
    "score": 1.3214116096496582,
    "summary": "The method refresh_artifacts updates file and folder cards in the store, prunes orphaned artifacts, and enriches files with progress tracking. It also handles repository cards, updating them if necessary, and manages deleted files. The code includes milestones for summarizing and embedding data.",
    "title": "CodeChunk FullIndexer::refresh_artifacts in crates/indexer/src/indexer.rs"
  }
]
```

Interpretation:

```text
The top result is now a function/method-level chunk ✅
The summary shown is the chunk summary, not the file-card summary ✅
matched_terms, total_matched_symbols, and why_matched are hidden ✅
matched_symbols is preserved ✅
```

---

## Search: Raw Record / No Collapse

Command:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "where is resolve_retrieval_config defined" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --no-collapse \
  --compact \
  --limit 4
```

Description:

Returns raw matching records instead of collapsing them into one file result.

`--no-collapse` is equivalent to:

```bash
--result-granularity record
```

Observed output:

```json
[
  {
    "description": "Symbol: resolve_retrieval_config\nKind: Function\nFile: crates/cli/src/main.rs",
    "entity_id": "crates/cli/src/main.rs::resolve_retrieval_config:498",
    "entity_type": "symbol",
    "matched_symbols": [
      "resolve_retrieval_config"
    ],
    "path": "crates/cli/src/main.rs",
    "record_id": "semantic:symbol:crates/cli/src/main.rs::resolve_retrieval_config:498",
    "score": 2.02943754196167,
    "summary": "fn resolve_retrieval_config(",
    "title": "Symbol resolve_retrieval_config in crates/cli/src/main.rs"
  },
  {
    "description": "Symbol: RetrievalConfig\nKind: Struct\nFile: crates/core-ir/src/models.rs",
    "entity_id": "crates/core-ir/src/models.rs::RetrievalConfig:516",
    "entity_type": "symbol",
    "matched_symbols": [
      "RetrievalConfig"
    ],
    "path": "crates/core-ir/src/models.rs",
    "record_id": "semantic:symbol:crates/core-ir/src/models.rs::RetrievalConfig:516",
    "score": 1.0542629957199097,
    "summary": "pub struct RetrievalConfig",
    "title": "Symbol RetrievalConfig in crates/core-ir/src/models.rs"
  },
  {
    "description": "Symbol: MatryoshkaConfig::retrieval_config\nKind: Method\nFile: crates/api/src/lib.rs",
    "entity_id": "crates/api/src/lib.rs::MatryoshkaConfig::retrieval_config:189",
    "entity_type": "symbol",
    "matched_symbols": [
      "retrieval_config"
    ],
    "path": "crates/api/src/lib.rs",
    "record_id": "semantic:symbol:crates/api/src/lib.rs::MatryoshkaConfig::retrieval_config:189",
    "score": 1.0455865859985352,
    "summary": "pub fn retrieval_config(&self) -> RetrievalConfig",
    "title": "Symbol MatryoshkaConfig::retrieval_config in crates/api/src/lib.rs"
  },
  {
    "description": "Symbol: MatryoshkaConfig::with_retrieval_config\nKind: Method\nFile: crates/api/src/lib.rs",
    "entity_id": "crates/api/src/lib.rs::MatryoshkaConfig::with_retrieval_config:182",
    "entity_type": "symbol",
    "matched_symbols": [
      "with_retrieval_config"
    ],
    "path": "crates/api/src/lib.rs",
    "record_id": "semantic:symbol:crates/api/src/lib.rs::MatryoshkaConfig::with_retrieval_config:182",
    "score": 1.0392932891845703,
    "summary": "pub fn with_retrieval_config(mut self, config: RetrievalConfig) -> Self",
    "title": "Symbol MatryoshkaConfig::with_retrieval_config in crates/api/src/lib.rs"
  }
]
```

Interpretation:

```text
The exact symbol record is returned directly ✅
The result is not collapsed into File crates/cli/src/main.rs ✅
Compact output keeps matched_symbols and removes noisy debug fields ✅
```

---

## Search: Symbol-Only Results

Command:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "where is resolve_retrieval_config defined" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --result-granularity symbol \
  --compact \
  --limit 3
```

Description:

Returns only symbol records. Use this when you explicitly want definitions,
methods, structs, classes, or named declarations instead of file or chunk cards.

Representative output:

```json
[
  {
    "description": "Symbol: resolve_retrieval_config\nKind: Function\nFile: crates/cli/src/main.rs",
    "entity_id": "crates/cli/src/main.rs::resolve_retrieval_config:498",
    "entity_type": "symbol",
    "matched_symbols": [
      "resolve_retrieval_config"
    ],
    "path": "crates/cli/src/main.rs",
    "record_id": "semantic:symbol:crates/cli/src/main.rs::resolve_retrieval_config:498",
    "score": 2.02943754196167,
    "summary": "fn resolve_retrieval_config(",
    "title": "Symbol resolve_retrieval_config in crates/cli/src/main.rs"
  }
]
```

---

## Search: Dense-Off Comparison

Command:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "where is resolve_retrieval_config defined" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --disable-dense \
  --no-dense-fallback \
  --no-collapse \
  --compact \
  --limit 4
```

Description:

Uses the same dense-enabled DB but disables dense query embedding and late-interaction at search time.

Expected output shape:

```json
[
  {
    "entity_type": "symbol",
    "matched_symbols": [
      "resolve_retrieval_config"
    ],
    "path": "crates/cli/src/main.rs",
    "record_id": "semantic:symbol:crates/cli/src/main.rs::resolve_retrieval_config:498",
    "summary": "fn resolve_retrieval_config(",
    "title": "Symbol resolve_retrieval_config in crates/cli/src/main.rs"
  }
]
```

Key difference from dense-on mode:

```text
No query embedding call is required.
No Late-interaction MaxSim evidence is produced.
Exact/FTS/symbol matching still works.
```

---

## Search: Dense Primary

Command:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "where is dense embedding disabled during indexing" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --retrieval-primary dense \
  --enable-dense \
  --limit 8
```

Description:

Runs dense-enabled search with dense as the requested primary retrieval mode.

Current M3 caveat:

```text
M3 makes dense configurable and skippable.
M4 will make SPLADE/dense lane scoring cleaner.
```

So in M3, even with `--retrieval-primary dense`, output can still include FTS/exact evidence.

Observed top result:

```text
crates/indexer/src/indexer.rs
```

This is correct because dense embedding gating during indexing lives in the indexer.

---

## Search: Dense Enabled but Late Interaction Disabled

Command:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "how are code chunks summarized with omlx" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --no-late-interaction \
  --limit 8
```

Description:

Uses dense record/query embeddings but skips late-interaction MaxSim.

Expected relevant areas:

```text
crates/enricher/src/mlx_chat.rs
crates/indexer/src/indexer.rs
crates/enricher/src/prompts.rs
```

---

## Prepare Command

Command:

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
  --json
```

Description:

Recommended high-level operation for IDE integration. It decides whether the repo needs:

- full index
- update
- repair
- rebuild search
- prewarm only

Expected output shape:

```json
{
  "status": "ready",
  "actions_taken": [
    "update",
    "prepare_results"
  ],
  "file_count": 35,
  "folder_count": 6,
  "symbol_count": 856,
  "semantic_record_count": 1956,
  "retrieval_index": {
    "semantic_records": 1956,
    "embedded_records": 1956,
    "fts_records": 1956,
    "late_vector_rows": 0,
    "dense_enabled": true
  }
}
```

Actual counts vary by DB and current repo state.

---

## Rebuild Semantic Index With Dense Disabled

Command:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs rebuild-semantic \
  /Users/rohit/cradle-embed \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --disable-dense \
  --progress-jsonl
```

Description:

Rebuilds semantic records and FTS from existing stored artifacts while skipping dense embeddings.

Important:

```text
rebuild-semantic does not accept --chunk-summary-model or --chunk-summary-concurrency.
```

Representative live corrected output:

```jsonl
{"type":"embedding_skipped","record_count":15,"reason":"dense embeddings disabled"}
{"type":"retrieval_index_health","report":{"semantic_records":15,"embedded_records":0,"fts_records":15,"late_vector_rows":0,"records_with_late_vectors":0,"retrieval_primary":"hybrid","dense_enabled":false,"dense_fallback_enabled":false,"late_interaction_enabled":false}}
```

---

## CLI Config Errors That Should Be Rejected

### Invalid: dense primary while dense disabled

Command:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "anything" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --retrieval-primary dense \
  --disable-dense
```

Expected output:

```text
--retrieval-primary dense requires dense embeddings; remove --disable-dense or choose another primary
```

### Invalid: dense fallback while dense disabled

Command:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "anything" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --disable-dense \
  --dense-fallback
```

Expected output:

```text
--dense-fallback requires dense embeddings; remove --disable-dense or use --no-dense-fallback
```

---

## Rust API Search Examples

### Default file-collapsed search

```rust
use matryoshka::{Matryoshka, MatryoshkaConfig, SearchOptions};

let api = Matryoshka::new(
    MatryoshkaConfig::new("/Users/rohit/cradle-embed")
        .with_db("/Users/rohit/cradle-embed/.matryoshka/matryoshka.db")
        .with_endpoint("http://127.0.0.1:44449", "2508")
        .with_models(
            "MercuriusDream--Qwen3.5-4B-MLX-mxfp8",
            "mlx-community--embeddinggemma-300m-bf16",
        ),
);

let hits = api.search(
    "where is dense embedding disabled during indexing",
    SearchOptions::default(),
)?;
```

### Chunk-level search

```rust
use matryoshka::{SearchOptions};
use matryoshka_search::SearchResultGranularity;

let hits = api.search(
    "how does update preserve unchanged generated chunk summaries",
    SearchOptions::default().with_result_granularity(SearchResultGranularity::Chunk),
)?;
```

### Dense disabled search

```rust
use matryoshka::{Matryoshka, MatryoshkaConfig, SearchOptions};

let api = Matryoshka::new(
    MatryoshkaConfig::new("/Users/rohit/cradle-embed")
        .with_db("/Users/rohit/cradle-embed/.matryoshka/matryoshka.db")
        .with_endpoint("http://127.0.0.1:44449", "2508")
        .with_models(
            "MercuriusDream--Qwen3.5-4B-MLX-mxfp8",
            "mlx-community--embeddinggemma-300m-bf16",
        )
        .with_dense_enabled(false)
        .with_dense_fallback_enabled(false),
);

let hits = api.search(
    "where is resolve_retrieval_config defined",
    SearchOptions::default(),
)?;
```

---

## Validation Commands

Command:

```bash
cd /Users/rohit/cradle-embed

cargo fmt --all
cargo check --workspace
cargo test -p matryoshka-search
cargo test --workspace
cargo build -p matryoshka-cli
```

Description:

Validates formatting, compilation, search tests, the full workspace, and the CLI binary.

Observed output summary:

```text
cargo check --workspace
  Finished `dev` profile ... ok

cargo test -p matryoshka-search
  15 passed; 0 failed

cargo test --workspace
  api facade: 7 passed
  indexer rust_core: 12 passed, 1 ignored live test
  parser: 16 passed
  resolver: 3 passed
  search: 15 passed
  watcher: 3 passed
  enricher: 5 passed, 2 ignored live oMLX tests
  doc-tests: passed

cargo build -p matryoshka-cli
  Finished `dev` profile ... ok
```

---

## Useful Search Queries

Try these against the current `cradle-embed` DB.

### Dense indexing behavior

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "where is dense embedding disabled during indexing" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --compact \
  --limit 5
```

Expected top area:

```text
crates/indexer/src/indexer.rs
```

### Update chunk-summary preservation

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "how does update preserve unchanged generated chunk summaries" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --result-granularity chunk \
  --compact \
  --limit 5
```

Expected top area:

```text
crates/indexer/src/indexer.rs
FullIndexer::refresh_chunk_summaries
```

### Doc comments attached to chunks

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "where are rust doc comments attached to code chunks" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --result-granularity chunk \
  --compact \
  --limit 5
```

Expected top areas:

```text
crates/parser/src/source_parser.rs
crates/indexer/src/indexer.rs
```

### oMLX chunk summarizer implementation

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "where is MlxChunkSummarizer implemented" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --result-granularity symbol \
  --compact \
  --limit 5
```

Expected top area:

```text
crates/enricher/src/mlx_chat.rs
MlxChunkSummarizer
```

### Search dense-disabled code path

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs search \
  "where does search skip query embeddings when dense is disabled" \
  --db /Users/rohit/cradle-embed/.matryoshka/matryoshka.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --enable-dense \
  --compact \
  --limit 5
```

Expected top area:

```text
crates/search/src/semantic_search.rs
SearchEngine::with_dense
SearchEngine::search
```

---

## Notes and Caveats

1. `--compact` only changes output shape. It does not change retrieval scoring.
2. `--result-granularity chunk` only returns existing `code_chunk` semantic records.
3. If newly-added code does not appear in search, run `update` or reindex first.
4. `--retrieval-primary splade` is reserved for M4. The SPLADE model is available at:

   ```text
   /Users/rohit/.omlx/models/naver--splade-code-06B
   ```

5. M3 makes dense optional and configurable. M4 will add SPLADE-primary retrieval.
