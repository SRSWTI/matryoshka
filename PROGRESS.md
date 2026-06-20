# Matryoshka Retrieval Overhaul — Progress & Architecture

This document tracks the multi-milestone effort to move Matryoshka from
file/folder/repo summaries + raw snippet/symbol embeddings toward
**function-level chunk summaries** with **SPLADE-primary, dense-optional**
retrieval.

The north-star rule for chunk summaries:

```
Good docstring/doc comment exists
  -> use it directly
  -> do not call LLM

No useful docstring/doc comment
  -> call local MLX/Raptor model
  -> generate 2-3-line summary
```

---

## Milestone Status

| Milestone | Status | Summary |
|---|---|---|
| M1 — Code chunk extraction + doc extraction | ✅ Complete | Parser emits structural `CodeChunkFact`s with full body + docstring/doc-comment summaries |
| M2 — Concurrent LLM chunk summaries + chunk semantic records | ✅ Complete | MLX/Raptor summarizes empty chunks concurrently; `code_chunk` semantic records built in target template; CLI/API flags wired |
| M2.5 — Real-repo verification + parser hardening | ✅ Complete | Verified DB invariants on `/Users/rohit/pi/packages/agent/src`; fixed TS prompt-string false positive and removed `Unknown` chunks from embedding path |
| M3 — Retrieval config, dense optional | ✅ Complete | Dense embeddings can be disabled at index/search/prepare/update/rebuild/prewarm time; health/progress treats dense-off as valid |
| M4 — SPLADE primary retrieval | ⬜ Not started | SPLADE sparse index + postings storage using omlx SPLADE model; SPLADE-first scoring |
| M5 — Measurement & recall comparison | ⬜ Not started | Retrieval diagnostics + eval harness for SPLADE-only vs hybrid/dense |

---

## Target Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              Source repo                 │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │            matryoshka-parser             │
                        │  tree-sitter AST -> symbols + chunks     │
                        │  docstring / doc-comment extraction      │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
                        ┌─────────────────────────────────────────┐
                        │           matryoshka-resolver            │
                        │  graph edges, folder hierarchy,          │
                        │  carries code_chunks into snapshot       │
                        └────────────────────┬────────────────────┘
                                             │
                                             ▼
          ┌──────────────────────────────────────────────────────────┐
          │                    matryoshka-indexer                      │
          │                                                            │
          │  parse -> resolve -> store snapshot                        │
          │  enrich file/folder/repo cards (existing)                  │
          │  summarize code chunks (M2: doc-first, LLM fallback)       │
          │  build semantic records (raw + card + code_chunk)          │
          │  embed (M3: optional dense) / SPLADE encode (M4)           │
          └────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
          ┌──────────────────────────────────────────────────────────┐
          │                 matryoshka-store-sqlite                    │
          │  files, folders, symbols, edges, cards                    │
          │  semantic_records + FTS5 + late-interaction vectors       │
          │  code_chunks table (M1)                                   │
          │  splade_postings table (M4, planned)                      │
          └────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
          ┌──────────────────────────────────────────────────────────┐
          │                   matryoshka-search                        │
          │                                                            │
          │  M1: code_chunk records are first-class searchable        │
          │  M3: dense becomes optional (RetrievalConfig)              │
          │  M4: SPLADE primary, dense optional fallback               │
          │  scoring: exact + SPLADE + (optional dense) + graph        │
          └────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
          ┌──────────────────────────────────────────────────────────┐
          │              matryoshka-api / matryoshka-cli               │
          │  prepare / index / update / search / read / prewarm        │
          │  M2: --chunk-summary-model, --chunk-summary-concurrency    │
          │  M3: --retrieval-primary, --enable/disable-dense           │
          │  M4: --enable-splade, --splade-model                       │
          └──────────────────────────────────────────────────────────┘
```

### Retrieval lane model (target after M4)

```
Lane 1 — Exact / symbolic
  path, symbol, signature, imports, calls, identifier subtokens
  good for: "handleResumeCountdown", "tests for X", "where is Foo used"

Lane 2 — Sparse semantic lexical (SPLADE)
  summary + signature + code
  good for: vocabulary mismatch, explainable matches, "rate limiting backoff"

Lane 3 — Dense embeddings (optional support leg)
  summary + signature + code
  good for: behavior/architecture queries, "where do we recover after handoff"
```

### Target scoring (SPLADE primary, dense optional)

```
final_score =
  exact_score   * 0.35
+ splade_score  * 0.50
+ dense_score   * 0.10   (only if dense enabled)
+ graph_score   * 0.10
+ card/chunk boosts
```

---

## Milestone 1 — Code Chunk Extraction ✅

### Goal

Extract one `CodeChunkFact` per function/method/class/struct symbol, preserving
the **full symbol body** (no truncation), and extract any leading docstring /
doc comment so the indexer can skip LLM summarization when a useful doc exists.

### Design rules honored

1. **AST drives chunk boundaries.** Tree-sitter is the primary symbol extractor;
   the line-based parsers (`parse_rust_symbol`, `parse_typescript_symbol`, etc.)
   are fallbacks only. `build_code_chunks` slices the full body from
   `symbol.start_line` to `symbol.end_line`.
2. **No truncation.** The full symbol body is stored in `code`. No 20-line cap,
   no "first 80 + last 40" heuristics.
3. **Docstring priority.** If a useful doc exists, it becomes the summary and
   `summary_source` records where it came from. No LLM call is needed for these.
4. **Generic docs are rejected.** `doc_is_useful` filters out `TODO`, `FIXME`,
   `placeholder`, `stub`, `handles event`, `helper`, `main`, and docs shorter
   than `MIN_USEFUL_DOC_SUMMARY_LEN` (12 chars).
5. **Contiguity enforced.** A doc is only attached if it is immediately above
   (Rust `///`, TS `/** */` / `//`) or inside the body (Python `"""`). A blank
   line or code between the doc and the symbol means no doc is attached.
6. **Exact symbols stay separate from embedding chunks.** Type aliases, constants,
   local declarations, and other `Unknown` chunk kinds remain available to the
   exact/symbol side, but are not converted into `CodeChunkFact`s for summary
   generation or semantic embedding.
7. **Backward compatible.** `RepositorySnapshot.code_chunks` is `#[serde(default)]`,
   so old snapshots still deserialize.

### Data model added (`crates/core-ir/src/models.rs`)

```rust
pub struct CodeChunkFact {
    pub chunk_id: String,
    pub file_id: String,
    pub symbol_id: Option<String>,
    pub path: String,
    pub symbol: Option<String>,
    pub qualified_name: Option<String>,
    pub kind: CodeChunkKind,
    pub signature: String,
    pub start_line: usize,
    pub end_line: usize,
    pub doc_summary: Option<String>,       // extracted doc, if useful
    pub generated_summary: Option<String>, // filled by M2 LLM summarizer
    pub summary: String,                   // doc_summary if useful, else generated
    pub summary_source: ChunkSummarySource,
    pub code: String,                      // full symbol body, no truncation
    pub source_hash: String,
}

pub enum CodeChunkKind { Function, Method, Class, Struct, Enum, Interface, Module, Unknown }

pub enum ChunkSummarySource { Docstring, DocComment, FileHeader, Llm, Heuristic, Empty }
```

Also:
- `SemanticEntityType::CodeChunk` variant added.
- `RepositorySnapshot.code_chunks: Vec<CodeChunkFact>` added (`#[serde(default)]`).
- `CHUNK_SCHEMA_VERSION` and `MIN_USEFUL_DOC_SUMMARY_LEN` constants added.
- `SymbolKind`, `CodeChunkKind`, `ChunkSummarySource` made `Copy`.

### Files touched in M1

| File | Change |
|---|---|
| `crates/core-ir/src/models.rs` | Added `CodeChunkFact`, `CodeChunkKind`, `ChunkSummarySource`, `SemanticEntityType::CodeChunk`, `RepositorySnapshot.code_chunks`, schema constants; made enums `Copy` |
| `crates/parser/src/source_parser.rs` | Extended `ParsedRepository` with `code_chunks`; `parse_file` builds chunks; `build_code_chunks`, `chunk_kind_for_symbol`, `doc_is_useful`, `doc_summary_source`, `extract_symbol_doc`, `extract_python_docstring`, `extract_leading_doc_comment`, `extract_typescript_doc`, `extract_jsdoc_block`, `extract_line_comments`, `typescript_variable_initializer_is_function_like`; parser regression coverage now 16 tests |
| `crates/resolver/src/graph_resolver.rs` | Carries `parsed.code_chunks` into `RepositorySnapshot.code_chunks` |
| `crates/store-sqlite/src/sqlite_store.rs` | New `code_chunks` table + indexes; `upsert_code_chunk(s)`, `load_all_code_chunks`, `load_code_chunks_for_file`, `delete_code_chunks_for_files`, `delete_code_chunks_for_paths`, `upsert_code_chunk_tx`; `replace_snapshot` clears/populates chunks; `prune_orphaned_artifacts` prunes orphan chunks; `OrphanPruneReport.code_chunks` field |
| `crates/indexer/src/indexer.rs` | `load_snapshot` loads `code_chunks`; `artifact_repair_set` handles `CodeChunk` records |
| `crates/search/src/semantic_search.rs` | `CodeChunk` treated as file-level result (collapses to parent file); `symbol_name_from_record` reads `qualified_name` from chunk metadata; `CollapsedHit::add` collects matched symbols from chunks; `is_file_level_result` includes `CodeChunk`; `hydrate_hits` handles `CodeChunk` |

### Doc extraction by language

| Language | Doc style | Where | Source label |
|---|---|---|---|
| Rust | `/// ...` line comments | immediately above symbol | `DocComment` |
| Rust | `//! ...` module docs | file header (not attached to functions) | `FileHeader` (planned) |
| Python | `"""..."""` triple-quoted | first string in body | `Docstring` |
| TypeScript | `/** ... */` JSDoc blocks | immediately above symbol | `DocComment` |
| TypeScript | `// ...` line comments | immediately above symbol | `DocComment` |

### Validation

- `cargo test --workspace` — all pass (16 parser tests incl. parser regression tests, 11 indexer integration tests, plus all search/store/api/watcher tests).
- Tested against generated sample repo (Rust + Python + TS) — 16 chunks, correct doc extraction.
- Tested against `/Users/rohit/pi` (real repo) — 24,695 chunks (2,363 `doc_comment`, 22,332 `empty`).
- Cross-verified two documented chunks (`isContextOverflow`, `sanitizeSurrogates`) against actual source — summaries match the JSDoc exactly.

### Bug found and fixed during M1

**TypeScript JSDoc leakage.** The original `extract_jsdoc_block` walked upward
without checking contiguity, so an undocumented function below a documented one
would inherit the wrong doc. Fixed by:
1. Requiring the line immediately above the symbol to be a comment.
2. Requiring every line inside a JSDoc block to start with `*` or be the `/**` opener.
3. Returning `None` if no `/**` opener is found.

Two regression tests added:
`typescript_jsdoc_does_not_leak_from_unrelated_function_above`,
`typescript_jsdoc_separated_by_blank_line_is_not_attached`.

After the fix, `/Users/rohit/pi` `doc_comment` count dropped 2719 → 2363
(356 false positives removed), and 0 empty chunks have a non-empty summary.

**TypeScript prompt string / unknown chunk over-inclusion.** A real DB check on
`/Users/rohit/pi/packages/agent/src` showed one `empty` function chunk for
`SUMMARIZATION_PROMPT` because the TS fallback classified a string constant as a
function when its text contained the word `function`. It also showed many
`Unknown` chunks from constants/type aliases/local declarations being sent to
chunk summarization, which is too noisy for the function/class/method chunking
model.

Fixed by:
1. Detecting TS variable function-ness from the initializer (`= ...`) instead of
   searching the whole declaration text for `function`.
2. Rejecting string/template literal initializers before checking `=>` or
   function expressions.
3. Skipping `CodeChunkKind::Unknown` in `build_code_chunks`, preserving exact
   symbols while keeping embedding chunks focused on functions/classes/methods
   and related structural declarations.

Regression test added:
`typescript_prompt_string_containing_function_is_not_chunked`.

Parser-only verification on `/Users/rohit/pi/packages/agent/src` after the fix:
`547` chunks total, `0 Unknown`; kinds are `185 Function`, `235 Method`,
`16 Class`, `111 Interface`; sources are `29 DocComment`, `518 Empty` before
LLM fallback.

### Test script

`scripts/test_milestone1_chunks.sh [REPO_PATH]` — builds the CLI, indexes a repo
(or a generated sample) offline, and dumps the `code_chunks` table for inspection.

---

## Milestone 2 — Concurrent LLM Chunk Summaries + Chunk Semantic Records ✅

### Goal

For every chunk with `summary_source == Empty`, call the local MLX/Raptor model
to generate a 2-3 line summary. Then build `code_chunk` semantic records in the
target template and persist them so search can use them.

### Progress

#### Done

- **`ChunkSummarizer` trait + `ChunkSummaryDraft`** (`crates/enricher/src/lib.rs`)
  - `ChunkSummaryDraft { chunk_id, summary }` — keyed by `chunk_id` so the
    indexer can map summaries back onto chunks.
  - `trait ChunkSummarizer { fn summarize_chunks(&self, chunks: &[CodeChunkFact]) -> Result<Vec<ChunkSummaryDraft>>; }`

- **Prompt builder** (`crates/enricher/src/prompts.rs`)
  - `chunk_summary_prompt(chunk)` — plain-text user message:
    ```
    Summarize this code chunk in 2-3 concise lines.

    path: crates/foo/src/bar.rs
    symbol: Foo::resume
    kind: method
    code:
    fn resume(&mut self) {
        self.countdown.cancel();
        self.mode = Mode::Attack;
    }
    ```
  - `chunk_summary_system_prompt()` — asks for strict JSON `{"summary": "..."}`.
  - `DEFAULT_CHUNK_SUMMARY_MODEL = "srswti--bodega-raptor-90m"`.
  - `MAX_CHUNK_CODE_CHARS = 8_000` — prompt-side cap only; stored `code` is never truncated.

- **`HeuristicChunkSummarizer`** (`crates/enricher/src/heuristic.rs`)
  - Fallback that produces a grounded one-line summary from symbol/kind/path/signature.
  - No LLM call. Used when MLX is unavailable or as a last resort.

- **`MlxChunkSummarizer`** (`crates/enricher/src/mlx_chat.rs`)
  - Concurrent chunk summarizer backed by an OpenAI-compatible chat endpoint (omlx).
  - Sends one request per chunk in parallel using a rayon thread pool.
  - `enable_thinking: false` passed via `chat_template_kwargs` to skip reasoning.
  - JSON-schema-constrained response (`SummaryDraft { summary }`).
  - Builders: `.with_model()`, `.with_concurrency()`, `.with_max_tokens()`.
  - Graceful partial-failure: collects successes, errors only if everything fails.
  - Reuses the existing `ChatRequest`/`parse_chat_response`/`cleanup_summary` helpers.

- **Live-tested against omlx** (`http://127.0.0.1:44449`, `srswti--bodega-raptor-90m`)
  - Single chunk: `"The function sets the mode to 'Attack' and returns true, canceling the countdown."`
  - 3 concurrent chunks in 2.16s — continuous batching on the omlx side handled it cleanly.
  - Tests: `mlx_chunk_summarizer_live`, `mlx_chunk_summarizer_concurrent_live` (ignored by default).

- **`rayon` added** to `crates/enricher/Cargo.toml` for the concurrent thread pool.

#### Done (indexer wiring + chunk semantic records)

- `crates/indexer/src/indexer.rs`:
  - `FullIndexer` now generic over `S: ChunkSummarizer` (3rd type param, defaults to `HeuristicChunkSummarizer`).
  - `refresh_chunk_summaries()` method: filters `Empty` chunks in affected files, calls `ChunkSummarizer`, maps drafts back by `chunk_id`, sets `generated_summary`/`summary`/`summary_source = Llm`, persists via `upsert_code_chunks`, builds `code_chunk` semantic records.
  - Heuristic fallback if LLM fails entirely (summaries prefixed with `[heuristic]`).
  - Incremental: only summarizes chunks in `affected_file_ids`.
  - `code_chunk_semantic_records()` helper builds records in the target template.
  - Chunk records added to `raw_records` so they get embedded + FTS-indexed.
- `crates/core-ir/src/models.rs`: added `EnrichingChunks`/`EnrichedChunks` progress events plus `EnrichingChunkBatch`/`EnrichedChunkBatch` for batch-level progress.
- `crates/api/src/lib.rs`: `MatryoshkaConfig` gains `chunk_summary_enabled`, `chunk_summary_model`, `chunk_summary_concurrency` + builders; `indexer_progress_state` handles all new events; `run_update_once_with_progress` passes concurrency to `MlxChunkSummarizer`.
- `crates/cli/src/main.rs`: `--chunk-summary-model`, `--chunk-summary-concurrency`, `--no-chunk-summaries` flags on `prepare`, `index`, and `update`; offline uses `HeuristicChunkSummarizer`, online uses `MlxChunkSummarizer`; `run_update_once`/`run_rebuild_semantic_once` helpers accept and forward chunk summary config.
- `crates/enricher/src/lib.rs`: `ChunkSummarizer` trait gains `summarize_chunks_with_progress` with a per-batch callback.
- `crates/enricher/src/mlx_chat.rs`: `MlxChunkSummarizer` implements `summarize_chunks_with_progress` with 32-chunk batches, emitting progress per batch.
- All call sites updated to pass a chunk summarizer (tests, CLI, API, prepare, watch loop, prewarm).

#### Live end-to-end test

Indexed a 2-function Rust file (1 documented, 1 undocumented) against `http://127.0.0.1:44449` with `srswti--bodega-raptor-90m`:
- `documented_function` → `doc_comment`, no LLM call.
- `undocumented_function` → `llm`, summary: `"The undocumented function takes two i32 parameters, sums them, and multiplies them, returning the sum."`
- Chunk semantic record built in target template (path/symbol/kind/signature/summary/code).
- `enriching_chunks`/`enriched_chunks` progress events emitted correctly.

Additional DB verification on `/Users/rohit/pi/packages/agent/src/.matryoshka/matryoshka.db` before the TS over-inclusion fix:
- `code_chunks` existed and was populated: `47 doc_comment`, `1137 llm`, `1 empty`.
- Doc-comment chunks had `doc_summary` and no `generated_summary`; LLM chunks had `generated_summary` and no `doc_summary`.
- `agentLoop`, `agentLoopContinue`, and `runLoop` correctly preserved source docs as `doc_comment`; undocumented chunks like `AgentEventSink` used `llm` fallback.
- `CodeChunk` semantic records used the target template (`path`, `symbol`, `kind`, `signature`, `summary`, `code`).
- The one missing `CodeChunk` semantic record was the empty/no-summary `SUMMARIZATION_PROMPT` false-positive fixed in M1 parser cleanup above.
- `embedding_batch` / `embedded_batch` events are still from the existing dense embedding pipeline and will become configurable in M3.

### LLM call shape (one chunk per request)

Input:
```
Summarize this code chunk in 2-3 concise lines.

path: crates/foo/src/bar.rs
symbol: Foo::resume
kind: method
code:
fn resume(&mut self) {
    self.countdown.cancel();
    self.mode = Mode::Attack;
}
```

Output:
```json
{"summary": "Cancels the active countdown and switches the object into attack mode."}
```

The caller already knows the `chunk_id` it sent, so the returned summary is
mapped back onto that chunk. Concurrency is handled by firing many requests in
parallel (the omlx server already does continuous batching internally).

### omlx endpoint config (live-tested)

```
base_url: http://127.0.0.1:44449
api_key: 2508
model: srswti--bodega-raptor-90m
chat_template_kwargs: {"enable_thinking": false}
response_format: json_schema (SummaryDraft { summary: String })
```

### Chunk semantic record template (implemented)

```
path: crates/foo/src/bar.rs
symbol: Foo::handle_resume_countdown
kind: method
signature: fn handle_resume_countdown(&mut self, ...)
summary: Resumes attack mode after handoff, cancels the countdown, and updates state.
code:
fn handle_resume_countdown(...) {
    ...
}
```

Metadata: `kind=code_chunk`, `summary_source`, `symbol_id`, `qualified_name`, `start_line`, `end_line`, `language`.

### Incremental behavior

Implemented:
- If changed file → regenerate summaries only for chunks in changed files / affected files.
- Only chunks with `summary_source == Empty` are sent to the LLM.
- Chunks with useful docs (`DocComment` / `Docstring`) are persisted directly and never sent to the LLM.

Still worth tightening in later cleanup:
- Make the skip/reuse decision explicit in progress events when a file hash is unchanged.
- Add a small chunk-summary cache keyed by `chunk_id + source_hash` if we need more aggressive reuse across rebuilds.

---

## Milestone 2.5 — Real-Repo Verification + Parser Hardening ✅

### Goal

Verify the full M1/M2 pipeline against a real TypeScript repo DB, confirm the
summary-source invariants, explain the remaining dense embedding progress events,
and harden the parser against noisy/non-structural chunks before starting M3.

### What we verified

DB inspected:

```text
/Users/rohit/pi/packages/agent/src/.matryoshka/matryoshka.db
```

Observed before parser cleanup:

```text
code_chunks:
  doc_comment: 47
  llm:         1137
  empty:       1
```

Invariants checked:
- `doc_comment` chunks had `doc_summary` populated and `generated_summary = null`.
- `llm` chunks had `generated_summary` populated and `doc_summary = null`.
- `agentLoop`, `agentLoopContinue`, and `runLoop` correctly used their source JSDoc as `doc_comment` summaries.
- Undocumented chunks such as `AgentEventSink` used the LLM fallback.
- `CodeChunk` semantic records existed and used the target text template:
  `path`, `symbol`, `kind`, `signature`, `summary`, `code`.

### Bug found and fixed

The one remaining `empty` chunk was:

```text
harness/compaction/compaction.ts::SUMMARIZATION_PROMPT:382
```

Root cause: the TypeScript fallback classified a string constant as a function
because the prompt text contained the word `function`.

Fixes in `crates/parser/src/source_parser.rs`:
1. Function-like TS variables are now detected from the initializer after `=`.
2. String/template literal initializers are rejected before checking for `=>` or
   `function` expressions.
3. `build_code_chunks()` skips `CodeChunkKind::Unknown`, so constants, type
   aliases, and local declarations stay on the exact/symbol side instead of being
   sent to LLM summarization and semantic embedding.
4. Added regression test:
   `typescript_prompt_string_containing_function_is_not_chunked`.

Parser-only verification after the fix on `/Users/rohit/pi/packages/agent/src`:

```text
total chunks: 547
Unknown:      0

Kinds:
  Function:  185
  Method:    235
  Class:     16
  Interface: 111

Sources before LLM:
  DocComment: 29
  Empty:      518
```

### Dense progress clarification

The observed events:

```json
{"type":"embedding_batch","records_in_batch":64}
{"type":"embedded_batch","records_in_batch":64}
```

are from the existing dense embedding pipeline, not the chunk summarizer. A
`record` here means one `semantic_records` row (`CodeChunk`, `Symbol`, `Snippet`,
`File`, `Folder`, or `Repo`) being sent to the dense embedder. M3 will make this
path optional/skippable.

### Validation

- `cargo test -p matryoshka-parser` — pass, 16 parser tests.
- `cargo test --workspace` — pass.
- `cargo build --release --bin matryoshka-rs` — pass.

### Files touched in M2.5

| File | Change |
|---|---|
| `crates/parser/src/source_parser.rs` | Tightened TypeScript function-like variable detection; skipped `Unknown` code chunks; added regression test |
| `PROGRESS.md` | Documented DB verification, parser cleanup, dense-event clarification, and updated roadmap |

### Reindex note

Existing DBs created before this fix still contain the old noisy chunks. Use a
fresh DB path or reindex after rebuilding the local binary:

```bash
/Users/rohit/cradle-embed/target/release/matryoshka-rs index /Users/rohit/pi/packages/agent/src \
  --db /Users/rohit/pi/packages/agent/src/.matryoshka/matryoshka_after_parser_fix.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --chunk-summary-model srswti--bodega-raptor-90m \
  --chunk-summary-concurrency 6 \
  --progress-jsonl
```

---

## Milestone 3 — Retrieval Config, Dense Optional ✅

### Goal

Make dense embeddings genuinely optional so we can run `exact + FTS` now, and
`exact + SPLADE` in M4, without embedding the query or semantic records. Dense
must remain available as a configurable fallback/support leg.

### Implemented behavior

- `--disable-dense` skips document embedding during `index`, `prepare`, `update`,
  and `rebuild-semantic`.
- Dense-disabled indexing still writes:
  - `semantic_records`
  - `semantic_records_fts`
  - code chunk records from M2
  - file/folder/repo cards
- Dense-disabled indexing does **not** write:
  - dense vectors in `semantic_records.payload_json.embedding`
  - `semantic_late_vectors`
- Dense-disabled search calls no query embedder path and skips late-interaction
  MaxSim scoring.
- Dense-disabled health is valid when:
  - `semantic_records > 0`
  - `fts_records > 0`
  - `embedded_records == 0`
  - `records_with_late_vectors == 0`
- Default behavior remains backward compatible: dense is enabled unless explicitly
  disabled or `--retrieval-primary fts` is selected without `--enable-dense`.

### Data model / progress events

`crates/core-ir/src/models.rs`:

```rust
pub enum RetrievalPrimary { Fts, Splade, Dense, Hybrid }

pub struct RetrievalConfig {
    pub primary: RetrievalPrimary,
    pub dense_enabled: bool,
    pub dense_fallback_enabled: bool,
}
```

`RetrievalIndexReport` now carries retrieval mode flags so CLI/API readiness can
judge dense-off indexes correctly:

```rust
pub struct RetrievalIndexReport {
    pub semantic_records: usize,
    pub embedded_records: usize,
    pub fts_records: usize,
    pub late_vector_rows: usize,
    pub records_with_late_vectors: usize,
    pub retrieval_primary: RetrievalPrimary,
    pub dense_enabled: bool,
    pub dense_fallback_enabled: bool,
    pub late_interaction_enabled: bool,
}
```

New progress event:

```json
{"type":"embedding_skipped","record_count":13,"reason":"dense embeddings disabled"}
```

This makes dense-off runs visibly active instead of appearing stuck between chunk
enrichment and database writes.

### CLI/API configuration

CLI flags added to relevant commands (`prepare`, `index`, `update`, `watch`,
`rebuild-semantic`, `search`, `op`, `prewarm`, `read-bundle`):

```text
--retrieval-primary <fts|splade|dense|hybrid>
--enable-dense
--disable-dense      # alias: --no-dense-embeddings
--dense-fallback
--no-dense-fallback
```

Rust API additions on `MatryoshkaConfig`:

```rust
.with_retrieval_primary(RetrievalPrimary::Hybrid)
.with_dense_enabled(false)
.with_dense_fallback_enabled(false)
.with_retrieval_config(RetrievalConfig { ... })
.retrieval_config()
```

### Files touched in M3

| File | Change |
|---|---|
| `crates/core-ir/src/models.rs` | Added `RetrievalPrimary`, `RetrievalConfig`; extended `RetrievalIndexReport`; added `EmbeddingSkipped` progress event |
| `crates/search/src/semantic_search.rs` | Added `SearchEngine::with_dense`; skips query embedding and late-interaction scoring when dense is disabled |
| `crates/indexer/src/indexer.rs` | Added `FullIndexer::with_retrieval_config` / `with_dense_embeddings_enabled`; gates record embedding and late-vector generation; emits `EmbeddingSkipped`; reports dense-aware health |
| `crates/indexer/src/lib.rs` | Re-exported `RetrievalConfig` and `RetrievalPrimary` |
| `crates/api/src/lib.rs` | Added dense/retrieval config builders; wired config through prepare/update/rebuild/prewarm/search; dense-aware readiness and progress-state mapping |
| `crates/cli/src/main.rs` | Added retrieval flags; wired config through index/update/watch/rebuild/search/op/prewarm/read-bundle/prepare helpers; dense-aware prepare summaries and daemon args |
| `crates/indexer/tests/rust_core.rs` | Added dense-disabled index/search regression test |
| `crates/api/tests/facade.rs` | Added dense-disabled prepare/search lifecycle regression test |
| `PROGRESS.md` | Marked M3 complete and documented behavior/validation |

### Validation

Automated:

```bash
cargo check --workspace
cargo test -p matryoshka-indexer dense_disabled_indexing_keeps_fts_search_without_embeddings
cargo test -p matryoshka-api prepare_with_dense_disabled_reaches_ready_without_embeddings
cargo test --workspace
```

All passed. Workspace test count after M3 includes:
- API facade: 7 passed
- Indexer integration: 12 passed, 1 ignored live MLX test
- Parser: 16 passed
- Search: 15 passed
- Resolver/watcher/enricher tests passed; live oMLX tests remain ignored by default.

Manual CLI smoke test:

```bash
./target/debug/matryoshka-rs index crates/indexer/tests/fixtures/mini_repo \
  --db /tmp/matryoshka_m3_dense_off.db \
  --offline \
  --disable-dense \
  --progress-jsonl
```

Observed:

```json
{"type":"embedding_skipped","record_count":13,"reason":"dense embeddings disabled"}
{"type":"retrieval_index_health","report":{"semantic_records":15,"embedded_records":0,"fts_records":15,"late_vector_rows":0,"records_with_late_vectors":0,"retrieval_primary":"hybrid","dense_enabled":false,"dense_fallback_enabled":false,"late_interaction_enabled":false}}
```

Dense-disabled search also worked without late-interaction evidence:

```bash
./target/debug/matryoshka-rs search "where is get_env_api_key defined" \
  --db /tmp/matryoshka_m3_dense_off.db \
  --offline \
  --disable-dense \
  --no-dense-fallback
```

Top hit: `src/config/env.py`, with exact/FTS/symbol explanations and no
`Late-interaction MaxSim` reason.

### M3 edge-case matrix — run on 2026-06-20

After the first M3 validation pass, a larger offline CLI matrix was run against
temporary repos/DBs under:

```text
/tmp/matryoshka_m3_edge_matrix
```

The matrix used the freshly built local binary:

```bash
cargo build -p matryoshka-cli
/Users/rohit/cradle-embed/target/debug/matryoshka-rs ...
```

Full machine-readable report:

```text
/tmp/matryoshka_m3_edge_matrix/report.json
```

Raw script summary:

```text
total: 36
pass:  31
fail:  5
error: 0
```

Manual review of those rows produced **4 real failures** plus **2 caveats**:
A7, A8, C7, and G3 are real failures; C2 and D6 work but have caveats.

#### Fresh index + config flows

| ID | Flow tested | Result | Observed output / notes |
|---|---|---|---|
| A1 | Fresh `index --offline` default dense | ✅ Pass | `semantic_records=18`, `fts_records=18`, `embedded_records=15`, `late_vectors=410`, `embedding_batch` emitted |
| A2 | Fresh `index --offline --disable-dense` | ✅ Pass | `semantic_records=18`, `fts_records=18`, `embedded_records=0`, `late_vectors=0`, `embedding_skipped` emitted |
| A3 | Fresh `index --retrieval-primary fts` | ✅ Pass | Defaults dense off: `embedded_records=0`, `late_vectors=0`, `embedding_skipped` emitted |
| A4 | Fresh `index --retrieval-primary fts --enable-dense` | ✅ Pass | FTS primary but dense enabled: `embedded_records=15`, `late_vectors=410` |
| A5 | `--enable-dense --disable-dense` | ✅ Pass | CLI errors: `choose either --enable-dense or --disable-dense, not both` |
| A6 | `--dense-fallback --no-dense-fallback` | ✅ Pass | CLI errors: `choose either --dense-fallback or --no-dense-fallback, not both` |
| A7 | `--retrieval-primary dense --disable-dense` | ❌ Fails validation expectation | CLI currently accepts this invalid config. Needs a guard: dense primary requires dense enabled. |
| A8 | `--disable-dense --dense-fallback` | ❌ Fails validation expectation | CLI currently accepts this ambiguous config. Needs a guard: dense fallback requires dense enabled and must conflict with disable-dense. |

#### Search flows

| ID | Flow tested | Result | Observed output / notes |
|---|---|---|---|
| B1 | Search default against dense-enabled DB | ✅ Pass | Hits: `src/util.rs`, `src/lib.rs`, `src/auth.rs`; `Late-interaction MaxSim` evidence present |
| B2 | Search `--disable-dense --no-dense-fallback` against dense-enabled DB | ✅ Pass | Same relevant hits; `Late-interaction MaxSim` evidence absent |
| B3 | Search default dense against dense-disabled DB | ✅ Pass | Search still works via exact/FTS; hits include `src/util.rs`; no late evidence because DB has no late vectors |
| B4 | Search dense disabled against dense-disabled DB | ✅ Pass | Search works via exact/FTS; no late evidence |

#### Update / incremental flows with dense disabled

| ID | Flow tested | Result | Observed output / notes |
|---|---|---|---|
| C1 | `update --disable-dense` with no source changes | ✅ Pass | No `enriching_file` events; stats unchanged; `embedded_records=0`, `late_vectors=0` |
| C2 | Modify one file, then `update --disable-dense` | ⚠️ Pass with caveat | Only changed file enriched: `enriching_file_paths=['src/util.rs']`; search found new `cache_key`; dense stayed zero. Caveat: `chunk_sources` became `{'empty': 2, 'llm': 1, 'doc_comment': 1}`, meaning unchanged no-doc chunks lost generated summaries during the update. This needs a fix before M4. |
| C3 | Add one file, then update | ✅ Pass | `enriching_file_paths=['src/added.rs']`; new path had semantic records and was searchable |
| C4 | Delete one file, then update | ✅ Pass | Deleted path had `0` semantic records; search no longer returned deleted path |
| C5 | Rename file | ✅ Pass | Old path `src/rename_me.rs` records `0`; new path `src/renamed.rs` records `5` |
| C6 | Move file between folders | ✅ Pass | Old path records `0`; new path `src/nested/move_me.rs` records `5`; new folder record present |
| C7 | Add then remove Rust doc comment on existing no-doc function | ❌ Fail | Expected chunk source to change `llm -> doc_comment -> llm/empty`; observed `initial=['llm']`, `after_add=['llm']`, `after_remove=['llm']`. Root cause: when no chunks need summarization, `refresh_chunk_summaries()` returns semantic records without upserting updated doc-only chunks. |
| C8 | Modify function body only | ✅ Pass | New body term was searchable in `src/docflow.rs` |
| C9 | Rename/signature change | ✅ Pass | Old symbol count `0`; new symbol count `1` |

#### Prepare flows

| ID | Flow tested | Result | Observed output / notes |
|---|---|---|---|
| D1 | `prepare --offline` new repo default dense | ✅ Pass | `status=ready`; `embedded_records=15`, `late_vectors=410` |
| D2 | `prepare --offline --disable-dense` new repo | ✅ Pass | `status=ready`; actions `['index', 'prepare_results']`; `embedded_records=0`, `late_vectors=0` |
| D3 | Prepare existing healthy dense-disabled DB | ✅ Pass | `status=ready`; actions `['update', 'prepare_results']` |
| D4 | Delete ready marker, then prepare dense-disabled DB | ✅ Pass | `status=ready`; actions `['update', 'prepare_results']`; marker recreated |
| D5 | Delete `semantic_records`, FTS, and late vectors, then prepare dense-disabled DB | ✅ Pass | Actions `['rebuild_search', 'prepare_results']`; rebuilt `semantic_records=15`, `fts_records=15`, `embedded_records=0` |
| D6 | Delete file/folder/repo cards, then prepare dense-disabled DB | ⚠️ Works but action label caveat | Cards were repaired (`file_cards=3`, `folder_cards=2`) and status was ready, but actions were `['update', 'prepare_results']` instead of an explicit `repair`. This is observability-only; behavior works. |
| D7 | Prepare dense-disabled DB again with default dense enabled | ✅ Pass | Detected dense missing and rebuilt search: actions `['rebuild_search', 'prepare_results']`; `embedded_records=12`, `late_vectors=389` |
| D8 | Prepare dense-enabled DB with `--disable-dense` | ✅ Pass | `status=ready`; dense-off health accepted. Existing dense vectors remain in DB (`embedded_records=15`, `late_vectors=410`) but are ignored by dense-disabled search. |

#### Rebuild flows

| ID | Flow tested | Result | Observed output / notes |
|---|---|---|---|
| E1 | `rebuild-semantic --offline` default dense | ✅ Pass | `semantic_records=15`, `fts_records=15`, `embedded_records=12`, `late_vectors=389` |
| E2 | `rebuild-semantic --offline --disable-dense` after dense DB | ✅ Pass | Dense artifacts purged for rebuilt records: `embedded_records=0`, `late_vectors=0`, `embedding_skipped` emitted |

#### Watch bounded probe

| ID | Flow tested | Result | Observed output / notes |
|---|---|---|---|
| F1 | Start `watch --disable-dense --skip-startup-update` and terminate after 1.5s | ✅ Pass | Watch started and printed: `watching /tmp/matryoshka_m3_edge_matrix/f_watch every 500ms with 500ms debounce`; process was intentionally terminated (`returncode=-15`). This verifies CLI config acceptance/startup only, not a full file-change watch loop. |

#### Chunk-summary flows

| ID | Flow tested | Result | Observed output / notes |
|---|---|---|---|
| G1 | Documented Rust chunk | ✅ Pass | `documented_chunk` source was `['doc_comment']` |
| G2 | Undocumented Rust chunk in offline mode | ✅ Pass with naming caveat | Summary was generated and source stored as `['llm']`. Because offline uses `HeuristicChunkSummarizer`, we may want to store `heuristic` instead of `llm` for more accurate provenance. |
| G3 | `index --offline --disable-dense --no-chunk-summaries` | ❌ Fail | Expected no generated summaries for undocumented chunks; observed `sources=['llm']`. Root cause: offline `index` / `update` branches do not call `.with_chunk_summary_enabled(!no_chunk_summaries)`, so the flag is ignored offline. Online branch is wired. |

#### Retrieval-health edge flow

| ID | Flow tested | Result | Observed output / notes |
|---|---|---|---|
| H1 | Delete FTS rows, then prepare dense-disabled DB | ✅ Pass | Detected missing FTS and rebuilt search: actions `['rebuild_search', 'prepare_results']`; `fts_records=15`, `embedded_records=0` |

### M3 live oMLX matrix — run on 2026-06-20

A smaller live matrix was also run against the actual oMLX server:

```text
base_url: http://127.0.0.1:44449
api_key: 2508
chat_model: MercuriusDream--Qwen3.5-4B-MLX-mxfp8
chunk_summary_model: srswti--bodega-raptor-90m
embedding_model: mlx-community--embeddinggemma-300m-bf16
```

Model probe:

```text
GET /v1/models -> 200
srswti--bodega-raptor-90m = present
MercuriusDream--Qwen3.5-4B-MLX-mxfp8 = present
mlx-community--embeddinggemma-300m-bf16 = present
```

Temporary root and report:

```text
/tmp/matryoshka_m3_omlx_matrix
/tmp/matryoshka_m3_omlx_matrix/report.json
```

Raw matrix result was `5/6` because the first `rebuild-semantic` command was
called with unsupported chunk-summary flags. Rerunning `rebuild-semantic` with
its actual CLI surface passed, so the functional live result is **6/6 pass**
with the same incremental chunk-summary caveat already found offline.

#### Live oMLX results

| ID | Flow tested | Result | Observed output / notes |
|---|---|---|---|
| L1 | Live `index` with oMLX, dense disabled | ✅ Pass | `semantic_records=16`, `fts_records=16`, `embedded_records=0`, `late_vectors=0`, `code_chunks=3`, `chunk_sources={'llm': 2, 'doc_comment': 1}`, `embedding_skipped` emitted |
| L2 | Live search against L1 DB with `--disable-dense --no-dense-fallback` | ✅ Pass | Hits included `src/util.rs`; no `Late-interaction MaxSim` evidence |
| L3 | Modify `src/util.rs`, then live `update --disable-dense` | ⚠️ Pass with caveat | Only changed file enriched: `enriching_file_paths=['src/util.rs']`; search found `live_update_marker`; `embedded_records=0`, `late_vectors=0`. Same caveat as offline: unchanged `api_entry` chunk became `summary_source=empty`, so unchanged generated summaries are not preserved correctly. |
| L4 | Live `rebuild-semantic --disable-dense` | ✅ Pass after corrected command | Correct command emitted `embedding_skipped` with `record_count=15`; final stats `semantic_records=15`, `fts_records=15`, `embedded_records=0`, `late_vectors=0`. First attempt failed only because `rebuild-semantic` does not accept `--chunk-summary-model` / `--chunk-summary-concurrency`. |
| L5 | Live `prepare --disable-dense` on fresh repo | ✅ Pass | `status=ready`, actions `['index', 'prepare_results']`; `semantic_records=16`, `fts_records=16`, `embedded_records=0`, `late_vectors=0` |
| L6 | Live `index` dense-enabled tiny repo | ✅ Pass | Verified embedding endpoint path: `semantic_records=8`, `fts_records=8`, `embedded_records=7`, `late_vectors=239`, `embedding_batch`/`embedded_batch` emitted |

Live chunk summaries observed from oMLX/Raptor:

```text
api_entry -> llm -> The function `api_entry` returns a string from the `util::helper()` function.
helper -> doc_comment -> Returns the original helper value.
undocumented_live_chunk -> llm -> A function returns a string 'live-undocumented'.
live_update_marker -> llm -> A function named live_update_marker returns a string 'live-update-marker'.
```

Corrected live rebuild command:

```bash
/Users/rohit/cradle-embed/target/debug/matryoshka-rs rebuild-semantic \
  /tmp/matryoshka_m3_omlx_matrix/live_dense_off \
  --db /tmp/matryoshka_m3_omlx_matrix/live_dense_off.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --disable-dense \
  --progress-jsonl
```

Observed corrected rebuild output:

```json
{"type":"embedding_skipped","record_count":15,"reason":"dense embeddings disabled"}
{"type":"retrieval_index_health","report":{"semantic_records":15,"embedded_records":0,"fts_records":15,"late_vector_rows":0,"records_with_late_vectors":0,"retrieval_primary":"hybrid","dense_enabled":false,"dense_fallback_enabled":false,"late_interaction_enabled":false}}
```

#### Fixes identified before M4

The offline and live matrices found the following concrete follow-ups, which are
marked completed in the next section:

1. ✅ **CLI config validation** (`crates/cli/src/main.rs`):
   - reject `--retrieval-primary dense --disable-dense`
   - reject `--disable-dense --dense-fallback`
2. ✅ **Offline no-chunk-summaries wiring** (`crates/cli/src/main.rs`):
   - apply `.with_chunk_summary_enabled(!no_chunk_summaries)` in the offline `index` and `update` branches, not just online branches.
3. ✅ **Incremental chunk-summary preservation** (`crates/indexer/src/indexer.rs`):
   - preserve existing generated summaries for unchanged chunks during `update`.
   - do not overwrite unchanged no-doc chunks back to `summary_source=empty`.
4. ✅ **Doc-comment update persistence** (`crates/indexer/src/indexer.rs`):
   - if a changed chunk now has a useful doc comment/docstring and no LLM call is needed, upsert the updated `CodeChunkFact` anyway.
5. ✅ **Provenance cleanup**:
   - oMLX/Raptor chunk summaries persist as `summary_source=llm`; heuristic summaries persist as `summary_source=heuristic`.

### M3 follow-up fixes — completed on 2026-06-20

The above issues were fixed one-by-one and validated with live oMLX only (no
offline test matrix was run for this pass).

#### Code changes

| Area | Files | Fix |
|---|---|---|
| CLI validation | `crates/cli/src/main.rs` | `resolve_retrieval_config()` now rejects `--retrieval-primary dense --disable-dense` and `--disable-dense --dense-fallback` |
| Chunk summary flag wiring | `crates/cli/src/main.rs` | Offline CLI `index`/`update` and CLI prepare helpers now pass `.with_chunk_summary_enabled(...)`; online path was already wired |
| Summary provenance | `crates/enricher/src/lib.rs`, `crates/enricher/src/heuristic.rs`, `crates/enricher/src/mlx_chat.rs`, `crates/indexer/src/indexer.rs` | `ChunkSummaryDraft` now carries `source`; oMLX summaries store `llm`, heuristic summaries store `heuristic` |
| Incremental summary preservation | `crates/indexer/src/indexer.rs` | `refresh_chunk_summaries()` merges unchanged chunks from the existing DB so generated summaries are preserved when unrelated files change |
| Doc-comment add/remove | `crates/indexer/src/indexer.rs` | Changed-file code chunks are replaced before upsert, avoiding stale duplicate chunks when doc comments shift line numbers; changed doc-only chunks are persisted even if no LLM call is needed |

Compile/build validation:

```bash
cargo fmt --all
cargo check --workspace
cargo build -p matryoshka-cli
```

All passed.

#### Live oMLX fix matrix

Temporary root/report:

```text
/tmp/matryoshka_m3_fix_omlx_matrix
/tmp/matryoshka_m3_fix_omlx_matrix/report.json
```

Model config:

```text
base_url: http://127.0.0.1:44449
chat_model: MercuriusDream--Qwen3.5-4B-MLX-mxfp8
chunk_summary_model: srswti--bodega-raptor-90m
embedding_model: mlx-community--embeddinggemma-300m-bf16
```

Result:

```text
total: 7
pass:  7
fail:  0
```

| ID | Flow tested live against oMLX | Result | Observed output / notes |
|---|---|---|---|
| P0 | `/v1/models` probe | ✅ Pass | All three models present |
| V1 | `--retrieval-primary dense --disable-dense` | ✅ Pass | CLI now errors: `--retrieval-primary dense requires dense embeddings; remove --disable-dense or choose another primary` |
| V2 | `--disable-dense --dense-fallback` | ✅ Pass | CLI now errors: `--dense-fallback requires dense embeddings; remove --disable-dense or use --no-dense-fallback` |
| V3 | Online `index --disable-dense --no-chunk-summaries` | ✅ Pass | `undocumented_chunk` stayed `summary_source=empty`; no `enriching_chunks`; `embedded_records=0`, `late_vectors=0` |
| V4 | Live index, modify only `src/util.rs`, then live update | ✅ Pass | Unchanged `api_entry` kept its `llm` summary exactly; new `live_update_marker` got `llm`; only `src/util.rs` enriched; dense stayed zero |
| V5 | Add/remove Rust doc comment | ✅ Pass | `docflow_target`: `llm -> doc_comment -> llm`; no duplicate stale chunks remained |
| V6 | Dense-enabled live sanity index | ✅ Pass | Dense path still works: `embedded_records=7`, `late_vectors=239`, `embedding_batch` emitted |

Key fixed observations:

```text
V4 api_entry before update:
  source=llm
  summary=A function in the 'api_entry' symbol that returns a string from the 'util::helper' function.

V4 api_entry after unrelated file update:
  source=llm
  summary=A function in the 'api_entry' symbol that returns a string from the 'util::helper' function.

V5 docflow_target:
  initial:      source=llm
  after doc:    source=doc_comment, summary=Explains the documented flow target.
  after remove: source=llm
```

#### Live oMLX validation on `cradle-embed`

The current repo was indexed using live oMLX with dense disabled and a temp DB:

```text
repo: /Users/rohit/cradle-embed
db:   /tmp/cradle_embed_m3_fix_omlx.db
```

Command shape:

```bash
./target/debug/matryoshka-rs index /Users/rohit/cradle-embed \
  --db /tmp/cradle_embed_m3_fix_omlx.db \
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

Index result:

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

No-change live update against the same DB:

```text
semantic_records: 1956
embedded_records: 0
fts_records: 1956
late_vector_rows: 0
records_with_late_vectors: 0
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

Dense-disabled search:

```bash
./target/debug/matryoshka-rs search "where is resolve_retrieval_config defined" \
  --db /tmp/cradle_embed_m3_fix_omlx.db \
  --base-url http://127.0.0.1:44449 \
  --api-key 2508 \
  --embedding-model mlx-community--embeddinggemma-300m-bf16 \
  --disable-dense \
  --no-dense-fallback
```

Top hit:

```text
crates/cli/src/main.rs
record_id: semantic:symbol:crates/cli/src/main.rs::resolve_retrieval_config:498
why_matched: exact token/symbol/path + SQLite FTS + Symbol query plan
Late-interaction MaxSim: absent
```

### M3.1 — Search Result Granularity + Compact Output ✅

After M3, code chunks were indexed and searched, but the default search response
still collapsed matching symbol/snippet/chunk records back into file-level results.
That meant chunk records participated in ranking, but the visible JSON often showed
file-card summaries instead of function/class/method summaries.

Implemented:

- `crates/search/src/semantic_search.rs`
  - Added `SearchResultGranularity`:
    - `file` — current/default behavior; collapse file/symbol/snippet/chunk hits into one file result.
    - `record` — no collapse; return raw matching records.
    - `symbol` — return only symbol records.
    - `chunk` — return only `code_chunk` records, i.e. function/class/method chunks.
  - Added `SearchEngine::with_result_granularity(...)`.
  - Changed non-file result hydration so `code_chunk` output now shows the chunk summary, symbol, kind, signature, line range, and summary source instead of replacing it with the file-card summary.

- `crates/api/src/lib.rs`
  - Added `SearchOptions.result_granularity` with default `file`.
  - Added `SearchOptions::with_result_granularity(...)`.
  - Wired the option through the Rust API search path.

- `crates/cli/src/main.rs`
  - Added CLI search/op flags:

    ```text
    --result-granularity <file|record|symbol|chunk>
    --no-collapse       # shortcut for --result-granularity record
    --compact           # aliases: --hide-match-details, --no-match-details
    ```

  - `--compact` removes these noisy JSON fields from search output:
    - `matched_terms`
    - `total_matched_symbols`
    - `why_matched`
  - `matched_symbols` is intentionally preserved.

Validation:

```bash
cargo fmt --all
cargo check --workspace
cargo test -p matryoshka-search
cargo test --workspace
cargo build -p matryoshka-cli
```

All passed.

Live oMLX search validation against the dense-enabled `cradle-embed` DB:

```bash
./target/debug/matryoshka-rs search \
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

Observed top chunk result:

```text
entity_type: code_chunk
path: crates/indexer/src/indexer.rs
symbol: FullIndexer::refresh_chunk_summaries
summary_source: doccomment
summary: Summarize code chunks that have no useful docstring/doc comment, persist the updated chunks to the store, and build `code_chunk` semantic records in the target template for retrieval...
```

Compact output confirmed absent:

```text
matched_terms: absent
total_matched_symbols: absent
why_matched: absent
matched_symbols: present
```

Additional live checks:

```bash
./target/debug/matryoshka-rs search \
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

Observed raw record-level output with top result:

```text
entity_type: symbol
path: crates/cli/src/main.rs
symbol: resolve_retrieval_config
```

---

## Milestone 4 — SPLADE Primary Retrieval ⬜

### Goal

Add SPLADE as the primary sparse semantic retriever, with dense as an optional
support leg.

### Planned changes

- New crate `crates/splade` (or `crates/sparse-search`) — `SparseEmbedder` trait, `SparseVector`, `SparseTerm`, HTTP client for the omlx SPLADE endpoint (`/Users/rohit/.omlx/models/naver--splade-code-06B`).
- `crates/store-sqlite/src/sqlite_store.rs` — `splade_postings` table (`term`, `record_id`, `weight`) + indexes; upsert/load/delete helpers.
- `crates/indexer/src/indexer.rs` — SPLADE-encode chunk/card records and store postings when `splade_enabled`.
- `crates/search/src/semantic_search.rs` — `collect_splade_candidates`; `CandidateEvidence.splade_score`; SPLADE-primary scoring.
- `crates/api/src/lib.rs` / `crates/cli/src/main.rs` — `--enable-splade`, `--splade-model`, `--splade-top-k`, `--retrieval-primary splade`.

### Deliverable

```bash
matryoshka-rs prepare /repo \
  --chunk-summary-model raptor-90m \
  --retrieval-primary splade \
  --disable-dense

matryoshka-rs search "resume attack mode after handoff" \
  --retrieval-primary splade \
  --disable-dense \
  --explain-retrieval
```

---

## Milestone 5 — Measurement & Recall Comparison ⬜

### Goal

Quantitatively compare retrieval modes so we can decide whether to keep dense at all.

### Planned changes

- `crates/search/src/semantic_search.rs` — `--explain-retrieval` diagnostics: per-hit score breakdown by stage (`exact`, `fts`, `splade`, `dense`, `graph`).
- `crates/search/tests/retrieval_modes.rs` (or CLI `eval-retrieval` command) — load a `queries.jsonl` with expected paths, compute top-1/top-5/top-10 recall + latency + index size for each mode.

### Comparison matrix

| Mode | Config |
|---|---|
| A | exact + FTS only (`--disable-dense`, no SPLADE yet) |
| B | exact + SPLADE (`--retrieval-primary splade --disable-dense`) |
| C | exact + SPLADE + dense fallback (`--dense-fallback`) |
| D | exact + SPLADE + current dense embeddings (`mlx-community--embeddinggemma-300m-bf16`) |

### Decision gate

If `exact + SPLADE` matches `exact + SPLADE + dense` on top-5/top-10 recall,
remove dense from the default path and keep it only as an optional feature flag.
