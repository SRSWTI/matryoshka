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
| M3 — Retrieval config, dense optional | ⬜ Next | Make dense embeddings optional at index/search time via CLI/API flags; stage-based candidate collection |
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

## Milestone 3 — Retrieval Config, Dense Optional ⬜

### Goal

Make dense embeddings genuinely optional so we can run `exact + FTS` (and later
`exact + SPLADE`) without embedding the query or the records.

### Planned changes

- `crates/core-ir/src/models.rs` (or new config module) — `RetrievalConfig`, `RetrievalPrimary` enum (`Fts`, `Splade`, `Dense`, `Hybrid`).
- `crates/indexer/src/indexer.rs` — make dense record embedding optional during `index`, `prepare`, `update`, and `rebuild-semantic`; skip dense vector writes and late-interaction vector generation when dense is disabled.
- `crates/search/src/semantic_search.rs` — `SearchEngine` takes `Option<M>` embedder; stage-based candidate collection (`collect_exact_candidates`, `collect_fts_candidates`, `collect_dense_candidates`); skip query embedding when dense disabled; disable late-interaction when dense disabled.
- `crates/store-sqlite/src/sqlite_store.rs` — make retrieval health reports explicit about dense-disabled indexes (`embedded_records=0` can be healthy when configured that way).
- `crates/api/src/lib.rs` — `MatryoshkaConfig` gains `retrieval_primary`, `dense_enabled`, `dense_fallback_enabled`.
- `crates/cli/src/main.rs` — `--retrieval-primary`, `--enable-dense`, `--disable-dense`, `--dense-fallback`, `--no-dense-fallback` flags on `prepare`/`index`/`update`/`search`/`op`/`prewarm`/`rebuild-semantic`.
- Progress events — emit a clear dense-skipped/embedding-skipped state so a no-dense index run does not look stalled or broken.

### Deliverable

```bash
matryoshka-rs index /repo --disable-dense --progress-jsonl
matryoshka-rs search "..." --disable-dense --no-dense-fallback
```
works without embedding records during indexing and without embedding the query during search.

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
