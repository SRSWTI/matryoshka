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
| M1 — Code chunk extraction + doc extraction | ✅ Complete | Parser emits `CodeChunkFact` per symbol with full body + docstring/doc-comment summaries |
| M2 — Concurrent LLM chunk summaries + chunk semantic records | ⬜ Not started | MLX/Raptor fills `generated_summary` for empty chunks; `code_chunk` semantic records created |
| M3 — Retrieval config, dense optional | ⬜ Not started | Make dense embeddings optional via CLI/API flags; stage-based candidate collection |
| M4 — SPLADE primary retrieval | ⬜ Not started | SPLADE sparse index + postings storage; SPLADE-first scoring |
| M5 — Measurement & recall comparison | ⬜ Not started | Retrieval diagnostics + eval harness for SPLADE-only vs hybrid |

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
6. **Backward compatible.** `RepositorySnapshot.code_chunks` is `#[serde(default)]`,
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
| `crates/parser/src/source_parser.rs` | Extended `ParsedRepository` with `code_chunks`; `parse_file` builds chunks; `build_code_chunks`, `chunk_kind_for_symbol`, `doc_is_useful`, `doc_summary_source`, `extract_symbol_doc`, `extract_python_docstring`, `extract_leading_doc_comment`, `extract_typescript_doc`, `extract_jsdoc_block`, `extract_line_comments`; 9 new tests |
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

- `cargo test --workspace` — all pass (15 parser tests incl. 9 new, 11 indexer integration tests, plus all search/store/api/watcher tests).
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

### Test script

`scripts/test_milestone1_chunks.sh [REPO_PATH]` — builds the CLI, indexes a repo
(or a generated sample) offline, and dumps the `code_chunks` table for inspection.

---

## Milestone 2 — Concurrent LLM Chunk Summaries + Chunk Semantic Records ⬜

### Goal

For every chunk with `summary_source == Empty`, call the local MLX/Raptor model
to generate a 2-3 line summary. Then build `code_chunk` semantic records in the
target template and persist them so search can use them.

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
{ "summary": "Cancels the active countdown and switches the object into attack mode." }
```

The caller already knows the `chunk_id` it sent, so the returned summary is
mapped back onto that chunk. Concurrency is handled by firing many requests in
parallel (the omlx server already does continuous batching internally).

### Planned changes

- `crates/enricher/src/lib.rs` — add `ChunkSummarizer` trait + `ChunkSummaryDraft`.
- `crates/enricher/src/heuristic.rs` — `HeuristicChunkSummarizer` fallback.
- `crates/enricher/src/mlx_chat.rs` — `MlxChunkSummarizer` with concurrent requests (rayon thread pool).
- `crates/enricher/src/prompts.rs` — `chunk_summary_prompt`.
- `crates/indexer/src/indexer.rs` — summarize chunks during refresh; only call LLM for `Empty` chunks; skip unchanged chunks by `source_hash`; build `code_chunk` semantic records.
- `crates/api/src/lib.rs` — `MatryoshkaConfig` gains `chunk_summary_enabled`, `chunk_summary_model`, `chunk_summary_concurrency`.
- `crates/cli/src/main.rs` — `--chunk-summary-model`, `--chunk-summary-concurrency`, `--no-chunk-summaries` flags on `prepare`/`index`/`update`/`rebuild-semantic`.

### Chunk semantic record template

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

- If file `source_hash` unchanged → skip chunk re-summary.
- If changed file → regenerate only chunks in changed files.
- Only chunks with `summary_source == Empty` (or generic/short docs) are sent to the LLM.

---

## Milestone 3 — Retrieval Config, Dense Optional ⬜

### Goal

Make dense embeddings genuinely optional so we can run `exact + FTS` (and later
`exact + SPLADE`) without embedding the query or the records.

### Planned changes

- `crates/core-ir/src/models.rs` (or new config module) — `RetrievalConfig`, `RetrievalPrimary` enum (`Fts`, `Splade`, `Dense`, `Hybrid`).
- `crates/search/src/semantic_search.rs` — `SearchEngine` takes `Option<M>` embedder; stage-based candidate collection (`collect_exact_candidates`, `collect_fts_candidates`, `collect_dense_candidates`); skip query embedding when dense disabled; disable late-interaction when dense disabled.
- `crates/api/src/lib.rs` — `MatryoshkaConfig` gains `retrieval_primary`, `dense_enabled`, `dense_fallback_enabled`.
- `crates/cli/src/main.rs` — `--retrieval-primary`, `--enable-dense`, `--disable-dense`, `--dense-fallback`, `--no-dense-fallback` flags on `prepare`/`index`/`update`/`search`/`op`/`prewarm`/`rebuild-semantic`.

### Deliverable

```bash
matryoshka-rs search "..." --disable-dense --no-late-interaction
```
works without embedding the query.

---

## Milestone 4 — SPLADE Primary Retrieval ⬜

### Goal

Add SPLADE as the primary sparse semantic retriever, with dense as an optional
support leg.

### Planned changes

- New crate `crates/splade` (or `crates/sparse-search`) — `SparseEmbedder` trait, `SparseVector`, `SparseTerm`, HTTP client for the omlx SPLADE endpoint (`naver--splade-code-06B`).
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
| A | exact + FTS (current-ish, no dense) |
| B | exact + SPLADE |
| C | exact + SPLADE + dense-256 fallback |
| D | exact + SPLADE + dense-1024 (current dense) |

### Decision gate

If `exact + SPLADE` matches `exact + SPLADE + dense` on top-5/top-10 recall,
remove dense from the default path and keep it only as an optional feature flag.
