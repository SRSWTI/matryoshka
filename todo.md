# Cradle Todo

## Phase 1 — Done

- [x] AST extraction for Python and TypeScript
- [x] Symbols, imports, call sites, and references
- [x] SQLite graph storage
- [x] LLM summaries, tags, and categories in DB
- [x] `cradle analyze`
- [x] `cradle retrieve`
- [x] `cradle visualize-db`
- [x] Real validation on `/Users/rohit/pi/packages/ai`

## Phase 2 — Done

- [x] Add embedding sidecar storage files
- [x] Use MLX EmbeddingGemma for semantic indexing
- [x] Embed graph-backed file and symbol content
- [x] Add `cradle semantic-index`
- [x] Add `axe_semantic_search`
- [x] Add `cradle semantic-search`
- [x] Run real semantic evaluation on `/Users/rohit/pi/packages/ai`

## Phase 3 — Done

- [x] Add exact DB lookup tools
- [x] Add `axe_file_search`
- [x] Add `axe_symbol_search`
- [x] Add `axe_import_search`
- [x] Add `axe_module_search`
- [x] Add `axe_reference_search`
- [x] Add `axe_call_search`

## Phase 4 — In Progress

- [x] Broader hybrid semantic + exact tool routing
- [x] Better graph neighborhood views
- [x] Focused file/symbol visualization
- [x] Cleaner stored signatures
- [ ] Better summary/category quality
- [ ] Optional FAISS acceleration

## Phase 5 — In Progress

- [x] Louvain community storage
- [x] K-centroid rollups
- [x] Repo/folder/module semantic routing over the existing repo/folder/file tree
- [ ] Incremental re-indexing
- [ ] Watchdog / auto-refresh flow

## Phase 6 — Retrieval Optimizations

- [ ] Add LLM-guided branch routing for broad or ambiguous queries

	Current hierarchy traversal is much better than flat search, but it still uses heuristic narrowing when choosing which communities, folders, or centroid branches to explore first. That is cheap and stable, but broad questions like `explain the architecture`, `how does request handling flow through the system`, or `where is inference orchestration implemented` can still route through slightly noisy branch labels or stop too early on one strong-looking branch.

	This should be treated as an optimization layer on top of the existing graph and centroid pipeline, not as a replacement for it. The graph, exact tools, community nodes, and centroids should still generate a small candidate frontier first. Then an LLM router can look at the query plus short summaries for the top candidate branches and decide which 1 to 3 branches are worth deeper expansion.

	What needs to be done:
	- Add a branch-routing stage after the current candidate communities/centroids are retrieved.
	- Feed the LLM a compact view of the top branch candidates: label, summary, path, representative files/symbols, and retrieval scores.
	- Have it choose a small number of branches plus an explicit depth or budget per branch.
	- Fall back to current heuristic routing for local or exact queries where an LLM hop is unnecessary.
	- Store routing decisions in debug output so branch selection stays inspectable.

	Example:
	- Query: `explain how model serving works end to end`
	- Current heuristic flow may over-commit to one server branch early.
	- LLM-guided routing should be able to keep both `server/api` and `engine/model-management` branches alive long enough to synthesize a more complete answer.

- [ ] Add LLM-controlled multi-branch deepening inside `axe_question`

	Right now answer synthesis is only as good as the evidence frontier that retrieval surfaces. Better indexing helps a lot, but it does not make missed branches impossible, because misses can still happen from summary quality, vector phrasing mismatch, or early pruning. For broad questions, `axe_question` should evolve from a single retrieve-then-answer pass into a small planner that decides whether the question needs one branch, several branches, or a deeper second pass.

	This is an optimization because it improves recall and answer completeness without changing SQLite or sidecar storage as the source of truth. The LLM should not hallucinate the search space; it should only decide how aggressively to deepen the already-retrieved branch candidates.

	What needs to be done:
	- Add a query-classification step inside `axe_question` to distinguish local, comparative, architectural, and causal questions.
	- Let the question layer request additional expansion into 2 to 3 retrieved communities/centroids when the query is broad.
	- Add stopping rules so the planner can stop when evidence converges instead of always exhausting the full search budget.
	- Merge evidence across selected branches before final answer synthesis.
	- Keep exact symbol/import/call/reference evidence in the final support set so the answer stays grounded.

	Example:
	- Query: `how does the system handle model downloads and serving configuration`
	- A single-branch answer may only describe downloader code or only describe settings code.
	- Multi-branch deepening should pull evidence from both `admin/download` and `model-settings/engine` related branches before synthesizing the answer.

- [ ] Add hierarchical summaries for communities and centroids

	If we want an LLM router or planner to make better decisions, it needs strong compressed descriptions of each branch. Today the summaries and categories are useful, but they are still not rich enough to reliably drive top-down planning on broad questions. Community nodes and centroid rollups need better surface-level summaries so the system can reason over branch meaning before opening leaf files and symbols.

	This is an optimization because it improves both routing quality and answer quality without changing the underlying graph facts. Better summaries make the existing communities and centroids more useful; they do not replace them.

	What needs to be done:
	- Generate or refresh short summaries for community nodes and centroid groups from their representative members.
	- Store branch summaries that explain responsibility, not just keywords or filenames.
	- Include representative symbols/files and distinguishing concepts in each summary payload.
	- Use these summaries in hierarchy search debug views, `axe_question`, and any future LLM router prompt.
	- Revisit current category/tag generation so it supports branch-level summaries instead of only node-level metadata.

	Example:
	- A centroid should summarize something like `request parsing and OpenAI-compatible API handlers` rather than just listing top files from `server.py`, `api/models.py`, and related modules.

- [ ] Add budgeted LLM routing policies instead of using LLMs on every query

	If an LLM is added to routing, it needs explicit cost and latency controls. Using it on every exact symbol lookup or narrow implementation query would add expense and unpredictability without improving outcomes. The system should decide when the extra planning step is worth paying for.

	This is an optimization because it preserves the speed and determinism of the current stack while selectively improving the hard queries that actually need adaptive planning.

	What needs to be done:
	- Add query-shape heuristics to decide when to skip LLM routing entirely.
	- Define separate budgets for local queries, medium ambiguous queries, and broad architecture questions.
	- Limit the number of branch candidates shown to the LLM.
	- Limit the number of second-pass expansions per query.
	- Record cost/latency telemetry alongside answer quality metrics.

	Example:
	- `who calls getEnvApiKey` should stay deterministic and exact-first.
	- `explain the architecture of request handling and model execution` should be allowed one top-level LLM routing pass and one selective deepening pass.

## Notes

- [x] Semantic search should use embeddings
- [x] `axe_*` DB tools should use exact or structured lookup
- [x] Keep SQLite as the main graph/source-of-truth store
- [x] Semantic vectors should live beside the SQLite graph, not replace it
- [x] Default semantic backend should use MLX on macOS
- [x] Keep `axe_semantic_search` retrieval-only and put grounded answer synthesis in `axe_question`
- [x] `cradle question`, `cradle hierarchy-search`, and `cradle visualize-focus` are implemented and tested
- [x] Louvain communities are stored in SQLite and consumed by hierarchy traversal
- [x] Semantic sidecars now include K-centroid rollups for structural parents
- [x] Current full test suite: `34 passed`
- [x] Real Pi AI artifacts refreshed: `/tmp/pi-ai-cradle-index.db`, `.cradle/pi-ai-semantic`, `.cradle/pi-ai-semantic-report.md`, `.cradle/pi-ai-axe-report.md`
- [ ] Hierarchy branch labels still need cleanup on some broad queries even when the final answer node is correct
- [ ] Treat LLM-guided routing as a later optimization layer over graph-backed retrieval, not as a replacement for exact lookup, communities, or centroids