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

- [ ] Louvain community storage
- [ ] K-centroid rollups
- [x] Repo/folder/module semantic routing over the existing repo/folder/file tree
- [ ] Incremental re-indexing
- [ ] Watchdog / auto-refresh flow

## Notes

- [x] Semantic search should use embeddings
- [x] `axe_*` DB tools should use exact or structured lookup
- [x] Keep SQLite as the main graph/source-of-truth store
- [x] Semantic vectors should live beside the SQLite graph, not replace it
- [x] Default semantic backend should use MLX on macOS
- [x] Keep `axe_semantic_search` retrieval-only and put grounded answer synthesis in `axe_question`
- [x] `cradle question`, `cradle hierarchy-search`, and `cradle visualize-focus` are implemented and tested
- [x] Current full test suite: `32 passed`
- [x] Real Pi AI artifacts refreshed: `/tmp/pi-ai-cradle-index.db`, `.cradle/pi-ai-semantic`, `.cradle/pi-ai-semantic-report.md`, `.cradle/pi-ai-axe-report.md`
- [ ] Hierarchy branch labels still need cleanup on some broad queries even when the final answer node is correct