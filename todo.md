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

## Phase 2 — Next

- [ ] Add embedding storage tables/files
- [ ] Pick Matryoshka embedding model
- [ ] Embed repo, folder, file, and symbol content
- [ ] Add `axe_semantic_search` for embedding-based retrieval

## Phase 3 — Next

- [ ] Add exact DB lookup tools
- [ ] Add `axe_file_search`
- [ ] Add `axe_symbol_search`
- [ ] Add `axe_import_search`
- [ ] Add `axe_module_search`
- [ ] Add `axe_reference_search`
- [ ] Add `axe_call_search`

## Phase 4 — Later

- [ ] Hybrid semantic + exact reranking
- [ ] Better graph neighborhood views
- [ ] Focused file/symbol visualization
- [ ] Cleaner stored signatures
- [ ] Better summary/category quality

## Phase 5 — Later

- [ ] Louvain community storage
- [ ] K-centroid rollups
- [ ] Repo/folder/module semantic routing
- [ ] Incremental re-indexing
- [ ] Watchdog / auto-refresh flow

## Notes

- [x] Semantic search should use embeddings
- [x] `axe_*` DB tools should use exact or structured lookup
- [x] Keep SQLite as the main graph/source-of-truth store