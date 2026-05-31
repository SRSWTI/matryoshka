# Cradle

Cradle is a local code-intelligence package for:

- repository analysis into SQLite
- exact graph-backed lookup
- semantic indexing and semantic search
- hierarchy-guided repository navigation
- focused graph visualization

## Install

Global CLI install with `uv`:

```bash
cd /Users/rohit/cradle-embed
uv tool install .
```

Editable local install:

```bash
cd /Users/rohit/cradle-embed
uv venv
uv pip install -e .
```

## Default Analyze Output

When you run:

```bash
cradle analyze /path/to/repo --model YOUR_MODEL --api-key YOUR_API_KEY
```

Cradle is already a working local code-intelligence stack built around a SQLite graph as the source of truth, with semantic vectors as a sidecar and several retrieval lanes layered on top.

**What The System Actually Is**
The core split is:

1. Ingest and structure the repo.
2. Persist that structure in SQLite.
3. Build optional semantic sidecars from the stored graph.
4. Query the graph through exact, deterministic, semantic, hierarchical, and QA entrypoints.

The important architectural point is that the LLM is only in the labeling phase during analyze. Query-time behavior is still deterministic and heuristic-driven, not LLM-routed. That matches the current roadmap in todo.md: Phase 6 is still future work.

**End-To-End Flow**
The real control flow starts in src/cradle/cli.py. The CLI fans out into these main paths:

1. `cradle analyze`
   Flow:
   src/cradle/pipeline.py collects source files, extracts AST facts per file, builds compact file packets, optionally sends those packets to the labeling engine, rolls file evidence upward into folder packets and a repo packet, then hands everything to src/cradle/graph_builder.py. That builder produces nodes, symbols, imports, call records, symbol references, and inherited node context. Then src/cradle/community_detection.py adds Louvain communities and theme/domain nodes. Finally src/cradle/storage.py wipes and rewrites the SQLite graph and rebuilds FTS indexes.

2. `cradle semantic-index`
   Flow:
   src/cradle/semantic_index.py loads nodes and symbols back out of SQLite, converts them into semantic records, embeds them through src/cradle/embeddings.py, builds centroid rollups for structural parents and virtual parents like communities/themes, and writes sidecar artifacts such as `manifest.json`, `nodes.records.json`, `nodes.vectors.npy`, `symbols.*`, and `node_centroids.*`. SQLite stays canonical; the vectors are derived artifacts.

3. `cradle retrieve`
   Flow:
   src/cradle/retrieval.py builds a `QueryPlan` from the raw query, infers whether the user wants callers, callees, implementations, or files, and scores symbols and nodes through exact matches, fuzzy path/name matches, FTS, and call/reference expansion. It then hydrates full hits through src/cradle/result_loader.py.

4. `cradle semantic-search`
   Flow:
   src/cradle/semantic_search.py embeds the query, does vector search over the sidecar, then reranks results with lexical and intent-aware bonuses. It also applies caller/callee expansion logic so semantic search does not stay purely vector-only.

5. `cradle hierarchy-search`
   Flow:
   src/cradle/hierarchical_search.py is the top-down router. It starts at repo/root nodes, walks structural children plus virtual children from communities and themes, uses centroid narrowing when a parent has many children, keeps a branch frontier, and only later lands on final file candidates and symbol candidates.

6. `cradle question`
   Flow:
   src/cradle/question_answering.py is not an LLM answer synthesizer. It runs hierarchy search first, then exact symbol/import/call/reference search, merges evidence, picks a preferred symbol when possible, reads code excerpts from disk, and emits a grounded text answer from retrieved evidence.

7. Visualization and dashboard
   Flow:
   src/cradle/db_visualization.py builds a static Markdown report from the DB.
   src/cradle/focus_visualization.py builds focused Mermaid neighborhoods around a resolved file or symbol.
   src/cradle/dashboard.py serves a local SPA over the SQLite DB and optional semantic sidecar. It exposes graph, themes, communities, embeddings, and per-node detail endpoints.

**How The Data Is Shaped**
The base types live in src/cradle/graph_models.py. The important objects are:

- `CodeNode`: repo, folder, file, community, or theme.
- `CodeSymbol`: stored symbol identity and signature.
- `ImportRecord`: internal/external import edges, including out-of-scope internal imports.
- `CallRecord`: caller to callee link, sometimes resolved to a stored symbol.
- `SymbolReferenceRecord`: imports and calls normalized into a reference surface.
- `NodeContextRecord`: inherited summaries/categories/tags from imported internal files.
- `RepositoryGraph`: the full in-memory graph before persistence.

The packet types in src/cradle/models.py are the pre-storage representations used for prompts and aggregation:

- `FilePacket` is a compressed summary of one file.
- `NodePacket` is an aggregated folder or repo packet.
- `LabelResult` is the LLM output for a file/folder/repo.

**What Each Major Module Does**
Here’s the cleanest mental map of the codebase.

- src/cradle/ast_extractor.py
  Tree-sitter extraction for Python and TypeScript. Produces symbols, imports, and call sites. It uses byte offsets correctly, which matters for non-ASCII source.

- src/cradle/pipeline.py
  Repository walker and packet builder. This is the main analyze orchestrator.

- src/cradle/labeling.py, src/cradle/prompts.py, src/cradle/llm_client.py, src/cradle/cache.py
  The only model-assisted part of the core pipeline. Structured prompt building, OpenAI-compatible calling, and SQLite-backed cache.

- src/cradle/graph_builder.py
  Converts analyzed files plus labels into the actual graph. This is where internal import resolution, call target resolution, references, and inherited node context are created.

- src/cradle/community_detection.py
  Builds community nodes from import/call coupling and theme nodes from dominant file categories.

- src/cradle/storage.py
  Owns the schema. It stores `repos`, `nodes`, `symbols`, `imports`, `call_sites`, `symbol_references`, `references`, `node_context`, `community_members`, `theme_members`, and `edges`, plus FTS surfaces.

- src/cradle/result_loader.py
  Hydrates DB rows back into typed hits with attached contexts/imports/calls/references.

- src/cradle/retrieval.py
  Exact/deterministic ranking lane.

- src/cradle/exact_search.py
  Specialized exact lookup surfaces for files, symbols, imports, modules, calls, and references.

- src/cradle/embeddings.py
  Embedding backend abstraction. MLX is preferred on macOS; sentence-transformers is fallback.

- src/cradle/semantic_index.py
  Builds and loads the sidecar semantic index, including centroid rollups.

- src/cradle/semantic_search.py
  Vector retrieval with lexical/intention-aware reranking.

- src/cradle/hierarchical_search.py
  Top-down semantic branch router across repo/folder/community/theme/file.

- src/cradle/question_answering.py
  Grounded answer composer over hierarchy plus exact evidence plus code excerpts.

- src/cradle/db_visualization.py, src/cradle/focus_visualization.py, src/cradle/dashboard.py
  Reporting and inspection surfaces.

- src/cradle/__init__.py
  Package export surface.

**What’s Important To Understand About Current Behavior**
A few implementation truths are worth calling out:

- SQLite is the primary truth. The semantic index is rebuilt from SQLite, not the other way around.
- Query-time reasoning is heuristic and grounded, not LLM-planned yet.
- Communities are structural, based on import and call coupling.
- Themes are semantic, based on labeled file categories.
- Node context is a real stored feature: files inherit summaries/tags/categories from internal imports and retrieval uses that.
- `axe_question` is intentionally conservative. It does not invent an answer; it assembles one from retrieved evidence.
- The semantic sidecar already includes centroid rollups for structural and virtual parents, so hierarchy search can narrow large child sets without brute-forcing every descendant.

**What Is Not There Yet**
From the code, the main planned-but-not-implemented items are still the ones you called out in the roadmap:

- incremental reindexing
- watchdog / auto-refresh
- better branch summary quality
- LLM-guided branch routing
- LLM-controlled multi-branch deepening in question answering
- budgeted routing policies
- optional FAISS is partially supported if installed, but NumPy is still the default path

So the current system is already solidly usable, but it is still a heuristic hierarchy router, not yet the adaptive planner you sketched for Phase 6.

If you want, the next useful step is one of these:

1. I can walk you through one concrete command end-to-end, like `cradle analyze` or `cradle question`, with the exact file-to-file call chain.
2. I can give you a dependency map of the whole package in diagram form.
3. I can review one subsystem deeply, for example hierarchical_search.py or semantic_index.py, and explain every major function.