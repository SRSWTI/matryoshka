# Cradle
### Hierarchical Semantic Code Intelligence — Project Specification

---

## Current Status

This document started as the target architecture. The project now has a real working foundation for code extraction, graph storage, deterministic retrieval, and SQLite-backed inspection. The embedding layer and Matryoshka routing are still next.

### What Is Implemented Now

- Tree-sitter extraction for Python and TypeScript
- Symbol extraction with line/column ranges
- Import extraction with internal vs external classification
- Call-site extraction and symbol reference extraction
- SQLite graph storage instead of JSON output
- Repo, folder, file, symbol, import, call, reference, context, and edge tables
- LLM-generated summaries, descriptions, tags, and categories persisted into the graph
- Deterministic retrieval over the SQLite DB
- DB visualization report with schema and sample stored rows

### Why We Implemented This First

We implemented the graph and SQLite layer before embeddings because this gives us a stable source of truth for the codebase.

- AST extraction gives grounded structural facts instead of guessing from raw text
- SQLite makes the index inspectable, queryable, and easy to debug
- calls, imports, references, and node context make retrieval useful for real coding questions
- summaries/tags/categories make the graph understandable to humans and usable for later ranking
- deterministic retrieval gives immediate value before the embedding layer is ready

This means the next embedding system will not be built on vague chunks. It will be built on real files, symbols, edges, and metadata that already exist.

### Current Storage Shape

Current analysis output is a SQLite graph with these main tables:

- `repos`
- `nodes`
- `node_categories`
- `node_tags`
- `symbols`
- `imports`
- `call_sites`
- `symbol_references`
- `references`
- `node_context`
- `edges`

These tables were chosen so we can support both kinds of future access:

- semantic lookup later through embeddings
- exact lookup now through graph and metadata queries

Examples:

- `nodes` stores file/folder/repo level metadata
- `symbols` stores exact symbol identities and positions
- `imports`, `call_sites`, and `symbol_references` store navigational relationships
- `node_context` stores inherited internal context from neighboring files
- `edges` gives a generic graph surface for visualization and future traversal

### Current Retrieval Shape

Current retrieval is not embedding-based yet. It is a practical hybrid of:

- exact and fuzzy symbol/name/path matching
- FTS-backed node and symbol lookup
- tag/category matches
- call/reference expansion for intent like `who calls X`
- node context inheritance from internal imports

This was implemented first because it solves real coding-session queries immediately:

- exact symbol lookup
- implementation lookup
- caller/callee lookup
- import/module lookup
- file/folder metadata lookup

This is useful now, and it also defines the exact entities that the embedding layer should later rank.

### What This Gives Us Right Now

The current system is already useful for:

- finding the file that implements a concept
- finding what calls a symbol
- finding what a symbol depends on
- inspecting stored metadata directly in SQLite
- building exact `axe_*` lookup tools on top of the DB

It is not yet the final semantic search system, but it is already the correct base layer for it.

### Real Example Shape

The current implementation has already been exercised on `/Users/rohit/pi/packages/ai` and stored in `/tmp/pi-ai-cradle-index.db`.

That run shows the storage model is already large enough and structured enough to support the next phase:

- 142 nodes
- 1065 symbols
- 615 imports
- 2767 call sites
- 3657 references
- 430 node-context rows
- 3523 edges

The current implementation can already answer queries like:

- `getEnvApiKey` resolves to `src/env-api-keys.ts`
- `streamBedrock` resolves to `src/providers/amazon-bedrock.ts`
- `who calls getEnvApiKey` surfaces real caller files like `test/cross-provider-handoff.test.ts`
- `where is oauth authentication handled` surfaces `src/utils/oauth` and related files

The current visualization layer can already show:

- table counts
- top files/folders/symbols
- actual SQL schema per table
- sample stored rows from the live DB

### What Is Not Implemented Yet

- embedding generation and storage
- Matryoshka multi-resolution retrieval
- Louvain/K-means based retrieval routing
- hybrid embedding + graph reranking
- specialized `axe_*` query tools over the DB
- incremental re-indexing and watchdog flow

---

## What Is Cradle?

Cradle is a local, in-process code intelligence system that lets a developer ask natural language questions about a codebase and get deterministic, semantically precise answers — without an LLM agent grepping files, without token-bloated context windows, and without a remote server.

The core idea: **index bottom-up, retrieve top-down.**

Today, the bottom-up indexing and graph persistence pieces are real. The top-down semantic embedding path is the next layer to build.

During pre-warming, Cradle builds a nested semantic index of the entire codebase — from raw AST symbols at the bottom, up through files, modules, folders, and the full repository — using graph community detection, Matryoshka embeddings, and call graph analysis. The result is a self-describing map of the codebase that the code itself drew.

At query time, Cradle navigates that map top-down: coarse cluster matching first, narrowing at each level, expanding via call graph at the bottom. No full-index search. No LLM at query time. Deterministic routing into a small candidate set, then precise semantic reranking.

The mental model: **a Russian Matryoshka doll.**

The outermost doll is the whole repo. Open it and you find macro clusters. Open those and you find folder-level communities. Keep going — modules, files, functions, symbols. Each doll knows its own shape and the shape of everything inside it. Query navigation is just opening the right dolls in the right order.

---

## The Name

**Cradle** — it holds the codebase from the ground up. Everything is built from the most primitive elements (AST nodes, import edges) and cradled into higher structures. Nothing is discarded. Nothing is averaged away. The structure is preserved at every level.

---

## Core Design Principles

1. **Bottom-up indexing, top-down retrieval.** Build meaning from atoms. Navigate from overview to detail.
2. **Structure emerges from the code, not from definitions.** Louvain community detection on the import graph discovers clusters. No human pre-labeling. No LLM hallucination of structure.
3. **No LLM at query time.** LLM is used once per cluster at pre-warm to assign a human-readable label. Never at retrieval time.
4. **Deterministic first, probabilistic second.** Import signature index and symbol-BM25 fire first (deterministic). Vector search runs only inside an already-narrowed candidate set.
5. **No averaged rollups.** Every folder/module node stores K cluster centroids (K-means over children), not a single averaged embedding. Specificity is preserved all the way up.
6. **Language agnostic core, rich metadata for Python and TypeScript.** AST via tree-sitter for all languages. Extra structural metadata (type hints, decorators, generics, export patterns) for Python and TypeScript.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                      CRADLE                         │
│                                                     │
│  ┌─────────────┐        ┌─────────────────────┐    │
│  │   INDEXER   │        │     RETRIEVER        │    │
│  │ (pre-warm)  │        │    (query time)      │    │
│  │             │        │                      │    │
│  │ bottom-up   │        │    top-down          │    │
│  │             │        │                      │    │
│  │ L0: Symbol  │        │  L4: Repo cluster    │    │
│  │ L1: File    │        │  L3: Folder cluster  │    │
│  │ L2: Module  │        │  L2: Module cluster  │    │
│  │ L3: Folder  │        │  L1: File            │    │
│  │ L4: Repo    │        │  L0: Symbol + graph  │    │
│  └─────────────┘        └─────────────────────┘    │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │               INDEX STORE (local)           │   │
│  │  - Matryoshka embedding vectors             │   │
│  │  - K-centroid maps per node                 │   │
│  │  - Louvain community graph                  │   │
│  │  - Symbol BM25 index                        │   │
│  │  - Import signature index                   │   │
│  │  - Call graph (directed, weighted)          │   │
│  │  - Tag vocabulary per node                  │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## The Indexing Pipeline — Bottom Up

### Level 0 — Symbol (Atomic)

The foundation. Every piece of structural meaning in the codebase lives here.

**What is extracted (via tree-sitter AST):**
- Function names, signatures, return types, parameter types
- Class names, base classes, decorators
- Constants and module-level variables
- Docstrings (if present)
- Line range of every symbol

**Call graph construction:**
- For every function: what functions does it call? (callees)
- For every function: what functions call it? (callers)
- Edges are directed and weighted by call frequency within the file
- Cross-file edges resolved via import resolution

**Import edge construction:**
- For every file: what does it import and from where?
- External packages tracked separately from internal imports
- Internal import graph = the graph Louvain will run on

**Output per file:**
```
symbol_table: {
  name, kind, signature, line_range,
  callers[], callees[], decorators[], docstring
}
import_edges: [ (this_file → imported_module), ... ]
external_packages: [ "jwt", "bcrypt", "sqlalchemy", ... ]
```

**Languages:**
- Python: full AST via tree-sitter-python. Extra: type hints, decorators, dataclass fields, `__init__` signatures
- TypeScript: full AST via tree-sitter-typescript. Extra: interfaces, generics, export types, async/await patterns
- All others: basic symbol extraction (functions, classes, imports)

---

### Level 1 — File

**Embedding:**
- Embed the symbol table (names + signatures as text), NOT the raw source code
- Use fine-tuned Matryoshka code embedding model (see Embedding Model section)
- Store full-dimension vector for fine-grained retrieval
- Also store 64-dim truncated vector for coarse-pass retrieval

**BM25 index:**
- Index only over symbol names and tag vocabulary
- Not over raw code text — symbol names are compressed developer intent
- `verifyJWTToken`, `AuthMiddleware`, `checkPermissions` are better retrieval signal than raw code

**Import signature vector:**
- Binary/weighted vector over external package vocabulary
- Files importing `[jwt, bcrypt, cryptography, oauth2]` cluster near each other naturally
- This is the primary input to Louvain at Level 2

**LLM summary (pre-warm only, once per file):**
- Input: symbol names + signatures + external packages
- Output: one-line purpose summary + 3-5 semantic tags
- Model: local LLM (small, fast — Mistral 7B or Qwen2.5-Coder is sufficient)
- Example output: `summary: "JWT token creation and validation", tags: ["auth", "jwt", "token", "middleware"]`

---

### Level 2 — Module / Subfolder

**Louvain community detection:**
- Input: import graph of all files within this subfolder
- Algorithm: Louvain modularity maximization
- Output: community assignments (which files naturally cluster together)
- This is structural truth — the codebase's own topology, not your opinion

**K-Means on file embeddings:**
- Input: all Level 1 file embedding vectors within this module
- K = min(5, num_files) — adaptive
- Store K centroid vectors, NOT an average
- Each centroid = a dominant concept cluster in this module
- Why: averaging would blur `auth` and `db_connection` into meaningless middle ground. Centroids keep them separate and searchable.

**Tag rollup:**
- Aggregate child file tags, weight by LLM-assigned importance
- Deduplicate and rank by frequency
- Module inherits the strongest tags from its files

**LLM cluster labeling (once per cluster per pre-warm):**
- Input: top external packages in this Louvain community + top tags
- Output: human-readable cluster label
- Example: `[jwt, bcrypt, sessions, oauth2]` → `"authentication and token management"`
- This label is stored but never used for retrieval routing — it is for human inspection only

---

### Level 3 — Folder

Same process as Level 2, one level up.

- Louvain runs on the module-level import graph (modules as nodes, cross-module imports as edges)
- K-Means on module centroid sets → folder-level centroids
- Tag rollup from module tags
- LLM labels folder-level clusters

The galaxy is taking shape.

---

### Level 4 — Repository

Final Louvain pass over the full folder-level graph.

- 4-8 macro clusters emerge: `auth`, `data layer`, `api routing`, `background jobs`, `config/infra`, etc.
- These are your galaxies
- Each has K centroids and a tag vocabulary
- The repo-level index is the entry point for every query

---

## The Embedding Model

### Choice: Fine-tuned Matryoshka Code Embedding

**Base model:** `nomic-embed-code` or `voyage-code-2`
- Pre-trained on code corpora — understands `verifyJWTToken ≈ authenticate_user`
- General language models do not reliably handle code vocabulary

**Training objective:**
```
Total Loss = MatryoshkaLoss(
               MultipleNegativesRankingLoss(model)
             )
```

- `MultipleNegativesRankingLoss`: contrastive — pulls matching query-code pairs together, pushes non-matching apart. In-batch negatives make training efficient.
- `MatryoshkaLoss`: wraps the above, applies the ranking loss at multiple embedding dimensions simultaneously (e.g. 64, 128, 256, 512, 768). Forces the model to encode coarse meaning in outer dimensions and fine-grained meaning in inner dimensions.

**Why Matryoshka specifically for Cradle:**
Cradle's traversal is multi-resolution by design. At repo level you want a fast coarse match. At symbol level you want full precision. Same model, same index — just truncate the vector at query time per traversal depth:

```
Repo/Folder level:   64-dim  → fast MaxSim, coarse match
Module level:       128-dim  → medium resolution
File level:         256-dim  → fine resolution
Symbol level:       768-dim  → full precision
```

No separate models. No extra storage per resolution. Storage savings of ~8x at the coarse pass with <2% performance loss (per Matryoshka benchmark results on code retrieval).

**Training datasets (from COIR benchmark):**

| Dataset | Why |
|---|---|
| `CodeSearchNetRetrieval` | Primary. NL→function retrieval, 6 languages, expert-labeled. |
| `CosQA` | Real web queries. Honest representation of how developers ask. |
| `CodeEditSearchRetrieval` | Functional similarity — what a function does, not what it's named. |
| `StackOverflowQA` | Developer intent queries in natural language. |

Evaluate on `BrightStackoverflowRetrieval` as adversarial stress test after training. Use **NDCG@10** as primary metric — it penalizes bad ranking, not just presence in top-10.

---

## The Retrieval Pipeline — Top Down

### Query: `"where does auth work"`

```
Step 1 — Encode query
  query_vector_64   = embed(query, dim=64)   ← coarse
  query_vector_768  = embed(query, dim=768)  ← precise

Step 2 — Repo level (galaxy)
  MaxSim(query_vector_64, repo_centroid_set)
  → "authentication and token management" cluster fires
  → descend into matched folder set only

Step 3 — Folder level (solar system)
  MaxSim(query_vector_128, folder_centroid_sets)
    inside matched galaxy only
  → 1-3 folders surfaced

Step 4 — Module level (planet)
  Tag index check first (deterministic):
    "auth" | "jwt" | "token" | "session" in tag vocab?
  Import signature index second (deterministic):
    files importing [jwt, bcrypt, oauth2] in this module?
  MaxSim(query_vector_256, module_centroids) last
  → 2-5 modules surfaced

Step 5 — File level (moon)
  Symbol-BM25 over candidate files:
    verifyToken, AuthMiddleware, checkPermissions, loginUser
  MaxSim(query_vector_768, file_vectors) for reranking
  Rerank score = semantic_score × import_centrality_weight

Step 6 — Symbol level (atom)
  Top-ranked files → extract matching functions via BM25
  Expand via call graph:
    callers of matched symbols (what uses auth?)
    callees of matched symbols (what does auth use?)
  Rank by call graph centrality:
    high in-degree = many things depend on this → surface higher

Return:
  top-K symbols with:
    - file path + line range
    - one-line LLM summary
    - callers list
    - callees list
    - tags
    - centrality score
```

### Why This Doesn't Blow Up at Scale

At every step, the candidate set shrinks:
- Repo level: entire codebase → 1-2 galaxy clusters
- Folder level: all folders → 2-4 folders
- Module level: all modules → 3-8 modules
- File level: all files → 5-20 files
- Symbol level: all symbols → 10-50 symbols

Vector search never runs against the full index. By the time MaxSim runs at symbol level, it is running against ~50 vectors, not 50,000. This is why latency stays low even on large monorepos.

---

## Hybrid Retrieval — The Two Tracks

Every query runs two parallel tracks that merge before final ranking:

```
query
  ├── DETERMINISTIC TRACK
  │     ├── Import signature index
  │     │   (exact package co-occurrence match)
  │     └── Symbol BM25
  │         (fuzzy match on symbol names)
  │
  └── SEMANTIC TRACK
        ├── MaxSim on K-centroid sets
        │   (Matryoshka vectors, multi-resolution)
        └── Call graph centrality weighting

        ↓
  MERGE + RERANK
  final_score = α × semantic_score
              + β × bm25_score
              + γ × import_signature_score
              + δ × call_graph_centrality
```

Weights `α β γ δ` are tunable. Default bias toward semantic + import signature. BM25 acts as a precision anchor. Call graph centrality acts as a significance amplifier — surfacing load-bearing code higher.

---

## Index Store Schema (Local, In-Process)

```
cradle_index/
├── meta.json                    ← repo root, languages, index version
├── graph/
│   ├── call_graph.bin           ← directed call graph (all symbols)
│   ├── import_graph.bin         ← file-level import edges
│   └── louvain_communities.json ← cluster assignments per level
├── embeddings/
│   ├── symbols.bin              ← L0: symbol-level vectors (768-dim)
│   ├── files.bin                ← L1: file-level vectors (768-dim)
│   ├── files_coarse.bin         ← L1: file-level vectors (64-dim)
│   └── centroids/
│       ├── modules.bin          ← L2: K centroids per module
│       ├── folders.bin          ← L3: K centroids per folder
│       └── repo.bin             ← L4: K centroids for repo
├── bm25/
│   └── symbols.index            ← BM25 over symbol names
├── tags/
│   └── tag_vocab.json           ← tag vocabulary per node, hierarchical
└── summaries/
    └── node_summaries.json      ← LLM-generated summaries per node
```

All stored locally. No server. No network. Mmap-friendly binary formats for zero-copy vector reads.

---

## Pre-warm Performance Targets

| Codebase size | Target index time |
|---|---|
| < 10k LOC | < 30 seconds |
| 10k–100k LOC | < 5 minutes |
| 100k–500k LOC | < 20 minutes |
| Monorepo 500k+ LOC | < 60 minutes |

Index is incremental — only re-indexes changed files and propagates changes upward through the hierarchy. A single file change does not trigger a full re-index.

---

## Query Latency Targets

| Operation | Target |
|---|---|
| Full top-down traversal | < 200ms |
| Coarse repo → folder routing | < 10ms |
| Symbol-level BM25 | < 20ms |
| Call graph expansion | < 15ms |
| Final reranking | < 50ms |

---

## CLI Interface

```bash
# Analyze a codebase into SQLite
cradle analyze ./my-repo --model <model> --output ./my-repo/.cradle/index.db

# Current retrieval
cradle retrieve ./my-repo/.cradle/index.db "where is auth handled"
cradle retrieve ./my-repo/.cradle/index.db "who calls processPayment"

# Current DB inspection
cradle visualize-db ./my-repo/.cradle/index.db --output ./.cradle/db-report.md

# Planned later
cradle search "where does auth work"
cradle inspect src/auth/middleware.py::verifyToken
cradle graph src/auth/middleware.py::verifyToken --depth 2
cradle reindex ./my-repo
cradle map ./my-repo
```

---

## What Cradle Is Not

- **Not an LLM coding agent.** Cradle does not generate code, suggest fixes, or reason about code. It retrieves and maps.
- **Not a remote service.** Fully local. Your code never leaves your machine.
- **Not a text search tool.** It does not grep strings. It navigates semantic structure.
- **Not RAG for LLMs (primarily).** Cradle can feed an LLM with precise context, but that is a downstream use case, not the core system.

---

## Implementation Stack

| Component | Tool |
|---|---|
| AST parsing | `tree-sitter` (Python + TypeScript bindings) |
| Call graph | custom resolver on top of tree-sitter output |
| Community detection | `python-louvain` (`community` package) or `igraph` |
| K-Means clustering | `scikit-learn` KMeans (fast, in-process) |
| Embedding model | fine-tuned `nomic-embed-code` via `sentence-transformers` |
| Matryoshka training | `sentence-transformers` MatryoshkaLoss |
| Training data | COIR: CodeSearchNet + CosQA + CodeEditSearch + StackOverflowQA |
| BM25 | `rank_bm25` |
| Vector similarity | `numpy` dot product (no vector DB — in-process arrays) |
| Index storage | `numpy` `.npy` + `msgpack` or `json` for metadata |
| CLI | `typer` or `click` |
| Local LLM (pre-warm summaries) | `ollama` + Mistral 7B / Qwen2.5-Coder-7B |

---

## Build Order

1. **tree-sitter AST extractor** — done
2. **Call graph resolver** — done
3. **SQLite graph store** — done
4. **LLM summaries/tags/categories persisted in DB** — done
5. **Deterministic retrieval over graph DB** — done
6. **DB visualization and inspection reports** — done
7. **Exact/structured `axe_*` DB tools** — next
8. **Embedding model selection and storage schema** — next
9. **Matryoshka embedding generation for nodes/symbols** — next
10. **Embedding-based `axe_semantic_search`** — next
11. **Hybrid semantic + exact reranking** — later
12. **Louvain/K-means centroid routing** — later
13. **Incremental re-indexer / watchdog** — later

---

*Cradle — the codebase knows its own shape. We just read it.*