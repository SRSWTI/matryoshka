# Matryoshka Rust Core

This workspace is the Rust-first rewrite of Matryoshka's agent-facing core.
It keeps the Python implementation as reference material and focuses the Rust
surface on prewarming, incremental refresh, semantic search, and rich read
cards.

## Crates

- `core-ir`: shared facts, cards, semantic records, provenance, and search/read types.
- `parser`: source walking and lightweight fact extraction.
- `resolver`: folder graph, import resolution, dependency edges, and initial records.
- `store-sqlite`: canonical SQLite persistence.
- `enricher`: rich file/folder/repo cards through MLX chat or offline heuristic fallback.
- `embed-client`: OpenAI-compatible `/v1/embeddings` client plus deterministic test embedder.
- `indexer`: full prewarm orchestration.
- `search`: hybrid semantic + lexical search over persisted semantic records.
- `read-api`: rich `read` and `read-more` card assembly.
- `watcher`: invalidation planning for future incremental refresh loops.
- `cli`: thin `matryoshka-rs` command wrapper.

## Local Smoke Test

```sh
cargo run -p matryoshka-cli -- index tests/fixtures/mini_repo \
  --db /tmp/matryoshka-rs-mini.db \
  --offline

cargo run -p matryoshka-cli -- search \
  --db /tmp/matryoshka-rs-mini.db \
  --offline \
  "api key loaded from environment"

cargo run -p matryoshka-cli -- read \
  --db /tmp/matryoshka-rs-mini.db \
  --repo-root tests/fixtures/mini_repo \
  src/auth/middleware.py
```

## MLX Endpoint Defaults

The non-offline path expects your local OpenAI-compatible MLX server:

- Base URL: `http://127.0.0.1:44445`
- API key: `2508`
- Embedding model: `mlx-community--embeddinggemma-300m-bf16`
- Enrichment model: `MercuriusDream--Qwen3.5-4B-MLX-mxfp8`

Chat enrichment sends `chat_template_kwargs.enable_thinking = false` by
default.
