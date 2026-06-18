# TODO

## Matryoshka Rust API

Prefer exposing Matryoshka prepare as a Rust library API for Jesco/Zed instead of only spawning the `matryoshka-rs` CLI.

Why:

- fewer process, environment, and binary-path issues
- easier cancellation integration
- typed progress events
- easier tests
- no need to configure a `MATRYOSHKA` binary path

Target shape:

```rust
let result = matryoshka_ops::prepare(PrepareOptions {
    repo_root,
    db,
    base_url,
    api_key,
    chat_model,
    embedding_model,
    ignores,
    limit,
    late_interaction: true,
}).await?;
```

The CLI should become a thin wrapper over the same API:

```text
matryoshka-rs prepare -> matryoshka_ops::prepare(...)
jesco_matryoshka -> matryoshka_ops::prepare(...)
```

The API result should mirror `prepare --json` so the CLI and IDE stay behaviorally identical.
