# Cradle Installation and Usage

## What You Get

Installing this package gives you:

- the `cradle` CLI
- the Python module `cradle`
- default analysis output inside the analyzed repository under `.cradle/`

When you run `cradle analyze /path/to/repo` without `--output`, Cradle now writes the SQLite DB to:

- `/path/to/repo/.cradle/<repo-name>.db`

Example:

- analyzing `/Users/rohit/pi` writes the DB to `/Users/rohit/pi/.cradle/pi.db`

## Install Options

### Option 1: Install globally with `uv tool`

This is the cleanest way to use `cradle` from anywhere on your machine.

```bash
cd /Users/rohit/cradle-embed
uv tool install .
```

If you update the code later and want to refresh the global install:

```bash
cd /Users/rohit/cradle-embed
uv tool install --reinstall .
```

After that, the command should be available globally:

```bash
cradle --help
```

### Option 2: Install into a virtualenv in editable mode

Use this if you want the package importable as a module while continuing to edit the source locally.

```bash
cd /Users/rohit/cradle-embed
uv venv
uv pip install -e .
```

Then run it as:

```bash
.venv/bin/cradle --help
```

or import it from Python:

```python
import cradle
```

## Dependencies

Core dependencies are declared in `pyproject.toml` and installed with the package.

Main runtime dependencies:

- `numpy`
- `networkx`
- `tree-sitter`
- `tree-sitter-python`
- `tree-sitter-typescript`
- `mlx-embeddings` on macOS

Optional CPU embedding backend:

```bash
uv pip install -e '.[cpu]'
```

Dev dependencies:

```bash
uv pip install -e '.[dev]'
```

## Basic Workflow In Any Repository

### 1. Analyze a repository

```bash
cradle analyze /path/to/repo \
  --model YOUR_MODEL \
  --api-key YOUR_API_KEY
```

What happens by default:

- Cradle creates `/path/to/repo/.cradle/` if it does not exist.
- The SQLite DB is written to `/path/to/repo/.cradle/<repo-name>.db`.
- Cradle skips common generated/virtualenv directories, nested `test`/`tests` directories, and files matched by the repository root `.gitignore`.

Example for `/Users/rohit/pi`:

```bash
cradle analyze /Users/rohit/pi \
  --model fa2a6d12ba62dae0eef63d36a1944f9c8170e183 \
  --api-key 2508 \
  --max-parallel-requests 8 \
  --max-tokens 600 \
  --thinking-budget 0 \
  --log-level INFO
```

That writes:

- DB: `/Users/rohit/pi/.cradle/pi.db`

### 2. Optionally exclude files or folders during analysis

```bash
cradle analyze /Users/rohit/pi \
  --model fa2a6d12ba62dae0eef63d36a1944f9c8170e183 \
  --api-key 2508 \
  --exclude-path tests \
  --exclude-path docs/generated \
  --exclude-path '*.d.ts' \
  --exclude-extension md
```

Supported exclusion styles:

- folder name: `tests`
- subtree: `docs/generated`
- exact file: `src/foo/bar.py`
- simple glob: `*.d.ts`
- extension: `--exclude-extension .md`

These exclusions are additive. Cradle already skips nested `test`/`tests` directories by default and also respects the repository root `.gitignore`.

### 3. Build the semantic index

```bash
cradle semantic-index /Users/rohit/pi/.cradle/pi.db \
  --model mlx-community/embeddinggemma-300m-bf16 \
  --backend mlx \
  --output-dir /Users/rohit/pi/.cradle/pi-semantic
```

### 4. Run retrieval commands

Semantic concept search:

```bash
cradle semantic-search /Users/rohit/pi/.cradle/pi.db \
  'where is oauth authentication handled' \
  --index-dir /Users/rohit/pi/.cradle/pi-semantic
```

Hierarchy search:

```bash
cradle hierarchy-search /Users/rohit/pi/.cradle/pi.db \
  'how are api keys loaded from environment' \
  --index-dir /Users/rohit/pi/.cradle/pi-semantic
```

Exact symbol search:

```bash
cradle symbol-search /Users/rohit/pi/.cradle/pi.db 'getEnvApiKey'
```

DB visualization:

```bash
cradle visualize-db /Users/rohit/pi/.cradle/pi.db \
  --output /Users/rohit/pi/.cradle/pi-db-report.md
```

Focused symbol/file neighborhood:

```bash
cradle visualize-focus /Users/rohit/pi/.cradle/pi.db 'getEnvApiKey' \
  --kind symbol \
  --output /Users/rohit/pi/.cradle/pi-focus.md
```

## Python Module Usage

You can also use Cradle directly from Python.

```python
from pathlib import Path

from cradle import CradlePipeline, PipelineConfig
from cradle.storage import CradleDatabase

repo_root = Path('/Users/rohit/pi')
pipeline = CradlePipeline(config=PipelineConfig())
graph = pipeline.analyze(repo_root)
CradleDatabase(repo_root / '.cradle' / f'{repo_root.name}.db').replace_graph(graph)
```

## Notes

- SQLite remains the source of truth.
- Semantic vectors live in a separate sidecar directory.
- The default DB path is only used when `--output` is omitted.
- If you want a custom DB path, keep using `--output`.
