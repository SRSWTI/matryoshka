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

Cradle writes the SQLite database to:

```text
/path/to/repo/.cradle/<repo-name>.db
```

Example:

```text
/Users/rohit/pi/.cradle/pi.db
```

## Full Usage

See [CRADLE_USAGE.md](CRADLE_USAGE.md) for:

- global install
- editable module install
- analysis commands
- semantic indexing
- retrieval commands
- exclusion flags
- Python module usage
