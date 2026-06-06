#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Clean dist ==="
rm -rf dist/ build/ src/*.egg-info

echo "=== Bump version ==="
CURRENT=$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "//;s/"//')
echo "Current version: $CURRENT"

# Patch version: 0.1.3 -> 0.1.4
IFS='.' read -r major minor patch <<< "$CURRENT"
NEW_PATCH=$((patch + 1))
NEW_VERSION="${major}.${minor}.${NEW_PATCH}"
echo "New version: $NEW_VERSION"

sed -i.bak "s/^version = \"${CURRENT}\"/version = \"${NEW_VERSION}\"/" pyproject.toml
rm pyproject.toml.bak

echo "=== Build ==="
python -m build

echo "=== Upload to PyPI ==="
twine upload --repository pypi dist/*

echo "=== Done: jesco-matryoshka v${NEW_VERSION} ==="
echo "View at: https://pypi.org/project/jesco-matryoshka/${NEW_VERSION}/"
