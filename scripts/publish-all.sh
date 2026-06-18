#!/usr/bin/env bash
set -euo pipefail

CRATES=(
  matryoshka-core-ir
  matryoshka-embed-client
  matryoshka-parser
  matryoshka-store-sqlite
  matryoshka-enricher
  matryoshka-resolver
  matryoshka-read-api
  matryoshka-search
  matryoshka-watcher
  matryoshka-indexer
  matryoshka
  matryoshka-cli
)

MATRYOSHKA_DEPS=(
  matryoshka-core-ir
  matryoshka-parser
  matryoshka-resolver
  matryoshka-store-sqlite
  matryoshka-enricher
  matryoshka-embed-client
  matryoshka-indexer
  matryoshka-search
  matryoshka-read-api
  matryoshka-watcher
  matryoshka
)

usage() {
  cat <<'EOF'
Usage:
  scripts/publish-all.sh --execute [--version X.Y.Z] [--allow-dirty]
  scripts/publish-all.sh --dry-run [--version X.Y.Z]

What it does:
  1. Reads the current workspace version from Cargo.toml.
  2. Bumps to the next patch version unless --version is supplied.
  3. Updates all internal Matryoshka workspace dependency versions.
  4. Runs cargo check.
  5. Publishes all crates in dependency order.

Examples:
  scripts/publish-all.sh --dry-run
  scripts/publish-all.sh --execute --version 0.1.2
  scripts/publish-all.sh --execute --allow-dirty
EOF
}

mode="dry-run"
requested_version=""
allow_dirty=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      mode="execute"
      shift
      ;;
    --dry-run)
      mode="dry-run"
      shift
      ;;
    --version)
      requested_version="${2:-}"
      if [[ -z "$requested_version" ]]; then
        echo "missing value for --version" >&2
        exit 2
      fi
      shift 2
      ;;
    --allow-dirty)
      allow_dirty="--allow-dirty"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

current_version="$(
  awk '
    /^\[workspace.package\]/ { in_workspace_package = 1; next }
    /^\[/ && in_workspace_package { exit }
    in_workspace_package && /^version = / {
      gsub(/"/, "", $3)
      print $3
      exit
    }
  ' Cargo.toml
)"

if [[ -z "$current_version" ]]; then
  echo "could not find [workspace.package] version in Cargo.toml" >&2
  exit 1
fi

next_patch_version() {
  local version="$1"
  local major minor patch
  IFS='.' read -r major minor patch <<<"$version"
  if [[ ! "$major" =~ ^[0-9]+$ || ! "$minor" =~ ^[0-9]+$ || ! "$patch" =~ ^[0-9]+$ ]]; then
    echo "cannot auto-bump non-simple semver: $version" >&2
    exit 1
  fi
  printf '%s.%s.%s\n' "$major" "$minor" "$((patch + 1))"
}

new_version="${requested_version:-$(next_patch_version "$current_version")}"

if [[ ! "$new_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "--version must be simple semver like 0.1.2; got: $new_version" >&2
  exit 2
fi

publish_flags=()
if [[ -n "$allow_dirty" ]]; then
  publish_flags+=("$allow_dirty")
elif [[ "$mode" == "execute" && -n "$(git status --porcelain)" ]]; then
  echo "worktree is dirty. Commit/stash changes or pass --allow-dirty." >&2
  git status --short >&2
  exit 1
fi

echo "Current version: $current_version"
echo "New version:     $new_version"
echo "Mode:            $mode"
echo
echo "Publish order:"
printf '  %s\n' "${CRATES[@]}"
echo

if [[ "$mode" == "dry-run" ]]; then
  echo "Dry run only. No files changed and nothing published."
  echo "Run with --execute to update Cargo.toml and publish."
  exit 0
fi

perl -0pi -e '
  my $new = $ENV{"NEW_VERSION"};
  s/(\[workspace\.package\][^\[]*?version\s*=\s*")[^"]+(")/$1$new$2/s;
' Cargo.toml

for crate in "${MATRYOSHKA_DEPS[@]}"; do
  CRATE_NAME="$crate" NEW_VERSION="$new_version" perl -0pi -e '
    my $crate = quotemeta($ENV{"CRATE_NAME"});
    my $new = $ENV{"NEW_VERSION"};
    s/($crate\s*=\s*\{\s*version\s*=\s*")[^"]+(")/$1$new$2/g;
  ' Cargo.toml
done

cargo check

for crate in "${CRATES[@]}"; do
  echo
  echo "Publishing $crate $new_version"
  cargo publish -p "$crate" "${publish_flags[@]}"
done

echo
echo "Published all Matryoshka crates at $new_version."
