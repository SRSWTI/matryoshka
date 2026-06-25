#!/usr/bin/env bash
#
# test_milestone1_chunks.sh
#
# End-to-end test for Milestone 1 (function-level code chunk extraction).
#
# What this script does:
#   1. Builds the matryoshka-rs CLI (release).
#   2. Indexes a repo (either one you pass in, or a generated sample repo) in
#      OFFLINE mode (no MLX / no embeddings needed).
#   3. Dumps the extracted `code_chunks` rows directly from the SQLite DB so you
#      can verify:
#        - full symbol bodies are preserved (no truncation)
#        - docstrings / doc comments are used as the summary when present
#        - undocumented chunks have summary_source = "empty"
#        - chunk boundaries match the AST symbol ranges
#
# Usage:
#   bash scripts/test_milestone1_chunks.sh [REPO_PATH]
#
#   If REPO_PATH is omitted, a throwaway sample repo with Rust/Python/TS files
#   is generated so you can see the expected behavior immediately.
#
# Examples:
#   bash scripts/test_milestone1_chunks.sh
#   bash scripts/test_milestone1_chunks.sh /Users/rohit/cradle-embed
#   bash scripts/test_milestone1_chunks.sh ~/projects/my-rust-crate
#
# Requirements:
#   - rust toolchain (cargo)
#   - sqlite3 CLI on PATH (macOS ships it)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CLI_BIN="$ROOT_DIR/target/release/matryoshka-rs"

# ------------------------------------------------------------------
# Resolve the target repo: either a user-supplied path or a generated sample.
# ------------------------------------------------------------------
if [[ $# -ge 1 && -n "$1" ]]; then
  TARGET_REPO="$1"
  if [[ ! -d "$TARGET_REPO" ]]; then
    echo "ERROR: repo path does not exist: $TARGET_REPO" >&2
    exit 1
  fi
  TARGET_REPO="$(cd "$TARGET_REPO" && pwd)"
  GENERATED_SAMPLE=0
else
  TARGET_REPO="$(mktemp -d)/sample_repo"
  GENERATED_SAMPLE=1
fi

# Use a dedicated DB path under the repo's .matryoshka dir so we never clobber
# an existing index. For user repos we append ".milestone1_test" to the db name
# to keep it clearly separated from a real index.
if [[ "$GENERATED_SAMPLE" -eq 1 ]]; then
  DB_PATH="$TARGET_REPO/.matryoshka/matryoshka.db"
else
  DB_PATH="$TARGET_REPO/.matryoshka/milestone1_test.db"
fi

echo "==> Building matryoshka-rs (release, this may take a moment)..."
cargo build --release -p matryoshka-cli 2>&1 | tail -3
echo

if [[ ! -x "$CLI_BIN" ]]; then
  echo "ERROR: matryoshka-rs binary not found at $CLI_BIN" >&2
  exit 1
fi

if [[ "$GENERATED_SAMPLE" -eq 1 ]]; then
  echo "==> No repo path provided; generating sample repo at $TARGET_REPO"
  mkdir -p "$TARGET_REPO/src" "$TARGET_REPO/python" "$TARGET_REPO/ts"

# ------------------------------------------------------------------
# Rust file: mix of documented / undocumented / generic-doc functions
# ------------------------------------------------------------------
  cat > "$TARGET_REPO/src/lib.rs" <<'RUST'
//! Crate-level module docs (not attached to a function).

/// Resumes attack mode after handoff.
///
/// Cancels the countdown and updates internal state.
pub fn handle_resume_countdown(state: &mut State) -> bool {
    state.countdown.cancel();
    state.mode = Mode::Attack;
    true
}

pub fn undocumented_helper(x: i32) -> i32 {
    x + 1
}

/// TODO
pub fn generic_doc_function() {}

pub struct Coordinator {
    countdown: Countdown,
    mode: Mode,
}

impl Coordinator {
    /// Cancels the active countdown with a reason and returns whether
    /// a countdown was actually cancelled.
    pub fn cancel_countdown(&mut self, reason: &str) -> bool {
        self.countdown.cancel(reason)
    }

    pub fn enter_attack_mode(&mut self) {
        self.mode = Mode::Attack;
    }
}
RUST

# ------------------------------------------------------------------
# Python file: docstrings (single + multi-line) and undocumented
# ------------------------------------------------------------------
  cat > "$TARGET_REPO/python/service.py" <<'PY'
"""Module-level docstring for the service."""


def refresh_token(token: str) -> str:
    """Refresh the given token and return the new value."""
    return token + "_new"


def undocumented_function(a, b):
    return a + b


class AttackCoordinator:
    """Coordinates attack-mode state transitions and countdown cancellation."""

    def handle_resume_countdown(self):
        """Resumes attack mode after handoff and cancels the countdown."""
        self.mode = "attack"
        self.countdown.cancel()

    def cancel_countdown(self, reason):
        return False
PY

# ------------------------------------------------------------------
# TypeScript file: JSDoc blocks and line comments
# ------------------------------------------------------------------
  cat > "$TARGET_REPO/ts/client.ts" <<'TS'
/**
 * Resumes attack mode after handoff.
 * Cancels the countdown.
 */
export function handleResumeCountdown(): void {
  cancelCountdown();
}

export function undocumented(): number {
  return 42;
}

// Quick line comment that should be picked up as a doc.
export function lineCommented(): void {
  console.log("hi");
}

export class ApiClient {
  /**
   * Fetches a token from the remote endpoint.
   */
  async fetchToken(): Promise<string> {
    return "token";
  }
}
TS
else
  echo "==> Using user-supplied repo: $TARGET_REPO"
fi
echo

mkdir -p "$(dirname "$DB_PATH")"

echo "==> Indexing repo in OFFLINE mode (no MLX required)..."
echo "    repo: $TARGET_REPO"
echo "    db:   $DB_PATH"
echo "    (streaming progress live; this may take a while on large repos)"
echo
"$CLI_BIN" index "$TARGET_REPO" \
  --offline \
  --db "$DB_PATH" \
  --ignore ".matryoshka" \
  --progress-jsonl
echo
echo "==> Indexing complete."

if [[ ! -f "$DB_PATH" ]]; then
  echo "ERROR: DB not created at $DB_PATH" >&2
  exit 1
fi

echo "==> Dumping code_chunks from SQLite"
echo "    DB: $DB_PATH"
echo
echo "=================================================================="
echo " CODE CHUNKS (path | symbol | kind | lines | summary_source | summary)"
echo "=================================================================="
sqlite3 -header -column "$DB_PATH" "
SELECT
  json_extract(payload_json, '\$.path')            AS path,
  json_extract(payload_json, '\$.qualified_name')  AS symbol,
  json_extract(payload_json, '\$.kind')            AS kind,
  (json_extract(payload_json, '\$.start_line') || '-' ||
   json_extract(payload_json, '\$.end_line'))      AS lines,
  json_extract(payload_json, '\$.summary_source')  AS summary_source,
  substr(json_extract(payload_json, '\$.summary'), 1, 60) AS summary
FROM code_chunks
ORDER BY path, start_line;
"
echo

echo "=================================================================="
echo " CHUNK COUNT BY SUMMARY SOURCE"
echo "=================================================================="
sqlite3 -header -column "$DB_PATH" "
SELECT
  json_extract(payload_json, '\$.summary_source') AS summary_source,
  COUNT(*) AS chunk_count
FROM code_chunks
GROUP BY summary_source
ORDER BY summary_source;
"
echo

echo "=================================================================="
echo " FULL DETAIL: first documented chunk (doc_comment or docstring)"
echo "=================================================================="
sqlite3 "$DB_PATH" "
SELECT json_extract(payload_json, '\$.qualified_name') || '  [' ||
       json_extract(payload_json, '\$.path') || ':' ||
       json_extract(payload_json, '\$.start_line') || '-' ||
       json_extract(payload_json, '\$.end_line') || ']  (' ||
       json_extract(payload_json, '\$.summary_source') || ')'
FROM code_chunks
WHERE json_extract(payload_json, '\$.summary_source') IN ('doc_comment', 'docstring')
ORDER BY path, start_line
LIMIT 1;
"
sqlite3 "$DB_PATH" "
SELECT json_extract(payload_json, '\$.code')
FROM code_chunks
WHERE json_extract(payload_json, '\$.summary_source') IN ('doc_comment', 'docstring')
ORDER BY path, start_line
LIMIT 1;
"
echo

echo "=================================================================="
echo " FULL DETAIL: first empty (undocumented) chunk — Milestone 2 target"
echo "=================================================================="
sqlite3 "$DB_PATH" "
SELECT json_extract(payload_json, '\$.qualified_name') || '  [' ||
       json_extract(payload_json, '\$.path') || ':' ||
       json_extract(payload_json, '\$.start_line') || '-' ||
       json_extract(payload_json, '\$.end_line') || ']'
FROM code_chunks
WHERE json_extract(payload_json, '\$.summary_source') = 'empty'
ORDER BY path, start_line
LIMIT 1;
"
sqlite3 "$DB_PATH" "
SELECT json_extract(payload_json, '\$.code')
FROM code_chunks
WHERE json_extract(payload_json, '\$.summary_source') = 'empty'
ORDER BY path, start_line
LIMIT 1;
"
echo

echo "==> Done."
if [[ "$GENERATED_SAMPLE" -eq 1 ]]; then
  echo "    Generated sample repo left at: $TARGET_REPO"
else
  echo "    Indexed repo: $TARGET_REPO"
fi
echo "    Inspect the DB manually with:"
echo "      sqlite3 \"$DB_PATH\""
echo "    e.g.: SELECT path, qualified_name, summary_source, summary FROM code_chunks;"
