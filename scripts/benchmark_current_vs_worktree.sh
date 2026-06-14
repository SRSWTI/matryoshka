#!/usr/bin/env bash
set -euo pipefail

baseline_root="${1:-/Users/rohit/cradle-embed}"
candidate_root="${2:-$(pwd)}"
sample_repo="${3:-$baseline_root}"
work_dir="${4:-$(mktemp -d /private/tmp/matryoshka-bench.XXXXXX)}"

baseline_db="$work_dir/baseline.db"
candidate_db="$work_dir/candidate.db"
baseline_out="$work_dir/baseline"
candidate_out="$work_dir/candidate"

mkdir -p "$baseline_out" "$candidate_out"

baseline_cli="$baseline_root/target/debug/matryoshka-rs"
candidate_cli="$candidate_root/target/debug/matryoshka-rs"

run_timed() {
  local label="$1"
  local out_dir="$2"
  shift 2
  echo "== $label"
  mkdir -p "$out_dir"
  /usr/bin/time -p "$@" >"$out_dir/stdout.txt" 2>"$out_dir/time.txt"
  cat "$out_dir/stdout.txt"
  cat "$out_dir/time.txt"
}

run_search() {
  local cli="$1"
  local db="$2"
  local label="$3"
  local out_dir="$4"
  local query="$5"
  mkdir -p "$out_dir"
  run_timed "$label" "$out_dir" "$cli" search --db "$db" --offline --limit 5 "$query"
}

echo "baseline_root=$baseline_root"
echo "candidate_root=$candidate_root"
echo "sample_repo=$sample_repo"
echo "work_dir=$work_dir"

echo "== build baseline"
cargo build -p matryoshka-cli --manifest-path "$baseline_root/Cargo.toml"

echo "== build candidate"
cargo build -p matryoshka-cli --manifest-path "$candidate_root/Cargo.toml"

run_timed "baseline index" "$baseline_out/index" \
  "$baseline_cli" index "$sample_repo" --db "$baseline_db" --offline

run_timed "candidate index" "$candidate_out/index" \
  "$candidate_cli" index "$sample_repo" --db "$candidate_db" --offline

if "$baseline_cli" prewarm --help >/dev/null 2>&1; then
  run_timed "baseline prewarm" "$baseline_out/prewarm" \
    "$baseline_cli" prewarm --db "$baseline_db" --offline --limit 5
else
  echo "== baseline prewarm"
  echo "unsupported"
fi

if "$candidate_cli" prewarm --help >/dev/null 2>&1; then
  run_timed "candidate prewarm" "$candidate_out/prewarm" \
    "$candidate_cli" prewarm --db "$candidate_db" --offline --limit 5
else
  echo "== candidate prewarm"
  echo "unsupported"
fi

queries=(
  "repository architecture"
  "where is SearchEngine defined"
  "where should I edit parser behavior"
  "dependency impact blast radius"
  "tests for indexer"
)

for query in "${queries[@]}"; do
  safe_name="$(echo "$query" | tr -c '[:alnum:]_' '_')"
  run_search "$baseline_cli" "$baseline_db" "baseline search: $query" "$baseline_out/search-$safe_name" "$query"
  run_search "$candidate_cli" "$candidate_db" "candidate search: $query" "$candidate_out/search-$safe_name" "$query"
done

if "$candidate_cli" op --help >/dev/null 2>&1; then
  run_timed "candidate op find-symbol" "$candidate_out/op-find-symbol" \
    "$candidate_cli" op --db "$candidate_db" --offline --limit 5 find-symbol "SearchEngine"
  run_timed "candidate op edit-target" "$candidate_out/op-edit-target" \
    "$candidate_cli" op --db "$candidate_db" --offline --limit 5 edit-target "parser behavior"
  run_timed "candidate op architecture" "$candidate_out/op-architecture" \
    "$candidate_cli" op --db "$candidate_db" --offline --limit 5 architecture "Matryoshka"
fi

run_timed "baseline read semantic_search" "$baseline_out/read-semantic-search" \
  "$baseline_cli" read --db "$baseline_db" --repo-root "$sample_repo" crates/search/src/semantic_search.rs

if "$candidate_cli" read-bundle --help >/dev/null 2>&1; then
  run_timed "candidate read-bundle search" "$candidate_out/read-bundle-search" \
    "$candidate_cli" read-bundle --db "$candidate_db" --repo-root "$sample_repo" --offline --mode edit --limit 6 --related 3 "SearchEngine search behavior"
fi

echo "benchmark artifacts: $work_dir"
