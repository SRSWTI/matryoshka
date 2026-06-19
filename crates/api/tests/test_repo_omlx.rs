use matryoshka::{
    CardsOptions, Matryoshka, MatryoshkaConfig, MatryoshkaEvent, PrepareOptions, PrepareStatus,
    ReadBundleOptions, RerankerOptions, SearchOptions, artifact_gap_count, progress_state_path,
    ready_marker_path,
};
use matryoshka_core_ir::MatryoshkaProgressEvent;
use rusqlite::Connection;
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:44447";
const DEFAULT_API_KEY: &str = "2508";
const DEFAULT_CHAT_MODEL: &str = "srswti--bodega-raptor-90m";
const DEFAULT_EMBED_MODEL: &str = "mlx-community--embeddinggemma-300m-bf16";
const DEFAULT_RERANK_MODEL: &str = "mlx-community--Qwen3-Reranker-0.6B-mxfp8";

#[test]
#[ignore = "requires MATRYOSHKA_TEST_REPO and a live oMLX server"]
fn test_repo_live_prepare_search_read_and_progress_work_through_rust_api() {
    let Some(repo) = std::env::var_os("MATRYOSHKA_TEST_REPO").map(PathBuf::from) else {
        eprintln!("set MATRYOSHKA_TEST_REPO=/path/to/test_repo to run this test");
        return;
    };
    assert!(
        repo.exists(),
        "test repo does not exist: {}",
        repo.display()
    );

    let db_dir = repo.join(".matryoshka/api-test-repo-live");
    if db_dir.exists() {
        fs::remove_dir_all(&db_dir).unwrap();
    }
    fs::create_dir_all(&db_dir).unwrap();
    let db = db_dir.join("matryoshka.db");
    let probe = repo.join("watcher/src/matryoshka_api_live_probe.rs");
    let _ = fs::remove_file(&probe);

    let api = Matryoshka::new(
        MatryoshkaConfig::new(&repo)
            .with_db(&db)
            .with_endpoint(base_url(), api_key())
            .with_models(chat_model(), embed_model())
            .with_ignored_paths([".matryoshka", "target"]),
    );

    let first = completed_summary(&run_prepare(&api, "no_db"));
    println!(
        "no_db actions={:?} files={} symbols={} records={} warmed={}",
        first.actions_taken,
        first.file_count,
        first.symbol_count,
        first.retrieval_index.semantic_records,
        first.prewarm.warmed_hit_count
    );
    assert_eq!(first.status, PrepareStatus::Ready);
    assert_eq!(first.actions_taken, vec!["index", "prepare_results"]);
    assert!(first.file_count >= 20);
    assert!(first.symbol_count > 0);
    assert_eq!(artifact_gap_count(&first.artifact_quality), 0);
    assert!(first.retrieval_index.semantic_records > 0);
    assert!(first.retrieval_index.fts_records > 0);
    assert!(first.retrieval_index.records_with_late_vectors > 0);
    assert!(first.prewarm.warmed_hit_count > 0);
    assert!(ready_marker_path(&db).exists());
    assert_ready_progress_state(&db);

    let search_hits = api
        .search(
            "watcher debounce added changed removed paths",
            SearchOptions::default(),
        )
        .unwrap();
    println!(
        "search top={:?}",
        search_hits
            .iter()
            .take(5)
            .map(|hit| (&hit.path, hit.score))
            .collect::<Vec<_>>()
    );
    assert!(!search_hits.is_empty());
    assert!(
        search_hits
            .iter()
            .any(|hit| hit.path == "watcher/src/poller.rs")
    );

    let reranked_hits = api
        .search(
            "watcher debounce added changed removed paths",
            SearchOptions {
                limit: 8,
                reranker: RerankerOptions::Omlx {
                    model: rerank_model(),
                    candidates: 20,
                },
            },
        )
        .unwrap();
    println!(
        "reranked search top={:?}",
        reranked_hits
            .iter()
            .take(5)
            .map(|hit| (&hit.path, hit.score))
            .collect::<Vec<_>>()
    );
    assert!(!reranked_hits.is_empty());

    let read = api.read("watcher/src/poller.rs").unwrap();
    println!(
        "read file={} symbols={} deps={} dependents={}",
        read.file.path,
        read.symbols.len(),
        read.depends_on.len(),
        read.dependents.len()
    );
    assert_eq!(read.file.path, "watcher/src/poller.rs");
    assert!(!read.symbols.is_empty());

    let bundle = api
        .read_bundle(ReadBundleOptions::new("watcher debounce update flow"))
        .unwrap();
    println!(
        "read_bundle primary={} related={}",
        bundle.primary.file.path,
        bundle.related.len()
    );
    assert!(!bundle.primary.file.path.is_empty());

    let cards = api.cards(CardsOptions { empty_only: false }).unwrap();
    let empty_cards = api.cards(CardsOptions { empty_only: true }).unwrap();
    println!("cards total={} empty={}", cards.len(), empty_cards.len());
    assert!(!cards.is_empty());
    assert!(empty_cards.is_empty());

    fs::remove_file(ready_marker_path(&db)).unwrap();
    let marker = completed_summary(&run_prepare(&api, "marker_missing"));
    println!("marker_missing actions={:?}", marker.actions_taken);
    assert_eq!(marker.status, PrepareStatus::Ready);
    assert_eq!(marker.actions_taken, vec!["update", "prepare_results"]);
    assert!(ready_marker_path(&db).exists());

    fs::write(
        &probe,
        "pub fn matryoshka_api_probe_added_symbol() -> &'static str { \"added\" }\n",
    )
    .unwrap();
    let added = completed_summary(&run_prepare(&api, "file_added"));
    println!(
        "file_added actions={:?} changed_files={}",
        added.actions_taken, added.changed_files
    );
    assert_eq!(added.status, PrepareStatus::Ready);
    assert!(added.changed_files >= 1);
    assert_eq!(
        count_file_cards(&db, "watcher/src/matryoshka_api_live_probe.rs"),
        1
    );
    assert!(count_semantic_records(&db, "watcher/src/matryoshka_api_live_probe.rs") > 0);
    assert!(count_late_vectors_for_path(&db, "watcher/src/matryoshka_api_live_probe.rs") > 0);
    assert!(fts_match_count(&db, "matryoshka_api_probe_added_symbol") > 0);

    fs::write(
        &probe,
        "pub fn matryoshka_api_probe_changed_symbol() -> &'static str { \"changed\" }\n",
    )
    .unwrap();
    let changed = completed_summary(&run_prepare(&api, "file_changed"));
    println!(
        "file_changed actions={:?} changed_files={}",
        changed.actions_taken, changed.changed_files
    );
    assert_eq!(changed.status, PrepareStatus::Ready);
    assert!(changed.changed_files >= 1);
    assert!(fts_match_count(&db, "matryoshka_api_probe_changed_symbol") > 0);
    assert_eq!(fts_match_count(&db, "matryoshka_api_probe_added_symbol"), 0);

    fs::remove_file(&probe).unwrap();
    let deleted = completed_summary(&run_prepare(&api, "file_deleted"));
    println!(
        "file_deleted actions={:?} removed_files={}",
        deleted.actions_taken, deleted.removed_files
    );
    assert_eq!(deleted.status, PrepareStatus::Ready);
    assert!(deleted.removed_files >= 1);
    assert_eq!(
        count_file_cards(&db, "watcher/src/matryoshka_api_live_probe.rs"),
        0
    );
    assert_eq!(
        count_semantic_records(&db, "watcher/src/matryoshka_api_live_probe.rs"),
        0
    );
    assert_eq!(
        count_late_vectors_for_path(&db, "watcher/src/matryoshka_api_live_probe.rs"),
        0
    );
    assert_eq!(
        fts_match_count(&db, "matryoshka_api_probe_changed_symbol"),
        0
    );

    seed_orphaned_cards(&db);
    seed_orphaned_semantic_artifacts(&db);
    assert!(orphan_file_cards(&db) > 0);
    assert!(orphan_folder_cards(&db) > 0);
    assert!(orphan_fts_records(&db) > 0);
    assert!(orphan_late_vectors(&db) > 0);
    let pruned = completed_summary(&run_prepare(&api, "orphan_prune"));
    println!("orphan_prune actions={:?}", pruned.actions_taken);
    assert_eq!(pruned.status, PrepareStatus::Ready);
    assert_eq!(orphan_file_cards(&db), 0);
    assert_eq!(orphan_folder_cards(&db), 0);
    assert_eq!(orphan_fts_records(&db), 0);
    assert_eq!(orphan_late_vectors(&db), 0);

    blank_two_card_summaries(&db);
    let repaired = completed_summary(&run_prepare(&api, "card_gaps"));
    println!("card_gaps actions={:?}", repaired.actions_taken);
    assert_eq!(repaired.status, PrepareStatus::Ready);
    assert_eq!(repaired.actions_taken, vec!["repair", "prepare_results"]);
    assert_eq!(artifact_gap_count(&repaired.artifact_quality), 0);
    assert!(
        api.cards(CardsOptions { empty_only: true })
            .unwrap()
            .is_empty()
    );

    delete_search_data(&db);
    let rebuilt = completed_summary(&run_prepare(&api, "search_missing"));
    println!(
        "search_missing actions={:?} records={} fts={} late_records={}",
        rebuilt.actions_taken,
        rebuilt.retrieval_index.semantic_records,
        rebuilt.retrieval_index.fts_records,
        rebuilt.retrieval_index.records_with_late_vectors
    );
    assert_eq!(rebuilt.status, PrepareStatus::Ready);
    assert_eq!(
        rebuilt.actions_taken,
        vec!["rebuild_search", "prepare_results"]
    );
    assert!(rebuilt.retrieval_index.semantic_records > 0);
    assert!(rebuilt.retrieval_index.fts_records > 0);
    assert!(rebuilt.retrieval_index.records_with_late_vectors > 0);

    let healthy = completed_summary(&run_prepare(&api, "healthy"));
    println!(
        "healthy actions={:?} changed_files={} removed_files={}",
        healthy.actions_taken, healthy.changed_files, healthy.removed_files
    );
    assert_eq!(healthy.status, PrepareStatus::Ready);
    assert_eq!(healthy.actions_taken, vec!["update", "prepare_results"]);
    assert_eq!(healthy.changed_files, 0);
    assert_eq!(healthy.removed_files, 0);
    assert_eq!(orphan_file_cards(&db), 0);
    assert_eq!(orphan_folder_cards(&db), 0);
    assert_eq!(orphan_fts_records(&db), 0);
    assert_eq!(orphan_late_vectors(&db), 0);
}

fn run_prepare(api: &Matryoshka, scenario: &str) -> Vec<MatryoshkaEvent> {
    let mut events = Vec::new();
    let summary = api
        .prepare_with_progress(
            PrepareOptions {
                limit: 4,
                queries: vec![
                    "watcher debounce added changed removed paths".into(),
                    "query planner reranker search results".into(),
                    "read bundle related dependencies".into(),
                ],
                write_progress_state: true,
            },
            |event| events.push(event),
        )
        .unwrap_or_else(|err| panic!("{scenario} prepare failed: {err:#}"));
    assert!(
        summary.is_ready(),
        "{scenario} prepare summary was not ready: {summary:#?}",
    );
    assert_progress_events_are_consistent(scenario, &events);
    assert_lock_events_are_consistent(scenario, &events);
    events
}

fn completed_summary(events: &[MatryoshkaEvent]) -> matryoshka::PrepareSummary {
    events
        .iter()
        .find_map(|event| match event {
            MatryoshkaEvent::PrepareCompleted { summary } => Some(summary.clone()),
            _ => None,
        })
        .expect("prepare should emit a completion event")
}

fn assert_progress_events_are_consistent(scenario: &str, events: &[MatryoshkaEvent]) {
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaEvent::PrepareStarted { .. })),
        "{scenario} did not emit prepare_started"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaEvent::PrepareDecision { .. })),
        "{scenario} did not emit prepare_decision"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaEvent::PrewarmStarted { .. })),
        "{scenario} did not emit prewarm_started"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaEvent::PrewarmCompleted { .. })),
        "{scenario} did not emit prewarm_completed"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaEvent::PrepareCompleted { .. })),
        "{scenario} did not emit prepare_completed"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaEvent::IndexerProgress { .. })),
        "{scenario} did not emit indexer progress"
    );

    for event in events {
        if let MatryoshkaEvent::IndexerProgress { progress, .. } = event {
            match progress {
                MatryoshkaProgressEvent::ParsingFile {
                    index, total_files, ..
                }
                | MatryoshkaProgressEvent::ParsedFile {
                    index, total_files, ..
                }
                | MatryoshkaProgressEvent::EnrichingFile {
                    index, total_files, ..
                }
                | MatryoshkaProgressEvent::EnrichedFile {
                    index, total_files, ..
                } => {
                    assert!(*total_files > 0, "{scenario} progress total was zero");
                    assert!(
                        *index <= *total_files,
                        "{scenario} progress index {index} exceeded total {total_files}"
                    );
                }
                MatryoshkaProgressEvent::EmbeddingBatch {
                    batch_index,
                    total_batches,
                    ..
                }
                | MatryoshkaProgressEvent::EmbeddedBatch {
                    batch_index,
                    total_batches,
                    ..
                } => {
                    assert!(
                        *total_batches > 0,
                        "{scenario} embedding batch total was zero"
                    );
                    assert!(
                        *batch_index <= *total_batches,
                        "{scenario} embedding batch {batch_index} exceeded total {total_batches}"
                    );
                }
                _ => {}
            }
        }
    }
}

fn assert_lock_events_are_consistent(scenario: &str, events: &[MatryoshkaEvent]) {
    let acquired = events
        .iter()
        .position(|event| matches!(event, MatryoshkaEvent::PrepareLockAcquired { .. }))
        .unwrap_or_else(|| panic!("{scenario} did not emit lock acquired"));
    let released = events
        .iter()
        .position(|event| matches!(event, MatryoshkaEvent::PrepareLockReleased { .. }))
        .unwrap_or_else(|| panic!("{scenario} did not emit lock released"));
    assert!(
        acquired < released,
        "{scenario} lock release should happen after acquisition"
    );
}

fn assert_ready_progress_state(db: &Path) {
    let state_path = progress_state_path(db);
    assert!(state_path.exists());
    let value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(state_path).unwrap()).unwrap();
    assert_eq!(value["status"], "ready");
    assert_eq!(value["percent"], 1.0);
}

fn conn(db: &Path) -> Connection {
    Connection::open(db).unwrap()
}

fn count_file_cards(db: &Path, file_id: &str) -> i64 {
    conn(db)
        .query_row(
            "select count(*) from file_cards where file_id = ?1",
            [file_id],
            |row| row.get(0),
        )
        .unwrap()
}

fn count_semantic_records(db: &Path, path: &str) -> i64 {
    conn(db)
        .query_row(
            "select count(*) from semantic_records where path = ?1",
            [path],
            |row| row.get(0),
        )
        .unwrap()
}

fn count_late_vectors_for_path(db: &Path, path: &str) -> i64 {
    conn(db)
        .query_row(
            "select count(*) from semantic_late_vectors where record_id in (select record_id from semantic_records where path = ?1)",
            [path],
            |row| row.get(0),
        )
        .unwrap()
}

fn fts_match_count(db: &Path, query: &str) -> i64 {
    conn(db)
        .query_row(
            "select count(*) from semantic_records_fts where semantic_records_fts match ?1",
            [query],
            |row| row.get(0),
        )
        .unwrap()
}

fn blank_two_card_summaries(db: &Path) {
    let conn = conn(db);
    conn.execute(
        "update file_cards set payload_json = json_set(payload_json, '$.summary', '') where file_id in ('watcher/src/poller.rs', 'search/src/semantic_search.rs')",
        [],
    )
    .unwrap();
}

fn delete_search_data(db: &Path) {
    let conn = conn(db);
    conn.execute("delete from semantic_late_vectors", [])
        .unwrap();
    conn.execute("delete from semantic_records_fts", [])
        .unwrap();
    conn.execute("delete from semantic_records", []).unwrap();
}

fn seed_orphaned_cards(db: &Path) {
    let conn = conn(db);
    conn.execute(
        r#"
        insert or replace into file_cards(file_id, source_hash, payload_json)
        select 'ignored/src/stale.rs',
               'stale',
               json_set(payload_json, '$.file_id', 'ignored/src/stale.rs', '$.summary', '')
        from file_cards
        limit 1
        "#,
        [],
    )
    .unwrap();
    conn.execute(
        r#"
        insert or replace into folder_cards(folder_id, payload_json)
        select 'ignored/src',
               json_set(payload_json, '$.folder_id', 'ignored/src', '$.summary', '')
        from folder_cards
        limit 1
        "#,
        [],
    )
    .unwrap();
}

fn seed_orphaned_semantic_artifacts(db: &Path) {
    let conn = conn(db);
    conn.execute(
        r#"
        insert or replace into semantic_records(record_id, entity_id, entity_type, path, source_hash, payload_json)
        values(
          'semantic:file_card:ignored/src/stale.rs',
          'ignored/src/stale.rs',
          'File',
          'ignored/src/stale.rs',
          'stale',
          '{"record_id":"semantic:file_card:ignored/src/stale.rs","entity_id":"ignored/src/stale.rs","entity_type":"file","title":"stale","content":"stale","path":"ignored/src/stale.rs","source_hash":"stale","embedding":[0.1],"metadata":{}}'
        )
        "#,
        [],
    )
    .unwrap();
    conn.execute(
        r#"
        insert into semantic_records_fts(record_id, title, path, content, metadata_text)
        values('semantic:missing:orphan', 'orphan', 'ignored/src/stale.rs', 'orphan stale', '')
        "#,
        [],
    )
    .unwrap();
    conn.execute(
        r#"
        insert or replace into semantic_late_vectors(record_id, token, ordinal, weight, embedding_json)
        values('semantic:missing:orphan', 'orphan', 0, 1.0, '[0.1]')
        "#,
        [],
    )
    .unwrap();
}

fn orphan_file_cards(db: &Path) -> i64 {
    conn(db)
        .query_row(
            "select count(*) from file_cards where file_id not in (select file_id from files)",
            [],
            |row| row.get(0),
        )
        .unwrap()
}

fn orphan_folder_cards(db: &Path) -> i64 {
    conn(db)
        .query_row(
            "select count(*) from folder_cards where folder_id not in (select folder_id from folders)",
            [],
            |row| row.get(0),
        )
        .unwrap()
}

fn orphan_fts_records(db: &Path) -> i64 {
    conn(db)
        .query_row(
            "select count(*) from semantic_records_fts where record_id not in (select record_id from semantic_records)",
            [],
            |row| row.get(0),
        )
        .unwrap()
}

fn orphan_late_vectors(db: &Path) -> i64 {
    conn(db)
        .query_row(
            "select count(*) from semantic_late_vectors where record_id not in (select record_id from semantic_records)",
            [],
            |row| row.get(0),
        )
        .unwrap()
}

fn base_url() -> String {
    std::env::var("MATRYOSHKA_MLX_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into())
}

fn api_key() -> String {
    std::env::var("MATRYOSHKA_MLX_API_KEY").unwrap_or_else(|_| DEFAULT_API_KEY.into())
}

fn chat_model() -> String {
    std::env::var("MATRYOSHKA_MLX_CHAT_MODEL").unwrap_or_else(|_| DEFAULT_CHAT_MODEL.into())
}

fn embed_model() -> String {
    std::env::var("MATRYOSHKA_MLX_EMBED_MODEL").unwrap_or_else(|_| DEFAULT_EMBED_MODEL.into())
}

fn rerank_model() -> String {
    std::env::var("MATRYOSHKA_OMLX_RERANK_MODEL").unwrap_or_else(|_| DEFAULT_RERANK_MODEL.into())
}
