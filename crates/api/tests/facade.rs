use matryoshka::{
    CardsOptions, Matryoshka, MatryoshkaCancelToken, MatryoshkaConfig, MatryoshkaEvent,
    PrepareOptions, PrepareStatus, ReadBundleOptions, SearchOptions, artifact_gap_count,
    is_cancelled_error, progress_state_path, ready_marker_path,
};
use matryoshka_core_ir::MatryoshkaProgressEvent;
use matryoshka_read_api::ReadApi;
use matryoshka_store_sqlite::MatryoshkaStore;
use rusqlite::Connection;
use serde_json::json;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;
use tempfile::TempDir;

#[test]
fn prepare_search_read_and_repair_lifecycle_work_through_rust_api() {
    let fixture = Fixture::new();
    let repo = fixture.repo();
    let db = repo.join(".matryoshka/matryoshka-api-test.db");
    let api = fixture.api(&db);

    let first_events = run_prepare(&api);
    let first = completed_summary(&first_events);
    assert_eq!(first.status, PrepareStatus::Ready);
    assert_eq!(first.actions_taken, vec!["index", "prepare_results"]);
    assert!(first.file_count >= 3);
    assert!(first.symbol_count > 0);
    assert_eq!(artifact_gap_count(&first.artifact_quality), 0);
    assert!(first.retrieval_index.semantic_records > 0);
    assert!(first.retrieval_index.fts_records > 0);
    assert!(first.retrieval_index.records_with_late_vectors > 0);
    assert!(first.prewarm.warmed_hit_count > 0);
    assert!(ready_marker_path(&db).exists());
    assert_ready_progress_state(&db);

    let hits = api
        .search(
            "watcher debounce changed removed paths",
            SearchOptions::default(),
        )
        .unwrap();
    assert!(!hits.is_empty());
    assert!(hits.iter().any(|hit| hit.path == "src/watcher.rs"));

    let read = api.read("src/watcher.rs").unwrap();
    assert_eq!(read.file.path, "src/watcher.rs");
    assert!(
        read.symbols
            .iter()
            .any(|symbol| symbol.name == "debounce_window")
    );
    let read_with_chunks = ReadApi::new(MatryoshkaStore::open(&db).unwrap(), &repo)
        .read_with_chunks("src/watcher.rs")
        .unwrap();
    assert!(read_with_chunks.symbols.is_empty());
    assert!(!read_with_chunks.chunks.is_empty());
    assert!(read_with_chunks.chunks.iter().any(|chunk| {
        chunk
            .symbol
            .as_deref()
            .is_some_and(|symbol| symbol.contains("debounce_window"))
    }));

    let bundle = api
        .read_bundle(ReadBundleOptions::new("watcher debounce flow"))
        .unwrap();
    assert!(!bundle.primary.file.path.is_empty());

    let cards = api.cards(CardsOptions { empty_only: false }).unwrap();
    assert!(!cards.is_empty());
    assert!(
        api.cards(CardsOptions { empty_only: true })
            .unwrap()
            .is_empty()
    );

    fs::remove_file(ready_marker_path(&db)).unwrap();
    let marker_events = run_prepare(&api);
    let marker = completed_summary(&marker_events);
    assert_eq!(marker.status, PrepareStatus::Ready);
    assert_eq!(marker.actions_taken, vec!["update", "prepare_results"]);
    assert!(ready_marker_path(&db).exists());

    fs::write(
        repo.join("src/probe.rs"),
        "pub fn matryoshka_probe_added_symbol() -> &'static str { \"added\" }\n",
    )
    .unwrap();
    let added = completed_summary(&run_prepare(&api));
    assert_eq!(added.status, PrepareStatus::Ready);
    assert!(added.changed_files >= 1);
    assert_eq!(count_file_cards(&db, "src/probe.rs"), 1);
    assert!(count_semantic_records(&db, "src/probe.rs") > 0);
    assert!(count_late_vectors_for_path(&db, "src/probe.rs") > 0);

    fs::write(
        repo.join("src/probe.rs"),
        "pub fn matryoshka_probe_changed_symbol() -> &'static str { \"changed\" }\n",
    )
    .unwrap();
    let changed = completed_summary(&run_prepare(&api));
    assert_eq!(changed.status, PrepareStatus::Ready);
    assert!(changed.changed_files >= 1);
    assert!(fts_match_count(&db, "matryoshka_probe_changed_symbol") > 0);
    assert_eq!(fts_match_count(&db, "matryoshka_probe_added_symbol"), 0);

    fs::remove_file(repo.join("src/probe.rs")).unwrap();
    let deleted = completed_summary(&run_prepare(&api));
    assert_eq!(deleted.status, PrepareStatus::Ready);
    assert!(deleted.removed_files >= 1);
    assert_eq!(count_file_cards(&db, "src/probe.rs"), 0);
    assert_eq!(count_semantic_records(&db, "src/probe.rs"), 0);
    assert_eq!(count_late_vectors_for_path(&db, "src/probe.rs"), 0);

    blank_two_card_summaries(&db);
    let repaired = completed_summary(&run_prepare(&api));
    assert_eq!(repaired.status, PrepareStatus::Ready);
    assert_eq!(repaired.actions_taken, vec!["repair", "prepare_results"]);
    assert_eq!(artifact_gap_count(&repaired.artifact_quality), 0);

    delete_search_data(&db);
    let rebuilt = completed_summary(&run_prepare(&api));
    assert_eq!(rebuilt.status, PrepareStatus::Ready);
    assert_eq!(
        rebuilt.actions_taken,
        vec!["rebuild_search", "prepare_results"]
    );
    assert!(rebuilt.retrieval_index.semantic_records > 0);
    assert!(rebuilt.retrieval_index.fts_records > 0);
    assert!(rebuilt.retrieval_index.records_with_late_vectors > 0);

    let healthy_events = run_prepare(&api);
    let healthy = completed_summary(&healthy_events);
    assert_eq!(healthy.status, PrepareStatus::Ready);
    assert_eq!(healthy.actions_taken, vec!["update", "prepare_results"]);
    assert_eq!(healthy.changed_files, 0);
    assert_eq!(healthy.removed_files, 0);
    assert_progress_events_are_consistent(&healthy_events);
}

#[test]
fn prepare_with_dense_disabled_reaches_ready_without_embeddings() {
    let fixture = Fixture::new();
    let repo = fixture.repo();
    let db = repo.join(".matryoshka/matryoshka-api-dense-off.db");
    let api = Matryoshka::new(
        fixture
            .config(&db)
            .with_dense_enabled(false)
            .with_dense_fallback_enabled(false),
    );

    let events = run_prepare(&api);
    let summary = completed_summary(&events);

    assert_eq!(summary.status, PrepareStatus::Ready);
    assert_eq!(summary.actions_taken, vec!["index", "prepare_results"]);
    assert!(summary.retrieval_index.semantic_records > 0);
    assert!(summary.retrieval_index.fts_records > 0);
    assert_eq!(summary.retrieval_index.embedded_records, 0);
    assert_eq!(summary.retrieval_index.late_vector_rows, 0);
    assert_eq!(summary.retrieval_index.records_with_late_vectors, 0);
    assert!(!summary.retrieval_index.dense_enabled);
    assert!(!summary.retrieval_index.dense_fallback_enabled);
    assert!(!summary.retrieval_index.late_interaction_enabled);
    assert!(summary.prewarm.warmed_hit_count > 0);
    assert!(events.iter().any(|event| matches!(
        event,
        MatryoshkaEvent::IndexerProgress {
            progress: MatryoshkaProgressEvent::EmbeddingSkipped { record_count, reason },
            ..
        } if *record_count > 0 && reason == "dense embeddings disabled"
    )));
    assert!(!events.iter().any(|event| matches!(
        event,
        MatryoshkaEvent::IndexerProgress {
            progress: MatryoshkaProgressEvent::EmbeddingBatch { .. }
                | MatryoshkaProgressEvent::EmbeddedBatch { .. },
            ..
        }
    )));
    assert_eq!(count_all_late_vectors(&db), 0);

    let hits = api
        .search(
            "watcher debounce changed removed paths",
            SearchOptions::default(),
        )
        .unwrap();
    assert!(!hits.is_empty());
    assert!(
        hits.iter().any(|hit| hit.path == "src/watcher.rs"),
        "{hits:?}"
    );
    assert!(hits.iter().all(|hit| {
        !hit.why_matched
            .iter()
            .any(|why| why.contains("Late-interaction MaxSim"))
    }));
    assert_ready_progress_state(&db);
}

#[test]
fn prepare_cancellation_before_start_emits_cancelled_state() {
    let fixture = Fixture::new();
    let repo = fixture.repo();
    let db = repo.join(".matryoshka/matryoshka-api-cancel-before.db");
    let api = fixture.api(&db);
    let cancel_token = MatryoshkaCancelToken::new();
    cancel_token.cancel();

    let mut events = Vec::new();
    let err = api
        .prepare_with_progress_and_cancel(PrepareOptions::default(), cancel_token, |event| {
            events.push(event)
        })
        .unwrap_err();

    assert!(is_cancelled_error(err.as_ref()));
    assert_cancelling_then_cancelled(&events);
    assert_cancelled_progress_state(&db);
    assert!(!ready_marker_path(&db).exists());
}

#[test]
fn prepare_cancellation_during_enrichment_stops_before_ready() {
    let fixture = Fixture::new();
    let repo = fixture.repo();
    let db = repo.join(".matryoshka/matryoshka-api-cancel-during.db");
    let api = fixture.api(&db);
    let cancel_token = MatryoshkaCancelToken::new();
    let cancel_from_callback = cancel_token.clone();

    let mut events = Vec::new();
    let err = api
        .prepare_with_progress_and_cancel(
            PrepareOptions {
                limit: 3,
                queries: vec!["watcher debounce changed removed paths".into()],
                write_progress_state: true,
            },
            cancel_token,
            |event| {
                if matches!(
                    &event,
                    MatryoshkaEvent::IndexerProgress {
                        progress: MatryoshkaProgressEvent::EnrichingFile { .. },
                        ..
                    }
                ) {
                    cancel_from_callback.cancel();
                }
                events.push(event);
            },
        )
        .unwrap_err();

    assert!(is_cancelled_error(err.as_ref()));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaEvent::IndexerProgress { .. }))
    );
    assert_cancelling_then_cancelled(&events);
    assert_cancelled_progress_state(&db);
    assert!(!ready_marker_path(&db).exists());
    let read_err = api.read("src/watcher.rs").unwrap_err();
    let read_err = format!("{read_err:#}");
    assert!(read_err.contains("Matryoshka prepare is not ready"));
    assert!(read_err.contains("cancelled"));

    let retry = completed_summary(&run_prepare(&api));
    assert_eq!(retry.status, PrepareStatus::Ready);
}

#[test]
fn prepare_failure_preserves_mlx_error_text_and_retry_reconciles() {
    let fixture = Fixture::new();
    let repo = fixture.repo();
    let db = repo.join(".matryoshka/matryoshka-api-mlx-error.db");
    let api = fixture.api(&db);
    let marker = "mlx-cache exploded while loading embeddinggemma exact marker";
    fixture.fail_next_chat_requests(9, marker);

    let mut events = Vec::new();
    let err = api
        .prepare_with_progress(
            PrepareOptions {
                limit: 3,
                queries: vec!["watcher debounce changed removed paths".into()],
                write_progress_state: true,
            },
            |event| events.push(event),
        )
        .unwrap_err();
    let err = format!("{err:#}");
    assert!(err.contains(marker), "{err}");
    assert!(!ready_marker_path(&db).exists());
    assert!(events.iter().any(|event| matches!(
        event,
        MatryoshkaEvent::PrepareFailed { message, .. } if message.contains(marker)
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        MatryoshkaEvent::ProgressState { state }
            if state.status == "failed" && state.last_error.as_deref().is_some_and(|error| error.contains(marker))
    )));
    assert_failed_progress_state(&db, marker);

    let search_err = api
        .search(
            "watcher debounce changed removed paths",
            SearchOptions::default(),
        )
        .unwrap_err();
    let search_err = format!("{search_err:#}");
    assert!(search_err.contains("Matryoshka prepare is not ready"));
    assert!(search_err.contains(marker));

    let retry = completed_summary(&run_prepare(&api));
    assert_eq!(retry.status, PrepareStatus::Ready);
    assert!(ready_marker_path(&db).exists());
    assert_ready_progress_state(&db);
}

#[test]
fn stale_running_prepare_state_tells_reads_to_resume_prepare() {
    let fixture = Fixture::new();
    let repo = fixture.repo();
    let db = repo.join(".matryoshka/matryoshka-api-stale-running.db");
    let api = fixture.api(&db);
    let state_path = progress_state_path(&db);
    fs::create_dir_all(state_path.parent().unwrap()).unwrap();
    fs::write(
        &state_path,
        serde_json::to_string_pretty(&json!({
            "operation": "prepare",
            "status": "running",
            "phase": "enriching_chunks",
            "message": "Understanding code",
            "percent": 0.76,
            "updated_at_unix_ms": 1
        }))
        .unwrap(),
    )
    .unwrap();

    let err = api.read("src/watcher.rs").unwrap_err();
    let err = format!("{err:#}");
    assert!(err.contains("Matryoshka prepare is not ready"), "{err}");
    assert!(
        err.contains(
            "previous prepare is still running or was interrupted at phase enriching_chunks"
        ),
        "{err}"
    );
    assert!(err.contains("run prepare again to resume"), "{err}");
}

#[test]
fn prepare_prunes_orphaned_artifacts_before_health_checks() {
    let fixture = Fixture::new();
    let repo = fixture.repo();
    let db = repo.join(".matryoshka/matryoshka-api-orphans.db");
    let api = fixture.api(&db);

    let first = completed_summary(&run_prepare(&api));
    assert_eq!(first.status, PrepareStatus::Ready);

    seed_orphaned_cards(&db);
    seed_orphaned_semantic_artifacts(&db);
    assert!(orphan_file_cards(&db) > 0);
    assert!(orphan_folder_cards(&db) > 0);
    assert!(orphan_fts_records(&db) > 0);
    assert!(orphan_late_vectors(&db) > 0);

    let repaired = completed_summary(&run_prepare(&api));
    assert_eq!(repaired.status, PrepareStatus::Ready);
    assert_eq!(artifact_gap_count(&repaired.artifact_quality), 0);
    assert_eq!(orphan_file_cards(&db), 0);
    assert_eq!(orphan_folder_cards(&db), 0);
    assert_eq!(orphan_fts_records(&db), 0);
    assert_eq!(orphan_late_vectors(&db), 0);
    assert!(
        api.cards(CardsOptions { empty_only: true })
            .unwrap()
            .is_empty()
    );
}

#[test]
fn prepare_calls_for_same_db_serialize_without_sqlite_lock_errors() {
    let fixture = Fixture::new();
    let repo = fixture.repo();
    let db = repo.join(".matryoshka/matryoshka-api-lock-same.db");
    let api = Arc::new(fixture.api(&db));
    let barrier = Arc::new(Barrier::new(2));

    let left = spawn_prepare(api.clone(), barrier.clone());
    let right = spawn_prepare(api, barrier);

    let left_events = left.join().unwrap();
    let right_events = right.join().unwrap();
    assert_eq!(completed_summary(&left_events).status, PrepareStatus::Ready);
    assert_eq!(
        completed_summary(&right_events).status,
        PrepareStatus::Ready
    );
    assert_lock_events_are_consistent(&left_events);
    assert_lock_events_are_consistent(&right_events);
}

#[test]
fn prepare_calls_for_different_db_paths_run_independently() {
    let fixture = Fixture::new();
    let repo = fixture.repo();
    let left_db = repo.join(".matryoshka/matryoshka-api-lock-left.db");
    let right_db = repo.join(".matryoshka/matryoshka-api-lock-right.db");
    let left_api = Arc::new(fixture.api(&left_db));
    let right_api = Arc::new(fixture.api(&right_db));
    let barrier = Arc::new(Barrier::new(2));

    let left = spawn_prepare(left_api, barrier.clone());
    let right = spawn_prepare(right_api, barrier);

    let left_events = left.join().unwrap();
    let right_events = right.join().unwrap();
    assert_eq!(completed_summary(&left_events).status, PrepareStatus::Ready);
    assert_eq!(
        completed_summary(&right_events).status,
        PrepareStatus::Ready
    );
    assert_lock_events_are_consistent(&left_events);
    assert_lock_events_are_consistent(&right_events);
}

struct Fixture {
    _tmp: TempDir,
    repo: PathBuf,
    mlx: FakeMlxServer,
}

impl Fixture {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let mlx = FakeMlxServer::new();
        let repo = tmp.path().join("repo");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("src/lib.rs"),
            r#"
pub mod search;
pub mod watcher;

pub struct RepoWatcher {
    pub interval_ms: u64,
}

impl RepoWatcher {
    pub fn poll_once(&self) -> bool {
        self.interval_ms > 0
    }
}
"#,
        )
        .unwrap();
        fs::write(
            repo.join("src/watcher.rs"),
            r#"
pub fn debounce_window(changed_paths: &[String], removed_paths: &[String]) -> usize {
    changed_paths.len() + removed_paths.len()
}

pub fn update_after_change(path: &str) -> String {
    format!("changed:{path}")
}
"#,
        )
        .unwrap();
        fs::write(
            repo.join("src/search.rs"),
            r#"
pub fn semantic_search(query: &str) -> Vec<String> {
    vec![format!("hit:{query}")]
}

pub fn read_next(file: &str) -> String {
    format!("read:{file}")
}
"#,
        )
        .unwrap();
        Self {
            _tmp: tmp,
            repo,
            mlx,
        }
    }

    fn repo(&self) -> PathBuf {
        self.repo.clone()
    }

    fn config(&self, db: &Path) -> MatryoshkaConfig {
        MatryoshkaConfig::new(&self.repo)
            .with_db(db)
            .with_ignored_paths([".matryoshka", "target"])
            .with_endpoint(self.mlx.base_url(), "2508")
            .with_models("fake-chat-model", "fake-embedding-model")
            .with_chunk_summary_model("fake-chunk-summary-model")
            .with_chunk_summary_concurrency(2)
    }

    fn api(&self, db: &Path) -> Matryoshka {
        Matryoshka::new(self.config(db))
    }

    fn fail_next_chat_requests(&self, count: usize, message: &str) {
        self.mlx.fail_next_chat_requests(count, message);
    }
}

struct FakeMlxServer {
    base_url: String,
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    state: Arc<FakeMlxState>,
    handle: Option<thread::JoinHandle<()>>,
}

#[derive(Default)]
struct FakeMlxState {
    chat_failures: Mutex<Option<(usize, String)>>,
}

impl FakeMlxServer {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let state = Arc::new(FakeMlxState::default());
        let server_stop = stop.clone();
        let server_state = state.clone();
        let handle = thread::spawn(move || {
            for stream in listener.incoming() {
                if server_stop.load(Ordering::SeqCst) {
                    break;
                }
                let Ok(stream) = stream else {
                    continue;
                };
                let state = server_state.clone();
                thread::spawn(move || handle_fake_mlx_connection(stream, state));
            }
        });

        Self {
            base_url: format!("http://{addr}"),
            addr,
            stop,
            state,
            handle: Some(handle),
        }
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn fail_next_chat_requests(&self, count: usize, message: &str) {
        *self.state.chat_failures.lock().unwrap() = Some((count, message.to_string()));
    }
}

impl Drop for FakeMlxServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect_timeout(&self.addr, Duration::from_millis(100));
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn handle_fake_mlx_connection(mut stream: TcpStream, state: Arc<FakeMlxState>) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let request = match read_http_request(&stream) {
        Ok(request) => request,
        Err(err) => {
            write_json_response(
                &mut stream,
                400,
                json!({ "error": { "message": format!("bad request: {err}"), "type": "bad_request" } }),
            );
            return;
        }
    };

    match (request.method.as_str(), request.path.as_str()) {
        ("POST", "/v1/chat/completions") => respond_chat(&mut stream, &state),
        ("POST", "/v1/embeddings") => respond_embeddings(&mut stream, &request.body),
        _ => write_json_response(
            &mut stream,
            404,
            json!({ "error": { "message": "unknown fake mlx route", "type": "not_found" } }),
        ),
    }
}

fn read_http_request(stream: &TcpStream) -> std::io::Result<HttpRequest> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut first_line = String::new();
    reader.read_line(&mut first_line)?;
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().unwrap_or(0);
            }
        }
    }

    let mut body = vec![0; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok(HttpRequest { method, path, body })
}

fn respond_chat(stream: &mut TcpStream, state: &FakeMlxState) {
    if let Some(message) = next_chat_failure(state) {
        write_json_response(
            stream,
            500,
            json!({ "error": { "message": message, "type": "server_error" } }),
        );
        return;
    }

    let content = json!({
        "summary": "Fake MLX prepared summary covering watcher debounce, semantic search, reads, symbols, chunks, changed paths, and removed paths."
    })
    .to_string();
    write_json_response(
        stream,
        200,
        json!({
            "choices": [{
                "message": { "content": content },
                "finish_reason": "stop"
            }]
        }),
    );
}

fn next_chat_failure(state: &FakeMlxState) -> Option<String> {
    let mut failures = state.chat_failures.lock().unwrap();
    let (remaining, message) = failures.as_mut()?;
    if *remaining == 0 {
        *failures = None;
        return None;
    }
    *remaining -= 1;
    let message = message.clone();
    if *remaining == 0 {
        *failures = None;
    }
    Some(message)
}

fn respond_embeddings(stream: &mut TcpStream, body: &[u8]) {
    let input = serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|value| value.get("input").cloned())
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default();
    let data = input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let text = value.as_str().unwrap_or_default();
            json!({
                "index": index,
                "embedding": fake_embedding(text),
            })
        })
        .collect::<Vec<_>>();
    write_json_response(stream, 200, json!({ "data": data }));
}

fn fake_embedding(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0; 32];
    for token in text.split(|ch: char| !ch.is_alphanumeric() && ch != '_') {
        if token.is_empty() {
            continue;
        }
        let mut hash = 0usize;
        for byte in token.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as usize);
        }
        vector[hash % 32] += 1.0;
    }
    if vector.iter().all(|value| *value == 0.0) {
        vector[0] = 1.0;
    }
    vector
}

fn write_json_response(stream: &mut TcpStream, status: u16, body: serde_json::Value) {
    let body = body.to_string();
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Unknown",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
}

fn run_prepare(api: &Matryoshka) -> Vec<MatryoshkaEvent> {
    let mut events = Vec::new();
    let summary = api
        .prepare_with_progress(
            PrepareOptions {
                limit: 3,
                queries: vec![
                    "watcher debounce changed removed paths".into(),
                    "semantic search read next".into(),
                ],
                write_progress_state: true,
            },
            |event| events.push(event),
        )
        .unwrap();
    assert!(
        summary.is_ready(),
        "prepare summary was not ready: {:#?}",
        summary
    );
    assert_progress_events_are_consistent(&events);
    assert_lock_events_are_consistent(&events);
    events
}

fn spawn_prepare(
    api: Arc<Matryoshka>,
    barrier: Arc<Barrier>,
) -> thread::JoinHandle<Vec<MatryoshkaEvent>> {
    thread::spawn(move || {
        barrier.wait();
        run_prepare(&api)
    })
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

fn assert_progress_events_are_consistent(events: &[MatryoshkaEvent]) {
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaEvent::PrepareStarted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaEvent::PrepareDecision { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaEvent::PrewarmStarted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaEvent::PrewarmCompleted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaEvent::PrepareCompleted { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaEvent::IndexerProgress { .. }))
    );
    assert!(events.iter().any(|event| matches!(
        event,
        MatryoshkaEvent::ProgressState { state }
            if state.operation == "prepare" && state.message == "Getting ready"
    )));

    let emitted_file_enrichment = events.iter().any(|event| {
        matches!(
            event,
            MatryoshkaEvent::IndexerProgress {
                progress: MatryoshkaProgressEvent::EnrichingFile { .. }
                    | MatryoshkaProgressEvent::EnrichedFile { .. },
                ..
            }
        )
    });
    if emitted_file_enrichment {
        assert!(events.iter().any(|event| matches!(
            event,
            MatryoshkaEvent::ProgressState { state }
                if state.operation == "prepare" && state.phase == "enriching_files"
        )));
    }

    let emitted_chunk_enrichment = events.iter().any(|event| {
        matches!(
            event,
            MatryoshkaEvent::IndexerProgress {
                progress: MatryoshkaProgressEvent::EnrichingChunks { .. }
                    | MatryoshkaProgressEvent::EnrichingChunkBatch { .. }
                    | MatryoshkaProgressEvent::EnrichedChunkBatch { .. }
                    | MatryoshkaProgressEvent::EnrichedChunks { .. },
                ..
            }
        )
    });
    if emitted_chunk_enrichment {
        assert!(events.iter().any(|event| matches!(
            event,
            MatryoshkaEvent::ProgressState { state }
                if state.operation == "prepare"
                    && state.phase == "enriching_chunks"
                    && state.files_done.is_none()
                    && state.files_total.is_none()
                    && (state.item_label.as_deref() == Some("chunks")
                        || state.item_label.as_deref() == Some("batches"))
        )));
    }

    let emitted_search_progress = events.iter().any(|event| {
        matches!(
            event,
            MatryoshkaEvent::IndexerProgress {
                progress: MatryoshkaProgressEvent::EmbeddingBatch { .. }
                    | MatryoshkaProgressEvent::EmbeddedBatch { .. }
                    | MatryoshkaProgressEvent::EmbeddingSkipped { .. },
                ..
            }
        )
    });
    if emitted_search_progress {
        assert!(events.iter().any(|event| matches!(
            event,
            MatryoshkaEvent::ProgressState { state }
                if state.operation == "prepare"
                    && state.files_done.is_none()
                    && state.files_total.is_none()
                    && ((state.phase == "embedding"
                        && state.item_label.as_deref() == Some("batches"))
                        || (state.phase == "embedding_skipped"
                            && state.item_label.as_deref() == Some("records")))
        )));
    }

    for event in events {
        if let MatryoshkaEvent::IndexerProgress {
            operation,
            action,
            progress,
        } = event
        {
            assert_eq!(operation, "prepare");
            assert!(action.is_some());
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
                    assert!(*total_files > 0);
                    assert!(*index <= *total_files);
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
                    assert!(*total_batches > 0);
                    assert!(*batch_index <= *total_batches);
                }
                _ => {}
            }
        }
    }
}

fn assert_lock_events_are_consistent(events: &[MatryoshkaEvent]) {
    let acquired = events
        .iter()
        .position(|event| matches!(event, MatryoshkaEvent::PrepareLockAcquired { .. }))
        .expect("prepare should emit lock acquired");
    let released = events
        .iter()
        .position(|event| matches!(event, MatryoshkaEvent::PrepareLockReleased { .. }))
        .expect("prepare should emit lock released");
    assert!(
        acquired < released,
        "lock acquired event should be emitted before lock released"
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

fn assert_cancelled_progress_state(db: &Path) {
    let state_path = progress_state_path(db);
    assert!(state_path.exists());
    let value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(state_path).unwrap()).unwrap();
    assert_eq!(value["status"], "cancelled");
    assert_eq!(value["phase"], "cancelled");
}

fn assert_failed_progress_state(db: &Path, expected_error: &str) {
    let state_path = progress_state_path(db);
    assert!(state_path.exists());
    let value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(state_path).unwrap()).unwrap();
    assert_eq!(value["status"], "failed");
    assert!(
        value["last_error"]
            .as_str()
            .is_some_and(|error| error.contains(expected_error)),
        "{value:#}"
    );
    assert!(
        value["error_stage"]
            .as_str()
            .is_some_and(|stage| !stage.trim().is_empty()),
        "{value:#}"
    );
}

fn assert_cancelling_then_cancelled(events: &[MatryoshkaEvent]) {
    let cancelling = events
        .iter()
        .position(|event| matches!(event, MatryoshkaEvent::PrepareCancelling { .. }))
        .expect("prepare should emit cancelling event");
    let cancelled = events
        .iter()
        .position(|event| matches!(event, MatryoshkaEvent::PrepareCancelled { .. }))
        .expect("prepare should emit cancelled event");
    assert!(
        cancelling < cancelled,
        "cancelling event should be emitted before cancelled event"
    );
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

fn count_all_late_vectors(db: &Path) -> i64 {
    conn(db)
        .query_row("select count(*) from semantic_late_vectors", [], |row| {
            row.get(0)
        })
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
        "update file_cards set payload_json = json_set(payload_json, '$.summary', '') where file_id in ('src/watcher.rs', 'src/search.rs')",
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
