use anyhow::{Result, anyhow};
use matryoshka_core_ir::{
    ArtifactQualityReport, FileCard, FileEnrichmentContext, FileFact, FolderCard,
    FolderEnrichmentContext, FolderFact, MatryoshkaProgressEvent, ReadCard, RepoCard,
    RetrievalConfig, RetrievalIndexReport, RetrievalPrimary, SearchHit, SymbolFact,
};
use matryoshka_embed_client::Embedder;
use matryoshka_embed_client::{DeterministicEmbedder, EndpointEmbedder};
use matryoshka_enricher::{
    CodeEnricher, HeuristicChunkSummarizer, HeuristicEnricher, MlxChatEnricher, MlxChunkSummarizer,
};
use matryoshka_indexer::{FullIndexer, SemanticRebuildSummary, UpdateSummary};
use matryoshka_parser::ParserConfig;
use matryoshka_read_api::{ReadApi, ReadBundle, ReadPackMode};
use matryoshka_search::{
    EndpointReranker, OmlxReranker, SearchEngine, SearchPrewarmSummary, SearchResultGranularity,
    default_prewarm_queries,
};
use matryoshka_store_sqlite::{
    CardSummaryRow, MatryoshkaStore, OrphanPruneReport, RetrievalIndexStats,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:44445";
pub const DEFAULT_API_KEY: &str = "2508";
pub const DEFAULT_EMBED_MODEL: &str = "mlx-community--embeddinggemma-300m-bf16";
pub const DEFAULT_CHAT_MODEL: &str = "MercuriusDream--Qwen3.5-4B-MLX-mxfp8";
pub const DEFAULT_OMLX_RERANK_MODEL: &str = "mlx-community--Qwen3-Reranker-0.6B-mxfp8";
pub const DEFAULT_CHUNK_SUMMARY_MODEL: &str = "srswti--bodega-raptor-90m";
pub const DEFAULT_CHUNK_SUMMARY_CONCURRENCY: usize = 6;
pub const MATRYOSHKA_DIR: &str = ".matryoshka";
pub const DEFAULT_DB_FILE: &str = "matryoshka.db";
pub const READY_MARKER_FILE: &str = ".jesco-prewarm-complete";

#[derive(Debug, Clone)]
pub struct Matryoshka {
    config: MatryoshkaConfig,
}

#[derive(Debug, Clone, Default)]
pub struct MatryoshkaCancelToken {
    cancelled: Arc<AtomicBool>,
}

impl MatryoshkaCancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(cancelled_error())
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone)]
pub struct MatryoshkaConfig {
    pub repo_root: PathBuf,
    pub db: PathBuf,
    pub offline: bool,
    pub base_url: String,
    pub api_key: String,
    pub embedding_model: String,
    pub chat_model: String,
    pub ignore: Vec<String>,
    pub late_interaction: bool,
    pub retrieval_primary: RetrievalPrimary,
    pub dense_enabled: bool,
    pub dense_fallback_enabled: bool,
    pub chunk_summary_enabled: bool,
    pub chunk_summary_model: String,
    pub chunk_summary_concurrency: usize,
}

impl MatryoshkaConfig {
    pub fn new(repo_root: impl Into<PathBuf>) -> Self {
        let repo_root = repo_root.into();
        let db = default_db_path(&repo_root);
        Self {
            repo_root,
            db,
            offline: false,
            base_url: DEFAULT_BASE_URL.into(),
            api_key: DEFAULT_API_KEY.into(),
            embedding_model: DEFAULT_EMBED_MODEL.into(),
            chat_model: DEFAULT_CHAT_MODEL.into(),
            ignore: Vec::new(),
            late_interaction: true,
            retrieval_primary: RetrievalPrimary::Hybrid,
            dense_enabled: true,
            dense_fallback_enabled: true,
            chunk_summary_enabled: true,
            chunk_summary_model: DEFAULT_CHUNK_SUMMARY_MODEL.into(),
            chunk_summary_concurrency: DEFAULT_CHUNK_SUMMARY_CONCURRENCY,
        }
    }

    pub fn with_db(mut self, db: impl Into<PathBuf>) -> Self {
        self.db = db.into();
        self
    }

    pub fn offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    pub fn with_endpoint(
        mut self,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        self.base_url = base_url.into();
        self.api_key = api_key.into();
        self
    }

    pub fn with_models(
        mut self,
        chat_model: impl Into<String>,
        embedding_model: impl Into<String>,
    ) -> Self {
        self.chat_model = chat_model.into();
        self.embedding_model = embedding_model.into();
        self
    }

    pub fn with_ignored_paths(
        mut self,
        ignore: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.ignore = ignore.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_late_interaction(mut self, enabled: bool) -> Self {
        self.late_interaction = enabled;
        self
    }

    pub fn with_retrieval_primary(mut self, primary: RetrievalPrimary) -> Self {
        self.retrieval_primary = primary;
        self
    }

    pub fn with_dense_enabled(mut self, enabled: bool) -> Self {
        self.dense_enabled = enabled;
        if !enabled {
            self.dense_fallback_enabled = false;
        }
        self
    }

    pub fn with_dense_fallback_enabled(mut self, enabled: bool) -> Self {
        self.dense_fallback_enabled = enabled;
        if enabled {
            self.dense_enabled = true;
        }
        self
    }

    pub fn with_retrieval_config(mut self, config: RetrievalConfig) -> Self {
        self.retrieval_primary = config.primary;
        self.dense_enabled = config.dense_enabled;
        self.dense_fallback_enabled = config.dense_fallback_enabled;
        self
    }

    pub fn retrieval_config(&self) -> RetrievalConfig {
        RetrievalConfig {
            primary: self.retrieval_primary,
            dense_enabled: self.dense_enabled,
            dense_fallback_enabled: self.dense_fallback_enabled,
        }
    }

    pub fn with_chunk_summary_enabled(mut self, enabled: bool) -> Self {
        self.chunk_summary_enabled = enabled;
        self
    }

    pub fn with_chunk_summary_model(mut self, model: impl Into<String>) -> Self {
        self.chunk_summary_model = model.into();
        self
    }

    pub fn with_chunk_summary_concurrency(mut self, concurrency: usize) -> Self {
        self.chunk_summary_concurrency = concurrency.max(1);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PrepareOptions {
    pub limit: usize,
    pub queries: Vec<String>,
    pub write_progress_state: bool,
}

impl Default for PrepareOptions {
    fn default() -> Self {
        Self {
            limit: 8,
            queries: Vec::new(),
            write_progress_state: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchOptions {
    pub limit: usize,
    pub reranker: RerankerOptions,
    #[serde(default)]
    pub result_granularity: SearchResultGranularity,
}

impl SearchOptions {
    pub fn with_result_granularity(mut self, granularity: SearchResultGranularity) -> Self {
        self.result_granularity = granularity;
        self
    }
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 8,
            reranker: RerankerOptions::None,
            result_granularity: SearchResultGranularity::File,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RerankerOptions {
    None,
    Chat { model: String },
    Omlx { model: String, candidates: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReadBundleOptions {
    pub query: String,
    pub limit: usize,
    pub related: usize,
    pub mode: ReadPackMode,
    pub search: SearchOptions,
}

impl ReadBundleOptions {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            limit: 4,
            related: 3,
            mode: ReadPackMode::Brief,
            search: SearchOptions::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CardsOptions {
    pub empty_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PrepareSummary {
    pub repo_root: PathBuf,
    pub db: PathBuf,
    pub ready_marker: PathBuf,
    pub logs_dir: PathBuf,
    pub status: PrepareStatus,
    pub actions_taken: Vec<String>,
    pub file_count: usize,
    pub folder_count: usize,
    pub symbol_count: usize,
    pub semantic_record_count: usize,
    pub changed_files: usize,
    pub removed_files: usize,
    pub changed_folders: usize,
    pub repo_card_updated: bool,
    pub artifact_quality: ArtifactQualityReport,
    pub retrieval_index: RetrievalIndexReport,
    pub prewarm: SearchPrewarmSummaryJson,
    pub embedding_model: String,
}

impl PrepareSummary {
    pub fn is_ready(&self) -> bool {
        self.status == PrepareStatus::Ready
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PrepareStatus {
    Ready,
    NeedsAttention,
}

impl PrepareStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::NeedsAttention => "needs_attention",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchPrewarmSummaryJson {
    pub fts_record_count: usize,
    pub query_count: usize,
    pub warmed_hit_count: usize,
}

impl From<SearchPrewarmSummary> for SearchPrewarmSummaryJson {
    fn from(value: SearchPrewarmSummary) -> Self {
        Self {
            fts_record_count: value.fts_record_count,
            query_count: value.query_count,
            warmed_hit_count: value.warmed_hit_count,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum MatryoshkaEvent {
    PrepareStarted {
        repo_root: PathBuf,
        db: PathBuf,
        existing_file_count: usize,
        existing_missing_text: usize,
        existing_search_missing: bool,
        ready_marker_exists: bool,
    },
    PrepareWaitingForLock {
        db: PathBuf,
        lock_path: PathBuf,
        waited_ms: u128,
    },
    PrepareLockAcquired {
        db: PathBuf,
        lock_path: PathBuf,
    },
    PrepareLockReleased {
        db: PathBuf,
        lock_path: PathBuf,
    },
    PrepareDecision {
        action: String,
        reason: String,
    },
    IndexerProgress {
        operation: String,
        progress: MatryoshkaProgressEvent,
    },
    PrewarmStarted {
        query_count: usize,
        limit: usize,
    },
    PrewarmCompleted {
        summary: SearchPrewarmSummaryJson,
    },
    PrepareCancelling {
        reason: String,
    },
    PrepareCancelled {
        reason: String,
    },
    PrepareCompleted {
        summary: PrepareSummary,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProgressState {
    pub operation: String,
    pub action: Option<String>,
    pub status: String,
    pub phase: String,
    pub message: String,
    pub percent: f32,
    pub current_file: Option<String>,
    pub files_done: Option<usize>,
    pub files_total: Option<usize>,
    pub items_done: Option<usize>,
    pub items_total: Option<usize>,
    pub item_label: Option<String>,
    pub updated_at_unix_ms: u128,
}

impl ProgressState {
    fn new(
        operation: &str,
        action: Option<&str>,
        status: &str,
        phase: &str,
        message: &str,
        percent: f32,
    ) -> Self {
        Self {
            operation: operation.into(),
            action: action.map(Into::into),
            status: status.into(),
            phase: phase.into(),
            message: message.into(),
            percent: percent.clamp(0.0, 1.0),
            current_file: None,
            files_done: None,
            files_total: None,
            items_done: None,
            items_total: None,
            item_label: None,
            updated_at_unix_ms: unix_millis(),
        }
    }

    fn with_file_progress(
        mut self,
        current_file: Option<String>,
        files_done: Option<usize>,
        files_total: Option<usize>,
    ) -> Self {
        self.current_file = current_file;
        self.files_done = files_done;
        self.files_total = files_total;
        self
    }

    fn with_item_progress(
        mut self,
        items_done: Option<usize>,
        items_total: Option<usize>,
        item_label: &str,
    ) -> Self {
        self.items_done = items_done;
        self.items_total = items_total;
        self.item_label = Some(item_label.into());
        self
    }
}

struct CancellableEnricher<E> {
    inner: E,
    cancel_token: MatryoshkaCancelToken,
}

impl<E> CancellableEnricher<E> {
    fn new(inner: E, cancel_token: MatryoshkaCancelToken) -> Self {
        Self {
            inner,
            cancel_token,
        }
    }
}

impl<E> CodeEnricher for CancellableEnricher<E>
where
    E: CodeEnricher,
{
    fn enrich_file(
        &self,
        file: &FileFact,
        symbols: &[SymbolFact],
        context: &FileEnrichmentContext,
    ) -> Result<FileCard> {
        self.cancel_token.check()?;
        let card = self.inner.enrich_file(file, symbols, context)?;
        self.cancel_token.check()?;
        Ok(card)
    }

    fn enrich_folder(
        &self,
        folder: &FolderFact,
        child_files: &[FileCard],
        child_folders: &[FolderCard],
        context: &FolderEnrichmentContext,
    ) -> Result<FolderCard> {
        self.cancel_token.check()?;
        let card = self
            .inner
            .enrich_folder(folder, child_files, child_folders, context)?;
        self.cancel_token.check()?;
        Ok(card)
    }

    fn enrich_repo(&self, repo_root: &str, folders: &[FolderCard]) -> Result<RepoCard> {
        self.cancel_token.check()?;
        let card = self.inner.enrich_repo(repo_root, folders)?;
        self.cancel_token.check()?;
        Ok(card)
    }
}

struct CancellableEmbedder<M> {
    inner: M,
    cancel_token: MatryoshkaCancelToken,
}

impl<M> CancellableEmbedder<M> {
    fn new(inner: M, cancel_token: MatryoshkaCancelToken) -> Self {
        Self {
            inner,
            cancel_token,
        }
    }
}

impl<M> Embedder for CancellableEmbedder<M>
where
    M: Embedder,
{
    fn model(&self) -> &str {
        self.inner.model()
    }

    fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        self.cancel_token.check()?;
        let embeddings = self.inner.embed(inputs)?;
        self.cancel_token.check()?;
        Ok(embeddings)
    }
}

impl Matryoshka {
    pub fn new(config: MatryoshkaConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &MatryoshkaConfig {
        &self.config
    }

    pub fn prepare(&self, options: PrepareOptions) -> Result<PrepareSummary> {
        self.prepare_with_progress(options, |_| {})
    }

    pub fn prepare_with_progress(
        &self,
        options: PrepareOptions,
        mut on_event: impl FnMut(MatryoshkaEvent),
    ) -> Result<PrepareSummary> {
        self.prepare_with_progress_and_cancel(options, MatryoshkaCancelToken::new(), |event| {
            on_event(event)
        })
    }

    pub fn prepare_with_progress_and_cancel(
        &self,
        options: PrepareOptions,
        cancel_token: MatryoshkaCancelToken,
        mut on_event: impl FnMut(MatryoshkaEvent),
    ) -> Result<PrepareSummary> {
        ensure_matryoshka_layout(&self.config.db)?;
        let mut progress_writer =
            ProgressStateWriter::new(&self.config.db, options.write_progress_state);
        let mut log = CommandLog::open(&self.config.db, "prepare")?;
        let logs_dir = logs_dir(&self.config.db);
        let ready_marker = ready_marker_path(&self.config.db);
        let parser_config = parser_config(self.config.ignore.clone());

        if cancel_token.is_cancelled() {
            return cancel_prepare(
                &mut on_event,
                &mut progress_writer,
                &mut log,
                "prepare was cancelled before it started",
            );
        }

        let prepare_lock =
            match acquire_prepare_lock(&self.config.db, &cancel_token, &mut on_event, &mut log) {
                Ok(lock) => lock,
                Err(err) if is_cancelled_error(err.as_ref()) => {
                    return cancel_prepare(
                        &mut on_event,
                        &mut progress_writer,
                        &mut log,
                        "prepare was cancelled while waiting for the project lock",
                    );
                }
                Err(err) => return Err(err),
            };

        let result = (|| -> Result<PrepareSummary> {
            let store = MatryoshkaStore::open(&self.config.db)?;
            let initial_prune = store.prune_orphaned_artifacts()?;
            log_prune_report(&mut log, "prepare_initial_prune", &initial_prune)?;

            let existing_file_count = store.load_all_files()?.len();
            let existing_gap_count = store
                .load_active_card_summaries()?
                .iter()
                .filter(|row| row.is_empty)
                .count();
            let existing_search_missing = retrieval_needs_rebuild(&retrieval_report_from_stats(
                store.retrieval_index_stats()?,
                self.config.retrieval_config(),
                self.config.late_interaction,
            ));
            let ready_marker_exists = ready_marker.exists();
            let mut actions_taken = Vec::new();

            let started = MatryoshkaEvent::PrepareStarted {
                repo_root: self.config.repo_root.clone(),
                db: self.config.db.clone(),
                existing_file_count,
                existing_missing_text: existing_gap_count,
                existing_search_missing,
                ready_marker_exists,
            };
            emit_event(&mut on_event, &mut progress_writer, started.clone());
            log.event(
                "prepare_started",
                json!({
                    "repo_root": self.config.repo_root,
                    "db": self.config.db,
                    "offline": self.config.offline,
                    "embedding_model": if self.config.offline { "deterministic" } else { self.config.embedding_model.as_str() },
                    "chat_model": if self.config.offline { "heuristic" } else { self.config.chat_model.as_str() },
                    "existing_file_count": existing_file_count,
                    "existing_missing_text": existing_gap_count,
                    "existing_search_missing": existing_search_missing,
                    "ready_marker_exists": ready_marker_exists,
                }),
            )?;

            let first_action = if existing_file_count == 0 {
                "index"
            } else if existing_gap_count > 0 {
                "repair"
            } else if existing_search_missing {
                "rebuild_search"
            } else {
                "update"
            };
            let first_reason = if existing_file_count == 0 {
                "no indexed files found"
            } else if existing_gap_count > 0 {
                "project map has gaps"
            } else if existing_search_missing {
                "search data is missing or incomplete"
            } else if !ready_marker_exists {
                "ready marker missing"
            } else {
                "refresh current project map"
            };
            prepare_decision(
                &mut on_event,
                &mut progress_writer,
                &mut log,
                first_action,
                first_reason,
            )?;
            if cancel_token.is_cancelled() {
                return cancel_prepare(
                    &mut on_event,
                    &mut progress_writer,
                    &mut log,
                    "prepare was cancelled before indexing started",
                );
            }

            let mut update = match self.run_update_once_with_progress(
                parser_config.clone(),
                Some(&mut log),
                &cancel_token,
                |progress| {
                    emit_event(
                        &mut on_event,
                        &mut progress_writer,
                        MatryoshkaEvent::IndexerProgress {
                            operation: first_action.to_string(),
                            progress,
                        },
                    );
                },
            ) {
                Ok(summary) => summary,
                Err(_err) if cancel_token.is_cancelled() => {
                    return cancel_prepare(
                        &mut on_event,
                        &mut progress_writer,
                        &mut log,
                        "prepare was cancelled while updating the project",
                    );
                }
                Err(err) => return Err(err),
            };
            actions_taken.push(first_action.to_string());

            if artifact_gap_count(&update.artifact_quality) > 0 && first_action != "repair" {
                if cancel_token.is_cancelled() {
                    return cancel_prepare(
                        &mut on_event,
                        &mut progress_writer,
                        &mut log,
                        "prepare was cancelled before repair",
                    );
                }
                prepare_decision(
                    &mut on_event,
                    &mut progress_writer,
                    &mut log,
                    "repair",
                    "project map has gaps",
                )?;
                update = match self.run_update_once_with_progress(
                    parser_config,
                    Some(&mut log),
                    &cancel_token,
                    |progress| {
                        emit_event(
                            &mut on_event,
                            &mut progress_writer,
                            MatryoshkaEvent::IndexerProgress {
                                operation: "repair".into(),
                                progress,
                            },
                        );
                    },
                ) {
                    Ok(summary) => summary,
                    Err(_err) if cancel_token.is_cancelled() => {
                        return cancel_prepare(
                            &mut on_event,
                            &mut progress_writer,
                            &mut log,
                            "prepare was cancelled while repairing the project",
                        );
                    }
                    Err(err) => return Err(err),
                };
                actions_taken.push("repair".into());
            }

            let mut artifact_quality = update.artifact_quality.clone();
            let mut retrieval_index = update.retrieval_index.clone();
            if retrieval_needs_rebuild(&retrieval_index) {
                if cancel_token.is_cancelled() {
                    return cancel_prepare(
                        &mut on_event,
                        &mut progress_writer,
                        &mut log,
                        "prepare was cancelled before rebuilding search",
                    );
                }
                prepare_decision(
                    &mut on_event,
                    &mut progress_writer,
                    &mut log,
                    "rebuild_search",
                    "search data is missing or incomplete",
                )?;
                let rebuild = match self.run_rebuild_semantic_once_with_progress(
                    Some(&mut log),
                    &cancel_token,
                    |progress| {
                        emit_event(
                            &mut on_event,
                            &mut progress_writer,
                            MatryoshkaEvent::IndexerProgress {
                                operation: "rebuild_search".into(),
                                progress,
                            },
                        );
                    },
                ) {
                    Ok(summary) => summary,
                    Err(_err) if cancel_token.is_cancelled() => {
                        return cancel_prepare(
                            &mut on_event,
                            &mut progress_writer,
                            &mut log,
                            "prepare was cancelled while rebuilding search",
                        );
                    }
                    Err(err) => return Err(err),
                };
                artifact_quality = rebuild.artifact_quality;
                if !actions_taken
                    .iter()
                    .any(|action| action == "rebuild_search")
                {
                    actions_taken.push("rebuild_search".into());
                }
            }

            let queries = if options.queries.is_empty() {
                default_prewarm_queries()
            } else {
                options.queries.clone()
            };
            if cancel_token.is_cancelled() {
                return cancel_prepare(
                    &mut on_event,
                    &mut progress_writer,
                    &mut log,
                    "prepare was cancelled before warming results",
                );
            }
            prepare_decision(
                &mut on_event,
                &mut progress_writer,
                &mut log,
                "prepare_results",
                "make first searches fast and precise",
            )?;
            emit_event(
                &mut on_event,
                &mut progress_writer,
                MatryoshkaEvent::PrewarmStarted {
                    query_count: queries.len(),
                    limit: options.limit,
                },
            );
            let prewarm =
                match self.run_prewarm_once(&queries, options.limit, Some(&mut log), &cancel_token)
                {
                    Ok(summary) => summary,
                    Err(_err) if cancel_token.is_cancelled() => {
                        return cancel_prepare(
                            &mut on_event,
                            &mut progress_writer,
                            &mut log,
                            "prepare was cancelled while warming results",
                        );
                    }
                    Err(err) => return Err(err),
                };
            let prewarm_json = SearchPrewarmSummaryJson::from(prewarm);
            emit_event(
                &mut on_event,
                &mut progress_writer,
                MatryoshkaEvent::PrewarmCompleted {
                    summary: prewarm_json.clone(),
                },
            );
            actions_taken.push("prepare_results".into());

            let final_prune = MatryoshkaStore::open(&self.config.db)?.prune_orphaned_artifacts()?;
            log_prune_report(&mut log, "prepare_final_prune", &final_prune)?;
            retrieval_index = retrieval_report_from_stats(
                MatryoshkaStore::open(&self.config.db)?.retrieval_index_stats()?,
                self.config.retrieval_config(),
                self.config.late_interaction,
            );
            let ready =
                artifact_gap_count(&artifact_quality) == 0 && retrieval_is_ready(&retrieval_index);
            let status = if ready {
                PrepareStatus::Ready
            } else {
                PrepareStatus::NeedsAttention
            };

            let summary = PrepareSummary {
                repo_root: self.config.repo_root.clone(),
                db: self.config.db.clone(),
                ready_marker,
                logs_dir,
                status,
                actions_taken,
                file_count: update.file_count,
                folder_count: update.folder_count,
                symbol_count: update.symbol_count,
                semantic_record_count: retrieval_index.semantic_records,
                changed_files: update.changed_files,
                removed_files: update.removed_files,
                changed_folders: update.changed_folders,
                repo_card_updated: update.repo_card_updated,
                artifact_quality,
                retrieval_index,
                prewarm: prewarm_json,
                embedding_model: update.embedding_model,
            };

            if summary.status == PrepareStatus::Ready {
                write_ready_marker(&summary)?;
            }
            log.event("prepare_completed", prepare_summary_json(&summary))?;
            emit_event(
                &mut on_event,
                &mut progress_writer,
                MatryoshkaEvent::PrepareCompleted {
                    summary: summary.clone(),
                },
            );
            Ok(summary)
        })();

        let release_result =
            release_prepare_lock(prepare_lock, &self.config.db, &mut on_event, &mut log);
        match (result, release_result) {
            (Ok(summary), Ok(())) => Ok(summary),
            (Ok(_), Err(err)) => Err(err),
            (Err(err), _) => Err(err),
        }
    }

    pub fn search(&self, query: &str, options: SearchOptions) -> Result<Vec<SearchHit>> {
        ensure_matryoshka_layout(&self.config.db)?;
        ensure_reranker_options(&options.reranker)?;
        let store = MatryoshkaStore::open(&self.config.db)?;
        let result_granularity = options.result_granularity;
        if self.config.offline {
            let engine = SearchEngine::new(store, DeterministicEmbedder::default())
                .with_dense(self.config.dense_enabled)
                .with_late_interaction(self.config.late_interaction)
                .with_result_granularity(result_granularity);
            search_with_reranker(engine, &self.config, query, options.limit, options.reranker)
        } else {
            let engine = SearchEngine::new(
                store,
                EndpointEmbedder::new(
                    self.config.base_url.clone(),
                    self.config.api_key.clone(),
                    self.config.embedding_model.clone(),
                ),
            )
            .with_dense(self.config.dense_enabled)
            .with_late_interaction(self.config.late_interaction)
            .with_result_granularity(result_granularity);
            search_with_reranker(engine, &self.config, query, options.limit, options.reranker)
        }
    }

    pub fn read(&self, file: &str) -> Result<ReadCard> {
        ensure_matryoshka_layout(&self.config.db)?;
        ReadApi::new(
            MatryoshkaStore::open(&self.config.db)?,
            self.config.repo_root.clone(),
        )
        .read(file)
    }

    pub fn read_bundle(&self, options: ReadBundleOptions) -> Result<ReadBundle> {
        let store = MatryoshkaStore::open(&self.config.db)?;
        let hits = self.search(
            &task_query(AgentTask::ReadNext, &options.query),
            SearchOptions {
                limit: options.limit,
                reranker: options.search.reranker.clone(),
                result_granularity: SearchResultGranularity::File,
            },
        )?;
        let file_ids = hits
            .iter()
            .filter_map(|hit| {
                store
                    .load_file(&hit.path)
                    .ok()
                    .flatten()
                    .map(|file| file.file_id)
            })
            .collect::<Vec<_>>();
        let Some(primary) = file_ids.first() else {
            return Err(anyhow!(
                "no file-level search hit found for read bundle query"
            ));
        };
        let related_file_ids =
            select_related_file_ids(primary, &file_ids[1..], &options.query, options.related);
        ReadApi::new(store, self.config.repo_root.clone()).read_bundle(
            primary,
            &related_file_ids,
            options.mode,
            options.related,
        )
    }

    pub fn cards(&self, options: CardsOptions) -> Result<Vec<CardSummaryRow>> {
        ensure_matryoshka_layout(&self.config.db)?;
        let store = MatryoshkaStore::open(&self.config.db)?;
        store.prune_orphaned_artifacts()?;
        let mut rows = store.load_active_card_summaries()?;
        if options.empty_only {
            rows.retain(|row| row.is_empty);
        }
        Ok(rows)
    }

    fn run_update_once_with_progress(
        &self,
        parser_config: ParserConfig,
        mut log: Option<&mut CommandLog>,
        cancel_token: &MatryoshkaCancelToken,
        mut progress: impl FnMut(MatryoshkaProgressEvent),
    ) -> Result<UpdateSummary> {
        cancel_token.check()?;
        if let Some(log) = log.as_deref_mut() {
            log.event(
                "update_started",
                json!({
                    "repo_root": self.config.repo_root,
                    "db": self.config.db,
                    "offline": self.config.offline,
                    "embedding_model": if self.config.offline { "deterministic" } else { self.config.embedding_model.as_str() },
                }),
            )?;
        }
        let store = MatryoshkaStore::open(&self.config.db)?;
        let summary = if self.config.offline {
            FullIndexer::new(
                store,
                CancellableEnricher::new(HeuristicEnricher, cancel_token.clone()),
                CancellableEmbedder::new(DeterministicEmbedder::default(), cancel_token.clone()),
                HeuristicChunkSummarizer,
            )
            .with_parser_config(parser_config)
            .with_retrieval_config(self.config.retrieval_config())
            .update_repo_with_progress(&self.config.repo_root, &mut progress)?
        } else {
            let enricher = MlxChatEnricher::new(&self.config.base_url, &self.config.api_key)
                .with_model(self.config.chat_model.clone());
            let embedder = EndpointEmbedder::new(
                &self.config.base_url,
                &self.config.api_key,
                self.config.embedding_model.clone(),
            );
            let chunk_summarizer =
                MlxChunkSummarizer::new(&self.config.base_url, &self.config.api_key)
                    .with_model(&self.config.chunk_summary_model)
                    .with_concurrency(self.config.chunk_summary_concurrency);
            FullIndexer::new(
                store,
                CancellableEnricher::new(enricher, cancel_token.clone()),
                CancellableEmbedder::new(embedder, cancel_token.clone()),
                chunk_summarizer,
            )
            .with_parser_config(parser_config)
            .with_retrieval_config(self.config.retrieval_config())
            .with_chunk_summary_enabled(self.config.chunk_summary_enabled)
            .update_repo_with_progress(&self.config.repo_root, &mut progress)?
        };
        cancel_token.check()?;
        if let Some(log) = log.as_deref_mut() {
            log.event("update_completed", update_summary_json(&summary))?;
        }
        Ok(summary)
    }

    fn run_rebuild_semantic_once_with_progress(
        &self,
        mut log: Option<&mut CommandLog>,
        cancel_token: &MatryoshkaCancelToken,
        mut progress: impl FnMut(MatryoshkaProgressEvent),
    ) -> Result<SemanticRebuildSummary> {
        cancel_token.check()?;
        if let Some(log) = log.as_deref_mut() {
            log.event(
                "semantic_rebuild_started",
                json!({
                    "repo_root": self.config.repo_root,
                    "db": self.config.db,
                    "offline": self.config.offline,
                    "embedding_model": if self.config.offline { "deterministic" } else { self.config.embedding_model.as_str() },
                }),
            )?;
        }
        let store = MatryoshkaStore::open(&self.config.db)?;
        let summary = if self.config.offline {
            FullIndexer::new(
                store,
                CancellableEnricher::new(HeuristicEnricher, cancel_token.clone()),
                CancellableEmbedder::new(DeterministicEmbedder::default(), cancel_token.clone()),
                HeuristicChunkSummarizer,
            )
            .with_retrieval_config(self.config.retrieval_config())
            .rebuild_semantic_index_with_progress(&self.config.repo_root, &mut progress)?
        } else {
            FullIndexer::new(
                store,
                CancellableEnricher::new(HeuristicEnricher, cancel_token.clone()),
                CancellableEmbedder::new(
                    EndpointEmbedder::new(
                        &self.config.base_url,
                        &self.config.api_key,
                        self.config.embedding_model.clone(),
                    ),
                    cancel_token.clone(),
                ),
                HeuristicChunkSummarizer,
            )
            .with_retrieval_config(self.config.retrieval_config())
            .rebuild_semantic_index_with_progress(&self.config.repo_root, &mut progress)?
        };
        cancel_token.check()?;
        if let Some(log) = log.as_deref_mut() {
            log.event(
                "semantic_rebuild_completed",
                semantic_rebuild_summary_json(&summary),
            )?;
        }
        Ok(summary)
    }

    fn run_prewarm_once(
        &self,
        queries: &[String],
        limit: usize,
        mut log: Option<&mut CommandLog>,
        cancel_token: &MatryoshkaCancelToken,
    ) -> Result<SearchPrewarmSummary> {
        cancel_token.check()?;
        if let Some(log) = log.as_deref_mut() {
            log.event(
                "prewarm_started",
                json!({
                    "db": self.config.db,
                    "offline": self.config.offline,
                    "embedding_model": if self.config.offline { "deterministic" } else { self.config.embedding_model.as_str() },
                    "limit": limit,
                    "query_count": queries.len(),
                    "late_interaction": self.config.late_interaction,
                }),
            )?;
        }
        let store = MatryoshkaStore::open(&self.config.db)?;
        let summary = if self.config.offline {
            SearchEngine::new(
                store,
                CancellableEmbedder::new(DeterministicEmbedder::default(), cancel_token.clone()),
            )
            .with_dense(self.config.dense_enabled)
            .with_late_interaction(self.config.late_interaction)
            .prewarm(queries, limit)?
        } else {
            SearchEngine::new(
                store,
                CancellableEmbedder::new(
                    EndpointEmbedder::new(
                        &self.config.base_url,
                        &self.config.api_key,
                        self.config.embedding_model.clone(),
                    ),
                    cancel_token.clone(),
                ),
            )
            .with_dense(self.config.dense_enabled)
            .with_late_interaction(self.config.late_interaction)
            .prewarm(queries, limit)?
        };
        cancel_token.check()?;
        if let Some(log) = log.as_deref_mut() {
            let retrieval_stats =
                MatryoshkaStore::open(&self.config.db)?.retrieval_index_stats()?;
            log.event(
                "prewarm_completed",
                json!({
                    "fts_records": summary.fts_record_count,
                    "queries": summary.query_count,
                    "warmed_hits": summary.warmed_hit_count,
                    "retrieval_index": retrieval_stats_json(&retrieval_stats),
                }),
            )?;
        }
        Ok(summary)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentTask {
    FindSymbol,
    FindBehavior,
    EditTarget,
    TraceDependency,
    Architecture,
    TestsFor,
    ReadNext,
}

fn search_with_reranker<M: matryoshka_embed_client::Embedder + 'static>(
    engine: SearchEngine<M>,
    config: &MatryoshkaConfig,
    query: &str,
    limit: usize,
    reranker: RerankerOptions,
) -> Result<Vec<SearchHit>> {
    match reranker {
        RerankerOptions::None => engine.search(query, limit),
        RerankerOptions::Chat { model } => engine
            .with_reranker(EndpointReranker::new(
                config.base_url.clone(),
                config.api_key.clone(),
                model,
            ))
            .search(query, limit),
        RerankerOptions::Omlx { model, candidates } => engine
            .with_reranker(
                OmlxReranker::new(config.base_url.clone(), config.api_key.clone(), model)
                    .with_max_candidates(candidates),
            )
            .search(query, limit),
    }
}

fn ensure_reranker_options(reranker: &RerankerOptions) -> Result<()> {
    match reranker {
        RerankerOptions::None => Ok(()),
        RerankerOptions::Chat { model } | RerankerOptions::Omlx { model, .. } => {
            if model.trim().is_empty() {
                Err(anyhow!("reranker model cannot be empty"))
            } else {
                Ok(())
            }
        }
    }
}

fn emit_event(
    on_event: &mut impl FnMut(MatryoshkaEvent),
    progress_writer: &mut ProgressStateWriter,
    event: MatryoshkaEvent,
) {
    progress_writer.record(&event);
    on_event(event);
}

fn prepare_decision(
    on_event: &mut impl FnMut(MatryoshkaEvent),
    progress_writer: &mut ProgressStateWriter,
    log: &mut CommandLog,
    action: &str,
    reason: &str,
) -> Result<()> {
    log.event(
        "prepare_decision",
        json!({
            "action": action,
            "reason": reason,
        }),
    )?;
    emit_event(
        on_event,
        progress_writer,
        MatryoshkaEvent::PrepareDecision {
            action: action.into(),
            reason: reason.into(),
        },
    );
    Ok(())
}

struct ProgressStateWriter {
    path: PathBuf,
    enabled: bool,
}

impl ProgressStateWriter {
    fn new(db: &Path, enabled: bool) -> Self {
        Self {
            path: db
                .parent()
                .unwrap_or_else(|| Path::new(MATRYOSHKA_DIR))
                .join("state")
                .join("progress.json"),
            enabled,
        }
    }

    fn record(&mut self, event: &MatryoshkaEvent) {
        if !self.enabled {
            return;
        }
        let state = progress_state_from_event(event);
        if let Some(parent) = self.path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(
            &self.path,
            serde_json::to_string_pretty(&state).unwrap_or_default(),
        );
    }
}

fn progress_state_from_event(event: &MatryoshkaEvent) -> ProgressState {
    match event {
        MatryoshkaEvent::PrepareWaitingForLock { waited_ms, .. } => progress_state(
            "prepare",
            "running",
            "waiting",
            &format!("Waiting for Matryoshka ({waited_ms} ms)"),
            0.0,
            None,
            None,
            None,
        ),
        MatryoshkaEvent::PrepareLockAcquired { .. } => progress_state(
            "prepare",
            "running",
            "locked",
            "Preparing Matryoshka",
            0.01,
            None,
            None,
            None,
        ),
        MatryoshkaEvent::PrepareLockReleased { .. } => progress_state(
            "prepare",
            "running",
            "released",
            "Matryoshka lock released",
            1.0,
            None,
            None,
            None,
        ),
        MatryoshkaEvent::PrepareStarted { .. } => progress_state(
            "prepare",
            "running",
            "starting",
            "Starting Matryoshka",
            0.0,
            None,
            None,
            None,
        ),
        MatryoshkaEvent::PrepareDecision { action, reason } => progress_state(
            "prepare",
            "running",
            action,
            reason,
            match action.as_str() {
                "index" | "update" | "repair" => 0.03,
                "rebuild_search" => 0.82,
                "prepare_results" => 0.94,
                _ => 0.05,
            },
            None,
            None,
            None,
        ),
        MatryoshkaEvent::IndexerProgress {
            operation,
            progress,
        } => indexer_progress_state(operation, progress),
        MatryoshkaEvent::PrewarmStarted { query_count, .. } => progress_state(
            "prepare",
            "running",
            "prewarming",
            &format!("Warming {query_count} search queries"),
            0.95,
            None,
            None,
            None,
        ),
        MatryoshkaEvent::PrewarmCompleted { .. } => progress_state(
            "prepare",
            "running",
            "prewarmed",
            "Search is warm",
            0.98,
            None,
            None,
            None,
        ),
        MatryoshkaEvent::PrepareCancelling { reason } => progress_state(
            "prepare",
            "cancelling",
            "cancelling",
            reason,
            0.99,
            None,
            None,
            None,
        ),
        MatryoshkaEvent::PrepareCancelled { reason } => progress_state(
            "prepare",
            "cancelled",
            "cancelled",
            reason,
            1.0,
            None,
            None,
            None,
        ),
        MatryoshkaEvent::PrepareCompleted { summary } => progress_state(
            "prepare",
            summary.status.as_str(),
            "ready",
            if summary.is_ready() {
                "Jesco is ready"
            } else {
                "Matryoshka needs attention"
            },
            1.0,
            None,
            Some(summary.file_count),
            Some(summary.file_count),
        ),
    }
}

fn cancel_prepare(
    on_event: &mut impl FnMut(MatryoshkaEvent),
    progress_writer: &mut ProgressStateWriter,
    log: &mut CommandLog,
    reason: &str,
) -> Result<PrepareSummary> {
    log.event(
        "prepare_cancelling",
        json!({
            "reason": reason,
        }),
    )?;
    emit_event(
        on_event,
        progress_writer,
        MatryoshkaEvent::PrepareCancelling {
            reason: reason.into(),
        },
    );
    log.event(
        "prepare_cancelled",
        json!({
            "reason": reason,
        }),
    )?;
    emit_event(
        on_event,
        progress_writer,
        MatryoshkaEvent::PrepareCancelled {
            reason: reason.into(),
        },
    );
    Err(cancelled_error())
}

fn cancelled_error() -> anyhow::Error {
    anyhow!("matryoshka prepare cancelled")
}

pub fn is_cancelled_error(error: &(dyn std::error::Error + 'static)) -> bool {
    error.to_string().contains("matryoshka prepare cancelled")
}

fn indexer_progress_state(operation: &str, event: &MatryoshkaProgressEvent) -> ProgressState {
    match event {
        MatryoshkaProgressEvent::Started { .. } => progress_state(
            operation,
            "running",
            "starting",
            "Starting project map",
            0.04,
            None,
            None,
            None,
        ),
        MatryoshkaProgressEvent::DiscoveringFiles => progress_state(
            operation,
            "running",
            "discovering",
            "Finding project files",
            0.05,
            None,
            None,
            None,
        ),
        MatryoshkaProgressEvent::FilesDiscovered { total_files } => progress_state(
            operation,
            "running",
            "discovered",
            "Project files found",
            0.08,
            None,
            Some(0),
            Some(*total_files),
        ),
        MatryoshkaProgressEvent::ParsingFile {
            path,
            index,
            total_files,
        }
        | MatryoshkaProgressEvent::ParsedFile {
            path,
            index,
            total_files,
        } => progress_state(
            operation,
            "running",
            "parsing",
            "Reading code structure",
            0.08 + ratio(*index, *total_files) * 0.22,
            Some(path.clone()),
            Some(*index),
            Some(*total_files),
        ),
        MatryoshkaProgressEvent::EnrichingFile {
            path,
            index,
            total_files,
        }
        | MatryoshkaProgressEvent::EnrichedFile {
            path,
            index,
            total_files,
        } => progress_state(
            operation,
            "running",
            "enriching",
            "Writing file summaries",
            0.30 + ratio(*index, *total_files) * 0.36,
            Some(path.clone()),
            Some(*index),
            Some(*total_files),
        ),
        MatryoshkaProgressEvent::EnrichingChunks { chunk_count } => progress_state(
            operation,
            "running",
            "summarizing_chunks",
            "Summarizing code chunks",
            0.66,
            None,
            Some(0),
            Some(*chunk_count),
        ),
        MatryoshkaProgressEvent::EnrichedChunks { chunk_count } => progress_state(
            operation,
            "running",
            "summarizing_chunks",
            "Code chunks summarized",
            0.76,
            None,
            Some(*chunk_count),
            Some(*chunk_count),
        ),
        MatryoshkaProgressEvent::EnrichingChunkBatch {
            batch_index,
            total_batches,
            ..
        }
        | MatryoshkaProgressEvent::EnrichedChunkBatch {
            batch_index,
            total_batches,
            ..
        } => progress_state(
            operation,
            "running",
            "summarizing_chunks",
            "Summarizing code chunks",
            0.66 + ratio(*batch_index, *total_batches) * 0.10,
            None,
            Some(*batch_index),
            Some(*total_batches),
        ),
        MatryoshkaProgressEvent::EmbeddingBatch {
            batch_index,
            total_batches,
            ..
        }
        | MatryoshkaProgressEvent::EmbeddedBatch {
            batch_index,
            total_batches,
            ..
        } => progress_state(
            operation,
            "running",
            "embedding",
            "Building search data",
            0.76 + ratio(*batch_index, *total_batches) * 0.14,
            None,
            Some(*batch_index),
            Some(*total_batches),
        ),
        MatryoshkaProgressEvent::EmbeddingSkipped { record_count, .. } => progress_state(
            operation,
            "running",
            "embedding_skipped",
            "Dense embeddings disabled; using exact/FTS search data",
            0.84,
            None,
            Some(*record_count),
            Some(*record_count),
        ),
        MatryoshkaProgressEvent::WritingDatabase { .. } => progress_state(
            operation,
            "running",
            "writing",
            "Saving Matryoshka",
            0.88,
            None,
            None,
            None,
        ),
        MatryoshkaProgressEvent::ArtifactQuality { .. } => progress_state(
            operation,
            "running",
            "checking",
            "Checking project map",
            0.91,
            None,
            None,
            None,
        ),
        MatryoshkaProgressEvent::RetrievalIndexHealth { .. } => progress_state(
            operation,
            "running",
            "checking_search",
            "Checking search",
            0.93,
            None,
            None,
            None,
        ),
        MatryoshkaProgressEvent::Completed { file_count, .. } => progress_state(
            operation,
            "running",
            "complete",
            "Project map complete",
            0.94,
            None,
            Some(*file_count),
            Some(*file_count),
        ),
        MatryoshkaProgressEvent::Failed { stage, message } => {
            progress_state(operation, "failed", stage, message, 0.0, None, None, None)
        }
    }
}

fn progress_state(
    operation: &str,
    status: &str,
    phase: &str,
    message: &str,
    percent: f32,
    current_file: Option<String>,
    files_done: Option<usize>,
    files_total: Option<usize>,
) -> ProgressState {
    ProgressState {
        operation: operation.into(),
        status: status.into(),
        phase: phase.into(),
        message: message.into(),
        percent: percent.clamp(0.0, 1.0),
        current_file,
        files_done,
        files_total,
        updated_at_unix_ms: unix_millis(),
    }
}

fn ratio(done: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        (done as f32 / total as f32).clamp(0.0, 1.0)
    }
}

struct CommandLog {
    file: File,
}

impl CommandLog {
    fn open(db: &Path, name: &str) -> Result<Self> {
        let path = log_path(db, name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self { file })
    }

    fn event(&mut self, event: &str, fields: serde_json::Value) -> Result<()> {
        let payload = json!({
            "ts_unix_ms": unix_millis(),
            "event": event,
            "fields": fields,
        });
        writeln!(self.file, "{payload}")?;
        self.file.flush()?;
        Ok(())
    }
}

const PREPARE_LOCK_TIMEOUT: Duration = Duration::from_secs(120);
const PREPARE_LOCK_POLL_INTERVAL: Duration = Duration::from_millis(200);
const PREPARE_LOCK_EVENT_INTERVAL: Duration = Duration::from_secs(1);

struct PrepareLockGuard {
    file: File,
    path: PathBuf,
}

impl Drop for PrepareLockGuard {
    fn drop(&mut self) {
        let _ = unlock_prepare_file(&self.file);
    }
}

fn acquire_prepare_lock(
    db: &Path,
    cancel_token: &MatryoshkaCancelToken,
    on_event: &mut impl FnMut(MatryoshkaEvent),
    log: &mut CommandLog,
) -> Result<PrepareLockGuard> {
    let lock_path = prepare_lock_path(db);
    if let Some(parent) = lock_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)?;
    let started = Instant::now();
    let mut last_wait_event: Option<Instant> = None;

    loop {
        cancel_token.check()?;
        if try_lock_prepare_file(&file)? {
            file.set_len(0)?;
            writeln!(
                &file,
                "{{\"pid\":{},\"acquired_at_unix_ms\":{}}}",
                std::process::id(),
                unix_millis()
            )?;
            log.event(
                "prepare_lock_acquired",
                json!({
                    "db": db,
                    "lock_path": lock_path,
                }),
            )?;
            on_event(MatryoshkaEvent::PrepareLockAcquired {
                db: db.to_path_buf(),
                lock_path: lock_path.clone(),
            });
            return Ok(PrepareLockGuard {
                file,
                path: lock_path,
            });
        }

        let waited = started.elapsed();
        if waited >= PREPARE_LOCK_TIMEOUT {
            return Err(anyhow!(
                "timed out after {} seconds waiting for Matryoshka prepare lock at {}",
                PREPARE_LOCK_TIMEOUT.as_secs(),
                lock_path.display()
            ));
        }
        if last_wait_event
            .map(|last| last.elapsed() >= PREPARE_LOCK_EVENT_INTERVAL)
            .unwrap_or(true)
        {
            let waited_ms = waited.as_millis();
            log.event(
                "prepare_waiting_for_lock",
                json!({
                    "db": db,
                    "lock_path": lock_path,
                    "waited_ms": waited_ms,
                }),
            )?;
            on_event(MatryoshkaEvent::PrepareWaitingForLock {
                db: db.to_path_buf(),
                lock_path: lock_path.clone(),
                waited_ms,
            });
            last_wait_event = Some(Instant::now());
        }
        thread::sleep(PREPARE_LOCK_POLL_INTERVAL);
    }
}

fn release_prepare_lock(
    lock: PrepareLockGuard,
    db: &Path,
    on_event: &mut impl FnMut(MatryoshkaEvent),
    log: &mut CommandLog,
) -> Result<()> {
    let lock_path = lock.path.clone();
    drop(lock);
    log.event(
        "prepare_lock_released",
        json!({
            "db": db,
            "lock_path": lock_path,
        }),
    )?;
    on_event(MatryoshkaEvent::PrepareLockReleased {
        db: db.to_path_buf(),
        lock_path,
    });
    Ok(())
}

fn prepare_lock_path(db: &Path) -> PathBuf {
    PathBuf::from(format!("{}.prepare.lock", db.display()))
}

fn log_prune_report(log: &mut CommandLog, event: &str, report: &OrphanPruneReport) -> Result<()> {
    log.event(
        event,
        json!({
            "file_cards": report.file_cards,
            "folder_cards": report.folder_cards,
            "semantic_records": report.semantic_records,
            "fts_records": report.fts_records,
            "late_vectors": report.late_vectors,
        }),
    )
}

#[cfg(unix)]
fn try_lock_prepare_file(file: &File) -> Result<bool> {
    use std::ffi::c_int;
    use std::io;
    use std::os::fd::AsRawFd;

    const LOCK_EX: c_int = 2;
    const LOCK_NB: c_int = 4;

    unsafe extern "C" {
        fn flock(fd: c_int, operation: c_int) -> c_int;
    }

    // `flock` is an OS advisory lock tied to this file descriptor and is released on drop.
    let rc = unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) };
    if rc == 0 {
        return Ok(true);
    }
    let err = io::Error::last_os_error();
    if err.kind() == io::ErrorKind::WouldBlock {
        Ok(false)
    } else {
        Err(err.into())
    }
}

#[cfg(unix)]
fn unlock_prepare_file(file: &File) -> Result<()> {
    use std::ffi::c_int;
    use std::os::fd::AsRawFd;

    const LOCK_UN: c_int = 8;

    unsafe extern "C" {
        fn flock(fd: c_int, operation: c_int) -> c_int;
    }

    let rc = unsafe { flock(file.as_raw_fd(), LOCK_UN) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error().into())
    }
}

#[cfg(not(unix))]
fn try_lock_prepare_file(_file: &File) -> Result<bool> {
    Ok(true)
}

#[cfg(not(unix))]
fn unlock_prepare_file(_file: &File) -> Result<()> {
    Ok(())
}

pub fn default_db_path(repo_root: &Path) -> PathBuf {
    repo_root.join(MATRYOSHKA_DIR).join(DEFAULT_DB_FILE)
}

pub fn ensure_matryoshka_layout(db: &Path) -> Result<()> {
    if let Some(parent) = db.parent() {
        fs::create_dir_all(parent)?;
        fs::create_dir_all(parent.join("logs"))?;
        fs::create_dir_all(parent.join("state"))?;
    }
    Ok(())
}

pub fn log_path(db: &Path, name: &str) -> PathBuf {
    db.parent()
        .unwrap_or_else(|| Path::new(MATRYOSHKA_DIR))
        .join("logs")
        .join(format!("{name}.jsonl"))
}

pub fn progress_state_path(db: &Path) -> PathBuf {
    db.parent()
        .unwrap_or_else(|| Path::new(MATRYOSHKA_DIR))
        .join("state")
        .join("progress.json")
}

pub fn ready_marker_path(db: &Path) -> PathBuf {
    db.parent()
        .unwrap_or_else(|| Path::new(MATRYOSHKA_DIR))
        .join(READY_MARKER_FILE)
}

pub fn artifact_gap_count(report: &ArtifactQualityReport) -> usize {
    report.file_cards_empty_summary
        + report.folder_cards_empty_summary
        + usize::from(!report.repo_card_has_summary)
}

pub fn retrieval_is_ready(report: &RetrievalIndexReport) -> bool {
    report.semantic_records > 0
        && report.fts_records > 0
        && (!report.dense_enabled || report.embedded_records > 0)
        && (!report.dense_enabled
            || !report.late_interaction_enabled
            || report.records_with_late_vectors > 0)
}

fn retrieval_needs_rebuild(report: &RetrievalIndexReport) -> bool {
    !retrieval_is_ready(report)
}

fn retrieval_report_from_stats(
    stats: RetrievalIndexStats,
    retrieval_config: RetrievalConfig,
    late_interaction: bool,
) -> RetrievalIndexReport {
    RetrievalIndexReport {
        semantic_records: stats.semantic_records,
        embedded_records: stats.embedded_records,
        fts_records: stats.fts_records,
        late_vector_rows: stats.late_vector_rows,
        records_with_late_vectors: stats.records_with_late_vectors,
        retrieval_primary: retrieval_config.primary,
        dense_enabled: retrieval_config.dense_enabled,
        dense_fallback_enabled: retrieval_config.dense_fallback_enabled,
        late_interaction_enabled: retrieval_config.dense_enabled && late_interaction,
    }
}

fn parser_config(ignore: Vec<String>) -> ParserConfig {
    ParserConfig::default().with_ignored_paths(ignore)
}

fn logs_dir(db: &Path) -> PathBuf {
    db.parent()
        .unwrap_or_else(|| Path::new(MATRYOSHKA_DIR))
        .join("logs")
}

fn write_ready_marker(summary: &PrepareSummary) -> Result<()> {
    if let Some(parent) = summary.ready_marker.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &summary.ready_marker,
        serde_json::to_string_pretty(&prepare_summary_json(summary))?,
    )?;
    Ok(())
}

pub fn prepare_summary_json(summary: &PrepareSummary) -> serde_json::Value {
    json!({
        "status": summary.status.as_str(),
        "repo_root": summary.repo_root,
        "db": summary.db,
        "ready_marker": summary.ready_marker,
        "logs": summary.logs_dir,
        "actions_taken": summary.actions_taken,
        "project_map": {
            "status": if artifact_gap_count(&summary.artifact_quality) == 0 {
                "ready"
            } else {
                "needs_attention"
            },
            "files": summary.file_count,
            "folders": summary.folder_count,
            "symbols": summary.symbol_count,
            "cards": {
                "file": summary.artifact_quality.file_cards,
                "folder": summary.artifact_quality.folder_cards,
                "repo": usize::from(summary.artifact_quality.repo_card_has_summary),
                "missing_text": artifact_gap_count(&summary.artifact_quality),
                "empty_file_samples": summary.artifact_quality.empty_file_summary_samples,
                "empty_folder_samples": summary.artifact_quality.empty_folder_summary_samples,
            },
        },
        "search": {
            "status": if retrieval_is_ready(&summary.retrieval_index) {
                "ready"
            } else {
                "needs_refresh"
            },
            "semantic_records": summary.semantic_record_count,
            "embedded_records": summary.retrieval_index.embedded_records,
            "fts_records": summary.retrieval_index.fts_records,
            "late_vector_rows": summary.retrieval_index.late_vector_rows,
            "records_with_late_vectors": summary.retrieval_index.records_with_late_vectors,
            "retrieval_primary": summary.retrieval_index.retrieval_primary,
            "dense_enabled": summary.retrieval_index.dense_enabled,
            "dense_fallback_enabled": summary.retrieval_index.dense_fallback_enabled,
            "late_interaction_enabled": summary.retrieval_index.late_interaction_enabled,
        },
        "changes": {
            "changed_files": summary.changed_files,
            "removed_files": summary.removed_files,
            "changed_folders": summary.changed_folders,
            "repo_card_updated": summary.repo_card_updated,
        },
        "prepare_results": {
            "fts_records": summary.prewarm.fts_record_count,
            "query_count": summary.prewarm.query_count,
            "warmed_hits": summary.prewarm.warmed_hit_count,
        },
        "embedding_model": summary.embedding_model,
    })
}

fn update_summary_json(summary: &UpdateSummary) -> serde_json::Value {
    json!({
        "files": summary.file_count,
        "folders": summary.folder_count,
        "symbols": summary.symbol_count,
        "semantic_records": summary.semantic_record_count,
        "artifact_quality": &summary.artifact_quality,
        "retrieval_index": &summary.retrieval_index,
        "changed_files": summary.changed_files,
        "removed_files": summary.removed_files,
        "changed_folders": summary.changed_folders,
        "repo_card_updated": summary.repo_card_updated,
        "embedding_model": summary.embedding_model,
    })
}

fn semantic_rebuild_summary_json(summary: &SemanticRebuildSummary) -> serde_json::Value {
    json!({
        "semantic_records": summary.semantic_record_count,
        "file_card_records": summary.file_card_record_count,
        "folder_card_records": summary.folder_card_record_count,
        "repo_card_records": summary.repo_card_record_count,
        "artifact_quality": &summary.artifact_quality,
        "retrieval_index": &summary.retrieval_index,
        "embedding_model": summary.embedding_model,
    })
}

fn retrieval_stats_json(stats: &RetrievalIndexStats) -> serde_json::Value {
    json!({
        "semantic_records": stats.semantic_records,
        "embedded_records": stats.embedded_records,
        "fts_records": stats.fts_records,
        "late_vector_rows": stats.late_vector_rows,
        "records_with_late_vectors": stats.records_with_late_vectors,
    })
}

fn task_query(task: AgentTask, query: &str) -> String {
    match task {
        AgentTask::FindSymbol => format!("where is {query} defined symbol definition usage"),
        AgentTask::FindBehavior => format!("how does {query} behavior logic responsibility work"),
        AgentTask::EditTarget => format!("where should I edit change fix implement {query}"),
        AgentTask::TraceDependency => {
            format!("trace dependency impact blast radius downstream upstream {query}")
        }
        AgentTask::Architecture => format!("repository architecture overview subsystem {query}"),
        AgentTask::TestsFor => format!("tests fixtures spec coverage for {query}"),
        AgentTask::ReadNext => {
            format!("read next before editing understand implementation {query}")
        }
    }
}

fn select_related_file_ids(
    primary: &str,
    candidates: &[String],
    query: &str,
    limit: usize,
) -> Vec<String> {
    let wants_tests = query_wants_tests(query);
    let mut seen = std::collections::BTreeSet::new();
    let mut scored = candidates
        .iter()
        .enumerate()
        .filter(|(_, file_id)| file_id.as_str() != primary)
        .filter(|(_, file_id)| seen.insert((*file_id).clone()))
        .filter(|(_, file_id)| wants_tests || !looks_like_low_signal_test_context(file_id))
        .map(|(index, file_id)| {
            let mut score = 0i32;
            if same_crate_area(primary, file_id) {
                score += 5;
            }
            if same_parent_folder(primary, file_id) {
                score += 3;
            }
            if same_top_level_area(primary, file_id) {
                score += 1;
            }
            (score, index, file_id.clone())
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    scored
        .into_iter()
        .take(limit)
        .map(|(_, _, file_id)| file_id)
        .collect()
}

fn same_crate_area(left: &str, right: &str) -> bool {
    path_segment(left, 0) == Some("crates")
        && path_segment(right, 0) == Some("crates")
        && path_segment(left, 1) == path_segment(right, 1)
}

fn same_top_level_area(left: &str, right: &str) -> bool {
    path_segment(left, 0).is_some() && path_segment(left, 0) == path_segment(right, 0)
}

fn same_parent_folder(left: &str, right: &str) -> bool {
    left.rsplit_once('/').map(|(parent, _)| parent)
        == right.rsplit_once('/').map(|(parent, _)| parent)
}

fn path_segment(path: &str, index: usize) -> Option<&str> {
    path.split('/').nth(index)
}

fn query_wants_tests(query: &str) -> bool {
    query
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .any(|token| {
            matches!(
                token.to_ascii_lowercase().as_str(),
                "test" | "tests" | "testing" | "fixture" | "fixtures" | "spec" | "coverage"
            )
        })
}

fn looks_like_low_signal_test_context(path: &str) -> bool {
    path.contains("/fixtures/")
        || path.contains("/tests/")
        || path.contains("/tests/fixtures/")
        || path.contains("/__tests__/")
        || path.ends_with("_test.rs")
        || path.ends_with("_test.py")
        || path.contains(".test.")
        || path.contains(".spec.")
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}
