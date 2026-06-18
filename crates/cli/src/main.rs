use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use matryoshka_embed_client::{DeterministicEmbedder, EndpointEmbedder};
use matryoshka_enricher::{HeuristicEnricher, MlxChatEnricher};
use matryoshka_indexer::{
    ArtifactQualityReport, FullIndexer, IndexSummary, MatryoshkaProgressEvent,
    RetrievalIndexReport, SemanticRebuildSummary, UpdateSummary,
};
use matryoshka_parser::ParserConfig;
use matryoshka_read_api::{ReadApi, ReadPackMode};
use matryoshka_search::{
    EndpointReranker, OmlxReranker, SearchEngine, SearchPrewarmSummary, default_prewarm_queries,
};
use matryoshka_store_sqlite::{CardSummaryRow, MatryoshkaStore, RetrievalIndexStats};
use matryoshka_watcher::RepoWatcher;
use serde_json::json;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:44445";
const DEFAULT_API_KEY: &str = "2508";
const DEFAULT_EMBED_MODEL: &str = "mlx-community--embeddinggemma-300m-bf16";
const DEFAULT_CHAT_MODEL: &str = "MercuriusDream--Qwen3.5-4B-MLX-mxfp8";
const DEFAULT_OMLX_RERANK_MODEL: &str = "mlx-community--Qwen3-Reranker-0.6B-mxfp8";
const MATRYOSHKA_DIR: &str = ".matryoshka";
const DEFAULT_DB_FILE: &str = "matryoshka.db";
const WATCH_PID_FILE: &str = "watch.pid";
const READY_MARKER_FILE: &str = ".jesco-prewarm-complete";

#[derive(Debug, Parser)]
#[command(name = "matryoshka-rs")]
#[command(about = "Rust-first Matryoshka code intelligence core")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Prepare {
        repo_root: PathBuf,
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        offline: bool,
        #[arg(long, default_value = DEFAULT_BASE_URL)]
        base_url: String,
        #[arg(long, default_value = DEFAULT_API_KEY)]
        api_key: String,
        #[arg(long = "embedding-model", visible_alias = "embed-model", default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
        #[arg(long = "model", visible_alias = "chat-model", default_value = DEFAULT_CHAT_MODEL)]
        chat_model: String,
        #[arg(long = "ignore", value_name = "PATH")]
        ignore: Vec<String>,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        #[arg(long = "query")]
        queries: Vec<String>,
        #[arg(long, default_value_t = false)]
        no_late_interaction: bool,
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    Index {
        repo_root: PathBuf,
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        offline: bool,
        #[arg(long, default_value = DEFAULT_BASE_URL)]
        base_url: String,
        #[arg(long, default_value = DEFAULT_API_KEY)]
        api_key: String,
        #[arg(long = "embedding-model", visible_alias = "embed-model", default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
        #[arg(long = "model", visible_alias = "chat-model", default_value = DEFAULT_CHAT_MODEL)]
        chat_model: String,
        #[arg(long, default_value_t = false)]
        progress_jsonl: bool,
        #[arg(long = "ignore", value_name = "PATH")]
        ignore: Vec<String>,
        #[arg(long, default_value_t = false)]
        watch: bool,
        #[arg(long, default_value_t = false)]
        watch_daemon: bool,
    },
    Update {
        repo_root: PathBuf,
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        offline: bool,
        #[arg(long, default_value = DEFAULT_BASE_URL)]
        base_url: String,
        #[arg(long, default_value = DEFAULT_API_KEY)]
        api_key: String,
        #[arg(long = "embedding-model", visible_alias = "embed-model", default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
        #[arg(long = "model", visible_alias = "chat-model", default_value = DEFAULT_CHAT_MODEL)]
        chat_model: String,
        #[arg(long, default_value_t = false)]
        progress_jsonl: bool,
        #[arg(long = "ignore", value_name = "PATH")]
        ignore: Vec<String>,
    },
    Watch {
        repo_root: PathBuf,
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        offline: bool,
        #[arg(long, default_value = DEFAULT_BASE_URL)]
        base_url: String,
        #[arg(long, default_value = DEFAULT_API_KEY)]
        api_key: String,
        #[arg(long = "embedding-model", visible_alias = "embed-model", default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
        #[arg(long = "model", visible_alias = "chat-model", default_value = DEFAULT_CHAT_MODEL)]
        chat_model: String,
        #[arg(long, default_value_t = 2_000)]
        interval_ms: u64,
        #[arg(long, default_value_t = 3_000)]
        debounce_ms: u64,
        #[arg(long = "ignore", value_name = "PATH")]
        ignore: Vec<String>,
        #[arg(long, default_value_t = false)]
        daemon: bool,
        #[arg(long, default_value_t = false)]
        skip_startup_update: bool,
    },
    RebuildSemantic {
        repo_root: PathBuf,
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        offline: bool,
        #[arg(long, default_value = DEFAULT_BASE_URL)]
        base_url: String,
        #[arg(long, default_value = DEFAULT_API_KEY)]
        api_key: String,
        #[arg(long = "embedding-model", visible_alias = "embed-model", default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
        #[arg(long, default_value_t = false)]
        progress_jsonl: bool,
    },
    Search {
        #[arg(long)]
        db: Option<PathBuf>,
        query: String,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        #[arg(long, default_value_t = false)]
        offline: bool,
        #[arg(long, default_value = DEFAULT_BASE_URL)]
        base_url: String,
        #[arg(long, default_value = DEFAULT_API_KEY)]
        api_key: String,
        #[arg(long = "embedding-model", visible_alias = "embed-model", default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
        #[arg(long, default_value_t = false)]
        rerank: bool,
        #[arg(long = "rerank-model", default_value = DEFAULT_CHAT_MODEL)]
        rerank_model: String,
        #[arg(long, default_value_t = false)]
        omlx_rerank: bool,
        #[arg(long = "omlx-rerank-model", default_value = DEFAULT_OMLX_RERANK_MODEL)]
        omlx_rerank_model: String,
        #[arg(long = "omlx-rerank-candidates", default_value_t = 20)]
        omlx_rerank_candidates: usize,
        #[arg(long, default_value_t = false)]
        no_late_interaction: bool,
    },
    Op {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(value_enum)]
        task: AgentTask,
        query: String,
        #[arg(long, default_value_t = 8)]
        limit: usize,
        #[arg(long, default_value_t = false)]
        offline: bool,
        #[arg(long, default_value = DEFAULT_BASE_URL)]
        base_url: String,
        #[arg(long, default_value = DEFAULT_API_KEY)]
        api_key: String,
        #[arg(long = "embedding-model", visible_alias = "embed-model", default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
        #[arg(long, default_value_t = false)]
        rerank: bool,
        #[arg(long = "rerank-model", default_value = DEFAULT_CHAT_MODEL)]
        rerank_model: String,
        #[arg(long, default_value_t = false)]
        omlx_rerank: bool,
        #[arg(long = "omlx-rerank-model", default_value = DEFAULT_OMLX_RERANK_MODEL)]
        omlx_rerank_model: String,
        #[arg(long = "omlx-rerank-candidates", default_value_t = 20)]
        omlx_rerank_candidates: usize,
        #[arg(long, default_value_t = false)]
        no_late_interaction: bool,
    },
    Prewarm {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long)]
        repo_root: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        offline: bool,
        #[arg(long, default_value = DEFAULT_BASE_URL)]
        base_url: String,
        #[arg(long, default_value = DEFAULT_API_KEY)]
        api_key: String,
        #[arg(long = "embedding-model", visible_alias = "embed-model", default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
        #[arg(long, default_value_t = 6)]
        limit: usize,
        #[arg(long = "query")]
        queries: Vec<String>,
        #[arg(long, default_value_t = false)]
        no_late_interaction: bool,
        #[arg(long, default_value_t = false)]
        ensure_fresh: bool,
        #[arg(long, default_value_t = false)]
        watch: bool,
        #[arg(long, default_value_t = false)]
        watch_daemon: bool,
    },
    Read {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long)]
        repo_root: Option<PathBuf>,
        file: String,
    },
    ReadBundle {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long)]
        repo_root: Option<PathBuf>,
        query: String,
        #[arg(long, default_value_t = 4)]
        limit: usize,
        #[arg(long, default_value_t = 3)]
        related: usize,
        #[arg(long, value_enum, default_value_t = CliReadPackMode::Brief)]
        mode: CliReadPackMode,
        #[arg(long, default_value_t = false)]
        offline: bool,
        #[arg(long, default_value = DEFAULT_BASE_URL)]
        base_url: String,
        #[arg(long, default_value = DEFAULT_API_KEY)]
        api_key: String,
        #[arg(long = "embedding-model", visible_alias = "embed-model", default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
        #[arg(long, default_value_t = false)]
        rerank: bool,
        #[arg(long = "rerank-model", default_value = DEFAULT_CHAT_MODEL)]
        rerank_model: String,
        #[arg(long, default_value_t = false)]
        omlx_rerank: bool,
        #[arg(long = "omlx-rerank-model", default_value = DEFAULT_OMLX_RERANK_MODEL)]
        omlx_rerank_model: String,
        #[arg(long = "omlx-rerank-candidates", default_value_t = 20)]
        omlx_rerank_candidates: usize,
        #[arg(long, default_value_t = false)]
        no_late_interaction: bool,
    },
    Cards {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long, default_value_t = false)]
        summaries: bool,
        #[arg(long, default_value_t = false)]
        empty: bool,
        #[arg(long, default_value_t = false)]
        json: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum AgentTask {
    FindSymbol,
    FindBehavior,
    EditTarget,
    TraceDependency,
    Architecture,
    TestsFor,
    ReadNext,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliReadPackMode {
    Brief,
    Edit,
    Flow,
}

impl From<CliReadPackMode> for ReadPackMode {
    fn from(value: CliReadPackMode) -> Self {
        match value {
            CliReadPackMode::Brief => ReadPackMode::Brief,
            CliReadPackMode::Edit => ReadPackMode::Edit,
            CliReadPackMode::Flow => ReadPackMode::Flow,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Prepare {
            repo_root,
            db,
            offline,
            base_url,
            api_key,
            embed_model,
            chat_model,
            ignore,
            limit,
            queries,
            no_late_interaction,
            json,
        } => {
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            let summary = run_prepare(PrepareOptions {
                repo_root,
                db,
                offline,
                base_url,
                api_key,
                embed_model,
                chat_model,
                ignore,
                limit,
                queries,
                late_interaction: !no_late_interaction,
            })?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&prepare_summary_json(&summary))?
                );
            } else {
                print_prepare_summary(&summary);
            }
        }
        Command::Index {
            repo_root,
            db,
            offline,
            base_url,
            api_key,
            embed_model,
            chat_model,
            progress_jsonl,
            ignore,
            watch,
            watch_daemon,
        } => {
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            let mut command_log = CommandLog::open(&db, "index")?;
            command_log.event(
                "index_started",
                json!({
                    "repo_root": repo_root,
                    "db": db,
                    "offline": offline,
                    "embedding_model": if offline { "deterministic" } else { embed_model.as_str() },
                    "chat_model": if offline { "heuristic" } else { chat_model.as_str() },
                }),
            )?;
            let store = MatryoshkaStore::open(&db)?;
            let parser_config = parser_config(ignore);
            if offline {
                let indexer =
                    FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default())
                        .with_parser_config(parser_config);
                let summary = if progress_jsonl {
                    indexer.index_repo_with_progress(&repo_root, print_progress_jsonl)?
                } else {
                    indexer.index_repo(&repo_root)?
                };
                command_log.event("index_completed", index_summary_json(&summary))?;
                if !progress_jsonl {
                    print_index_summary(summary);
                }
            } else {
                let enricher =
                    MlxChatEnricher::new(&base_url, &api_key).with_model(chat_model.clone());
                let embedder = EndpointEmbedder::new(&base_url, &api_key, embed_model.clone());
                let indexer =
                    FullIndexer::new(store, enricher, embedder).with_parser_config(parser_config);
                let summary = if progress_jsonl {
                    indexer.index_repo_with_progress(&repo_root, print_progress_jsonl)?
                } else {
                    indexer.index_repo(&repo_root)?
                };
                command_log.event("index_completed", index_summary_json(&summary))?;
                if !progress_jsonl {
                    print_index_summary(summary);
                }
            }
            if watch || watch_daemon {
                start_watch_after_index(
                    &repo_root,
                    &db,
                    offline,
                    &base_url,
                    &api_key,
                    &embed_model,
                    &chat_model,
                    watch_daemon,
                )?;
            }
        }
        Command::Update {
            repo_root,
            db,
            offline,
            base_url,
            api_key,
            embed_model,
            chat_model,
            progress_jsonl,
            ignore,
        } => {
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            let mut command_log = CommandLog::open(&db, "update")?;
            command_log.event(
                "update_started",
                json!({
                    "repo_root": repo_root,
                    "db": db,
                    "offline": offline,
                    "embedding_model": if offline { "deterministic" } else { embed_model.as_str() },
                    "chat_model": if offline { "heuristic" } else { chat_model.as_str() },
                }),
            )?;
            let store = MatryoshkaStore::open(&db)?;
            let parser_config = parser_config(ignore);
            if offline {
                let indexer =
                    FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default())
                        .with_parser_config(parser_config);
                let summary = if progress_jsonl {
                    indexer.update_repo_with_progress(repo_root, print_progress_jsonl)?
                } else {
                    indexer.update_repo(repo_root)?
                };
                command_log.event("update_completed", update_summary_json(&summary))?;
                if !progress_jsonl {
                    print_update_summary(summary);
                }
            } else {
                let enricher = MlxChatEnricher::new(&base_url, &api_key).with_model(chat_model);
                let embedder = EndpointEmbedder::new(&base_url, &api_key, embed_model);
                let indexer =
                    FullIndexer::new(store, enricher, embedder).with_parser_config(parser_config);
                let summary = if progress_jsonl {
                    indexer.update_repo_with_progress(repo_root, print_progress_jsonl)?
                } else {
                    indexer.update_repo(repo_root)?
                };
                command_log.event("update_completed", update_summary_json(&summary))?;
                if !progress_jsonl {
                    print_update_summary(summary);
                }
            }
        }
        Command::Watch {
            repo_root,
            db,
            offline,
            base_url,
            api_key,
            embed_model,
            chat_model,
            interval_ms,
            debounce_ms,
            ignore,
            daemon,
            skip_startup_update,
        } => {
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            let options = WatchLoopOptions {
                repo_root,
                db,
                offline,
                base_url,
                api_key,
                embed_model,
                chat_model,
                interval_ms,
                debounce_ms,
                ignore,
                skip_startup_update,
            };
            if daemon {
                spawn_watch_daemon(&options)?;
                return Ok(());
            }
            run_watch_loop(options)?;
        }
        Command::Search {
            db,
            query,
            limit,
            offline,
            base_url,
            api_key,
            embed_model,
            rerank,
            rerank_model,
            omlx_rerank,
            omlx_rerank_model,
            omlx_rerank_candidates,
            no_late_interaction,
        } => {
            let db = resolve_db_path(db, None)?;
            ensure_matryoshka_layout(&db)?;
            ensure_single_reranker(rerank, omlx_rerank)?;
            let store = MatryoshkaStore::open(&db)?;
            let late_interaction = !no_late_interaction;
            let hits = if offline && omlx_rerank {
                SearchEngine::new(store, DeterministicEmbedder::default())
                    .with_late_interaction(late_interaction)
                    .with_reranker(
                        OmlxReranker::new(base_url, api_key, omlx_rerank_model)
                            .with_max_candidates(omlx_rerank_candidates),
                    )
                    .search(&query, limit)?
            } else if offline {
                SearchEngine::new(store, DeterministicEmbedder::default())
                    .with_late_interaction(late_interaction)
                    .search(&query, limit)?
            } else if omlx_rerank {
                SearchEngine::new(
                    store,
                    EndpointEmbedder::new(base_url.clone(), api_key.clone(), embed_model),
                )
                .with_late_interaction(late_interaction)
                .with_reranker(
                    OmlxReranker::new(base_url, api_key, omlx_rerank_model)
                        .with_max_candidates(omlx_rerank_candidates),
                )
                .search(&query, limit)?
            } else if rerank {
                SearchEngine::new(
                    store,
                    EndpointEmbedder::new(base_url.clone(), api_key.clone(), embed_model),
                )
                .with_late_interaction(late_interaction)
                .with_reranker(EndpointReranker::new(base_url, api_key, rerank_model))
                .search(&query, limit)?
            } else {
                SearchEngine::new(store, EndpointEmbedder::new(base_url, api_key, embed_model))
                    .with_late_interaction(late_interaction)
                    .search(&query, limit)?
            };
            println!("{}", serde_json::to_string_pretty(&hits)?);
        }
        Command::Op {
            db,
            task,
            query,
            limit,
            offline,
            base_url,
            api_key,
            embed_model,
            rerank,
            rerank_model,
            omlx_rerank,
            omlx_rerank_model,
            omlx_rerank_candidates,
            no_late_interaction,
        } => {
            let db = resolve_db_path(db, None)?;
            ensure_matryoshka_layout(&db)?;
            ensure_single_reranker(rerank, omlx_rerank)?;
            let store = MatryoshkaStore::open(&db)?;
            let task_query = task_query(task, &query);
            let late_interaction = !no_late_interaction;
            let hits = if offline && omlx_rerank {
                SearchEngine::new(store, DeterministicEmbedder::default())
                    .with_late_interaction(late_interaction)
                    .with_reranker(
                        OmlxReranker::new(base_url, api_key, omlx_rerank_model)
                            .with_max_candidates(omlx_rerank_candidates),
                    )
                    .search(&task_query, limit)?
            } else if offline {
                SearchEngine::new(store, DeterministicEmbedder::default())
                    .with_late_interaction(late_interaction)
                    .search(&task_query, limit)?
            } else if omlx_rerank {
                SearchEngine::new(
                    store,
                    EndpointEmbedder::new(base_url.clone(), api_key.clone(), embed_model),
                )
                .with_late_interaction(late_interaction)
                .with_reranker(
                    OmlxReranker::new(base_url, api_key, omlx_rerank_model)
                        .with_max_candidates(omlx_rerank_candidates),
                )
                .search(&task_query, limit)?
            } else if rerank {
                SearchEngine::new(
                    store,
                    EndpointEmbedder::new(base_url.clone(), api_key.clone(), embed_model),
                )
                .with_late_interaction(late_interaction)
                .with_reranker(EndpointReranker::new(base_url, api_key, rerank_model))
                .search(&task_query, limit)?
            } else {
                SearchEngine::new(store, EndpointEmbedder::new(base_url, api_key, embed_model))
                    .with_late_interaction(late_interaction)
                    .search(&task_query, limit)?
            };
            println!("{}", serde_json::to_string_pretty(&hits)?);
        }
        Command::Prewarm {
            db,
            repo_root,
            offline,
            base_url,
            api_key,
            embed_model,
            limit,
            queries,
            no_late_interaction,
            ensure_fresh,
            watch,
            watch_daemon,
        } => {
            let repo_root = resolve_optional_repo_root(repo_root)?;
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            let mut command_log = CommandLog::open(&db, "prewarm")?;
            command_log.event(
                "prewarm_started",
                json!({
                    "repo_root": repo_root,
                    "db": db,
                    "offline": offline,
                    "embedding_model": if offline { "deterministic" } else { embed_model.as_str() },
                    "ensure_fresh": ensure_fresh,
                    "limit": limit,
                }),
            )?;
            if ensure_fresh {
                let summary = run_update_once(
                    &repo_root,
                    &db,
                    offline,
                    &base_url,
                    &api_key,
                    &embed_model,
                    DEFAULT_CHAT_MODEL,
                    ParserConfig::default(),
                    Some(&mut command_log),
                )?;
                print_update_summary(summary);
            }
            let store = MatryoshkaStore::open(&db)?;
            let queries = if queries.is_empty() {
                default_prewarm_queries()
            } else {
                queries
            };
            let late_interaction = !no_late_interaction;
            let summary = if offline {
                SearchEngine::new(store, DeterministicEmbedder::default())
                    .with_late_interaction(late_interaction)
                    .prewarm(&queries, limit)?
            } else {
                SearchEngine::new(
                    store,
                    EndpointEmbedder::new(base_url.clone(), api_key.clone(), embed_model.clone()),
                )
                .with_late_interaction(late_interaction)
                .prewarm(&queries, limit)?
            };
            println!("fts_records: {}", summary.fts_record_count);
            println!("queries: {}", summary.query_count);
            println!("warmed_hits: {}", summary.warmed_hit_count);
            let retrieval_stats = MatryoshkaStore::open(&db)?.retrieval_index_stats()?;
            println!("embedded_records: {}", retrieval_stats.embedded_records);
            println!("late_vector_rows: {}", retrieval_stats.late_vector_rows);
            println!(
                "records_with_late_vectors: {}",
                retrieval_stats.records_with_late_vectors
            );
            command_log.event(
                "prewarm_completed",
                json!({
                    "fts_records": summary.fts_record_count,
                    "queries": summary.query_count,
                    "warmed_hits": summary.warmed_hit_count,
                    "retrieval_index": {
                        "semantic_records": retrieval_stats.semantic_records,
                        "embedded_records": retrieval_stats.embedded_records,
                        "fts_records": retrieval_stats.fts_records,
                        "late_vector_rows": retrieval_stats.late_vector_rows,
                        "records_with_late_vectors": retrieval_stats.records_with_late_vectors,
                    },
                }),
            )?;
            if watch || watch_daemon {
                start_watch_after_index(
                    &repo_root,
                    &db,
                    offline,
                    &base_url,
                    &api_key,
                    &embed_model,
                    DEFAULT_CHAT_MODEL,
                    watch_daemon,
                )?;
            }
        }
        Command::RebuildSemantic {
            repo_root,
            db,
            offline,
            base_url,
            api_key,
            embed_model,
            progress_jsonl,
        } => {
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            let mut command_log = CommandLog::open(&db, "semantic-rebuild")?;
            command_log.event(
                "semantic_rebuild_started",
                json!({
                    "repo_root": repo_root,
                    "db": db,
                    "offline": offline,
                    "embedding_model": if offline { "deterministic" } else { embed_model.as_str() },
                }),
            )?;
            let store = MatryoshkaStore::open(&db)?;
            let summary = if offline {
                let indexer =
                    FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default());
                if progress_jsonl {
                    indexer.rebuild_semantic_index_with_progress(repo_root, print_progress_jsonl)?
                } else {
                    indexer.rebuild_semantic_index(repo_root)?
                }
            } else {
                let indexer = FullIndexer::new(
                    store,
                    HeuristicEnricher,
                    EndpointEmbedder::new(base_url, api_key, embed_model),
                );
                if progress_jsonl {
                    indexer.rebuild_semantic_index_with_progress(repo_root, print_progress_jsonl)?
                } else {
                    indexer.rebuild_semantic_index(repo_root)?
                }
            };
            command_log.event(
                "semantic_rebuild_completed",
                semantic_rebuild_summary_json(&summary),
            )?;
            if !progress_jsonl {
                print_semantic_rebuild_summary(summary);
            }
        }
        Command::Read {
            db,
            repo_root,
            file,
        } => {
            let repo_root = resolve_optional_repo_root(repo_root)?;
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            let read = ReadApi::new(MatryoshkaStore::open(&db)?, repo_root);
            println!("{}", serde_json::to_string_pretty(&read.read(&file)?)?);
        }
        Command::ReadBundle {
            db,
            repo_root,
            query,
            limit,
            related,
            mode,
            offline,
            base_url,
            api_key,
            embed_model,
            rerank,
            rerank_model,
            omlx_rerank,
            omlx_rerank_model,
            omlx_rerank_candidates,
            no_late_interaction,
        } => {
            let repo_root = resolve_optional_repo_root(repo_root)?;
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            ensure_single_reranker(rerank, omlx_rerank)?;
            let store = MatryoshkaStore::open(&db)?;
            let late_interaction = !no_late_interaction;
            let hits = if offline && omlx_rerank {
                SearchEngine::new(store.clone(), DeterministicEmbedder::default())
                    .with_late_interaction(late_interaction)
                    .with_reranker(
                        OmlxReranker::new(base_url, api_key, omlx_rerank_model)
                            .with_max_candidates(omlx_rerank_candidates),
                    )
                    .search(&task_query(AgentTask::ReadNext, &query), limit)?
            } else if offline {
                SearchEngine::new(store.clone(), DeterministicEmbedder::default())
                    .with_late_interaction(late_interaction)
                    .search(&task_query(AgentTask::ReadNext, &query), limit)?
            } else if omlx_rerank {
                SearchEngine::new(
                    store.clone(),
                    EndpointEmbedder::new(base_url.clone(), api_key.clone(), embed_model),
                )
                .with_late_interaction(late_interaction)
                .with_reranker(
                    OmlxReranker::new(base_url, api_key, omlx_rerank_model)
                        .with_max_candidates(omlx_rerank_candidates),
                )
                .search(&task_query(AgentTask::ReadNext, &query), limit)?
            } else if rerank {
                SearchEngine::new(
                    store.clone(),
                    EndpointEmbedder::new(base_url.clone(), api_key.clone(), embed_model),
                )
                .with_late_interaction(late_interaction)
                .with_reranker(EndpointReranker::new(base_url, api_key, rerank_model))
                .search(&task_query(AgentTask::ReadNext, &query), limit)?
            } else {
                SearchEngine::new(
                    store.clone(),
                    EndpointEmbedder::new(base_url, api_key, embed_model),
                )
                .with_late_interaction(late_interaction)
                .search(&task_query(AgentTask::ReadNext, &query), limit)?
            };
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
                anyhow::bail!("no file-level search hit found for read bundle query");
            };
            let related_file_ids =
                select_related_file_ids(primary, &file_ids[1..], &query, related);
            let read = ReadApi::new(store, repo_root);
            let bundle = read.read_bundle(primary, &related_file_ids, mode.into(), related)?;
            println!("{}", serde_json::to_string_pretty(&bundle)?);
        }
        Command::Cards {
            db,
            summaries: _,
            empty,
            json,
        } => {
            let db = resolve_db_path(db, None)?;
            ensure_matryoshka_layout(&db)?;
            let store = MatryoshkaStore::open(&db)?;
            let mut rows = store.load_card_summaries()?;
            if empty {
                rows.retain(|row| row.is_empty);
            }
            if json {
                println!("{}", serde_json::to_string_pretty(&rows)?);
            } else {
                print_card_summaries(&db, &rows, empty);
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct PrepareOptions {
    repo_root: PathBuf,
    db: PathBuf,
    offline: bool,
    base_url: String,
    api_key: String,
    embed_model: String,
    chat_model: String,
    ignore: Vec<String>,
    limit: usize,
    queries: Vec<String>,
    late_interaction: bool,
}

#[derive(Debug, Clone)]
struct PrepareSummary {
    repo_root: PathBuf,
    db: PathBuf,
    ready_marker: PathBuf,
    logs_dir: PathBuf,
    status: String,
    actions_taken: Vec<String>,
    file_count: usize,
    folder_count: usize,
    symbol_count: usize,
    semantic_record_count: usize,
    changed_files: usize,
    removed_files: usize,
    changed_folders: usize,
    repo_card_updated: bool,
    artifact_quality: ArtifactQualityReport,
    retrieval_index: RetrievalIndexReport,
    prewarm: SearchPrewarmSummary,
    embedding_model: String,
}

#[derive(Debug, Clone)]
struct WatchLoopOptions {
    repo_root: PathBuf,
    db: PathBuf,
    offline: bool,
    base_url: String,
    api_key: String,
    embed_model: String,
    chat_model: String,
    interval_ms: u64,
    debounce_ms: u64,
    ignore: Vec<String>,
    skip_startup_update: bool,
}

struct CommandLog {
    path: PathBuf,
    file: File,
}

impl CommandLog {
    fn open(db: &Path, name: &str) -> Result<Self> {
        let path = log_path(db, name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self { path, file })
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

fn run_prepare(options: PrepareOptions) -> Result<PrepareSummary> {
    ensure_matryoshka_layout(&options.db)?;
    let mut log = CommandLog::open(&options.db, "prepare")?;
    let logs_dir = logs_dir(&options.db);
    let ready_marker = ready_marker_path(&options.db);
    let parser_config = parser_config(options.ignore.clone());
    let existing_file_count = indexed_file_count(&options.db).unwrap_or(0);
    let existing_gap_count = existing_card_gap_count(&options.db).unwrap_or(0);
    let existing_search_missing =
        existing_retrieval_needs_rebuild(&options.db, options.late_interaction).unwrap_or(false);
    let ready_marker_exists = ready_marker.exists();
    let mut actions_taken = Vec::new();

    log.event(
        "prepare_started",
        json!({
            "repo_root": options.repo_root,
            "db": options.db,
            "offline": options.offline,
            "embedding_model": if options.offline { "deterministic" } else { options.embed_model.as_str() },
            "chat_model": if options.offline { "heuristic" } else { options.chat_model.as_str() },
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
    log.event(
        "prepare_decision",
        json!({
            "action": first_action,
            "reason": if existing_file_count == 0 {
                "no indexed files found"
            } else if existing_gap_count > 0 {
                "project map has gaps"
            } else if existing_search_missing {
                "search data is missing or incomplete"
            } else if !ready_marker_exists {
                "ready marker missing"
            } else {
                "refresh current project map"
            },
        }),
    )?;
    let mut update = run_update_once(
        &options.repo_root,
        &options.db,
        options.offline,
        &options.base_url,
        &options.api_key,
        &options.embed_model,
        &options.chat_model,
        parser_config.clone(),
        Some(&mut log),
    )?;
    actions_taken.push(first_action.to_string());

    if artifact_gap_count(&update.artifact_quality) > 0 && first_action != "repair" {
        log.event(
            "prepare_decision",
            json!({
                "action": "repair",
                "reason": "project map has gaps",
                "missing_text": artifact_gap_count(&update.artifact_quality),
            }),
        )?;
        update = run_update_once(
            &options.repo_root,
            &options.db,
            options.offline,
            &options.base_url,
            &options.api_key,
            &options.embed_model,
            &options.chat_model,
            parser_config,
            Some(&mut log),
        )?;
        actions_taken.push("repair".to_string());
    }

    let mut artifact_quality = update.artifact_quality.clone();
    let mut retrieval_index = update.retrieval_index.clone();
    if retrieval_needs_rebuild(&retrieval_index, options.late_interaction) {
        log.event(
            "prepare_decision",
            json!({
                "action": "rebuild_search",
                "reason": "search data is missing or incomplete",
                "retrieval_index": retrieval_report_json(&retrieval_index),
            }),
        )?;
        let rebuild = run_rebuild_semantic_once(
            &options.repo_root,
            &options.db,
            options.offline,
            &options.base_url,
            &options.api_key,
            &options.embed_model,
            Some(&mut log),
        )?;
        artifact_quality = rebuild.artifact_quality;
        if !actions_taken
            .iter()
            .any(|action| action == "rebuild_search")
        {
            actions_taken.push("rebuild_search".to_string());
        }
    }

    let queries = if options.queries.is_empty() {
        default_prewarm_queries()
    } else {
        options.queries.clone()
    };
    log.event(
        "prepare_decision",
        json!({
            "action": "prepare_results",
            "reason": "make first searches fast and precise",
            "queries": queries,
            "limit": options.limit,
        }),
    )?;
    let prewarm = run_prewarm_once(
        &options.db,
        options.offline,
        &options.base_url,
        &options.api_key,
        &options.embed_model,
        &queries,
        options.limit,
        options.late_interaction,
        Some(&mut log),
    )?;
    actions_taken.push("prepare_results".to_string());

    retrieval_index = retrieval_report_from_stats(
        MatryoshkaStore::open(&options.db)?.retrieval_index_stats()?,
        options.late_interaction,
    );
    let ready = artifact_gap_count(&artifact_quality) == 0
        && retrieval_is_ready(&retrieval_index, options.late_interaction);
    let status = if ready { "ready" } else { "needs_attention" }.to_string();

    let summary = PrepareSummary {
        repo_root: options.repo_root,
        db: options.db,
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
        prewarm,
        embedding_model: update.embedding_model,
    };

    if summary.status == "ready" {
        write_ready_marker(&summary)?;
    }
    log.event("prepare_completed", prepare_summary_json(&summary))?;
    Ok(summary)
}

fn resolve_optional_repo_root(repo_root: Option<PathBuf>) -> Result<PathBuf> {
    Ok(match repo_root {
        Some(repo_root) => repo_root,
        None => std::env::current_dir()?,
    })
}

fn resolve_db_path(db: Option<PathBuf>, repo_root: Option<&Path>) -> Result<PathBuf> {
    Ok(match db {
        Some(db) => db,
        None => repo_root
            .map(default_db_path)
            .unwrap_or(default_db_path(&std::env::current_dir()?)),
    })
}

fn default_db_path(repo_root: &Path) -> PathBuf {
    repo_root.join(MATRYOSHKA_DIR).join(DEFAULT_DB_FILE)
}

fn ensure_matryoshka_layout(db: &Path) -> Result<()> {
    if let Some(parent) = db.parent() {
        fs::create_dir_all(parent)?;
        fs::create_dir_all(parent.join("logs"))?;
    }
    Ok(())
}

fn log_path(db: &Path, name: &str) -> PathBuf {
    db.parent()
        .unwrap_or_else(|| Path::new(MATRYOSHKA_DIR))
        .join("logs")
        .join(format!("{name}.jsonl"))
}

fn pid_path(db: &Path) -> PathBuf {
    db.parent()
        .unwrap_or_else(|| Path::new(MATRYOSHKA_DIR))
        .join(WATCH_PID_FILE)
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

fn start_watch_after_index(
    repo_root: &Path,
    db: &Path,
    offline: bool,
    base_url: &str,
    api_key: &str,
    embed_model: &str,
    chat_model: &str,
    daemon: bool,
) -> Result<()> {
    let options = WatchLoopOptions {
        repo_root: repo_root.to_path_buf(),
        db: db.to_path_buf(),
        offline,
        base_url: base_url.to_string(),
        api_key: api_key.to_string(),
        embed_model: embed_model.to_string(),
        chat_model: chat_model.to_string(),
        interval_ms: 2_000,
        debounce_ms: 3_000,
        ignore: Vec::new(),
        skip_startup_update: false,
    };
    if daemon {
        spawn_watch_daemon(&options)
    } else {
        run_watch_loop(options)
    }
}

fn spawn_watch_daemon(options: &WatchLoopOptions) -> Result<()> {
    ensure_matryoshka_layout(&options.db)?;
    let json_log_path = log_path(&options.db, "watch");
    let stdout_log_path = log_path(&options.db, "watch.stdout");
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&stdout_log_path)?;
    let err_file = log_file.try_clone()?;
    let mut command = ProcessCommand::new(std::env::current_exe()?);
    command
        .arg("watch")
        .arg(&options.repo_root)
        .arg("--db")
        .arg(&options.db)
        .arg("--base-url")
        .arg(&options.base_url)
        .arg("--api-key")
        .arg(&options.api_key)
        .arg("--embedding-model")
        .arg(&options.embed_model)
        .arg("--model")
        .arg(&options.chat_model)
        .arg("--interval-ms")
        .arg(options.interval_ms.to_string())
        .arg("--debounce-ms")
        .arg(options.debounce_ms.to_string())
        .current_dir(&options.repo_root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(err_file));
    #[cfg(unix)]
    command.process_group(0);
    if options.offline {
        command.arg("--offline");
    }
    if options.skip_startup_update {
        command.arg("--skip-startup-update");
    }
    for ignored in &options.ignore {
        command.arg("--ignore").arg(ignored);
    }
    let child = command.spawn()?;
    let pid_path = pid_path(&options.db);
    fs::write(&pid_path, format!("{}\n", child.id()))?;
    println!("watch_daemon_pid: {}", child.id());
    println!("watch_pid_file: {}", pid_path.display());
    println!("watch_log: {}", json_log_path.display());
    println!("watch_stdout_log: {}", stdout_log_path.display());
    Ok(())
}

fn run_watch_loop(options: WatchLoopOptions) -> Result<()> {
    ensure_matryoshka_layout(&options.db)?;
    let parser_config = parser_config(options.ignore.clone());
    let mut log = CommandLog::open(&options.db, "watch")?;
    log.event(
        "watch_started",
        json!({
            "repo_root": options.repo_root,
            "db": options.db,
            "offline": options.offline,
            "interval_ms": options.interval_ms,
            "debounce_ms": options.debounce_ms,
            "startup_update": !options.skip_startup_update,
        }),
    )?;

    if !options.skip_startup_update {
        let summary = run_update_once(
            &options.repo_root,
            &options.db,
            options.offline,
            &options.base_url,
            &options.api_key,
            &options.embed_model,
            &options.chat_model,
            parser_config.clone(),
            Some(&mut log),
        )?;
        print_update_summary(summary);
    }

    let mut watcher = RepoWatcher::new(&options.repo_root)?
        .with_parser_config(parser_config.clone())?
        .with_poll_interval(Duration::from_millis(options.interval_ms))
        .with_debounce_window(Duration::from_millis(options.debounce_ms));
    println!(
        "watching {} every {}ms with {}ms debounce",
        options.repo_root.display(),
        options.interval_ms,
        options.debounce_ms
    );
    println!("watch_log: {}", log.path.display());

    let mut poll_count = 0usize;
    loop {
        poll_count = poll_count.saturating_add(1);
        if poll_count % 25 == 0 {
            log.event(
                "watch_heartbeat",
                json!({
                    "poll_count": poll_count,
                    "interval_ms": options.interval_ms,
                }),
            )?;
        }
        if let Some(batch) = watcher.poll()? {
            println!(
                "change batch detected: changed={} added={} removed={}",
                batch.changed_paths.len(),
                batch.added_paths.len(),
                batch.removed_paths.len()
            );
            log.event(
                "change_batch",
                json!({
                    "changed_paths": batch.changed_paths,
                    "added_paths": batch.added_paths,
                    "removed_paths": batch.removed_paths,
                }),
            )?;
            let summary = run_update_once(
                &options.repo_root,
                &options.db,
                options.offline,
                &options.base_url,
                &options.api_key,
                &options.embed_model,
                &options.chat_model,
                parser_config.clone(),
                Some(&mut log),
            )?;
            print_update_summary(summary);
        }
        thread::sleep(watcher.poll_interval());
    }
}

#[allow(clippy::too_many_arguments)]
fn run_update_once(
    repo_root: &Path,
    db: &Path,
    offline: bool,
    base_url: &str,
    api_key: &str,
    embed_model: &str,
    chat_model: &str,
    parser_config: ParserConfig,
    mut log: Option<&mut CommandLog>,
) -> Result<UpdateSummary> {
    if let Some(log) = log.as_deref_mut() {
        log.event(
            "update_started",
            json!({
                "repo_root": repo_root,
                "db": db,
                "offline": offline,
                "embedding_model": if offline { "deterministic" } else { embed_model },
            }),
        )?;
    }
    let store = MatryoshkaStore::open(db)?;
    let summary = if offline {
        FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default())
            .with_parser_config(parser_config)
            .update_repo(repo_root)?
    } else {
        let enricher = MlxChatEnricher::new(base_url, api_key).with_model(chat_model.to_string());
        let embedder = EndpointEmbedder::new(base_url, api_key, embed_model.to_string());
        FullIndexer::new(store, enricher, embedder)
            .with_parser_config(parser_config)
            .update_repo(repo_root)?
    };
    if let Some(log) = log.as_deref_mut() {
        log.event(
            "update_completed",
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
            }),
        )?;
    }
    Ok(summary)
}

fn run_rebuild_semantic_once(
    repo_root: &Path,
    db: &Path,
    offline: bool,
    base_url: &str,
    api_key: &str,
    embed_model: &str,
    mut log: Option<&mut CommandLog>,
) -> Result<SemanticRebuildSummary> {
    if let Some(log) = log.as_deref_mut() {
        log.event(
            "semantic_rebuild_started",
            json!({
                "repo_root": repo_root,
                "db": db,
                "offline": offline,
                "embedding_model": if offline { "deterministic" } else { embed_model },
            }),
        )?;
    }
    let store = MatryoshkaStore::open(db)?;
    let summary = if offline {
        FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default())
            .rebuild_semantic_index(repo_root)?
    } else {
        FullIndexer::new(
            store,
            HeuristicEnricher,
            EndpointEmbedder::new(base_url, api_key, embed_model.to_string()),
        )
        .rebuild_semantic_index(repo_root)?
    };
    if let Some(log) = log.as_deref_mut() {
        log.event(
            "semantic_rebuild_completed",
            semantic_rebuild_summary_json(&summary),
        )?;
    }
    Ok(summary)
}

#[allow(clippy::too_many_arguments)]
fn run_prewarm_once(
    db: &Path,
    offline: bool,
    base_url: &str,
    api_key: &str,
    embed_model: &str,
    queries: &[String],
    limit: usize,
    late_interaction: bool,
    mut log: Option<&mut CommandLog>,
) -> Result<SearchPrewarmSummary> {
    if let Some(log) = log.as_deref_mut() {
        log.event(
            "prewarm_started",
            json!({
                "db": db,
                "offline": offline,
                "embedding_model": if offline { "deterministic" } else { embed_model },
                "limit": limit,
                "query_count": queries.len(),
                "late_interaction": late_interaction,
            }),
        )?;
    }
    let store = MatryoshkaStore::open(db)?;
    let summary = if offline {
        SearchEngine::new(store, DeterministicEmbedder::default())
            .with_late_interaction(late_interaction)
            .prewarm(queries, limit)?
    } else {
        SearchEngine::new(
            store,
            EndpointEmbedder::new(base_url, api_key, embed_model.to_string()),
        )
        .with_late_interaction(late_interaction)
        .prewarm(queries, limit)?
    };
    if let Some(log) = log.as_deref_mut() {
        let retrieval_stats = MatryoshkaStore::open(db)?.retrieval_index_stats()?;
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

fn indexed_file_count(db: &Path) -> Result<usize> {
    Ok(MatryoshkaStore::open(db)?.load_all_files()?.len())
}

fn existing_card_gap_count(db: &Path) -> Result<usize> {
    Ok(MatryoshkaStore::open(db)?
        .load_card_summaries()?
        .iter()
        .filter(|row| row.is_empty)
        .count())
}

fn existing_retrieval_needs_rebuild(db: &Path, late_interaction: bool) -> Result<bool> {
    Ok(retrieval_needs_rebuild(
        &retrieval_report_from_stats(
            MatryoshkaStore::open(db)?.retrieval_index_stats()?,
            late_interaction,
        ),
        late_interaction,
    ))
}

fn artifact_gap_count(report: &ArtifactQualityReport) -> usize {
    report.file_cards_empty_summary
        + report.folder_cards_empty_summary
        + usize::from(!report.repo_card_has_summary)
}

fn retrieval_needs_rebuild(report: &RetrievalIndexReport, late_interaction: bool) -> bool {
    !retrieval_is_ready(report, late_interaction)
}

fn retrieval_is_ready(report: &RetrievalIndexReport, late_interaction: bool) -> bool {
    report.semantic_records > 0
        && report.embedded_records > 0
        && report.fts_records > 0
        && (!late_interaction || report.records_with_late_vectors > 0)
}

fn retrieval_report_from_stats(
    stats: RetrievalIndexStats,
    late_interaction: bool,
) -> RetrievalIndexReport {
    RetrievalIndexReport {
        semantic_records: stats.semantic_records,
        embedded_records: stats.embedded_records,
        fts_records: stats.fts_records,
        late_vector_rows: stats.late_vector_rows,
        records_with_late_vectors: stats.records_with_late_vectors,
        late_interaction_enabled: late_interaction,
    }
}

fn logs_dir(db: &Path) -> PathBuf {
    db.parent()
        .unwrap_or_else(|| Path::new(MATRYOSHKA_DIR))
        .join("logs")
}

fn ready_marker_path(db: &Path) -> PathBuf {
    db.parent()
        .unwrap_or_else(|| Path::new(MATRYOSHKA_DIR))
        .join(READY_MARKER_FILE)
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

fn ensure_single_reranker(chat_rerank: bool, omlx_rerank: bool) -> Result<()> {
    if chat_rerank && omlx_rerank {
        anyhow::bail!("choose either --rerank or --omlx-rerank, not both");
    }
    Ok(())
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

fn print_progress_jsonl(event: MatryoshkaProgressEvent) {
    println!(
        "{}",
        serde_json::to_string(&event).expect("progress event should serialize")
    );
}

fn parser_config(ignore: Vec<String>) -> ParserConfig {
    ParserConfig::default().with_ignored_paths(ignore)
}

fn print_prepare_summary(summary: &PrepareSummary) {
    if summary.status == "ready" {
        println!("Jesco is ready.");
    } else {
        println!("Matryoshka needs attention.");
    }
    println!();
    println!(
        "project_map: {}",
        if artifact_gap_count(&summary.artifact_quality) == 0 {
            "ready"
        } else {
            "needs_attention"
        }
    );
    println!(
        "search: {}",
        if retrieval_is_ready(
            &summary.retrieval_index,
            summary.retrieval_index.late_interaction_enabled
        ) {
            "ready"
        } else {
            "needs_refresh"
        }
    );
    println!("files: {}", summary.file_count);
    println!("folders: {}", summary.folder_count);
    println!("symbols: {}", summary.symbol_count);
    println!("changed_files: {}", summary.changed_files);
    println!("removed_files: {}", summary.removed_files);
    println!(
        "map_gaps: {}",
        artifact_gap_count(&summary.artifact_quality)
    );
    println!("prepared_queries: {}", summary.prewarm.query_count);
    println!("prepared_hits: {}", summary.prewarm.warmed_hit_count);
    println!("actions_taken: {}", summary.actions_taken.join(", "));
    println!("db: {}", summary.db.display());
    println!("ready_marker: {}", summary.ready_marker.display());
    println!("logs: {}", summary.logs_dir.display());
}

fn print_index_summary(summary: IndexSummary) {
    println!("files: {}", summary.file_count);
    println!("folders: {}", summary.folder_count);
    println!("symbols: {}", summary.symbol_count);
    println!("semantic_records: {}", summary.semantic_record_count);
    print_artifact_quality(&summary.artifact_quality);
    print_retrieval_index(&summary.retrieval_index);
    println!("embedding_model: {}", summary.embedding_model);
}

fn print_update_summary(summary: UpdateSummary) {
    println!("files: {}", summary.file_count);
    println!("folders: {}", summary.folder_count);
    println!("symbols: {}", summary.symbol_count);
    println!("semantic_records: {}", summary.semantic_record_count);
    print_artifact_quality(&summary.artifact_quality);
    print_retrieval_index(&summary.retrieval_index);
    println!("changed_files: {}", summary.changed_files);
    println!("removed_files: {}", summary.removed_files);
    println!("changed_folders: {}", summary.changed_folders);
    println!("repo_card_updated: {}", summary.repo_card_updated);
    println!("embedding_model: {}", summary.embedding_model);
}

fn print_semantic_rebuild_summary(summary: SemanticRebuildSummary) {
    println!("semantic_records: {}", summary.semantic_record_count);
    println!("file_card_records: {}", summary.file_card_record_count);
    println!("folder_card_records: {}", summary.folder_card_record_count);
    println!("repo_card_records: {}", summary.repo_card_record_count);
    print_artifact_quality(&summary.artifact_quality);
    print_retrieval_index(&summary.retrieval_index);
    println!("embedding_model: {}", summary.embedding_model);
}

fn print_card_summaries(db: &Path, rows: &[CardSummaryRow], empty_only: bool) {
    if empty_only {
        println!("# Matryoshka Empty Card Summaries");
    } else {
        println!("# Matryoshka Card Summaries");
    }
    println!();
    println!("- Database: `{}`", db.display());
    println!("- Cards returned: {}", rows.len());
    println!(
        "- File cards: {}",
        rows.iter().filter(|row| row.card_type == "file").count()
    );
    println!(
        "- Folder cards: {}",
        rows.iter().filter(|row| row.card_type == "folder").count()
    );
    println!(
        "- Repo cards: {}",
        rows.iter().filter(|row| row.card_type == "repo").count()
    );
    println!(
        "- Empty summaries: {}",
        rows.iter().filter(|row| row.is_empty).count()
    );
    println!();

    if rows.is_empty() {
        if empty_only {
            println!("No empty card summaries found.");
        } else {
            println!("No card summaries found in this database.");
        }
        return;
    }

    let mut current_type = "";
    for row in rows {
        if row.card_type != current_type {
            current_type = &row.card_type;
            println!("## {} Cards", card_type_title(current_type));
            println!();
        }
        println!("### `{}`", row.id);
        println!();
        println!("- Type: {}", row.card_type);
        println!(
            "- Summary status: {}",
            if row.is_empty { "empty" } else { "present" }
        );
        println!();
        println!("Summary:");
        println!();
        if row.is_empty {
            println!("_No summary is currently stored for this card._");
        } else {
            print_markdown_quote(&row.summary);
        }
        println!();
    }
}

fn card_type_title(card_type: &str) -> &'static str {
    match card_type {
        "file" => "File",
        "folder" => "Folder",
        "repo" => "Repo",
        _ => "Unknown",
    }
}

fn print_markdown_quote(text: &str) {
    for line in text.lines() {
        if line.trim().is_empty() {
            println!(">");
        } else {
            println!("> {}", line);
        }
    }
}

fn print_artifact_quality(report: &ArtifactQualityReport) {
    println!(
        "file_card_summaries: {}/{}",
        report.file_cards_with_summary, report.file_cards
    );
    println!(
        "folder_card_summaries: {}/{}",
        report.folder_cards_with_summary, report.folder_cards
    );
    println!("repo_card_has_summary: {}", report.repo_card_has_summary);
    if !report.empty_file_summary_samples.is_empty() {
        println!(
            "empty_file_summary_samples: {}",
            report.empty_file_summary_samples.join(", ")
        );
    }
    if !report.empty_folder_summary_samples.is_empty() {
        println!(
            "empty_folder_summary_samples: {}",
            report.empty_folder_summary_samples.join(", ")
        );
    }
}

fn print_retrieval_index(report: &RetrievalIndexReport) {
    println!("embedded_records: {}", report.embedded_records);
    println!("fts_records: {}", report.fts_records);
    println!("late_vector_rows: {}", report.late_vector_rows);
    println!(
        "records_with_late_vectors: {}",
        report.records_with_late_vectors
    );
}

fn prepare_summary_json(summary: &PrepareSummary) -> serde_json::Value {
    json!({
        "status": summary.status,
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
            "status": if retrieval_is_ready(
                &summary.retrieval_index,
                summary.retrieval_index.late_interaction_enabled,
            ) {
                "ready"
            } else {
                "needs_refresh"
            },
            "semantic_records": summary.semantic_record_count,
            "embedded_records": summary.retrieval_index.embedded_records,
            "fts_records": summary.retrieval_index.fts_records,
            "late_vector_rows": summary.retrieval_index.late_vector_rows,
            "records_with_late_vectors": summary.retrieval_index.records_with_late_vectors,
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

fn retrieval_report_json(report: &RetrievalIndexReport) -> serde_json::Value {
    json!({
        "semantic_records": report.semantic_records,
        "embedded_records": report.embedded_records,
        "fts_records": report.fts_records,
        "late_vector_rows": report.late_vector_rows,
        "records_with_late_vectors": report.records_with_late_vectors,
        "late_interaction_enabled": report.late_interaction_enabled,
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

fn index_summary_json(summary: &IndexSummary) -> serde_json::Value {
    json!({
        "files": summary.file_count,
        "folders": summary.folder_count,
        "symbols": summary.symbol_count,
        "semantic_records": summary.semantic_record_count,
        "artifact_quality": &summary.artifact_quality,
        "retrieval_index": &summary.retrieval_index,
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
