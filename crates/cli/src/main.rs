use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use matryoshka_embed_client::{DeterministicEmbedder, EndpointEmbedder};
use matryoshka_enricher::{HeuristicEnricher, MlxChatEnricher};
use matryoshka_indexer::{
    FullIndexer, IndexSummary, MatryoshkaProgressEvent, SemanticRebuildSummary, UpdateSummary,
};
use matryoshka_parser::ParserConfig;
use matryoshka_read_api::{ReadApi, ReadPackMode};
use matryoshka_search::{EndpointReranker, OmlxReranker, SearchEngine, default_prewarm_queries};
use matryoshka_store_sqlite::MatryoshkaStore;
use matryoshka_watcher::RepoWatcher;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:44445";
const DEFAULT_API_KEY: &str = "2508";
const DEFAULT_EMBED_MODEL: &str = "mlx-community--embeddinggemma-300m-bf16";
const DEFAULT_CHAT_MODEL: &str = "MercuriusDream--Qwen3.5-4B-MLX-mxfp8";
const DEFAULT_OMLX_RERANK_MODEL: &str = "mlx-community--Qwen3-Reranker-0.6B-mxfp8";

#[derive(Debug, Parser)]
#[command(name = "matryoshka-rs")]
#[command(about = "Rust-first Matryoshka code intelligence core")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Index {
        repo_root: PathBuf,
        #[arg(long)]
        db: PathBuf,
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
    Update {
        repo_root: PathBuf,
        #[arg(long)]
        db: PathBuf,
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
        db: PathBuf,
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
    },
    RebuildSemantic {
        repo_root: PathBuf,
        #[arg(long)]
        db: PathBuf,
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
        db: PathBuf,
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
        db: PathBuf,
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
        db: PathBuf,
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
    },
    Read {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        repo_root: PathBuf,
        file: String,
    },
    ReadBundle {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        repo_root: PathBuf,
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
        } => {
            let store = MatryoshkaStore::open(&db)?;
            let parser_config = parser_config(ignore);
            if offline {
                let indexer =
                    FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default())
                        .with_parser_config(parser_config);
                let summary = if progress_jsonl {
                    indexer.index_repo_with_progress(repo_root, print_progress_jsonl)?
                } else {
                    indexer.index_repo(repo_root)?
                };
                if !progress_jsonl {
                    print_index_summary(summary);
                }
            } else {
                let enricher = MlxChatEnricher::new(&base_url, &api_key).with_model(chat_model);
                let embedder = EndpointEmbedder::new(&base_url, &api_key, embed_model);
                let indexer =
                    FullIndexer::new(store, enricher, embedder).with_parser_config(parser_config);
                let summary = if progress_jsonl {
                    indexer.index_repo_with_progress(repo_root, print_progress_jsonl)?
                } else {
                    indexer.index_repo(repo_root)?
                };
                if !progress_jsonl {
                    print_index_summary(summary);
                }
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
        } => {
            let parser_config = parser_config(ignore);
            let mut watcher = RepoWatcher::new(&repo_root)?
                .with_parser_config(parser_config.clone())?
                .with_poll_interval(Duration::from_millis(interval_ms))
                .with_debounce_window(Duration::from_millis(debounce_ms));
            println!(
                "watching {} every {}ms with {}ms debounce",
                repo_root.display(),
                interval_ms,
                debounce_ms
            );
            loop {
                if let Some(batch) = watcher.poll()? {
                    println!(
                        "change batch detected: changed={} added={} removed={}",
                        batch.changed_paths.len(),
                        batch.added_paths.len(),
                        batch.removed_paths.len()
                    );
                    let store = MatryoshkaStore::open(&db)?;
                    if offline {
                        let indexer = FullIndexer::new(
                            store,
                            HeuristicEnricher,
                            DeterministicEmbedder::default(),
                        )
                        .with_parser_config(parser_config.clone());
                        print_update_summary(indexer.update_repo(&repo_root)?);
                    } else {
                        let enricher = MlxChatEnricher::new(&base_url, &api_key)
                            .with_model(chat_model.clone());
                        let embedder =
                            EndpointEmbedder::new(&base_url, &api_key, embed_model.clone());
                        let indexer = FullIndexer::new(store, enricher, embedder)
                            .with_parser_config(parser_config.clone());
                        print_update_summary(indexer.update_repo(&repo_root)?);
                    }
                }
                thread::sleep(watcher.poll_interval());
            }
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
            offline,
            base_url,
            api_key,
            embed_model,
            limit,
            queries,
            no_late_interaction,
        } => {
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
                SearchEngine::new(store, EndpointEmbedder::new(base_url, api_key, embed_model))
                    .with_late_interaction(late_interaction)
                    .prewarm(&queries, limit)?
            };
            println!("fts_records: {}", summary.fts_record_count);
            println!("queries: {}", summary.query_count);
            println!("warmed_hits: {}", summary.warmed_hit_count);
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
            if !progress_jsonl {
                print_semantic_rebuild_summary(summary);
            }
        }
        Command::Read {
            db,
            repo_root,
            file,
        } => {
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
    }
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

fn print_index_summary(summary: IndexSummary) {
    println!("files: {}", summary.file_count);
    println!("folders: {}", summary.folder_count);
    println!("symbols: {}", summary.symbol_count);
    println!("semantic_records: {}", summary.semantic_record_count);
    println!("embedding_model: {}", summary.embedding_model);
}

fn print_update_summary(summary: UpdateSummary) {
    println!("files: {}", summary.file_count);
    println!("folders: {}", summary.folder_count);
    println!("symbols: {}", summary.symbol_count);
    println!("semantic_records: {}", summary.semantic_record_count);
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
    println!("embedding_model: {}", summary.embedding_model);
}
