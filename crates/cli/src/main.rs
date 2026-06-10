use anyhow::Result;
use clap::{Parser, Subcommand};
use matryoshka_embed_client::{DeterministicEmbedder, EndpointEmbedder};
use matryoshka_enricher::{HeuristicEnricher, MlxChatEnricher};
use matryoshka_indexer::{FullIndexer, IndexSummary, UpdateSummary};
use matryoshka_read_api::ReadApi;
use matryoshka_search::SearchEngine;
use matryoshka_store_sqlite::MatryoshkaStore;
use matryoshka_watcher::RepoWatcher;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:44445";
const DEFAULT_API_KEY: &str = "2508";
const DEFAULT_EMBED_MODEL: &str = "mlx-community--embeddinggemma-300m-bf16";
const DEFAULT_CHAT_MODEL: &str = "MercuriusDream--Qwen3.5-4B-MLX-mxfp8";

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
        #[arg(long, default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
        #[arg(long, default_value = DEFAULT_CHAT_MODEL)]
        chat_model: String,
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
        #[arg(long, default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
        #[arg(long, default_value = DEFAULT_CHAT_MODEL)]
        chat_model: String,
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
        #[arg(long, default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
        #[arg(long, default_value = DEFAULT_CHAT_MODEL)]
        chat_model: String,
        #[arg(long, default_value_t = 2_000)]
        interval_ms: u64,
        #[arg(long, default_value_t = 3_000)]
        debounce_ms: u64,
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
        #[arg(long, default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
    },
    Read {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        repo_root: PathBuf,
        file: String,
    },
    ReadMore {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        repo_root: PathBuf,
        file: String,
    },
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
        } => {
            let store = MatryoshkaStore::open(&db)?;
            if offline {
                let indexer =
                    FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default());
                let summary = indexer.index_repo(repo_root)?;
                print_index_summary(summary);
            } else {
                let enricher = MlxChatEnricher::new(&base_url, &api_key).with_model(chat_model);
                let embedder = EndpointEmbedder::new(&base_url, &api_key, embed_model);
                let indexer = FullIndexer::new(store, enricher, embedder);
                let summary = indexer.index_repo(repo_root)?;
                print_index_summary(summary);
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
        } => {
            let store = MatryoshkaStore::open(&db)?;
            if offline {
                let indexer =
                    FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default());
                let summary = indexer.update_repo(repo_root)?;
                print_update_summary(summary);
            } else {
                let enricher = MlxChatEnricher::new(&base_url, &api_key).with_model(chat_model);
                let embedder = EndpointEmbedder::new(&base_url, &api_key, embed_model);
                let indexer = FullIndexer::new(store, enricher, embedder);
                let summary = indexer.update_repo(repo_root)?;
                print_update_summary(summary);
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
        } => {
            let mut watcher = RepoWatcher::new(&repo_root)?
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
                        );
                        print_update_summary(indexer.update_repo(&repo_root)?);
                    } else {
                        let enricher =
                            MlxChatEnricher::new(&base_url, &api_key).with_model(chat_model.clone());
                        let embedder =
                            EndpointEmbedder::new(&base_url, &api_key, embed_model.clone());
                        let indexer = FullIndexer::new(store, enricher, embedder);
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
        } => {
            let store = MatryoshkaStore::open(&db)?;
            let hits = if offline {
                SearchEngine::new(store, DeterministicEmbedder::default()).search(&query, limit)?
            } else {
                SearchEngine::new(store, EndpointEmbedder::new(base_url, api_key, embed_model))
                    .search(&query, limit)?
            };
            println!("{}", serde_json::to_string_pretty(&hits)?);
        }
        Command::Read {
            db,
            repo_root,
            file,
        } => {
            let read = ReadApi::new(MatryoshkaStore::open(&db)?, repo_root);
            println!("{}", serde_json::to_string_pretty(&read.read(&file)?)?);
        }
        Command::ReadMore {
            db,
            repo_root,
            file,
        } => {
            let read = ReadApi::new(MatryoshkaStore::open(&db)?, repo_root);
            println!("{}", serde_json::to_string_pretty(&read.read_more(&file)?)?);
        }
    }
    Ok(())
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
