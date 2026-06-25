use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use matryoshka::{
    EnrichmentOptions as ApiEnrichmentOptions, EnrichmentSummary, Matryoshka, MatryoshkaConfig,
    PrepareOptions as ApiPrepareOptions, PrepareSummary as ApiPrepareSummary,
    enrichment_summary_json,
};
use matryoshka_embed_client::EndpointEmbedder;
use matryoshka_enricher::{MlxChatEnricher, MlxChunkSummarizer};
use matryoshka_indexer::{
    ArtifactQualityReport, EnrichmentReadinessReport, FullIndexer, IndexSummary,
    MatryoshkaProgressEvent, RetrievalConfig, RetrievalIndexReport, RetrievalPrimary,
    SemanticRebuildSummary, UpdateSummary,
};
use matryoshka_parser::{ParserConfig, SourceParser};
use matryoshka_read_api::{ReadApi, ReadPackMode};
use matryoshka_search::{
    EndpointReranker, OmlxReranker, SearchEngine, SearchPrewarmSummary, SearchResultGranularity,
    default_prewarm_queries,
};
use matryoshka_store_sqlite::{CardSummaryRow, MatryoshkaStore};
use matryoshka_watcher::RepoWatcher;
use serde_json::{Value, json};
use std::collections::BTreeSet;
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
const DEFAULT_CHUNK_SUMMARY_MODEL: &str = "srswti--bodega-raptor-90m";
const DEFAULT_CHUNK_SUMMARY_CONCURRENCY: usize = 6;
const MATRYOSHKA_DIR: &str = ".matryoshka";
const DEFAULT_DB_FILE: &str = "matryoshka.db";
const WATCH_PID_FILE: &str = "watch.pid";

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
        #[arg(long, default_value_t = false)]
        progress_jsonl: bool,
        #[arg(long = "chunk-summary-model", default_value = DEFAULT_CHUNK_SUMMARY_MODEL)]
        chunk_summary_model: String,
        #[arg(long = "chunk-summary-concurrency", default_value_t = DEFAULT_CHUNK_SUMMARY_CONCURRENCY)]
        chunk_summary_concurrency: usize,
        #[arg(long, default_value_t = false)]
        no_chunk_summaries: bool,
        #[arg(long, default_value_t = false)]
        enrich_now: bool,
        #[arg(long = "retrieval-primary", value_enum, default_value_t = CliRetrievalPrimary::Hybrid)]
        retrieval_primary: CliRetrievalPrimary,
        #[arg(long = "enable-dense", default_value_t = false)]
        enable_dense: bool,
        #[arg(
            long = "disable-dense",
            visible_alias = "no-dense-embeddings",
            default_value_t = false
        )]
        disable_dense: bool,
        #[arg(long = "dense-fallback", default_value_t = false)]
        dense_fallback: bool,
        #[arg(long = "no-dense-fallback", default_value_t = false)]
        no_dense_fallback: bool,
    },
    Index {
        repo_root: PathBuf,
        #[arg(long)]
        db: Option<PathBuf>,
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
        #[arg(long = "chunk-summary-model", default_value = DEFAULT_CHUNK_SUMMARY_MODEL)]
        chunk_summary_model: String,
        #[arg(long = "chunk-summary-concurrency", default_value_t = DEFAULT_CHUNK_SUMMARY_CONCURRENCY)]
        chunk_summary_concurrency: usize,
        #[arg(long, default_value_t = false)]
        no_chunk_summaries: bool,
        #[arg(long = "retrieval-primary", value_enum, default_value_t = CliRetrievalPrimary::Hybrid)]
        retrieval_primary: CliRetrievalPrimary,
        #[arg(long = "enable-dense", default_value_t = false)]
        enable_dense: bool,
        #[arg(
            long = "disable-dense",
            visible_alias = "no-dense-embeddings",
            default_value_t = false
        )]
        disable_dense: bool,
        #[arg(long = "dense-fallback", default_value_t = false)]
        dense_fallback: bool,
        #[arg(long = "no-dense-fallback", default_value_t = false)]
        no_dense_fallback: bool,
    },
    Update {
        repo_root: PathBuf,
        #[arg(long)]
        db: Option<PathBuf>,
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
        #[arg(long = "chunk-summary-model", default_value = DEFAULT_CHUNK_SUMMARY_MODEL)]
        chunk_summary_model: String,
        #[arg(long = "chunk-summary-concurrency", default_value_t = DEFAULT_CHUNK_SUMMARY_CONCURRENCY)]
        chunk_summary_concurrency: usize,
        #[arg(long, default_value_t = false)]
        no_chunk_summaries: bool,
        #[arg(long = "retrieval-primary", value_enum, default_value_t = CliRetrievalPrimary::Hybrid)]
        retrieval_primary: CliRetrievalPrimary,
        #[arg(long = "enable-dense", default_value_t = false)]
        enable_dense: bool,
        #[arg(
            long = "disable-dense",
            visible_alias = "no-dense-embeddings",
            default_value_t = false
        )]
        disable_dense: bool,
        #[arg(long = "dense-fallback", default_value_t = false)]
        dense_fallback: bool,
        #[arg(long = "no-dense-fallback", default_value_t = false)]
        no_dense_fallback: bool,
    },
    Watch {
        repo_root: PathBuf,
        #[arg(long)]
        db: Option<PathBuf>,
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
        #[arg(long = "retrieval-primary", value_enum, default_value_t = CliRetrievalPrimary::Hybrid)]
        retrieval_primary: CliRetrievalPrimary,
        #[arg(long = "enable-dense", default_value_t = false)]
        enable_dense: bool,
        #[arg(
            long = "disable-dense",
            visible_alias = "no-dense-embeddings",
            default_value_t = false
        )]
        disable_dense: bool,
        #[arg(long = "dense-fallback", default_value_t = false)]
        dense_fallback: bool,
        #[arg(long = "no-dense-fallback", default_value_t = false)]
        no_dense_fallback: bool,
    },
    RebuildSemantic {
        repo_root: PathBuf,
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long, default_value = DEFAULT_BASE_URL)]
        base_url: String,
        #[arg(long, default_value = DEFAULT_API_KEY)]
        api_key: String,
        #[arg(long = "embedding-model", visible_alias = "embed-model", default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
        #[arg(long, default_value_t = false)]
        progress_jsonl: bool,
        #[arg(long = "retrieval-primary", value_enum, default_value_t = CliRetrievalPrimary::Hybrid)]
        retrieval_primary: CliRetrievalPrimary,
        #[arg(long = "enable-dense", default_value_t = false)]
        enable_dense: bool,
        #[arg(
            long = "disable-dense",
            visible_alias = "no-dense-embeddings",
            default_value_t = false
        )]
        disable_dense: bool,
        #[arg(long = "dense-fallback", default_value_t = false)]
        dense_fallback: bool,
        #[arg(long = "no-dense-fallback", default_value_t = false)]
        no_dense_fallback: bool,
    },
    Search {
        #[arg(long)]
        db: Option<PathBuf>,
        query: String,
        #[arg(long, default_value_t = 8)]
        limit: usize,
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
        #[arg(long = "retrieval-primary", value_enum, default_value_t = CliRetrievalPrimary::Hybrid)]
        retrieval_primary: CliRetrievalPrimary,
        #[arg(long = "enable-dense", default_value_t = false)]
        enable_dense: bool,
        #[arg(
            long = "disable-dense",
            visible_alias = "no-dense-embeddings",
            default_value_t = false
        )]
        disable_dense: bool,
        #[arg(long = "dense-fallback", default_value_t = false)]
        dense_fallback: bool,
        #[arg(long = "no-dense-fallback", default_value_t = false)]
        no_dense_fallback: bool,
        #[arg(long = "result-granularity", value_enum, default_value_t = CliSearchResultGranularity::File)]
        result_granularity: CliSearchResultGranularity,
        #[arg(long = "no-collapse", default_value_t = false)]
        no_collapse: bool,
        #[arg(
            long = "compact",
            visible_alias = "hide-match-details",
            visible_alias = "no-match-details",
            default_value_t = false
        )]
        compact: bool,
    },
    Op {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(value_enum)]
        task: AgentTask,
        query: String,
        #[arg(long, default_value_t = 8)]
        limit: usize,
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
        #[arg(long = "retrieval-primary", value_enum, default_value_t = CliRetrievalPrimary::Hybrid)]
        retrieval_primary: CliRetrievalPrimary,
        #[arg(long = "enable-dense", default_value_t = false)]
        enable_dense: bool,
        #[arg(
            long = "disable-dense",
            visible_alias = "no-dense-embeddings",
            default_value_t = false
        )]
        disable_dense: bool,
        #[arg(long = "dense-fallback", default_value_t = false)]
        dense_fallback: bool,
        #[arg(long = "no-dense-fallback", default_value_t = false)]
        no_dense_fallback: bool,
        #[arg(long = "result-granularity", value_enum, default_value_t = CliSearchResultGranularity::File)]
        result_granularity: CliSearchResultGranularity,
        #[arg(long = "no-collapse", default_value_t = false)]
        no_collapse: bool,
        #[arg(
            long = "compact",
            visible_alias = "hide-match-details",
            visible_alias = "no-match-details",
            default_value_t = false
        )]
        compact: bool,
    },
    Prewarm {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long)]
        repo_root: Option<PathBuf>,
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
        #[arg(long = "retrieval-primary", value_enum, default_value_t = CliRetrievalPrimary::Hybrid)]
        retrieval_primary: CliRetrievalPrimary,
        #[arg(long = "enable-dense", default_value_t = false)]
        enable_dense: bool,
        #[arg(
            long = "disable-dense",
            visible_alias = "no-dense-embeddings",
            default_value_t = false
        )]
        disable_dense: bool,
        #[arg(long = "dense-fallback", default_value_t = false)]
        dense_fallback: bool,
        #[arg(long = "no-dense-fallback", default_value_t = false)]
        no_dense_fallback: bool,
    },
    Read {
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long)]
        repo_root: Option<PathBuf>,
        #[arg(
            long = "chunks",
            visible_alias = "include-chunks",
            default_value_t = false
        )]
        chunks: bool,
        #[arg(
            long = "json",
            help = "Emit legacy full symbol objects instead of compact symbol outlines",
            default_value_t = false
        )]
        json: bool,
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
        #[arg(long = "retrieval-primary", value_enum, default_value_t = CliRetrievalPrimary::Hybrid)]
        retrieval_primary: CliRetrievalPrimary,
        #[arg(long = "enable-dense", default_value_t = false)]
        enable_dense: bool,
        #[arg(
            long = "disable-dense",
            visible_alias = "no-dense-embeddings",
            default_value_t = false
        )]
        disable_dense: bool,
        #[arg(long = "dense-fallback", default_value_t = false)]
        dense_fallback: bool,
        #[arg(long = "no-dense-fallback", default_value_t = false)]
        no_dense_fallback: bool,
    },
    Enrich {
        repo_root: PathBuf,
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long, default_value = DEFAULT_BASE_URL)]
        base_url: String,
        #[arg(long, default_value = DEFAULT_API_KEY)]
        api_key: String,
        #[arg(long = "embedding-model", visible_alias = "embed-model", default_value = DEFAULT_EMBED_MODEL)]
        embed_model: String,
        #[arg(long = "model", visible_alias = "chat-model", default_value = DEFAULT_CHAT_MODEL)]
        chat_model: String,
        #[arg(long = "chunk-summary-model", default_value = DEFAULT_CHUNK_SUMMARY_MODEL)]
        chunk_summary_model: String,
        #[arg(long = "chunk-summary-concurrency", default_value_t = DEFAULT_CHUNK_SUMMARY_CONCURRENCY)]
        chunk_summary_concurrency: usize,
        #[arg(long = "max-files", default_value_t = 1)]
        max_files: usize,
        #[arg(long, default_value_t = false)]
        status: bool,
        #[arg(long, default_value_t = false)]
        json: bool,
        #[arg(long, default_value_t = false)]
        progress_jsonl: bool,
        #[arg(long, default_value_t = false)]
        no_chunk_summaries: bool,
        #[arg(long = "retrieval-primary", value_enum, default_value_t = CliRetrievalPrimary::Hybrid)]
        retrieval_primary: CliRetrievalPrimary,
        #[arg(long = "enable-dense", default_value_t = false)]
        enable_dense: bool,
        #[arg(
            long = "disable-dense",
            visible_alias = "no-dense-embeddings",
            default_value_t = false
        )]
        disable_dense: bool,
        #[arg(long = "dense-fallback", default_value_t = false)]
        dense_fallback: bool,
        #[arg(long = "no-dense-fallback", default_value_t = false)]
        no_dense_fallback: bool,
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
    /// Parse a repo and dump extracted code chunks (parser-only, no embeddings).
    /// Used to verify Milestone 1 chunk/doc extraction in isolation.
    Chunks {
        repo_root: PathBuf,
        #[arg(long = "ignore", value_name = "PATH")]
        ignore: Vec<String>,
        /// Only emit chunks with this summary source (doc_comment, docstring, empty, all).
        #[arg(long, default_value = "all")]
        source: String,
        /// Emit pretty JSON instead of the default table.
        #[arg(long, default_value_t = false)]
        json: bool,
        /// Also print the full code body for each chunk (verbose).
        #[arg(long, default_value_t = false)]
        with_code: bool,
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliRetrievalPrimary {
    Fts,
    Splade,
    Dense,
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CliSearchResultGranularity {
    File,
    Record,
    Symbol,
    Chunk,
}

impl From<CliSearchResultGranularity> for SearchResultGranularity {
    fn from(value: CliSearchResultGranularity) -> Self {
        match value {
            CliSearchResultGranularity::File => SearchResultGranularity::File,
            CliSearchResultGranularity::Record => SearchResultGranularity::Record,
            CliSearchResultGranularity::Symbol => SearchResultGranularity::Symbol,
            CliSearchResultGranularity::Chunk => SearchResultGranularity::Chunk,
        }
    }
}

impl From<CliRetrievalPrimary> for RetrievalPrimary {
    fn from(value: CliRetrievalPrimary) -> Self {
        match value {
            CliRetrievalPrimary::Fts => RetrievalPrimary::Fts,
            CliRetrievalPrimary::Splade => RetrievalPrimary::Splade,
            CliRetrievalPrimary::Dense => RetrievalPrimary::Dense,
            CliRetrievalPrimary::Hybrid => RetrievalPrimary::Hybrid,
        }
    }
}

fn resolve_search_result_granularity(
    granularity: CliSearchResultGranularity,
    no_collapse: bool,
) -> Result<SearchResultGranularity> {
    if no_collapse
        && !matches!(
            granularity,
            CliSearchResultGranularity::File | CliSearchResultGranularity::Record
        )
    {
        anyhow::bail!(
            "--no-collapse conflicts with --result-granularity {granularity:?}; use one result granularity selector"
        );
    }
    if no_collapse {
        Ok(SearchResultGranularity::Record)
    } else {
        Ok(granularity.into())
    }
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

fn resolve_retrieval_config(
    primary: CliRetrievalPrimary,
    enable_dense: bool,
    disable_dense: bool,
    dense_fallback: bool,
    no_dense_fallback: bool,
) -> Result<RetrievalConfig> {
    if enable_dense && disable_dense {
        anyhow::bail!("choose either --enable-dense or --disable-dense, not both");
    }
    if dense_fallback && no_dense_fallback {
        anyhow::bail!("choose either --dense-fallback or --no-dense-fallback, not both");
    }
    if disable_dense && dense_fallback {
        anyhow::bail!(
            "--dense-fallback requires dense embeddings; remove --disable-dense or use --no-dense-fallback"
        );
    }

    let primary = RetrievalPrimary::from(primary);
    if matches!(primary, RetrievalPrimary::Dense) && disable_dense {
        anyhow::bail!(
            "--retrieval-primary dense requires dense embeddings; remove --disable-dense or choose another primary"
        );
    }
    let dense_enabled = if disable_dense {
        false
    } else if enable_dense || dense_fallback {
        true
    } else {
        !matches!(primary, RetrievalPrimary::Fts)
    };
    let dense_fallback_enabled = if no_dense_fallback || !dense_enabled {
        false
    } else if dense_fallback {
        true
    } else {
        dense_enabled
    };

    Ok(RetrievalConfig {
        primary,
        dense_enabled,
        dense_fallback_enabled,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Prepare {
            repo_root,
            db,
            base_url,
            api_key,
            embed_model,
            chat_model,
            ignore,
            limit,
            queries,
            no_late_interaction,
            json,
            progress_jsonl,
            chunk_summary_model,
            chunk_summary_concurrency,
            no_chunk_summaries,
            enrich_now,
            retrieval_primary,
            enable_dense,
            disable_dense,
            dense_fallback,
            no_dense_fallback,
        } => {
            let retrieval_config = resolve_retrieval_config(
                retrieval_primary,
                enable_dense,
                disable_dense,
                dense_fallback,
                no_dense_fallback,
            )?;
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            let summary = run_prepare_via_api(
                PrepareOptions {
                    repo_root,
                    db,
                    base_url,
                    api_key,
                    embed_model,
                    chat_model,
                    ignore,
                    limit,
                    queries,
                    late_interaction: !no_late_interaction,
                    retrieval_config,
                    chunk_summary_model,
                    chunk_summary_concurrency,
                    no_chunk_summaries,
                    enrich_now,
                },
                progress_jsonl,
            )?;
            if !progress_jsonl {
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&prepare_summary_json(&summary))?
                    );
                } else {
                    print_prepare_summary(&summary);
                }
            }
        }
        Command::Index {
            repo_root,
            db,
            base_url,
            api_key,
            embed_model,
            chat_model,
            progress_jsonl,
            ignore,
            watch,
            watch_daemon,
            chunk_summary_model,
            chunk_summary_concurrency,
            no_chunk_summaries,
            retrieval_primary,
            enable_dense,
            disable_dense,
            dense_fallback,
            no_dense_fallback,
        } => {
            let retrieval_config = resolve_retrieval_config(
                retrieval_primary,
                enable_dense,
                disable_dense,
                dense_fallback,
                no_dense_fallback,
            )?;
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            let mut command_log = CommandLog::open(&db, "index")?;
            command_log.event(
                "index_started",
                json!({
                    "repo_root": repo_root,
                    "db": db,
                    "embedding_model": embed_model.as_str(),
                    "chat_model": chat_model.as_str(),
                }),
            )?;
            let store = MatryoshkaStore::open(&db)?;
            let parser_config = parser_config(ignore);
            let mut progress_writer = CliProgressStateWriter::new(&db, "index");
            let enricher = MlxChatEnricher::new(&base_url, &api_key).with_model(chat_model.clone());
            let embedder = EndpointEmbedder::new(&base_url, &api_key, embed_model.clone());
            let chunk_summarizer = MlxChunkSummarizer::new(&base_url, &api_key)
                .with_model(&chunk_summary_model)
                .with_concurrency(chunk_summary_concurrency);
            let indexer = FullIndexer::new(store, enricher, embedder, chunk_summarizer)
                .with_parser_config(parser_config)
                .with_retrieval_config(retrieval_config)
                .with_chunk_summary_enabled(!no_chunk_summaries);
            let summary = indexer.index_repo_with_progress(&repo_root, |event| {
                record_cli_progress(&mut progress_writer, progress_jsonl, event);
            })?;
            command_log.event("index_completed", index_summary_json(&summary))?;
            if !progress_jsonl {
                print_index_summary(summary);
            }
            if watch || watch_daemon {
                start_watch_after_index(
                    &repo_root,
                    &db,
                    &base_url,
                    &api_key,
                    &embed_model,
                    &chat_model,
                    retrieval_config,
                    watch_daemon,
                )?;
            }
        }
        Command::Update {
            repo_root,
            db,
            base_url,
            api_key,
            embed_model,
            chat_model,
            progress_jsonl,
            ignore,
            chunk_summary_model,
            chunk_summary_concurrency,
            no_chunk_summaries,
            retrieval_primary,
            enable_dense,
            disable_dense,
            dense_fallback,
            no_dense_fallback,
        } => {
            let retrieval_config = resolve_retrieval_config(
                retrieval_primary,
                enable_dense,
                disable_dense,
                dense_fallback,
                no_dense_fallback,
            )?;
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            let mut command_log = CommandLog::open(&db, "update")?;
            command_log.event(
                "update_started",
                json!({
                    "repo_root": repo_root,
                    "db": db,
                    "embedding_model": embed_model.as_str(),
                    "chat_model": chat_model.as_str(),
                }),
            )?;
            let store = MatryoshkaStore::open(&db)?;
            let parser_config = parser_config(ignore);
            let mut progress_writer = CliProgressStateWriter::new(&db, "update");
            let enricher = MlxChatEnricher::new(&base_url, &api_key).with_model(chat_model);
            let embedder = EndpointEmbedder::new(&base_url, &api_key, embed_model);
            let chunk_summarizer = MlxChunkSummarizer::new(&base_url, &api_key)
                .with_model(&chunk_summary_model)
                .with_concurrency(chunk_summary_concurrency);
            let indexer = FullIndexer::new(store, enricher, embedder, chunk_summarizer)
                .with_parser_config(parser_config)
                .with_retrieval_config(retrieval_config)
                .with_chunk_summary_enabled(!no_chunk_summaries);
            let summary = indexer.update_repo_with_progress(&repo_root, |event| {
                record_cli_progress(&mut progress_writer, progress_jsonl, event);
            })?;
            command_log.event("update_completed", update_summary_json(&summary))?;
            if !progress_jsonl {
                print_update_summary(summary);
            }
        }
        Command::Watch {
            repo_root,
            db,
            base_url,
            api_key,
            embed_model,
            chat_model,
            interval_ms,
            debounce_ms,
            ignore,
            daemon,
            skip_startup_update,
            retrieval_primary,
            enable_dense,
            disable_dense,
            dense_fallback,
            no_dense_fallback,
        } => {
            let retrieval_config = resolve_retrieval_config(
                retrieval_primary,
                enable_dense,
                disable_dense,
                dense_fallback,
                no_dense_fallback,
            )?;
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            let options = WatchLoopOptions {
                repo_root,
                db,
                base_url,
                api_key,
                embed_model,
                chat_model,
                interval_ms,
                debounce_ms,
                ignore,
                skip_startup_update,
                retrieval_config,
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
            base_url,
            api_key,
            embed_model,
            rerank,
            rerank_model,
            omlx_rerank,
            omlx_rerank_model,
            omlx_rerank_candidates,
            no_late_interaction,
            retrieval_primary,
            enable_dense,
            disable_dense,
            dense_fallback,
            no_dense_fallback,
            result_granularity,
            no_collapse,
            compact,
        } => {
            let result_granularity =
                resolve_search_result_granularity(result_granularity, no_collapse)?;
            let retrieval_config = resolve_retrieval_config(
                retrieval_primary,
                enable_dense,
                disable_dense,
                dense_fallback,
                no_dense_fallback,
            )?;
            let db = resolve_db_path(db, None)?;
            ensure_matryoshka_layout(&db)?;
            ensure_single_reranker(rerank, omlx_rerank)?;
            let late_interaction = !no_late_interaction;
            ensure_cli_prepare_ready(&db, retrieval_config, late_interaction)?;
            let store = MatryoshkaStore::open(&db)?;
            let hits = if omlx_rerank {
                SearchEngine::new(
                    store,
                    EndpointEmbedder::new(base_url.clone(), api_key.clone(), embed_model),
                )
                .with_dense(retrieval_config.dense_enabled)
                .with_late_interaction(late_interaction)
                .with_result_granularity(result_granularity)
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
                .with_dense(retrieval_config.dense_enabled)
                .with_late_interaction(late_interaction)
                .with_result_granularity(result_granularity)
                .with_reranker(EndpointReranker::new(base_url, api_key, rerank_model))
                .search(&query, limit)?
            } else {
                SearchEngine::new(store, EndpointEmbedder::new(base_url, api_key, embed_model))
                    .with_dense(retrieval_config.dense_enabled)
                    .with_late_interaction(late_interaction)
                    .with_result_granularity(result_granularity)
                    .search(&query, limit)?
            };
            print_search_hits(serde_json::to_value(&hits)?, compact)?;
        }
        Command::Op {
            db,
            task,
            query,
            limit,
            base_url,
            api_key,
            embed_model,
            rerank,
            rerank_model,
            omlx_rerank,
            omlx_rerank_model,
            omlx_rerank_candidates,
            no_late_interaction,
            retrieval_primary,
            enable_dense,
            disable_dense,
            dense_fallback,
            no_dense_fallback,
            result_granularity,
            no_collapse,
            compact,
        } => {
            let result_granularity =
                resolve_search_result_granularity(result_granularity, no_collapse)?;
            let retrieval_config = resolve_retrieval_config(
                retrieval_primary,
                enable_dense,
                disable_dense,
                dense_fallback,
                no_dense_fallback,
            )?;
            let db = resolve_db_path(db, None)?;
            ensure_matryoshka_layout(&db)?;
            ensure_single_reranker(rerank, omlx_rerank)?;
            let task_query = task_query(task, &query);
            let late_interaction = !no_late_interaction;
            ensure_cli_prepare_ready(&db, retrieval_config, late_interaction)?;
            let store = MatryoshkaStore::open(&db)?;
            let hits = if omlx_rerank {
                SearchEngine::new(
                    store,
                    EndpointEmbedder::new(base_url.clone(), api_key.clone(), embed_model),
                )
                .with_dense(retrieval_config.dense_enabled)
                .with_late_interaction(late_interaction)
                .with_result_granularity(result_granularity)
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
                .with_dense(retrieval_config.dense_enabled)
                .with_late_interaction(late_interaction)
                .with_result_granularity(result_granularity)
                .with_reranker(EndpointReranker::new(base_url, api_key, rerank_model))
                .search(&task_query, limit)?
            } else {
                SearchEngine::new(store, EndpointEmbedder::new(base_url, api_key, embed_model))
                    .with_dense(retrieval_config.dense_enabled)
                    .with_late_interaction(late_interaction)
                    .with_result_granularity(result_granularity)
                    .search(&task_query, limit)?
            };
            print_search_hits(serde_json::to_value(&hits)?, compact)?;
        }
        Command::Prewarm {
            db,
            repo_root,
            base_url,
            api_key,
            embed_model,
            limit,
            queries,
            no_late_interaction,
            ensure_fresh,
            watch,
            watch_daemon,
            retrieval_primary,
            enable_dense,
            disable_dense,
            dense_fallback,
            no_dense_fallback,
        } => {
            let retrieval_config = resolve_retrieval_config(
                retrieval_primary,
                enable_dense,
                disable_dense,
                dense_fallback,
                no_dense_fallback,
            )?;
            let repo_root = resolve_optional_repo_root(repo_root)?;
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            let mut command_log = CommandLog::open(&db, "prewarm")?;
            command_log.event(
                "prewarm_started",
                json!({
                    "repo_root": repo_root,
                    "db": db,
                    "embedding_model": embed_model.as_str(),
                    "ensure_fresh": ensure_fresh,
                    "limit": limit,
                }),
            )?;
            if ensure_fresh {
                let summary = run_update_once(
                    &repo_root,
                    &db,
                    &base_url,
                    &api_key,
                    &embed_model,
                    DEFAULT_CHAT_MODEL,
                    ParserConfig::default(),
                    DEFAULT_CHUNK_SUMMARY_MODEL,
                    DEFAULT_CHUNK_SUMMARY_CONCURRENCY,
                    true,
                    retrieval_config,
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
            let summary = SearchEngine::new(
                store,
                EndpointEmbedder::new(base_url.clone(), api_key.clone(), embed_model.clone()),
            )
            .with_dense(retrieval_config.dense_enabled)
            .with_late_interaction(late_interaction)
            .prewarm(&queries, limit)?;
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
                    &base_url,
                    &api_key,
                    &embed_model,
                    DEFAULT_CHAT_MODEL,
                    retrieval_config,
                    watch_daemon,
                )?;
            }
        }
        Command::RebuildSemantic {
            repo_root,
            db,
            base_url,
            api_key,
            embed_model,
            progress_jsonl,
            retrieval_primary,
            enable_dense,
            disable_dense,
            dense_fallback,
            no_dense_fallback,
        } => {
            let retrieval_config = resolve_retrieval_config(
                retrieval_primary,
                enable_dense,
                disable_dense,
                dense_fallback,
                no_dense_fallback,
            )?;
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            let mut command_log = CommandLog::open(&db, "semantic-rebuild")?;
            command_log.event(
                "semantic_rebuild_started",
                json!({
                    "repo_root": repo_root,
                    "db": db,
                    "embedding_model": embed_model.as_str(),
                }),
            )?;
            let store = MatryoshkaStore::open(&db)?;
            let mut progress_writer = CliProgressStateWriter::new(&db, "rebuild-semantic");
            let indexer = FullIndexer::new(
                store,
                MlxChatEnricher::new(&base_url, &api_key).with_model(DEFAULT_CHAT_MODEL),
                EndpointEmbedder::new(base_url.clone(), api_key.clone(), embed_model),
                MlxChunkSummarizer::new(base_url, api_key).with_model(DEFAULT_CHUNK_SUMMARY_MODEL),
            )
            .with_retrieval_config(retrieval_config);
            let summary = indexer.rebuild_semantic_index_with_progress(&repo_root, |event| {
                record_cli_progress(&mut progress_writer, progress_jsonl, event);
            })?;
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
            chunks,
            json: json_output,
            file,
        } => {
            let repo_root = resolve_optional_repo_root(repo_root)?;
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            ensure_cli_prepare_ready(&db, RetrievalConfig::default(), true)?;
            let read = ReadApi::new(MatryoshkaStore::open(&db)?, repo_root);
            let value = if chunks {
                serde_json::to_value(read.read_with_chunks(&file)?)?
            } else if json_output {
                serde_json::to_value(read.read(&file)?)?
            } else {
                serde_json::to_value(read.read_compact(&file)?)?
            };
            println!("{}", serde_json::to_string_pretty(&value)?);
        }
        Command::ReadBundle {
            db,
            repo_root,
            query,
            limit,
            related,
            mode,
            base_url,
            api_key,
            embed_model,
            rerank,
            rerank_model,
            omlx_rerank,
            omlx_rerank_model,
            omlx_rerank_candidates,
            no_late_interaction,
            retrieval_primary,
            enable_dense,
            disable_dense,
            dense_fallback,
            no_dense_fallback,
        } => {
            let retrieval_config = resolve_retrieval_config(
                retrieval_primary,
                enable_dense,
                disable_dense,
                dense_fallback,
                no_dense_fallback,
            )?;
            let repo_root = resolve_optional_repo_root(repo_root)?;
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            ensure_single_reranker(rerank, omlx_rerank)?;
            let late_interaction = !no_late_interaction;
            ensure_cli_prepare_ready(&db, retrieval_config, late_interaction)?;
            let store = MatryoshkaStore::open(&db)?;
            let hits = if omlx_rerank {
                SearchEngine::new(
                    store.clone(),
                    EndpointEmbedder::new(base_url.clone(), api_key.clone(), embed_model),
                )
                .with_dense(retrieval_config.dense_enabled)
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
                .with_dense(retrieval_config.dense_enabled)
                .with_late_interaction(late_interaction)
                .with_reranker(EndpointReranker::new(base_url, api_key, rerank_model))
                .search(&task_query(AgentTask::ReadNext, &query), limit)?
            } else {
                SearchEngine::new(
                    store.clone(),
                    EndpointEmbedder::new(base_url, api_key, embed_model),
                )
                .with_dense(retrieval_config.dense_enabled)
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
        Command::Enrich {
            repo_root,
            db,
            base_url,
            api_key,
            embed_model,
            chat_model,
            chunk_summary_model,
            chunk_summary_concurrency,
            max_files,
            status,
            json,
            progress_jsonl,
            no_chunk_summaries,
            retrieval_primary,
            enable_dense,
            disable_dense,
            dense_fallback,
            no_dense_fallback,
        } => {
            let retrieval_config = resolve_retrieval_config(
                retrieval_primary,
                enable_dense,
                disable_dense,
                dense_fallback,
                no_dense_fallback,
            )?;
            let db = resolve_db_path(db, Some(&repo_root))?;
            ensure_matryoshka_layout(&db)?;
            let api = Matryoshka::new(
                MatryoshkaConfig::new(&repo_root)
                    .with_db(&db)
                    .with_endpoint(&base_url, &api_key)
                    .with_models(&chat_model, &embed_model)
                    .with_retrieval_config(retrieval_config)
                    .with_llm_enrichment_enabled(true)
                    .with_chunk_summary_enabled(!no_chunk_summaries)
                    .with_chunk_summary_model(&chunk_summary_model)
                    .with_chunk_summary_concurrency(chunk_summary_concurrency),
            );
            if status {
                let report = api.enrichment_status()?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&report)?);
                } else {
                    print_enrichment_status(&report);
                }
                return Ok(());
            }

            let summary = api.enrich_once_with_progress(
                ApiEnrichmentOptions {
                    max_files,
                    write_progress_state: true,
                },
                |event| {
                    if progress_jsonl {
                        match serde_json::to_string(&event) {
                            Ok(line) => println!("{line}"),
                            Err(err) => println!(
                                "{}",
                                json!({
                                    "event": "progress_serialization_failed",
                                    "message": err.to_string(),
                                })
                            ),
                        }
                    }
                },
            )?;
            if !progress_jsonl {
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&enrichment_summary_json(&summary))?
                    );
                } else {
                    print_enrichment_summary(&summary);
                }
            }
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
        Command::Chunks {
            repo_root,
            ignore,
            source,
            json,
            with_code,
        } => {
            let parser_config = parser_config(ignore);
            let parser = SourceParser::new(parser_config);
            let parsed = parser.parse_repo(&repo_root)?;
            let source_filter = source.trim().to_ascii_lowercase().replace('_', "");
            let mut chunks = parsed.code_chunks;
            if !source_filter.is_empty() && source_filter != "all" {
                chunks.retain(|chunk| {
                    format!("{:?}", chunk.summary_source)
                        .to_ascii_lowercase()
                        .replace('_', "")
                        == source_filter
                });
            }

            if json {
                let payload: Vec<serde_json::Value> = chunks
                    .iter()
                    .map(|chunk| {
                        let mut value = serde_json::json!({
                            "chunk_id": chunk.chunk_id,
                            "path": chunk.path,
                            "symbol": chunk.symbol,
                            "qualified_name": chunk.qualified_name,
                            "kind": format!("{:?}", chunk.kind),
                            "signature": chunk.signature,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                            "summary_source": format!("{:?}", chunk.summary_source),
                            "summary": chunk.summary,
                            "doc_summary": chunk.doc_summary,
                        });
                        if with_code {
                            value["code"] = serde_json::Value::String(chunk.code.clone());
                        }
                        value
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&payload)?);
            } else {
                println!("chunks: {}", chunks.len());
                println!(
                    "{:<60} {:<40} {:<10} {:<10} {:<14} {}",
                    "path", "symbol", "kind", "lines", "source", "summary"
                );
                println!("{}", "-".repeat(160));
                for chunk in &chunks {
                    let lines = format!("{}-{}", chunk.start_line, chunk.end_line);
                    let summary_preview: String = chunk.summary.chars().take(70).collect();
                    println!(
                        "{:<60} {:<40} {:<10} {:<10} {:<14} {}",
                        truncate_str(&chunk.path, 60),
                        truncate_str(chunk.qualified_name.as_deref().unwrap_or(""), 40),
                        format!("{:?}", chunk.kind),
                        lines,
                        format!("{:?}", chunk.summary_source),
                        summary_preview
                    );
                    if with_code {
                        println!("--- code ---");
                        println!("{}", chunk.code);
                        println!("--- end code ---");
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct PrepareOptions {
    repo_root: PathBuf,
    db: PathBuf,
    base_url: String,
    api_key: String,
    embed_model: String,
    chat_model: String,
    ignore: Vec<String>,
    limit: usize,
    queries: Vec<String>,
    late_interaction: bool,
    retrieval_config: RetrievalConfig,
    chunk_summary_model: String,
    chunk_summary_concurrency: usize,
    no_chunk_summaries: bool,
    enrich_now: bool,
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
    enrichment: EnrichmentReadinessReport,
    retrieval_index: RetrievalIndexReport,
    prewarm: SearchPrewarmSummary,
    embedding_model: String,
}

fn run_prepare_via_api(options: PrepareOptions, progress_jsonl: bool) -> Result<PrepareSummary> {
    let config = MatryoshkaConfig::new(&options.repo_root)
        .with_db(&options.db)
        .with_endpoint(&options.base_url, &options.api_key)
        .with_models(&options.chat_model, &options.embed_model)
        .with_ignored_paths(options.ignore.clone())
        .with_late_interaction(options.late_interaction)
        .with_retrieval_config(options.retrieval_config)
        .with_llm_enrichment_enabled(options.enrich_now)
        .with_chunk_summary_enabled(!options.no_chunk_summaries)
        .with_chunk_summary_model(&options.chunk_summary_model)
        .with_chunk_summary_concurrency(options.chunk_summary_concurrency);
    let api = Matryoshka::new(config);
    let summary = api.prepare_with_progress(
        ApiPrepareOptions {
            limit: options.limit,
            queries: options.queries,
            write_progress_state: true,
        },
        |event| {
            if progress_jsonl {
                match serde_json::to_string(&event) {
                    Ok(line) => println!("{line}"),
                    Err(err) => println!(
                        "{}",
                        json!({
                            "event": "progress_serialization_failed",
                            "message": err.to_string(),
                        })
                    ),
                }
            }
        },
    )?;
    Ok(prepare_summary_from_api(summary))
}

fn prepare_summary_from_api(summary: ApiPrepareSummary) -> PrepareSummary {
    PrepareSummary {
        repo_root: summary.repo_root,
        db: summary.db,
        ready_marker: summary.ready_marker,
        logs_dir: summary.logs_dir,
        status: summary.status.as_str().into(),
        actions_taken: summary.actions_taken,
        file_count: summary.file_count,
        folder_count: summary.folder_count,
        symbol_count: summary.symbol_count,
        semantic_record_count: summary.semantic_record_count,
        changed_files: summary.changed_files,
        removed_files: summary.removed_files,
        changed_folders: summary.changed_folders,
        repo_card_updated: summary.repo_card_updated,
        artifact_quality: summary.artifact_quality,
        enrichment: summary.enrichment,
        retrieval_index: summary.retrieval_index,
        prewarm: SearchPrewarmSummary {
            fts_record_count: summary.prewarm.fts_record_count,
            query_count: summary.prewarm.query_count,
            warmed_hit_count: summary.prewarm.warmed_hit_count,
        },
        embedding_model: summary.embedding_model,
    }
}

#[derive(Debug, Clone)]
struct WatchLoopOptions {
    repo_root: PathBuf,
    db: PathBuf,
    base_url: String,
    api_key: String,
    embed_model: String,
    chat_model: String,
    interval_ms: u64,
    debounce_ms: u64,
    ignore: Vec<String>,
    skip_startup_update: bool,
    retrieval_config: RetrievalConfig,
}

struct CommandLog {
    path: PathBuf,
    file: File,
}

struct CliProgressStateWriter {
    operation: String,
    path: PathBuf,
    enriched_files: BTreeSet<String>,
    last_percent: f32,
}

impl CliProgressStateWriter {
    fn new(db: &Path, operation: &str) -> Self {
        Self {
            operation: operation.into(),
            path: progress_state_path(db),
            enriched_files: BTreeSet::new(),
            last_percent: 0.0,
        }
    }

    fn record(&mut self, event: &MatryoshkaProgressEvent) {
        let state = self.state_for_event(event);
        if let Some(parent) = self.path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(
            &self.path,
            serde_json::to_string_pretty(&state).unwrap_or_default(),
        );
    }

    fn state_for_event(&mut self, event: &MatryoshkaProgressEvent) -> Value {
        match event {
            MatryoshkaProgressEvent::Started { .. } => self.state(
                "running",
                "starting",
                "Getting ready",
                0.02,
                None,
                None,
                None,
            ),
            MatryoshkaProgressEvent::DiscoveringFiles => self.state(
                "running",
                "discovering_files",
                "Looking through the project",
                0.04,
                None,
                None,
                None,
            ),
            MatryoshkaProgressEvent::FilesDiscovered { total_files } => self.state(
                "running",
                "discovering_files",
                "Looking through the project",
                0.06,
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
            } => self.state(
                "running",
                "reading_files",
                "Reading code structure",
                0.06 + progress_ratio(*index, *total_files) * 0.22,
                Some(path.clone()),
                Some(*index),
                Some(*total_files),
            ),
            MatryoshkaProgressEvent::EnrichingFile {
                path, total_files, ..
            } => self.state(
                "running",
                "enriching_files",
                "Understanding files",
                0.30 + progress_ratio(self.enriched_files.len(), *total_files) * 0.36,
                Some(path.clone()),
                Some(self.enriched_files.len()),
                Some(*total_files),
            ),
            MatryoshkaProgressEvent::EnrichedFile {
                path, total_files, ..
            } => {
                self.enriched_files.insert(path.clone());
                self.state(
                    "running",
                    "enriching_files",
                    "Understanding files",
                    0.30 + progress_ratio(self.enriched_files.len(), *total_files) * 0.36,
                    Some(path.clone()),
                    Some(self.enriched_files.len()),
                    Some(*total_files),
                )
            }
            MatryoshkaProgressEvent::EnrichingChunks { chunk_count } => self.item_state(
                "running",
                "enriching_chunks",
                "Understanding code",
                0.66,
                Some(0),
                Some(*chunk_count),
                "chunks",
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
            } => self.item_state(
                "running",
                "enriching_chunks",
                "Understanding code",
                0.66 + progress_ratio(*batch_index, *total_batches) * 0.10,
                Some(*batch_index),
                Some(*total_batches),
                "batches",
            ),
            MatryoshkaProgressEvent::EnrichedChunks { chunk_count } => self.item_state(
                "running",
                "enriching_chunks",
                "Understanding code",
                0.76,
                Some(*chunk_count),
                Some(*chunk_count),
                "chunks",
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
            } => self.item_state(
                "running",
                "embedding",
                "Preparing search",
                0.76 + progress_ratio(*batch_index, *total_batches) * 0.14,
                Some(*batch_index),
                Some(*total_batches),
                "batches",
            ),
            MatryoshkaProgressEvent::EmbeddingSkipped { record_count, .. } => self.item_state(
                "running",
                "embedding_skipped",
                "Preparing text search",
                0.90,
                Some(*record_count),
                Some(*record_count),
                "records",
            ),
            MatryoshkaProgressEvent::WritingDatabase { records_written } => self.item_state(
                "running",
                "saving",
                "Saving updates",
                (self.last_percent + 0.01).min(0.92),
                *records_written,
                None,
                "records",
            ),
            MatryoshkaProgressEvent::ArtifactQuality { .. } => self.state(
                "running",
                "checking",
                "Checking everything",
                0.94,
                None,
                None,
                None,
            ),
            MatryoshkaProgressEvent::RetrievalIndexHealth { .. } => self.state(
                "running",
                "checking",
                "Checking everything",
                0.96,
                None,
                None,
                None,
            ),
            MatryoshkaProgressEvent::Completed { file_count, .. } => self.state(
                "completed",
                "complete",
                "Ready",
                1.0,
                None,
                Some(*file_count),
                Some(*file_count),
            ),
            MatryoshkaProgressEvent::Failed { .. } => self.state(
                "failed",
                "failed",
                "Needs attention",
                self.last_percent,
                None,
                None,
                None,
            ),
        }
    }

    fn state(
        &mut self,
        status: &str,
        phase: &str,
        message: &str,
        percent: f32,
        current_file: Option<String>,
        files_done: Option<usize>,
        files_total: Option<usize>,
    ) -> Value {
        self.state_with_counters(
            status,
            phase,
            message,
            percent,
            current_file,
            files_done,
            files_total,
            None,
            None,
            None,
        )
    }

    fn item_state(
        &mut self,
        status: &str,
        phase: &str,
        message: &str,
        percent: f32,
        items_done: Option<usize>,
        items_total: Option<usize>,
        item_label: &str,
    ) -> Value {
        self.state_with_counters(
            status,
            phase,
            message,
            percent,
            None,
            None,
            None,
            items_done,
            items_total,
            Some(item_label),
        )
    }

    fn state_with_counters(
        &mut self,
        status: &str,
        phase: &str,
        message: &str,
        percent: f32,
        current_file: Option<String>,
        files_done: Option<usize>,
        files_total: Option<usize>,
        items_done: Option<usize>,
        items_total: Option<usize>,
        item_label: Option<&str>,
    ) -> Value {
        let percent = if status == "failed" {
            percent
        } else {
            percent.max(self.last_percent)
        }
        .clamp(0.0, 1.0);
        self.last_percent = self.last_percent.max(percent);
        json!({
            "operation": self.operation.clone(),
            "action": Value::Null,
            "status": status,
            "phase": phase,
            "message": message,
            "percent": percent,
            "current_file": current_file,
            "files_done": files_done,
            "files_total": files_total,
            "items_done": items_done,
            "items_total": items_total,
            "item_label": item_label,
            "updated_at_unix_ms": unix_millis(),
        })
    }
}

fn record_cli_progress(
    writer: &mut CliProgressStateWriter,
    progress_jsonl: bool,
    event: MatryoshkaProgressEvent,
) {
    writer.record(&event);
    if progress_jsonl {
        print_progress_jsonl(event);
    }
}

fn progress_ratio(done: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        (done as f32 / total as f32).clamp(0.0, 1.0)
    }
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
    base_url: &str,
    api_key: &str,
    embed_model: &str,
    chat_model: &str,
    retrieval_config: RetrievalConfig,
    daemon: bool,
) -> Result<()> {
    let options = WatchLoopOptions {
        repo_root: repo_root.to_path_buf(),
        db: db.to_path_buf(),
        base_url: base_url.to_string(),
        api_key: api_key.to_string(),
        embed_model: embed_model.to_string(),
        chat_model: chat_model.to_string(),
        interval_ms: 2_000,
        debounce_ms: 3_000,
        ignore: Vec::new(),
        skip_startup_update: false,
        retrieval_config,
    };
    if daemon {
        spawn_watch_daemon(&options)
    } else {
        run_watch_loop(options)
    }
}

fn append_retrieval_args(command: &mut ProcessCommand, config: RetrievalConfig) {
    let primary = match config.primary {
        RetrievalPrimary::Fts => "fts",
        RetrievalPrimary::Splade => "splade",
        RetrievalPrimary::Dense => "dense",
        RetrievalPrimary::Hybrid => "hybrid",
    };
    command.arg("--retrieval-primary").arg(primary);
    if config.dense_enabled {
        command.arg("--enable-dense");
    } else {
        command.arg("--disable-dense");
    }
    if config.dense_fallback_enabled {
        command.arg("--dense-fallback");
    } else {
        command.arg("--no-dense-fallback");
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
    append_retrieval_args(&mut command, options.retrieval_config);
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
            "interval_ms": options.interval_ms,
            "debounce_ms": options.debounce_ms,
            "startup_update": !options.skip_startup_update,
        }),
    )?;

    if !options.skip_startup_update {
        let summary = run_update_once(
            &options.repo_root,
            &options.db,
            &options.base_url,
            &options.api_key,
            &options.embed_model,
            &options.chat_model,
            parser_config.clone(),
            DEFAULT_CHUNK_SUMMARY_MODEL,
            DEFAULT_CHUNK_SUMMARY_CONCURRENCY,
            true,
            options.retrieval_config,
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
                &options.base_url,
                &options.api_key,
                &options.embed_model,
                &options.chat_model,
                parser_config.clone(),
                DEFAULT_CHUNK_SUMMARY_MODEL,
                DEFAULT_CHUNK_SUMMARY_CONCURRENCY,
                true,
                options.retrieval_config,
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
    base_url: &str,
    api_key: &str,
    embed_model: &str,
    chat_model: &str,
    parser_config: ParserConfig,
    chunk_summary_model: &str,
    chunk_summary_concurrency: usize,
    chunk_summary_enabled: bool,
    retrieval_config: RetrievalConfig,
    mut log: Option<&mut CommandLog>,
) -> Result<UpdateSummary> {
    if let Some(log) = log.as_deref_mut() {
        log.event(
            "update_started",
            json!({
                "repo_root": repo_root,
                "db": db,
                "embedding_model": embed_model,
            }),
        )?;
    }
    let store = MatryoshkaStore::open(db)?;
    let enricher = MlxChatEnricher::new(base_url, api_key).with_model(chat_model.to_string());
    let embedder = EndpointEmbedder::new(base_url, api_key, embed_model.to_string());
    let chunk_summarizer = MlxChunkSummarizer::new(base_url, api_key)
        .with_model(chunk_summary_model)
        .with_concurrency(chunk_summary_concurrency);
    let summary = FullIndexer::new(store, enricher, embedder, chunk_summarizer)
        .with_parser_config(parser_config)
        .with_retrieval_config(retrieval_config)
        .with_chunk_summary_enabled(chunk_summary_enabled)
        .update_repo(repo_root)?;
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

fn artifact_gap_count(report: &ArtifactQualityReport) -> usize {
    report.file_cards_empty_summary
        + report.folder_cards_empty_summary
        + usize::from(!report.repo_card_has_summary)
}

fn retrieval_is_ready(report: &RetrievalIndexReport) -> bool {
    report.semantic_records > 0
        && report.fts_records > 0
        && (!report.dense_enabled || report.embedded_records > 0)
        && (!report.dense_enabled
            || !report.late_interaction_enabled
            || report.records_with_late_vectors > 0)
}

fn progress_state_path(db: &Path) -> PathBuf {
    db.parent()
        .unwrap_or_else(|| Path::new(MATRYOSHKA_DIR))
        .join("state")
        .join("progress.json")
}

fn ready_marker_path(db: &Path) -> PathBuf {
    db.parent()
        .unwrap_or_else(|| Path::new(MATRYOSHKA_DIR))
        .join(".jesco-prewarm-complete")
}

fn ensure_cli_prepare_ready(
    db: &Path,
    retrieval_config: RetrievalConfig,
    late_interaction: bool,
) -> Result<()> {
    if !ready_marker_path(db).exists() {
        anyhow::bail!(
            "Matryoshka prepare is not ready for {}; run prepare first{}",
            db.display(),
            prepare_state_error_hint(db)
        );
    }

    let store = MatryoshkaStore::open(db)?;

    let stats = store.retrieval_index_stats()?;
    let report = RetrievalIndexReport {
        semantic_records: stats.semantic_records,
        embedded_records: stats.embedded_records,
        fts_records: stats.fts_records,
        late_vector_rows: stats.late_vector_rows,
        records_with_late_vectors: stats.records_with_late_vectors,
        retrieval_primary: retrieval_config.primary,
        dense_enabled: retrieval_config.dense_enabled,
        dense_fallback_enabled: retrieval_config.dense_fallback_enabled,
        late_interaction_enabled: late_interaction && retrieval_config.dense_enabled,
    };
    if !retrieval_is_ready(&report) {
        anyhow::bail!(
            "Matryoshka prepare is not ready: retrieval index is incomplete (semantic_records={}, fts_records={}, embedded_records={}, records_with_late_vectors={}); run prepare again{}",
            report.semantic_records,
            report.fts_records,
            report.embedded_records,
            report.records_with_late_vectors,
            prepare_state_error_hint(db)
        );
    }

    Ok(())
}

fn prepare_state_error_hint(db: &Path) -> String {
    let Ok(raw) = fs::read_to_string(progress_state_path(db)) else {
        return String::new();
    };
    let Ok(value) = serde_json::from_str::<Value>(&raw) else {
        return String::new();
    };
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let phase = value
        .get("phase")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let message = value
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let last_error = value
        .get("last_error")
        .and_then(Value::as_str)
        .unwrap_or_default();

    if !last_error.trim().is_empty() {
        format!("; last prepare error: {last_error}")
    } else if status == "running" {
        let phase = if phase.trim().is_empty() {
            "unknown"
        } else {
            phase
        };
        format!(
            "; previous prepare is still running or was interrupted at phase {phase}; run prepare again to resume"
        )
    } else if !status.is_empty() && status != "ready" {
        format!("; last prepare status: {status} ({phase}) {message}")
    } else {
        String::new()
    }
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

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

fn print_search_hits(mut value: Value, compact: bool) -> Result<()> {
    if compact {
        strip_search_match_details(&mut value);
    }
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn strip_search_match_details(value: &mut Value) {
    match value {
        Value::Array(items) => {
            for item in items {
                strip_search_match_details(item);
            }
        }
        Value::Object(map) => {
            map.remove("matched_terms");
            map.remove("total_matched_symbols");
            map.remove("why_matched");
            for item in map.values_mut() {
                strip_search_match_details(item);
            }
        }
        _ => {}
    }
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
        if retrieval_is_ready(&summary.retrieval_index) {
            "ready"
        } else {
            "needs_refresh"
        }
    );
    println!("enrichment: {:?}", summary.enrichment.status);
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

fn print_enrichment_status(report: &EnrichmentReadinessReport) {
    println!("enrichment: {:?}", report.status);
    println!("files: {}/{}", report.file_cards_ready, report.files_total);
    println!(
        "folders: {}/{}",
        report.folder_cards_ready, report.folders_total
    );
    println!("chunks: {}/{}", report.chunks_ready, report.chunks_total);
    println!("repo_card_ready: {}", report.repo_card_ready);
    println!("pending_total: {}", report.pending_total());
    println!("stale_file_cards: {}", report.file_cards_stale);
    if !report.pending_files_sample.is_empty() {
        println!(
            "pending_files_sample: {}",
            report.pending_files_sample.join(", ")
        );
    }
    if !report.pending_folders_sample.is_empty() {
        println!(
            "pending_folders_sample: {}",
            report.pending_folders_sample.join(", ")
        );
    }
    if !report.pending_chunks_sample.is_empty() {
        println!(
            "pending_chunks_sample: {}",
            report.pending_chunks_sample.join(", ")
        );
    }
}

fn print_enrichment_summary(summary: &EnrichmentSummary) {
    println!("enrichment_batch: complete");
    println!("selected_files: {}", summary.selected_files);
    println!("selected_folders: {}", summary.selected_folders);
    println!("repo_card_updated: {}", summary.repo_card_updated);
    println!("before_status: {:?}", summary.before.status);
    println!("after_status: {:?}", summary.after.status);
    println!("before_pending: {}", summary.before.pending_total());
    println!("after_pending: {}", summary.after.pending_total());
    println!("semantic_records: {}", summary.semantic_record_count);
    print_artifact_quality(&summary.artifact_quality);
    print_retrieval_index(&summary.retrieval_index);
    println!("embedding_model: {}", summary.embedding_model);
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
        "enrichment": &summary.enrichment,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_enrich_now_flag_is_explicit_and_defaults_off() {
        let default_args = Args::try_parse_from(["matryoshka-rs", "prepare", "/tmp/repo"]).unwrap();
        match default_args.command {
            Command::Prepare { enrich_now, .. } => assert!(!enrich_now),
            _ => panic!("expected prepare command"),
        }

        let enriched_args =
            Args::try_parse_from(["matryoshka-rs", "prepare", "/tmp/repo", "--enrich-now"])
                .unwrap();
        match enriched_args.command {
            Command::Prepare { enrich_now, .. } => assert!(enrich_now),
            _ => panic!("expected prepare command"),
        }
    }
}
