mod heuristic;
mod mlx_chat;
mod prompts;

pub use heuristic::*;
pub use mlx_chat::*;
pub use prompts::*;

use anyhow::Result;
use matryoshka_core_ir::{
    CodeChunkFact, FileCard, FileEnrichmentContext, FileFact, FolderCard, FolderEnrichmentContext,
    FolderFact, RepoCard, SymbolFact,
};

pub trait CodeEnricher {
    fn enrich_file(
        &self,
        file: &FileFact,
        symbols: &[SymbolFact],
        context: &FileEnrichmentContext,
    ) -> Result<FileCard>;
    fn enrich_folder(
        &self,
        folder: &FolderFact,
        child_files: &[FileCard],
        child_folders: &[FolderCard],
        context: &FolderEnrichmentContext,
    ) -> Result<FolderCard>;
    fn enrich_repo(&self, repo_root: &str, folders: &[FolderCard]) -> Result<RepoCard>;
}

/// A generated summary for a single code chunk, keyed by `chunk_id` so the
/// indexer can map it back onto the originating `CodeChunkFact`.
#[derive(Debug, Clone)]
pub struct ChunkSummaryDraft {
    pub chunk_id: String,
    pub summary: String,
}

/// Summarizes code chunks that have no useful docstring/doc comment.
///
/// Implementations should only be called for chunks where `summary_source`
/// is `Empty` (or otherwise needs generation). Chunks with useful docs are
/// used directly and never passed to the summarizer.
pub trait ChunkSummarizer {
    fn summarize_chunks(&self, chunks: &[CodeChunkFact]) -> Result<Vec<ChunkSummaryDraft>>;

    /// Summarize chunks with a progress callback. The callback is invoked once
    /// per batch with `(batch_index, total_batches, chunks_in_batch)`.
    ///
    /// Default implementation delegates to `summarize_chunks` without progress.
    fn summarize_chunks_with_progress(
        &self,
        chunks: &[CodeChunkFact],
        progress: &mut dyn FnMut(usize, usize, usize),
    ) -> Result<Vec<ChunkSummaryDraft>> {
        let _ = progress;
        self.summarize_chunks(chunks)
    }
}
