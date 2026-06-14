mod heuristic;
mod mlx_chat;
mod prompts;

pub use heuristic::*;
pub use mlx_chat::*;
pub use prompts::*;

use anyhow::Result;
use matryoshka_core_ir::{
    FileCard, FileEnrichmentContext, FileFact, FolderCard, FolderEnrichmentContext, FolderFact,
    RepoCard, SymbolFact,
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
