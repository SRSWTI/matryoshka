use anyhow::{Result, anyhow};
use matryoshka_core_ir::{ReadCard, SnippetFact};
use matryoshka_store_sqlite::MatryoshkaStore;
use std::fs;
use std::path::{Path, PathBuf};

pub struct ReadApi {
    store: MatryoshkaStore,
    repo_root: PathBuf,
}

impl ReadApi {
    pub fn new(store: MatryoshkaStore, repo_root: impl Into<PathBuf>) -> Self {
        Self {
            store,
            repo_root: repo_root.into(),
        }
    }

    pub fn read(&self, file_id: &str) -> Result<ReadCard> {
        self.read_impl(file_id, false)
    }

    pub fn read_more(&self, file_id: &str) -> Result<ReadCard> {
        self.read_impl(file_id, true)
    }

    fn read_impl(&self, file_id: &str, include_source_blocks: bool) -> Result<ReadCard> {
        let file = self
            .store
            .load_file(file_id)?
            .ok_or_else(|| anyhow!("unknown file id {file_id}"))?;
        let file_card = self.store.load_file_card(file_id)?;
        let folder_card = self.store.load_folder_card(&file.parent_folder_id)?;
        let symbols = self.store.load_symbols_for_file(file_id)?;
        let (incoming_edges, outgoing_edges) = self.store.load_edges_for_entity(file_id)?;
        let symbol_blocks = if include_source_blocks {
            self.symbol_blocks(&file.path, &symbols)?
        } else {
            Vec::new()
        };
        let import_lines = if include_source_blocks {
            self.import_lines(&file.path)?
        } else {
            Vec::new()
        };
        Ok(ReadCard {
            imports: file.imports.clone(),
            snippets: file.snippets.clone(),
            file,
            file_card,
            folder_card,
            symbols,
            incoming_edges,
            outgoing_edges,
            symbol_blocks,
            import_lines,
        })
    }

    fn symbol_blocks(
        &self,
        file_path: &str,
        symbols: &[matryoshka_core_ir::SymbolFact],
    ) -> Result<Vec<SnippetFact>> {
        let source_path = self.repo_root.join(file_path);
        let lines = read_lines(&source_path)?;
        Ok(symbols
            .iter()
            .map(|symbol| {
                let start = symbol.start_line.saturating_sub(1);
                let end = symbol.end_line.min(lines.len());
                SnippetFact {
                    snippet_id: format!("{}#{}-{}", file_path, symbol.start_line, end),
                    file_id: file_path.into(),
                    title: symbol.qualified_name.clone(),
                    start_line: symbol.start_line,
                    end_line: end,
                    text: lines[start..end].join("\n"),
                }
            })
            .collect())
    }

    fn import_lines(&self, file_path: &str) -> Result<Vec<String>> {
        let source_path = self.repo_root.join(file_path);
        Ok(read_lines(&source_path)?
            .into_iter()
            .filter(|line| {
                let trimmed = line.trim_start();
                trimmed.starts_with("import ")
                    || trimmed.starts_with("from ")
                    || trimmed.starts_with("use ")
                    || trimmed.starts_with("mod ")
                    || trimmed.starts_with("pub mod ")
                    || trimmed.starts_with("pub use ")
                    || trimmed.starts_with("export ")
            })
            .collect())
    }
}

fn read_lines(path: &Path) -> Result<Vec<String>> {
    Ok(fs::read_to_string(path)?
        .lines()
        .map(ToString::to_string)
        .collect())
}
