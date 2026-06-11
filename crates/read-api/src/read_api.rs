use anyhow::{Result, anyhow};
use matryoshka_core_ir::{
    DependencyInterpretation, EdgeFact, FileCard, FileFact, FolderCard, ImportFact, ReadCard,
    ReadDependencies, ReadDependency, ReadFileOverview, ReadFolderOverview, ReadImport, ReadSymbol,
    SymbolBehavior, SymbolFact,
};
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
        let file = self
            .store
            .load_file(file_id)?
            .ok_or_else(|| anyhow!("unknown file id {file_id}"))?;
        let file_card = self.store.load_file_card(file_id)?;
        let folder_card = self.store.load_folder_card(&file.parent_folder_id)?;
        let mut symbols = self.store.load_symbols_for_file(file_id)?;
        symbols.sort_by_key(|symbol| symbol.start_line);
        let (incoming_edges, outgoing_edges) = self.store.load_edges_for_entity(file_id)?;
        let source_lines = read_lines(&self.repo_root.join(&file.path)).unwrap_or_default();
        let module_docs = module_docs(&source_lines, &file.language);
        let card_is_heuristic = file_card
            .as_ref()
            .and_then(|card| card.provenance.model.as_deref())
            == Some("heuristic");
        let folder_card_is_heuristic = folder_card
            .as_ref()
            .and_then(|card| card.provenance.model.as_deref())
            == Some("heuristic");
        Ok(ReadCard {
            file: file_overview(&file),
            summary: read_summary(&file, file_card.as_ref(), &module_docs, card_is_heuristic),
            description: read_description(file_card.as_ref(), &module_docs, card_is_heuristic),
            folder: folder_card
                .as_ref()
                .filter(|_| !folder_card_is_heuristic)
                .map(folder_overview),
            symbols: read_symbols(
                &symbols,
                file_card.as_ref(),
                &source_lines,
                card_is_heuristic,
            ),
            imports: read_imports(
                &file.imports,
                file_card.as_ref().filter(|_| !card_is_heuristic),
            ),
            dependencies: read_dependencies(&incoming_edges, &outgoing_edges),
            agent_hints: read_hints(file_card.as_ref(), card_is_heuristic),
        })
    }
}

fn file_overview(file: &FileFact) -> ReadFileOverview {
    ReadFileOverview {
        file_id: file.file_id.clone(),
        path: file.path.clone(),
        name: file.name.clone(),
        language: file.language.clone(),
        parent_folder_id: file.parent_folder_id.clone(),
        source_hash: file.source_hash.clone(),
        line_count: file.line_count,
    }
}

fn folder_overview(card: &FolderCard) -> ReadFolderOverview {
    ReadFolderOverview {
        folder_id: card.folder_id.clone(),
        summary: card.summary.clone(),
        responsibility: card.responsibility.clone(),
    }
}

fn read_summary(
    file: &FileFact,
    card: Option<&FileCard>,
    module_docs: &[String],
    card_is_heuristic: bool,
) -> Option<String> {
    if !card_is_heuristic {
        if let Some(summary) = card
            .map(|card| card.summary.trim())
            .filter(|summary| !summary.is_empty())
        {
            return Some(summary.to_string());
        }
    }
    module_docs.first().cloned().or_else(|| {
        card.map(|card| card.summary.trim().to_string())
            .filter(|summary| !summary.is_empty())
            .or_else(|| {
                Some(format!(
                    "{} is a {} file with {} lines.",
                    file.path, file.language, file.line_count
                ))
            })
    })
}

fn read_description(
    card: Option<&FileCard>,
    module_docs: &[String],
    card_is_heuristic: bool,
) -> Option<String> {
    let mut parts = Vec::new();
    if card_is_heuristic {
        parts.extend(module_docs.iter().cloned());
    } else if let Some(card) = card {
        push_labeled(&mut parts, "Role", &card.role);
        push_joined(&mut parts, "Behaviors", &card.primary_behaviors);
        push_joined(&mut parts, "Owns", &card.owns_behaviors);
        push_joined(&mut parts, "Delegates to", &card.delegates_to);
        push_joined(&mut parts, "Side effects", &card.side_effects);
        push_joined(&mut parts, "Blast radius", &card.blast_radius);
    } else {
        parts.extend(module_docs.iter().cloned());
    }
    non_empty(parts.join("\n"))
}

fn read_symbols(
    symbols: &[SymbolFact],
    card: Option<&FileCard>,
    source_lines: &[String],
    card_is_heuristic: bool,
) -> Vec<ReadSymbol> {
    symbols
        .iter()
        .map(|symbol| {
            let symbol_behavior = card
                .and_then(|card| matching_symbol_behavior(symbol, &card.important_symbols))
                .and_then(|behavior| {
                    meaningful_symbol_behavior(
                        &behavior.behavior,
                        &symbol.signature,
                        card_is_heuristic,
                    )
                });
            ReadSymbol {
                name: symbol.name.clone(),
                qualified_name: symbol.qualified_name.clone(),
                kind: symbol.kind.clone(),
                signature: symbol.signature.clone(),
                start_line: symbol.start_line,
                end_line: symbol.end_line,
                doc: symbol_doc(source_lines, symbol.start_line),
                behavior: symbol_behavior,
            }
        })
        .collect()
}

fn matching_symbol_behavior<'a>(
    symbol: &SymbolFact,
    behaviors: &'a [SymbolBehavior],
) -> Option<&'a SymbolBehavior> {
    behaviors.iter().find(|behavior| {
        behavior.symbol_id == symbol.symbol_id
            || behavior.name == symbol.name
            || symbol.qualified_name.ends_with(&behavior.name)
    })
}

fn meaningful_symbol_behavior(
    behavior: &str,
    signature: &str,
    card_is_heuristic: bool,
) -> Option<String> {
    let trimmed = behavior.trim();
    if trimmed.is_empty() {
        return None;
    }
    if card_is_heuristic
        && (trimmed == signature.trim()
            || trimmed.starts_with("fn ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("enum ")
            || trimmed.starts_with("impl "))
    {
        return None;
    }
    Some(trimmed.to_string())
}

fn read_imports(imports: &[ImportFact], card: Option<&FileCard>) -> Vec<ReadImport> {
    let interpreted = card
        .map(|card| card.imports_interpreted.as_slice())
        .unwrap_or_default();
    imports
        .iter()
        .map(|import| ReadImport {
            module: import.module.clone(),
            names: import.names.clone(),
            line: import.line,
            is_internal: import.is_internal,
            resolved_file_id: import.resolved_file_id.clone(),
            purpose: import_purpose(import, interpreted),
        })
        .collect()
}

fn import_purpose(import: &ImportFact, interpreted: &[DependencyInterpretation]) -> Option<String> {
    interpreted
        .iter()
        .find(|item| {
            import.resolved_file_id.as_deref() == Some(item.target_id.as_str())
                || import.resolved_file_id.as_deref() == Some(item.target_path.as_str())
                || item.target_path.contains(&import.module.replace('.', "/"))
                || import.module.contains(&item.target_path.replace('/', "."))
        })
        .map(|item| item.why.clone())
        .filter(|why| !why.trim().is_empty())
}

fn read_dependencies(incoming: &[EdgeFact], outgoing: &[EdgeFact]) -> ReadDependencies {
    ReadDependencies {
        incoming: incoming
            .iter()
            .map(|edge| read_dependency(edge, edge.source_id.clone()))
            .collect(),
        outgoing: outgoing
            .iter()
            .map(|edge| read_dependency(edge, edge.target_id.clone()))
            .collect(),
    }
}

fn read_dependency(edge: &EdgeFact, entity_id: String) -> ReadDependency {
    ReadDependency {
        entity_id,
        kind: edge.kind.clone(),
        detail: edge.detail.clone(),
    }
}

fn read_hints(card: Option<&FileCard>, card_is_heuristic: bool) -> Vec<String> {
    if card_is_heuristic {
        return Vec::new();
    }
    card.map(|card| card.agent_read_hints.clone())
        .unwrap_or_default()
}

fn module_docs(lines: &[String], language: &str) -> Vec<String> {
    match language {
        "rust" => rust_module_docs(lines),
        "python" => python_module_docs(lines),
        _ => Vec::new(),
    }
}

fn rust_module_docs(lines: &[String]) -> Vec<String> {
    let mut docs = Vec::new();
    for line in lines.iter().take(80) {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//!") {
            docs.push(strip_doc_marker(trimmed).to_string());
        } else if trimmed.is_empty() && docs.is_empty() {
            continue;
        } else if trimmed.is_empty() && !docs.is_empty() {
            docs.push(String::new());
        } else {
            break;
        }
    }
    normalize_doc_blocks(docs)
}

fn python_module_docs(lines: &[String]) -> Vec<String> {
    let first_non_empty = lines.iter().position(|line| !line.trim().is_empty());
    let Some(start) = first_non_empty else {
        return Vec::new();
    };
    let trimmed = lines[start].trim();
    let quote = if trimmed.starts_with("\"\"\"") {
        "\"\"\""
    } else if trimmed.starts_with("'''") {
        "'''"
    } else {
        return Vec::new();
    };
    let mut docs = Vec::new();
    for (offset, line) in lines[start..].iter().enumerate() {
        let mut text = line.trim().to_string();
        if offset == 0 {
            text = text.trim_start_matches(quote).to_string();
        }
        if text.ends_with(quote) {
            docs.push(text.trim_end_matches(quote).trim().to_string());
            break;
        }
        docs.push(text);
    }
    normalize_doc_blocks(docs)
}

fn symbol_doc(lines: &[String], start_line: usize) -> Option<String> {
    if start_line <= 1 || lines.is_empty() {
        return None;
    }
    let mut docs = Vec::new();
    let mut index = start_line.saturating_sub(2);
    loop {
        let trimmed = lines.get(index)?.trim_start();
        if is_doc_comment(trimmed) {
            docs.push(strip_doc_marker(trimmed).to_string());
        } else if trimmed.is_empty() && !docs.is_empty() {
            docs.push(String::new());
        } else {
            break;
        }
        if index == 0 {
            break;
        }
        index -= 1;
    }
    docs.reverse();
    non_empty(normalize_doc_blocks(docs).join("\n"))
}

fn is_doc_comment(line: &str) -> bool {
    line.starts_with("///")
        || line.starts_with("//!")
        || line.starts_with("#")
        || line.starts_with("*")
}

fn strip_doc_marker(line: &str) -> &str {
    line.trim_start()
        .trim_start_matches("///")
        .trim_start_matches("//!")
        .trim_start_matches("//")
        .trim_start_matches('#')
        .trim_start_matches('*')
        .trim()
}

fn normalize_doc_blocks(lines: Vec<String>) -> Vec<String> {
    let joined = lines
        .join("\n")
        .split("\n\n")
        .map(|block| {
            block
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|block| !block.is_empty())
        .collect::<Vec<_>>();
    joined
}

fn push_labeled(parts: &mut Vec<String>, label: &str, value: &str) {
    if !value.trim().is_empty() {
        parts.push(format!("{label}: {}", value.trim()));
    }
}

fn push_joined(parts: &mut Vec<String>, label: &str, values: &[String]) {
    if !values.is_empty() {
        push_labeled(parts, label, &values.join("; "));
    }
}

fn non_empty(value: String) -> Option<String> {
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

fn read_lines(path: &Path) -> Result<Vec<String>> {
    Ok(fs::read_to_string(path)?
        .lines()
        .map(ToString::to_string)
        .collect())
}
