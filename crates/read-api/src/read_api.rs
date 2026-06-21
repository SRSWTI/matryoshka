use anyhow::{Result, anyhow};
use matryoshka_core_ir::{
    CodeChunkFact, CompactReadCard, DependencyInterpretation, EdgeFact, EdgeKind, FileCard,
    FileFact, FolderCard, ImportFact, ReadCard, ReadCodeChunk, ReadDependency, ReadFileOverview,
    ReadFolderOverview, ReadImports, ReadInternalImport, ReadSymbol, SymbolBehavior, SymbolFact,
    SymbolKind,
};
use matryoshka_store_sqlite::MatryoshkaStore;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_READ_DEPENDENCIES: usize = 20;

pub struct ReadApi {
    store: MatryoshkaStore,
    repo_root: PathBuf,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReadPackMode {
    Brief,
    Edit,
    Flow,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PackedReadCard {
    pub file: ReadFileOverview,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub symbols: Vec<ReadSymbol>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub internal_imports: Vec<ReadInternalImport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependents: Vec<ReadDependency>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<ReadDependency>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub omitted: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReadBundle {
    pub mode: ReadPackMode,
    pub primary: PackedReadCard,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related: Vec<PackedReadCard>,
}

impl ReadApi {
    pub fn new(store: MatryoshkaStore, repo_root: impl Into<PathBuf>) -> Self {
        Self {
            store,
            repo_root: repo_root.into(),
        }
    }

    pub fn read(&self, file_id: &str) -> Result<ReadCard> {
        self.read_inner(file_id, false)
    }

    pub fn read_json(&self, file_id: &str) -> Result<ReadCard> {
        self.read(file_id)
    }

    pub fn read_compact(&self, file_id: &str) -> Result<CompactReadCard> {
        self.read(file_id).map(compact_read_card)
    }

    pub fn read_with_chunks(&self, file_id: &str) -> Result<ReadCard> {
        self.read_inner(file_id, true)
    }

    pub fn read_compact_with_chunks(&self, file_id: &str) -> Result<CompactReadCard> {
        self.read_with_chunks(file_id).map(compact_read_card)
    }

    fn read_inner(&self, file_id: &str, include_chunks: bool) -> Result<ReadCard> {
        let file = self
            .store
            .load_file(file_id)?
            .ok_or_else(|| anyhow!("unknown file id {file_id}"))?;
        let file_card = self.store.load_file_card(file_id)?;
        let folder_card = self.store.load_folder_card(&file.parent_folder_id)?;
        let mut symbols = self.store.load_symbols_for_file(file_id)?;
        symbols.sort_by_key(|symbol| symbol.start_line);
        let (incoming_edges, outgoing_edges) = self.store.load_edges_for_entity(file_id)?;
        let chunks = if include_chunks {
            read_chunks(&self.store.load_code_chunks_for_file(file_id)?)
        } else {
            Vec::new()
        };
        let source_lines = read_lines(&self.repo_root.join(&file.path)).unwrap_or_default();
        Ok(ReadCard {
            file: file_overview(&file),
            summary: read_summary(file_card.as_ref()),
            description: read_description(file_card.as_ref()),
            folder: folder_card.as_ref().map(folder_overview),
            symbols: if include_chunks {
                Vec::new()
            } else {
                read_symbols(&symbols, file_card.as_ref(), &source_lines)
            },
            chunks,
            imports: read_imports(&file.imports, file_card.as_ref()),
            total_dependents: collapsed_dependency_count(&incoming_edges, DependencySide::Incoming),
            dependents: read_dependencies(&incoming_edges, DependencySide::Incoming),
            total_depends_on: collapsed_dependency_count(&outgoing_edges, DependencySide::Outgoing),
            depends_on: read_dependencies(&outgoing_edges, DependencySide::Outgoing),
        })
    }

    pub fn read_packed(&self, file_id: &str, mode: ReadPackMode) -> Result<PackedReadCard> {
        self.read(file_id).map(|card| pack_read_card(card, mode))
    }

    pub fn read_bundle(
        &self,
        primary_file_id: &str,
        related_file_ids: &[String],
        mode: ReadPackMode,
        max_related: usize,
    ) -> Result<ReadBundle> {
        let primary = self.read_packed(primary_file_id, mode)?;
        let mut related = Vec::new();
        for file_id in related_file_ids {
            if file_id == primary_file_id
                || related
                    .iter()
                    .any(|card: &PackedReadCard| &card.file.file_id == file_id)
            {
                continue;
            }
            if related.len() >= max_related {
                break;
            }
            if let Ok(card) = self.read_packed(file_id, mode) {
                related.push(card);
            }
        }
        Ok(ReadBundle {
            mode,
            primary,
            related,
        })
    }
}

fn pack_read_card(card: ReadCard, mode: ReadPackMode) -> PackedReadCard {
    let mut omitted = Vec::new();
    let (symbol_limit, dep_limit, include_description) = match mode {
        ReadPackMode::Brief => (8, 3, false),
        ReadPackMode::Edit => (16, 8, true),
        ReadPackMode::Flow => (10, 12, true),
    };

    let symbol_count = card.symbols.len();
    let dependent_count = card.dependents.len();
    let depends_on_count = card.depends_on.len();
    let import_count = card.imports.internal.len();

    if symbol_count > symbol_limit {
        omitted.push(format!(
            "omitted {} symbols",
            symbol_count.saturating_sub(symbol_limit)
        ));
    }
    if import_count > dep_limit {
        omitted.push(format!(
            "omitted {} internal imports",
            import_count.saturating_sub(dep_limit)
        ));
    }
    if dependent_count > dep_limit {
        omitted.push(format!(
            "omitted {} dependents",
            dependent_count.saturating_sub(dep_limit)
        ));
    }
    if depends_on_count > dep_limit {
        omitted.push(format!(
            "omitted {} dependencies",
            depends_on_count.saturating_sub(dep_limit)
        ));
    }

    PackedReadCard {
        file: card.file,
        summary: card.summary,
        description: include_description.then_some(card.description).flatten(),
        symbols: card.symbols.into_iter().take(symbol_limit).collect(),
        internal_imports: card.imports.internal.into_iter().take(dep_limit).collect(),
        dependents: card.dependents.into_iter().take(dep_limit).collect(),
        depends_on: card.depends_on.into_iter().take(dep_limit).collect(),
        omitted,
    }
}

fn compact_read_card(card: ReadCard) -> CompactReadCard {
    CompactReadCard {
        file: card.file,
        summary: card.summary,
        description: card.description,
        folder: card.folder,
        symbols: card.symbols.iter().map(compact_read_symbol).collect(),
        chunks: card.chunks,
        imports: card.imports,
        dependents: card.dependents,
        depends_on: card.depends_on,
        total_dependents: card.total_dependents,
        total_depends_on: card.total_depends_on,
    }
}

fn compact_read_symbol(symbol: &ReadSymbol) -> String {
    let mut parts = Vec::new();
    push_inline(&mut parts, &symbol.lines);
    parts.push(symbol_kind_label(symbol.kind).to_string());
    push_inline(&mut parts, &symbol.qualified_name);
    let mut outline = parts.join(" ");
    let signature = normalize_inline(&symbol.signature);
    if !signature.is_empty() {
        if outline.is_empty() {
            outline.push_str(&signature);
        } else {
            outline.push_str(" :: ");
            outline.push_str(&signature);
        }
    }
    outline
}

fn push_inline(parts: &mut Vec<String>, value: &str) {
    let value = normalize_inline(value);
    if !value.is_empty() {
        parts.push(value);
    }
}

fn normalize_inline(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn symbol_kind_label(kind: SymbolKind) -> &'static str {
    match kind {
        SymbolKind::Function => "function",
        SymbolKind::Method => "method",
        SymbolKind::Class => "class",
        SymbolKind::Struct => "struct",
        SymbolKind::Enum => "enum",
        SymbolKind::Interface => "interface",
        SymbolKind::TypeAlias => "type_alias",
        SymbolKind::Constant => "constant",
        SymbolKind::Unknown => "unknown",
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

fn read_summary(card: Option<&FileCard>) -> Option<String> {
    card.map(|card| card.summary.trim().to_string())
        .filter(|summary| !summary.is_empty())
}

fn read_description(card: Option<&FileCard>) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(card) = card {
        push_labeled(&mut parts, "Role", &card.role);
        push_joined(&mut parts, "Behaviors", &card.primary_behaviors);
        push_joined(&mut parts, "Owns", &card.owns_behaviors);
        push_joined(&mut parts, "Delegates to", &card.delegates_to);
        push_joined(&mut parts, "Side effects", &card.side_effects);
        push_joined(&mut parts, "Blast radius", &card.blast_radius);
    }
    non_empty(parts.join("\n"))
}

fn read_symbols(
    symbols: &[SymbolFact],
    card: Option<&FileCard>,
    source_lines: &[String],
) -> Vec<ReadSymbol> {
    symbols
        .iter()
        .map(|symbol| {
            let symbol_behavior = card
                .and_then(|card| matching_symbol_behavior(symbol, &card.important_symbols))
                .and_then(|behavior| {
                    meaningful_symbol_behavior(&behavior.behavior, &symbol.signature)
                });
            ReadSymbol {
                name: symbol.name.clone(),
                qualified_name: symbol.qualified_name.clone(),
                kind: symbol.kind.clone(),
                signature: symbol.signature.clone(),
                lines: format!("{}-{}", symbol.start_line, symbol.end_line),
                doc: symbol_doc(source_lines, symbol.start_line),
                behavior: symbol_behavior,
            }
        })
        .collect()
}

fn read_chunks(chunks: &[CodeChunkFact]) -> Vec<ReadCodeChunk> {
    chunks
        .iter()
        .map(|chunk| ReadCodeChunk {
            chunk_id: chunk.chunk_id.clone(),
            symbol: chunk.symbol.clone(),
            qualified_name: chunk.qualified_name.clone(),
            kind: chunk.kind,
            signature: chunk.signature.clone(),
            lines: format!("{}-{}", chunk.start_line, chunk.end_line),
            summary_source: chunk.summary_source,
            summary: chunk.summary.clone(),
            doc_summary: chunk.doc_summary.clone(),
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

fn meaningful_symbol_behavior(behavior: &str, signature: &str) -> Option<String> {
    let trimmed = behavior.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed == signature.trim()
        || trimmed.starts_with("fn ")
        || trimmed.starts_with("struct ")
        || trimmed.starts_with("enum ")
        || trimmed.starts_with("impl ")
    {
        return None;
    }
    Some(trimmed.to_string())
}

fn read_imports(imports: &[ImportFact], card: Option<&FileCard>) -> ReadImports {
    let interpreted = card
        .map(|card| card.imports_interpreted.as_slice())
        .unwrap_or_default();

    let mut external = Vec::new();
    let mut internal: BTreeMap<(String, Option<String>), InternalImportGroup> = BTreeMap::new();
    for import in imports {
        if import.is_internal {
            internal
                .entry((import.module.clone(), import.resolved_file_id.clone()))
                .or_insert_with(|| {
                    InternalImportGroup::new(import.module.clone(), import.resolved_file_id.clone())
                })
                .add(import, interpreted);
        } else {
            external.push(import.module.clone());
        }
    }

    external.sort();
    external.dedup();

    ReadImports {
        external: non_empty(external.join(", ")),
        internal: internal
            .into_values()
            .map(InternalImportGroup::into_read_import)
            .collect(),
    }
}

#[derive(Debug, Clone)]
struct InternalImportGroup {
    module: String,
    path: Option<String>,
    names: Vec<String>,
    why: Option<String>,
}

impl InternalImportGroup {
    fn new(module: String, path: Option<String>) -> Self {
        Self {
            module,
            path,
            names: Vec::new(),
            why: None,
        }
    }

    fn add(&mut self, import: &ImportFact, interpreted: &[DependencyInterpretation]) {
        for name in &import.names {
            if !self.names.contains(name) {
                self.names.push(name.clone());
            }
        }
        if self.why.is_none() {
            self.why = import_purpose(import, interpreted);
        }
    }

    fn into_read_import(self) -> ReadInternalImport {
        ReadInternalImport {
            module: self.module,
            path: self.path,
            names: joined_names(&self.names),
            why: self.why,
        }
    }
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

fn joined_names(names: &[String]) -> Option<String> {
    if names.is_empty() {
        None
    } else {
        Some(names.join(", "))
    }
}

#[derive(Debug, Clone, Copy)]
enum DependencySide {
    Incoming,
    Outgoing,
}

fn collapsed_dependency_count(edges: &[EdgeFact], side: DependencySide) -> usize {
    collapsed_dependencies(edges, side).len()
}

fn read_dependencies(edges: &[EdgeFact], side: DependencySide) -> Vec<ReadDependency> {
    collapsed_dependencies(edges, side)
        .into_values()
        .take(MAX_READ_DEPENDENCIES)
        .map(|group| group.into_read_dependency(side))
        .collect()
}

fn collapsed_dependencies(
    edges: &[EdgeFact],
    side: DependencySide,
) -> BTreeMap<String, DependencyGroup> {
    let mut groups: BTreeMap<String, DependencyGroup> = BTreeMap::new();
    for edge in edges {
        if edge.kind == EdgeKind::Contains {
            continue;
        }
        let path = match side {
            DependencySide::Incoming => edge.source_id.clone(),
            DependencySide::Outgoing => edge.target_id.clone(),
        };
        if path.trim().is_empty() {
            continue;
        }
        groups
            .entry(path.clone())
            .or_insert_with(|| DependencyGroup::new(path))
            .add(edge);
    }
    groups
}

#[derive(Debug, Clone)]
struct DependencyGroup {
    path: String,
    relationships: Vec<String>,
    details: Vec<String>,
}

impl DependencyGroup {
    fn new(path: String) -> Self {
        Self {
            path,
            relationships: Vec::new(),
            details: Vec::new(),
        }
    }

    fn add(&mut self, edge: &EdgeFact) {
        let relationship = edge_relationship(edge);
        if !self.relationships.contains(&relationship) {
            self.relationships.push(relationship);
        }
        let detail = edge.detail.trim();
        if !detail.is_empty() && !self.details.iter().any(|existing| existing == detail) {
            self.details.push(detail.to_string());
        }
    }

    fn into_read_dependency(self, side: DependencySide) -> ReadDependency {
        ReadDependency {
            path: self.path,
            relationships: self.relationships.join(", "),
            why: dependency_why(side, &self.relationships, &self.details),
        }
    }
}

fn edge_relationship(edge: &EdgeFact) -> String {
    match edge.kind {
        EdgeKind::Contains => "contains",
        EdgeKind::Imports => "imports",
        EdgeKind::Calls => "calls",
        EdgeKind::References => "references",
        EdgeKind::DependsOn => "depends_on",
    }
    .to_string()
}

fn dependency_why(
    side: DependencySide,
    relationships: &[String],
    details: &[String],
) -> Option<String> {
    if let Some(detail) = details.first().filter(|detail| !detail.trim().is_empty()) {
        return Some(detail.clone());
    }

    let joined = relationships.join(", ");
    let text = match side {
        DependencySide::Incoming => format!("Uses this file via {joined}."),
        DependencySide::Outgoing => format!("This file uses it via {joined}."),
    };
    non_empty(text)
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
