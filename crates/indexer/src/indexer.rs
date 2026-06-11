use anyhow::{Result, bail};
use matryoshka_core_ir::{
    FileCard, FileEnrichmentContext, FileFact, FolderCard, FolderEnrichmentContext, ImportContext,
    RelatedFileContext, RepoCard, RepositorySnapshot, SemanticEntityType, SemanticRecord,
};
use matryoshka_embed_client::Embedder;
use matryoshka_enricher::CodeEnricher;
use matryoshka_parser::{ParserConfig, SourceParser};
use matryoshka_resolver::GraphResolver;
use matryoshka_store_sqlite::MatryoshkaStore;
use rayon::prelude::*;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub struct FullIndexer<E, M> {
    store: MatryoshkaStore,
    enricher: E,
    embedder: M,
    parser_config: ParserConfig,
}

impl<E, M> FullIndexer<E, M>
where
    E: CodeEnricher + Sync,
    M: Embedder + Sync,
{
    pub fn new(store: MatryoshkaStore, enricher: E, embedder: M) -> Self {
        Self {
            store,
            enricher,
            embedder,
            parser_config: ParserConfig::default(),
        }
    }

    pub fn with_parser_config(mut self, parser_config: ParserConfig) -> Self {
        self.parser_config = parser_config;
        self
    }

    pub fn index_repo(&self, repo_root: impl AsRef<Path>) -> Result<IndexSummary> {
        let parser = SourceParser::new(self.parser_config.clone());
        let parsed = parser.parse_repo(repo_root.as_ref())?;
        let snapshot = GraphResolver::resolve(parsed);
        self.store.replace_snapshot(&snapshot)?;
        let file_ids = snapshot
            .files
            .iter()
            .map(|file| file.file_id.clone())
            .collect::<BTreeSet<_>>();
        let folder_ids = snapshot
            .folders
            .iter()
            .map(|folder| folder.folder_id.clone())
            .collect::<BTreeSet<_>>();
        let records = self.refresh_artifacts(
            &snapshot,
            &file_ids,
            &folder_ids,
            true,
            BTreeSet::new(),
            BTreeSet::new(),
        )?;

        Ok(IndexSummary {
            db_path: PathBuf::new(),
            file_count: snapshot.files.len(),
            folder_count: snapshot.folders.len(),
            symbol_count: snapshot.symbols.len(),
            semantic_record_count: records,
            embedding_model: self.embedder.model().into(),
        })
    }

    pub fn update_repo(&self, repo_root: impl AsRef<Path>) -> Result<UpdateSummary> {
        let repo_root = repo_root.as_ref();
        let old_snapshot = load_snapshot(&self.store, repo_root)?;
        if old_snapshot.files.is_empty() {
            let summary = self.index_repo(repo_root)?;
            return Ok(UpdateSummary {
                file_count: summary.file_count,
                folder_count: summary.folder_count,
                symbol_count: summary.symbol_count,
                semantic_record_count: summary.semantic_record_count,
                changed_files: summary.file_count,
                removed_files: 0,
                changed_folders: summary.folder_count,
                repo_card_updated: true,
                embedding_model: summary.embedding_model,
            });
        }

        let parser = SourceParser::new(self.parser_config.clone());
        let parsed = parser.parse_repo(repo_root)?;
        let new_snapshot = GraphResolver::resolve(parsed);
        let delta = compute_delta(&old_snapshot, &new_snapshot);

        if delta.is_noop() {
            self.store.clear_invalidation_queue()?;
            return Ok(UpdateSummary {
                file_count: new_snapshot.files.len(),
                folder_count: new_snapshot.folders.len(),
                symbol_count: new_snapshot.symbols.len(),
                semantic_record_count: self.store.load_all_semantic_records()?.len(),
                changed_files: 0,
                removed_files: 0,
                changed_folders: 0,
                repo_card_updated: false,
                embedding_model: self.embedder.model().into(),
            });
        }

        mark_invalidation_queue(&self.store, &old_snapshot, &new_snapshot, &delta)?;
        apply_structural_delta(&self.store, &old_snapshot, &new_snapshot, &delta)?;
        let semantic_record_count = self.refresh_artifacts(
            &new_snapshot,
            &delta.affected_file_ids,
            &delta.affected_folder_ids,
            delta.repo_card_stale,
            delta.removed_file_ids.clone(),
            delta.removed_folder_ids.clone(),
        )?;
        self.store.clear_invalidation_queue()?;

        Ok(UpdateSummary {
            file_count: new_snapshot.files.len(),
            folder_count: new_snapshot.folders.len(),
            symbol_count: new_snapshot.symbols.len(),
            semantic_record_count,
            changed_files: delta.changed_or_added_file_ids.len(),
            removed_files: delta.removed_file_ids.len(),
            changed_folders: delta.affected_folder_ids.len(),
            repo_card_updated: delta.repo_card_stale,
            embedding_model: self.embedder.model().into(),
        })
    }

    pub fn rebuild_semantic_index(
        &self,
        repo_root: impl AsRef<Path>,
    ) -> Result<SemanticRebuildSummary> {
        let repo_root = repo_root.as_ref();
        let snapshot = load_snapshot(&self.store, repo_root)?;
        if snapshot.files.is_empty() {
            bail!("no indexed files found in store; run index first");
        }

        let file_cards = self.store.load_all_file_cards()?;
        let folder_cards = self.store.load_all_folder_cards()?;
        let repo_card = self.store.load_repo_card(&snapshot.repo_root)?;

        let mut raw_records = raw_semantic_records(&snapshot);
        embed_selected_records(&self.embedder, &mut raw_records, |record| {
            matches!(
                record.entity_type,
                SemanticEntityType::Snippet | SemanticEntityType::Symbol
            )
        })?;

        let mut card_records =
            card_semantic_records(&file_cards, &folder_cards, repo_card.as_ref());
        embed_records(&self.embedder, &mut card_records)?;

        let file_card_record_count = file_cards.len();
        let folder_card_record_count = folder_cards.len();
        let repo_card_record_count = usize::from(repo_card.is_some());

        raw_records.extend(card_records);
        self.store.replace_semantic_records(&raw_records)?;

        Ok(SemanticRebuildSummary {
            semantic_record_count: raw_records.len(),
            file_card_record_count,
            folder_card_record_count,
            repo_card_record_count,
            embedding_model: self.embedder.model().into(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct IndexSummary {
    pub db_path: PathBuf,
    pub file_count: usize,
    pub folder_count: usize,
    pub symbol_count: usize,
    pub semantic_record_count: usize,
    pub embedding_model: String,
}

#[derive(Debug, Clone)]
pub struct UpdateSummary {
    pub file_count: usize,
    pub folder_count: usize,
    pub symbol_count: usize,
    pub semantic_record_count: usize,
    pub changed_files: usize,
    pub removed_files: usize,
    pub changed_folders: usize,
    pub repo_card_updated: bool,
    pub embedding_model: String,
}

#[derive(Debug, Clone)]
pub struct SemanticRebuildSummary {
    pub semantic_record_count: usize,
    pub file_card_record_count: usize,
    pub folder_card_record_count: usize,
    pub repo_card_record_count: usize,
    pub embedding_model: String,
}

#[derive(Debug, Clone)]
struct SnapshotDelta {
    changed_or_added_file_ids: BTreeSet<String>,
    removed_file_ids: BTreeSet<String>,
    affected_file_ids: BTreeSet<String>,
    removed_folder_ids: BTreeSet<String>,
    affected_folder_ids: BTreeSet<String>,
    structural_entity_ids: BTreeSet<String>,
    raw_semantic_paths: BTreeSet<String>,
    card_semantic_paths: BTreeSet<String>,
    repo_card_stale: bool,
}

impl SnapshotDelta {
    fn is_noop(&self) -> bool {
        self.changed_or_added_file_ids.is_empty()
            && self.removed_file_ids.is_empty()
            && self.removed_folder_ids.is_empty()
            && self.affected_folder_ids.is_empty()
    }
}

fn enrichment_concurrency() -> usize {
    std::env::var("MATRYOSHKA_ENRICH_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(6)
}

fn embedding_batch_size() -> usize {
    std::env::var("MATRYOSHKA_EMBED_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(64)
}

pub fn embed_records(embedder: &impl Embedder, records: &mut [SemanticRecord]) -> Result<()> {
    embed_selected_records(embedder, records, |_| true)
}

fn embed_selected_records<F>(
    embedder: &impl Embedder,
    records: &mut [SemanticRecord],
    include: F,
) -> Result<()>
where
    F: Fn(&SemanticRecord) -> bool,
{
    let batch_size = embedding_batch_size();
    let selected_indices = records
        .iter()
        .enumerate()
        .filter_map(|(index, record)| include(record).then_some(index))
        .collect::<Vec<_>>();
    for batch in selected_indices.chunks(batch_size) {
        let inputs = batch
            .iter()
            .map(|index| semantic_embedding_input(&records[*index]))
            .collect::<Vec<_>>();
        let embeddings = embedder.embed(&inputs)?;
        for (index, embedding) in batch.iter().zip(embeddings) {
            records[*index].embedding = Some(embedding);
        }
    }
    Ok(())
}

fn semantic_embedding_input(record: &SemanticRecord) -> String {
    format!(
        "title: {}\npath: {}\n{}",
        record.title, record.path, record.content
    )
}

fn raw_semantic_records(snapshot: &RepositorySnapshot) -> Vec<SemanticRecord> {
    let source_hash_by_file_id = snapshot
        .files
        .iter()
        .map(|file| (file.file_id.as_str(), file.source_hash.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut semantic_records = Vec::new();
    for file in &snapshot.files {
        semantic_records.push(SemanticRecord {
            record_id: format!("semantic:file:{}", file.file_id),
            entity_id: file.file_id.clone(),
            entity_type: SemanticEntityType::File,
            title: format!("File {}", file.path),
            content: raw_file_record_content(file),
            path: file.path.clone(),
            source_hash: file.source_hash.clone(),
            embedding: None,
            metadata: BTreeMap::from([("kind".into(), json!("file_fact"))]),
        });
        for snippet in &file.snippets {
            semantic_records.push(SemanticRecord {
                record_id: format!("semantic:snippet:{}", snippet.snippet_id),
                entity_id: snippet.snippet_id.clone(),
                entity_type: SemanticEntityType::Snippet,
                title: format!("Snippet {} in {}", snippet.title, file.path),
                content: snippet.text.clone(),
                path: file.path.clone(),
                source_hash: file.source_hash.clone(),
                embedding: None,
                metadata: BTreeMap::from([
                    ("file_id".into(), json!(file.file_id)),
                    ("start_line".into(), json!(snippet.start_line)),
                ]),
            });
        }
    }
    for symbol in &snapshot.symbols {
        semantic_records.push(SemanticRecord {
            record_id: format!("semantic:symbol:{}", symbol.symbol_id),
            entity_id: symbol.symbol_id.clone(),
            entity_type: SemanticEntityType::Symbol,
            title: format!("Symbol {} in {}", symbol.qualified_name, symbol.path),
            content: format!(
                "symbol: {}\nkind: {:?}\nsignature: {}\nfile: {}",
                symbol.qualified_name, symbol.kind, symbol.signature, symbol.path
            ),
            path: symbol.path.clone(),
            source_hash: source_hash_by_file_id
                .get(symbol.file_id.as_str())
                .cloned()
                .unwrap_or_default(),
            embedding: None,
            metadata: BTreeMap::from([("kind".into(), json!("symbol_fact"))]),
        });
    }
    semantic_records
}

fn raw_file_record_content(file: &matryoshka_core_ir::FileFact) -> String {
    let imports = file
        .imports
        .iter()
        .map(|import| import.module.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let snippets = file
        .snippets
        .iter()
        .map(|snippet| snippet.title.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "path: {}\nlanguage: {}\nimports: {}\nimportant snippets: {}\nlines: {}",
        file.path, file.language, imports, snippets, file.line_count
    )
}

fn load_snapshot(store: &MatryoshkaStore, repo_root: &Path) -> Result<RepositorySnapshot> {
    Ok(RepositorySnapshot {
        repo_root: store
            .load_repo_root()?
            .unwrap_or_else(|| repo_root.to_string_lossy().to_string()),
        indexed_at: chrono::Utc::now(),
        files: store.load_all_files()?,
        folders: store.load_all_folders()?,
        symbols: store.load_all_symbols()?,
        edges: store.load_all_edges()?,
        semantic_records: store.load_all_semantic_records()?,
    })
}

fn compute_delta(old: &RepositorySnapshot, new: &RepositorySnapshot) -> SnapshotDelta {
    let old_files = old
        .files
        .iter()
        .map(|file| (file.file_id.clone(), file))
        .collect::<BTreeMap<_, _>>();
    let new_files = new
        .files
        .iter()
        .map(|file| (file.file_id.clone(), file))
        .collect::<BTreeMap<_, _>>();
    let old_folders = old
        .folders
        .iter()
        .map(|folder| (folder.folder_id.clone(), folder))
        .collect::<BTreeMap<_, _>>();
    let new_folders = new
        .folders
        .iter()
        .map(|folder| (folder.folder_id.clone(), folder))
        .collect::<BTreeMap<_, _>>();

    let changed_or_added_file_ids = new_files
        .iter()
        .filter_map(|(file_id, file)| match old_files.get(file_id) {
            Some(old_file) if old_file.source_hash == file.source_hash => None,
            _ => Some(file_id.clone()),
        })
        .collect::<BTreeSet<_>>();

    let removed_file_ids = old_files
        .keys()
        .filter(|file_id| !new_files.contains_key(*file_id))
        .cloned()
        .collect::<BTreeSet<_>>();

    let added_folder_ids = new_folders
        .keys()
        .filter(|folder_id| !old_folders.contains_key(*folder_id))
        .cloned()
        .collect::<BTreeSet<_>>();
    let removed_folder_ids = old_folders
        .keys()
        .filter(|folder_id| !new_folders.contains_key(*folder_id))
        .cloned()
        .collect::<BTreeSet<_>>();

    let old_file_contexts = build_file_contexts(old);
    let new_file_contexts = build_file_contexts(new);
    let mut affected_file_ids = changed_or_added_file_ids.clone();
    let mut affected_folder_ids = added_folder_ids.clone();
    let mut raw_semantic_paths = BTreeSet::new();
    let mut card_semantic_paths = added_folder_ids.clone();

    for file_id in changed_or_added_file_ids
        .iter()
        .chain(removed_file_ids.iter())
        .cloned()
    {
        if let Some(file) = old_files.get(&file_id) {
            affected_folder_ids.insert(file.parent_folder_id.clone());
            raw_semantic_paths.insert(file.path.clone());
            card_semantic_paths.insert(file.path.clone());
        }
        if let Some(file) = new_files.get(&file_id) {
            affected_folder_ids.insert(file.parent_folder_id.clone());
            raw_semantic_paths.insert(file.path.clone());
            card_semantic_paths.insert(file.path.clone());
        }

        for neighbor in related_files(old, &old_file_contexts, &file_id)
            .into_iter()
            .chain(related_files(new, &new_file_contexts, &file_id))
        {
            affected_file_ids.insert(neighbor);
        }
    }

    for file_id in &affected_file_ids {
        if let Some(file) = old_files.get(file_id) {
            affected_folder_ids.insert(file.parent_folder_id.clone());
            raw_semantic_paths.insert(file.path.clone());
            card_semantic_paths.insert(file.path.clone());
        }
        if let Some(file) = new_files.get(file_id) {
            affected_folder_ids.insert(file.parent_folder_id.clone());
            raw_semantic_paths.insert(file.path.clone());
            card_semantic_paths.insert(file.path.clone());
        }
    }

    let mut repo_card_stale = !removed_file_ids.is_empty()
        || !removed_folder_ids.is_empty()
        || !added_folder_ids.is_empty();
    if changed_or_added_file_ids.len() >= 4 {
        repo_card_stale = true;
    }

    for file_id in &changed_or_added_file_ids {
        let old_targets = internal_targets(old_files.get(file_id));
        let new_targets = internal_targets(new_files.get(file_id));
        if old_targets != new_targets {
            repo_card_stale = true;
        }
        for target_id in old_targets.symmetric_difference(&new_targets) {
            if let Some(file) = new_files
                .get(target_id)
                .or_else(|| old_files.get(target_id))
            {
                affected_file_ids.insert(file.file_id.clone());
                affected_folder_ids.insert(file.parent_folder_id.clone());
                raw_semantic_paths.insert(file.path.clone());
                card_semantic_paths.insert(file.path.clone());
            }
        }
    }

    for folder_id in &removed_folder_ids {
        card_semantic_paths.insert(folder_id.clone());
    }

    let structural_entity_ids = affected_file_ids
        .iter()
        .cloned()
        .chain(affected_folder_ids.iter().cloned())
        .chain(removed_file_ids.iter().cloned())
        .chain(removed_folder_ids.iter().cloned())
        .collect::<BTreeSet<_>>();

    SnapshotDelta {
        changed_or_added_file_ids,
        removed_file_ids,
        affected_file_ids,
        removed_folder_ids,
        affected_folder_ids,
        structural_entity_ids,
        raw_semantic_paths,
        card_semantic_paths,
        repo_card_stale,
    }
}

fn related_files(
    snapshot: &RepositorySnapshot,
    contexts: &BTreeMap<String, FileEnrichmentContext>,
    file_id: &str,
) -> BTreeSet<String> {
    let mut related = BTreeSet::new();
    if let Some(context) = contexts.get(file_id) {
        for import in &context.internal_imports {
            if let Some(target_id) = &import.resolved_file_id {
                related.insert(target_id.clone());
            }
        }
        for related_file in &context.imported_by_files {
            related.insert(related_file.file_id.clone());
        }
    }
    for file in &snapshot.files {
        if file.file_id == file_id {
            continue;
        }
        if file
            .imports
            .iter()
            .any(|import| import.resolved_file_id.as_deref() == Some(file_id))
        {
            related.insert(file.file_id.clone());
        }
    }
    related
}

fn internal_targets(file: Option<&&FileFact>) -> BTreeSet<String> {
    file.into_iter()
        .flat_map(|file| file.imports.iter())
        .filter_map(|import| import.resolved_file_id.clone())
        .collect()
}

fn mark_invalidation_queue(
    store: &MatryoshkaStore,
    old: &RepositorySnapshot,
    new: &RepositorySnapshot,
    delta: &SnapshotDelta,
) -> Result<()> {
    let old_files = old
        .files
        .iter()
        .map(|file| (file.file_id.clone(), file))
        .collect::<BTreeMap<_, _>>();
    let new_files = new
        .files
        .iter()
        .map(|file| (file.file_id.clone(), file))
        .collect::<BTreeMap<_, _>>();

    for file_id in &delta.changed_or_added_file_ids {
        let parent_folder_id = new_files
            .get(file_id)
            .map(|file| file.parent_folder_id.as_str())
            .or_else(|| {
                old_files
                    .get(file_id)
                    .map(|file| file.parent_folder_id.as_str())
            });
        store.mark_stale("file", file_id, "file content changed")?;
        if let Some(folder_id) = parent_folder_id {
            store.mark_stale("folder", folder_id, "child file changed")?;
        }
    }
    for file_id in &delta.removed_file_ids {
        store.mark_stale("file", file_id, "file removed")?;
    }
    for folder_id in &delta.affected_folder_ids {
        store.mark_stale("folder", folder_id, "folder needs refreshed interpretation")?;
    }
    if delta.repo_card_stale {
        store.mark_stale("repo", "repo", "repository map changed")?;
    }
    Ok(())
}

fn apply_structural_delta(
    store: &MatryoshkaStore,
    old: &RepositorySnapshot,
    new: &RepositorySnapshot,
    delta: &SnapshotDelta,
) -> Result<()> {
    let new_files = new
        .files
        .iter()
        .map(|file| (file.file_id.clone(), file.clone()))
        .collect::<BTreeMap<_, _>>();
    let new_folders = new
        .folders
        .iter()
        .map(|folder| (folder.folder_id.clone(), folder.clone()))
        .collect::<BTreeMap<_, _>>();

    store.delete_file_cards(&delta.removed_file_ids.iter().cloned().collect::<Vec<_>>())?;
    store.delete_folder_cards(&delta.removed_folder_ids.iter().cloned().collect::<Vec<_>>())?;
    store.delete_files(&delta.removed_file_ids.iter().cloned().collect::<Vec<_>>())?;
    store.delete_symbols_for_files(
        &delta
            .changed_or_added_file_ids
            .iter()
            .chain(delta.removed_file_ids.iter())
            .cloned()
            .collect::<Vec<_>>(),
    )?;
    store.delete_folders(&delta.removed_folder_ids.iter().cloned().collect::<Vec<_>>())?;
    store.delete_edges_for_entities(
        &delta
            .structural_entity_ids
            .iter()
            .cloned()
            .collect::<Vec<_>>(),
    )?;
    store.delete_semantic_records_for_paths(
        &delta
            .raw_semantic_paths
            .iter()
            .chain(delta.card_semantic_paths.iter())
            .cloned()
            .collect::<Vec<_>>(),
    )?;

    let changed_files = delta
        .changed_or_added_file_ids
        .iter()
        .filter_map(|file_id| new_files.get(file_id).cloned())
        .collect::<Vec<_>>();
    let changed_symbols = new
        .symbols
        .iter()
        .filter(|symbol| delta.changed_or_added_file_ids.contains(&symbol.file_id))
        .cloned()
        .collect::<Vec<_>>();
    let changed_folders = delta
        .affected_folder_ids
        .iter()
        .filter_map(|folder_id| new_folders.get(folder_id).cloned())
        .collect::<Vec<_>>();
    let changed_edges = new
        .edges
        .iter()
        .filter(|edge| {
            delta.structural_entity_ids.contains(&edge.source_id)
                || delta.structural_entity_ids.contains(&edge.target_id)
        })
        .cloned()
        .collect::<Vec<_>>();
    let changed_raw_records = new
        .semantic_records
        .iter()
        .filter(|record| delta.raw_semantic_paths.contains(&record.path))
        .cloned()
        .collect::<Vec<_>>();

    store.upsert_files(&changed_files)?;
    store.upsert_folders(&changed_folders)?;
    store.upsert_symbols(&changed_symbols)?;
    store.upsert_edges(&changed_edges)?;
    store.upsert_semantic_records(&changed_raw_records)?;

    let _ = old;
    Ok(())
}

impl<E, M> FullIndexer<E, M>
where
    E: CodeEnricher + Sync,
    M: Embedder + Sync,
{
    fn refresh_artifacts(
        &self,
        snapshot: &RepositorySnapshot,
        affected_file_ids: &BTreeSet<String>,
        affected_folder_ids: &BTreeSet<String>,
        repo_card_stale: bool,
        removed_file_ids: BTreeSet<String>,
        removed_folder_ids: BTreeSet<String>,
    ) -> Result<usize> {
        let file_contexts = build_file_contexts(snapshot);
        let folder_contexts = build_folder_contexts(snapshot);
        let enrichment_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(enrichment_concurrency())
            .build()?;

        let file_cards = enrichment_pool.install(|| {
            snapshot
                .files
                .par_iter()
                .filter(|file| affected_file_ids.contains(&file.file_id))
                .map(|file| {
                    let symbols = snapshot
                        .symbols
                        .iter()
                        .filter(|symbol| symbol.file_id == file.file_id)
                        .cloned()
                        .collect::<Vec<_>>();
                    let context = file_contexts
                        .get(&file.file_id)
                        .cloned()
                        .unwrap_or_else(|| empty_file_context(&file.parent_folder_id));
                    self.enricher.enrich_file(file, &symbols, &context)
                })
                .collect::<Result<Vec<_>>>()
        })?;

        for card in &file_cards {
            self.store.upsert_file_card(card)?;
        }

        let folder_cards = enrichment_pool.install(|| {
            let store = self.store.clone();
            snapshot
                .folders
                .par_iter()
                .filter(|folder| affected_folder_ids.contains(&folder.folder_id))
                .map(|folder| {
                    let child_cards = folder
                        .child_file_ids
                        .iter()
                        .filter_map(|file_id| {
                            file_cards
                                .iter()
                                .find(|card| card.file_id == *file_id)
                                .cloned()
                                .or_else(|| store.load_file_card(file_id).ok().flatten())
                        })
                        .collect::<Vec<_>>();
                    let context = folder_contexts
                        .get(&folder.folder_id)
                        .cloned()
                        .unwrap_or_else(empty_folder_context);
                    self.enricher.enrich_folder(folder, &child_cards, &context)
                })
                .collect::<Result<Vec<_>>>()
        })?;

        for card in &folder_cards {
            self.store.upsert_folder_card(card)?;
        }

        if !removed_file_ids.is_empty() {
            self.store
                .delete_file_cards(&removed_file_ids.into_iter().collect::<Vec<_>>())?;
        }
        if !removed_folder_ids.is_empty() {
            self.store
                .delete_folder_cards(&removed_folder_ids.into_iter().collect::<Vec<_>>())?;
        }

        let repo_card = if repo_card_stale {
            let all_folder_cards = snapshot
                .folders
                .iter()
                .map(|folder| {
                    folder_cards
                        .iter()
                        .find(|card| card.folder_id == folder.folder_id)
                        .cloned()
                        .or_else(|| {
                            self.store
                                .load_folder_card(&folder.folder_id)
                                .ok()
                                .flatten()
                        })
                        .unwrap_or_else(|| FolderCard {
                            folder_id: folder.folder_id.clone(),
                            summary: String::new(),
                            responsibility: String::new(),
                            behavior_intents: Vec::new(),
                            edit_intents: Vec::new(),
                            retrieval_tags: Vec::new(),
                            contains_kinds_of_files: Vec::new(),
                            incoming_dependencies_meaning: Vec::new(),
                            outgoing_dependencies_meaning: Vec::new(),
                            key_entrypoints: Vec::new(),
                            common_behaviors: Vec::new(),
                            subareas: Vec::new(),
                            agent_guidance: Vec::new(),
                            search_phrases: Vec::new(),
                            provenance: matryoshka_core_ir::Provenance::source_only(""),
                        })
                })
                .collect::<Vec<_>>();
            let repo_card = self
                .enricher
                .enrich_repo(&snapshot.repo_root, &all_folder_cards)?;
            self.store.upsert_repo_card(&repo_card)?;
            Some(repo_card)
        } else {
            None
        };

        let mut raw_records = selected_embedded_raw_records(snapshot, affected_file_ids);
        embed_records(&self.embedder, &mut raw_records)?;
        self.store.upsert_semantic_records(&raw_records)?;

        let mut card_records =
            card_semantic_records(&file_cards, &folder_cards, repo_card.as_ref());
        embed_records(&self.embedder, &mut card_records)?;
        self.store.upsert_semantic_records(&card_records)?;
        Ok(self.store.load_all_semantic_records()?.len())
    }
}

fn card_semantic_records(
    file_cards: &[FileCard],
    folder_cards: &[FolderCard],
    repo_card: Option<&RepoCard>,
) -> Vec<SemanticRecord> {
    let mut records = Vec::new();
    for card in file_cards {
        records.push(SemanticRecord {
            record_id: format!("semantic:file_card:{}", card.file_id),
            entity_id: card.file_id.clone(),
            entity_type: SemanticEntityType::File,
            title: format!("FileCard {}", card.file_id),
            content: format!(
                "summary: {}\nrole: {}\nownership: {:?}\nowns behaviors: {}\ndelegates to: {}\nbehaviors: {}\nbehavior intents: {}\nedit intents: {}\nretrieval tags: {}\nimports: {}\nused_by: {}\nblast_radius: {}\nread hints: {}\nsearch phrases: {}",
                card.summary,
                card.role,
                card.ownership_kind,
                card.owns_behaviors.join("; "),
                card.delegates_to.join("; "),
                card.primary_behaviors.join("; "),
                card.behavior_intents.join("; "),
                card.edit_intents.join("; "),
                card.retrieval_tags.join("; "),
                card.imports_interpreted
                    .iter()
                    .map(|item| format!("{} -> {}", item.target_path, item.why))
                    .collect::<Vec<_>>()
                    .join("; "),
                card.used_by_interpreted
                    .iter()
                    .map(|item| format!("{} -> {}", item.target_path, item.why))
                    .collect::<Vec<_>>()
                    .join("; "),
                card.blast_radius.join("; "),
                card.agent_read_hints.join("; "),
                card.search_phrases.join("; ")
            ),
            path: card.file_id.clone(),
            source_hash: card.provenance.source_hash.clone(),
            embedding: None,
            metadata: BTreeMap::from([
                ("kind".into(), json!("file_card")),
                ("behavior_intents".into(), json!(card.behavior_intents)),
                ("edit_intents".into(), json!(card.edit_intents)),
                ("retrieval_tags".into(), json!(card.retrieval_tags)),
                ("ownership_kind".into(), json!(card.ownership_kind)),
                ("owns_behaviors".into(), json!(card.owns_behaviors)),
                ("delegates_to".into(), json!(card.delegates_to)),
            ]),
        });
    }
    for card in folder_cards {
        records.push(SemanticRecord {
            record_id: format!("semantic:folder_card:{}", card.folder_id),
            entity_id: card.folder_id.clone(),
            entity_type: SemanticEntityType::Folder,
            title: format!("FolderCard {}", card.folder_id),
            content: format!(
                "summary: {}\nresponsibility: {}\nbehaviors: {}\nbehavior intents: {}\nedit intents: {}\nretrieval tags: {}\nincoming dependencies: {}\noutgoing dependencies: {}\nentrypoints: {}\nguidance: {}\nsearch phrases: {}",
                card.summary,
                card.responsibility,
                card.common_behaviors.join("; "),
                card.behavior_intents.join("; "),
                card.edit_intents.join("; "),
                card.retrieval_tags.join("; "),
                card.incoming_dependencies_meaning.join("; "),
                card.outgoing_dependencies_meaning.join("; "),
                card.key_entrypoints.join("; "),
                card.agent_guidance.join("; "),
                card.search_phrases.join("; ")
            ),
            path: card.folder_id.clone(),
            source_hash: card.provenance.source_hash.clone(),
            embedding: None,
            metadata: BTreeMap::from([
                ("kind".into(), json!("folder_card")),
                ("behavior_intents".into(), json!(card.behavior_intents)),
                ("edit_intents".into(), json!(card.edit_intents)),
                ("retrieval_tags".into(), json!(card.retrieval_tags)),
            ]),
        });
    }
    if let Some(card) = repo_card {
        records.push(repo_card_semantic_record(card));
    }
    records
}

fn repo_card_semantic_record(card: &RepoCard) -> SemanticRecord {
    SemanticRecord {
        record_id: format!("semantic:repo_card:{}", card.repo_root),
        entity_id: card.repo_root.clone(),
        entity_type: SemanticEntityType::Repo,
        title: format!("RepoCard {}", card.repo_root),
        content: format!(
            "summary: {}\nbehavior intents: {}\nedit intents: {}\nretrieval tags: {}\nsubsystems: {}\nflows: {}\nentrypoints: {}\nhigh risk areas: {}\nnavigation hints: {}\nsearch phrases: {}",
            card.summary,
            card.behavior_intents.join("; "),
            card.edit_intents.join("; "),
            card.retrieval_tags.join("; "),
            card.top_level_subsystems
                .iter()
                .map(|item| format!("{} -> {}", item.name, item.responsibility))
                .collect::<Vec<_>>()
                .join("; "),
            card.cross_subsystem_flows.join("; "),
            card.entrypoints.join("; "),
            card.high_risk_areas.join("; "),
            card.agent_navigation_hints.join("; "),
            card.search_phrases.join("; ")
        ),
        path: card.repo_root.clone(),
        source_hash: card.provenance.source_hash.clone(),
        embedding: None,
        metadata: BTreeMap::from([
            ("kind".into(), json!("repo_card")),
            ("behavior_intents".into(), json!(card.behavior_intents)),
            ("edit_intents".into(), json!(card.edit_intents)),
            ("retrieval_tags".into(), json!(card.retrieval_tags)),
        ]),
    }
}

fn selected_embedded_raw_records(
    snapshot: &RepositorySnapshot,
    affected_file_ids: &BTreeSet<String>,
) -> Vec<SemanticRecord> {
    let affected_paths = snapshot
        .files
        .iter()
        .filter(|file| affected_file_ids.contains(&file.file_id))
        .map(|file| file.path.clone())
        .collect::<BTreeSet<_>>();

    snapshot
        .semantic_records
        .iter()
        .filter(|record| affected_paths.contains(&record.path))
        .filter(|record| {
            matches!(
                record.entity_type,
                SemanticEntityType::Snippet | SemanticEntityType::Symbol
            )
        })
        .cloned()
        .collect()
}

fn build_file_contexts(
    snapshot: &matryoshka_core_ir::RepositorySnapshot,
) -> BTreeMap<String, FileEnrichmentContext> {
    let file_by_id = snapshot
        .files
        .iter()
        .map(|file| (file.file_id.clone(), file))
        .collect::<BTreeMap<_, _>>();
    let mut imported_by = BTreeMap::<String, Vec<RelatedFileContext>>::new();

    for file in &snapshot.files {
        for import in file.imports.iter().filter(|import| import.is_internal) {
            if let Some(target_id) = &import.resolved_file_id {
                if let Some(target_file) = file_by_id.get(target_id) {
                    imported_by
                        .entry(target_id.clone())
                        .or_default()
                        .push(RelatedFileContext {
                            file_id: file.file_id.clone(),
                            path: file.path.clone(),
                            relationship: "imported_by".into(),
                            detail: format!(
                                "{} imports {} via {}.",
                                file.path, target_file.path, import.module
                            ),
                        });
                }
            }
        }
    }

    snapshot
        .files
        .iter()
        .map(|file| {
            let sibling_file_ids = snapshot
                .files
                .iter()
                .filter(|candidate| {
                    candidate.parent_folder_id == file.parent_folder_id
                        && candidate.file_id != file.file_id
                })
                .map(|candidate| candidate.file_id.clone())
                .take(12)
                .collect::<Vec<_>>();
            let internal_imports = file
                .imports
                .iter()
                .filter(|import| import.is_internal)
                .map(|import| ImportContext {
                    module: import.module.clone(),
                    names: import.names.clone(),
                    line: import.line,
                    dependency_kind: "internal".into(),
                    resolved_file_id: import.resolved_file_id.clone(),
                    resolved_path: import
                        .resolved_file_id
                        .as_ref()
                        .and_then(|id| file_by_id.get(id))
                        .map(|target| target.path.clone()),
                })
                .collect::<Vec<_>>();
            let external_imports = file
                .imports
                .iter()
                .filter(|import| !import.is_internal)
                .map(|import| ImportContext {
                    module: import.module.clone(),
                    names: import.names.clone(),
                    line: import.line,
                    dependency_kind: "external".into(),
                    resolved_file_id: None,
                    resolved_path: None,
                })
                .collect::<Vec<_>>();

            (
                file.file_id.clone(),
                FileEnrichmentContext {
                    parent_folder_id: file.parent_folder_id.clone(),
                    sibling_file_ids,
                    internal_imports,
                    external_imports,
                    imported_by_files: imported_by.remove(&file.file_id).unwrap_or_default(),
                },
            )
        })
        .collect()
}

fn build_folder_contexts(
    snapshot: &matryoshka_core_ir::RepositorySnapshot,
) -> BTreeMap<String, FolderEnrichmentContext> {
    let file_by_id = snapshot
        .files
        .iter()
        .map(|file| (file.file_id.clone(), file))
        .collect::<BTreeMap<_, _>>();
    let mut incoming = BTreeMap::<String, Vec<RelatedFileContext>>::new();
    let mut outgoing = BTreeMap::<String, Vec<RelatedFileContext>>::new();

    for file in &snapshot.files {
        for import in file.imports.iter().filter(|import| import.is_internal) {
            let Some(target_id) = &import.resolved_file_id else {
                continue;
            };
            let Some(target_file) = file_by_id.get(target_id) else {
                continue;
            };
            if file.parent_folder_id == target_file.parent_folder_id {
                continue;
            }
            outgoing
                .entry(file.parent_folder_id.clone())
                .or_default()
                .push(RelatedFileContext {
                    file_id: target_file.file_id.clone(),
                    path: target_file.path.clone(),
                    relationship: "depends_on_folder".into(),
                    detail: format!(
                        "{} imports {} via {}.",
                        file.path, target_file.path, import.module
                    ),
                });
            incoming
                .entry(target_file.parent_folder_id.clone())
                .or_default()
                .push(RelatedFileContext {
                    file_id: file.file_id.clone(),
                    path: file.path.clone(),
                    relationship: "depended_on_by_folder".into(),
                    detail: format!(
                        "{} depends on {} via {}.",
                        file.path, target_file.path, import.module
                    ),
                });
        }
    }

    snapshot
        .folders
        .iter()
        .map(|folder| {
            let representative_child_files = folder
                .child_file_ids
                .iter()
                .take(8)
                .filter_map(|file_id| file_by_id.get(file_id))
                .map(|file| RelatedFileContext {
                    file_id: file.file_id.clone(),
                    path: file.path.clone(),
                    relationship: "child_file".into(),
                    detail: format!("Representative file inside {}.", folder.path),
                })
                .collect::<Vec<_>>();

            let context = FolderEnrichmentContext {
                parent_folder_id: folder.parent_folder_id.clone(),
                incoming_dependencies: dedupe_related(
                    incoming.remove(&folder.folder_id).unwrap_or_default(),
                ),
                outgoing_dependencies: dedupe_related(
                    outgoing.remove(&folder.folder_id).unwrap_or_default(),
                ),
                representative_child_files,
            };
            (folder.folder_id.clone(), context)
        })
        .collect()
}

fn dedupe_related(items: Vec<RelatedFileContext>) -> Vec<RelatedFileContext> {
    let mut seen = BTreeSet::new();
    items
        .into_iter()
        .filter(|item| seen.insert((item.file_id.clone(), item.detail.clone())))
        .collect()
}

fn empty_file_context(parent_folder_id: &str) -> FileEnrichmentContext {
    FileEnrichmentContext {
        parent_folder_id: parent_folder_id.into(),
        sibling_file_ids: Vec::new(),
        internal_imports: Vec::new(),
        external_imports: Vec::new(),
        imported_by_files: Vec::new(),
    }
}

fn empty_folder_context() -> FolderEnrichmentContext {
    FolderEnrichmentContext {
        parent_folder_id: None,
        incoming_dependencies: Vec::new(),
        outgoing_dependencies: Vec::new(),
        representative_child_files: Vec::new(),
    }
}
