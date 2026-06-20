use anyhow::Result;
use matryoshka_core_ir::{
    ArtifactQualityReport, ChunkSummarySource, CodeChunkFact, FileCard, FileEnrichmentContext,
    FileFact, FolderCard, FolderEnrichmentContext, ImportContext, LateInteractionVector,
    MatryoshkaProgressEvent, RelatedFileContext, RepoCard, RepositorySnapshot,
    RetrievalIndexReport, SemanticEntityType, SemanticRecord,
};
use matryoshka_embed_client::Embedder;
use matryoshka_enricher::{ChunkSummarizer, CodeEnricher, HeuristicChunkSummarizer};
use matryoshka_parser::{ParserConfig, SourceParser};
use matryoshka_resolver::GraphResolver;
use matryoshka_store_sqlite::MatryoshkaStore;
use rayon::prelude::*;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

pub struct FullIndexer<E, M, S = HeuristicChunkSummarizer> {
    store: MatryoshkaStore,
    enricher: E,
    embedder: M,
    chunk_summarizer: S,
    parser_config: ParserConfig,
    chunk_summary_enabled: bool,
}

impl<E, M, S> FullIndexer<E, M, S>
where
    E: CodeEnricher + Sync,
    M: Embedder + Sync,
    S: ChunkSummarizer + Sync,
{
    pub fn new(store: MatryoshkaStore, enricher: E, embedder: M, chunk_summarizer: S) -> Self {
        Self {
            store,
            enricher,
            embedder,
            chunk_summarizer,
            parser_config: ParserConfig::default(),
            chunk_summary_enabled: true,
        }
    }

    pub fn with_parser_config(mut self, parser_config: ParserConfig) -> Self {
        self.parser_config = parser_config;
        self
    }

    pub fn with_chunk_summary_enabled(mut self, enabled: bool) -> Self {
        self.chunk_summary_enabled = enabled;
        self
    }

    pub fn index_repo(&self, repo_root: impl AsRef<Path>) -> Result<IndexSummary> {
        self.index_repo_with_progress(repo_root, |_| {})
    }

    pub fn index_repo_with_progress(
        &self,
        repo_root: impl AsRef<Path>,
        mut progress: impl FnMut(MatryoshkaProgressEvent),
    ) -> Result<IndexSummary> {
        let repo_root = repo_root.as_ref();
        progress(MatryoshkaProgressEvent::Started { total_steps: None });
        let parser = SourceParser::new(self.parser_config.clone());
        let parsed = match parser.parse_repo_with_progress(repo_root, |event| progress(event)) {
            Ok(parsed) => parsed,
            Err(err) => return fail_with_progress("parsing", err, &mut progress),
        };
        let snapshot = GraphResolver::resolve(parsed);
        progress(MatryoshkaProgressEvent::WritingDatabase {
            records_written: Some(
                snapshot.files.len()
                    + snapshot.folders.len()
                    + snapshot.symbols.len()
                    + snapshot.edges.len()
                    + snapshot.semantic_records.len(),
            ),
        });
        if let Err(err) = self.store.replace_snapshot(&snapshot) {
            return fail_with_progress("writing_database", err, &mut progress);
        }
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
        let artifacts = self.refresh_artifacts(
            &snapshot,
            &file_ids,
            &folder_ids,
            true,
            BTreeSet::new(),
            BTreeSet::new(),
            &mut progress,
        )?;

        let summary = IndexSummary {
            db_path: PathBuf::new(),
            file_count: snapshot.files.len(),
            folder_count: snapshot.folders.len(),
            symbol_count: snapshot.symbols.len(),
            semantic_record_count: artifacts.semantic_record_count,
            artifact_quality: artifacts.artifact_quality,
            retrieval_index: artifacts.retrieval_index,
            embedding_model: self.embedder.model().into(),
        };
        progress(MatryoshkaProgressEvent::Completed {
            file_count: summary.file_count,
            folder_count: summary.folder_count,
            symbol_count: summary.symbol_count,
            semantic_record_count: summary.semantic_record_count,
            embedding_model: summary.embedding_model.clone(),
        });
        Ok(summary)
    }

    pub fn update_repo(&self, repo_root: impl AsRef<Path>) -> Result<UpdateSummary> {
        self.update_repo_with_progress(repo_root, |_| {})
    }

    pub fn update_repo_with_progress(
        &self,
        repo_root: impl AsRef<Path>,
        mut progress: impl FnMut(MatryoshkaProgressEvent),
    ) -> Result<UpdateSummary> {
        let repo_root = repo_root.as_ref();
        progress(MatryoshkaProgressEvent::Started { total_steps: None });
        let old_snapshot = match load_snapshot(&self.store, repo_root) {
            Ok(snapshot) => snapshot,
            Err(err) => return fail_with_progress("loading_snapshot", err, &mut progress),
        };
        if old_snapshot.files.is_empty() {
            let summary = self.index_repo_with_progress(repo_root, |event| progress(event))?;
            return Ok(UpdateSummary {
                file_count: summary.file_count,
                folder_count: summary.folder_count,
                symbol_count: summary.symbol_count,
                semantic_record_count: summary.semantic_record_count,
                artifact_quality: summary.artifact_quality,
                retrieval_index: summary.retrieval_index,
                changed_files: summary.file_count,
                removed_files: 0,
                changed_folders: summary.folder_count,
                repo_card_updated: true,
                embedding_model: summary.embedding_model,
            });
        }

        let parser = SourceParser::new(self.parser_config.clone());
        let parsed = match parser.parse_repo_with_progress(repo_root, |event| progress(event)) {
            Ok(parsed) => parsed,
            Err(err) => return fail_with_progress("parsing", err, &mut progress),
        };
        let new_snapshot = GraphResolver::resolve(parsed);
        if let Err(err) = self.store.prune_orphaned_artifacts() {
            return fail_with_progress("pruning_orphaned_artifacts", err, &mut progress);
        }
        let delta = compute_delta(&old_snapshot, &new_snapshot);

        if delta.is_noop() {
            let repair = match artifact_repair_set(&self.store, &new_snapshot) {
                Ok(repair) => repair,
                Err(err) => return fail_with_progress("checking_artifacts", err, &mut progress),
            };
            if !repair.is_empty() {
                let artifacts = self.refresh_artifacts(
                    &new_snapshot,
                    &repair.affected_file_ids,
                    &repair.affected_folder_ids,
                    repair.repo_card_stale,
                    BTreeSet::new(),
                    BTreeSet::new(),
                    &mut progress,
                )?;
                if let Err(err) = self.store.clear_invalidation_queue() {
                    return fail_with_progress("clearing_invalidation_queue", err, &mut progress);
                }
                let summary = UpdateSummary {
                    file_count: new_snapshot.files.len(),
                    folder_count: new_snapshot.folders.len(),
                    symbol_count: new_snapshot.symbols.len(),
                    semantic_record_count: artifacts.semantic_record_count,
                    artifact_quality: artifacts.artifact_quality,
                    retrieval_index: artifacts.retrieval_index,
                    changed_files: 0,
                    removed_files: 0,
                    changed_folders: repair.affected_folder_ids.len(),
                    repo_card_updated: repair.repo_card_stale,
                    embedding_model: self.embedder.model().into(),
                };
                progress(MatryoshkaProgressEvent::Completed {
                    file_count: summary.file_count,
                    folder_count: summary.folder_count,
                    symbol_count: summary.symbol_count,
                    semantic_record_count: summary.semantic_record_count,
                    embedding_model: summary.embedding_model.clone(),
                });
                return Ok(summary);
            }

            if let Err(err) = self.store.clear_invalidation_queue() {
                return fail_with_progress("clearing_invalidation_queue", err, &mut progress);
            }
            let semantic_record_count = match self.store.load_all_semantic_records() {
                Ok(records) => records.len(),
                Err(err) => {
                    return fail_with_progress("loading_semantic_records", err, &mut progress);
                }
            };
            let diagnostics = match self
                .current_index_diagnostics(&new_snapshot.repo_root, &mut progress)
            {
                Ok(diagnostics) => diagnostics,
                Err(err) => return fail_with_progress("checking_index_health", err, &mut progress),
            };
            let summary = UpdateSummary {
                file_count: new_snapshot.files.len(),
                folder_count: new_snapshot.folders.len(),
                symbol_count: new_snapshot.symbols.len(),
                semantic_record_count,
                artifact_quality: diagnostics.artifact_quality,
                retrieval_index: diagnostics.retrieval_index,
                changed_files: 0,
                removed_files: 0,
                changed_folders: 0,
                repo_card_updated: false,
                embedding_model: self.embedder.model().into(),
            };
            progress(MatryoshkaProgressEvent::Completed {
                file_count: summary.file_count,
                folder_count: summary.folder_count,
                symbol_count: summary.symbol_count,
                semantic_record_count: summary.semantic_record_count,
                embedding_model: summary.embedding_model.clone(),
            });
            return Ok(summary);
        }

        if let Err(err) = mark_invalidation_queue(&self.store, &old_snapshot, &new_snapshot, &delta)
        {
            return fail_with_progress("marking_invalidation_queue", err, &mut progress);
        }
        if let Err(err) = apply_structural_delta(&self.store, &old_snapshot, &new_snapshot, &delta)
        {
            return fail_with_progress("applying_structural_delta", err, &mut progress);
        }
        let artifacts = self.refresh_artifacts(
            &new_snapshot,
            &delta.affected_file_ids,
            &delta.affected_folder_ids,
            delta.repo_card_stale,
            delta.removed_file_ids.clone(),
            delta.removed_folder_ids.clone(),
            &mut progress,
        )?;
        if let Err(err) = self.store.clear_invalidation_queue() {
            return fail_with_progress("clearing_invalidation_queue", err, &mut progress);
        }

        let summary = UpdateSummary {
            file_count: new_snapshot.files.len(),
            folder_count: new_snapshot.folders.len(),
            symbol_count: new_snapshot.symbols.len(),
            semantic_record_count: artifacts.semantic_record_count,
            artifact_quality: artifacts.artifact_quality,
            retrieval_index: artifacts.retrieval_index,
            changed_files: delta.changed_or_added_file_ids.len(),
            removed_files: delta.removed_file_ids.len(),
            changed_folders: delta.affected_folder_ids.len(),
            repo_card_updated: delta.repo_card_stale,
            embedding_model: self.embedder.model().into(),
        };
        progress(MatryoshkaProgressEvent::Completed {
            file_count: summary.file_count,
            folder_count: summary.folder_count,
            symbol_count: summary.symbol_count,
            semantic_record_count: summary.semantic_record_count,
            embedding_model: summary.embedding_model.clone(),
        });
        Ok(summary)
    }

    pub fn rebuild_semantic_index(
        &self,
        repo_root: impl AsRef<Path>,
    ) -> Result<SemanticRebuildSummary> {
        self.rebuild_semantic_index_with_progress(repo_root, |_| {})
    }

    pub fn rebuild_semantic_index_with_progress(
        &self,
        repo_root: impl AsRef<Path>,
        mut progress: impl FnMut(MatryoshkaProgressEvent),
    ) -> Result<SemanticRebuildSummary> {
        let repo_root = repo_root.as_ref();
        progress(MatryoshkaProgressEvent::Started { total_steps: None });
        let snapshot = match load_snapshot(&self.store, repo_root) {
            Ok(snapshot) => snapshot,
            Err(err) => return fail_with_progress("loading_snapshot", err, &mut progress),
        };
        if snapshot.files.is_empty() {
            return fail_with_progress(
                "rebuild_semantic",
                anyhow::anyhow!("no indexed files found in store; run index first"),
                &mut progress,
            );
        }

        if let Err(err) = self.store.prune_orphaned_artifacts() {
            return fail_with_progress("pruning_orphaned_artifacts", err, &mut progress);
        }
        let file_cards = match self.store.load_active_file_cards() {
            Ok(cards) => cards,
            Err(err) => return fail_with_progress("loading_file_cards", err, &mut progress),
        };
        let folder_cards = match self.store.load_active_folder_cards() {
            Ok(cards) => cards,
            Err(err) => return fail_with_progress("loading_folder_cards", err, &mut progress),
        };
        let repo_card = match self.store.load_repo_card(&snapshot.repo_root) {
            Ok(card) => card,
            Err(err) => return fail_with_progress("loading_repo_card", err, &mut progress),
        };

        let mut raw_records = raw_semantic_records(&snapshot);
        let raw_batches = selected_batch_count(&raw_records, |record| {
            matches!(
                record.entity_type,
                SemanticEntityType::Snippet | SemanticEntityType::Symbol
            )
        });

        let mut card_records =
            card_semantic_records(&file_cards, &folder_cards, repo_card.as_ref());
        let card_batches = selected_batch_count(&card_records, |_| true);
        let mut embedding_progress = EmbeddingProgress::new(raw_batches + card_batches);

        if let Err(err) = embed_selected_records(
            &self.embedder,
            &mut raw_records,
            |record| {
                matches!(
                    record.entity_type,
                    SemanticEntityType::Snippet | SemanticEntityType::Symbol
                )
            },
            Some(&mut progress),
            Some(&mut embedding_progress),
        ) {
            return fail_with_progress("embedding_raw_records", err, &mut progress);
        }

        if let Err(err) = embed_selected_records(
            &self.embedder,
            &mut card_records,
            |_| true,
            Some(&mut progress),
            Some(&mut embedding_progress),
        ) {
            return fail_with_progress("embedding_card_records", err, &mut progress);
        }

        let file_card_record_count = file_cards.len();
        let folder_card_record_count = folder_cards.len();
        let repo_card_record_count = usize::from(repo_card.is_some());

        raw_records.extend(card_records);
        let record_ids = raw_records
            .iter()
            .map(|record| record.record_id.clone())
            .collect::<Vec<_>>();
        let late_vectors = match build_late_interaction_vectors(&self.embedder, &raw_records) {
            Ok(vectors) => vectors,
            Err(err) => {
                return fail_with_progress("embedding_late_interaction", err, &mut progress);
            }
        };
        progress(MatryoshkaProgressEvent::WritingDatabase {
            records_written: Some(raw_records.len()),
        });
        if let Err(err) = self.store.replace_semantic_records(&raw_records) {
            return fail_with_progress("writing_database", err, &mut progress);
        }
        if let Err(err) = self
            .store
            .replace_late_interaction_vectors(&record_ids, &late_vectors)
        {
            return fail_with_progress("writing_late_interaction", err, &mut progress);
        }

        let artifact_quality =
            quality_report_from_cards(&file_cards, &folder_cards, repo_card.as_ref());
        progress(MatryoshkaProgressEvent::ArtifactQuality {
            report: artifact_quality.clone(),
        });
        let retrieval_index = match self.retrieval_index_report() {
            Ok(report) => report,
            Err(err) => return fail_with_progress("checking_index_health", err, &mut progress),
        };
        progress(MatryoshkaProgressEvent::RetrievalIndexHealth {
            report: retrieval_index.clone(),
        });

        let summary = SemanticRebuildSummary {
            semantic_record_count: raw_records.len(),
            file_card_record_count,
            folder_card_record_count,
            repo_card_record_count,
            artifact_quality,
            retrieval_index,
            embedding_model: self.embedder.model().into(),
        };
        progress(MatryoshkaProgressEvent::Completed {
            file_count: snapshot.files.len(),
            folder_count: snapshot.folders.len(),
            symbol_count: snapshot.symbols.len(),
            semantic_record_count: summary.semantic_record_count,
            embedding_model: summary.embedding_model.clone(),
        });
        Ok(summary)
    }
}

#[derive(Debug, Clone)]
pub struct IndexSummary {
    pub db_path: PathBuf,
    pub file_count: usize,
    pub folder_count: usize,
    pub symbol_count: usize,
    pub semantic_record_count: usize,
    pub artifact_quality: ArtifactQualityReport,
    pub retrieval_index: RetrievalIndexReport,
    pub embedding_model: String,
}

#[derive(Debug, Clone)]
pub struct UpdateSummary {
    pub file_count: usize,
    pub folder_count: usize,
    pub symbol_count: usize,
    pub semantic_record_count: usize,
    pub artifact_quality: ArtifactQualityReport,
    pub retrieval_index: RetrievalIndexReport,
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
    pub artifact_quality: ArtifactQualityReport,
    pub retrieval_index: RetrievalIndexReport,
    pub embedding_model: String,
}

#[derive(Debug, Clone)]
struct ArtifactRefreshReport {
    semantic_record_count: usize,
    artifact_quality: ArtifactQualityReport,
    retrieval_index: RetrievalIndexReport,
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

#[derive(Debug, Clone, Default)]
struct ArtifactRepairSet {
    affected_file_ids: BTreeSet<String>,
    affected_folder_ids: BTreeSet<String>,
    repo_card_stale: bool,
}

impl ArtifactRepairSet {
    fn is_empty(&self) -> bool {
        self.affected_file_ids.is_empty()
            && self.affected_folder_ids.is_empty()
            && !self.repo_card_stale
    }

    fn repair_file(&mut self, file: &FileFact) {
        self.affected_file_ids.insert(file.file_id.clone());
        self.affected_folder_ids
            .insert(file.parent_folder_id.clone());
        self.repo_card_stale = true;
    }

    fn repair_folder(&mut self, folder_id: impl Into<String>) {
        self.affected_folder_ids.insert(folder_id.into());
        self.repo_card_stale = true;
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
    embed_selected_records(embedder, records, |_| true, None, None)
}

fn embed_selected_records<F>(
    embedder: &impl Embedder,
    records: &mut [SemanticRecord],
    include: F,
    mut progress: Option<&mut dyn FnMut(MatryoshkaProgressEvent)>,
    mut embedding_progress: Option<&mut EmbeddingProgress>,
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
        if let Some(state) = embedding_progress.as_deref_mut() {
            state.next_batch_index += 1;
            if let Some(progress) = progress.as_deref_mut() {
                progress(MatryoshkaProgressEvent::EmbeddingBatch {
                    batch_index: state.next_batch_index,
                    total_batches: state.total_batches,
                    records_in_batch: batch.len(),
                });
            }
        }
        let inputs = batch
            .iter()
            .map(|index| semantic_embedding_input(&records[*index]))
            .collect::<Vec<_>>();
        let embeddings = embedder.embed(&inputs)?;
        for (index, embedding) in batch.iter().zip(embeddings) {
            records[*index].embedding = Some(embedding);
        }
        if let Some(state) = embedding_progress.as_deref_mut() {
            if let Some(progress) = progress.as_deref_mut() {
                progress(MatryoshkaProgressEvent::EmbeddedBatch {
                    batch_index: state.next_batch_index,
                    total_batches: state.total_batches,
                    records_in_batch: batch.len(),
                });
            }
        }
    }
    Ok(())
}

fn selected_batch_count<F>(records: &[SemanticRecord], include: F) -> usize
where
    F: Fn(&SemanticRecord) -> bool,
{
    let selected = records.iter().filter(|record| include(record)).count();
    if selected == 0 {
        0
    } else {
        selected.div_ceil(embedding_batch_size())
    }
}

fn semantic_embedding_input(record: &SemanticRecord) -> String {
    format!(
        "title: {}\npath: {}\n{}",
        record.title, record.path, record.content
    )
}

fn late_interaction_enabled() -> bool {
    std::env::var("MATRYOSHKA_LATE_INTERACTION")
        .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
        .unwrap_or(true)
}

fn late_interaction_max_tokens() -> usize {
    std::env::var("MATRYOSHKA_LATE_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(48)
}

fn late_interaction_embedding_input(token: &str) -> String {
    format!("code search token: {token}")
}

fn build_late_interaction_vectors(
    embedder: &impl Embedder,
    records: &[SemanticRecord],
) -> Result<Vec<LateInteractionVector>> {
    if !late_interaction_enabled() || records.is_empty() {
        return Ok(Vec::new());
    }

    let max_tokens = late_interaction_max_tokens();
    let mut token_slots = Vec::<(String, usize, String, f32)>::new();
    for record in records {
        for (ordinal, (token, weight)) in late_interaction_tokens(record, max_tokens)
            .into_iter()
            .enumerate()
        {
            token_slots.push((record.record_id.clone(), ordinal, token, weight));
        }
    }

    let mut vectors = Vec::with_capacity(token_slots.len());
    for batch in token_slots.chunks(embedding_batch_size()) {
        let inputs = batch
            .iter()
            .map(|(_, _, token, _)| late_interaction_embedding_input(token))
            .collect::<Vec<_>>();
        let embeddings = embedder.embed(&inputs)?;
        for ((record_id, ordinal, token, weight), embedding) in batch.iter().zip(embeddings) {
            vectors.push(LateInteractionVector {
                record_id: record_id.clone(),
                token: token.clone(),
                ordinal: *ordinal,
                weight: *weight,
                embedding,
            });
        }
    }
    Ok(vectors)
}

fn late_interaction_tokens(record: &SemanticRecord, max_tokens: usize) -> Vec<(String, f32)> {
    let mut weighted = BTreeMap::<String, f32>::new();
    add_weighted_tokens(&record.path, 1.25, &mut weighted);
    add_weighted_tokens(&record.title, 1.35, &mut weighted);
    add_weighted_tokens(&record.content, 1.0, &mut weighted);
    if matches!(
        record.entity_type,
        SemanticEntityType::Symbol | SemanticEntityType::Snippet
    ) {
        add_weighted_tokens(&record.entity_id, 1.4, &mut weighted);
    }

    let mut tokens = weighted.into_iter().collect::<Vec<_>>();
    tokens.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    tokens.truncate(max_tokens);
    tokens
}

fn add_weighted_tokens(text: &str, weight: f32, out: &mut BTreeMap<String, f32>) {
    for raw in text.split(|ch: char| !ch.is_alphanumeric() && ch != '_') {
        for token in code_identifier_terms(raw) {
            if is_late_stopword(&token) {
                continue;
            }
            let entry = out.entry(token).or_insert(0.0);
            *entry = (*entry + weight).min(4.0);
        }
    }
}

fn code_identifier_terms(raw: &str) -> Vec<String> {
    let normalized = raw.trim_matches('_');
    if normalized.len() < 2 {
        return Vec::new();
    }

    let mut terms = Vec::new();
    let lower = normalized.to_ascii_lowercase();
    if lower.len() >= 2 {
        terms.push(lower);
    }
    for part in normalized.split('_') {
        let part = part.trim();
        if part.len() >= 2 {
            terms.extend(split_camel_terms(part));
        }
    }
    terms.sort();
    terms.dedup();
    terms
}

fn split_camel_terms(value: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    let mut previous_lowercase = false;
    for ch in value.chars() {
        if ch.is_uppercase() && previous_lowercase && !current.is_empty() {
            let term = current.to_ascii_lowercase();
            if term.len() >= 2 {
                terms.push(term);
            }
            current.clear();
        }
        previous_lowercase = ch.is_lowercase() || ch.is_ascii_digit();
        current.push(ch);
    }
    let tail = current.to_ascii_lowercase();
    if tail.len() >= 2 {
        terms.push(tail);
    }
    terms
}

fn is_late_stopword(token: &str) -> bool {
    const STOPWORDS: &[&str] = &[
        "the", "and", "for", "with", "from", "into", "that", "this", "file", "code", "pub", "use",
        "impl", "self", "let", "mut", "fn", "struct", "enum", "class", "def", "return", "none",
        "some", "true", "false",
    ];
    STOPWORDS.contains(&token)
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

fn quality_report_from_cards(
    file_cards: &[FileCard],
    folder_cards: &[FolderCard],
    repo_card: Option<&RepoCard>,
) -> ArtifactQualityReport {
    const SAMPLE_LIMIT: usize = 12;

    let file_cards_with_summary = file_cards
        .iter()
        .filter(|card| has_useful_summary(&card.summary))
        .count();
    let folder_cards_with_summary = folder_cards
        .iter()
        .filter(|card| has_useful_summary(&card.summary))
        .count();
    let empty_file_summary_samples = file_cards
        .iter()
        .filter(|card| !has_useful_summary(&card.summary))
        .map(|card| card.file_id.clone())
        .take(SAMPLE_LIMIT)
        .collect();
    let empty_folder_summary_samples = folder_cards
        .iter()
        .filter(|card| !has_useful_summary(&card.summary))
        .map(|card| card.folder_id.clone())
        .take(SAMPLE_LIMIT)
        .collect();

    ArtifactQualityReport {
        file_cards: file_cards.len(),
        file_cards_with_summary,
        file_cards_empty_summary: file_cards.len().saturating_sub(file_cards_with_summary),
        folder_cards: folder_cards.len(),
        folder_cards_with_summary,
        folder_cards_empty_summary: folder_cards.len().saturating_sub(folder_cards_with_summary),
        repo_card_has_summary: repo_card
            .map(|card| has_useful_summary(&card.summary))
            .unwrap_or(false),
        empty_file_summary_samples,
        empty_folder_summary_samples,
    }
}

fn has_useful_summary(summary: &str) -> bool {
    !summary.trim().is_empty()
}

fn file_card_needs_quality_repair(card: &FileCard) -> bool {
    !has_useful_summary(&card.summary)
}

fn folder_card_needs_quality_repair(card: &FolderCard) -> bool {
    !has_useful_summary(&card.summary)
}

fn repo_card_needs_quality_repair(card: &RepoCard) -> bool {
    !has_useful_summary(&card.summary)
}

#[derive(Debug)]
struct EmbeddingProgress {
    next_batch_index: usize,
    total_batches: usize,
}

impl EmbeddingProgress {
    fn new(total_batches: usize) -> Self {
        Self {
            next_batch_index: 0,
            total_batches,
        }
    }
}

enum ProgressMessage<T> {
    Event(MatryoshkaProgressEvent),
    Finished(Result<T>),
}

fn fail_with_progress<T>(
    stage: &str,
    err: anyhow::Error,
    progress: &mut dyn FnMut(MatryoshkaProgressEvent),
) -> Result<T> {
    progress(MatryoshkaProgressEvent::Failed {
        stage: stage.to_string(),
        message: format!("{err:#}"),
    });
    Err(err)
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
        code_chunks: store.load_all_code_chunks()?,
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

fn artifact_repair_set(
    store: &MatryoshkaStore,
    snapshot: &RepositorySnapshot,
) -> Result<ArtifactRepairSet> {
    let mut repair = ArtifactRepairSet::default();
    let files_by_id = snapshot
        .files
        .iter()
        .map(|file| (file.file_id.as_str(), file))
        .collect::<BTreeMap<_, _>>();
    let files_by_path = snapshot
        .files
        .iter()
        .map(|file| (file.path.as_str(), file))
        .collect::<BTreeMap<_, _>>();

    let file_cards = store.load_active_file_cards()?;
    let file_cards_by_id = file_cards
        .iter()
        .map(|card| (card.file_id.as_str(), card))
        .collect::<BTreeMap<_, _>>();
    for file in &snapshot.files {
        match file_cards_by_id.get(file.file_id.as_str()) {
            Some(card)
                if card.provenance.source_hash == file.source_hash
                    && !file_card_needs_quality_repair(card) => {}
            _ => repair.repair_file(file),
        }
    }

    let folder_cards = store.load_active_folder_cards()?;
    let folder_cards_by_id = folder_cards
        .iter()
        .map(|card| (card.folder_id.as_str(), card))
        .collect::<BTreeMap<_, _>>();
    for folder in &snapshot.folders {
        match folder_cards_by_id.get(folder.folder_id.as_str()) {
            Some(card) if !folder_card_needs_quality_repair(card) => {}
            _ => repair.repair_folder(folder.folder_id.clone()),
        }
    }

    let repo_card = store.load_repo_card(&snapshot.repo_root)?;
    match repo_card.as_ref() {
        Some(card) if !repo_card_needs_quality_repair(card) => {}
        _ => repair.repo_card_stale = true,
    }

    let semantic_records = store.load_all_semantic_records()?;
    let semantic_record_ids = semantic_records
        .iter()
        .map(|record| record.record_id.as_str())
        .collect::<BTreeSet<_>>();
    for record in raw_semantic_records(snapshot) {
        if semantic_record_ids.contains(record.record_id.as_str()) {
            continue;
        }
        if let Some(file) = files_by_path.get(record.path.as_str()) {
            repair.repair_file(file);
        }
    }

    let card_records = card_semantic_records(&file_cards, &folder_cards, repo_card.as_ref());
    for record in card_records {
        if semantic_record_ids.contains(record.record_id.as_str()) {
            continue;
        }
        match record.entity_type {
            SemanticEntityType::File => {
                if let Some(file) = files_by_id.get(record.entity_id.as_str()) {
                    repair.repair_file(file);
                }
            }
            SemanticEntityType::Folder => {
                repair.repair_folder(record.entity_id);
            }
            SemanticEntityType::Repo => {
                repair.repo_card_stale = true;
            }
            SemanticEntityType::Symbol
            | SemanticEntityType::Snippet
            | SemanticEntityType::CodeChunk => {
                if let Some(file) = files_by_path.get(record.path.as_str()) {
                    repair.repair_file(file);
                }
            }
        }
    }

    Ok(repair)
}

impl<E, M, S> FullIndexer<E, M, S>
where
    E: CodeEnricher + Sync,
    M: Embedder + Sync,
    S: ChunkSummarizer + Sync,
{
    fn refresh_artifacts(
        &self,
        snapshot: &RepositorySnapshot,
        affected_file_ids: &BTreeSet<String>,
        affected_folder_ids: &BTreeSet<String>,
        repo_card_stale: bool,
        removed_file_ids: BTreeSet<String>,
        removed_folder_ids: BTreeSet<String>,
        progress: &mut dyn FnMut(MatryoshkaProgressEvent),
    ) -> Result<ArtifactRefreshReport> {
        if let Err(err) = self.store.prune_orphaned_artifacts() {
            return fail_with_progress("pruning_orphaned_artifacts", err, progress);
        }
        let file_contexts = build_file_contexts(snapshot);
        let folder_contexts = build_folder_contexts(snapshot);
        let enrichment_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(enrichment_concurrency())
            .build()
            .map_err(anyhow::Error::from)?;

        let file_cards = match self.collect_file_cards_with_progress(
            snapshot,
            affected_file_ids,
            &file_contexts,
            &enrichment_pool,
            progress,
        ) {
            Ok(cards) => cards,
            Err(err) => return fail_with_progress("enriching_file_cards", err, progress),
        };

        progress(MatryoshkaProgressEvent::WritingDatabase {
            records_written: Some(file_cards.len()),
        });
        for card in &file_cards {
            if let Err(err) = self.store.upsert_file_card(card) {
                return fail_with_progress("writing_database", err, progress);
            }
        }

        let folder_cards = match self.collect_folder_cards_bottom_up(
            snapshot,
            affected_folder_ids,
            &file_cards,
            &folder_contexts,
            &enrichment_pool,
        ) {
            Ok(cards) => cards,
            Err(err) => return fail_with_progress("enriching_folder_cards", err, progress),
        };

        progress(MatryoshkaProgressEvent::WritingDatabase {
            records_written: Some(folder_cards.len()),
        });
        for card in &folder_cards {
            if let Err(err) = self.store.upsert_folder_card(card) {
                return fail_with_progress("writing_database", err, progress);
            }
        }

        if !removed_file_ids.is_empty() {
            if let Err(err) = self
                .store
                .delete_file_cards(&removed_file_ids.into_iter().collect::<Vec<_>>())
            {
                return fail_with_progress("writing_database", err, progress);
            }
        }
        if !removed_folder_ids.is_empty() {
            if let Err(err) = self
                .store
                .delete_folder_cards(&removed_folder_ids.into_iter().collect::<Vec<_>>())
            {
                return fail_with_progress("writing_database", err, progress);
            }
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
            let repo_card = match self
                .enricher
                .enrich_repo(&snapshot.repo_root, &all_folder_cards)
            {
                Ok(card) => card,
                Err(err) => return fail_with_progress("enriching_repo_card", err, progress),
            };
            progress(MatryoshkaProgressEvent::WritingDatabase {
                records_written: Some(1),
            });
            if let Err(err) = self.store.upsert_repo_card(&repo_card) {
                return fail_with_progress("writing_database", err, progress);
            }
            Some(repo_card)
        } else {
            None
        };

        // ---- Milestone 2: summarize code chunks that have no useful doc ----
        let chunk_records =
            match self.refresh_chunk_summaries(snapshot, affected_file_ids, progress) {
                Ok(records) => records,
                Err(err) => return fail_with_progress("summarizing_code_chunks", err, progress),
            };

        let mut raw_records = selected_embedded_raw_records(snapshot, affected_file_ids);
        raw_records.extend(chunk_records);
        let mut card_records =
            card_semantic_records(&file_cards, &folder_cards, repo_card.as_ref());
        let raw_batches = selected_batch_count(&raw_records, |_| true);
        let card_batches = selected_batch_count(&card_records, |_| true);
        let mut embedding_progress = EmbeddingProgress::new(raw_batches + card_batches);

        if let Err(err) = embed_selected_records(
            &self.embedder,
            &mut raw_records,
            |_| true,
            Some(progress),
            Some(&mut embedding_progress),
        ) {
            return fail_with_progress("embedding_raw_records", err, progress);
        }
        let raw_record_ids = raw_records
            .iter()
            .map(|record| record.record_id.clone())
            .collect::<Vec<_>>();
        let raw_late_vectors = match build_late_interaction_vectors(&self.embedder, &raw_records) {
            Ok(vectors) => vectors,
            Err(err) => return fail_with_progress("embedding_late_interaction", err, progress),
        };
        progress(MatryoshkaProgressEvent::WritingDatabase {
            records_written: Some(raw_records.len()),
        });
        if let Err(err) = self.store.upsert_semantic_records(&raw_records) {
            return fail_with_progress("writing_database", err, progress);
        }
        if let Err(err) = self
            .store
            .replace_late_interaction_vectors(&raw_record_ids, &raw_late_vectors)
        {
            return fail_with_progress("writing_late_interaction", err, progress);
        }

        if let Err(err) = embed_selected_records(
            &self.embedder,
            &mut card_records,
            |_| true,
            Some(progress),
            Some(&mut embedding_progress),
        ) {
            return fail_with_progress("embedding_card_records", err, progress);
        }
        let card_record_ids = card_records
            .iter()
            .map(|record| record.record_id.clone())
            .collect::<Vec<_>>();
        let card_late_vectors = match build_late_interaction_vectors(&self.embedder, &card_records)
        {
            Ok(vectors) => vectors,
            Err(err) => return fail_with_progress("embedding_late_interaction", err, progress),
        };
        progress(MatryoshkaProgressEvent::WritingDatabase {
            records_written: Some(card_records.len()),
        });
        if let Err(err) = self.store.upsert_semantic_records(&card_records) {
            return fail_with_progress("writing_database", err, progress);
        }
        if let Err(err) = self
            .store
            .replace_late_interaction_vectors(&card_record_ids, &card_late_vectors)
        {
            return fail_with_progress("writing_late_interaction", err, progress);
        }
        if let Err(err) = self.store.prune_orphaned_artifacts() {
            return fail_with_progress("pruning_orphaned_artifacts", err, progress);
        }

        let semantic_record_count = match self.store.load_all_semantic_records() {
            Ok(records) => records.len(),
            Err(err) => return fail_with_progress("loading_semantic_records", err, progress),
        };
        let diagnostics = match self.current_index_diagnostics(&snapshot.repo_root, progress) {
            Ok(diagnostics) => diagnostics,
            Err(err) => return fail_with_progress("checking_index_health", err, progress),
        };
        Ok(ArtifactRefreshReport {
            semantic_record_count,
            artifact_quality: diagnostics.artifact_quality,
            retrieval_index: diagnostics.retrieval_index,
        })
    }

    fn current_index_diagnostics(
        &self,
        repo_root: &str,
        progress: &mut dyn FnMut(MatryoshkaProgressEvent),
    ) -> Result<ArtifactRefreshReport> {
        let file_cards = self.store.load_active_file_cards()?;
        let folder_cards = self.store.load_active_folder_cards()?;
        let repo_card = self.store.load_repo_card(repo_root)?;
        let artifact_quality =
            quality_report_from_cards(&file_cards, &folder_cards, repo_card.as_ref());
        progress(MatryoshkaProgressEvent::ArtifactQuality {
            report: artifact_quality.clone(),
        });

        let retrieval_index = self.retrieval_index_report()?;
        progress(MatryoshkaProgressEvent::RetrievalIndexHealth {
            report: retrieval_index.clone(),
        });

        Ok(ArtifactRefreshReport {
            semantic_record_count: retrieval_index.semantic_records,
            artifact_quality,
            retrieval_index,
        })
    }

    fn retrieval_index_report(&self) -> Result<RetrievalIndexReport> {
        let stats = self.store.retrieval_index_stats()?;
        Ok(RetrievalIndexReport {
            semantic_records: stats.semantic_records,
            embedded_records: stats.embedded_records,
            fts_records: stats.fts_records,
            late_vector_rows: stats.late_vector_rows,
            records_with_late_vectors: stats.records_with_late_vectors,
            late_interaction_enabled: late_interaction_enabled(),
        })
    }

    fn collect_folder_cards_bottom_up(
        &self,
        snapshot: &RepositorySnapshot,
        affected_folder_ids: &BTreeSet<String>,
        refreshed_file_cards: &[FileCard],
        folder_contexts: &BTreeMap<String, FolderEnrichmentContext>,
        enrichment_pool: &rayon::ThreadPool,
    ) -> Result<Vec<FolderCard>> {
        let active_file_ids = snapshot
            .files
            .iter()
            .map(|file| file.file_id.as_str())
            .collect::<BTreeSet<_>>();
        let active_folder_ids = snapshot
            .folders
            .iter()
            .map(|folder| folder.folder_id.as_str())
            .collect::<BTreeSet<_>>();
        let existing_file_cards = self
            .store
            .load_active_file_cards()?
            .into_iter()
            .filter(|card| active_file_ids.contains(card.file_id.as_str()))
            .map(|card| (card.file_id.clone(), card))
            .collect::<BTreeMap<_, _>>();
        let refreshed_file_cards = refreshed_file_cards
            .iter()
            .map(|card| (card.file_id.clone(), card.clone()))
            .collect::<BTreeMap<_, _>>();
        let existing_folder_cards = self
            .store
            .load_active_folder_cards()?
            .into_iter()
            .filter(|card| active_folder_ids.contains(card.folder_id.as_str()))
            .map(|card| (card.folder_id.clone(), card))
            .collect::<BTreeMap<_, _>>();
        let folders_by_id = snapshot
            .folders
            .iter()
            .map(|folder| (folder.folder_id.clone(), folder))
            .collect::<BTreeMap<_, _>>();
        let mut folders_by_depth = BTreeMap::<usize, Vec<&matryoshka_core_ir::FolderFact>>::new();

        for folder_id in affected_folder_ids {
            if let Some(folder) = folders_by_id.get(folder_id) {
                folders_by_depth
                    .entry(folder_depth(&folder.folder_id))
                    .or_default()
                    .push(*folder);
            }
        }

        let mut refreshed_folder_cards = BTreeMap::<String, FolderCard>::new();
        for (_, folders) in folders_by_depth.into_iter().rev() {
            let cards = enrichment_pool.install(|| {
                folders
                    .par_iter()
                    .map(|folder| {
                        let child_file_cards = folder
                            .child_file_ids
                            .iter()
                            .filter_map(|file_id| {
                                refreshed_file_cards
                                    .get(file_id)
                                    .cloned()
                                    .or_else(|| existing_file_cards.get(file_id).cloned())
                            })
                            .collect::<Vec<_>>();
                        let child_folder_cards = folder
                            .child_folder_ids
                            .iter()
                            .filter_map(|folder_id| {
                                refreshed_folder_cards
                                    .get(folder_id)
                                    .cloned()
                                    .or_else(|| existing_folder_cards.get(folder_id).cloned())
                            })
                            .collect::<Vec<_>>();
                        let context = folder_contexts
                            .get(&folder.folder_id)
                            .cloned()
                            .unwrap_or_else(empty_folder_context);
                        self.enricher.enrich_folder(
                            folder,
                            &child_file_cards,
                            &child_folder_cards,
                            &context,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })?;

            for card in cards {
                refreshed_folder_cards.insert(card.folder_id.clone(), card);
            }
        }

        Ok(refreshed_folder_cards.into_values().collect())
    }

    fn collect_file_cards_with_progress(
        &self,
        snapshot: &RepositorySnapshot,
        affected_file_ids: &BTreeSet<String>,
        file_contexts: &BTreeMap<String, FileEnrichmentContext>,
        enrichment_pool: &rayon::ThreadPool,
        progress: &mut dyn FnMut(MatryoshkaProgressEvent),
    ) -> Result<Vec<FileCard>> {
        let files_to_enrich = snapshot
            .files
            .iter()
            .filter(|file| affected_file_ids.contains(&file.file_id))
            .collect::<Vec<_>>();
        let total_files = files_to_enrich.len();
        if total_files == 0 {
            return Ok(Vec::new());
        }

        let (tx, rx) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(|| {
                let progress_tx = tx.clone();
                let result = enrichment_pool
                    .install(|| {
                        files_to_enrich
                            .par_iter()
                            .enumerate()
                            .map_init(
                                || progress_tx.clone(),
                                |event_tx, (position, file)| {
                                    let index = position + 1;
                                    let path = file.path.clone();
                                    let _ = event_tx.send(ProgressMessage::Event(
                                        MatryoshkaProgressEvent::EnrichingFile {
                                            path: path.clone(),
                                            index,
                                            total_files,
                                        },
                                    ));
                                    let symbols = snapshot
                                        .symbols
                                        .iter()
                                        .filter(|symbol| symbol.file_id == file.file_id)
                                        .cloned()
                                        .collect::<Vec<_>>();
                                    let context =
                                        file_contexts.get(&file.file_id).cloned().unwrap_or_else(
                                            || empty_file_context(&file.parent_folder_id),
                                        );
                                    let card =
                                        self.enricher.enrich_file(file, &symbols, &context)?;
                                    let _ = event_tx.send(ProgressMessage::Event(
                                        MatryoshkaProgressEvent::EnrichedFile {
                                            path,
                                            index,
                                            total_files,
                                        },
                                    ));
                                    Ok((position, card))
                                },
                            )
                            .collect::<Result<Vec<_>>>()
                    })
                    .map(|mut indexed_cards| {
                        indexed_cards.sort_by_key(|(position, _)| *position);
                        indexed_cards
                            .into_iter()
                            .map(|(_, card)| card)
                            .collect::<Vec<_>>()
                    });
                let _ = tx.send(ProgressMessage::Finished(result));
            });

            let mut finished = None;
            while let Ok(message) = rx.recv() {
                match message {
                    ProgressMessage::Event(event) => progress(event),
                    ProgressMessage::Finished(result) => {
                        finished = Some(result);
                        break;
                    }
                }
            }

            finished.unwrap_or_else(|| {
                Err(anyhow::anyhow!(
                    "file enrichment progress channel closed unexpectedly"
                ))
            })
        })
    }

    /// Summarize code chunks that have no useful docstring/doc comment, persist
    /// the updated chunks to the store, and build `code_chunk` semantic records
    /// in the target template for retrieval.
    ///
    /// Only chunks with `summary_source == Empty` (or generic/short docs) are
    /// sent to the LLM. Chunks with useful docs are used directly. Chunks in
    /// files whose `source_hash` is unchanged are skipped entirely.
    fn refresh_chunk_summaries(
        &self,
        snapshot: &RepositorySnapshot,
        affected_file_ids: &BTreeSet<String>,
        progress: &mut dyn FnMut(MatryoshkaProgressEvent),
    ) -> Result<Vec<SemanticRecord>> {
        if !self.chunk_summary_enabled {
            return Ok(Vec::new());
        }

        // Collect chunks that need summarization: affected files + Empty source.
        let affected: BTreeSet<&str> = affected_file_ids.iter().map(String::as_str).collect();
        let chunks_to_summarize: Vec<CodeChunkFact> = snapshot
            .code_chunks
            .iter()
            .filter(|chunk| {
                // Only summarize chunks in affected files (incremental).
                if !affected_file_ids.is_empty() && !affected.contains(chunk.file_id.as_str()) {
                    return false;
                }
                // Only summarize chunks that actually need a generated summary.
                chunk.summary_source == ChunkSummarySource::Empty
            })
            .cloned()
            .collect();

        if chunks_to_summarize.is_empty() {
            // Still build semantic records for all chunks (docs + any previously
            // generated summaries) so they're searchable.
            return Ok(code_chunk_semantic_records(&snapshot.code_chunks));
        }

        progress(MatryoshkaProgressEvent::EnrichingChunks {
            chunk_count: chunks_to_summarize.len(),
        });

        let drafts = match self.chunk_summarizer.summarize_chunks_with_progress(
            &chunks_to_summarize,
            &mut |batch_index, total_batches, chunks_in_batch| {
                progress(MatryoshkaProgressEvent::EnrichingChunkBatch {
                    batch_index,
                    total_batches,
                    chunks_in_batch,
                });
                progress(MatryoshkaProgressEvent::EnrichedChunkBatch {
                    batch_index,
                    total_batches,
                    chunks_in_batch,
                });
            },
        ) {
            Ok(drafts) => drafts,
            Err(_err) => {
                // If LLM summarization fails entirely, fall back to heuristic
                // summaries so chunks still get a (grounded) summary record.
                let fallback = matryoshka_enricher::HeuristicChunkSummarizer;
                fallback
                    .summarize_chunks(&chunks_to_summarize)?
                    .into_iter()
                    .map(|mut d| {
                        // Mark as heuristic so we know it wasn't LLM-generated.
                        d.summary = format!("[heuristic] {}", d.summary);
                        d
                    })
                    .collect()
            }
        };

        // Map drafts back onto chunks by chunk_id.
        let drafts_by_id: BTreeMap<String, String> = drafts
            .into_iter()
            .map(|d| (d.chunk_id, d.summary))
            .collect();

        // Build the full updated chunk list: update summarized chunks, keep
        // the rest as-is.
        let updated_chunks: Vec<CodeChunkFact> = snapshot
            .code_chunks
            .iter()
            .map(|chunk| {
                if let Some(summary) = drafts_by_id.get(&chunk.chunk_id) {
                    let mut updated = chunk.clone();
                    updated.generated_summary = Some(summary.clone());
                    updated.summary = summary.clone();
                    updated.summary_source = ChunkSummarySource::Llm;
                    updated
                } else {
                    chunk.clone()
                }
            })
            .collect();

        // Persist updated chunks to the store.
        progress(MatryoshkaProgressEvent::WritingDatabase {
            records_written: Some(updated_chunks.len()),
        });
        if let Err(err) = self.store.upsert_code_chunks(&updated_chunks) {
            return fail_with_progress("writing_code_chunks", err, progress);
        }

        progress(MatryoshkaProgressEvent::EnrichedChunks {
            chunk_count: drafts_by_id.len(),
        });

        // Build semantic records for ALL chunks (docs + generated).
        Ok(code_chunk_semantic_records(&updated_chunks))
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

/// Build `code_chunk` semantic records in the target template:
///
/// ```text
/// path: crates/foo/src/bar.rs
/// symbol: Foo::handle_resume_countdown
/// kind: method
/// signature: fn handle_resume_countdown(&mut self, ...)
/// summary: Resumes attack mode after handoff, cancels the countdown, and updates state.
/// code:
/// fn handle_resume_countdown(...) {
///     ...
/// }
/// ```
fn code_chunk_semantic_records(chunks: &[CodeChunkFact]) -> Vec<SemanticRecord> {
    chunks
        .iter()
        .filter(|chunk| !chunk.summary.is_empty())
        .map(|chunk| {
            let symbol = chunk
                .qualified_name
                .as_deref()
                .or(chunk.symbol.as_deref())
                .unwrap_or("<unknown>");
            let kind = format!("{:?}", chunk.kind).to_ascii_lowercase();
            let content = format!(
                "path: {}\nsymbol: {}\nkind: {}\nsignature: {}\nsummary: {}\ncode:\n{}",
                chunk.path, symbol, kind, chunk.signature, chunk.summary, chunk.code
            );
            let title = format!("CodeChunk {} in {}", symbol, chunk.path);
            let record_id = format!("semantic:code_chunk:{}", chunk.chunk_id);
            let metadata = BTreeMap::from([
                ("kind".into(), json!("code_chunk")),
                (
                    "summary_source".into(),
                    json!(format!("{:?}", chunk.summary_source).to_ascii_lowercase()),
                ),
                ("symbol_id".into(), json!(chunk.symbol_id)),
                ("qualified_name".into(), json!(chunk.qualified_name)),
                ("start_line".into(), json!(chunk.start_line)),
                ("end_line".into(), json!(chunk.end_line)),
            ]);
            SemanticRecord {
                record_id,
                entity_id: chunk.chunk_id.clone(),
                entity_type: SemanticEntityType::CodeChunk,
                title,
                content,
                path: chunk.path.clone(),
                source_hash: chunk.source_hash.clone(),
                embedding: None,
                metadata,
            }
        })
        .collect()
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

fn folder_depth(folder_id: &str) -> usize {
    if folder_id == "repo" {
        0
    } else {
        folder_id.split('/').count()
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
