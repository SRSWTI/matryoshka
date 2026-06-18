use anyhow::Result;
use matryoshka_core_ir::{
    FileCard, FolderCard, LateInteractionVector, RepoCard, SearchHit, SemanticEntityType,
    SemanticRecord,
};
use matryoshka_embed_client::{Embedder, cosine};
use matryoshka_store_sqlite::MatryoshkaStore;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

use crate::{QueryPlan, RerankCandidate, Reranker, SearchMode, TestPreference, plan_query};

const MAX_MATCHED_SYMBOLS: usize = 12;
const DEFAULT_CANDIDATE_MULTIPLIER: usize = 24;
const LATE_INTERACTION_CANDIDATE_LIMIT: usize = 80;
const LATE_INTERACTION_BOOST: f32 = 0.30;

pub struct SearchEngine<M> {
    store: MatryoshkaStore,
    embedder: M,
    reranker: Option<Box<dyn Reranker>>,
    late_interaction: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchPrewarmSummary {
    pub fts_record_count: usize,
    pub query_count: usize,
    pub warmed_hit_count: usize,
}

impl<M: Embedder> SearchEngine<M> {
    pub fn new(store: MatryoshkaStore, embedder: M) -> Self {
        Self {
            store,
            embedder,
            reranker: None,
            late_interaction: true,
        }
    }

    pub fn with_reranker<R: Reranker + 'static>(mut self, reranker: R) -> Self {
        self.reranker = Some(Box::new(reranker));
        self
    }

    pub fn with_late_interaction(mut self, enabled: bool) -> Self {
        self.late_interaction = enabled;
        self
    }

    pub fn prewarm(
        &self,
        queries: &[String],
        limit_per_query: usize,
    ) -> Result<SearchPrewarmSummary> {
        let fts_record_count = self.store.rebuild_semantic_fts()?;
        let mut warmed_hit_count = 0usize;
        for query in queries {
            warmed_hit_count += self.search(query, limit_per_query)?.len();
        }
        Ok(SearchPrewarmSummary {
            fts_record_count,
            query_count: queries.len(),
            warmed_hit_count,
        })
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let plan = plan_query(query);
        let query_embedding = self
            .embedder
            .embed(&[query.to_string()])?
            .pop()
            .unwrap_or_default();
        let records = self.store.load_all_semantic_records()?;
        let query_tokens = tokens(query);
        let candidates = collect_candidates(
            &self.store,
            &records,
            &query_embedding,
            &query_tokens,
            query,
            limit,
            &plan,
        )?;
        let candidate_records = candidates
            .keys()
            .filter_map(|record_id| records.iter().find(|record| &record.record_id == record_id))
            .cloned()
            .collect::<Vec<_>>();
        let mut hits = candidate_records
            .iter()
            .filter_map(|record| {
                let evidence = candidates
                    .get(&record.record_id)
                    .cloned()
                    .unwrap_or_default();
                score_record(record, &query_embedding, &query_tokens, &plan, &evidence)
            })
            .collect::<Vec<_>>();
        self.apply_late_interaction(&query_tokens, &mut hits)?;
        self.apply_reranker(query, &mut hits, &candidate_records)?;
        hits = collapse_file_hits(hits, &candidate_records);
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.path.cmp(&right.path))
        });
        hits.truncate(limit);
        self.hydrate_hits(&mut hits, &candidate_records)?;
        Ok(hits)
    }

    fn apply_late_interaction(
        &self,
        query_tokens: &[String],
        hits: &mut [SearchHit],
    ) -> Result<()> {
        if !self.late_interaction || query_tokens.is_empty() || hits.is_empty() {
            return Ok(());
        }

        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.path.cmp(&right.path))
        });
        let record_ids = hits
            .iter()
            .take(LATE_INTERACTION_CANDIDATE_LIMIT)
            .map(|hit| hit.record_id.clone())
            .collect::<Vec<_>>();
        let doc_vectors = self.store.load_late_interaction_vectors(&record_ids)?;
        if doc_vectors.is_empty() {
            return Ok(());
        }

        let query_inputs = query_tokens
            .iter()
            .map(|token| late_interaction_embedding_input(token))
            .collect::<Vec<_>>();
        let query_vectors = self.embedder.embed(&query_inputs)?;
        if query_vectors.is_empty() {
            return Ok(());
        }

        for hit in hits {
            let Some(vectors) = doc_vectors.get(&hit.record_id) else {
                continue;
            };
            let score = late_interaction_score(&query_vectors, vectors);
            if score <= 0.0 {
                continue;
            }
            hit.score += score * LATE_INTERACTION_BOOST;
            hit.why_matched
                .push("Late-interaction MaxSim matched indexed code-token vectors".into());
        }

        Ok(())
    }

    fn apply_reranker(
        &self,
        query: &str,
        hits: &mut [SearchHit],
        records: &[SemanticRecord],
    ) -> Result<()> {
        let Some(reranker) = self.reranker.as_ref() else {
            return Ok(());
        };
        if hits.is_empty() {
            return Ok(());
        }

        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.path.cmp(&right.path))
        });
        let records_by_id = records
            .iter()
            .map(|record| (record.record_id.as_str(), record))
            .collect::<BTreeMap<_, _>>();
        let candidates = hits
            .iter()
            .take(40)
            .filter_map(|hit| {
                let record = records_by_id.get(hit.record_id.as_str()).copied()?;
                Some(RerankCandidate::from_record(record, hit.score))
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Ok(());
        }

        let scores = reranker.rerank(query, &candidates)?;
        let scores_by_id = scores
            .into_iter()
            .map(|score| (score.record_id.clone(), score))
            .collect::<BTreeMap<_, _>>();
        for hit in hits {
            let Some(score) = scores_by_id.get(&hit.record_id) else {
                continue;
            };
            hit.score += score.score * 0.24;
            match score.reason.as_deref() {
                Some(reason) => hit
                    .why_matched
                    .push(format!("Reranker preferred this result: {reason}")),
                None => hit
                    .why_matched
                    .push("Reranker preferred this result".into()),
            }
        }

        Ok(())
    }

    fn hydrate_hits(&self, hits: &mut [SearchHit], records: &[SemanticRecord]) -> Result<()> {
        let file_cards = self
            .store
            .load_all_file_cards()?
            .into_iter()
            .map(|card| (card.file_id.clone(), card))
            .collect::<BTreeMap<_, _>>();
        let folder_cards = self
            .store
            .load_all_folder_cards()?
            .into_iter()
            .map(|card| (card.folder_id.clone(), card))
            .collect::<BTreeMap<_, _>>();
        let repo_card = self
            .store
            .load_repo_root()?
            .and_then(|repo_root| self.store.load_repo_card(&repo_root).ok().flatten());
        let records_by_id = records
            .iter()
            .map(|record| (record.record_id.as_str(), record))
            .collect::<BTreeMap<_, _>>();

        for hit in hits {
            let record = records_by_id.get(hit.record_id.as_str()).copied();
            match hit.entity_type {
                matryoshka_core_ir::SemanticEntityType::File => {
                    if let Some(card) = file_cards
                        .get(&hit.path)
                        .or_else(|| file_cards.get(&hit.entity_id))
                    {
                        apply_file_card(hit, card, record);
                    } else if let Some(record) = record {
                        apply_record_fallback(hit, record);
                    }
                }
                matryoshka_core_ir::SemanticEntityType::Folder => {
                    if let Some(card) = folder_cards
                        .get(&hit.path)
                        .or_else(|| folder_cards.get(&hit.entity_id))
                    {
                        apply_folder_card(hit, card);
                    } else if let Some(record) = record {
                        apply_record_fallback(hit, record);
                    }
                }
                matryoshka_core_ir::SemanticEntityType::Repo => {
                    if let Some(card) = repo_card.as_ref() {
                        apply_repo_card(hit, card);
                    } else if let Some(record) = record {
                        apply_record_fallback(hit, record);
                    }
                }
                matryoshka_core_ir::SemanticEntityType::Snippet
                | matryoshka_core_ir::SemanticEntityType::Symbol => {
                    if let Some(card) = file_cards
                        .get(&hit.path)
                        .or_else(|| file_cards.get(&hit.entity_id))
                    {
                        apply_file_card(hit, card, record);
                    } else if let Some(record) = record {
                        apply_record_fallback(hit, record);
                    }
                }
            }
        }

        Ok(())
    }
}

pub fn default_prewarm_queries() -> Vec<String> {
    [
        "repository architecture",
        "where are symbols defined",
        "where should I edit behavior",
        "dependency impact blast radius",
        "tests fixtures integration",
        "read next implementation owner",
    ]
    .into_iter()
    .map(ToString::to_string)
    .collect()
}

#[derive(Debug, Clone, Default)]
struct CandidateEvidence {
    fts_rank: Option<f32>,
    dense_rank: Option<usize>,
    exact_score: f32,
    graph_score: f32,
    card_hint: bool,
}

fn collect_candidates(
    store: &MatryoshkaStore,
    records: &[SemanticRecord],
    query_embedding: &[f32],
    query_tokens: &[String],
    query: &str,
    limit: usize,
    plan: &QueryPlan,
) -> Result<BTreeMap<String, CandidateEvidence>> {
    let candidate_limit = candidate_limit(limit);
    let mut candidates = BTreeMap::<String, CandidateEvidence>::new();

    for hit in store.search_semantic_fts(query, candidate_limit)? {
        let evidence = candidates.entry(hit.record_id).or_default();
        evidence.fts_rank = Some(hit.rank);
    }

    for record in records {
        let exact_score = exact_candidate_score(record, query_tokens, plan);
        if exact_score > 0.0 {
            candidates
                .entry(record.record_id.clone())
                .or_default()
                .exact_score += exact_score;
        }
        if plan.include_repo_card && record_kind(record) == "repo_card" {
            candidates
                .entry(record.record_id.clone())
                .or_default()
                .card_hint = true;
        }
        if plan.include_folder_cards && record_kind(record) == "folder_card" {
            candidates
                .entry(record.record_id.clone())
                .or_default()
                .card_hint = true;
        }
        if plan.include_graph_neighbors {
            let graph_score = graph_candidate_score(record, query_tokens);
            if graph_score > 0.0 {
                candidates
                    .entry(record.record_id.clone())
                    .or_default()
                    .graph_score += graph_score;
            }
        }
    }

    if !query_embedding.is_empty() {
        let mut dense = records
            .iter()
            .filter_map(|record| {
                let embedding = record.embedding.as_ref()?;
                (!embedding.is_empty())
                    .then(|| (record.record_id.clone(), cosine(query_embedding, embedding)))
            })
            .collect::<Vec<_>>();
        dense.sort_by(|left, right| right.1.total_cmp(&left.1));
        for (rank, (record_id, _)) in dense.into_iter().take(candidate_limit).enumerate() {
            candidates.entry(record_id).or_default().dense_rank = Some(rank);
        }
    }

    if candidates.is_empty() {
        for record in records.iter().take(candidate_limit) {
            candidates.entry(record.record_id.clone()).or_default();
        }
    }

    Ok(candidates)
}

fn candidate_limit(limit: usize) -> usize {
    limit
        .max(8)
        .saturating_mul(DEFAULT_CANDIDATE_MULTIPLIER)
        .max(96)
}

fn collapse_file_hits(hits: Vec<SearchHit>, records: &[SemanticRecord]) -> Vec<SearchHit> {
    let records_by_id = records
        .iter()
        .map(|record| (record.record_id.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    let mut groups: BTreeMap<String, CollapsedHit> = BTreeMap::new();
    let mut standalone = Vec::new();

    for hit in hits {
        let Some(record) = records_by_id.get(hit.record_id.as_str()).copied() else {
            standalone.push(hit);
            continue;
        };
        if !is_file_level_result(record) {
            standalone.push(hit);
            continue;
        }

        let key = format!("file:{}", hit.path);
        groups
            .entry(key)
            .or_insert_with(|| CollapsedHit::new(hit.path.clone()))
            .add(hit, record);
    }

    standalone.extend(groups.into_values().filter_map(CollapsedHit::into_hit));
    standalone
}

fn is_file_level_result(record: &SemanticRecord) -> bool {
    matches!(
        record.entity_type,
        SemanticEntityType::File | SemanticEntityType::Symbol | SemanticEntityType::Snippet
    )
}

fn exact_candidate_score(
    record: &SemanticRecord,
    query_tokens: &[String],
    plan: &QueryPlan,
) -> f32 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let haystack = format!("{} {} {}", record.title, record.path, record.content).to_lowercase();
    let token_hits = query_tokens
        .iter()
        .filter(|token| haystack.contains(token.as_str()))
        .count();
    let mut score = token_hits as f32 * 0.05 * plan.lexical_weight;

    if matches!(record.entity_type, SemanticEntityType::Symbol) {
        let symbol = symbol_name_from_record(record)
            .unwrap_or_default()
            .to_lowercase();
        if query_tokens
            .iter()
            .any(|token| !symbol.is_empty() && token.eq_ignore_ascii_case(&symbol))
        {
            score += 0.35 * plan.symbol_weight;
        }
    }

    if query_tokens
        .iter()
        .any(|token| record.path.to_lowercase().contains(token))
    {
        score += 0.08 * plan.lexical_weight;
    }

    score
}

fn graph_candidate_score(record: &SemanticRecord, query_tokens: &[String]) -> f32 {
    let haystack = format!("{} {}", record.content, metadata_text(record)).to_lowercase();
    let mut score = 0.0;
    if query_tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "dependency" | "dependencies" | "depends" | "dependent" | "downstream" | "upstream"
        )
    }) && (haystack.contains("depends")
        || haystack.contains("dependency")
        || haystack.contains("incoming")
        || haystack.contains("outgoing")
        || haystack.contains("used by"))
    {
        score += 0.25;
    }
    if query_tokens
        .iter()
        .any(|token| matches!(token.as_str(), "impact" | "breaks" | "blast"))
        && (haystack.contains("blast_radius") || haystack.contains("blast radius"))
    {
        score += 0.25;
    }
    score
}

fn metadata_text(record: &SemanticRecord) -> String {
    fn collect(value: &Value, out: &mut Vec<String>) {
        match value {
            Value::String(text) => out.push(text.clone()),
            Value::Array(items) => {
                for item in items {
                    collect(item, out);
                }
            }
            Value::Object(map) => {
                for (key, value) in map {
                    out.push(key.clone());
                    collect(value, out);
                }
            }
            Value::Number(number) => out.push(number.to_string()),
            Value::Bool(value) => out.push(value.to_string()),
            Value::Null => {}
        }
    }
    let mut parts = Vec::new();
    for (key, value) in &record.metadata {
        parts.push(key.clone());
        collect(value, &mut parts);
    }
    parts.join(" ")
}

#[derive(Debug, Clone)]
struct CollapsedHit {
    path: String,
    best: Option<SearchHit>,
    file_card: Option<SearchHit>,
    matched_terms: BTreeSet<String>,
    matched_symbols: BTreeSet<String>,
    why: BTreeSet<String>,
    max_score: f32,
    related_hit_count: usize,
}

impl CollapsedHit {
    fn new(path: String) -> Self {
        Self {
            path,
            best: None,
            file_card: None,
            matched_terms: BTreeSet::new(),
            matched_symbols: BTreeSet::new(),
            why: BTreeSet::new(),
            max_score: 0.0,
            related_hit_count: 0,
        }
    }

    fn add(&mut self, hit: SearchHit, record: &SemanticRecord) {
        self.max_score = self.max_score.max(hit.score);
        self.related_hit_count += 1;
        self.matched_terms.extend(hit.matched_terms.iter().cloned());
        self.matched_symbols
            .extend(hit.matched_symbols.iter().cloned());
        self.why.extend(hit.why_matched.iter().cloned());

        if matches!(record.entity_type, SemanticEntityType::Symbol) {
            if let Some(symbol) = symbol_name_from_record(record) {
                self.matched_symbols.insert(symbol);
            }
        }

        if record_kind(record) == "file_card"
            || matches!(record.entity_type, SemanticEntityType::File)
        {
            if self
                .file_card
                .as_ref()
                .map(|existing| hit.score > existing.score)
                .unwrap_or(true)
            {
                self.file_card = Some(hit.clone());
            }
        }

        if self
            .best
            .as_ref()
            .map(|existing| hit.score > existing.score)
            .unwrap_or(true)
        {
            self.best = Some(hit);
        }
    }

    fn into_hit(self) -> Option<SearchHit> {
        let mut hit = self.file_card.or(self.best)?;
        let total_matched_symbols = self.matched_symbols.len();
        let matched_symbols = self
            .matched_symbols
            .iter()
            .take(MAX_MATCHED_SYMBOLS)
            .cloned()
            .collect::<Vec<_>>();
        hit.entity_id = self.path.clone();
        hit.path = self.path;
        hit.entity_type = SemanticEntityType::File;
        hit.title = format!("File {}", hit.path);
        hit.score =
            self.max_score + ((self.related_hit_count.saturating_sub(1) as f32) * 0.015).min(0.06);
        hit.matched_terms = self.matched_terms.into_iter().collect();
        hit.matched_symbols = matched_symbols;
        hit.total_matched_symbols = total_matched_symbols;
        let mut why_matched = self.why.into_iter().collect::<Vec<_>>();
        let reranker_reason_count = why_matched
            .iter()
            .filter(|reason| reason.starts_with("Reranker preferred this result"))
            .count();
        if reranker_reason_count > 1 {
            why_matched.retain(|reason| !reason.starts_with("Reranker preferred this result"));
            why_matched.push(format!(
                "Reranker preferred {reranker_reason_count} matching records in this file"
            ));
        }
        hit.why_matched = why_matched;
        if !hit.matched_symbols.is_empty() {
            if total_matched_symbols > hit.matched_symbols.len() {
                hit.why_matched.push(format!(
                    "Matched {total_matched_symbols} symbols in this file, including: {}",
                    hit.matched_symbols.join(", ")
                ));
            } else {
                hit.why_matched.push(format!(
                    "Matched symbols in this file: {}",
                    hit.matched_symbols.join(", ")
                ));
            }
        }
        if self.related_hit_count > 1 {
            hit.why_matched.push(format!(
                "{count} indexed file, symbol, or snippet records from this file matched and were shown as one result",
                count = self.related_hit_count
            ));
        }
        Some(hit)
    }
}

fn score_record(
    record: &SemanticRecord,
    query_embedding: &[f32],
    query_tokens: &[String],
    plan: &QueryPlan,
    evidence: &CandidateEvidence,
) -> Option<SearchHit> {
    let semantic_score = record
        .embedding
        .as_ref()
        .filter(|embedding| !embedding.is_empty() && !query_embedding.is_empty())
        .map(|embedding| cosine(query_embedding, embedding))
        .unwrap_or_default();
    let mut score = semantic_score * plan.semantic_weight;
    let haystack = format!("{} {} {}", record.title, record.path, record.content).to_lowercase();
    let record_kind = record_kind(record);
    let edit_like_query = is_edit_like_query(query_tokens);
    let behavior_like_query = is_behavior_like_query(query_tokens);
    let entrypoint_query = is_entrypoint_query(query_tokens);
    let folder_query = is_folder_query(query_tokens);
    let repo_query = is_repo_query(query_tokens);
    let test_query = plan.test_preference == TestPreference::Prefer || is_test_query(query_tokens);
    let owner_query = (edit_like_query || behavior_like_query) && !entrypoint_query;
    let facade = looks_like_facade(record);
    let behavior_owner = looks_like_behavior_owner(record);
    let mut why = Vec::new();
    if looks_like_test_path(&record.path)
        && plan.test_preference == TestPreference::Penalize
        && !test_query
    {
        score -= 0.28;
        why.push("Test file de-prioritized because the query does not ask for tests".into());
    } else if looks_like_test_path(&record.path) && plan.test_preference == TestPreference::Prefer {
        score += 0.22;
        why.push("Test file preferred because the query asks for tests".into());
    } else if !looks_like_test_path(&record.path)
        && plan.test_preference == TestPreference::Penalize
        && matches!(
            record.entity_type,
            SemanticEntityType::File | SemanticEntityType::Symbol | SemanticEntityType::Snippet
        )
    {
        score += 0.04;
        why.push("Implementation/source file preferred for a non-test query".into());
    }
    if semantic_score > 0.2 {
        why.push("Summary/content is semantically close to the query".into());
    }
    if let Some(rank) = evidence.fts_rank {
        let fts_boost = (0.24 / (1.0 + rank.abs())).max(0.04) * plan.lexical_weight;
        score += fts_boost;
        why.push(
            "SQLite FTS matched exact query terms in path, title, content, or metadata".into(),
        );
    }
    if let Some(rank) = evidence.dense_rank {
        score += (0.08 / (rank as f32 + 1.0).sqrt()) * plan.semantic_weight;
    }
    if evidence.exact_score > 0.0 {
        score += evidence.exact_score;
        why.push("Exact token, symbol, or path candidate matched the query".into());
    }
    if evidence.graph_score > 0.0 {
        score += evidence.graph_score * plan.graph_weight;
        why.push("Graph/dependency-oriented indexed context matched the query plan".into());
    }
    if evidence.card_hint {
        score += 0.16 * plan.card_weight;
        why.push("Query plan requested repository or folder card context".into());
    }
    let matched_terms = query_tokens
        .iter()
        .filter(|token| haystack.contains(token.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let token_hits = matched_terms.len();
    if token_hits > 0 {
        score += token_hits as f32 * 0.08;
        why.push(format!(
            "Path, title, or indexed text contains: {}",
            matched_terms.join(", ")
        ));
    }
    let behavior_hits = metadata_token_hits(record, "behavior_intents", query_tokens);
    if behavior_hits > 0 {
        let weight = if facade && owner_query { 0.03 } else { 0.07 };
        score += behavior_hits as f32 * weight;
        why.push("Behavior phrases match the query".into());
    }
    let owns_behavior_hits = metadata_token_hits(record, "owns_behaviors", query_tokens);
    if owns_behavior_hits > 0 {
        score += owns_behavior_hits as f32 * if owner_query { 0.14 } else { 0.08 };
        why.push("This result claims ownership of matching behavior".into());
    }
    let edit_hits = metadata_token_hits(record, "edit_intents", query_tokens);
    if edit_hits > 0 {
        let edit_query = query_tokens.iter().any(|token| {
            matches!(
                token.as_str(),
                "edit" | "change" | "debug" | "fix" | "refactor" | "modify" | "add"
            )
        });
        let weight = if facade && owner_query {
            0.035
        } else if edit_query {
            0.10
        } else {
            0.05
        };
        score += edit_hits as f32 * weight;
        why.push("Edit/search phrases match the query".into());
    }
    let tag_hits = metadata_token_hits(record, "retrieval_tags", query_tokens);
    if tag_hits > 0 {
        let weight = if facade && owner_query { 0.025 } else { 0.06 };
        score += tag_hits as f32 * weight;
        why.push("Retrieval tags match the query concepts".into());
    }
    match record_kind {
        "file_card" => {
            score += 0.12 * plan.card_weight;
            why.push("Matched enriched file-card text".into());
            if facade && owner_query {
                score -= 0.55;
                why.push(
                    "Entrypoint/facade result was de-prioritized for an implementation query"
                        .into(),
                );
            } else if behavior_owner && owner_query {
                score += 0.28;
                why.push("Implementation file appears to own the requested behavior".into());
            } else if facade && entrypoint_query {
                score += 0.18;
                why.push("Entrypoint/facade result fits a public-surface query".into());
            }
        }
        "folder_card" => {
            score += 0.08 * plan.card_weight;
            why.push("Matched enriched folder-card text".into());
            if matches!(plan.mode, SearchMode::FindSymbol) && !folder_query {
                score -= 0.38;
                why.push("Folder result was de-prioritized for a file-level symbol query".into());
            }
            if edit_like_query && !folder_query {
                score -= 0.20;
                why.push("Folder result was de-prioritized for a file-level edit query".into());
            }
            if matches!(
                plan.mode,
                SearchMode::ArchitectureOverview | SearchMode::FindBehavior | SearchMode::ReadNext
            ) {
                score += 0.12 * plan.card_weight;
                why.push("Folder card fits the planned search mode".into());
            }
        }
        "repo_card" => {
            score += 0.16 * plan.card_weight;
            why.push("Matched the repository-level map".into());
            if matches!(plan.mode, SearchMode::FindSymbol) && !repo_query {
                score -= 0.48;
                why.push("Repository map was de-prioritized for a file-level symbol query".into());
            }
            if matches!(plan.mode, SearchMode::EditTarget) && !repo_query {
                score -= 0.32;
                why.push("Repository map was de-prioritized for a file-level edit query".into());
            }
            if repo_query || plan.include_repo_card {
                score += 0.30;
                why.push("Repository architecture query fits the repo map".into());
            }
        }
        "file_fact" => {
            score -= 0.05;
            why.push("Raw structural file record matched, but enriched cards are preferred".into());
        }
        "symbol_fact" => {
            if token_hits > 0 {
                score += 0.12 * plan.symbol_weight;
                why.push("Matched a concrete symbol in this file".into());
            }
            if matches!(plan.mode, SearchMode::FindSymbol) {
                score += 0.12 * plan.symbol_weight;
                why.push("Symbol query plan preferred concrete symbol records".into());
            }
        }
        _ => {}
    }
    if haystack.contains("import")
        && query_tokens
            .iter()
            .any(|token| token == "dependency" || token == "import" || token == "resolution")
    {
        score += 0.05;
        why.push("Import/dependency context matches the query".into());
    }
    if query_tokens.iter().any(|token| token == "import")
        && query_tokens
            .iter()
            .any(|token| token == "resolution" || token == "resolve")
        && (haystack.contains("import resolution")
            || haystack.contains("resolve_import")
            || haystack.contains("imports onto concrete repository files"))
    {
        score += 0.14;
        why.push("Import-resolution wording matches directly".into());
    }
    if query_tokens
        .iter()
        .any(|token| token == "depends" || token == "dependent" || token == "downstream")
        && (haystack.contains("used_by")
            || haystack.contains("incoming dependencies")
            || haystack.contains("downstream files")
            || haystack.contains("what depends on"))
    {
        score += 0.12;
        why.push("Downstream/dependency direction matches the query".into());
    }
    if haystack.contains("snippet")
        || matches!(
            record.entity_type,
            matryoshka_core_ir::SemanticEntityType::Snippet
        )
    {
        if token_hits > 0 {
            score += 0.05;
            why.push("Matched a source snippet in this file".into());
        }
    }
    if why.is_empty() && score < 0.15 {
        return None;
    }
    Some(SearchHit {
        entity_id: record.entity_id.clone(),
        record_id: record.record_id.clone(),
        path: record.path.clone(),
        title: record.title.clone(),
        entity_type: record.entity_type.clone(),
        summary: None,
        description: None,
        behaviors: Vec::new(),
        matched_terms,
        matched_symbols: symbol_name_from_record(record).into_iter().collect(),
        total_matched_symbols: usize::from(symbol_name_from_record(record).is_some()),
        score,
        why_matched: why,
    })
}

fn apply_file_card(hit: &mut SearchHit, card: &FileCard, record: Option<&SemanticRecord>) {
    hit.summary = non_empty(card.summary.clone());
    let mut description_parts = Vec::new();
    push_labeled(&mut description_parts, "Role", &card.role);
    let owns = clean_behaviors(card.owns_behaviors.clone());
    if !owns.is_empty() {
        push_labeled(&mut description_parts, "Owns", &owns.join("; "));
    }
    if !card.delegates_to.is_empty() {
        push_labeled(
            &mut description_parts,
            "Delegates to",
            &card.delegates_to.join("; "),
        );
    }
    let side_effects = clean_description_items(&card.side_effects);
    if !side_effects.is_empty() {
        push_labeled(
            &mut description_parts,
            "Side effects",
            &side_effects.join("; "),
        );
    }
    if !card.blast_radius.is_empty() {
        push_labeled(
            &mut description_parts,
            "Blast radius",
            &card.blast_radius.join("; "),
        );
    }
    if let Some(record) = record {
        if matches!(
            record.entity_type,
            matryoshka_core_ir::SemanticEntityType::Snippet
        ) {
            push_labeled(&mut description_parts, "Matched snippet", &record.content);
        } else if matches!(
            record.entity_type,
            matryoshka_core_ir::SemanticEntityType::Symbol
        ) {
            push_labeled(
                &mut description_parts,
                "Matched symbol",
                &symbol_description(card, record),
            );
        }
    }
    hit.description = non_empty(description_parts.join("\n"));
    hit.behaviors = clean_behaviors(prefer_non_empty(&[
        &card.owns_behaviors,
        &card.primary_behaviors,
        &card.behavior_intents,
    ]));
}

fn apply_folder_card(hit: &mut SearchHit, card: &FolderCard) {
    hit.summary = non_empty(card.summary.clone());
    let mut description_parts = Vec::new();
    push_labeled(
        &mut description_parts,
        "Responsibility",
        &card.responsibility,
    );
    if !card.contains_kinds_of_files.is_empty() {
        push_labeled(
            &mut description_parts,
            "Contains",
            &card.contains_kinds_of_files.join("; "),
        );
    }
    if !card.outgoing_dependencies_meaning.is_empty() {
        push_labeled(
            &mut description_parts,
            "Uses",
            &card.outgoing_dependencies_meaning.join("; "),
        );
    }
    if !card.incoming_dependencies_meaning.is_empty() {
        push_labeled(
            &mut description_parts,
            "Used by",
            &card.incoming_dependencies_meaning.join("; "),
        );
    }
    hit.description = non_empty(description_parts.join("\n"));
    hit.behaviors = clean_behaviors(prefer_non_empty(&[
        &card.common_behaviors,
        &card.behavior_intents,
    ]));
}

fn apply_repo_card(hit: &mut SearchHit, card: &RepoCard) {
    hit.summary = non_empty(card.summary.clone());
    let mut description_parts = Vec::new();
    if !card.top_level_subsystems.is_empty() {
        push_labeled(
            &mut description_parts,
            "Subsystems",
            &card
                .top_level_subsystems
                .iter()
                .map(|subsystem| format!("{}: {}", subsystem.name, subsystem.responsibility))
                .collect::<Vec<_>>()
                .join("; "),
        );
    }
    if !card.cross_subsystem_flows.is_empty() {
        push_labeled(
            &mut description_parts,
            "Flows",
            &card.cross_subsystem_flows.join("; "),
        );
    }
    if !card.high_risk_areas.is_empty() {
        push_labeled(
            &mut description_parts,
            "High-risk areas",
            &card.high_risk_areas.join("; "),
        );
    }
    hit.description = non_empty(description_parts.join("\n"));
    hit.behaviors = clean_behaviors(prefer_non_empty(&[
        &card.behavior_intents,
        &card.cross_subsystem_flows,
    ]));
}

fn apply_record_fallback(hit: &mut SearchHit, record: &SemanticRecord) {
    hit.summary = non_empty(record.content.clone());
}

fn symbol_description(card: &FileCard, record: &SemanticRecord) -> String {
    let symbol_note = card
        .important_symbols
        .iter()
        .find(|symbol| {
            symbol.symbol_id == record.entity_id
                || record.title.contains(&symbol.name)
                || record.entity_id.contains(&symbol.name)
        })
        .map(|symbol| format!("{}: {} {}", symbol.name, symbol.role, symbol.behavior));
    match symbol_note {
        Some(note) => format!("{note}\n{}", record.content),
        None => record.content.clone(),
    }
}

fn symbol_name_from_record(record: &SemanticRecord) -> Option<String> {
    if !matches!(record.entity_type, SemanticEntityType::Symbol) {
        return None;
    }
    record
        .entity_id
        .rsplit_once("::")
        .map(|(_, symbol)| symbol)
        .unwrap_or(record.entity_id.as_str())
        .split(':')
        .next()
        .map(str::trim)
        .filter(|symbol| !symbol.is_empty())
        .map(ToString::to_string)
}

fn prefer_non_empty(groups: &[&Vec<String>]) -> Vec<String> {
    groups
        .iter()
        .find(|items| !items.is_empty())
        .map(|items| (*items).clone())
        .unwrap_or_default()
}

fn clean_behaviors(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .filter(|value| !is_generic_behavior(value))
        .take(8)
        .collect()
}

fn clean_description_items(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .filter(|value| !is_generic_behavior(value))
        .collect()
}

fn is_generic_behavior(value: &str) -> bool {
    let normalized = value.to_ascii_lowercase();
    normalized == "provide behavior used by downstream dependents"
        || normalized == "coordinate internal dependencies used by this file"
        || normalized == "coordinate internal dependencies used by this folder"
        || normalized.starts_with("no side effects were proven statically")
        || normalized.starts_with("acts as a general module")
}

fn push_labeled(parts: &mut Vec<String>, label: &str, value: &str) {
    if !value.trim().is_empty() {
        parts.push(format!("{label}: {}", value.trim()));
    }
}

fn non_empty(value: String) -> Option<String> {
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
}

fn record_kind(record: &SemanticRecord) -> &str {
    record
        .metadata
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
}

fn metadata_token_hits(record: &SemanticRecord, key: &str, query_tokens: &[String]) -> usize {
    let Some(items) = record.metadata.get(key).and_then(Value::as_array) else {
        return 0;
    };
    let joined = items
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();

    query_tokens
        .iter()
        .filter(|token| joined.contains(token.as_str()))
        .count()
}

fn has_metadata_tag(record: &SemanticRecord, key: &str, needle: &str) -> bool {
    record
        .metadata
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .any(|item| item.eq_ignore_ascii_case(needle))
        })
        .unwrap_or(false)
}

fn looks_like_facade(record: &SemanticRecord) -> bool {
    metadata_string(record, "ownership_kind").as_deref() == Some("facade")
        || record.path.ends_with("/lib.rs")
        || record.path.ends_with("/mod.rs")
        || has_metadata_tag(record, "retrieval_tags", "artifact:facade")
}

fn looks_like_behavior_owner(record: &SemanticRecord) -> bool {
    metadata_string(record, "ownership_kind").as_deref() == Some("implementation")
        || metadata_array_len(record, "owns_behaviors") > 0
        || looks_like_implementation(record)
}

fn looks_like_implementation(record: &SemanticRecord) -> bool {
    !looks_like_facade(record)
        && (has_metadata_tag(record, "retrieval_tags", "artifact:implementation")
            || record.path.ends_with(".rs")
            || record.path.ends_with(".py")
            || record.path.ends_with(".ts")
            || record.path.ends_with(".tsx"))
}

fn metadata_string(record: &SemanticRecord, key: &str) -> Option<String> {
    record
        .metadata
        .get(key)
        .and_then(Value::as_str)
        .map(ToString::to_string)
}

fn metadata_array_len(record: &SemanticRecord, key: &str) -> usize {
    record
        .metadata
        .get(key)
        .and_then(Value::as_array)
        .map(|items| items.len())
        .unwrap_or(0)
}

fn is_edit_like_query(query_tokens: &[String]) -> bool {
    query_tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "edit"
                | "change"
                | "debug"
                | "fix"
                | "refactor"
                | "modify"
                | "add"
                | "remove"
                | "implement"
                | "update"
        )
    })
}

fn is_behavior_like_query(query_tokens: &[String]) -> bool {
    query_tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "behavior" | "flow" | "logic" | "responsibility" | "resolve" | "resolution"
        )
    })
}

fn is_entrypoint_query(query_tokens: &[String]) -> bool {
    query_tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "entrypoint" | "exports" | "export" | "reexport" | "public" | "surface" | "api"
        )
    })
}

fn is_folder_query(query_tokens: &[String]) -> bool {
    query_tokens
        .iter()
        .any(|token| matches!(token.as_str(), "folder" | "module" | "area" | "subsystem"))
}

fn is_repo_query(query_tokens: &[String]) -> bool {
    query_tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "repo" | "repository" | "architecture" | "overview" | "subsystem" | "subsystems"
        )
    })
}

fn is_test_query(query_tokens: &[String]) -> bool {
    query_tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "test"
                | "tests"
                | "testing"
                | "pytest"
                | "unit"
                | "integration"
                | "spec"
                | "fixture"
                | "mock"
                | "assert"
                | "coverage"
        )
    })
}

fn looks_like_test_path(path: &str) -> bool {
    path.starts_with("tests/")
        || path.contains("/tests/")
        || path.contains("/__tests__/")
        || path.ends_with("_test.rs")
        || path.ends_with("_test.py")
        || path.contains(".test.")
        || path.contains(".spec.")
}

fn late_interaction_embedding_input(token: &str) -> String {
    format!("code search token: {token}")
}

fn late_interaction_score(
    query_vectors: &[Vec<f32>],
    doc_vectors: &[LateInteractionVector],
) -> f32 {
    if query_vectors.is_empty() || doc_vectors.is_empty() {
        return 0.0;
    }

    let mut total = 0.0;
    let mut matched = 0usize;
    for query_vector in query_vectors {
        if query_vector.is_empty() {
            continue;
        }
        let best = doc_vectors
            .iter()
            .filter(|doc| !doc.embedding.is_empty() && doc.embedding.len() == query_vector.len())
            .map(|doc| {
                let weight = doc.weight.max(0.1).sqrt();
                cosine(query_vector, &doc.embedding) * weight
            })
            .max_by(|left, right| left.total_cmp(right))
            .unwrap_or(0.0);
        if best > 0.0 {
            total += best;
            matched += 1;
        }
    }

    if matched == 0 {
        0.0
    } else {
        (total / matched as f32).clamp(0.0, 1.5)
    }
}

fn tokens(query: &str) -> Vec<String> {
    const STOPWORDS: &[&str] = &[
        "the", "and", "for", "with", "from", "into", "that", "this", "what", "where", "when",
        "which", "does", "file", "code", "handled", "about", "there", "their",
    ];

    query
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .map(|token| token.to_lowercase())
        .filter(|token| token.len() > 2)
        .filter(|token| !STOPWORDS.contains(&token.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use matryoshka_core_ir::{SemanticEntityType, SemanticRecord};
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn collapse_file_hits_keeps_one_result_with_matched_symbols() {
        let file_record = SemanticRecord {
            record_id: "file".into(),
            entity_id: "octane/cli/mesh.py".into(),
            entity_type: SemanticEntityType::File,
            title: "FileCard octane/cli/mesh.py".into(),
            content: "mesh peer discovery".into(),
            path: "octane/cli/mesh.py".into(),
            source_hash: "a".into(),
            embedding: Some(vec![1.0]),
            metadata: BTreeMap::from([("kind".into(), json!("file_card"))]),
        };
        let symbol_record = SemanticRecord {
            record_id: "symbol".into(),
            entity_id: "octane/cli/mesh.py::resolve_peer:448".into(),
            entity_type: SemanticEntityType::Symbol,
            title: "Symbol resolve_peer in octane/cli/mesh.py".into(),
            content: "resolve peer endpoint".into(),
            path: "octane/cli/mesh.py".into(),
            source_hash: "a".into(),
            embedding: Some(vec![1.0]),
            metadata: BTreeMap::from([("kind".into(), json!("symbol_fact"))]),
        };
        let records = vec![file_record, symbol_record];
        let hits = vec![
            SearchHit {
                entity_id: "octane/cli/mesh.py".into(),
                record_id: "file".into(),
                path: "octane/cli/mesh.py".into(),
                title: "FileCard octane/cli/mesh.py".into(),
                entity_type: SemanticEntityType::File,
                summary: None,
                description: None,
                behaviors: Vec::new(),
                matched_terms: vec!["mesh".into()],
                matched_symbols: Vec::new(),
                total_matched_symbols: 0,
                score: 0.4,
                why_matched: vec!["Matched enriched file-card text".into()],
            },
            SearchHit {
                entity_id: "octane/cli/mesh.py::resolve_peer:448".into(),
                record_id: "symbol".into(),
                path: "octane/cli/mesh.py".into(),
                title: "Symbol resolve_peer in octane/cli/mesh.py".into(),
                entity_type: SemanticEntityType::Symbol,
                summary: None,
                description: None,
                behaviors: Vec::new(),
                matched_terms: vec!["mesh".into()],
                matched_symbols: vec!["resolve_peer".into()],
                total_matched_symbols: 1,
                score: 0.5,
                why_matched: vec!["Matched a concrete symbol in this file".into()],
            },
        ];

        let collapsed = collapse_file_hits(hits, &records);

        assert_eq!(collapsed.len(), 1);
        assert_eq!(collapsed[0].entity_type, SemanticEntityType::File);
        assert_eq!(collapsed[0].path, "octane/cli/mesh.py");
        assert_eq!(collapsed[0].matched_symbols, vec!["resolve_peer"]);
        assert!(
            collapsed[0]
                .why_matched
                .iter()
                .any(|why| { why.contains("2 indexed file, symbol, or snippet records") })
        );
    }

    #[test]
    fn behavior_cleanup_drops_generic_fallback_phrases() {
        let cleaned = clean_behaviors(vec![
            "Provide behavior used by downstream dependents".into(),
            "Coordinate internal dependencies used by this file".into(),
            "discovers mesh peers across LAN and Tailscale".into(),
        ]);

        assert_eq!(
            cleaned,
            vec!["discovers mesh peers across LAN and Tailscale"]
        );
    }

    #[test]
    fn late_interaction_score_uses_maxsim_over_doc_tokens() {
        let query_vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc_vectors = vec![
            LateInteractionVector {
                record_id: "a".into(),
                token: "search".into(),
                ordinal: 0,
                weight: 1.0,
                embedding: vec![1.0, 0.0],
            },
            LateInteractionVector {
                record_id: "a".into(),
                token: "ranking".into(),
                ordinal: 1,
                weight: 1.0,
                embedding: vec![0.0, 1.0],
            },
        ];

        let score = late_interaction_score(&query_vectors, &doc_vectors);

        assert!(score > 0.99);
    }

    #[test]
    fn behavior_edit_queries_prefer_implementation_files_over_facades() {
        let query_tokens = tokens("debug import resolution behavior");
        let query_embedding = vec![1.0, 0.0];

        let facade = SemanticRecord {
            record_id: "1".into(),
            entity_id: "resolver/src/lib.rs".into(),
            entity_type: SemanticEntityType::File,
            title: "FileCard resolver/src/lib.rs".into(),
            content: "import resolution behavior debug exports".into(),
            path: "resolver/src/lib.rs".into(),
            source_hash: "a".into(),
            embedding: Some(vec![1.0, 0.0]),
            metadata: BTreeMap::from([
                ("kind".into(), json!("file_card")),
                (
                    "retrieval_tags".into(),
                    json!(["artifact:facade", "behavior:import-resolution"]),
                ),
                ("ownership_kind".into(), json!("facade")),
                (
                    "edit_intents".into(),
                    json!([
                        "debug import resolution behavior",
                        "change import resolution behavior",
                        "fix import resolution behavior",
                        "refactor import resolution behavior"
                    ]),
                ),
                (
                    "delegates_to".into(),
                    json!(["resolver/src/graph_resolver.rs"]),
                ),
            ]),
        };

        let implementation = SemanticRecord {
            record_id: "2".into(),
            entity_id: "resolver/src/graph_resolver.rs".into(),
            entity_type: SemanticEntityType::File,
            title: "FileCard resolver/src/graph_resolver.rs".into(),
            content: "import resolution behavior debug implementation".into(),
            path: "resolver/src/graph_resolver.rs".into(),
            source_hash: "b".into(),
            embedding: Some(vec![1.0, 0.0]),
            metadata: BTreeMap::from([
                ("kind".into(), json!("file_card")),
                (
                    "retrieval_tags".into(),
                    json!(["artifact:implementation", "behavior:import-resolution"]),
                ),
                ("ownership_kind".into(), json!("implementation")),
                (
                    "owns_behaviors".into(),
                    json!(["import resolution behavior"]),
                ),
                ("edit_intents".into(), json!(["debug import resolution"])),
            ]),
        };

        let plan = plan_query("debug import resolution behavior");
        let evidence = CandidateEvidence::default();
        let facade_hit =
            score_record(&facade, &query_embedding, &query_tokens, &plan, &evidence).unwrap();
        let implementation_hit = score_record(
            &implementation,
            &query_embedding,
            &query_tokens,
            &plan,
            &evidence,
        )
        .unwrap();

        assert!(implementation_hit.score > facade_hit.score);
        assert!(
            implementation_hit
                .why_matched
                .iter()
                .any(|item| item.contains("Implementation file appears to own"))
        );
        assert!(
            facade_hit
                .why_matched
                .iter()
                .any(|item| item.contains("facade result was de-prioritized"))
        );
    }

    #[test]
    fn symbol_queries_prefer_concrete_files_over_matching_repo_cards() {
        let query = "where is advisor called before implementation";
        let query_tokens = tokens(query);
        let query_embedding = vec![1.0, 0.0];
        let plan = plan_query(query);
        assert_eq!(plan.mode, SearchMode::FindSymbol);

        let file = SemanticRecord {
            record_id: "file".into(),
            entity_id: "src/main.rs".into(),
            entity_type: SemanticEntityType::File,
            title: "File src/main.rs".into(),
            content: "advisor call implementation transcript_for_advisor".into(),
            path: "src/main.rs".into(),
            source_hash: "a".into(),
            embedding: Some(vec![1.0, 0.0]),
            metadata: BTreeMap::from([("kind".into(), json!("file_card"))]),
        };
        let repo = SemanticRecord {
            record_id: "repo".into(),
            entity_id: "repo".into(),
            entity_type: SemanticEntityType::Repo,
            title: "RepoCard repo".into(),
            content: "before implementation advisor repository overview".into(),
            path: "repo".into(),
            source_hash: "a".into(),
            embedding: Some(vec![1.0, 0.0]),
            metadata: BTreeMap::from([("kind".into(), json!("repo_card"))]),
        };
        let evidence = CandidateEvidence {
            fts_rank: Some(0.0),
            exact_score: 0.2,
            ..CandidateEvidence::default()
        };

        let file_hit = score_record(&file, &query_embedding, &query_tokens, &plan, &evidence)
            .expect("file should score");
        let repo_hit = score_record(&repo, &query_embedding, &query_tokens, &plan, &evidence)
            .expect("repo should score");

        assert!(file_hit.score > repo_hit.score);
        assert!(
            repo_hit
                .why_matched
                .iter()
                .any(|item| item.contains("file-level symbol query"))
        );
    }

    #[test]
    fn entrypoint_queries_can_still_prefer_facades() {
        let query_tokens = tokens("resolver public exports entrypoint");
        let query_embedding = vec![1.0, 0.0];

        let facade = SemanticRecord {
            record_id: "1".into(),
            entity_id: "resolver/src/lib.rs".into(),
            entity_type: SemanticEntityType::File,
            title: "FileCard resolver/src/lib.rs".into(),
            content: "resolver public exports entrypoint reexport surface".into(),
            path: "resolver/src/lib.rs".into(),
            source_hash: "a".into(),
            embedding: Some(vec![1.0, 0.0]),
            metadata: BTreeMap::from([
                ("kind".into(), json!("file_card")),
                ("ownership_kind".into(), json!("facade")),
                ("retrieval_tags".into(), json!(["artifact:facade"])),
            ]),
        };

        let implementation = SemanticRecord {
            record_id: "2".into(),
            entity_id: "resolver/src/graph_resolver.rs".into(),
            entity_type: SemanticEntityType::File,
            title: "FileCard resolver/src/graph_resolver.rs".into(),
            content: "resolver graph implementation import resolution".into(),
            path: "resolver/src/graph_resolver.rs".into(),
            source_hash: "b".into(),
            embedding: Some(vec![1.0, 0.0]),
            metadata: BTreeMap::from([
                ("kind".into(), json!("file_card")),
                ("ownership_kind".into(), json!("implementation")),
                ("owns_behaviors".into(), json!(["import resolution"])),
            ]),
        };

        let plan = plan_query("resolver public exports entrypoint");
        let evidence = CandidateEvidence::default();
        let facade_hit =
            score_record(&facade, &query_embedding, &query_tokens, &plan, &evidence).unwrap();
        let implementation_hit = score_record(
            &implementation,
            &query_embedding,
            &query_tokens,
            &plan,
            &evidence,
        )
        .unwrap();

        assert!(facade_hit.score > implementation_hit.score);
        assert!(
            facade_hit
                .why_matched
                .iter()
                .any(|item| item.contains("public-surface query"))
        );
    }
}
