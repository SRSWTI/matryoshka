use anyhow::Result;
use matryoshka_core_ir::{
    FileCard, FolderCard, RepoCard, SearchHit, SemanticEntityType, SemanticRecord,
};
use matryoshka_embed_client::{Embedder, cosine};
use matryoshka_store_sqlite::MatryoshkaStore;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const MAX_MATCHED_SYMBOLS: usize = 12;

pub struct SearchEngine<M> {
    store: MatryoshkaStore,
    embedder: M,
}

impl<M: Embedder> SearchEngine<M> {
    pub fn new(store: MatryoshkaStore, embedder: M) -> Self {
        Self { store, embedder }
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let query_embedding = self
            .embedder
            .embed(&[query.to_string()])?
            .pop()
            .unwrap_or_default();
        let records = self.store.load_all_semantic_records()?;
        let query_tokens = tokens(query);
        let mut hits = records
            .iter()
            .filter_map(|record| score_record(record, &query_embedding, &query_tokens))
            .collect::<Vec<_>>();
        hits = collapse_file_hits(hits, &records);
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.path.cmp(&right.path))
        });
        hits.truncate(limit);
        self.hydrate_hits(&mut hits, &records)?;
        Ok(hits)
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
        hit.why_matched = self.why.into_iter().collect();
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
) -> Option<SearchHit> {
    let embedding = record.embedding.as_ref()?;
    if embedding.is_empty() || query_embedding.is_empty() {
        return None;
    }
    let semantic_score = cosine(query_embedding, embedding);
    let mut score = semantic_score;
    let haystack = format!("{} {} {}", record.title, record.path, record.content).to_lowercase();
    let record_kind = record_kind(record);
    let edit_like_query = is_edit_like_query(query_tokens);
    let behavior_like_query = is_behavior_like_query(query_tokens);
    let entrypoint_query = is_entrypoint_query(query_tokens);
    let folder_query = is_folder_query(query_tokens);
    let repo_query = is_repo_query(query_tokens);
    let test_query = is_test_query(query_tokens);
    let owner_query = (edit_like_query || behavior_like_query) && !entrypoint_query;
    let facade = looks_like_facade(record);
    let behavior_owner = looks_like_behavior_owner(record);
    let mut why = Vec::new();
    if looks_like_test_path(&record.path) && !test_query {
        score -= 0.28;
        why.push("Test file de-prioritized because the query does not ask for tests".into());
    } else if !looks_like_test_path(&record.path)
        && !test_query
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
            score += 0.12;
            why.push("Matched the enriched file summary".into());
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
            score += 0.08;
            why.push("Matched the enriched folder responsibility".into());
            if edit_like_query && !folder_query {
                score -= 0.20;
                why.push("Folder result was de-prioritized for a file-level edit query".into());
            }
        }
        "repo_card" => {
            score += 0.16;
            why.push("Matched the repository-level map".into());
            if repo_query {
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
                score += 0.12;
                why.push("Matched a concrete symbol in this file".into());
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
                why_matched: vec!["Matched the enriched file summary".into()],
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

        let facade_hit = score_record(&facade, &query_embedding, &query_tokens).unwrap();
        let implementation_hit =
            score_record(&implementation, &query_embedding, &query_tokens).unwrap();

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

        let facade_hit = score_record(&facade, &query_embedding, &query_tokens).unwrap();
        let implementation_hit =
            score_record(&implementation, &query_embedding, &query_tokens).unwrap();

        assert!(facade_hit.score > implementation_hit.score);
        assert!(
            facade_hit
                .why_matched
                .iter()
                .any(|item| item.contains("public-surface query"))
        );
    }
}
