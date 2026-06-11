use anyhow::Result;
use matryoshka_core_ir::{FileCard, FolderCard, RepoCard, SearchHit, SemanticRecord};
use matryoshka_embed_client::{Embedder, cosine};
use matryoshka_store_sqlite::MatryoshkaStore;
use serde_json::Value;
use std::collections::BTreeMap;

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
    let owner_query = (edit_like_query || behavior_like_query) && !entrypoint_query;
    let facade = looks_like_facade(record);
    let behavior_owner = looks_like_behavior_owner(record);
    let mut why = Vec::new();
    if semantic_score > 0.2 {
        why.push("semantic behavior match".into());
    }
    let matched_terms = query_tokens
        .iter()
        .filter(|token| haystack.contains(token.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let token_hits = matched_terms.len();
    if token_hits > 0 {
        score += token_hits as f32 * 0.08;
        why.push(format!("{} lexical/path/symbol token matches", token_hits));
    }
    let behavior_hits = metadata_token_hits(record, "behavior_intents", query_tokens);
    if behavior_hits > 0 {
        let weight = if facade && owner_query { 0.03 } else { 0.07 };
        score += behavior_hits as f32 * weight;
        why.push(format!("{behavior_hits} behavior-intent matches"));
    }
    let owns_behavior_hits = metadata_token_hits(record, "owns_behaviors", query_tokens);
    if owns_behavior_hits > 0 {
        score += owns_behavior_hits as f32 * if owner_query { 0.14 } else { 0.08 };
        why.push(format!("{owns_behavior_hits} owned-behavior matches"));
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
        why.push(format!("{edit_hits} edit-intent matches"));
    }
    let tag_hits = metadata_token_hits(record, "retrieval_tags", query_tokens);
    if tag_hits > 0 {
        let weight = if facade && owner_query { 0.025 } else { 0.06 };
        score += tag_hits as f32 * weight;
        why.push(format!("{tag_hits} retrieval-tag matches"));
    }
    match record_kind {
        "file_card" => {
            score += 0.12;
            why.push("rich file-card boost".into());
            if facade && owner_query {
                score -= 0.55;
                why.push("facade penalty for edit/behavior query".into());
            } else if behavior_owner && owner_query {
                score += 0.28;
                why.push("behavior-owner boost".into());
            } else if facade && entrypoint_query {
                score += 0.18;
                why.push("facade surface boost".into());
            }
        }
        "folder_card" => {
            score += 0.08;
            why.push("folder-responsibility boost".into());
            if edit_like_query && !folder_query {
                score -= 0.20;
                why.push("folder penalty for file-level edit query".into());
            }
        }
        "repo_card" => {
            score += 0.16;
            why.push("repo-map boost".into());
            if repo_query {
                score += 0.30;
                why.push("repository architecture boost".into());
            }
        }
        "file_fact" => {
            score -= 0.05;
            why.push("raw-file penalty".into());
        }
        "symbol_fact" => {
            if token_hits > 0 {
                score += 0.12;
                why.push("exact symbol boost".into());
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
        why.push("import-neighborhood boost".into());
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
        why.push("import-resolution phrase boost".into());
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
        why.push("dependency-direction boost".into());
    }
    if haystack.contains("snippet")
        || matches!(
            record.entity_type,
            matryoshka_core_ir::SemanticEntityType::Snippet
        )
    {
        if token_hits > 0 {
            score += 0.05;
            why.push("snippet-level match".into());
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
        key_behaviors: Vec::new(),
        agent_hints: Vec::new(),
        matched_terms,
        score,
        why_matched: why,
    })
}

fn apply_file_card(hit: &mut SearchHit, card: &FileCard, record: Option<&SemanticRecord>) {
    hit.summary = non_empty(card.summary.clone());
    let mut description_parts = Vec::new();
    push_labeled(&mut description_parts, "Role", &card.role);
    if !card.owns_behaviors.is_empty() {
        push_labeled(
            &mut description_parts,
            "Owns",
            &card.owns_behaviors.join("; "),
        );
    }
    if !card.delegates_to.is_empty() {
        push_labeled(
            &mut description_parts,
            "Delegates to",
            &card.delegates_to.join("; "),
        );
    }
    if !card.side_effects.is_empty() {
        push_labeled(
            &mut description_parts,
            "Side effects",
            &card.side_effects.join("; "),
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
    hit.key_behaviors = prefer_non_empty(&[
        &card.owns_behaviors,
        &card.primary_behaviors,
        &card.behavior_intents,
    ]);
    hit.agent_hints = card.agent_read_hints.clone();
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
    hit.key_behaviors = prefer_non_empty(&[&card.common_behaviors, &card.behavior_intents]);
    hit.agent_hints = card.agent_guidance.clone();
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
    hit.key_behaviors = prefer_non_empty(&[&card.behavior_intents, &card.cross_subsystem_flows]);
    hit.agent_hints = card.agent_navigation_hints.clone();
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

fn prefer_non_empty(groups: &[&Vec<String>]) -> Vec<String> {
    groups
        .iter()
        .find(|items| !items.is_empty())
        .map(|items| (*items).clone())
        .unwrap_or_default()
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
                .any(|item| item.contains("behavior-owner boost"))
        );
        assert!(
            facade_hit
                .why_matched
                .iter()
                .any(|item| item.contains("facade penalty"))
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
                .any(|item| item.contains("facade surface boost"))
        );
    }
}
