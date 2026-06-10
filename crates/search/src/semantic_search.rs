use anyhow::Result;
use matryoshka_core_ir::{SearchHit, SemanticRecord};
use matryoshka_embed_client::{Embedder, cosine};
use matryoshka_store_sqlite::MatryoshkaStore;
use serde_json::Value;

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
        Ok(hits)
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
    let mut why = Vec::new();
    if semantic_score > 0.2 {
        why.push("semantic behavior match".into());
    }
    let token_hits = query_tokens
        .iter()
        .filter(|token| haystack.contains(token.as_str()))
        .count();
    if token_hits > 0 {
        score += token_hits as f32 * 0.08;
        why.push(format!("{} lexical/path/symbol token matches", token_hits));
    }
    let behavior_hits = metadata_token_hits(record, "behavior_intents", query_tokens);
    if behavior_hits > 0 {
        score += behavior_hits as f32 * 0.07;
        why.push(format!("{behavior_hits} behavior-intent matches"));
    }
    let edit_hits = metadata_token_hits(record, "edit_intents", query_tokens);
    if edit_hits > 0 {
        let edit_query = query_tokens.iter().any(|token| {
            matches!(
                token.as_str(),
                "edit" | "change" | "debug" | "fix" | "refactor" | "modify" | "add"
            )
        });
        score += edit_hits as f32 * if edit_query { 0.10 } else { 0.05 };
        why.push(format!("{edit_hits} edit-intent matches"));
    }
    let tag_hits = metadata_token_hits(record, "retrieval_tags", query_tokens);
    if tag_hits > 0 {
        score += tag_hits as f32 * 0.06;
        why.push(format!("{tag_hits} retrieval-tag matches"));
    }
    match record_kind {
        "file_card" => {
            score += 0.12;
            why.push("rich file-card boost".into());
            if looks_like_facade(record) && (edit_like_query || behavior_like_query) && !entrypoint_query
            {
                score -= 0.18;
                why.push("facade penalty for edit/behavior query".into());
            } else if looks_like_implementation(record)
                && (edit_like_query || behavior_like_query)
                && !entrypoint_query
            {
                score += 0.08;
                why.push("implementation-file boost".into());
            }
        }
        "folder_card" => {
            score += 0.08;
            why.push("folder-responsibility boost".into());
            if edit_like_query && !folder_query {
                score -= 0.07;
                why.push("folder penalty for file-level edit query".into());
            }
        }
        "file_fact" => {
            score -= 0.05;
            why.push("raw-file penalty".into());
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
        && query_tokens.iter().any(|token| token == "resolution" || token == "resolve")
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
        score,
        why_matched: why,
    })
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
    record.path.ends_with("/lib.rs")
        || record.path.ends_with("/mod.rs")
        || has_metadata_tag(record, "retrieval_tags", "artifact:facade")
}

fn looks_like_implementation(record: &SemanticRecord) -> bool {
    !looks_like_facade(record)
        && (has_metadata_tag(record, "retrieval_tags", "artifact:implementation")
            || record.path.ends_with(".rs")
            || record.path.ends_with(".py")
            || record.path.ends_with(".ts")
            || record.path.ends_with(".tsx"))
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
                ("retrieval_tags".into(), json!(["artifact:facade", "behavior:import-resolution"])),
                ("edit_intents".into(), json!(["debug import resolution behavior"])),
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
                ("edit_intents".into(), json!(["debug import resolution behavior"])),
            ]),
        };

        let facade_hit = score_record(&facade, &query_embedding, &query_tokens).unwrap();
        let implementation_hit =
            score_record(&implementation, &query_embedding, &query_tokens).unwrap();

        assert!(implementation_hit.score > facade_hit.score);
        assert!(implementation_hit
            .why_matched
            .iter()
            .any(|item| item.contains("implementation-file boost")));
        assert!(facade_hit
            .why_matched
            .iter()
            .any(|item| item.contains("facade penalty")));
    }
}
