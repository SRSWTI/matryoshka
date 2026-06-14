use anyhow::{Context, Result, anyhow};
use matryoshka_core_ir::SemanticRecord;
use reqwest::blocking::Client;
use reqwest::header::CONNECTION;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::time::Duration;

const MAX_RERANK_CANDIDATES: usize = 20;
const MAX_CONTENT_CHARS: usize = 1_400;

pub trait Reranker {
    fn rerank(&self, query: &str, candidates: &[RerankCandidate]) -> Result<Vec<RerankScore>>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct RerankCandidate {
    pub record_id: String,
    pub title: String,
    pub path: String,
    pub content: String,
    pub current_score: f32,
}

impl RerankCandidate {
    pub fn from_record(record: &SemanticRecord, current_score: f32) -> Self {
        Self {
            record_id: record.record_id.clone(),
            title: record.title.clone(),
            path: record.path.clone(),
            content: truncate_chars(&record.content, MAX_CONTENT_CHARS),
            current_score,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RerankScore {
    pub record_id: String,
    pub score: f32,
    pub reason: Option<String>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoopReranker;

impl Reranker for NoopReranker {
    fn rerank(&self, _query: &str, _candidates: &[RerankCandidate]) -> Result<Vec<RerankScore>> {
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone)]
pub struct EndpointReranker {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl EndpointReranker {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::builder()
                .http1_only()
                .pool_max_idle_per_host(0)
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(180))
                .build()
                .expect("failed to build reranker client"),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }
}

impl Reranker for EndpointReranker {
    fn rerank(&self, query: &str, candidates: &[RerankCandidate]) -> Result<Vec<RerankScore>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: "You rerank code-intelligence search results for a coding agent. Return only JSON. Prefer concrete implementation, exact symbols, and files the agent should read or edit next. Do not invent record IDs.".into(),
                },
                ChatMessage {
                    role: "user",
                    content: serde_json::to_string(&json!({
                        "query": query,
                        "instructions": "Return only the best 10 candidates as {\"scores\":[{\"record_id\":\"...\",\"score\":0.0-1.0,\"reason\":\"short reason\"}]}. Higher score means more useful for the query. Keep reasons under 10 words.",
                        "candidates": candidates
                            .iter()
                            .take(MAX_RERANK_CANDIDATES)
                            .map(|candidate| {
                                json!({
                                    "record_id": candidate.record_id,
                                    "title": candidate.title,
                                    "path": candidate.path,
                                    "current_score": candidate.current_score,
                                    "content": candidate.content,
                                })
                            })
                            .collect::<Vec<_>>()
                    }))?,
                },
            ],
            max_tokens: 900,
            temperature: 0.0,
            chat_template_kwargs: json!({ "enable_thinking": false }),
            response_format: json!({ "type": "json_object" }),
            stream: false,
        };

        let response: ChatResponse = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header(CONNECTION, "close")
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .context("failed to call reranker chat endpoint")?
            .error_for_status()
            .context("reranker chat endpoint returned an error")?
            .json()
            .context("failed to parse reranker chat response")?;

        let content = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.as_deref())
            .ok_or_else(|| anyhow!("reranker response did not contain message content"))?;
        let parsed = match parse_rerank_response(content) {
            Ok(scores) => scores,
            Err(_) => return Ok(Vec::new()),
        };

        let valid_ids = candidates
            .iter()
            .map(|candidate| candidate.record_id.as_str())
            .collect::<BTreeSet<_>>();
        Ok(parsed
            .into_iter()
            .filter(|score| valid_ids.contains(score.record_id.as_str()))
            .map(|score| RerankScore {
                record_id: score.record_id,
                score: score.score.clamp(0.0, 1.0),
                reason: score.reason.filter(|reason| !reason.trim().is_empty()),
            })
            .collect())
    }
}

#[derive(Debug, Clone)]
pub struct OmlxReranker {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
    max_candidates: usize,
}

impl OmlxReranker {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::builder()
                .http1_only()
                .pool_max_idle_per_host(0)
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(180))
                .build()
                .expect("failed to build oMLX reranker client"),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: model.into(),
            max_candidates: MAX_RERANK_CANDIDATES,
        }
    }

    pub fn with_max_candidates(mut self, max_candidates: usize) -> Self {
        self.max_candidates = max_candidates.max(1);
        self
    }
}

impl Reranker for OmlxReranker {
    fn rerank(&self, query: &str, candidates: &[RerankCandidate]) -> Result<Vec<RerankScore>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let candidates = candidates
            .iter()
            .take(self.max_candidates)
            .collect::<Vec<_>>();
        let documents = candidates
            .iter()
            .map(|candidate| rerank_document(candidate))
            .collect::<Vec<_>>();
        let request = OmlxRerankRequest {
            model: self.model.clone(),
            query,
            documents,
            top_n: Some(candidates.len()),
            return_documents: false,
        };

        let response: OmlxRerankResponse = self
            .client
            .post(format!("{}/v1/rerank", self.base_url))
            .header(CONNECTION, "close")
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .context("failed to call oMLX rerank endpoint")?
            .error_for_status()
            .context("oMLX rerank endpoint returned an error")?
            .json()
            .context("failed to parse oMLX rerank response")?;

        Ok(response
            .results
            .into_iter()
            .filter_map(|result| {
                let candidate = candidates.get(result.index)?;
                Some(RerankScore {
                    record_id: candidate.record_id.clone(),
                    score: result.relevance_score.clamp(0.0, 1.0),
                    reason: Some(format!("oMLX rerank score {:.3}", result.relevance_score)),
                })
            })
            .collect())
    }
}

#[derive(Debug, Serialize)]
struct OmlxRerankRequest<'a> {
    model: String,
    query: &'a str,
    documents: Vec<String>,
    top_n: Option<usize>,
    return_documents: bool,
}

#[derive(Debug, Deserialize)]
struct OmlxRerankResponse {
    results: Vec<OmlxRerankResult>,
}

#[derive(Debug, Deserialize)]
struct OmlxRerankResult {
    index: usize,
    relevance_score: f32,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: String,
    messages: Vec<ChatMessage<'a>>,
    max_tokens: u32,
    temperature: f32,
    chat_template_kwargs: Value,
    response_format: Value,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RerankEnvelope {
    scores: Vec<RerankScoreWire>,
}

#[derive(Debug, Deserialize)]
struct RerankScoreWire {
    record_id: String,
    score: f32,
    reason: Option<String>,
}

fn parse_rerank_response(content: &str) -> Result<Vec<RerankScoreWire>> {
    let value = serde_json::from_str::<Value>(content).or_else(|_| {
        let extracted = extract_json_value(content)?;
        serde_json::from_str::<Value>(extracted).map_err(anyhow::Error::from)
    })?;
    parse_rerank_value(value)
}

fn parse_rerank_value(value: Value) -> Result<Vec<RerankScoreWire>> {
    if value.is_array() {
        return serde_json::from_value(value).context("failed to parse reranker score array");
    }

    if let Ok(envelope) = serde_json::from_value::<RerankEnvelope>(value.clone()) {
        return Ok(envelope.scores);
    }

    if let Some(object) = value.as_object() {
        for key in ["scores", "results", "rankings", "reranked", "candidates"] {
            if let Some(items) = object.get(key).filter(|items| items.is_array()) {
                return serde_json::from_value(items.clone())
                    .with_context(|| format!("failed to parse reranker field {key}"));
            }
        }
    }

    Err(anyhow!("reranker JSON did not contain a score list"))
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut out = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

fn rerank_document(candidate: &RerankCandidate) -> String {
    format!(
        "title: {}\npath: {}\nscore: {:.4}\ncontent:\n{}",
        candidate.title, candidate.path, candidate.current_score, candidate.content
    )
}

fn extract_json_value(content: &str) -> Result<&str> {
    let object_start = content.find('{');
    let array_start = content.find('[');
    let start = match (object_start, array_start) {
        (Some(object), Some(array)) => object.min(array),
        (Some(object), None) => object,
        (None, Some(array)) => array,
        (None, None) => {
            return Err(anyhow!(
                "reranker response did not contain a JSON object or array"
            ));
        }
    };
    let object_end = content.rfind('}');
    let array_end = content.rfind(']');
    let end = match (object_end, array_end) {
        (Some(object), Some(array)) => object.max(array),
        (Some(object), None) => object,
        (None, Some(array)) => array,
        (None, None) => {
            return Err(anyhow!(
                "reranker response did not contain a complete JSON object or array"
            ));
        }
    };
    Ok(&content[start..=end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use matryoshka_core_ir::SemanticEntityType;
    use std::collections::BTreeMap;

    #[test]
    fn candidate_truncates_long_record_content() {
        let record = SemanticRecord {
            record_id: "record-a".into(),
            entity_id: "src/lib.rs".into(),
            entity_type: SemanticEntityType::File,
            title: "File src/lib.rs".into(),
            content: "x".repeat(MAX_CONTENT_CHARS + 20),
            path: "src/lib.rs".into(),
            source_hash: "abc".into(),
            embedding: None,
            metadata: BTreeMap::new(),
        };

        let candidate = RerankCandidate::from_record(&record, 0.42);

        assert_eq!(candidate.record_id, "record-a");
        assert!(candidate.content.len() < record.content.len());
        assert!(candidate.content.ends_with("..."));
    }

    #[test]
    fn noop_reranker_returns_no_adjustments() {
        let reranker = NoopReranker;
        assert!(reranker.rerank("query", &[]).unwrap().is_empty());
    }

    #[test]
    fn omlx_rerank_document_includes_path_title_and_content() {
        let candidate = RerankCandidate {
            record_id: "r".into(),
            title: "Symbol search".into(),
            path: "crates/search/src/lib.rs".into(),
            content: "pub struct SearchEngine;".into(),
            current_score: 0.75,
        };

        let document = rerank_document(&candidate);

        assert!(document.contains("Symbol search"));
        assert!(document.contains("crates/search/src/lib.rs"));
        assert!(document.contains("pub struct SearchEngine;"));
    }

    #[test]
    fn parses_bare_array_reranker_response() {
        let scores = parse_rerank_response(
            r#"[{"record_id":"a","score":1.0,"reason":"exact"},{"record_id":"b","score":0.2}]"#,
        )
        .unwrap();

        assert_eq!(scores.len(), 2);
        assert_eq!(scores[0].record_id, "a");
        assert_eq!(scores[0].reason.as_deref(), Some("exact"));
    }

    #[test]
    fn parses_enveloped_reranker_response() {
        let scores = parse_rerank_response(
            r#"{"rankings":[{"record_id":"a","score":0.8,"reason":"nearby"}]}"#,
        )
        .unwrap();

        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].record_id, "a");
    }
}
