use crate::{
    CodeEnricher, DEFAULT_CHUNK_SUMMARY_MODEL, ENRICHMENT_MODEL, HeuristicEnricher,
    file_summary_enrichment_prompt, folder_summary_enrichment_prompt,
    repo_summary_enrichment_prompt,
};
use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use matryoshka_core_ir::{
    ChunkSummarySource, FileCard, FileEnrichmentContext, FileFact, FolderCard,
    FolderEnrichmentContext, FolderFact, Provenance, RepoCard, SubareaSummary, SymbolFact,
};
use reqwest::blocking::Client;
use reqwest::blocking::Response;
use reqwest::header::CONNECTION;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader};
use std::time::Duration;

const CHAT_COMPLETION_ATTEMPTS: usize = 3;

#[derive(Debug, Clone)]
pub struct MlxChatEnricher {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl MlxChatEnricher {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .http1_only()
                .pool_max_idle_per_host(0)
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(600))
                .build()
                .expect("failed to build MLX chat client"),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: ENRICHMENT_MODEL.into(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    fn complete_json_with_schema(
        &self,
        prompt: Value,
        response_format: Value,
        max_tokens: u32,
    ) -> Result<Value> {
        let prompt_content = serde_json::to_string_pretty(&prompt)?;
        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: "You are a senior engineer. Summarize code artifacts concisely. Be concrete and grounded in the provided code context. Return only valid JSON matching the requested shape.",
                },
                ChatMessage {
                    role: "user",
                    content: &prompt_content,
                },
            ],
            max_tokens,
            temperature: 0.0,
            chat_template_kwargs: json!({ "enable_thinking": false }),
            response_format,
            stream: true,
        };

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header(CONNECTION, "close")
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .context("failed to call chat endpoint")?
            .error_for_status()
            .context("chat endpoint returned an error")?;

        let response = parse_chat_response(response)?;

        let choice = response
            .choices
            .first()
            .ok_or_else(|| anyhow!("chat response did not contain choices"))?;
        let content = choice
            .message
            .content
            .as_deref()
            .ok_or_else(|| anyhow!("chat response did not contain message content"))?;

        serde_json::from_str(content)
            .or_else(|_| {
                extract_json_object(content)
                    .and_then(|json| serde_json::from_str(json).map_err(Into::into))
            })
            .with_context(|| {
                let preview = content.chars().take(1200).collect::<String>();
                format!(
                    "chat response did not contain valid JSON (finish_reason={:?}, preview={preview:?})",
                    choice.finish_reason
                )
            })
    }

    fn complete_typed<T>(
        &self,
        prompt: Value,
        name: &str,
        description: &str,
        max_tokens: u32,
    ) -> Result<T>
    where
        T: JsonSchema + DeserializeOwned,
    {
        let response_format = json_schema_payload::<T>(name, description);
        let mut attempt = 1;
        let mut previous_errors = Vec::new();
        loop {
            let result = self
                .complete_json_with_schema(prompt.clone(), response_format.clone(), max_tokens)
                .and_then(|value| {
                    serde_json::from_value(value)
                        .context("structured response did not match target type")
                });
            match result {
                Ok(value) => return Ok(value),
                Err(error) if attempt < CHAT_COMPLETION_ATTEMPTS => {
                    previous_errors.push(format!("attempt {attempt}: {error:#}"));
                    std::thread::sleep(chat_retry_delay(attempt));
                    attempt += 1;
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        let previous = if previous_errors.is_empty() {
                            String::new()
                        } else {
                            format!("; previous errors: {}", previous_errors.join(" | "))
                        };
                        format!(
                            "MLX structured chat failed after {attempt} attempts for schema {name}{previous}"
                        )
                    });
                }
            }
        }
    }
}

impl CodeEnricher for MlxChatEnricher {
    fn enrich_file(
        &self,
        file: &FileFact,
        symbols: &[SymbolFact],
        context: &FileEnrichmentContext,
    ) -> Result<FileCard> {
        let mut card = HeuristicEnricher.enrich_file(file, symbols, context)?;
        let mut prompt_hashes = Vec::new();

        let prompt = file_summary_enrichment_prompt(file, symbols, context);
        prompt_hashes.push(hash_json(&prompt)?);
        let input_hash = hash_json(&json!(prompt_hashes))?;
        let draft = match self
            .complete_typed::<SummaryDraft>(
                prompt,
                "file_card_summary_draft",
                "Summary-only enrichment for a code-intelligence file card",
                700,
            )
            .with_context(|| format!("MLX file enrichment failed for {}", file.path))
        {
            Ok(draft) => draft,
            Err(error) => {
                return empty_file_card_after_enrichment_failure(
                    card,
                    file,
                    input_hash,
                    &self.model,
                    error,
                );
            }
        };
        card.summary = cleanup_summary(draft.summary);

        card.provenance = Provenance {
            source_hash: file.source_hash.clone(),
            input_hash: Some(input_hash),
            model: Some(self.model.clone()),
            schema_version: matryoshka_core_ir::CARD_SCHEMA_VERSION,
            generated_at: Utc::now(),
        };
        if card.summary.trim().len() < 40 {
            card.risk_notes
                .push("Enrichment quality warning: summary was shorter than expected.".into());
        }
        Ok(card)
    }

    fn enrich_folder(
        &self,
        folder: &FolderFact,
        child_files: &[FileCard],
        child_folders: &[FolderCard],
        context: &FolderEnrichmentContext,
    ) -> Result<FolderCard> {
        if child_files.is_empty() && child_folders.is_empty() {
            return empty_folder_card_after_enrichment_failure(
                folder,
                child_files,
                child_folders,
                context,
                None,
                &self.model,
                format!(
                    "folder {} has no child file cards or child folder cards to ground MLX enrichment",
                    folder.folder_id
                ),
            );
        }

        if child_files.is_empty() {
            return roll_up_folder_card_from_child_folders(
                folder,
                child_folders,
                context,
                &self.model,
            );
        }

        let child_values = child_files
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()?;
        let child_folder_values = child_folders
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()?;
        let mut prompt_hashes = Vec::new();

        let prompt =
            folder_summary_enrichment_prompt(folder, &child_values, &child_folder_values, context);
        prompt_hashes.push(hash_json(&prompt)?);
        let input_hash = hash_json(&json!(prompt_hashes))?;
        let draft = match self
            .complete_typed::<SummaryDraft>(
                prompt,
                "folder_card_summary_draft",
                "Summary-only enrichment for a code-intelligence folder card",
                700,
            )
            .with_context(|| format!("MLX folder enrichment failed for {}", folder.folder_id))
        {
            Ok(draft) => draft,
            Err(error) => {
                return empty_folder_card_after_enrichment_failure(
                    folder,
                    child_files,
                    child_folders,
                    context,
                    Some(input_hash),
                    &self.model,
                    format!("{error:#}"),
                );
            }
        };

        let mut card =
            HeuristicEnricher.enrich_folder(folder, child_files, child_folders, context)?;
        card.summary = cleanup_summary(draft.summary);

        card.provenance = Provenance {
            source_hash: child_files
                .iter()
                .map(|card| card.provenance.source_hash.as_str())
                .chain(
                    child_folders
                        .iter()
                        .map(|card| card.provenance.source_hash.as_str()),
                )
                .collect::<Vec<_>>()
                .join(":"),
            input_hash: Some(input_hash),
            model: Some(self.model.clone()),
            schema_version: matryoshka_core_ir::CARD_SCHEMA_VERSION,
            generated_at: Utc::now(),
        };
        Ok(card)
    }

    fn enrich_repo(&self, repo_root: &str, folders: &[FolderCard]) -> Result<RepoCard> {
        let mut card = HeuristicEnricher.enrich_repo(repo_root, folders)?;
        let prompt = repo_summary_enrichment_prompt(repo_root, folders);
        let input_hash = hash_json(&prompt)?;
        match self
            .complete_typed::<SummaryDraft>(
                prompt,
                "repo_card_summary_draft",
                "Summary-only enrichment for a code-intelligence repo card",
                700,
            )
            .with_context(|| format!("MLX repo enrichment failed for {repo_root}"))
        {
            Ok(draft) => {
                card.summary = cleanup_summary(draft.summary);
                card.provenance.input_hash = Some(input_hash);
            }
            Err(error) => {
                card.high_risk_areas.push(format!(
                    "MLX repo summary failed after retries; heuristic summary retained: {error:#}"
                ));
            }
        }
        card.provenance.model = Some(self.model.clone());
        Ok(card)
    }
}

fn chat_retry_delay(attempt: usize) -> Duration {
    Duration::from_millis(500 * attempt as u64)
}

fn empty_file_card_after_enrichment_failure(
    mut card: FileCard,
    file: &FileFact,
    input_hash: String,
    model: &str,
    error: anyhow::Error,
) -> Result<FileCard> {
    card.summary.clear();
    card.role.clear();
    card.primary_behaviors.clear();
    card.behavior_intents.clear();
    card.edit_intents.clear();
    card.owns_behaviors.clear();
    card.side_effects.clear();
    card.agent_read_hints.clear();
    card.search_phrases.clear();
    card.risk_notes = vec![format!(
        "MLX file enrichment failed after retries; summary intentionally left empty: {error:#}"
    )];
    card.provenance = Provenance {
        source_hash: file.source_hash.clone(),
        input_hash: Some(input_hash),
        model: Some(model.to_string()),
        schema_version: matryoshka_core_ir::CARD_SCHEMA_VERSION,
        generated_at: Utc::now(),
    };
    Ok(card)
}

fn empty_folder_card_after_enrichment_failure(
    folder: &FolderFact,
    child_files: &[FileCard],
    child_folders: &[FolderCard],
    context: &FolderEnrichmentContext,
    input_hash: Option<String>,
    model: &str,
    message: impl Into<String>,
) -> Result<FolderCard> {
    Ok(FolderCard {
        folder_id: folder.folder_id.clone(),
        summary: String::new(),
        responsibility: String::new(),
        behavior_intents: Vec::new(),
        edit_intents: Vec::new(),
        retrieval_tags: structural_folder_tags(folder, child_files, child_folders),
        contains_kinds_of_files: child_files
            .iter()
            .map(|card| card.file_id.clone())
            .take(12)
            .collect(),
        incoming_dependencies_meaning: context
            .incoming_dependencies
            .iter()
            .take(12)
            .map(|item| item.detail.clone())
            .collect(),
        outgoing_dependencies_meaning: context
            .outgoing_dependencies
            .iter()
            .take(12)
            .map(|item| item.detail.clone())
            .collect(),
        key_entrypoints: context
            .representative_child_files
            .iter()
            .map(|item| item.path.clone())
            .chain(child_folders.iter().map(|card| card.folder_id.clone()))
            .take(8)
            .collect(),
        common_behaviors: Vec::new(),
        subareas: subareas_from_child_folders(folder, child_folders),
        agent_guidance: vec![message.into()],
        search_phrases: Vec::new(),
        provenance: Provenance {
            source_hash: folder_source_hash(child_files, child_folders),
            input_hash,
            model: Some(model.to_string()),
            schema_version: matryoshka_core_ir::CARD_SCHEMA_VERSION,
            generated_at: Utc::now(),
        },
    })
}

fn roll_up_folder_card_from_child_folders(
    folder: &FolderFact,
    child_folders: &[FolderCard],
    context: &FolderEnrichmentContext,
    model: &str,
) -> Result<FolderCard> {
    let child_summaries = child_folders
        .iter()
        .filter_map(|card| child_folder_rollup_line(card))
        .take(12)
        .collect::<Vec<_>>();
    let child_responsibilities = child_folders
        .iter()
        .filter_map(|card| {
            first_non_empty([card.responsibility.as_str(), card.summary.as_str()])
                .map(|text| format!("{}: {text}", card.folder_id))
        })
        .take(12)
        .collect::<Vec<_>>();
    let input_hash = hash_json(&json!({
        "folder_id": folder.folder_id,
        "child_folders": child_folders
            .iter()
            .map(|card| json!({
                "folder_id": &card.folder_id,
                "summary": &card.summary,
                "responsibility": &card.responsibility,
                "common_behaviors": &card.common_behaviors,
                "provenance": &card.provenance,
            }))
            .collect::<Vec<_>>(),
    }))?;

    Ok(FolderCard {
        folder_id: folder.folder_id.clone(),
        summary: child_summaries.join("\n"),
        responsibility: child_responsibilities.join("\n"),
        behavior_intents: sanitize_string_items(
            child_folders
                .iter()
                .flat_map(|card| card.behavior_intents.clone())
                .collect(),
            16,
        ),
        edit_intents: sanitize_string_items(
            child_folders
                .iter()
                .flat_map(|card| card.edit_intents.clone())
                .collect(),
            16,
        ),
        retrieval_tags: structural_folder_tags(folder, &[], child_folders),
        contains_kinds_of_files: sanitize_string_items(
            child_folders
                .iter()
                .flat_map(|card| card.contains_kinds_of_files.clone())
                .collect(),
            16,
        ),
        incoming_dependencies_meaning: context
            .incoming_dependencies
            .iter()
            .take(12)
            .map(|item| item.detail.clone())
            .collect(),
        outgoing_dependencies_meaning: context
            .outgoing_dependencies
            .iter()
            .take(12)
            .map(|item| item.detail.clone())
            .collect(),
        key_entrypoints: sanitize_string_items(
            child_folders
                .iter()
                .flat_map(|card| card.key_entrypoints.clone())
                .chain(child_folders.iter().map(|card| card.folder_id.clone()))
                .collect(),
            12,
        ),
        common_behaviors: sanitize_string_items(
            child_folders
                .iter()
                .flat_map(|card| card.common_behaviors.clone())
                .collect(),
            16,
        ),
        subareas: subareas_from_child_folders(folder, child_folders),
        agent_guidance: sanitize_string_items(
            child_folders
                .iter()
                .flat_map(|card| card.agent_guidance.clone())
                .collect(),
            8,
        ),
        search_phrases: sanitize_string_items(
            child_folders
                .iter()
                .flat_map(|card| card.search_phrases.clone())
                .collect(),
            16,
        ),
        provenance: Provenance {
            source_hash: folder_source_hash(&[], child_folders),
            input_hash: Some(input_hash),
            model: Some(model.to_string()),
            schema_version: matryoshka_core_ir::CARD_SCHEMA_VERSION,
            generated_at: Utc::now(),
        },
    })
}

fn child_folder_rollup_line(card: &FolderCard) -> Option<String> {
    first_non_empty([card.summary.as_str(), card.responsibility.as_str()])
        .map(|text| format!("{}: {text}", card.folder_id))
}

fn first_non_empty<'a>(items: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    items
        .into_iter()
        .map(str::trim)
        .find(|item| !item.is_empty())
}

fn folder_source_hash(child_files: &[FileCard], child_folders: &[FolderCard]) -> String {
    child_files
        .iter()
        .map(|card| card.provenance.source_hash.as_str())
        .chain(
            child_folders
                .iter()
                .map(|card| card.provenance.source_hash.as_str()),
        )
        .collect::<Vec<_>>()
        .join(":")
}

fn structural_folder_tags(
    folder: &FolderFact,
    child_files: &[FileCard],
    child_folders: &[FolderCard],
) -> Vec<String> {
    sanitize_retrieval_tags(
        std::iter::once("entity:folder".to_string())
            .chain(std::iter::once(format!(
                "path:{}",
                folder.path.replace('/', "-")
            )))
            .chain(std::iter::once(format!(
                "folder:{}",
                folder.folder_id.replace('/', "-")
            )))
            .chain(
                child_files
                    .iter()
                    .flat_map(|card| card.retrieval_tags.clone()),
            )
            .chain(
                child_folders
                    .iter()
                    .flat_map(|card| card.retrieval_tags.clone()),
            )
            .collect(),
        24,
    )
}

fn subareas_from_child_folders(
    folder: &FolderFact,
    child_folders: &[FolderCard],
) -> Vec<SubareaSummary> {
    folder
        .child_folder_ids
        .iter()
        .map(|id| {
            let responsibility = child_folders
                .iter()
                .find(|card| card.folder_id == *id)
                .and_then(|card| {
                    first_non_empty([card.responsibility.as_str(), card.summary.as_str()])
                        .map(str::to_string)
                })
                .unwrap_or_default();
            SubareaSummary {
                id: id.clone(),
                name: id.clone(),
                responsibility,
            }
        })
        .collect()
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
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatStreamChunk {
    choices: Vec<ChatStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatStreamChoice {
    delta: ChatStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatStreamDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatErrorEnvelope {
    error: ChatErrorBody,
}

#[derive(Debug, Deserialize)]
struct ChatErrorBody {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
struct SummaryDraft {
    summary: String,
}

fn cleanup_summary(summary: String) -> String {
    summary
        .lines()
        .map(|line| {
            line.trim()
                .trim_start_matches("- ")
                .trim_start_matches("* ")
                .trim()
        })
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn sanitize_string_items(items: Vec<String>, limit: usize) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    items
        .into_iter()
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .filter(|item| seen.insert(item.to_lowercase()))
        .take(limit)
        .collect()
}

fn sanitize_retrieval_tags(items: Vec<String>, limit: usize) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    items
        .into_iter()
        .map(|item| {
            item.trim()
                .to_ascii_lowercase()
                .chars()
                .map(|ch| {
                    if ch.is_ascii_alphanumeric() || ch == ':' || ch == '-' || ch == '_' {
                        ch
                    } else {
                        '-'
                    }
                })
                .collect::<String>()
        })
        .map(|item| collapse_dashes(&item))
        .filter(|item| !item.is_empty())
        .filter(|item| seen.insert(item.clone()))
        .take(limit)
        .collect()
}

fn collapse_dashes(value: &str) -> String {
    let mut collapsed = String::new();
    let mut previous_dash = false;
    for ch in value.chars() {
        if ch == '-' {
            if !previous_dash {
                collapsed.push(ch);
            }
            previous_dash = true;
        } else {
            collapsed.push(ch);
            previous_dash = false;
        }
    }
    collapsed.trim_matches('-').to_string()
}

fn json_schema_payload<T: JsonSchema>(name: &str, description: &str) -> Value {
    let schema = schemars::schema_for!(T);
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "description": description,
            "strict": true,
            "schema": serde_json::to_value(&schema).expect("schema should serialize"),
        }
    })
}

fn parse_chat_response(response: Response) -> Result<ChatResponse> {
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned();

    if content_type.contains("text/event-stream") {
        return parse_streaming_chat_response(response);
    }

    let body = response
        .text()
        .context("failed to read chat response body")?;
    serde_json::from_str::<ChatResponse>(body.trim_start()).context("failed to parse chat response")
}

fn parse_streaming_chat_response(response: Response) -> Result<ChatResponse> {
    let reader = BufReader::new(response);
    parse_streaming_chat_reader(reader)
}

fn parse_streaming_chat_reader<R: BufRead>(reader: R) -> Result<ChatResponse> {
    let mut content = String::new();
    let mut finish_reason = None;
    for line in reader.lines() {
        let line = line.context("failed to read streaming chat response line")?;
        let trimmed = line.trim();
        if trimmed.is_empty() || !trimmed.starts_with("data:") {
            continue;
        }

        let payload = trimmed.trim_start_matches("data:").trim();
        if payload == "[DONE]" {
            break;
        }

        if let Ok(error) = serde_json::from_str::<ChatErrorEnvelope>(payload) {
            return Err(anyhow!(
                "chat stream error from server: {} ({})",
                error.error.message,
                error
                    .error
                    .error_type
                    .unwrap_or_else(|| "unknown_error".into())
            ));
        }

        let chunk: ChatStreamChunk = serde_json::from_str(payload)
            .with_context(|| format!("failed to parse streaming chat chunk: {payload}"))?;
        for choice in chunk.choices {
            if let Some(delta) = choice.delta.content {
                content.push_str(&delta);
            }
            if choice.finish_reason.is_some() {
                finish_reason = choice.finish_reason;
            }
        }
    }

    Ok(ChatResponse {
        choices: vec![ChatChoice {
            message: ChatMessageResponse {
                content: Some(content),
            },
            finish_reason,
        }],
    })
}

fn extract_json_object(content: &str) -> Result<&str> {
    let start = content
        .find('{')
        .ok_or_else(|| anyhow!("no JSON object start found"))?;
    let end = content
        .rfind('}')
        .ok_or_else(|| anyhow!("no JSON object end found"))?;
    Ok(&content[start..=end])
}

fn hash_json(value: &Value) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(serde_json::to_vec(value)?);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn summary_schema_payload_contains_summary_field() {
        let payload = json_schema_payload::<SummaryDraft>(
            "summary_draft",
            "Concise grounded summary for a code-intelligence card",
        );
        let schema = payload
            .get("json_schema")
            .and_then(|value| value.get("schema"))
            .cloned()
            .expect("schema payload should contain schema");

        assert_eq!(schema.get("type").and_then(Value::as_str), Some("object"));
        let properties = schema
            .get("properties")
            .and_then(Value::as_object)
            .expect("schema should expose properties");
        assert!(properties.contains_key("summary"));
    }

    #[test]
    fn streaming_parser_reassembles_content_chunks() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"summary\\\":\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" \\\"ok\\\"}\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let response = parse_streaming_chat_reader(Cursor::new(sse)).unwrap();
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("{\"summary\": \"ok\"}")
        );
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn streaming_parser_surfaces_server_error_envelopes() {
        let sse = concat!(
            "data: {\"error\":{\"message\":\"cache exploded\",\"type\":\"server_error\"}}\n\n",
            "data: [DONE]\n\n"
        );
        let error = parse_streaming_chat_reader(Cursor::new(sse)).unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("cache exploded"));
        assert!(message.contains("server_error"));
    }

    #[test]
    fn parent_folder_rollup_uses_child_folder_cards_without_placeholder_prose() {
        let folder = FolderFact {
            folder_id: "gateway".into(),
            path: "gateway".into(),
            name: "gateway".into(),
            parent_folder_id: None,
            child_file_ids: Vec::new(),
            child_folder_ids: vec!["gateway/apps".into(), "gateway/crates".into()],
        };
        let context = FolderEnrichmentContext {
            parent_folder_id: None,
            incoming_dependencies: Vec::new(),
            outgoing_dependencies: Vec::new(),
            representative_child_files: Vec::new(),
        };
        let child_folders = vec![
            test_folder_card(
                "gateway/apps",
                "Runs the gateway daemon entrypoint and process wiring.",
                "Owns daemon startup.",
            ),
            test_folder_card(
                "gateway/crates",
                "Contains reusable gateway provider crates.",
                "Owns reusable provider code.",
            ),
        ];

        let card =
            roll_up_folder_card_from_child_folders(&folder, &child_folders, &context, "mlx-test")
                .unwrap();

        assert!(card.summary.contains("gateway/apps"));
        assert!(card.summary.contains("gateway/crates"));
        assert!(card.summary.contains("Runs the gateway daemon"));
        assert!(
            card.summary
                .contains("Contains reusable gateway provider crates")
        );
        assert!(!card.summary.contains("central hub"));
        assert!(
            card.subareas
                .iter()
                .all(|subarea| !subarea.responsibility.contains("awaiting"))
        );
    }

    fn test_folder_card(id: &str, summary: &str, responsibility: &str) -> FolderCard {
        FolderCard {
            folder_id: id.into(),
            summary: summary.into(),
            responsibility: responsibility.into(),
            behavior_intents: Vec::new(),
            edit_intents: Vec::new(),
            retrieval_tags: vec![format!("folder:{}", id.replace('/', "-"))],
            contains_kinds_of_files: Vec::new(),
            incoming_dependencies_meaning: Vec::new(),
            outgoing_dependencies_meaning: Vec::new(),
            key_entrypoints: Vec::new(),
            common_behaviors: Vec::new(),
            subareas: Vec::new(),
            agent_guidance: Vec::new(),
            search_phrases: Vec::new(),
            provenance: Provenance::source_only(id),
        }
    }
}

// ============================================================================
// Chunk summarizer (Milestone 2)
// ============================================================================

use crate::{
    ChunkSummarizer, ChunkSummaryDraft, chunk_summary_prompt, chunk_summary_system_prompt,
};
use matryoshka_core_ir::CodeChunkFact;
use rayon::prelude::*;

/// Concurrent chunk summarizer backed by an OpenAI-compatible chat endpoint
/// (omlx). Sends one request per chunk in parallel using a rayon thread pool.
///
/// The omlx server already does continuous batching internally, so we just fire
/// many blocking requests concurrently up to `concurrency`.
#[derive(Debug, Clone)]
pub struct MlxChunkSummarizer {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
    concurrency: usize,
    max_tokens: u32,
}

impl MlxChunkSummarizer {
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            client: Client::builder()
                .http1_only()
                .pool_max_idle_per_host(0)
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(120))
                .build()
                .expect("failed to build MLX chunk summarizer client"),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: DEFAULT_CHUNK_SUMMARY_MODEL.into(),
            concurrency: 6,
            max_tokens: 160,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Summarize a single chunk synchronously. Returns the cleaned summary text.
    fn summarize_one(&self, chunk: &CodeChunkFact) -> Result<String> {
        let prompt = chunk_summary_prompt(chunk);
        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: chunk_summary_system_prompt(),
                },
                ChatMessage {
                    role: "user",
                    content: &prompt,
                },
            ],
            max_tokens: self.max_tokens,
            temperature: 0.0,
            chat_template_kwargs: json!({ "enable_thinking": false }),
            response_format: json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "chunk_summary",
                    "description": "A 2-3 line summary of a code chunk",
                    "strict": true,
                    "schema": serde_json::to_value(&schemars::schema_for!(SummaryDraft))
                        .expect("schema should serialize"),
                }
            }),
            stream: false,
        };

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header(CONNECTION, "close")
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .context("failed to call chunk summary endpoint")?
            .error_for_status()
            .context("chunk summary endpoint returned an error")?;

        let chat_response = parse_chat_response(response)?;
        let content = chat_response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .ok_or_else(|| anyhow!("chunk summary response had no content"))?;

        let summary = serde_json::from_str::<SummaryDraft>(&content)
            .or_else(|_| {
                extract_json_object(&content)
                    .and_then(|json| serde_json::from_str(json).map_err(Into::into))
            })
            .with_context(|| {
                let preview = content.chars().take(500).collect::<String>();
                format!("chunk summary response was not valid JSON: {preview:?}")
            })?;
        Ok(cleanup_summary(summary.summary))
    }
}

impl ChunkSummarizer for MlxChunkSummarizer {
    fn summarize_chunks(&self, chunks: &[CodeChunkFact]) -> Result<Vec<ChunkSummaryDraft>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.concurrency)
            .build()
            .map_err(anyhow::Error::from)?;

        let drafts: Vec<Result<ChunkSummaryDraft>> = pool.install(|| {
            chunks
                .par_iter()
                .map(|chunk| {
                    let summary = self.summarize_one(chunk)?;
                    Ok(ChunkSummaryDraft {
                        chunk_id: chunk.chunk_id.clone(),
                        summary,
                        source: ChunkSummarySource::Llm,
                    })
                })
                .collect()
        });

        // Collect successes; collect failures into a single error if everything failed.
        let mut ok = Vec::with_capacity(chunks.len());
        let mut errors = Vec::new();
        for result in drafts {
            match result {
                Ok(draft) => ok.push(draft),
                Err(err) => errors.push(format!("{err:#}")),
            }
        }
        if ok.is_empty() && !errors.is_empty() {
            return Err(anyhow!(
                "all chunk summary requests failed; first error: {}",
                errors.into_iter().next().unwrap_or_default()
            ));
        }
        Ok(ok)
    }

    fn summarize_chunks_with_progress(
        &self,
        chunks: &[CodeChunkFact],
        progress: &mut dyn FnMut(usize, usize, usize),
    ) -> Result<Vec<ChunkSummaryDraft>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        const BATCH_SIZE: usize = 32;
        let total_batches = chunks.len().div_ceil(BATCH_SIZE);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.concurrency)
            .build()
            .map_err(anyhow::Error::from)?;

        let mut all_drafts = Vec::with_capacity(chunks.len());
        for (batch_index, batch) in chunks.chunks(BATCH_SIZE).enumerate() {
            progress(batch_index + 1, total_batches, batch.len());

            let drafts: Vec<Result<ChunkSummaryDraft>> = pool.install(|| {
                batch
                    .par_iter()
                    .map(|chunk| {
                        let summary = self.summarize_one(chunk)?;
                        Ok(ChunkSummaryDraft {
                            chunk_id: chunk.chunk_id.clone(),
                            summary,
                            source: ChunkSummarySource::Llm,
                        })
                    })
                    .collect()
            });

            for result in drafts {
                match result {
                    Ok(draft) => all_drafts.push(draft),
                    Err(_err) => {
                        // Individual chunk failures are tolerated; only fail if
                        // the entire batch failed.
                    }
                }
            }
        }

        if all_drafts.is_empty() {
            return Err(anyhow!(
                "all chunk summary requests failed across {} batches",
                total_batches
            ));
        }
        Ok(all_drafts)
    }
}

#[cfg(test)]
mod chunk_summarizer_tests {
    use super::*;
    use crate::HeuristicChunkSummarizer;
    use matryoshka_core_ir::{ChunkSummarySource, CodeChunkFact, CodeChunkKind};

    fn make_chunk(symbol: &str, code: &str) -> CodeChunkFact {
        CodeChunkFact {
            chunk_id: format!("test::{}:1", symbol),
            file_id: "src/lib.rs".into(),
            symbol_id: Some(format!("src/lib.rs::{}:1", symbol)),
            path: "src/lib.rs".into(),
            symbol: Some(symbol.into()),
            qualified_name: Some(symbol.into()),
            kind: CodeChunkKind::Function,
            signature: format!("fn {}()", symbol),
            start_line: 1,
            end_line: code.lines().count(),
            doc_summary: None,
            generated_summary: None,
            summary: String::new(),
            summary_source: ChunkSummarySource::Empty,
            code: code.into(),
            source_hash: "test".into(),
        }
    }

    #[test]
    fn heuristic_chunk_summarizer_produces_grounded_summary() {
        let chunk = make_chunk(
            "handle_resume_countdown",
            "fn handle_resume_countdown(state: &mut State) -> bool {\n    state.countdown.cancel();\n    true\n}",
        );
        let summarizer = HeuristicChunkSummarizer;
        let drafts = summarizer.summarize_chunks(&[chunk.clone()]).unwrap();
        assert_eq!(drafts.len(), 1);
        assert_eq!(drafts[0].chunk_id, chunk.chunk_id);
        assert!(drafts[0].summary.contains("handle_resume_countdown"));
        assert!(drafts[0].summary.contains("src/lib.rs"));
    }

    #[test]
    #[ignore = "requires a live omlx server at 127.0.0.1:44449"]
    fn mlx_chunk_summarizer_live() {
        let chunk = make_chunk(
            "handle_resume_countdown",
            "fn handle_resume_countdown(state: &mut State) -> bool {\n    state.countdown.cancel();\n    state.mode = Mode::Attack;\n    true\n}",
        );
        let summarizer = MlxChunkSummarizer::new("http://127.0.0.1:44449", "2508")
            .with_model("srswti--bodega-raptor-90m")
            .with_concurrency(2);
        let drafts = summarizer.summarize_chunks(&[chunk]).unwrap();
        assert_eq!(drafts.len(), 1);
        let summary = &drafts[0].summary;
        println!("generated summary: {summary}");
        assert!(!summary.is_empty());
        // The summary should mention at least one of the key behaviors.
        let lower = summary.to_ascii_lowercase();
        assert!(
            lower.contains("countdown")
                || lower.contains("attack")
                || lower.contains("cancel")
                || lower.contains("mode"),
            "summary should be grounded in the code: {summary}"
        );
    }

    #[test]
    #[ignore = "requires a live omlx server at 127.0.0.1:44449"]
    fn mlx_chunk_summarizer_concurrent_live() {
        // Fire several chunks concurrently to verify the thread pool + omlx
        // continuous batching path works.
        let chunks = vec![
            make_chunk(
                "cancel_countdown",
                "fn cancel_countdown(&mut self) {\n    self.timer.cancel();\n}",
            ),
            make_chunk(
                "enter_attack_mode",
                "fn enter_attack_mode(&mut self) {\n    self.mode = Mode::Attack;\n}",
            ),
            make_chunk(
                "reset_state",
                "fn reset_state(&mut self) {\n    self.mode = Mode::Idle;\n    self.target = None;\n}",
            ),
        ];
        let summarizer = MlxChunkSummarizer::new("http://127.0.0.1:44449", "2508")
            .with_model("srswti--bodega-raptor-90m")
            .with_concurrency(3);
        let drafts = summarizer.summarize_chunks(&chunks).unwrap();
        assert_eq!(drafts.len(), 3, "all three chunks should get summaries");
        for draft in &drafts {
            assert!(!draft.summary.is_empty(), "summary should not be empty");
            println!("  {}: {}", draft.chunk_id, draft.summary);
        }
    }
}
