use crate::{
    CodeEnricher, DEFAULT_CHUNK_SUMMARY_MODEL, ENRICHMENT_MODEL, file_summary_enrichment_prompt,
    folder_summary_enrichment_prompt, repo_summary_enrichment_prompt,
};
use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use matryoshka_core_ir::{
    ChunkSummarySource, DependencyInterpretation, FileCard, FileEnrichmentContext, FileFact,
    FileOwnershipKind, FolderCard, FolderEnrichmentContext, FolderFact, Provenance, RepoCard,
    SubareaSummary, SymbolBehavior, SymbolFact,
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
            .context("failed to call chat endpoint")
            .and_then(|response| response_with_body_on_error(response, "chat endpoint"))?;

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
        let mut prompt_hashes = Vec::new();

        let prompt = file_summary_enrichment_prompt(file, symbols, context);
        prompt_hashes.push(hash_json(&prompt)?);
        let input_hash = hash_json(&json!(prompt_hashes))?;
        let draft = self
            .complete_typed::<SummaryDraft>(
                prompt,
                "file_card_summary_draft",
                "Summary-only enrichment for a code-intelligence file card",
                700,
            )
            .with_context(|| format!("MLX file enrichment failed for {}", file.path))?;
        let mut card = strict_file_card_from_summary(
            file,
            symbols,
            context,
            cleanup_summary(draft.summary),
            input_hash,
            &self.model,
        );
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
            return Err(anyhow!(
                "MLX folder enrichment failed for {}: folder has no child file cards or child folder cards to ground enrichment",
                folder.folder_id
            ));
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
        let draft = self
            .complete_typed::<SummaryDraft>(
                prompt,
                "folder_card_summary_draft",
                "Summary-only enrichment for a code-intelligence folder card",
                700,
            )
            .with_context(|| format!("MLX folder enrichment failed for {}", folder.folder_id))?;

        Ok(strict_folder_card_from_summary(
            folder,
            child_files,
            child_folders,
            context,
            cleanup_summary(draft.summary),
            input_hash,
            &self.model,
        ))
    }

    fn enrich_repo(&self, repo_root: &str, folders: &[FolderCard]) -> Result<RepoCard> {
        let prompt = repo_summary_enrichment_prompt(repo_root, folders);
        let input_hash = hash_json(&prompt)?;
        let draft = self
            .complete_typed::<SummaryDraft>(
                prompt,
                "repo_card_summary_draft",
                "Summary-only enrichment for a code-intelligence repo card",
                700,
            )
            .with_context(|| format!("MLX repo enrichment failed for {repo_root}"))?;

        Ok(strict_repo_card_from_summary(
            repo_root,
            folders,
            cleanup_summary(draft.summary),
            input_hash,
            &self.model,
        ))
    }
}

fn chat_retry_delay(attempt: usize) -> Duration {
    Duration::from_millis(500 * attempt as u64)
}

fn strict_file_card_from_summary(
    file: &FileFact,
    symbols: &[SymbolFact],
    context: &FileEnrichmentContext,
    summary: String,
    input_hash: String,
    model: &str,
) -> FileCard {
    let important_symbols = symbols
        .iter()
        .take(24)
        .map(|symbol| SymbolBehavior {
            symbol_id: symbol.symbol_id.clone(),
            name: symbol.name.clone(),
            role: format!("{:?}", symbol.kind),
            behavior: symbol.signature.clone(),
        })
        .collect::<Vec<_>>();
    let key_entities = sanitize_string_items(
        symbols
            .iter()
            .flat_map(|symbol| [symbol.name.clone(), symbol.qualified_name.clone()])
            .collect(),
        32,
    );
    let imports_interpreted = context
        .internal_imports
        .iter()
        .filter_map(|import| {
            let target_id = import.resolved_file_id.clone()?;
            Some(DependencyInterpretation {
                target_id,
                target_path: import
                    .resolved_path
                    .clone()
                    .unwrap_or_else(|| import.module.clone()),
                why: format!("resolved internal import {}", import.module),
                dependency_kind: import.dependency_kind.clone(),
            })
        })
        .collect::<Vec<_>>();
    let used_by_interpreted = context
        .imported_by_files
        .iter()
        .map(|file| DependencyInterpretation {
            target_id: file.file_id.clone(),
            target_path: file.path.clone(),
            why: file.detail.clone(),
            dependency_kind: file.relationship.clone(),
        })
        .collect::<Vec<_>>();
    let external_systems = sanitize_string_items(
        context
            .external_imports
            .iter()
            .map(|import| import.module.clone())
            .collect(),
        24,
    );
    let search_phrases = sanitize_string_items(
        std::iter::once(file.path.clone())
            .chain(std::iter::once(file.name.clone()))
            .chain(symbols.iter().map(|symbol| symbol.qualified_name.clone()))
            .collect(),
        32,
    );

    FileCard {
        file_id: file.file_id.clone(),
        summary,
        role: format!("{} source file", file.language),
        primary_behaviors: Vec::new(),
        behavior_intents: Vec::new(),
        edit_intents: Vec::new(),
        retrieval_tags: sanitize_retrieval_tags(
            std::iter::once("entity:file".to_string())
                .chain(std::iter::once(format!("language:{}", file.language)))
                .chain(std::iter::once(format!(
                    "path:{}",
                    file.path.replace('/', "-")
                )))
                .chain(
                    symbols
                        .iter()
                        .map(|symbol| format!("symbol:{}", symbol.name)),
                )
                .collect(),
            32,
        ),
        ownership_kind: FileOwnershipKind::Unknown,
        owns_behaviors: Vec::new(),
        delegates_to: imports_interpreted
            .iter()
            .map(|item| item.target_path.clone())
            .collect(),
        side_effects: Vec::new(),
        key_entities,
        external_systems,
        important_symbols,
        imports_interpreted,
        used_by_interpreted,
        blast_radius: Vec::new(),
        agent_read_hints: Vec::new(),
        search_phrases,
        risk_notes: Vec::new(),
        provenance: Provenance {
            source_hash: file.source_hash.clone(),
            input_hash: Some(input_hash),
            model: Some(model.to_string()),
            schema_version: matryoshka_core_ir::CARD_SCHEMA_VERSION,
            generated_at: Utc::now(),
        },
    }
}

fn strict_folder_card_from_summary(
    folder: &FolderFact,
    child_files: &[FileCard],
    child_folders: &[FolderCard],
    context: &FolderEnrichmentContext,
    summary: String,
    input_hash: String,
    model: &str,
) -> FolderCard {
    let key_entrypoints = sanitize_string_items(
        context
            .representative_child_files
            .iter()
            .map(|item| item.path.clone())
            .chain(child_folders.iter().map(|card| card.folder_id.clone()))
            .collect(),
        16,
    );

    FolderCard {
        folder_id: folder.folder_id.clone(),
        summary: summary.clone(),
        responsibility: summary,
        behavior_intents: Vec::new(),
        edit_intents: Vec::new(),
        retrieval_tags: structural_folder_tags(folder, child_files, child_folders),
        contains_kinds_of_files: sanitize_string_items(
            child_files
                .iter()
                .map(|card| card.file_id.clone())
                .chain(child_folders.iter().map(|card| card.folder_id.clone()))
                .collect(),
            32,
        ),
        incoming_dependencies_meaning: context
            .incoming_dependencies
            .iter()
            .map(|item| item.detail.clone())
            .collect(),
        outgoing_dependencies_meaning: context
            .outgoing_dependencies
            .iter()
            .map(|item| item.detail.clone())
            .collect(),
        key_entrypoints,
        common_behaviors: Vec::new(),
        subareas: subareas_from_child_folders(folder, child_folders),
        agent_guidance: Vec::new(),
        search_phrases: sanitize_string_items(
            std::iter::once(folder.folder_id.clone())
                .chain(child_files.iter().map(|card| card.file_id.clone()))
                .chain(child_folders.iter().map(|card| card.folder_id.clone()))
                .collect(),
            32,
        ),
        provenance: Provenance {
            source_hash: folder_source_hash(child_files, child_folders),
            input_hash: Some(input_hash),
            model: Some(model.to_string()),
            schema_version: matryoshka_core_ir::CARD_SCHEMA_VERSION,
            generated_at: Utc::now(),
        },
    }
}

fn strict_repo_card_from_summary(
    repo_root: &str,
    folders: &[FolderCard],
    summary: String,
    input_hash: String,
    model: &str,
) -> RepoCard {
    let top_level_subsystems = folders
        .iter()
        .filter(|folder| folder.folder_id != "repo")
        .take(16)
        .map(|folder| SubareaSummary {
            id: folder.folder_id.clone(),
            name: folder.folder_id.clone(),
            responsibility: folder.responsibility.clone(),
        })
        .collect::<Vec<_>>();
    let entrypoints = sanitize_string_items(
        folders
            .iter()
            .flat_map(|folder| folder.key_entrypoints.clone())
            .collect(),
        32,
    );
    let search_phrases = sanitize_string_items(
        std::iter::once(repo_root.to_string())
            .chain(folders.iter().map(|folder| folder.folder_id.clone()))
            .collect(),
        32,
    );

    RepoCard {
        repo_root: repo_root.to_string(),
        summary,
        behavior_intents: Vec::new(),
        edit_intents: Vec::new(),
        retrieval_tags: sanitize_retrieval_tags(
            std::iter::once("entity:repo".to_string())
                .chain(
                    folders
                        .iter()
                        .flat_map(|folder| folder.retrieval_tags.clone()),
                )
                .collect(),
            32,
        ),
        top_level_subsystems,
        cross_subsystem_flows: Vec::new(),
        entrypoints,
        high_risk_areas: Vec::new(),
        agent_navigation_hints: Vec::new(),
        search_phrases,
        provenance: Provenance {
            source_hash: folder_source_hash(&[], folders),
            input_hash: Some(input_hash),
            model: Some(model.to_string()),
            schema_version: matryoshka_core_ir::CARD_SCHEMA_VERSION,
            generated_at: Utc::now(),
        },
    }
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

fn response_with_body_on_error(response: Response, label: &str) -> Result<Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response
        .text()
        .unwrap_or_else(|err| format!("<failed to read error body: {err}>"));
    Err(anyhow!("{label} returned {status}: {body}"))
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
    fn strict_folder_card_uses_mlx_summary_without_placeholder_prose() {
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

        let card = strict_folder_card_from_summary(
            &folder,
            &[],
            &child_folders,
            &context,
            "Owns the gateway source layout and routes engineers to app and crate areas.".into(),
            "input-hash".into(),
            "mlx-test",
        );

        assert!(card.summary.contains("gateway source layout"));
        assert_eq!(card.provenance.model.as_deref(), Some("mlx-test"));
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
            .context("failed to call chunk summary endpoint")
            .and_then(|response| response_with_body_on_error(response, "chunk summary endpoint"))?;

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

        let mut ok = Vec::with_capacity(chunks.len());
        let mut errors = Vec::new();
        for result in drafts {
            match result {
                Ok(draft) => ok.push(draft),
                Err(err) => errors.push(format!("{err:#}")),
            }
        }
        if !errors.is_empty() {
            return Err(anyhow!(
                "{} chunk summary request(s) failed; first error: {}",
                errors.len(),
                errors.first().cloned().unwrap_or_default()
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
        let mut errors = Vec::new();
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
                    Err(err) => errors.push(format!("{err:#}")),
                }
            }
        }

        if !errors.is_empty() {
            return Err(anyhow!(
                "{} chunk summary request(s) failed across {} batches; first error: {}",
                errors.len(),
                total_batches,
                errors.first().cloned().unwrap_or_default()
            ));
        }
        Ok(all_drafts)
    }
}

#[cfg(test)]
mod chunk_summarizer_tests {
    use super::*;
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
