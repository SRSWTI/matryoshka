use crate::{
    CodeEnricher, ENRICHMENT_MODEL, HeuristicEnricher, file_single_pass_enrichment_prompt,
    folder_single_pass_enrichment_prompt,
};
use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use matryoshka_core_ir::{
    FileCard, FileEnrichmentContext, FileFact, FolderCard, FolderEnrichmentContext, FolderFact,
    Provenance, RepoCard, SubareaSummary, SymbolBehavior, SymbolFact,
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
                    content: "Return only valid JSON matching the requested shape. Be specific, behavioral, and useful for a coding agent.",
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
        let heuristic_risk_notes = card.risk_notes.clone();
        let mut prompt_hashes = Vec::new();

        let prompt = file_single_pass_enrichment_prompt(file, symbols, context);
        prompt_hashes.push(hash_json(&prompt)?);
        let input_hash = hash_json(&json!(prompt_hashes))?;
        let draft = match self
            .complete_typed::<FileCardSinglePassDraft>(
                prompt,
                "file_card_single_pass_draft",
                "Single-pass behavioral, retrieval, and editing-risk enrichment for a code-intelligence file card",
                2600,
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
        apply_file_single_pass_draft(&mut card, draft, file, symbols, context);

        card.risk_notes =
            remove_heuristic_placeholder_notes(card.risk_notes, &heuristic_risk_notes);
        card.provenance = Provenance {
            source_hash: file.source_hash.clone(),
            input_hash: Some(input_hash),
            model: Some(self.model.clone()),
            schema_version: matryoshka_core_ir::CARD_SCHEMA_VERSION,
            generated_at: Utc::now(),
        };
        if card.summary.trim().len() < 40 || card.role.trim().len() < 20 {
            card.risk_notes.push(
                "Enrichment quality warning: summary or role was shorter than expected.".into(),
            );
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

        let prompt = folder_single_pass_enrichment_prompt(
            folder,
            &child_values,
            &child_folder_values,
            context,
        );
        prompt_hashes.push(hash_json(&prompt)?);
        let input_hash = hash_json(&json!(prompt_hashes))?;
        let draft = match self
            .complete_typed::<FolderCardSinglePassDraft>(
                prompt,
                "folder_card_single_pass_draft",
                "Single-pass responsibility, retrieval, and editing-risk enrichment for a code-intelligence folder card",
                2200,
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

        let mut card = FolderCard {
            folder_id: folder.folder_id.clone(),
            summary: draft.summary.trim().to_string(),
            responsibility: draft.responsibility.trim().to_string(),
            behavior_intents: Vec::new(),
            edit_intents: Vec::new(),
            retrieval_tags: Vec::new(),
            contains_kinds_of_files: sanitize_string_items(draft.contains_kinds_of_files, 8),
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
            key_entrypoints: Vec::new(),
            common_behaviors: Vec::new(),
            subareas: sanitize_subareas(draft.subareas, folder),
            agent_guidance: Vec::new(),
            search_phrases: Vec::new(),
            provenance: Provenance::source_only(""),
        };

        let anchors = folder_anchor_strings(folder, child_files, child_folders, context, &card);
        card.behavior_intents = sanitize_grounded_strings(draft.behavior_intents, &anchors, 12);
        card.common_behaviors = sanitize_grounded_strings(draft.common_behaviors, &anchors, 12);
        card.key_entrypoints =
            sanitize_key_entrypoints(draft.key_entrypoints, folder, child_files, child_folders, 8);
        card.edit_intents = sanitize_grounded_strings(draft.edit_intents, &anchors, 12);
        card.retrieval_tags = sanitize_grounded_tags(
            sanitize_retrieval_tags(draft.retrieval_tags, 24),
            &anchors,
            &FileFact {
                file_id: folder.folder_id.clone(),
                path: folder.path.clone(),
                name: folder.name.clone(),
                language: "folder".into(),
                parent_folder_id: folder
                    .parent_folder_id
                    .clone()
                    .unwrap_or_else(|| "repo".into()),
                source_hash: String::new(),
                line_count: 0,
                imports: Vec::new(),
                snippets: Vec::new(),
            },
            folder.folder_id.as_str(),
        );
        card.agent_guidance = sanitize_grounded_strings(draft.agent_guidance, &anchors, 6);
        card.search_phrases = sanitize_search_phrases(draft.search_phrases, &anchors, 12);

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
struct FileCardSinglePassDraft {
    summary: String,
    role: String,
    primary_behaviors: Vec<String>,
    behavior_intents: Vec<String>,
    edit_intents: Vec<String>,
    retrieval_tags: Vec<String>,
    search_phrases: Vec<String>,
    agent_read_hints: Vec<String>,
    side_effects: Vec<String>,
    key_entities: Vec<String>,
    external_systems: Vec<String>,
    important_symbols: Vec<SymbolBehavior>,
    risk_notes: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[schemars(deny_unknown_fields)]
struct FolderCardSinglePassDraft {
    summary: String,
    responsibility: String,
    behavior_intents: Vec<String>,
    contains_kinds_of_files: Vec<String>,
    common_behaviors: Vec<String>,
    subareas: Vec<SubareaSummary>,
    key_entrypoints: Vec<String>,
    edit_intents: Vec<String>,
    retrieval_tags: Vec<String>,
    agent_guidance: Vec<String>,
    search_phrases: Vec<String>,
}

fn apply_file_single_pass_draft(
    card: &mut FileCard,
    draft: FileCardSinglePassDraft,
    file: &FileFact,
    symbols: &[SymbolFact],
    context: &FileEnrichmentContext,
) {
    card.summary = draft.summary;
    card.role = draft.role;
    let primary_behaviors = sanitize_string_items(draft.primary_behaviors, 8);
    if !primary_behaviors.is_empty() {
        card.primary_behaviors = primary_behaviors;
    }
    card.owns_behaviors = card.primary_behaviors.iter().take(6).cloned().collect();

    let anchors = file_anchor_strings(file, symbols, context, card);
    let behavior_intents = sanitize_grounded_strings(draft.behavior_intents, &anchors, 12);
    if !behavior_intents.is_empty() {
        card.behavior_intents = behavior_intents;
    }
    let edit_intents = sanitize_grounded_strings(draft.edit_intents, &anchors, 12);
    if !edit_intents.is_empty() {
        card.edit_intents = edit_intents;
    }
    let retrieval_tags = sanitize_grounded_tags(
        sanitize_retrieval_tags(draft.retrieval_tags, 24),
        &anchors,
        file,
        context.parent_folder_id.as_str(),
    );
    if !retrieval_tags.is_empty() {
        card.retrieval_tags = merge_retrieval_tags(retrieval_tags, card.retrieval_tags.clone(), 24);
    }
    let search_phrases = sanitize_search_phrases(draft.search_phrases, &anchors, 12);
    if !search_phrases.is_empty() {
        card.search_phrases = search_phrases;
    }
    let agent_read_hints = sanitize_grounded_strings(draft.agent_read_hints, &anchors, 6);
    if !agent_read_hints.is_empty() {
        card.agent_read_hints = agent_read_hints;
    }
    card.side_effects =
        sanitize_side_effects(draft.side_effects, file, context, &card.side_effects);
    card.key_entities = sanitize_key_entities(
        draft.key_entities,
        file,
        symbols,
        context,
        &card.key_entities,
    );
    card.external_systems =
        sanitize_external_systems(draft.external_systems, context, &card.external_systems);
    let important_symbols = sanitize_important_symbols(draft.important_symbols, symbols);
    if !important_symbols.is_empty() {
        card.important_symbols = important_symbols;
    }
    card.risk_notes
        .extend(sanitize_risk_notes(draft.risk_notes, &anchors, file, 6));
}

fn sanitize_important_symbols(
    symbols_from_model: Vec<SymbolBehavior>,
    actual_symbols: &[SymbolFact],
) -> Vec<SymbolBehavior> {
    let allowed = actual_symbols
        .iter()
        .map(|symbol| symbol.symbol_id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    symbols_from_model
        .into_iter()
        .filter(|symbol| allowed.contains(symbol.symbol_id.as_str()))
        .take(8)
        .collect()
}

fn file_anchor_strings(
    file: &FileFact,
    symbols: &[SymbolFact],
    context: &FileEnrichmentContext,
    card: &FileCard,
) -> Vec<String> {
    let mut anchors = vec![
        file.path.clone(),
        file.name.clone(),
        context.parent_folder_id.clone(),
        card.role.clone(),
        card.summary.clone(),
    ];
    anchors.extend(card.primary_behaviors.clone());
    anchors.extend(card.behavior_intents.clone());
    anchors.extend(card.edit_intents.clone());
    anchors.extend(card.retrieval_tags.clone());
    anchors.extend(symbols.iter().map(|symbol| symbol.name.clone()));
    anchors.extend(symbols.iter().map(|symbol| symbol.qualified_name.clone()));
    anchors.extend(
        context
            .internal_imports
            .iter()
            .map(|import| import.module.clone()),
    );
    anchors.extend(
        context
            .internal_imports
            .iter()
            .filter_map(|import| import.resolved_path.clone()),
    );
    anchors.extend(
        context
            .external_imports
            .iter()
            .map(|import| import.module.clone()),
    );
    anchors
}

fn folder_anchor_strings(
    folder: &FolderFact,
    child_files: &[FileCard],
    child_folders: &[FolderCard],
    context: &FolderEnrichmentContext,
    card: &FolderCard,
) -> Vec<String> {
    let mut anchors = vec![
        folder.folder_id.clone(),
        folder.path.clone(),
        folder.name.clone(),
        card.summary.clone(),
        card.responsibility.clone(),
    ];
    anchors.extend(card.behavior_intents.clone());
    anchors.extend(card.edit_intents.clone());
    anchors.extend(card.retrieval_tags.clone());
    anchors.extend(card.common_behaviors.clone());
    anchors.extend(child_files.iter().map(|file| file.file_id.clone()));
    anchors.extend(
        child_files
            .iter()
            .flat_map(|file| file.primary_behaviors.clone()),
    );
    anchors.extend(child_folders.iter().map(|folder| folder.folder_id.clone()));
    anchors.extend(
        child_folders
            .iter()
            .flat_map(|folder| [folder.summary.clone(), folder.responsibility.clone()]),
    );
    anchors.extend(
        child_folders
            .iter()
            .flat_map(|folder| folder.common_behaviors.clone()),
    );
    anchors.extend(
        context
            .incoming_dependencies
            .iter()
            .map(|item| item.path.clone()),
    );
    anchors.extend(
        context
            .outgoing_dependencies
            .iter()
            .map(|item| item.path.clone()),
    );
    anchors
}

fn sanitize_grounded_strings(items: Vec<String>, anchors: &[String], limit: usize) -> Vec<String> {
    let anchor_tokens = collect_anchor_tokens(anchors);
    sanitize_string_items(items, limit * 3)
        .into_iter()
        .filter(|item| is_grounded_phrase(item, &anchor_tokens))
        .take(limit)
        .collect()
}

fn sanitize_grounded_tags(
    tags: Vec<String>,
    anchors: &[String],
    file_like: &FileFact,
    folder_id: &str,
) -> Vec<String> {
    let anchor_tokens = collect_anchor_tokens(anchors);
    let exact_path_tag = format!(
        "path:{}",
        collapse_dashes(&file_like.path.replace('/', "-"))
    );
    let exact_folder_tag = format!("folder:{}", collapse_dashes(&folder_id.replace('/', "-")));
    tags.into_iter()
        .filter(|tag| {
            let prefix = tag.split(':').next().unwrap_or_default();
            match prefix {
                "artifact" | "entity" | "language" | "role" | "core" | "ownership" => true,
                "path" => tag == &exact_path_tag,
                "folder" => tag == &exact_folder_tag,
                "behavior" | "edit" | "dependency" => {
                    let tokens = phrase_tokens(tag);
                    !tokens.is_empty() && tokens.iter().any(|token| anchor_tokens.contains(token))
                }
                _ => false,
            }
        })
        .take(24)
        .collect()
}

fn sanitize_side_effects(
    items: Vec<String>,
    file: &FileFact,
    context: &FileEnrichmentContext,
    fallback: &[String],
) -> Vec<String> {
    let external = context
        .external_imports
        .iter()
        .map(|import| import.module.to_lowercase())
        .collect::<Vec<_>>();
    let has_database = external
        .iter()
        .any(|item| item.contains("sql") || item.contains("sqlite"));
    let has_filesystem = external
        .iter()
        .any(|item| item.contains("path") || item.contains("fs"));
    let has_network = external
        .iter()
        .any(|item| item.contains("reqwest") || item.contains("http") || item.contains("hyper"));
    let sanitized = sanitize_string_items(items, 8)
        .into_iter()
        .filter(|item| {
            let lowered = item.to_lowercase();
            if lowered.contains("uncertain") || lowered.contains("potential") {
                return false;
            }
            (has_database
                && (lowered.contains("database")
                    || lowered.contains("sqlite")
                    || lowered.contains("sql")))
                || (has_filesystem
                    && (lowered.contains("filesystem")
                        || lowered.contains("file")
                        || lowered.contains("directory")
                        || lowered.contains("path")))
                || (has_network
                    && (lowered.contains("network")
                        || lowered.contains("http")
                        || lowered.contains("request")
                        || lowered.contains("stream")))
                || lowered.contains(&file.name.to_lowercase())
                || lowered.contains(&file.path.to_lowercase())
        })
        .take(6)
        .collect::<Vec<_>>();

    if sanitized.is_empty() {
        fallback.to_vec()
    } else {
        merge_preferred_strings(sanitized, fallback.to_vec(), 6)
    }
}

fn sanitize_key_entities(
    items: Vec<String>,
    file: &FileFact,
    symbols: &[SymbolFact],
    context: &FileEnrichmentContext,
    fallback: &[String],
) -> Vec<String> {
    let allowed = std::iter::once(file.name.to_lowercase())
        .chain(std::iter::once(file.path.to_lowercase()))
        .chain(symbols.iter().map(|symbol| symbol.name.to_lowercase()))
        .chain(
            symbols
                .iter()
                .map(|symbol| symbol.qualified_name.to_lowercase()),
        )
        .chain(
            context
                .internal_imports
                .iter()
                .filter_map(|import| import.resolved_path.as_ref())
                .map(|path| path.to_lowercase()),
        )
        .collect::<std::collections::BTreeSet<_>>();
    let sanitized = sanitize_string_items(items, 16)
        .into_iter()
        .filter(|item| {
            let lowered = item.to_lowercase();
            allowed.iter().any(|allowed_item| {
                lowered.contains(allowed_item) || allowed_item.contains(&lowered)
            })
        })
        .take(10)
        .collect::<Vec<_>>();
    if sanitized.is_empty() {
        fallback.to_vec()
    } else {
        merge_preferred_strings(sanitized, fallback.to_vec(), 10)
    }
}

fn sanitize_external_systems(
    items: Vec<String>,
    context: &FileEnrichmentContext,
    fallback: &[String],
) -> Vec<String> {
    let allowed = context
        .external_imports
        .iter()
        .map(|import| import.module.to_lowercase())
        .collect::<std::collections::BTreeSet<_>>();
    let sanitized = sanitize_string_items(items, 12)
        .into_iter()
        .filter(|item| {
            let lowered = item.to_lowercase();
            allowed.iter().any(|allowed_item| {
                lowered.contains(allowed_item) || allowed_item.contains(&lowered)
            })
        })
        .take(8)
        .collect::<Vec<_>>();
    if sanitized.is_empty() {
        fallback.to_vec()
    } else {
        merge_preferred_strings(sanitized, fallback.to_vec(), 8)
    }
}

fn sanitize_search_phrases(items: Vec<String>, anchors: &[String], limit: usize) -> Vec<String> {
    let anchor_tokens = collect_anchor_tokens(anchors);
    sanitize_string_items(items, limit * 3)
        .into_iter()
        .filter(|item| item.len() >= 12)
        .filter(|item| {
            let tokens = phrase_tokens(item);
            tokens.iter().any(|token| anchor_tokens.contains(token))
        })
        .take(limit)
        .collect()
}

fn sanitize_risk_notes(
    items: Vec<String>,
    anchors: &[String],
    file: &FileFact,
    limit: usize,
) -> Vec<String> {
    let anchor_tokens = collect_anchor_tokens(anchors);
    sanitize_string_items(items, limit * 3)
        .into_iter()
        .filter(|item| {
            let lowered = item.to_lowercase();
            lowered.contains(&file.name.to_lowercase())
                || lowered.contains(&file.path.to_lowercase())
                || phrase_tokens(item)
                    .iter()
                    .any(|token| anchor_tokens.contains(token))
        })
        .take(limit)
        .collect()
}

fn sanitize_subareas(items: Vec<SubareaSummary>, folder: &FolderFact) -> Vec<SubareaSummary> {
    let allowed_child_folders = folder
        .child_folder_ids
        .iter()
        .map(|id| id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    if allowed_child_folders.is_empty() {
        return Vec::new();
    }

    items
        .into_iter()
        .filter_map(|item| {
            let id = item.id.trim().to_string();
            let name = item.name.trim().to_string();
            let responsibility = item.responsibility.trim().to_string();
            if id.is_empty() || name.is_empty() || responsibility.is_empty() {
                return None;
            }
            if !allowed_child_folders.contains(id.as_str())
                && !allowed_child_folders.contains(name.as_str())
            {
                return None;
            }
            Some(SubareaSummary {
                id,
                name,
                responsibility,
            })
        })
        .take(6)
        .collect()
}

fn sanitize_key_entrypoints(
    items: Vec<String>,
    folder: &FolderFact,
    child_files: &[FileCard],
    child_folders: &[FolderCard],
    limit: usize,
) -> Vec<String> {
    let allowed = child_files
        .iter()
        .map(|card| card.file_id.as_str())
        .chain(folder.child_file_ids.iter().map(|id| id.as_str()))
        .chain(child_folders.iter().map(|card| card.folder_id.as_str()))
        .chain(folder.child_folder_ids.iter().map(|id| id.as_str()))
        .collect::<std::collections::BTreeSet<_>>();
    sanitize_string_items(items, limit * 2)
        .into_iter()
        .filter(|item| allowed.contains(item.as_str()))
        .take(limit)
        .collect()
}

fn collect_anchor_tokens(items: &[String]) -> std::collections::BTreeSet<String> {
    items.iter().flat_map(|item| phrase_tokens(item)).collect()
}

fn phrase_tokens(item: &str) -> std::collections::BTreeSet<String> {
    item.split(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != '/')
        .map(|token| token.to_ascii_lowercase())
        .filter(|token| token.len() > 2)
        .collect()
}

fn is_grounded_phrase(item: &str, anchor_tokens: &std::collections::BTreeSet<String>) -> bool {
    let tokens = phrase_tokens(item);
    !tokens.is_empty() && tokens.iter().any(|token| anchor_tokens.contains(token))
}

fn merge_preferred_strings(
    preferred: Vec<String>,
    fallback: Vec<String>,
    limit: usize,
) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    preferred
        .into_iter()
        .chain(fallback)
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .filter(|item| seen.insert(item.to_lowercase()))
        .take(limit)
        .collect()
}

fn merge_retrieval_tags(
    preferred: Vec<String>,
    fallback: Vec<String>,
    limit: usize,
) -> Vec<String> {
    let structural_fallback = fallback.into_iter().filter(|tag| {
        matches!(
            tag.split(':').next().unwrap_or_default(),
            "artifact"
                | "entity"
                | "language"
                | "path"
                | "folder"
                | "role"
                | "dependency"
                | "core"
                | "ownership"
        )
    });
    merge_preferred_strings(preferred, structural_fallback.collect(), limit)
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

fn remove_heuristic_placeholder_notes(
    current_notes: Vec<String>,
    heuristic_notes: &[String],
) -> Vec<String> {
    let heuristic_set = heuristic_notes
        .iter()
        .map(|note| note.to_lowercase())
        .collect::<std::collections::BTreeSet<_>>();

    current_notes
        .into_iter()
        .filter(|note| {
            let lowered = note.to_lowercase();
            !lowered.contains("heuristic card:") && !heuristic_set.contains(&lowered)
        })
        .collect()
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
    fn staged_file_schema_payload_contains_nested_type_definitions() {
        let payload = json_schema_payload::<FileCardSinglePassDraft>(
            "file_card_single_pass_draft",
            "Single-pass behavioral, retrieval, and editing-risk enrichment for a code-intelligence file card",
        );
        let schema = payload
            .get("json_schema")
            .and_then(|value| value.get("schema"))
            .cloned()
            .expect("schema payload should contain schema");

        assert_eq!(schema.get("type").and_then(Value::as_str), Some("object"));
        assert!(schema.get("properties").is_some());
        assert!(contains_key(&schema, "$ref"));
        assert!(
            schema.get("definitions").is_some() || schema.get("$defs").is_some(),
            "nested schema references must be accompanied by definitions"
        );
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

    fn contains_key(value: &Value, needle: &str) -> bool {
        match value {
            Value::Object(map) => {
                map.contains_key(needle) || map.values().any(|entry| contains_key(entry, needle))
            }
            Value::Array(items) => items.iter().any(|entry| contains_key(entry, needle)),
            _ => false,
        }
    }
}
