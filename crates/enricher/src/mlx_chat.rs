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
        let value = self.complete_json_with_schema(
            prompt,
            json_schema_payload::<T>(name, description),
            max_tokens,
        )?;
        serde_json::from_value(value).context("structured response did not match target type")
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
        let is_facade = is_thin_facade(file, symbols, context);

        if !is_facade {
            let prompt = file_single_pass_enrichment_prompt(file, symbols, context);
            prompt_hashes.push(hash_json(&prompt)?);
            let draft = self.complete_typed::<FileCardSinglePassDraft>(
                prompt,
                "file_card_single_pass_draft",
                "Single-pass behavioral, retrieval, and editing-risk enrichment for a code-intelligence file card",
                2600,
            );
            match draft {
                Ok(draft) => apply_file_single_pass_draft(&mut card, draft, file, symbols, context),
                Err(error) => card.risk_notes.push(format!(
                    "MLX file enrichment degraded to heuristic synthesis: {error:#}"
                )),
            }
        }

        card.risk_notes =
            remove_heuristic_placeholder_notes(card.risk_notes, &heuristic_risk_notes);
        card.provenance = Provenance {
            source_hash: file.source_hash.clone(),
            input_hash: Some(hash_json(&json!(prompt_hashes))?),
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
        context: &FolderEnrichmentContext,
    ) -> Result<FolderCard> {
        let mut card = HeuristicEnricher.enrich_folder(folder, child_files, context)?;
        let child_values = child_files
            .iter()
            .map(serde_json::to_value)
            .collect::<Result<Vec<_>, _>>()?;
        let mut prompt_hashes = Vec::new();

        let prompt = folder_single_pass_enrichment_prompt(folder, &child_values, context);
        prompt_hashes.push(hash_json(&prompt)?);
        let draft = self.complete_typed::<FolderCardSinglePassDraft>(
            prompt,
            "folder_card_single_pass_draft",
            "Single-pass responsibility, retrieval, and editing-risk enrichment for a code-intelligence folder card",
            2200,
        );
        match draft {
            Ok(draft) => {
                let anchors = folder_anchor_strings(folder, child_files, context, &card);
                card.summary = draft.summary;
                card.responsibility = draft.responsibility;
                let behavior_intents =
                    sanitize_grounded_strings(draft.behavior_intents, &anchors, 12);
                if !behavior_intents.is_empty() {
                    card.behavior_intents =
                        merge_preferred_strings(behavior_intents, card.behavior_intents, 12);
                }
                let common_behaviors =
                    sanitize_grounded_strings(draft.common_behaviors, &anchors, 12);
                if !common_behaviors.is_empty() {
                    card.common_behaviors =
                        merge_preferred_strings(common_behaviors, card.common_behaviors, 12);
                }
                let key_entrypoints =
                    sanitize_key_entrypoints(draft.key_entrypoints, folder, child_files, 8);
                if !key_entrypoints.is_empty() {
                    card.key_entrypoints = key_entrypoints;
                }
                let edit_intents = sanitize_grounded_strings(draft.edit_intents, &anchors, 12);
                if !edit_intents.is_empty() {
                    card.edit_intents =
                        merge_preferred_strings(edit_intents, card.edit_intents, 12);
                }
                let retrieval_tags = sanitize_grounded_tags(
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
                if !retrieval_tags.is_empty() {
                    card.retrieval_tags =
                        merge_retrieval_tags(retrieval_tags, card.retrieval_tags, 24);
                }
                let contains_kinds_of_files =
                    sanitize_string_items(draft.contains_kinds_of_files, 8);
                if !contains_kinds_of_files.is_empty() {
                    card.contains_kinds_of_files = contains_kinds_of_files;
                }
                let subareas = sanitize_subareas(draft.subareas, folder);
                if !subareas.is_empty() {
                    card.subareas = subareas;
                }
                let agent_guidance = sanitize_grounded_strings(draft.agent_guidance, &anchors, 6);
                if !agent_guidance.is_empty() {
                    card.agent_guidance =
                        merge_preferred_strings(agent_guidance, card.agent_guidance, 6);
                }
                let search_phrases = sanitize_search_phrases(draft.search_phrases, &anchors, 12);
                if !search_phrases.is_empty() {
                    card.search_phrases =
                        merge_preferred_strings(search_phrases, card.search_phrases, 12);
                }
            }
            Err(error) => {
                card.agent_guidance.push(format!(
                    "MLX folder enrichment degraded to heuristic synthesis: {error:#}"
                ));
            }
        }

        card.provenance = Provenance {
            source_hash: child_files
                .iter()
                .map(|card| card.provenance.source_hash.as_str())
                .collect::<Vec<_>>()
                .join(":"),
            input_hash: Some(hash_json(&json!(prompt_hashes))?),
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

fn is_thin_facade(
    file: &FileFact,
    symbols: &[SymbolFact],
    context: &FileEnrichmentContext,
) -> bool {
    (file.name == "lib.rs" || file.name == "mod.rs")
        && file.line_count <= 20
        && symbols.is_empty()
        && context.internal_imports.is_empty()
        && context.external_imports.is_empty()
        && !context.sibling_file_ids.is_empty()
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
    limit: usize,
) -> Vec<String> {
    let allowed = child_files
        .iter()
        .map(|card| card.file_id.as_str())
        .chain(folder.child_file_ids.iter().map(|id| id.as_str()))
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
