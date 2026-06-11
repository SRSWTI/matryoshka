use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const IR_SCHEMA_VERSION: u32 = 1;
pub const CARD_SCHEMA_VERSION: u32 = 1;
pub const SEMANTIC_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Provenance {
    pub source_hash: String,
    pub input_hash: Option<String>,
    pub model: Option<String>,
    pub schema_version: u32,
    pub generated_at: DateTime<Utc>,
}

impl Provenance {
    pub fn source_only(source_hash: impl Into<String>) -> Self {
        Self {
            source_hash: source_hash.into(),
            input_hash: None,
            model: None,
            schema_version: IR_SCHEMA_VERSION,
            generated_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RepositorySnapshot {
    pub repo_root: String,
    pub indexed_at: DateTime<Utc>,
    pub files: Vec<FileFact>,
    pub folders: Vec<FolderFact>,
    pub symbols: Vec<SymbolFact>,
    pub edges: Vec<EdgeFact>,
    pub semantic_records: Vec<SemanticRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileFact {
    pub file_id: String,
    pub path: String,
    pub name: String,
    pub language: String,
    pub parent_folder_id: String,
    pub source_hash: String,
    pub line_count: usize,
    pub imports: Vec<ImportFact>,
    pub snippets: Vec<SnippetFact>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct RelatedFileContext {
    pub file_id: String,
    pub path: String,
    pub relationship: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ImportContext {
    pub module: String,
    pub names: Vec<String>,
    pub line: usize,
    pub dependency_kind: String,
    pub resolved_file_id: Option<String>,
    pub resolved_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct FileEnrichmentContext {
    pub parent_folder_id: String,
    pub sibling_file_ids: Vec<String>,
    pub internal_imports: Vec<ImportContext>,
    pub external_imports: Vec<ImportContext>,
    pub imported_by_files: Vec<RelatedFileContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FolderFact {
    pub folder_id: String,
    pub path: String,
    pub name: String,
    pub parent_folder_id: Option<String>,
    pub child_file_ids: Vec<String>,
    pub child_folder_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct FolderEnrichmentContext {
    pub parent_folder_id: Option<String>,
    pub incoming_dependencies: Vec<RelatedFileContext>,
    pub outgoing_dependencies: Vec<RelatedFileContext>,
    pub representative_child_files: Vec<RelatedFileContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SymbolFact {
    pub symbol_id: String,
    pub file_id: String,
    pub path: String,
    pub name: String,
    pub qualified_name: String,
    pub kind: SymbolKind,
    pub signature: String,
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Interface,
    TypeAlias,
    Constant,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImportFact {
    pub module: String,
    pub names: Vec<String>,
    pub line: usize,
    pub resolved_file_id: Option<String>,
    pub is_internal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnippetFact {
    pub snippet_id: String,
    pub file_id: String,
    pub title: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EdgeFact {
    pub edge_id: String,
    pub source_id: String,
    pub target_id: String,
    pub kind: EdgeKind,
    pub weight: u32,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    Contains,
    Imports,
    Calls,
    References,
    DependsOn,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FileOwnershipKind {
    Facade,
    Implementation,
    Mixed,
    Unknown,
}

impl Default for FileOwnershipKind {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileCard {
    pub file_id: String,
    pub summary: String,
    pub role: String,
    pub primary_behaviors: Vec<String>,
    #[serde(default)]
    pub behavior_intents: Vec<String>,
    #[serde(default)]
    pub edit_intents: Vec<String>,
    #[serde(default)]
    pub retrieval_tags: Vec<String>,
    #[serde(default)]
    pub ownership_kind: FileOwnershipKind,
    #[serde(default)]
    pub owns_behaviors: Vec<String>,
    #[serde(default)]
    pub delegates_to: Vec<String>,
    pub side_effects: Vec<String>,
    pub key_entities: Vec<String>,
    pub external_systems: Vec<String>,
    pub important_symbols: Vec<SymbolBehavior>,
    pub imports_interpreted: Vec<DependencyInterpretation>,
    pub used_by_interpreted: Vec<DependencyInterpretation>,
    pub blast_radius: Vec<String>,
    pub agent_read_hints: Vec<String>,
    pub search_phrases: Vec<String>,
    pub risk_notes: Vec<String>,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FolderCard {
    pub folder_id: String,
    pub summary: String,
    pub responsibility: String,
    #[serde(default)]
    pub behavior_intents: Vec<String>,
    #[serde(default)]
    pub edit_intents: Vec<String>,
    #[serde(default)]
    pub retrieval_tags: Vec<String>,
    pub contains_kinds_of_files: Vec<String>,
    pub incoming_dependencies_meaning: Vec<String>,
    pub outgoing_dependencies_meaning: Vec<String>,
    pub key_entrypoints: Vec<String>,
    pub common_behaviors: Vec<String>,
    pub subareas: Vec<SubareaSummary>,
    pub agent_guidance: Vec<String>,
    pub search_phrases: Vec<String>,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RepoCard {
    pub repo_root: String,
    pub summary: String,
    #[serde(default)]
    pub behavior_intents: Vec<String>,
    #[serde(default)]
    pub edit_intents: Vec<String>,
    #[serde(default)]
    pub retrieval_tags: Vec<String>,
    pub top_level_subsystems: Vec<SubareaSummary>,
    pub cross_subsystem_flows: Vec<String>,
    pub entrypoints: Vec<String>,
    pub high_risk_areas: Vec<String>,
    pub agent_navigation_hints: Vec<String>,
    pub search_phrases: Vec<String>,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SymbolBehavior {
    pub symbol_id: String,
    pub name: String,
    pub role: String,
    pub behavior: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct DependencyInterpretation {
    pub target_id: String,
    pub target_path: String,
    pub why: String,
    pub dependency_kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct SubareaSummary {
    pub id: String,
    pub name: String,
    pub responsibility: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SemanticRecord {
    pub record_id: String,
    pub entity_id: String,
    pub entity_type: SemanticEntityType,
    pub title: String,
    pub content: String,
    pub path: String,
    pub source_hash: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SemanticEntityType {
    File,
    Folder,
    Symbol,
    Snippet,
    Repo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchHit {
    pub entity_id: String,
    pub record_id: String,
    pub path: String,
    pub title: String,
    pub entity_type: SemanticEntityType,
    pub score: f32,
    pub why_matched: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReadCard {
    pub file: FileFact,
    pub file_card: Option<FileCard>,
    pub folder_card: Option<FolderCard>,
    pub symbols: Vec<SymbolFact>,
    pub imports: Vec<ImportFact>,
    pub incoming_edges: Vec<EdgeFact>,
    pub outgoing_edges: Vec<EdgeFact>,
    pub snippets: Vec<SnippetFact>,
    pub symbol_blocks: Vec<SnippetFact>,
    pub import_lines: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InvalidationSet {
    pub file_ids: Vec<String>,
    pub folder_ids: Vec<String>,
    pub repo_stale: bool,
    pub reason: String,
}
