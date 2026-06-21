use chrono::{DateTime, Utc};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const IR_SCHEMA_VERSION: u32 = 1;
pub const CARD_SCHEMA_VERSION: u32 = 1;
pub const SEMANTIC_SCHEMA_VERSION: u32 = 1;
pub const CHUNK_SCHEMA_VERSION: u32 = 1;

/// Minimum length (in characters) for a docstring/doc comment to be considered
/// "useful" and used directly without invoking the LLM summarizer.
pub const MIN_USEFUL_DOC_SUMMARY_LEN: usize = 12;

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
    #[serde(default)]
    pub code_chunks: Vec<CodeChunkFact>,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
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

/// A first-class semantic chunk derived from a tree-sitter symbol boundary
/// (function, method, class, struct, etc.). Unlike `SnippetFact`, a code chunk
/// preserves the full symbol body, carries an optional docstring/doc comment,
/// and is the unit that gets summarized and embedded for retrieval.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeChunkFact {
    pub chunk_id: String,
    pub file_id: String,
    pub symbol_id: Option<String>,
    pub path: String,
    pub symbol: Option<String>,
    pub qualified_name: Option<String>,
    pub kind: CodeChunkKind,
    pub signature: String,
    pub start_line: usize,
    pub end_line: usize,
    /// Docstring / doc comment extracted directly from source, if present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_summary: Option<String>,
    /// Summary produced by the LLM chunk summarizer, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_summary: Option<String>,
    /// The summary actually used for retrieval (doc_summary if useful, else generated).
    pub summary: String,
    pub summary_source: ChunkSummarySource,
    /// Full source text of the chunk (no truncation).
    pub code: String,
    pub source_hash: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CodeChunkKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Interface,
    Module,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ChunkSummarySource {
    /// Summary came from a Python docstring.
    Docstring,
    /// Summary came from a leading doc comment (Rust `///`, TS `/** */`, etc.).
    DocComment,
    /// Summary came from a file/module-level header doc.
    FileHeader,
    /// Summary was generated by the LLM chunk summarizer.
    Llm,
    /// No summary is available yet.
    Empty,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LateInteractionVector {
    pub record_id: String,
    pub token: String,
    pub ordinal: usize,
    pub weight: f32,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SemanticEntityType {
    File,
    Folder,
    Symbol,
    Snippet,
    CodeChunk,
    Repo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchHit {
    pub entity_id: String,
    pub record_id: String,
    pub path: String,
    pub title: String,
    pub entity_type: SemanticEntityType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub behaviors: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub matched_terms: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub matched_symbols: Vec<String>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub total_matched_symbols: usize,
    pub score: f32,
    pub why_matched: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReadCard {
    pub file: ReadFileOverview,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub folder: Option<ReadFolderOverview>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub symbols: Vec<ReadSymbol>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chunks: Vec<ReadCodeChunk>,
    pub imports: ReadImports,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependents: Vec<ReadDependency>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<ReadDependency>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub total_dependents: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub total_depends_on: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompactReadCard {
    pub file: ReadFileOverview,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub folder: Option<ReadFolderOverview>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub symbols: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chunks: Vec<ReadCodeChunk>,
    pub imports: ReadImports,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependents: Vec<ReadDependency>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<ReadDependency>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub total_dependents: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub total_depends_on: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadFileOverview {
    pub file_id: String,
    pub path: String,
    pub name: String,
    pub language: String,
    pub parent_folder_id: String,
    pub source_hash: String,
    pub line_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadFolderOverview {
    pub folder_id: String,
    pub summary: String,
    pub responsibility: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadSymbol {
    pub name: String,
    pub qualified_name: String,
    pub kind: SymbolKind,
    pub signature: String,
    pub lines: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub behavior: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadCodeChunk {
    pub chunk_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qualified_name: Option<String>,
    pub kind: CodeChunkKind,
    pub signature: String,
    pub lines: String,
    pub summary_source: ChunkSummarySource,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadImports {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub internal: Vec<ReadInternalImport>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadInternalImport {
    pub module: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub names: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReadDependency {
    pub path: String,
    pub relationships: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InvalidationSet {
    pub file_ids: Vec<String>,
    pub folder_ids: Vec<String>,
    pub repo_stale: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ArtifactQualityReport {
    pub file_cards: usize,
    pub file_cards_with_summary: usize,
    pub file_cards_empty_summary: usize,
    pub folder_cards: usize,
    pub folder_cards_with_summary: usize,
    pub folder_cards_empty_summary: usize,
    pub repo_card_has_summary: bool,
    pub empty_file_summary_samples: Vec<String>,
    pub empty_folder_summary_samples: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalPrimary {
    Fts,
    Splade,
    Dense,
    #[default]
    Hybrid,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetrievalConfig {
    pub primary: RetrievalPrimary,
    pub dense_enabled: bool,
    pub dense_fallback_enabled: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            primary: RetrievalPrimary::Hybrid,
            dense_enabled: true,
            dense_fallback_enabled: true,
        }
    }
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetrievalIndexReport {
    pub semantic_records: usize,
    pub embedded_records: usize,
    pub fts_records: usize,
    pub late_vector_rows: usize,
    pub records_with_late_vectors: usize,
    #[serde(default)]
    pub retrieval_primary: RetrievalPrimary,
    #[serde(default = "default_true")]
    pub dense_enabled: bool,
    #[serde(default = "default_true")]
    pub dense_fallback_enabled: bool,
    pub late_interaction_enabled: bool,
}

impl Default for RetrievalIndexReport {
    fn default() -> Self {
        Self {
            semantic_records: 0,
            embedded_records: 0,
            fts_records: 0,
            late_vector_rows: 0,
            records_with_late_vectors: 0,
            retrieval_primary: RetrievalPrimary::Hybrid,
            dense_enabled: true,
            dense_fallback_enabled: true,
            late_interaction_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MatryoshkaProgressEvent {
    Started {
        total_steps: Option<usize>,
    },
    DiscoveringFiles,
    FilesDiscovered {
        total_files: usize,
    },
    ParsingFile {
        path: String,
        index: usize,
        total_files: usize,
    },
    ParsedFile {
        path: String,
        index: usize,
        total_files: usize,
    },
    EnrichingFile {
        path: String,
        index: usize,
        total_files: usize,
    },
    EnrichedFile {
        path: String,
        index: usize,
        total_files: usize,
    },
    EnrichingChunks {
        chunk_count: usize,
    },
    EnrichingChunkBatch {
        batch_index: usize,
        total_batches: usize,
        chunks_in_batch: usize,
    },
    EnrichedChunkBatch {
        batch_index: usize,
        total_batches: usize,
        chunks_in_batch: usize,
    },
    EnrichedChunks {
        chunk_count: usize,
    },
    EmbeddingBatch {
        batch_index: usize,
        total_batches: usize,
        records_in_batch: usize,
    },
    EmbeddedBatch {
        batch_index: usize,
        total_batches: usize,
        records_in_batch: usize,
    },
    EmbeddingSkipped {
        record_count: usize,
        reason: String,
    },
    WritingDatabase {
        records_written: Option<usize>,
    },
    ArtifactQuality {
        report: ArtifactQualityReport,
    },
    RetrievalIndexHealth {
        report: RetrievalIndexReport,
    },
    Completed {
        file_count: usize,
        folder_count: usize,
        symbol_count: usize,
        semantic_record_count: usize,
        embedding_model: String,
    },
    Failed {
        stage: String,
        message: String,
    },
}
