use crate::{ChunkSummarizer, ChunkSummaryDraft, CodeEnricher};
use anyhow::Result;
use chrono::Utc;
use matryoshka_core_ir::{
    CodeChunkFact, DependencyInterpretation, FileCard, FileEnrichmentContext, FileFact,
    FileOwnershipKind, FolderCard, FolderEnrichmentContext, FolderFact, Provenance,
    RelatedFileContext, RepoCard, SubareaSummary, SymbolBehavior, SymbolFact,
};

#[derive(Debug, Default, Clone)]
pub struct HeuristicEnricher;

impl CodeEnricher for HeuristicEnricher {
    fn enrich_file(
        &self,
        file: &FileFact,
        symbols: &[SymbolFact],
        context: &FileEnrichmentContext,
    ) -> Result<FileCard> {
        let symbol_names = symbols
            .iter()
            .filter(|symbol| symbol.file_id == file.file_id)
            .map(|symbol| symbol.name.clone())
            .collect::<Vec<_>>();
        let is_facade = is_thin_facade(file, &symbol_names, context);
        let top_symbols = symbol_names.iter().take(4).cloned().collect::<Vec<_>>();
        let behavior_intents = Vec::new();
        let edit_intents = Vec::new();
        let retrieval_tags =
            retrieval_tags(file, &behavior_intents, &edit_intents, context, is_facade);
        let ownership_kind = ownership_kind(file, symbols, context, is_facade);
        let owns_behaviors = Vec::new();
        let delegates_to = delegates_to(context, is_facade);

        Ok(FileCard {
            file_id: file.file_id.clone(),
            summary: heuristic_file_summary(file, &symbol_names, context),
            role: heuristic_file_role(file, &symbol_names, context),
            primary_behaviors: Vec::new(),
            behavior_intents,
            edit_intents,
            retrieval_tags,
            ownership_kind,
            owns_behaviors,
            delegates_to,
            side_effects: side_effects(file, context),
            key_entities: key_entities(file, &symbol_names, context),
            external_systems: context
                .external_imports
                .iter()
                .map(|import| import.module.clone())
                .collect(),
            important_symbols: symbols
                .iter()
                .filter(|symbol| symbol.file_id == file.file_id)
                .take(12)
                .map(|symbol| SymbolBehavior {
                    symbol_id: symbol.symbol_id.clone(),
                    name: symbol.name.clone(),
                    role: format!("Top-level {:?} declared in this file.", symbol.kind),
                    behavior: symbol.signature.clone(),
                })
                .collect(),
            imports_interpreted: imports_interpreted(file, context),
            used_by_interpreted: context
                .imported_by_files
                .iter()
                .map(to_dependent_interpretation)
                .collect(),
            blast_radius: blast_radius(file, context),
            agent_read_hints: agent_read_hints(file, &top_symbols, context),
            search_phrases: Vec::new(),
            risk_notes: risk_notes(file, context),
            provenance: Provenance {
                source_hash: file.source_hash.clone(),
                input_hash: None,
                model: Some("heuristic".into()),
                schema_version: matryoshka_core_ir::CARD_SCHEMA_VERSION,
                generated_at: Utc::now(),
            },
        })
    }

    fn enrich_folder(
        &self,
        folder: &FolderFact,
        child_files: &[FileCard],
        child_folders: &[FolderCard],
        context: &FolderEnrichmentContext,
    ) -> Result<FolderCard> {
        let behavior_intents = Vec::new();
        let common_behaviors = Vec::new();
        let edit_intents = Vec::new();
        let retrieval_tags = dedupe_take(
            child_files
                .iter()
                .flat_map(|card| card.retrieval_tags.clone())
                .chain(
                    child_folders
                        .iter()
                        .flat_map(|card| card.retrieval_tags.clone()),
                )
                .chain(std::iter::once(format!("folder:{}", tagify(&folder.path))))
                .chain(std::iter::once(format!("layer:{}", tagify(&folder.name)))),
            20,
        );
        Ok(FolderCard {
            folder_id: folder.folder_id.clone(),
            summary: heuristic_folder_summary(folder, child_files, child_folders),
            responsibility: heuristic_folder_responsibility(folder, child_files, child_folders),
            behavior_intents,
            edit_intents,
            retrieval_tags,
            contains_kinds_of_files: folder_file_kinds(child_files),
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
                .take(8)
                .collect(),
            common_behaviors: common_behaviors.clone(),
            subareas: folder
                .child_folder_ids
                .iter()
                .map(|id| SubareaSummary {
                    id: id.clone(),
                    name: id.clone(),
                    responsibility: child_folders
                        .iter()
                        .find(|card| card.folder_id == *id)
                        .and_then(|card| {
                            (!card.responsibility.trim().is_empty())
                                .then(|| card.responsibility.clone())
                        })
                        .unwrap_or_default(),
                })
                .collect(),
            agent_guidance: Vec::new(),
            search_phrases: Vec::new(),
            provenance: Provenance {
                source_hash: child_files
                    .iter()
                    .map(|card| card.provenance.source_hash.as_str())
                    .collect::<Vec<_>>()
                    .join(":"),
                input_hash: None,
                model: Some("heuristic".into()),
                schema_version: matryoshka_core_ir::CARD_SCHEMA_VERSION,
                generated_at: Utc::now(),
            },
        })
    }

    fn enrich_repo(&self, repo_root: &str, folders: &[FolderCard]) -> Result<RepoCard> {
        let top_level_subsystems = folders
            .iter()
            .filter(|folder| !folder.folder_id.contains('/'))
            .map(|folder| SubareaSummary {
                id: folder.folder_id.clone(),
                name: folder.folder_id.clone(),
                responsibility: folder.responsibility.clone(),
            })
            .collect::<Vec<_>>();
        let repo_name = std::path::Path::new(repo_root)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(repo_root);
        let subsystem_names = top_level_subsystems
            .iter()
            .map(|item| item.name.clone())
            .take(6)
            .collect::<Vec<_>>();
        Ok(RepoCard {
            repo_root: repo_root.into(),
            summary: format!(
                "Repository {repo_name} contains {} enriched folders across {} top-level subsystems. The main navigation surface is organized around {}. Start with subsystem folders, then read file cards to move from public facades into behavior-owning implementation files.",
                folders.len(),
                top_level_subsystems.len(),
                if subsystem_names.is_empty() {
                    "the indexed code map".into()
                } else {
                    subsystem_names.join(", ")
                }
            ),
            behavior_intents: dedupe_take(
                folders
                    .iter()
                    .flat_map(|folder| folder.behavior_intents.clone()),
                20,
            ),
            edit_intents: dedupe_take(
                folders
                    .iter()
                    .flat_map(|folder| folder.edit_intents.clone()),
                20,
            ),
            retrieval_tags: dedupe_take(
                folders
                    .iter()
                    .flat_map(|folder| folder.retrieval_tags.clone())
                    .chain(std::iter::once("entity:repo".into()))
                    .chain(std::iter::once("artifact:repo-card".into()))
                    .chain(std::iter::once(format!("repo:{}", tagify(repo_name)))),
                32,
            ),
            top_level_subsystems: top_level_subsystems.clone(),
            cross_subsystem_flows: folders
                .iter()
                .flat_map(|folder| folder.outgoing_dependencies_meaning.clone())
                .take(16)
                .collect(),
            entrypoints: folders
                .iter()
                .flat_map(|folder| folder.key_entrypoints.clone())
                .take(16)
                .collect(),
            high_risk_areas: folders
                .iter()
                .filter(|folder| !folder.incoming_dependencies_meaning.is_empty())
                .take(8)
                .map(|folder| {
                    format!(
                        "{} has {} incoming dependency signals and deserves blast-radius checks before edits.",
                        folder.folder_id,
                        folder.incoming_dependencies_meaning.len()
                    )
                })
                .collect(),
            agent_navigation_hints: vec![
                "Use semantic search first for behavior, then read the returned file or folder card before opening raw source."
                    .into(),
                "For public API or entrypoint changes, inspect facade files first and then follow delegates_to or behavior-owner files."
                    .into(),
                "For architecture questions, start at the repo card and top-level folders before drilling into implementation files."
                    .into(),
            ],
            search_phrases: vec![
                format!("repository map {repo_name}"),
                "top level subsystem responsibilities".into(),
                format!("{repo_name} architecture overview"),
            ],
            provenance: Provenance {
                source_hash: folders
                    .iter()
                    .map(|folder| folder.provenance.source_hash.as_str())
                    .collect::<Vec<_>>()
                    .join(":"),
                input_hash: None,
                model: Some("heuristic".into()),
                schema_version: matryoshka_core_ir::CARD_SCHEMA_VERSION,
                generated_at: Utc::now(),
            },
        })
    }
}

fn heuristic_file_summary(
    file: &FileFact,
    symbol_names: &[String],
    context: &FileEnrichmentContext,
) -> String {
    let mut parts = Vec::new();
    parts.push(format!(
        "{} is a {} file with {} lines.",
        file.path, file.language, file.line_count
    ));
    if !symbol_names.is_empty() {
        parts.push(format!(
            "It defines {}.",
            symbol_names
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !context.internal_imports.is_empty() {
        parts.push(format!(
            "It uses internal code from {}.",
            context
                .internal_imports
                .iter()
                .take(3)
                .map(|item| item.module.clone())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    } else if !context.external_imports.is_empty() {
        parts.push(format!(
            "It uses external crates or modules such as {}.",
            context
                .external_imports
                .iter()
                .take(3)
                .map(|item| item.module.clone())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    parts.join(" ")
}

fn heuristic_file_role(
    file: &FileFact,
    symbol_names: &[String],
    context: &FileEnrichmentContext,
) -> String {
    let family = file_role_family(file);
    let symbol_text = if symbol_names.is_empty() {
        "its module-level code".to_string()
    } else {
        symbol_names
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "Acts as a {family} file for {}. It has {} internal imports and {} known dependents.",
        symbol_text,
        context.internal_imports.len(),
        context.imported_by_files.len()
    )
}

fn heuristic_folder_summary(
    folder: &FolderFact,
    child_files: &[FileCard],
    child_folders: &[FolderCard],
) -> String {
    let file_examples = folder_owner_files(child_files)
        .into_iter()
        .chain(folder_surface_files(child_files))
        .take(5)
        .collect::<Vec<_>>();
    let child_folder_examples = child_folders
        .iter()
        .map(|card| card.folder_id.clone())
        .take(5)
        .collect::<Vec<_>>();
    let mut parts = vec![format!(
        "{} groups {} direct files and {} child folders.",
        folder.path,
        folder.child_file_ids.len(),
        folder.child_folder_ids.len()
    )];
    if !file_examples.is_empty() {
        parts.push(format!(
            "Important child files include {}.",
            file_examples.join(", ")
        ));
    }
    if !child_folder_examples.is_empty() {
        parts.push(format!(
            "Important child folders include {}.",
            child_folder_examples.join(", ")
        ));
    }
    parts.join(" ")
}

fn heuristic_folder_responsibility(
    folder: &FolderFact,
    child_files: &[FileCard],
    child_folders: &[FolderCard],
) -> String {
    let kinds = folder_file_kinds(child_files);
    let kind_text = if kinds.is_empty() {
        "code organization".to_string()
    } else {
        kinds.into_iter().take(4).collect::<Vec<_>>().join("; ")
    };
    format!(
        "{} owns {} across {} files and {} child folders.",
        folder.path,
        kind_text,
        child_files.len(),
        child_folders.len()
    )
}

fn folder_owner_files(child_files: &[FileCard]) -> Vec<String> {
    dedupe_take(
        child_files
            .iter()
            .filter(|card| {
                matches!(
                    card.ownership_kind,
                    FileOwnershipKind::Implementation | FileOwnershipKind::Mixed
                )
            })
            .map(|card| card.file_id.clone()),
        4,
    )
}

fn folder_surface_files(child_files: &[FileCard]) -> Vec<String> {
    dedupe_take(
        child_files
            .iter()
            .filter(|card| card.ownership_kind == FileOwnershipKind::Facade)
            .map(|card| card.file_id.clone()),
        4,
    )
}

fn folder_file_kinds(child_files: &[FileCard]) -> Vec<String> {
    let owner_files = folder_owner_files(child_files);
    let surface_files = folder_surface_files(child_files);
    let supporting_files = dedupe_take(
        child_files
            .iter()
            .filter(|card| {
                !matches!(
                    card.ownership_kind,
                    FileOwnershipKind::Implementation | FileOwnershipKind::Mixed
                ) && card.ownership_kind != FileOwnershipKind::Facade
            })
            .map(|card| card.file_id.clone()),
        4,
    );

    let mut kinds = Vec::new();
    if !owner_files.is_empty() {
        kinds.push(format!(
            "Behavior-owner implementation files such as {}.",
            owner_files.join(", ")
        ));
    }
    if !surface_files.is_empty() {
        kinds.push(format!(
            "Facade or entrypoint files such as {} that route readers to deeper implementation.",
            surface_files.join(", ")
        ));
    }
    if !supporting_files.is_empty() {
        kinds.push(format!(
            "Supporting modules such as {} that round out the folder's behavior surface.",
            supporting_files.join(", ")
        ));
    }
    if kinds.is_empty() {
        kinds.push("Source files grouped under this path that collectively implement the folder's behavior.".into());
    }
    kinds
}

fn file_role_family(file: &FileFact) -> &'static str {
    if file.path.contains("sqlite") || file.path.contains("store") || file.path.contains("storage")
    {
        "persistence-oriented"
    } else if file.path.contains("search")
        || file.path.contains("index")
        || file.path.contains("embed")
    {
        "retrieval or indexing"
    } else if file.path.contains("resolver") {
        "resolution"
    } else if file.path.contains("parser") {
        "parsing"
    } else if file.path.contains("watcher") || file.path.contains("invalidate") {
        "change-detection"
    } else {
        "general"
    }
}

fn side_effects(file: &FileFact, context: &FileEnrichmentContext) -> Vec<String> {
    let mut effects = Vec::new();
    let external = context
        .external_imports
        .iter()
        .map(|import| import.module.as_str())
        .collect::<Vec<_>>();
    if external
        .iter()
        .any(|name| name.contains("rusqlite") || name.contains("sql"))
    {
        effects.push("Likely performs database IO through SQLite-related dependencies.".into());
    }
    if external
        .iter()
        .any(|name| name.contains("fs") || name.contains("path"))
    {
        effects.push(
            "Touches filesystem paths or local file metadata as part of its behavior.".into(),
        );
    }
    if effects.is_empty() {
        effects.push(format!(
            "No side effects were proven statically for {}; inspect source for IO, mutation, or process boundaries.",
            file.path
        ));
    }
    effects
}

fn key_entities(
    file: &FileFact,
    symbol_names: &[String],
    context: &FileEnrichmentContext,
) -> Vec<String> {
    let mut entities = symbol_names.iter().take(10).cloned().collect::<Vec<_>>();
    entities.extend(
        context
            .internal_imports
            .iter()
            .filter_map(|import| import.resolved_path.clone())
            .take(4),
    );
    if entities.is_empty() {
        entities.push(file.name.clone());
    }
    entities
}

fn imports_interpreted(
    file: &FileFact,
    context: &FileEnrichmentContext,
) -> Vec<DependencyInterpretation> {
    let mut interpreted = context
        .internal_imports
        .iter()
        .map(|import| DependencyInterpretation {
            target_id: import
                .resolved_file_id
                .clone()
                .unwrap_or_else(|| import.module.clone()),
            target_path: import
                .resolved_path
                .clone()
                .unwrap_or_else(|| import.module.clone()),
            why: format!(
                "{} imports {} to support behavior in {}.",
                file.path, import.module, file.path
            ),
            dependency_kind: "internal".into(),
        })
        .collect::<Vec<_>>();
    interpreted.extend(context.external_imports.iter().take(6).map(|import| {
        DependencyInterpretation {
            target_id: import.module.clone(),
            target_path: import.module.clone(),
            why: format!(
                "{} imports external dependency {}.",
                file.path, import.module
            ),
            dependency_kind: "external".into(),
        }
    }));
    interpreted
}

fn to_dependent_interpretation(related: &RelatedFileContext) -> DependencyInterpretation {
    DependencyInterpretation {
        target_id: related.file_id.clone(),
        target_path: related.path.clone(),
        why: related.detail.clone(),
        dependency_kind: "dependent".into(),
    }
}

fn blast_radius(file: &FileFact, context: &FileEnrichmentContext) -> Vec<String> {
    let mut notes = Vec::new();
    if !context.imported_by_files.is_empty() {
        notes.push(format!(
            "Changes to {} can affect {} indexed dependents, especially {}.",
            file.path,
            context.imported_by_files.len(),
            context
                .imported_by_files
                .iter()
                .take(3)
                .map(|related| related.path.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !context.internal_imports.is_empty() {
        notes.push(format!(
            "{} also depends on {} internal modules, so interface or data-shape changes can cascade outward and inward.",
            file.path,
            context.internal_imports.len()
        ));
    }
    if notes.is_empty() {
        notes.push(format!(
            "Changes to {} can affect files importing its exported symbols or module behavior.",
            file.path
        ));
    }
    notes
}

fn agent_read_hints(
    file: &FileFact,
    top_symbols: &[String],
    context: &FileEnrichmentContext,
) -> Vec<String> {
    let mut hints = vec![format!(
        "Read {} when investigating behavior around {}.",
        file.path,
        top_symbols
            .first()
            .cloned()
            .unwrap_or_else(|| file.name.clone())
    )];
    if !context.imported_by_files.is_empty() {
        hints.push(
            "Read this before editing its dependents, because exported behavior here can propagate outward."
                .into(),
        );
    }
    if !context.internal_imports.is_empty() {
        hints.push("Read this when tracing behavior across its internal dependencies.".into());
    }
    hints
}

fn is_thin_facade(
    file: &FileFact,
    symbol_names: &[String],
    context: &FileEnrichmentContext,
) -> bool {
    (file.name == "lib.rs" || file.name == "mod.rs")
        && file.line_count <= 20
        && symbol_names.is_empty()
        && context.internal_imports.is_empty()
        && context.external_imports.is_empty()
        && !context.sibling_file_ids.is_empty()
}

fn retrieval_tags(
    file: &FileFact,
    behavior_intents: &[String],
    edit_intents: &[String],
    context: &FileEnrichmentContext,
    is_facade: bool,
) -> Vec<String> {
    let mut tags = vec![
        format!(
            "artifact:{}",
            if is_facade {
                "facade"
            } else {
                "implementation"
            }
        ),
        format!("entity:file"),
        format!("language:{}", tagify(&file.language)),
        format!("path:{}", tagify(&file.path)),
        format!("folder:{}", tagify(&context.parent_folder_id)),
        format!("role:{}", tagify(file_role_family(file))),
        format!(
            "ownership:{}",
            if is_facade {
                "surface"
            } else {
                "behavior-owner"
            }
        ),
    ];
    tags.extend(
        behavior_intents
            .iter()
            .take(8)
            .map(|intent| format!("behavior:{}", tagify(intent))),
    );
    tags.extend(
        edit_intents
            .iter()
            .take(8)
            .map(|intent| format!("edit:{}", tagify(intent))),
    );
    if !context.internal_imports.is_empty() {
        tags.push("dependency:imports-internal".into());
    }
    if !context.imported_by_files.is_empty() {
        tags.push("dependency:upstream".into());
    }
    if !context.external_imports.is_empty() {
        tags.push("dependency:external".into());
    }
    dedupe_take(tags, 32)
}

fn ownership_kind(
    file: &FileFact,
    symbols: &[SymbolFact],
    context: &FileEnrichmentContext,
    is_facade: bool,
) -> FileOwnershipKind {
    if is_facade {
        return FileOwnershipKind::Facade;
    }
    if !symbols.is_empty() || !file.snippets.is_empty() {
        return FileOwnershipKind::Implementation;
    }
    if !context.imported_by_files.is_empty() && !context.sibling_file_ids.is_empty() {
        return FileOwnershipKind::Mixed;
    }
    FileOwnershipKind::Unknown
}

fn delegates_to(context: &FileEnrichmentContext, is_facade: bool) -> Vec<String> {
    if !is_facade {
        return Vec::new();
    }
    dedupe_take(
        context
            .sibling_file_ids
            .iter()
            .filter(|id| !id.ends_with("/lib.rs") && !id.ends_with("/mod.rs"))
            .cloned(),
        8,
    )
}

fn tagify(value: &str) -> String {
    let mut tag = String::new();
    let mut previous_dash = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            tag.push(ch.to_ascii_lowercase());
            previous_dash = false;
        } else if !previous_dash {
            tag.push('-');
            previous_dash = true;
        }
    }
    tag.trim_matches('-').to_string()
}

fn dedupe_take(items: impl IntoIterator<Item = String>, limit: usize) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    items
        .into_iter()
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .filter(|item| seen.insert(item.to_lowercase()))
        .take(limit)
        .collect()
}

fn risk_notes(file: &FileFact, context: &FileEnrichmentContext) -> Vec<String> {
    let mut notes = vec![
        "Heuristic card: run MLX enrichment for deeper behavior, side effect, and dependency interpretation."
            .into(),
    ];
    if context.imported_by_files.len() > 3 {
        notes.push(format!(
            "{} is upstream of several files, so API changes deserve extra care.",
            file.path
        ));
    }
    notes
}

/// Fallback chunk summarizer that produces a grounded one-line summary from
/// the chunk's symbol name, kind, signature, and path — without calling any
/// LLM. Used when MLX is unavailable or as a last resort.
#[derive(Debug, Default, Clone)]
pub struct HeuristicChunkSummarizer;

impl ChunkSummarizer for HeuristicChunkSummarizer {
    fn summarize_chunks(&self, chunks: &[CodeChunkFact]) -> Result<Vec<ChunkSummaryDraft>> {
        Ok(chunks
            .iter()
            .map(|chunk| ChunkSummaryDraft {
                chunk_id: chunk.chunk_id.clone(),
                summary: heuristic_chunk_summary(chunk),
            })
            .collect())
    }
}

fn heuristic_chunk_summary(chunk: &CodeChunkFact) -> String {
    let kind = format!("{:?}", chunk.kind).to_ascii_lowercase();
    let symbol = chunk
        .qualified_name
        .as_deref()
        .or(chunk.symbol.as_deref())
        .unwrap_or("symbol");
    let signature = chunk.signature.trim();
    if signature.is_empty() {
        format!("{} {} defined in {}.", kind, symbol, chunk.path)
    } else {
        format!(
            "{} {} defined in {} with signature: {}",
            kind, symbol, chunk.path, signature
        )
    }
}
