use crate::CodeEnricher;
use anyhow::Result;
use chrono::Utc;
use matryoshka_core_ir::{
    DependencyInterpretation, FileCard, FileEnrichmentContext, FileFact, FolderCard,
    FolderEnrichmentContext, FolderFact, Provenance, RelatedFileContext, RepoCard, SubareaSummary,
    SymbolBehavior, SymbolFact,
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
        let intent_phrases = intent_phrases(file, &symbol_names, context, is_facade);
        let top_symbols = symbol_names.iter().take(4).cloned().collect::<Vec<_>>();
        let behavior = file_role(file, &top_symbols, context, &intent_phrases, is_facade);
        let dependency_summary = dependency_summary(file, context);
        let behavior_intents = behavior_intents(file, &intent_phrases, context, is_facade);
        let edit_intents = edit_intents(file, &symbol_names, context, is_facade);
        let retrieval_tags = retrieval_tags(file, &behavior_intents, &edit_intents, context, is_facade);
        let summary = format!(
            "{} is a {} file in {} with {} top-level symbols and {} imports. {} {}",
            file.path,
            file.language,
            context.parent_folder_id,
            symbol_names.len(),
            file.imports.len(),
            behavior,
            dependency_summary
        );

        Ok(FileCard {
            file_id: file.file_id.clone(),
            summary,
            role: behavior.clone(),
            primary_behaviors: primary_behaviors(
                file,
                &top_symbols,
                context,
                &intent_phrases,
                is_facade,
            ),
            behavior_intents,
            edit_intents,
            retrieval_tags,
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
            search_phrases: search_phrases(
                file,
                &top_symbols,
                context,
                &intent_phrases,
                is_facade,
            ),
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
        context: &FolderEnrichmentContext,
    ) -> Result<FolderCard> {
        let child_names = child_files
            .iter()
            .map(|card| card.file_id.clone())
            .collect::<Vec<_>>();
        let behavior_intents = dedupe_take(
            child_files
                .iter()
                .flat_map(|card| card.behavior_intents.clone())
                .chain(child_files.iter().flat_map(|card| card.primary_behaviors.clone())),
            12,
        );
        let edit_intents = dedupe_take(
            child_files
                .iter()
                .flat_map(|card| card.edit_intents.clone())
                .chain(std::iter::once(format!("edit code under {}", folder.path))),
            12,
        );
        let retrieval_tags = dedupe_take(
            child_files
                .iter()
                .flat_map(|card| card.retrieval_tags.clone())
                .chain(std::iter::once(format!("folder:{}", tagify(&folder.path))))
                .chain(std::iter::once(format!("layer:{}", tagify(&folder.name)))),
            20,
        );
        Ok(FolderCard {
            folder_id: folder.folder_id.clone(),
            summary: format!(
                "{} groups {} direct files and {} direct subfolders. It has {} incoming cross-file dependencies and {} outgoing cross-file dependencies in the current graph.",
                folder.path,
                folder.child_file_ids.len(),
                folder.child_folder_ids.len(),
                context.incoming_dependencies.len(),
                context.outgoing_dependencies.len(),
            ),
            responsibility: format!(
                "Coordinates the behavior represented by {} and acts as the {} layer of the codebase.",
                child_names.join(", "),
                folder.name
            ),
            behavior_intents,
            edit_intents,
            retrieval_tags,
            contains_kinds_of_files: child_files
                .iter()
                .flat_map(|card| card.primary_behaviors.clone())
                .take(8)
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
                .chain(child_names.iter().take(4).cloned())
                .take(8)
                .collect(),
            common_behaviors: child_files
                .iter()
                .flat_map(|card| card.primary_behaviors.clone())
                .take(12)
                .collect(),
            subareas: folder
                .child_folder_ids
                .iter()
                .map(|id| SubareaSummary {
                    id: id.clone(),
                    name: id.clone(),
                    responsibility: "Nested folder awaiting rich enrichment.".into(),
                })
                .collect(),
            agent_guidance: vec![format!(
                "Start at {} when a query mentions files or behaviors grouped under this path or when cross-file dependency flow converges here.",
                folder.path,
            )],
            search_phrases: vec![
                format!("folder responsibility {}", folder.path),
                format!("code under {}", folder.path),
                format!("{} dependency hub", folder.path),
            ],
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
        Ok(RepoCard {
            repo_root: repo_root.into(),
            summary: format!(
                "Repository at {repo_root} contains {} enriched folders and is indexed for agent search/read workflows.",
                folders.len()
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
                    .chain(std::iter::once("artifact:repo-card".into())),
                32,
            ),
            top_level_subsystems: folders
                .iter()
                .filter(|folder| !folder.folder_id.contains('/'))
                .map(|folder| SubareaSummary {
                    id: folder.folder_id.clone(),
                    name: folder.folder_id.clone(),
                    responsibility: folder.responsibility.clone(),
                })
                .collect(),
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
            high_risk_areas: vec![
                "Use dependency and caller/callee edges to inspect blast radius before editing central files."
                    .into(),
            ],
            agent_navigation_hints: vec![
                "Use semantic search first for behavior, then read the returned file card before opening raw source."
                    .into(),
            ],
            search_phrases: vec![
                format!("repository map {repo_root}"),
                "top level subsystem responsibilities".into(),
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

fn file_role(
    file: &FileFact,
    top_symbols: &[String],
    context: &FileEnrichmentContext,
    intent_phrases: &[String],
    is_facade: bool,
) -> String {
    if is_facade {
        let sibling = context
            .sibling_file_ids
            .iter()
            .map(|id| id.as_str())
            .take(3)
            .collect::<Vec<_>>()
            .join(", ");
        if sibling.is_empty() {
            return format!(
                "Acts as a thin crate or module entrypoint in {} and mainly re-exports implementation from sibling modules instead of owning deep behavior directly.",
                context.parent_folder_id
            );
        }
        return format!(
            "Acts as a thin crate or module entrypoint in {} and mainly exposes sibling implementation such as {} instead of owning deep behavior directly.",
            context.parent_folder_id,
            sibling
        );
    }

    if let Some(intent) = intent_phrases.first() {
        return format!(
            "Acts as a {} module in {} and primarily handles {}.",
            file_role_family(file),
            context.parent_folder_id,
            intent
        );
    }

    let joined = if top_symbols.is_empty() {
        file.name.clone()
    } else {
        top_symbols.join(", ")
    };
    format!(
        "Acts as a {} module in {} and owns the logic around {}.",
        file_role_family(file),
        context.parent_folder_id,
        joined
    )
}

fn file_role_family(file: &FileFact) -> &'static str {
    if file.path.contains("sqlite") || file.path.contains("store") || file.path.contains("storage") {
        "persistence-oriented"
    } else if file.path.contains("search") || file.path.contains("index") || file.path.contains("embed") {
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

fn dependency_summary(file: &FileFact, context: &FileEnrichmentContext) -> String {
    match (
        context.internal_imports.len(),
        context.imported_by_files.len(),
    ) {
        (0, 0) => format!(
            "It currently has no resolved internal dependency edges in the code map for {}.",
            file.path
        ),
        (imports, 0) => format!(
            "It depends on {imports} resolved internal modules, but no importing files were resolved yet."
        ),
        (0, dependents) => format!(
            "It is depended on by {dependents} files in the index, even though it does not resolve internal imports itself."
        ),
        (imports, dependents) => format!(
            "It depends on {imports} resolved internal modules and is used by {dependents} files in the current index."
        ),
    }
}

fn primary_behaviors(
    file: &FileFact,
    top_symbols: &[String],
    context: &FileEnrichmentContext,
    intent_phrases: &[String],
    is_facade: bool,
) -> Vec<String> {
    let mut behaviors = Vec::new();
    if is_facade {
        behaviors.push(format!(
            "Provides the {} entrypoint or re-export surface for sibling implementation modules.",
            context.parent_folder_id
        ));
        if !context.sibling_file_ids.is_empty() {
            behaviors.push(format!(
                "Points readers toward sibling implementation files such as {}.",
                context
                    .sibling_file_ids
                    .iter()
                    .take(3)
                    .map(|id| id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    if let Some(first_symbol) = top_symbols.first() {
        behaviors.push(format!(
            "Defines {} through {}.",
            file.name, first_symbol
        ));
    }
    behaviors.extend(
        intent_phrases
            .iter()
            .take(3)
            .map(|intent| format!("Handles {}.", intent)),
    );
    if !context.internal_imports.is_empty() {
        behaviors.push(format!(
            "Coordinates {} resolved internal dependencies.",
            context.internal_imports.len()
        ));
    }
    if !context.imported_by_files.is_empty() {
        behaviors.push(format!(
            "Serves {} downstream files that depend on its exports or module behavior.",
            context.imported_by_files.len()
        ));
    }
    if behaviors.is_empty() {
        behaviors.push(format!(
            "Keeps {} source logic and module-level declarations available to the codebase.",
            file.path
        ));
    }
    behaviors
}

fn side_effects(file: &FileFact, context: &FileEnrichmentContext) -> Vec<String> {
    let mut effects = Vec::new();
    let external = context
        .external_imports
        .iter()
        .map(|import| import.module.as_str())
        .collect::<Vec<_>>();
    if external.iter().any(|name| name.contains("rusqlite") || name.contains("sql")) {
        effects.push("Likely performs database IO through SQLite-related dependencies.".into());
    }
    if external.iter().any(|name| name.contains("fs") || name.contains("path")) {
        effects.push("Touches filesystem paths or local file metadata as part of its behavior.".into());
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
    interpreted.extend(
        context
            .external_imports
            .iter()
            .take(6)
            .map(|import| DependencyInterpretation {
                target_id: import.module.clone(),
                target_path: import.module.clone(),
                why: format!(
                    "{} imports external dependency {}.",
                    file.path, import.module
                ),
                dependency_kind: "external".into(),
            }),
    );
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
        top_symbols.first().cloned().unwrap_or_else(|| file.name.clone())
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

fn search_phrases(
    file: &FileFact,
    top_symbols: &[String],
    context: &FileEnrichmentContext,
    intent_phrases: &[String],
    is_facade: bool,
) -> Vec<String> {
    let mut phrases = vec![
        format!("behavior implemented in {}", file.path),
        format!("{} {}", file.language, file.name),
    ];
    if is_facade {
        phrases.push(format!("{} crate entrypoint", context.parent_folder_id));
        phrases.push(format!("{} module exports", context.parent_folder_id));
        phrases.push(format!("reexports from {}", file.path));
    } else {
        phrases.extend(intent_phrases.iter().cloned());
    }
    if !top_symbols.is_empty() {
        phrases.push(format!("{} {}", file.language, top_symbols.join(" ")));
    }
    if !context.internal_imports.is_empty() {
        phrases.push(format!("{} internal dependency flow", file.path));
    }
    if !context.imported_by_files.is_empty() {
        phrases.push(format!("what depends on {}", file.path));
    }
    phrases
}

fn intent_phrases(
    file: &FileFact,
    symbol_names: &[String],
    context: &FileEnrichmentContext,
    is_facade: bool,
) -> Vec<String> {
    if is_facade {
        return Vec::new();
    }

    let lowered_symbols = symbol_names
        .iter()
        .map(|name| name.to_lowercase())
        .collect::<Vec<_>>();
    let lowered_imports = context
        .internal_imports
        .iter()
        .map(|import| import.module.to_lowercase())
        .chain(
            context
                .external_imports
                .iter()
                .map(|import| import.module.to_lowercase()),
        )
        .collect::<Vec<_>>();
    let joined_symbols = lowered_symbols.join(" ");
    let joined_imports = lowered_imports.join(" ");
    let path = file.path.to_lowercase();
    let mut intents = Vec::new();

    if path.contains("resolver")
        || joined_symbols.contains("resolve_import")
        || joined_symbols.contains("import_module_paths")
        || joined_symbols.contains("workspace_crate_candidates")
    {
        intents.push("import resolution and repository dependency graph construction".into());
        intents.push("mapping Rust, Python, or workspace-crate imports onto concrete repository files".into());
    }
    if path.contains("parser")
        || joined_symbols.contains("parse_rust_import")
        || joined_symbols.contains("parse_python_import")
        || joined_symbols.contains("parse_symbols")
    {
        intents.push("source parsing, import extraction, and symbol extraction".into());
    }
    if joined_symbols.contains("parse_rust_symbols")
        || joined_symbols.contains("find_rust_block_end")
        || joined_symbols.contains("update_brace_depth")
    {
        intents.push("Rust symbol extraction, impl-method qualification, and block boundary detection".into());
    }
    if path.contains("search") || joined_symbols.contains("score_record") {
        intents.push("hybrid search ranking across embeddings, lexical hits, and graph-aware boosts".into());
    }
    if path.contains("indexer") || joined_symbols.contains("embed_records") {
        intents.push("building semantic records, cards, and embeddings for the indexed code map".into());
    }
    if path.contains("watcher") || path.contains("invalidation") {
        intents.push("change detection, invalidation, and incremental refresh planning".into());
    }
    if path.contains("store") || path.contains("sqlite") || joined_imports.contains("rusqlite") {
        intents.push("persisting facts, cards, and semantic records in SQLite".into());
    }
    if joined_symbols.contains("read")
        || joined_symbols.contains("load_file_card")
        || joined_symbols.contains("load_folder_card")
    {
        intents.push("assembling read-oriented file and folder context for an agent".into());
    }
    if path.contains("embed") || joined_imports.contains("embeddings") {
        intents.push("embedding generation and vector preparation for semantic retrieval".into());
    }

    intents.dedup();
    intents
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

fn behavior_intents(
    file: &FileFact,
    intent_phrases: &[String],
    context: &FileEnrichmentContext,
    is_facade: bool,
) -> Vec<String> {
    let mut intents = Vec::new();
    if is_facade {
        intents.push(format!("Expose {} module entrypoint and re-export surface", context.parent_folder_id));
    }
    intents.extend(intent_phrases.iter().cloned());
    if !context.imported_by_files.is_empty() {
        intents.push("Provide behavior used by downstream dependents".into());
    }
    if !context.internal_imports.is_empty() {
        intents.push("Coordinate internal dependencies used by this file".into());
    }
    if intents.is_empty() {
        intents.push(format!("Maintain {} source behavior", file.path));
    }
    dedupe_take(intents, 10)
}

fn edit_intents(
    file: &FileFact,
    symbol_names: &[String],
    context: &FileEnrichmentContext,
    is_facade: bool,
) -> Vec<String> {
    let mut intents = Vec::new();
    if is_facade {
        intents.push(format!("change public exports for {}", context.parent_folder_id));
        intents.push(format!("route readers from {} to implementation modules", file.path));
    }
    for symbol in symbol_names.iter().take(5) {
        intents.push(format!("change behavior around {symbol}"));
    }
    if file.path.contains("sqlite") || file.path.contains("store") {
        intents.push("change SQLite persistence behavior".into());
        intents.push("debug database loading or schema issues".into());
    }
    if file.path.contains("resolver") {
        intents.push("change import resolution behavior".into());
        intents.push("debug dependency graph construction".into());
    }
    if file.path.contains("parser") {
        intents.push("change source parsing or symbol extraction".into());
        intents.push("debug missing files or symbols in the index".into());
    }
    if file.path.contains("search") {
        intents.push("change semantic search ranking behavior".into());
    }
    if file.path.contains("indexer") {
        intents.push("change repository prewarm or semantic record generation".into());
    }
    if !context.imported_by_files.is_empty() {
        intents.push("assess blast radius before changing exported behavior".into());
    }
    dedupe_take(intents, 12)
}

fn retrieval_tags(
    file: &FileFact,
    behavior_intents: &[String],
    edit_intents: &[String],
    context: &FileEnrichmentContext,
    is_facade: bool,
) -> Vec<String> {
    let mut tags = vec![
        format!("artifact:{}", if is_facade { "facade" } else { "implementation" }),
        format!("entity:file"),
        format!("language:{}", tagify(&file.language)),
        format!("path:{}", tagify(&file.path)),
        format!("folder:{}", tagify(&context.parent_folder_id)),
        format!("role:{}", tagify(file_role_family(file))),
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
