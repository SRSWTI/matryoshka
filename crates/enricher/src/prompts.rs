use matryoshka_core_ir::{
    FileEnrichmentContext, FileFact, FolderEnrichmentContext, FolderFact, SymbolFact,
};
use serde_json::{Value, json};

pub const ENRICHMENT_MODEL: &str = "MercuriusDream--Qwen3.5-4B-MLX-mxfp8";

pub fn file_core_enrichment_prompt(
    file: &FileFact,
    symbols: &[SymbolFact],
    context: &FileEnrichmentContext,
) -> Value {
    json!({
        "task": "Create the core behavioral understanding for a code-intelligence FileCard. Focus on what the file does, why it exists, which behaviors it owns, its side effects, dynamic behavior intents, and the most important symbols. Do not include dependency interpretations, blast radius, edit intents, tags, or search phrases in this pass.",
        "output_rules": [
            "Return strict JSON only.",
            "Prefer quality over exhaustiveness.",
            "Each string should usually be one sentence.",
            "Keep the summary to 120-220 words.",
            "Keep the role to 2-4 sentences.",
            "For arrays, include only the most important items.",
            "Use 4-8 primary_behaviors.",
            "Use 4-10 behavior_intents as concise behavioral responsibilities or conceptual query targets.",
            "Use 0-5 side_effects, and say uncertain when side effects are unclear.",
            "Use 0-6 key_entities.",
            "Use 0-4 external_systems and only include truly external packages, services, storage, network, or OS concerns.",
            "Use 0-8 important_symbols and focus on symbols that matter for navigation or editing."
        ],
        "required_json_shape": {
            "summary": "rich paragraph",
            "role": "why this file exists",
            "primary_behaviors": ["behavior strings"],
            "behavior_intents": ["dynamic behavioral responsibilities and query intents"],
            "side_effects": ["side effects or explicit uncertainty"],
            "key_entities": ["important domain/API/config entities"],
            "external_systems": ["external packages, services, filesystems, networks, DBs"],
            "important_symbols": [{"symbol_id": "...", "name": "...", "role": "...", "behavior": "..."}]
        },
        "file_context": compact_file_context(file),
        "graph_context": compact_file_graph_context(context),
        "symbols": compact_symbols(symbols),
    })
}

pub fn file_dependencies_enrichment_prompt(
    file: &FileFact,
    symbols: &[SymbolFact],
    context: &FileEnrichmentContext,
    core_summary: &Value,
) -> Value {
    json!({
        "task": "Create the dependency and blast-radius interpretation for a code-intelligence FileCard. Focus on the most meaningful imports, the most meaningful dependents or callers implied by the available context, and what edits would likely affect. Do not repeat a large summary of the file.",
        "output_rules": [
            "Return strict JSON only.",
            "Prefer quality over exhaustiveness.",
            "Use 0-6 imports_interpreted and only include dependencies that materially explain behavior.",
            "Use 0-6 used_by_interpreted and only include dependents that materially affect blast radius.",
            "Use 2-5 blast_radius items.",
            "If dependent information is weak, say so clearly instead of inventing certainty."
        ],
        "required_json_shape": {
            "imports_interpreted": [{"target_id": "...", "target_path": "...", "why": "...", "dependency_kind": "..."}],
            "used_by_interpreted": [{"target_id": "...", "target_path": "...", "why": "...", "dependency_kind": "..."}],
            "blast_radius": ["what changes can affect"]
        },
        "file_context": compact_file_context(file),
        "graph_context": compact_file_graph_context(context),
        "symbols": compact_symbols(symbols),
        "core_summary": core_summary,
    })
}

pub fn file_navigation_enrichment_prompt(
    file: &FileFact,
    symbols: &[SymbolFact],
    context: &FileEnrichmentContext,
    core_summary: &Value,
    dependency_summary: &Value,
) -> Value {
    json!({
        "task": "Create the navigation and retrieval layer for a code-intelligence FileCard. Focus on when an agent should read this file, what edit intents should route here, decomposed retrieval tags, search phrasings, and editing risks. Do not restate the full behavior summary.",
        "output_rules": [
            "Return strict JSON only.",
            "Prefer short behavior-oriented phrases.",
            "Use 2-5 agent_read_hints.",
            "Use 4-10 edit_intents as concrete editing/debugging/refactoring tasks.",
            "Use 8-18 retrieval_tags as compact decomposed tags like behavior:import-resolution, edit:change-parser, artifact:implementation, dependency:upstream.",
            "Use 6-12 search_phrases.",
            "Use 0-4 risk_notes.",
            "Make search phrases varied and likely to match real coding-agent queries.",
            "Tags should be lowercase, hyphenated where useful, and should not invent facts not present in the graph or source context."
        ],
        "required_json_shape": {
            "agent_read_hints": ["when to read this file"],
            "edit_intents": ["editing/debugging tasks that should route here"],
            "retrieval_tags": ["compact decomposed retrieval tags"],
            "search_phrases": ["natural language search phrases"],
            "risk_notes": ["editing risks"]
        },
        "file_context": compact_file_context(file),
        "graph_context": compact_file_graph_context(context),
        "symbols": compact_symbols(symbols),
        "core_summary": core_summary,
        "dependency_summary": dependency_summary,
    })
}

pub fn folder_core_enrichment_prompt(
    folder: &FolderFact,
    child_file_cards: &[Value],
    context: &FolderEnrichmentContext,
) -> Value {
    json!({
        "task": "Create the core responsibility map for a FolderCard. Explain what responsibility this folder owns, what kinds of child files it contains, how they collaborate, dynamic behavior intents, and the most important common behaviors. Do not include guidance/search phrases in this pass.",
        "output_rules": [
            "Return strict JSON only.",
            "Prefer quality over exhaustiveness.",
            "Keep the summary to 120-220 words.",
            "Keep responsibility to 2-4 sentences.",
            "Use 4-10 behavior_intents.",
            "Use 3-6 contains_kinds_of_files items.",
            "Use 3-6 common_behaviors items.",
            "Use 0-6 subareas and only when they are meaningful."
        ],
        "required_json_shape": {
            "summary": "rich paragraph",
            "responsibility": "main ownership of this folder",
            "behavior_intents": ["dynamic folder-level responsibilities and query intents"],
            "contains_kinds_of_files": ["kinds of files and behaviors"],
            "common_behaviors": ["shared behaviors"],
            "subareas": [{"id": "...", "name": "...", "responsibility": "..."}]
        },
        "folder_context": compact_folder_context(folder),
        "graph_context": compact_folder_graph_context(context),
        "child_file_cards": compact_child_file_cards(child_file_cards),
    })
}

pub fn folder_navigation_enrichment_prompt(
    folder: &FolderFact,
    child_file_cards: &[Value],
    context: &FolderEnrichmentContext,
    core_summary: &Value,
) -> Value {
    json!({
        "task": "Create the dependency interpretation and navigation layer for a FolderCard. Focus on cross-folder dependencies, key entrypoints, edit intents, decomposed retrieval tags, when an agent should start here, and search phrases that should retrieve this folder.",
        "output_rules": [
            "Return strict JSON only.",
            "Prefer quality over exhaustiveness.",
            "Use 2-5 incoming_dependencies_meaning items.",
            "Use 2-5 outgoing_dependencies_meaning items.",
            "Use 2-6 key_entrypoints items.",
            "Use 4-10 edit_intents.",
            "Use 8-18 retrieval_tags as compact decomposed tags.",
            "Use 2-5 agent_guidance items.",
            "Use 6-10 search_phrases."
        ],
        "required_json_shape": {
            "incoming_dependencies_meaning": ["who depends on this folder and why"],
            "outgoing_dependencies_meaning": ["what this folder depends on and why"],
            "key_entrypoints": ["important child files/symbols"],
            "edit_intents": ["editing/debugging tasks that should start here"],
            "retrieval_tags": ["compact decomposed retrieval tags"],
            "agent_guidance": ["when to start here"],
            "search_phrases": ["natural language search phrases"]
        },
        "folder_context": compact_folder_context(folder),
        "graph_context": compact_folder_graph_context(context),
        "child_file_cards": compact_child_file_cards(child_file_cards),
        "core_summary": core_summary,
    })
}

fn compact_file_context(file: &FileFact) -> Value {
    let internal_imports: Vec<Value> = file
        .imports
        .iter()
        .filter(|import| import.is_internal)
        .take(10)
        .map(|import| {
            json!({
                "module": import.module,
                "names": import.names,
                "resolved_file_id": import.resolved_file_id,
            })
        })
        .collect();
    let external_imports: Vec<Value> = file
        .imports
        .iter()
        .filter(|import| !import.is_internal)
        .take(8)
        .map(|import| {
            json!({
                "module": import.module,
                "names": import.names,
            })
        })
        .collect();
    let snippets: Vec<Value> = file
        .snippets
        .iter()
        .take(4)
        .map(|snippet| {
            json!({
                "title": snippet.title,
                "start_line": snippet.start_line,
                "end_line": snippet.end_line,
                "text_excerpt": truncate(&snippet.text, 500),
            })
        })
        .collect();

    json!({
        "file_id": file.file_id,
        "path": file.path,
        "name": file.name,
        "language": file.language,
        "parent_folder_id": file.parent_folder_id,
        "line_count": file.line_count,
        "internal_import_count": file.imports.iter().filter(|import| import.is_internal).count(),
        "external_import_count": file.imports.iter().filter(|import| !import.is_internal).count(),
        "internal_imports_sample": internal_imports,
        "external_imports_sample": external_imports,
        "snippets_sample": snippets,
    })
}

fn compact_symbols(symbols: &[SymbolFact]) -> Vec<Value> {
    symbols
        .iter()
        .take(12)
        .map(|symbol| {
            json!({
                "symbol_id": symbol.symbol_id,
                "name": symbol.name,
                "qualified_name": symbol.qualified_name,
                "kind": symbol.kind,
                "signature": truncate(&symbol.signature, 220),
                "start_line": symbol.start_line,
                "end_line": symbol.end_line,
            })
        })
        .collect()
}

fn compact_file_graph_context(context: &FileEnrichmentContext) -> Value {
    json!({
        "parent_folder_id": context.parent_folder_id,
        "sibling_file_ids": context.sibling_file_ids.iter().take(8).collect::<Vec<_>>(),
        "internal_imports": context.internal_imports.iter().take(10).collect::<Vec<_>>(),
        "external_imports": context.external_imports.iter().take(8).collect::<Vec<_>>(),
        "imported_by_files": context.imported_by_files.iter().take(8).collect::<Vec<_>>(),
    })
}

fn compact_folder_context(folder: &FolderFact) -> Value {
    json!({
        "folder_id": folder.folder_id,
        "path": folder.path,
        "name": folder.name,
        "parent_folder_id": folder.parent_folder_id,
        "child_file_ids": folder.child_file_ids.iter().take(24).collect::<Vec<_>>(),
        "child_folder_ids": folder.child_folder_ids.iter().take(24).collect::<Vec<_>>(),
        "child_file_count": folder.child_file_ids.len(),
        "child_folder_count": folder.child_folder_ids.len(),
    })
}

fn compact_folder_graph_context(context: &FolderEnrichmentContext) -> Value {
    json!({
        "parent_folder_id": context.parent_folder_id,
        "incoming_dependencies": context.incoming_dependencies.iter().take(10).collect::<Vec<_>>(),
        "outgoing_dependencies": context.outgoing_dependencies.iter().take(10).collect::<Vec<_>>(),
        "representative_child_files": context.representative_child_files.iter().take(8).collect::<Vec<_>>(),
    })
}

fn compact_child_file_cards(child_file_cards: &[Value]) -> Vec<Value> {
    child_file_cards
        .iter()
        .take(12)
        .map(|card| {
            json!({
                "file_id": card.get("file_id").cloned().unwrap_or(Value::Null),
                "summary": truncate_value_str(card.get("summary"), 320),
                "role": truncate_value_str(card.get("role"), 220),
                "primary_behaviors": truncate_value_string_array(card.get("primary_behaviors"), 4, 140),
                "behavior_intents": truncate_value_string_array(card.get("behavior_intents"), 5, 100),
                "edit_intents": truncate_value_string_array(card.get("edit_intents"), 5, 100),
                "retrieval_tags": truncate_value_string_array(card.get("retrieval_tags"), 8, 80),
                "ownership_kind": card.get("ownership_kind").cloned().unwrap_or(Value::Null),
                "owns_behaviors": truncate_value_string_array(card.get("owns_behaviors"), 5, 100),
                "delegates_to": truncate_value_string_array(card.get("delegates_to"), 5, 120),
                "blast_radius": truncate_value_string_array(card.get("blast_radius"), 3, 140),
                "agent_read_hints": truncate_value_string_array(card.get("agent_read_hints"), 3, 140),
                "search_phrases": truncate_value_string_array(card.get("search_phrases"), 4, 80),
            })
        })
        .collect()
}

fn truncate(text: &str, max_chars: usize) -> String {
    let mut iter = text.chars();
    let truncated: String = iter.by_ref().take(max_chars).collect();
    if iter.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

fn truncate_value_str(value: Option<&Value>, max_chars: usize) -> Value {
    value
        .and_then(Value::as_str)
        .map(|text| Value::String(truncate(text, max_chars)))
        .unwrap_or(Value::Null)
}

fn truncate_value_string_array(value: Option<&Value>, max_items: usize, max_chars: usize) -> Value {
    Value::Array(
        value
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .take(max_items)
                    .map(|item| Value::String(truncate(item, max_chars)))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
    )
}
