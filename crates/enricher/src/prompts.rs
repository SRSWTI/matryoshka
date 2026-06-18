use matryoshka_core_ir::{
    FileEnrichmentContext, FileFact, FolderCard, FolderEnrichmentContext, FolderFact, SymbolFact,
};
use serde_json::{Value, json};

pub const ENRICHMENT_MODEL: &str = "MercuriusDream--Qwen3.5-4B-MLX-mxfp8";

pub fn file_summary_enrichment_prompt(
    file: &FileFact,
    symbols: &[SymbolFact],
    context: &FileEnrichmentContext,
) -> Value {
    json!({
        "task": "Summarize this code file for a code-intelligence index.",
        "rules": [
            "Write 3-5 short sentences.",
            "Cover only: the purpose and key behavior/data flow.",
            "Avoid generic filler.",
            "Use exact names from the code when useful.",
            "Return strict JSON with exactly one field: summary."
        ],
        "required_json_shape": {
            "summary": "concise grounded file summary"
        },
        "file_context": compact_file_context(file),
        "graph_context": compact_file_graph_context(context),
        "symbols": compact_symbols(symbols),
    })
}

pub fn folder_summary_enrichment_prompt(
    folder: &FolderFact,
    child_file_cards: &[Value],
    child_folder_cards: &[Value],
    context: &FolderEnrichmentContext,
) -> Value {
    json!({
        "task": "Summarize this code folder for a code-intelligence index.",
        "rules": [
            "Write 3-5 short sentences.",
            "Cover only: the folder purpose and key behavior/data flow across direct child files and folders.",
            "Avoid generic filler.",
            "Use exact file, folder, or symbol names when useful.",
            "Return strict JSON with exactly one field: summary."
        ],
        "required_json_shape": {
            "summary": "concise grounded folder summary"
        },
        "folder_context": compact_folder_context(folder),
        "graph_context": compact_folder_graph_context(context),
        "child_file_cards": compact_child_file_cards(child_file_cards),
        "child_folder_cards": compact_child_folder_cards(child_folder_cards),
    })
}

pub fn repo_summary_enrichment_prompt(repo_root: &str, folders: &[FolderCard]) -> Value {
    json!({
        "task": "Summarize this repository for a code-intelligence index.",
        "rules": [
            "Write 3-5 short sentences.",
            "Cover only: the repository purpose and key behavior/data flow across top-level folders.",
            "Avoid generic filler.",
            "Use exact folder names when useful.",
            "Return strict JSON with exactly one field: summary."
        ],
        "required_json_shape": {
            "summary": "concise grounded repository summary"
        },
        "repo_root": repo_root,
        "folder_cards": compact_repo_folder_cards(folders),
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
                "ownership_kind": card.get("ownership_kind").cloned().unwrap_or(Value::Null),
                "delegates_to": truncate_value_string_array(card.get("delegates_to"), 5, 120),
            })
        })
        .collect()
}

fn compact_child_folder_cards(child_folder_cards: &[Value]) -> Vec<Value> {
    child_folder_cards
        .iter()
        .take(12)
        .map(|card| {
            json!({
                "folder_id": card.get("folder_id").cloned().unwrap_or(Value::Null),
                "summary": truncate_value_str(card.get("summary"), 360),
                "responsibility": truncate_value_str(card.get("responsibility"), 260),
                "key_entrypoints": truncate_value_string_array(card.get("key_entrypoints"), 5, 120),
            })
        })
        .collect()
}

fn compact_repo_folder_cards(folder_cards: &[FolderCard]) -> Vec<Value> {
    folder_cards
        .iter()
        .take(24)
        .map(|card| {
            json!({
                "folder_id": card.folder_id,
                "summary": truncate(&card.summary, 360),
                "child_focus": truncate(&card.responsibility, 220),
                "entrypoints": card.key_entrypoints.iter().take(5).collect::<Vec<_>>(),
                "subareas": card.subareas.iter().take(5).collect::<Vec<_>>(),
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
