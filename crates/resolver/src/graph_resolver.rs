use chrono::Utc;
use matryoshka_core_ir::{
    EdgeFact, EdgeKind, FolderFact, RepositorySnapshot, SemanticEntityType, SemanticRecord,
};
use matryoshka_parser::ParsedRepository;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub struct GraphResolver;

impl GraphResolver {
    pub fn resolve(parsed: ParsedRepository) -> RepositorySnapshot {
        let mut files = parsed.files;
        let folders = build_folders(&files);
        let mut edges = Vec::new();
        let file_ids: BTreeSet<String> = files.iter().map(|file| file.file_id.clone()).collect();
        let module_aliases = module_aliases(&parsed.repo_root, &files);

        for folder in &folders {
            for file_id in &folder.child_file_ids {
                edges.push(edge(
                    &folder.folder_id,
                    file_id,
                    EdgeKind::Contains,
                    1,
                    "folder contains file",
                ));
            }
            for child_folder_id in &folder.child_folder_ids {
                edges.push(edge(
                    &folder.folder_id,
                    child_folder_id,
                    EdgeKind::Contains,
                    1,
                    "folder contains folder",
                ));
            }
        }

        for file in &mut files {
            for import in &mut file.imports {
                let target = resolve_import(&import.module, &file.file_id, &file_ids, &module_aliases);
                import.resolved_file_id = target.clone();
                import.is_internal = target.is_some() || import.is_internal;
                if let Some(target_id) = target {
                    edges.push(edge(
                        &file.file_id,
                        &target_id,
                        EdgeKind::Imports,
                        3,
                        &format!("imports {}", import.module),
                    ));
                    edges.push(edge(
                        &file.file_id,
                        &target_id,
                        EdgeKind::DependsOn,
                        2,
                        "resolved internal dependency",
                    ));
                }
            }
        }

        let mut semantic_records = Vec::new();
        for file in &files {
            semantic_records.push(SemanticRecord {
                record_id: format!("semantic:file:{}", file.file_id),
                entity_id: file.file_id.clone(),
                entity_type: SemanticEntityType::File,
                title: format!("File {}", file.path),
                content: file_record_content(file),
                path: file.path.clone(),
                source_hash: file.source_hash.clone(),
                embedding: None,
                metadata: BTreeMap::from([("kind".into(), json!("file_fact"))]),
            });
            for snippet in &file.snippets {
                semantic_records.push(SemanticRecord {
                    record_id: format!("semantic:snippet:{}", snippet.snippet_id),
                    entity_id: snippet.snippet_id.clone(),
                    entity_type: SemanticEntityType::Snippet,
                    title: format!("Snippet {} in {}", snippet.title, file.path),
                    content: snippet.text.clone(),
                    path: file.path.clone(),
                    source_hash: file.source_hash.clone(),
                    embedding: None,
                    metadata: BTreeMap::from([
                        ("file_id".into(), json!(file.file_id)),
                        ("start_line".into(), json!(snippet.start_line)),
                    ]),
                });
            }
        }
        for symbol in &parsed.symbols {
            semantic_records.push(SemanticRecord {
                record_id: format!("semantic:symbol:{}", symbol.symbol_id),
                entity_id: symbol.symbol_id.clone(),
                entity_type: SemanticEntityType::Symbol,
                title: format!("Symbol {} in {}", symbol.qualified_name, symbol.path),
                content: format!(
                    "symbol: {}\nkind: {:?}\nsignature: {}\nfile: {}",
                    symbol.qualified_name, symbol.kind, symbol.signature, symbol.path
                ),
                path: symbol.path.clone(),
                source_hash: files
                    .iter()
                    .find(|file| file.file_id == symbol.file_id)
                    .map(|file| file.source_hash.clone())
                    .unwrap_or_default(),
                embedding: None,
                metadata: BTreeMap::from([("kind".into(), json!("symbol_fact"))]),
            });
        }

        RepositorySnapshot {
            repo_root: parsed.repo_root.to_string_lossy().to_string(),
            indexed_at: Utc::now(),
            files,
            folders,
            symbols: parsed.symbols,
            edges,
            semantic_records,
        }
    }
}

fn build_folders(files: &[matryoshka_core_ir::FileFact]) -> Vec<FolderFact> {
    let mut folder_ids: BTreeSet<String> = BTreeSet::from(["repo".to_string()]);
    for file in files {
        let mut current = Path::new(&file.path).parent();
        while let Some(path) = current {
            let id = path.to_string_lossy().replace('\\', "/");
            if !id.is_empty() {
                folder_ids.insert(id);
            }
            current = path.parent();
        }
    }

    folder_ids
        .iter()
        .map(|folder_id| {
            let child_file_ids = files
                .iter()
                .filter(|file| &file.parent_folder_id == folder_id)
                .map(|file| file.file_id.clone())
                .collect::<Vec<_>>();
            let child_folder_ids = folder_ids
                .iter()
                .filter(|candidate| parent_folder_id(candidate) == Some(folder_id.clone()))
                .cloned()
                .collect::<Vec<_>>();
            FolderFact {
                folder_id: folder_id.clone(),
                path: folder_id.clone(),
                name: if folder_id == "repo" {
                    "repo".into()
                } else {
                    Path::new(folder_id)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or(folder_id)
                        .into()
                },
                parent_folder_id: parent_folder_id(folder_id),
                child_file_ids,
                child_folder_ids,
            }
        })
        .collect()
}

fn parent_folder_id(path: &str) -> Option<String> {
    if path == "repo" {
        return None;
    }
    Path::new(path)
        .parent()
        .and_then(|parent| parent.to_str())
        .filter(|parent| !parent.is_empty())
        .map(|parent| parent.replace('\\', "/"))
        .or_else(|| Some("repo".into()))
}

fn resolve_import(
    module: &str,
    importer_file_id: &str,
    file_ids: &BTreeSet<String>,
    module_aliases: &BTreeSet<String>,
) -> Option<String> {
    for module_path in import_module_paths(module, importer_file_id, module_aliases) {
        for candidate in file_candidates(&module_path) {
            if file_ids.contains(&candidate) {
                return Some(candidate);
            }
        }
        for candidate in workspace_crate_candidates(&module_path) {
            if file_ids.contains(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

fn file_candidates(module_path: &str) -> Vec<String> {
    if module_path.is_empty() {
        return vec!["__init__.py".into(), "mod.rs".into()];
    }
    vec![
        format!("{module_path}.py"),
        format!("{module_path}.ts"),
        format!("{module_path}.tsx"),
        format!("{module_path}.rs"),
        format!("{module_path}/__init__.py"),
        format!("{module_path}/mod.rs"),
        format!("{module_path}/index.ts"),
    ]
}

fn workspace_crate_candidates(module_path: &str) -> Vec<String> {
    let mut parts = module_path.split('/').filter(|segment| !segment.is_empty());
    let Some(crate_alias) = parts.next() else {
        return Vec::new();
    };
    let Some(crate_suffix) = crate_alias.strip_prefix("matryoshka_") else {
        return Vec::new();
    };
    let crate_dir = crate_suffix.replace('_', "-");
    let rest = parts.collect::<Vec<_>>().join("/");
    let mut candidates = vec![format!("{crate_dir}/src/lib.rs")];
    if !rest.is_empty() {
        candidates.push(format!("{crate_dir}/src/{rest}.rs"));
        candidates.push(format!("{crate_dir}/src/{rest}/mod.rs"));
    }
    candidates
}

fn import_module_paths(
    module: &str,
    importer_file_id: &str,
    module_aliases: &BTreeSet<String>,
) -> Vec<String> {
    let mut paths = Vec::new();

    if module.starts_with('.') {
        let dot_count = module.chars().take_while(|ch| *ch == '.').count();
        let bare = module.trim_start_matches('.');
        let importer_parent = Path::new(importer_file_id)
            .parent()
            .and_then(|path| path.to_str())
            .unwrap_or_default();
        let mut segments = importer_parent
            .split('/')
            .filter(|segment| !segment.is_empty())
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let ascents = dot_count.saturating_sub(1);
        for _ in 0..ascents {
            segments.pop();
        }
        if !bare.is_empty() {
            segments.extend(
                bare.split('.')
                    .filter(|segment| !segment.is_empty())
                    .map(ToString::to_string),
            );
        }
        paths.push(segments.join("/"));
    }

    let normalized = module
        .trim_start_matches("./")
        .trim_start_matches("../")
        .trim_start_matches("crate.")
        .trim_start_matches("self.")
        .trim_start_matches("super.");

    if !normalized.is_empty() {
        paths.push(normalized.replace('.', "/"));
        for alias in module_aliases {
            if normalized == alias {
                paths.push(String::new());
                paths.push(alias.replace('.', "/"));
            }
            if let Some(stripped) = normalized.strip_prefix(&format!("{alias}.")) {
                paths.push(stripped.replace('.', "/"));
            }
        }
    }

    let mut unique = BTreeSet::new();
    paths
        .into_iter()
        .filter(|path| unique.insert(path.clone()))
        .collect()
}

fn module_aliases(repo_root: &Path, files: &[matryoshka_core_ir::FileFact]) -> BTreeSet<String> {
    let mut aliases = BTreeSet::new();
    if let Some(name) = repo_root.file_name().and_then(|name| name.to_str()) {
        aliases.insert(name.to_string());
    }
    for file in files {
        if file.name == "__init__.py" || file.name == "mod.rs" {
            if let Some(parent_name) = Path::new(&file.path)
                .parent()
                .and_then(|path| path.file_name())
                .and_then(|name| name.to_str())
            {
                aliases.insert(parent_name.to_string());
            }
        }
    }
    aliases
}

fn edge(source_id: &str, target_id: &str, kind: EdgeKind, weight: u32, detail: &str) -> EdgeFact {
    EdgeFact {
        edge_id: format!("{kind:?}:{source_id}->{target_id}:{detail}"),
        source_id: source_id.into(),
        target_id: target_id.into(),
        kind,
        weight,
        detail: detail.into(),
    }
}

fn file_record_content(file: &matryoshka_core_ir::FileFact) -> String {
    let imports = file
        .imports
        .iter()
        .map(|import| import.module.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let snippets = file
        .snippets
        .iter()
        .map(|snippet| snippet.title.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "path: {}\nlanguage: {}\nimports: {}\nimportant snippets: {}\nlines: {}",
        file.path, file.language, imports, snippets, file.line_count
    )
}

#[cfg(test)]
mod tests {
    use super::{import_module_paths, module_aliases, resolve_import};
    use matryoshka_core_ir::FileFact;
    use std::collections::BTreeSet;
    use std::path::Path;

    #[test]
    fn resolves_repo_package_prefixed_python_imports() {
        let file_ids = BTreeSet::from([
            "__init__.py".to_string(),
            "ast_extractor.py".to_string(),
            "pipeline.py".to_string(),
        ]);
        let files = vec![
            fake_file("__init__.py", "__init__.py"),
            fake_file("ast_extractor.py", "ast_extractor.py"),
            fake_file("pipeline.py", "pipeline.py"),
        ];
        let aliases = module_aliases(Path::new("/tmp/matryoshka"), &files);

        let resolved = resolve_import(
            "matryoshka.ast_extractor",
            "pipeline.py",
            &file_ids,
            &aliases,
        );
        assert_eq!(resolved.as_deref(), Some("ast_extractor.py"));
    }

    #[test]
    fn resolves_relative_python_imports() {
        let file_ids = BTreeSet::from([
            "pkg/graph.py".to_string(),
            "pkg/sub/module.py".to_string(),
        ]);
        let aliases = BTreeSet::new();
        let resolved = resolve_import("..graph", "pkg/sub/module.py", &file_ids, &aliases);
        assert_eq!(resolved.as_deref(), Some("pkg/graph.py"));

        let paths = import_module_paths("..graph", "pkg/sub/module.py", &aliases);
        assert!(paths.iter().any(|path| path == "pkg/graph"));
    }

    #[test]
    fn resolves_workspace_crate_imports() {
        let file_ids = BTreeSet::from([
            "core-ir/src/lib.rs".to_string(),
            "store-sqlite/src/sqlite_store.rs".to_string(),
        ]);
        let aliases = BTreeSet::new();
        let resolved = resolve_import(
            "matryoshka_core_ir",
            "store-sqlite/src/sqlite_store.rs",
            &file_ids,
            &aliases,
        );
        assert_eq!(resolved.as_deref(), Some("core-ir/src/lib.rs"));
    }

    fn fake_file(file_id: &str, path: &str) -> FileFact {
        FileFact {
            file_id: file_id.into(),
            path: path.into(),
            name: Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(path)
                .into(),
            language: "python".into(),
            parent_folder_id: "repo".into(),
            source_hash: "hash".into(),
            line_count: 1,
            imports: Vec::new(),
            snippets: Vec::new(),
        }
    }
}
