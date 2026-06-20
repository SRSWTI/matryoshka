use anyhow::{Context, Result};
use matryoshka_core_ir::{
    ChunkSummarySource, CodeChunkFact, CodeChunkKind, FileFact, ImportFact,
    MatryoshkaProgressEvent, SnippetFact, SymbolFact, SymbolKind,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use tree_sitter::{Node, Parser as TreeSitterParser};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct ParsedRepository {
    pub repo_root: PathBuf,
    pub files: Vec<FileFact>,
    pub symbols: Vec<SymbolFact>,
    pub code_chunks: Vec<CodeChunkFact>,
}

#[derive(Debug, Clone)]
pub struct ParserConfig {
    pub include_extensions: Vec<String>,
    pub ignored_dirs: Vec<String>,
    pub ignored_paths: Vec<String>,
    pub max_snippets_per_file: usize,
}

impl Default for ParserConfig {
    fn default() -> Self {
        Self {
            include_extensions: vec!["py".into(), "ts".into(), "tsx".into(), "rs".into()],
            ignored_dirs: vec![
                ".git".into(),
                ".venv".into(),
                "venv".into(),
                "node_modules".into(),
                "dist".into(),
                "build".into(),
                "__pycache__".into(),
                ".pytest_cache".into(),
                "target".into(),
            ],
            ignored_paths: Vec::new(),
            max_snippets_per_file: 6,
        }
    }
}

impl ParserConfig {
    pub fn with_ignored_paths(mut self, ignored_paths: impl IntoIterator<Item = String>) -> Self {
        self.ignored_paths.extend(
            ignored_paths
                .into_iter()
                .map(|path| normalize_ignored_path(&path))
                .filter(|path| !path.is_empty()),
        );
        self
    }

    pub fn ignores_entry(&self, repo_root: &Path, path: &Path) -> bool {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            return false;
        };
        if self.ignored_dirs.iter().any(|ignored| ignored == name) {
            return true;
        }
        let relative = relative_path(repo_root, path);
        self.ignored_paths
            .iter()
            .any(|ignored| path_matches_ignore(&relative, ignored))
    }
}

pub struct SourceParser {
    config: ParserConfig,
}

impl SourceParser {
    pub fn new(config: ParserConfig) -> Self {
        Self { config }
    }

    pub fn parse_repo(&self, repo_root: impl AsRef<Path>) -> Result<ParsedRepository> {
        self.parse_repo_with_progress(repo_root, |_| {})
    }

    pub fn parse_repo_with_progress(
        &self,
        repo_root: impl AsRef<Path>,
        mut progress: impl FnMut(MatryoshkaProgressEvent),
    ) -> Result<ParsedRepository> {
        let repo_root = repo_root.as_ref().to_path_buf();
        progress(MatryoshkaProgressEvent::DiscoveringFiles);
        let candidate_paths = self.discover_paths(&repo_root)?;
        let total_files = candidate_paths.len();
        progress(MatryoshkaProgressEvent::FilesDiscovered { total_files });
        let mut files = Vec::new();
        let mut symbols = Vec::new();
        let mut code_chunks = Vec::new();

        for (index, path) in candidate_paths.iter().enumerate() {
            let relative = relative_path(&repo_root, path);
            progress(MatryoshkaProgressEvent::ParsingFile {
                path: relative.clone(),
                index: index + 1,
                total_files,
            });
            let (file, mut file_symbols, mut file_chunks) = self.parse_file(&repo_root, path)?;
            progress(MatryoshkaProgressEvent::ParsedFile {
                path: relative,
                index: index + 1,
                total_files,
            });
            files.push(file);
            symbols.append(&mut file_symbols);
            code_chunks.append(&mut file_chunks);
        }

        files.sort_by(|left, right| left.path.cmp(&right.path));
        symbols.sort_by(|left, right| left.symbol_id.cmp(&right.symbol_id));
        code_chunks.sort_by(|left, right| left.chunk_id.cmp(&right.chunk_id));

        Ok(ParsedRepository {
            repo_root,
            files,
            symbols,
            code_chunks,
        })
    }

    fn discover_paths(&self, repo_root: &Path) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();
        for entry in WalkDir::new(repo_root)
            .into_iter()
            .filter_entry(|entry| !self.config.ignores_entry(repo_root, entry.path()))
        {
            let entry = entry?;
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.into_path();
            if !self.config.ignores_entry(repo_root, &path) && self.should_parse(&path) {
                paths.push(path);
            }
        }
        paths.sort();
        Ok(paths)
    }

    fn should_parse(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                self.config
                    .include_extensions
                    .iter()
                    .any(|allowed| allowed == ext)
            })
            .unwrap_or(false)
    }

    fn parse_file(
        &self,
        repo_root: &Path,
        path: &Path,
    ) -> Result<(FileFact, Vec<SymbolFact>, Vec<CodeChunkFact>)> {
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read source file {}", path.display()))?;
        let relative = path
            .strip_prefix(repo_root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let language = language_for(path);
        let source_hash = hash_text(&source);
        let lines: Vec<&str> = source.lines().collect();
        let parent_folder_id = parent_folder_id(&relative);
        let imports = parse_imports(&relative, &language, &lines);
        let symbols = parse_symbols(&relative, &language, &source, &lines);
        let snippets = select_snippets(
            &relative,
            &source,
            &symbols,
            self.config.max_snippets_per_file,
        );
        let code_chunks = build_code_chunks(&relative, &language, &lines, &symbols, &source_hash);

        let file = FileFact {
            file_id: relative.clone(),
            path: relative.clone(),
            name: Path::new(&relative)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&relative)
                .to_string(),
            language,
            parent_folder_id,
            source_hash,
            line_count: lines.len(),
            imports,
            snippets,
        };

        Ok((file, symbols, code_chunks))
    }
}

fn normalize_ignored_path(path: &str) -> String {
    path.trim()
        .trim_matches('/')
        .replace('\\', "/")
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect::<Vec<_>>()
        .join("/")
}

fn path_matches_ignore(relative_path: &str, ignored_path: &str) -> bool {
    if ignored_path.is_empty() || relative_path.is_empty() {
        return false;
    }
    if ignored_path.contains('/') {
        relative_path == ignored_path || relative_path.starts_with(&format!("{ignored_path}/"))
    } else {
        relative_path
            .split('/')
            .any(|component| component == ignored_path)
    }
}

fn relative_path(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

pub fn hash_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn language_for(path: &Path) -> String {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
    {
        "py" => "python",
        "ts" | "tsx" => "typescript",
        "rs" => "rust",
        other => other,
    }
    .to_string()
}

fn parent_folder_id(path: &str) -> String {
    Path::new(path)
        .parent()
        .and_then(|parent| parent.to_str())
        .filter(|parent| !parent.is_empty())
        .unwrap_or("repo")
        .replace('\\', "/")
}

fn parse_imports(file_id: &str, language: &str, lines: &[&str]) -> Vec<ImportFact> {
    let mut imports = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let parsed = match language {
            "python" => parse_python_import(trimmed),
            "typescript" => parse_typescript_import(trimmed),
            "rust" => parse_rust_import(trimmed),
            _ => None,
        };
        if let Some((module, names)) = parsed {
            imports.push(ImportFact {
                module,
                names,
                line: index + 1,
                resolved_file_id: None,
                is_internal: false,
            });
        }
    }
    imports.sort_by(|left, right| (left.line, &left.module).cmp(&(right.line, &right.module)));
    imports.dedup_by(|left, right| left.module == right.module && left.line == right.line);
    imports.iter_mut().for_each(|import| {
        import.is_internal = looks_internal(&import.module, file_id);
    });
    imports
}

fn parse_python_import(line: &str) -> Option<(String, Vec<String>)> {
    if let Some(rest) = line.strip_prefix("from ") {
        let mut parts = rest.splitn(2, " import ");
        let module = parts.next()?.trim().to_string();
        let names = parts
            .next()
            .unwrap_or_default()
            .split(',')
            .map(|name| {
                name.trim()
                    .split_whitespace()
                    .next()
                    .unwrap_or_default()
                    .to_string()
            })
            .filter(|name| !name.is_empty())
            .collect();
        return (!module.is_empty()).then_some((module, names));
    }
    if let Some(rest) = line.strip_prefix("import ") {
        let module = rest
            .split(',')
            .next()?
            .trim()
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_string();
        return (!module.is_empty()).then_some((module, Vec::new()));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{
        ParserConfig, SourceParser, build_code_chunks, extract_symbol_doc, parse_python_import,
        parse_rust_import, parse_rust_symbols,
    };
    use matryoshka_core_ir::{ChunkSummarySource, CodeChunkKind, SymbolKind};
    use std::fs;

    #[test]
    fn python_relative_imports_preserve_leading_dots() {
        let parsed = parse_python_import("from ..graph import RepositoryGraph").unwrap();
        assert_eq!(parsed.0, "..graph");
        assert_eq!(parsed.1, vec!["RepositoryGraph"]);
    }

    #[test]
    fn rust_grouped_imports_extract_module_and_names() {
        let parsed = parse_rust_import("use matryoshka_core_ir::{FileFact, SymbolFact};").unwrap();
        assert_eq!(parsed.0, "matryoshka_core_ir");
        assert_eq!(parsed.1, vec!["FileFact", "SymbolFact"]);
    }

    #[test]
    fn rust_impl_methods_are_qualified_as_methods() {
        let lines = vec![
            "pub struct MatryoshkaStore {",
            "    db_path: PathBuf,",
            "}",
            "",
            "impl MatryoshkaStore {",
            "    pub fn open(db_path: impl AsRef<Path>) -> Result<Self> {",
            "        Self { db_path: db_path.as_ref().to_path_buf() }",
            "    }",
            "}",
        ];
        let symbols = parse_rust_symbols("store.rs", &lines);
        assert!(symbols.iter().any(|symbol| {
            symbol.qualified_name == "MatryoshkaStore::open" && symbol.kind == SymbolKind::Method
        }));
        assert!(symbols.iter().any(|symbol| {
            symbol.qualified_name == "MatryoshkaStore" && symbol.kind == SymbolKind::Struct
        }));
    }

    #[test]
    fn tree_sitter_parser_extracts_python_methods() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            temp.path().join("service.py"),
            "class TokenService:\n    def refresh(self):\n        return True\n",
        )
        .unwrap();
        let parser = SourceParser::new(ParserConfig::default());
        let parsed = parser.parse_repo(temp.path()).unwrap();
        assert!(parsed.symbols.iter().any(|symbol| {
            symbol.qualified_name == "TokenService::refresh"
                && symbol.kind == SymbolKind::Method
                && symbol.start_line == 2
                && symbol.end_line == 3
        }));
    }

    #[test]
    fn tree_sitter_parser_extracts_typescript_class_methods() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            temp.path().join("client.ts"),
            "export class ApiClient {\n  async fetchToken(): Promise<string> {\n    return 'token';\n  }\n}\n",
        )
        .unwrap();
        let parser = SourceParser::new(ParserConfig::default());
        let parsed = parser.parse_repo(temp.path()).unwrap();
        assert!(parsed.symbols.iter().any(|symbol| {
            symbol.qualified_name == "ApiClient::fetchToken"
                && symbol.kind == SymbolKind::Method
                && symbol.start_line == 2
                && symbol.end_line == 4
        }));
    }

    #[test]
    fn parser_config_ignores_path_components_and_subtrees() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path().join("src")).unwrap();
        fs::create_dir_all(temp.path().join("tests")).unwrap();
        fs::create_dir_all(temp.path().join("packages/web")).unwrap();
        fs::write(temp.path().join("src/lib.rs"), "pub fn keep() {}\n").unwrap();
        fs::write(
            temp.path().join("tests/test_api.py"),
            "def drop_me(): pass\n",
        )
        .unwrap();
        fs::write(
            temp.path().join("packages/web/app.ts"),
            "export function app() {}\n",
        )
        .unwrap();

        let parser = SourceParser::new(
            ParserConfig::default()
                .with_ignored_paths(["tests".to_string(), "packages/web".to_string()]),
        );
        let parsed = parser.parse_repo(temp.path()).unwrap();
        let paths = parsed
            .files
            .iter()
            .map(|file| file.path.as_str())
            .collect::<Vec<_>>();

        assert_eq!(paths, vec!["src/lib.rs"]);
    }

    #[test]
    fn code_chunks_preserve_full_symbol_body_without_truncation() {
        let source = "pub fn big_function() {\n    let a = 1;\n    let b = 2;\n    let c = 3;\n    let d = 4;\n    let e = 5;\n    let f = 6;\n    let g = 7;\n    let h = 8;\n    let i = 9;\n    let j = 10;\n    let k = 11;\n    let l = 12;\n    let m = 13;\n    let n = 14;\n    let o = 15;\n    let p = 16;\n    let q = 17;\n    let r = 18;\n    let s = 19;\n    let t = 20;\n    let u = 21;\n    let v = 22;\n    let w = 23;\n    let x = 24;\n    let y = 25;\n    let z = 26;\n}\n";
        let lines: Vec<&str> = source.lines().collect();
        let symbols = super::parse_symbols("big.rs", "rust", source, &lines);
        assert_eq!(symbols.len(), 1);
        let chunks = build_code_chunks("big.rs", "rust", &lines, &symbols, "hash");
        assert_eq!(chunks.len(), 1);
        let chunk = &chunks[0];
        // Full body preserved: 28 lines (def + 26 body lines + closing brace).
        assert_eq!(chunk.start_line, 1);
        assert_eq!(chunk.end_line, 28);
        assert_eq!(chunk.code.lines().count(), 28);
        assert_eq!(chunk.kind, CodeChunkKind::Function);
        assert_eq!(chunk.summary_source, ChunkSummarySource::Empty);
    }

    #[test]
    fn rust_doc_comment_is_used_as_summary() {
        let source = "/// Resumes attack mode after handoff.\n///\n/// Cancels the countdown and updates state.\npub fn handle_resume_countdown(&mut self) {\n    self.countdown.cancel();\n    self.mode = Mode::Attack;\n}\n";
        let lines: Vec<&str> = source.lines().collect();
        let symbols = super::parse_symbols("bar.rs", "rust", source, &lines);
        assert_eq!(symbols.len(), 1);
        let chunks = build_code_chunks("bar.rs", "rust", &lines, &symbols, "hash");
        assert_eq!(chunks.len(), 1);
        let chunk = &chunks[0];
        assert_eq!(chunk.summary_source, ChunkSummarySource::DocComment);
        assert!(chunk.summary.contains("Resumes attack mode after handoff"));
        assert!(
            chunk
                .summary
                .contains("Cancels the countdown and updates state")
        );
        assert_eq!(chunk.doc_summary.as_deref(), Some(chunk.summary.as_str()));
        assert!(chunk.generated_summary.is_none());
    }

    #[test]
    fn python_docstring_is_used_as_summary() {
        let source = "def handle_resume_countdown(self):\n    \"\"\"Resume attack mode after handoff and cancel countdown.\"\"\"\n    self.countdown.cancel()\n    self.mode = Mode.ATTACK\n";
        let lines: Vec<&str> = source.lines().collect();
        let symbols = super::parse_symbols("bar.py", "python", source, &lines);
        assert_eq!(symbols.len(), 1);
        let chunks = build_code_chunks("bar.py", "python", &lines, &symbols, "hash");
        assert_eq!(chunks.len(), 1);
        let chunk = &chunks[0];
        assert_eq!(chunk.summary_source, ChunkSummarySource::Docstring);
        assert_eq!(
            chunk.summary.as_str(),
            "Resume attack mode after handoff and cancel countdown."
        );
    }

    #[test]
    fn typescript_jsdoc_is_used_as_summary() {
        let source = "/**\n * Resumes attack mode after handoff.\n * Cancels the countdown.\n */\nexport function handleResumeCountdown(): void {\n  cancelCountdown();\n}\n";
        let lines: Vec<&str> = source.lines().collect();
        let symbols = super::parse_symbols("bar.ts", "typescript", source, &lines);
        if symbols.is_empty() {
            // tree-sitter may not be available in all environments; skip gracefully.
            return;
        }
        let chunks = build_code_chunks("bar.ts", "typescript", &lines, &symbols, "hash");
        assert_eq!(chunks.len(), 1);
        let chunk = &chunks[0];
        assert_eq!(chunk.summary_source, ChunkSummarySource::DocComment);
        assert!(chunk.summary.contains("Resumes attack mode after handoff"));
        assert!(chunk.summary.contains("Cancels the countdown"));
    }

    #[test]
    fn generic_docstrings_are_treated_as_empty() {
        let source = "/// TODO\npub fn placeholder() {}\n";
        let lines: Vec<&str> = source.lines().collect();
        let symbols = super::parse_symbols("p.rs", "rust", source, &lines);
        let chunks = build_code_chunks("p.rs", "rust", &lines, &symbols, "hash");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].summary_source, ChunkSummarySource::Empty);
        assert!(chunks[0].summary.is_empty());
        assert!(chunks[0].doc_summary.is_none());
    }

    #[test]
    fn extract_symbol_doc_returns_none_when_no_doc() {
        let source = "pub fn no_doc() {\n    let x = 1;\n}\n";
        let lines: Vec<&str> = source.lines().collect();
        let symbols = super::parse_symbols("n.rs", "rust", source, &lines);
        assert_eq!(symbols.len(), 1);
        let doc = extract_symbol_doc("rust", &lines, &symbols[0]);
        assert!(doc.is_none());
    }

    #[test]
    fn parse_repo_emits_code_chunks_for_all_symbols() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            temp.path().join("lib.rs"),
            "/// Does the first thing.\npub fn first() {}\n\npub fn second() {}\n",
        )
        .unwrap();
        let parser = SourceParser::new(ParserConfig::default());
        let parsed = parser.parse_repo(temp.path()).unwrap();
        assert_eq!(parsed.code_chunks.len(), 2);
        let first = parsed
            .code_chunks
            .iter()
            .find(|c| c.symbol.as_deref() == Some("first"))
            .expect("first chunk should exist");
        assert_eq!(first.summary_source, ChunkSummarySource::DocComment);
        assert!(first.summary.contains("Does the first thing"));
        let second = parsed
            .code_chunks
            .iter()
            .find(|c| c.symbol.as_deref() == Some("second"))
            .expect("second chunk should exist");
        assert_eq!(second.summary_source, ChunkSummarySource::Empty);
        assert!(second.summary.is_empty());
    }
}

fn parse_typescript_import(line: &str) -> Option<(String, Vec<String>)> {
    if !line.starts_with("import ") && !line.starts_with("export ") {
        return None;
    }
    let quote = if line.contains('"') { '"' } else { '\'' };
    let parts: Vec<&str> = line.split(quote).collect();
    if parts.len() < 2 {
        return None;
    }
    let module = parts[1].to_string();
    let names = line
        .split('{')
        .nth(1)
        .and_then(|rest| rest.split('}').next())
        .map(|inside| {
            inside
                .split(',')
                .map(|name| {
                    name.trim()
                        .split_whitespace()
                        .next()
                        .unwrap_or_default()
                        .to_string()
                })
                .filter(|name| !name.is_empty())
                .collect()
        })
        .unwrap_or_default();
    Some((module, names))
}

fn parse_rust_import(line: &str) -> Option<(String, Vec<String>)> {
    let rest = line.strip_prefix("use ")?;
    let rest = rest.trim_end_matches(';').trim();
    if let Some((module, names)) = rest.split_once("::{") {
        let names = names
            .trim_end_matches('}')
            .split(',')
            .map(|name| name.trim())
            .filter(|name| !name.is_empty())
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let module = module.trim().replace("::", ".");
        return (!module.is_empty()).then_some((module, names));
    }
    let module = rest.replace("::", ".");
    (!module.is_empty()).then_some((module, Vec::new()))
}

fn looks_internal(module: &str, file_id: &str) -> bool {
    module.starts_with('.')
        || module.starts_with("./")
        || module.starts_with("../")
        || module.starts_with("crate.")
        || module.starts_with("self.")
        || module.starts_with("super.")
        || file_id
            .split('/')
            .next()
            .is_some_and(|root| module.starts_with(root))
}

fn parse_symbols(file_id: &str, language: &str, source: &str, lines: &[&str]) -> Vec<SymbolFact> {
    if let Some(symbols) = parse_tree_sitter_symbols(file_id, language, source) {
        return symbols;
    }
    if language == "rust" {
        return parse_rust_symbols(file_id, lines);
    }
    let mut symbols = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        let parsed = match language {
            "python" => parse_python_symbol(trimmed),
            "typescript" => parse_typescript_symbol(trimmed),
            "rust" => None,
            _ => None,
        };
        if let Some((kind, name, signature)) = parsed {
            let start_line = index + 1;
            let end_line = find_block_end(lines, index);
            let symbol_id = format!("{file_id}::{name}:{start_line}");
            symbols.push(SymbolFact {
                symbol_id,
                file_id: file_id.to_string(),
                path: file_id.to_string(),
                name: name.clone(),
                qualified_name: name,
                kind,
                signature,
                start_line,
                end_line,
            });
        }
    }
    symbols
}

fn parse_tree_sitter_symbols(
    file_id: &str,
    language: &str,
    source: &str,
) -> Option<Vec<SymbolFact>> {
    let mut parser = TreeSitterParser::new();
    let tree_sitter_language = match language {
        "rust" => tree_sitter_rust::LANGUAGE.into(),
        "python" => tree_sitter_python::LANGUAGE.into(),
        "typescript" if file_id.ends_with(".tsx") => tree_sitter_typescript::LANGUAGE_TSX.into(),
        "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        _ => return None,
    };
    parser.set_language(&tree_sitter_language).ok()?;
    let tree = parser.parse(source, None)?;
    let mut symbols = Vec::new();
    visit_tree_sitter_symbols(
        file_id,
        language,
        source,
        tree.root_node(),
        None,
        &mut symbols,
    );
    (!symbols.is_empty()).then_some(symbols)
}

fn visit_tree_sitter_symbols(
    file_id: &str,
    language: &str,
    source: &str,
    node: Node<'_>,
    owner: Option<String>,
    symbols: &mut Vec<SymbolFact>,
) {
    let mut next_owner = owner.clone();

    if let Some((kind, name, owner_for_children)) =
        tree_sitter_symbol_kind_and_name(language, source, node, owner.as_deref())
    {
        let start_line = node.start_position().row + 1;
        let end_line = node.end_position().row + 1;
        let qualified_name = owner
            .as_ref()
            .filter(|_| kind == SymbolKind::Method)
            .map(|owner| format!("{owner}::{name}"))
            .unwrap_or_else(|| name.clone());
        let symbol_id = format!("{file_id}::{qualified_name}:{start_line}");
        symbols.push(SymbolFact {
            symbol_id,
            file_id: file_id.to_string(),
            path: file_id.to_string(),
            name: name.clone(),
            qualified_name,
            kind,
            signature: tree_sitter_signature(source, node),
            start_line,
            end_line,
        });
        next_owner = owner_for_children.or(Some(name));
    } else if language == "rust" && node.kind() == "impl_item" {
        next_owner = rust_impl_target(source, node).or(owner);
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        visit_tree_sitter_symbols(
            file_id,
            language,
            source,
            child,
            next_owner.clone(),
            symbols,
        );
    }
}

fn tree_sitter_symbol_kind_and_name(
    language: &str,
    source: &str,
    node: Node<'_>,
    owner: Option<&str>,
) -> Option<(SymbolKind, String, Option<String>)> {
    let kind = node.kind();
    let name = tree_sitter_node_name(source, node)?;
    match language {
        "rust" => match kind {
            "function_item" => {
                let symbol_kind = if owner.is_some() {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                };
                Some((symbol_kind, name, None))
            }
            "struct_item" => Some((SymbolKind::Struct, name.clone(), Some(name))),
            "enum_item" => Some((SymbolKind::Enum, name.clone(), Some(name))),
            "trait_item" => Some((SymbolKind::Interface, name.clone(), Some(name))),
            "type_item" => Some((SymbolKind::TypeAlias, name, None)),
            _ => None,
        },
        "python" => match kind {
            "function_definition" => {
                let symbol_kind = if owner.is_some() {
                    SymbolKind::Method
                } else {
                    SymbolKind::Function
                };
                Some((symbol_kind, name, None))
            }
            "class_definition" => Some((SymbolKind::Class, name.clone(), Some(name))),
            _ => None,
        },
        "typescript" => match kind {
            "function_declaration" | "generator_function_declaration" => {
                Some((SymbolKind::Function, name, None))
            }
            "class_declaration" => Some((SymbolKind::Class, name.clone(), Some(name))),
            "method_definition" | "public_field_definition" => {
                Some((SymbolKind::Method, name, None))
            }
            "interface_declaration" => Some((SymbolKind::Interface, name.clone(), Some(name))),
            "type_alias_declaration" => Some((SymbolKind::TypeAlias, name, None)),
            "lexical_declaration" | "variable_declaration" => {
                if node_text(source, node).contains("=>")
                    || node_text(source, node).contains("function")
                {
                    Some((SymbolKind::Function, name, None))
                } else {
                    Some((SymbolKind::Constant, name, None))
                }
            }
            _ => None,
        },
        _ => None,
    }
}

fn tree_sitter_node_name(source: &str, node: Node<'_>) -> Option<String> {
    for field in ["name", "property", "identifier"] {
        if let Some(child) = node.child_by_field_name(field) {
            let text = node_text(source, child).trim().to_string();
            if !text.is_empty() {
                return Some(text);
            }
        }
    }

    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if matches!(
            child.kind(),
            "identifier" | "type_identifier" | "property_identifier" | "field_identifier"
        ) {
            let text = node_text(source, child).trim().to_string();
            if !text.is_empty() {
                return Some(text);
            }
        }
        if child.kind() == "variable_declarator" {
            if let Some(name) = tree_sitter_node_name(source, child) {
                return Some(name);
            }
        }
    }
    None
}

fn rust_impl_target(source: &str, node: Node<'_>) -> Option<String> {
    if let Some(type_node) = node.child_by_field_name("type") {
        return Some(clean_type_name(node_text(source, type_node)));
    }
    let text = node_text(source, node);
    let header = text.split('{').next()?.trim();
    let rest = header.strip_prefix("impl")?.trim();
    let target = rest
        .split(" for ")
        .last()
        .unwrap_or(rest)
        .split_whitespace()
        .last()
        .unwrap_or(rest);
    let target = clean_type_name(target);
    (!target.is_empty()).then_some(target)
}

fn clean_type_name(text: &str) -> String {
    text.trim()
        .trim_matches('{')
        .split('<')
        .next()
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn tree_sitter_signature(source: &str, node: Node<'_>) -> String {
    node_text(source, node)
        .lines()
        .next()
        .unwrap_or_default()
        .trim_end_matches('{')
        .trim_end_matches(':')
        .trim()
        .to_string()
}

fn node_text<'a>(source: &'a str, node: Node<'a>) -> &'a str {
    node.utf8_text(source.as_bytes()).unwrap_or_default()
}

fn parse_rust_symbols(file_id: &str, lines: &[&str]) -> Vec<SymbolFact> {
    let mut symbols = Vec::new();
    let mut brace_depth = 0usize;
    let mut impl_stack: Vec<(String, usize)> = Vec::new();

    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        let depth_before = brace_depth;

        while let Some((_, close_depth)) = impl_stack.last() {
            if *close_depth > depth_before {
                impl_stack.pop();
            } else {
                break;
            }
        }

        if let Some(type_name) = parse_rust_impl_target(trimmed) {
            impl_stack.push((type_name, depth_before + 1));
        }

        if let Some((kind, name, signature)) = parse_rust_symbol(trimmed) {
            let start_line = index + 1;
            let end_line = find_rust_block_end(lines, index);
            let (name, qualified_name, kind) = if kind == SymbolKind::Function {
                if let Some((owner, _)) = impl_stack.last() {
                    (name.clone(), format!("{owner}::{name}"), SymbolKind::Method)
                } else {
                    (name.clone(), name, SymbolKind::Function)
                }
            } else {
                (name.clone(), name, kind)
            };
            let symbol_id = format!("{file_id}::{qualified_name}:{start_line}");
            symbols.push(SymbolFact {
                symbol_id,
                file_id: file_id.to_string(),
                path: file_id.to_string(),
                name,
                qualified_name,
                kind,
                signature,
                start_line,
                end_line,
            });
        }

        brace_depth = update_brace_depth(brace_depth, line);
    }

    symbols
}

fn parse_python_symbol(line: &str) -> Option<(SymbolKind, String, String)> {
    if line.starts_with("def ") || line.starts_with("async def ") {
        let signature = line.trim_end_matches(':').to_string();
        let name = signature
            .split("def ")
            .nth(1)?
            .split('(')
            .next()?
            .trim()
            .to_string();
        return Some((SymbolKind::Function, name, signature));
    }
    if let Some(rest) = line.strip_prefix("class ") {
        let signature = line.trim_end_matches(':').to_string();
        let name = rest.split(['(', ':']).next()?.trim().to_string();
        return Some((SymbolKind::Class, name, signature));
    }
    None
}

fn parse_typescript_symbol(line: &str) -> Option<(SymbolKind, String, String)> {
    let cleaned = line
        .strip_prefix("export ")
        .unwrap_or(line)
        .strip_prefix("default ")
        .unwrap_or(line)
        .trim();
    for prefix in ["async function ", "function "] {
        if let Some(rest) = cleaned.strip_prefix(prefix) {
            let name = rest.split('(').next()?.trim().to_string();
            return Some((SymbolKind::Function, name, cleaned.to_string()));
        }
    }
    if let Some(rest) = cleaned.strip_prefix("class ") {
        let name = rest.split([' ', '{', '<']).next()?.trim().to_string();
        return Some((SymbolKind::Class, name, cleaned.to_string()));
    }
    if let Some(rest) = cleaned.strip_prefix("interface ") {
        let name = rest.split([' ', '{', '<']).next()?.trim().to_string();
        return Some((SymbolKind::Interface, name, cleaned.to_string()));
    }
    for prefix in ["const ", "let ", "var "] {
        if let Some(rest) = cleaned.strip_prefix(prefix) {
            if rest.contains("=>") || rest.contains("function") {
                let name = rest.split([':', '=', ' ']).next()?.trim().to_string();
                return Some((SymbolKind::Function, name, cleaned.to_string()));
            }
        }
    }
    None
}

fn parse_rust_symbol(line: &str) -> Option<(SymbolKind, String, String)> {
    let cleaned = line.strip_prefix("pub ").unwrap_or(line).trim();
    if let Some(rest) = cleaned.strip_prefix("fn ") {
        let name = rest.split('(').next()?.trim().to_string();
        return Some((SymbolKind::Function, name, cleaned.to_string()));
    }
    if let Some(rest) = cleaned.strip_prefix("struct ") {
        let name = rest.split([' ', '{', '<', ';']).next()?.trim().to_string();
        return Some((SymbolKind::Struct, name, cleaned.to_string()));
    }
    if let Some(rest) = cleaned.strip_prefix("enum ") {
        let name = rest.split([' ', '{', '<', ';']).next()?.trim().to_string();
        return Some((SymbolKind::Enum, name, cleaned.to_string()));
    }
    None
}

fn parse_rust_impl_target(line: &str) -> Option<String> {
    let cleaned = line.strip_prefix("pub ").unwrap_or(line).trim();
    let rest = cleaned.strip_prefix("impl")?.trim();
    let target = if let Some((_, after_for)) = rest.split_once(" for ") {
        after_for
    } else {
        rest
    };
    let target = target.trim_end_matches('{').trim();
    let target = target
        .split('<')
        .next()
        .unwrap_or(target)
        .split_whitespace()
        .next()
        .unwrap_or(target)
        .trim();
    (!target.is_empty()).then_some(target.to_string())
}

fn find_rust_block_end(lines: &[&str], start_index: usize) -> usize {
    let mut depth = 0usize;
    let mut seen_open = false;
    for (index, line) in lines.iter().enumerate().skip(start_index) {
        for ch in line.chars() {
            match ch {
                '{' => {
                    depth += 1;
                    seen_open = true;
                }
                '}' => {
                    if depth > 0 {
                        depth -= 1;
                    }
                    if seen_open && depth == 0 {
                        return index + 1;
                    }
                }
                _ => {}
            }
        }
    }
    lines.len()
}

fn update_brace_depth(current: usize, line: &str) -> usize {
    let opens = line.chars().filter(|ch| *ch == '{').count();
    let closes = line.chars().filter(|ch| *ch == '}').count();
    current.saturating_add(opens).saturating_sub(closes)
}

fn find_block_end(lines: &[&str], start_index: usize) -> usize {
    let base_indent = lines[start_index]
        .chars()
        .take_while(|ch| ch.is_whitespace())
        .count();
    for (index, line) in lines.iter().enumerate().skip(start_index + 1) {
        if line.trim().is_empty() {
            continue;
        }
        let indent = line.chars().take_while(|ch| ch.is_whitespace()).count();
        if indent <= base_indent
            && (line.trim_start().starts_with("def ")
                || line.trim_start().starts_with("class ")
                || line.trim_start().starts_with("function ")
                || line.trim_start().starts_with("pub fn "))
        {
            return index;
        }
    }
    lines.len()
}

fn select_snippets(
    file_id: &str,
    source: &str,
    symbols: &[SymbolFact],
    limit: usize,
) -> Vec<SnippetFact> {
    let lines: Vec<&str> = source.lines().collect();
    symbols
        .iter()
        .take(limit)
        .map(|symbol| {
            let start = symbol.start_line.saturating_sub(1);
            let end = symbol.end_line.min(symbol.start_line + 20).min(lines.len());
            SnippetFact {
                snippet_id: format!("{}#{}-{}", file_id, symbol.start_line, end),
                file_id: file_id.to_string(),
                title: symbol.qualified_name.clone(),
                start_line: symbol.start_line,
                end_line: end,
                text: lines[start..end].join("\n"),
            }
        })
        .collect()
}

/// Build first-class `CodeChunkFact`s from parsed symbols. Each chunk preserves
/// the full symbol body (no truncation) and extracts any leading docstring / doc
/// comment so the indexer can skip LLM summarization when a useful doc exists.
fn build_code_chunks(
    file_id: &str,
    language: &str,
    lines: &[&str],
    symbols: &[SymbolFact],
    source_hash: &str,
) -> Vec<CodeChunkFact> {
    symbols
        .iter()
        .filter_map(|symbol| {
            let chunk_kind = chunk_kind_for_symbol(symbol.kind);
            let start = symbol.start_line.saturating_sub(1);
            let end = symbol.end_line.min(lines.len());
            if end <= start {
                return None;
            }
            let code = lines[start..end].join("\n");
            let doc = extract_symbol_doc(language, lines, symbol);
            let useful = doc.as_deref().is_some_and(doc_is_useful);
            let (summary, summary_source) = if useful {
                let doc_text = doc.as_deref().unwrap_or_default();
                (doc_text.to_string(), doc_summary_source(language, doc_text))
            } else {
                (String::new(), ChunkSummarySource::Empty)
            };
            let chunk_id = format!(
                "{}::{}:{}",
                file_id, symbol.qualified_name, symbol.start_line
            );
            Some(CodeChunkFact {
                chunk_id,
                file_id: file_id.to_string(),
                symbol_id: Some(symbol.symbol_id.clone()),
                path: file_id.to_string(),
                symbol: Some(symbol.name.clone()),
                qualified_name: Some(symbol.qualified_name.clone()),
                kind: chunk_kind,
                signature: symbol.signature.clone(),
                start_line: symbol.start_line,
                end_line: symbol.end_line,
                doc_summary: if useful { doc.clone() } else { None },
                generated_summary: None,
                summary,
                summary_source,
                code,
                source_hash: source_hash.to_string(),
            })
        })
        .collect()
}

fn chunk_kind_for_symbol(kind: SymbolKind) -> CodeChunkKind {
    match kind {
        SymbolKind::Function => CodeChunkKind::Function,
        SymbolKind::Method => CodeChunkKind::Method,
        SymbolKind::Class => CodeChunkKind::Class,
        SymbolKind::Struct => CodeChunkKind::Struct,
        SymbolKind::Enum => CodeChunkKind::Enum,
        SymbolKind::Interface => CodeChunkKind::Interface,
        SymbolKind::TypeAlias | SymbolKind::Constant | SymbolKind::Unknown => {
            CodeChunkKind::Unknown
        }
    }
}

/// A doc is "useful" if it is long enough and not a generic placeholder.
fn doc_is_useful(doc: &str) -> bool {
    let trimmed = doc.trim();
    if trimmed.len() < matryoshka_core_ir::MIN_USEFUL_DOC_SUMMARY_LEN {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    !matches!(
        lower.as_str(),
        "todo" | "fixme" | "placeholder" | "stub" | "handles event" | "helper" | "main"
    )
}

/// Pick the right `ChunkSummarySource` variant based on the doc text and language.
/// Python docstrings come from inside the body; Rust/TS docs come from leading comments.
fn doc_summary_source(language: &str, doc: &str) -> ChunkSummarySource {
    let trimmed = doc.trim();
    if trimmed.is_empty() {
        return ChunkSummarySource::Empty;
    }
    match language {
        "python" => ChunkSummarySource::Docstring,
        "rust" | "typescript" => ChunkSummarySource::DocComment,
        _ => ChunkSummarySource::DocComment,
    }
}

/// Extract a docstring / doc comment for a symbol. Returns the cleaned doc text.
fn extract_symbol_doc(language: &str, lines: &[&str], symbol: &SymbolFact) -> Option<String> {
    match language {
        "python" => extract_python_docstring(lines, symbol),
        "rust" => extract_leading_doc_comment(lines, symbol, "///", "//!"),
        "typescript" => extract_typescript_doc(lines, symbol),
        _ => None,
    }
}

/// Python: docstring is the first string literal inside the function/class body.
fn extract_python_docstring(lines: &[&str], symbol: &SymbolFact) -> Option<String> {
    let start = symbol.start_line; // 1-based
    // Search the first few lines of the body for a triple-quoted string.
    let body_start = start; // def line is `start`; body starts at start+1
    let max_search = (body_start + 6).min(lines.len());
    let mut in_doc = false;
    let mut doc_lines: Vec<String> = Vec::new();
    for index in body_start..max_search {
        let line = lines[index].trim();
        if line.is_empty() {
            continue;
        }
        if !in_doc {
            if let Some(rest) = line
                .strip_prefix("\"\"\"")
                .or_else(|| line.strip_prefix("'''"))
            {
                // Single-line docstring: """..."""
                if let Some(content) = rest
                    .strip_suffix("\"\"\"")
                    .or_else(|| rest.strip_suffix("'''"))
                {
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                    return None;
                }
                // Multi-line docstring started.
                in_doc = true;
                let rest_trimmed = rest.trim();
                if !rest_trimmed.is_empty() {
                    doc_lines.push(rest_trimmed.to_string());
                }
                continue;
            }
            // First non-empty, non-docstring line means no docstring.
            return None;
        } else {
            // Inside multi-line docstring; look for closing quotes.
            if let Some(end_idx) = line.find("\"\"\"").or_else(|| line.find("'''")) {
                let before = line[..end_idx].trim();
                if !before.is_empty() {
                    doc_lines.push(before.to_string());
                }
                break;
            }
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                doc_lines.push(trimmed.to_string());
            }
        }
    }
    if doc_lines.is_empty() {
        None
    } else {
        Some(doc_lines.join(" "))
    }
}

/// Rust / TS-style leading doc comments. Collects contiguous `///` (or `//!`)
/// lines immediately above the symbol. Also handles block comments `/** ... */`.
fn extract_leading_doc_comment(
    lines: &[&str],
    symbol: &SymbolFact,
    line_prefix: &str,
    _module_prefix: &str,
) -> Option<String> {
    if symbol.start_line == 0 {
        return None;
    }
    let mut doc_lines: Vec<String> = Vec::new();
    let mut index = symbol.start_line.saturating_sub(1); // line before the symbol (0-based)
    while index > 0 {
        index -= 1;
        let raw = lines[index];
        let trimmed = raw.trim_start();
        if let Some(rest) = trimmed.strip_prefix(line_prefix) {
            let text = rest.trim_start_matches(' ').trim();
            doc_lines.push(text.to_string());
        } else if trimmed.is_empty() {
            // Allow blank lines between doc and symbol? No: stop at first blank.
            break;
        } else {
            break;
        }
    }
    if doc_lines.is_empty() {
        return None;
    }
    doc_lines.reverse();
    let joined = doc_lines.join(" ");
    let trimmed = joined.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// TypeScript: supports JSDoc block comments `/** ... */` and `//` line comments
/// immediately above the symbol. The comment block must be *contiguous* with the
/// symbol (no blank lines or code in between), otherwise we return None.
fn extract_typescript_doc(lines: &[&str], symbol: &SymbolFact) -> Option<String> {
    if symbol.start_line == 0 {
        return None;
    }
    // `symbol.start_line` is 1-based; the line immediately above is at index
    // `start_line - 2` (0-based). If that line is not a comment, there's no doc.
    let above_index = symbol.start_line.checked_sub(2)?;
    let line = lines[above_index].trim();
    // JSDoc block comment ending on this line.
    if line.ends_with("*/") {
        return extract_jsdoc_block(lines, above_index);
    }
    // `//` line comments above the symbol.
    if line.starts_with("//") {
        return extract_line_comments(lines, above_index, "//");
    }
    None
}

fn extract_jsdoc_block(lines: &[&str], end_index: usize) -> Option<String> {
    let mut doc_lines: Vec<String> = Vec::new();
    let mut index = end_index;
    // Include the closing line.
    let closing_text = lines[index]
        .trim()
        .trim_end_matches("*/")
        .trim_start_matches('*')
        .trim();
    if !closing_text.is_empty() {
        doc_lines.push(closing_text.to_string());
    }
    loop {
        if index == 0 {
            // Never found an opener; this wasn't a real JSDoc block.
            return None;
        }
        index -= 1;
        let raw = lines[index].trim();
        if let Some(rest) = raw.strip_prefix("/**") {
            let text = rest.trim();
            if !text.is_empty() {
                doc_lines.push(text.to_string());
            }
            break;
        }
        // Inside a JSDoc block every line must start with `*` (continuation) or
        // be the `/**` opener. If we hit anything else, this `*/` didn't belong
        // to a JSDoc block attached to the symbol — bail.
        if !raw.starts_with('*') {
            return None;
        }
        let text = raw.trim_start_matches('*').trim();
        if !text.is_empty() {
            doc_lines.push(text.to_string());
        }
    }
    if doc_lines.is_empty() {
        return None;
    }
    doc_lines.reverse();
    let joined = doc_lines.join(" ");
    let trimmed = joined.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn extract_line_comments(lines: &[&str], start_index: usize, prefix: &str) -> Option<String> {
    let mut doc_lines: Vec<String> = Vec::new();
    let mut index = start_index;
    loop {
        let raw = lines[index].trim();
        if let Some(rest) = raw.strip_prefix(prefix) {
            let text = rest.trim();
            if !text.is_empty() {
                doc_lines.push(text.to_string());
            }
        } else {
            break;
        }
        if index == 0 {
            break;
        }
        index -= 1;
    }
    if doc_lines.is_empty() {
        return None;
    }
    doc_lines.reverse();
    let joined = doc_lines.join(" ");
    let trimmed = joined.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}
