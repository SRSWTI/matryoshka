use anyhow::{Context, Result};
use matryoshka_core_ir::{
    FileFact, ImportFact, MatryoshkaProgressEvent, SnippetFact, SymbolFact, SymbolKind,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct ParsedRepository {
    pub repo_root: PathBuf,
    pub files: Vec<FileFact>,
    pub symbols: Vec<SymbolFact>,
}

#[derive(Debug, Clone)]
pub struct ParserConfig {
    pub include_extensions: Vec<String>,
    pub ignored_dirs: Vec<String>,
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
            max_snippets_per_file: 6,
        }
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

        for (index, path) in candidate_paths.iter().enumerate() {
            let relative = relative_path(&repo_root, path);
            progress(MatryoshkaProgressEvent::ParsingFile {
                path: relative.clone(),
                index: index + 1,
                total_files,
            });
            let (file, mut file_symbols) = self.parse_file(&repo_root, path)?;
            progress(MatryoshkaProgressEvent::ParsedFile {
                path: relative,
                index: index + 1,
                total_files,
            });
            files.push(file);
            symbols.append(&mut file_symbols);
        }

        files.sort_by(|left, right| left.path.cmp(&right.path));
        symbols.sort_by(|left, right| left.symbol_id.cmp(&right.symbol_id));

        Ok(ParsedRepository {
            repo_root,
            files,
            symbols,
        })
    }

    fn discover_paths(&self, repo_root: &Path) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();
        for entry in WalkDir::new(repo_root).into_iter().filter_entry(|entry| {
            !entry
                .file_name()
                .to_str()
                .map(|name| {
                    self.config
                        .ignored_dirs
                        .iter()
                        .any(|ignored| ignored == name)
                })
                .unwrap_or(false)
        }) {
            let entry = entry?;
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.into_path();
            if self.should_parse(&path) {
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

    fn parse_file(&self, repo_root: &Path, path: &Path) -> Result<(FileFact, Vec<SymbolFact>)> {
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
        let symbols = parse_symbols(&relative, &language, &lines);
        let snippets = select_snippets(
            &relative,
            &source,
            &symbols,
            self.config.max_snippets_per_file,
        );

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

        Ok((file, symbols))
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
    use super::{parse_python_import, parse_rust_import, parse_rust_symbols};
    use matryoshka_core_ir::SymbolKind;

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

fn parse_symbols(file_id: &str, language: &str, lines: &[&str]) -> Vec<SymbolFact> {
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
