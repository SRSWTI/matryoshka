use anyhow::{Context, Result};
use matryoshka_parser::{ParserConfig, hash_text};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use walkdir::WalkDir;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeBatch {
    pub changed_paths: Vec<String>,
    pub added_paths: Vec<String>,
    pub removed_paths: Vec<String>,
}

impl ChangeBatch {
    pub fn is_empty(&self) -> bool {
        self.changed_paths.is_empty() && self.added_paths.is_empty() && self.removed_paths.is_empty()
    }
}

pub struct RepoWatcher {
    repo_root: PathBuf,
    parser_config: ParserConfig,
    poll_interval: Duration,
    debounce_window: Duration,
    baseline: BTreeMap<String, String>,
    pending: Option<PendingBatch>,
}

struct PendingBatch {
    batch: ChangeBatch,
    last_change_at: Instant,
    next_state: BTreeMap<String, String>,
}

impl RepoWatcher {
    pub fn new(repo_root: impl AsRef<Path>) -> Result<Self> {
        let repo_root = repo_root.as_ref().to_path_buf();
        let parser_config = ParserConfig::default();
        let baseline = scan_repo_state(&repo_root, &parser_config)?;
        Ok(Self {
            repo_root,
            parser_config,
            poll_interval: Duration::from_secs(2),
            debounce_window: Duration::from_secs(3),
            baseline,
            pending: None,
        })
    }

    pub fn with_parser_config(mut self, parser_config: ParserConfig) -> Result<Self> {
        self.baseline = scan_repo_state(&self.repo_root, &parser_config)?;
        self.parser_config = parser_config;
        Ok(self)
    }

    pub fn with_poll_interval(mut self, poll_interval: Duration) -> Self {
        self.poll_interval = poll_interval;
        self
    }

    pub fn with_debounce_window(mut self, debounce_window: Duration) -> Self {
        self.debounce_window = debounce_window;
        self
    }

    pub fn poll_interval(&self) -> Duration {
        self.poll_interval
    }

    pub fn poll(&mut self) -> Result<Option<ChangeBatch>> {
        let current = scan_repo_state(&self.repo_root, &self.parser_config)?;
        let diff = diff_states(&self.baseline, &current);
        let now = Instant::now();

        if diff.is_empty() {
            if let Some(pending) = self.pending.take() {
                if now.duration_since(pending.last_change_at) >= self.debounce_window {
                    self.baseline = pending.next_state;
                    return Ok(Some(pending.batch));
                }
                self.pending = Some(pending);
            }
            return Ok(None);
        }

        match self.pending.take() {
            Some(mut pending) => {
                if pending.next_state == current
                    && now.duration_since(pending.last_change_at) >= self.debounce_window
                {
                    self.baseline = current;
                    return Ok(Some(pending.batch));
                }
                merge_batches(&mut pending.batch, diff);
                pending.last_change_at = now;
                pending.next_state = current;
                self.pending = Some(pending);
            }
            None => {
                self.pending = Some(PendingBatch {
                    batch: diff,
                    last_change_at: now,
                    next_state: current,
                });
            }
        }

        Ok(None)
    }
}

fn diff_states(previous: &BTreeMap<String, String>, current: &BTreeMap<String, String>) -> ChangeBatch {
    let mut changed_paths = Vec::new();
    let mut added_paths = Vec::new();
    let mut removed_paths = Vec::new();

    for (path, hash) in current {
        match previous.get(path) {
            Some(previous_hash) if previous_hash == hash => {}
            Some(_) => changed_paths.push(path.clone()),
            None => added_paths.push(path.clone()),
        }
    }

    for path in previous.keys() {
        if !current.contains_key(path) {
            removed_paths.push(path.clone());
        }
    }

    ChangeBatch {
        changed_paths,
        added_paths,
        removed_paths,
    }
}

fn merge_batches(target: &mut ChangeBatch, incoming: ChangeBatch) {
    target.changed_paths = merge_path_lists(&target.changed_paths, &incoming.changed_paths);
    target.added_paths = merge_path_lists(&target.added_paths, &incoming.added_paths);
    target.removed_paths = merge_path_lists(&target.removed_paths, &incoming.removed_paths);
}

fn merge_path_lists(existing: &[String], incoming: &[String]) -> Vec<String> {
    existing
        .iter()
        .chain(incoming.iter())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn scan_repo_state(repo_root: &Path, parser_config: &ParserConfig) -> Result<BTreeMap<String, String>> {
    let mut state = BTreeMap::new();

    for entry in WalkDir::new(repo_root).into_iter().filter_entry(|entry| {
        !entry
            .file_name()
            .to_str()
            .map(|name| parser_config.ignored_dirs.iter().any(|ignored| ignored == name))
            .unwrap_or(false)
    }) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        if !should_track(path, parser_config) {
            continue;
        }

        let relative = path
            .strip_prefix(repo_root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let source = fs::read_to_string(path)
            .with_context(|| format!("failed to read watched source file {}", path.display()))?;
        state.insert(relative, hash_text(&source));
    }

    Ok(state)
}

fn should_track(path: &Path, parser_config: &ParserConfig) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| parser_config.include_extensions.iter().any(|allowed| allowed == ext))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::RepoWatcher;
    use std::fs;
    use std::time::Duration;

    #[test]
    fn poll_emits_debounced_change_batch() {
        let temp = tempfile::tempdir().unwrap();
        let repo_root = temp.path();
        let source = repo_root.join("lib.rs");
        fs::write(&source, "pub fn a() {}\n").unwrap();

        let mut watcher = RepoWatcher::new(repo_root)
            .unwrap()
            .with_poll_interval(Duration::from_millis(5))
            .with_debounce_window(Duration::from_millis(10));

        fs::write(&source, "pub fn b() {}\n").unwrap();
        assert!(watcher.poll().unwrap().is_none());
        std::thread::sleep(Duration::from_millis(15));
        let batch = watcher.poll().unwrap().unwrap();
        assert_eq!(batch.changed_paths, vec!["lib.rs"]);
    }
}
