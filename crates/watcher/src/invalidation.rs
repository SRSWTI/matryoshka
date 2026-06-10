use anyhow::Result;
use matryoshka_core_ir::InvalidationSet;
use matryoshka_store_sqlite::MatryoshkaStore;

pub struct InvalidationPlanner {
    store: MatryoshkaStore,
}

impl InvalidationPlanner {
    pub fn new(store: MatryoshkaStore) -> Self {
        Self { store }
    }

    pub fn mark_file_changed(
        &self,
        file_id: &str,
        parent_folder_id: Option<&str>,
    ) -> Result<InvalidationSet> {
        self.store
            .mark_stale("file", file_id, "file content changed")?;
        if let Some(folder_id) = parent_folder_id {
            self.store
                .mark_stale("folder", folder_id, "child file changed")?;
        }
        self.store
            .mark_stale("repo", "repo", "file change may affect repository map")?;
        Ok(InvalidationSet {
            file_ids: vec![file_id.into()],
            folder_ids: parent_folder_id
                .map(|id| vec![id.into()])
                .unwrap_or_default(),
            repo_stale: true,
            reason: "file content changed".into(),
        })
    }

    pub fn mark_cross_folder_dependency_changed(
        &self,
        source_folder_id: &str,
        target_folder_id: &str,
    ) -> Result<InvalidationSet> {
        self.store.mark_stale(
            "folder",
            source_folder_id,
            "cross-folder dependency changed",
        )?;
        self.store.mark_stale(
            "folder",
            target_folder_id,
            "cross-folder dependency changed",
        )?;
        self.store
            .mark_stale("repo", "repo", "cross-folder dependency changed")?;
        Ok(InvalidationSet {
            file_ids: Vec::new(),
            folder_ids: vec![source_folder_id.into(), target_folder_id.into()],
            repo_stale: true,
            reason: "cross-folder dependency changed".into(),
        })
    }
}
