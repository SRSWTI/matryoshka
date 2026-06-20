use matryoshka_core_ir::MatryoshkaProgressEvent;
use matryoshka_embed_client::DeterministicEmbedder;
use matryoshka_embed_client::EndpointEmbedder;
use matryoshka_enricher::HeuristicChunkSummarizer;
use matryoshka_enricher::HeuristicEnricher;
use matryoshka_enricher::MlxChatEnricher;
use matryoshka_indexer::FullIndexer;
use matryoshka_read_api::{ReadApi, ReadPackMode};
use matryoshka_search::{SearchEngine, default_prewarm_queries};
use matryoshka_store_sqlite::MatryoshkaStore;
use std::fs;

#[test]
fn indexes_searches_and_reads_file_cards() {
    let repo_root = std::path::PathBuf::from("tests/fixtures/mini_repo");
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("index.db");
    let store = MatryoshkaStore::open(&db_path).unwrap();
    let indexer = FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default(), HeuristicChunkSummarizer);

    let summary = indexer.index_repo(&repo_root).unwrap();

    assert_eq!(summary.file_count, 2);
    assert!(summary.folder_count >= 3);
    assert!(summary.symbol_count >= 2);
    assert!(summary.semantic_record_count >= 6);

    let records = MatryoshkaStore::open(&db_path)
        .unwrap()
        .load_all_semantic_records()
        .unwrap();
    assert!(records.iter().any(|record| {
        record
            .metadata
            .get("kind")
            .and_then(serde_json::Value::as_str)
            == Some("repo_card")
    }));
    assert!(records.iter().any(|record| {
        matches!(
            record.entity_type,
            matryoshka_core_ir::SemanticEntityType::Snippet
                | matryoshka_core_ir::SemanticEntityType::Symbol
        ) && record
            .embedding
            .as_ref()
            .is_some_and(|embedding| !embedding.is_empty())
    }));
    let fts_hits = MatryoshkaStore::open(&db_path)
        .unwrap()
        .search_semantic_fts("get_env_api_key", 8)
        .unwrap();
    assert!(
        fts_hits.iter().any(|hit| hit.record_id.contains("env.py")),
        "{fts_hits:?}"
    );
    let late_record_ids = records
        .iter()
        .map(|record| record.record_id.clone())
        .collect::<Vec<_>>();
    let late_vectors = MatryoshkaStore::open(&db_path)
        .unwrap()
        .load_late_interaction_vectors(&late_record_ids)
        .unwrap();
    assert!(
        late_vectors.values().any(|vectors| !vectors.is_empty()),
        "expected indexed late-interaction vectors"
    );

    let search = SearchEngine::new(
        MatryoshkaStore::open(&db_path).unwrap(),
        DeterministicEmbedder::default(),
    );
    let repo_hits = search.search("repository architecture", 5).unwrap();
    assert!(repo_hits.iter().any(|hit| {
        matches!(
            hit.entity_type,
            matryoshka_core_ir::SemanticEntityType::Repo
        )
    }));

    let hits = search.search("api key loaded from environment", 5).unwrap();
    assert!(!hits.is_empty());
    assert!(hits.iter().any(|hit| hit.path.contains("env.py")));
    assert!(hits.iter().any(|hit| !hit.why_matched.is_empty()));
    assert!(hits.iter().any(|hit| { hit.path.contains("env.py") }));

    let symbol_hits = search
        .search("where is get_env_api_key defined", 5)
        .unwrap();
    assert!(
        symbol_hits.iter().any(|hit| hit.path.contains("env.py")),
        "{symbol_hits:?}"
    );
    assert!(symbol_hits.iter().any(|hit| {
        hit.why_matched
            .iter()
            .any(|why| why.contains("SQLite FTS") || why.contains("Symbol query plan"))
    }));
    assert!(symbol_hits.iter().any(|hit| {
        hit.why_matched
            .iter()
            .any(|why| why.contains("Late-interaction MaxSim"))
    }));

    let prewarm = search.prewarm(&default_prewarm_queries(), 3).unwrap();
    assert!(prewarm.fts_record_count >= records.len());
    assert_eq!(prewarm.query_count, default_prewarm_queries().len());
    assert!(prewarm.warmed_hit_count > 0);

    let read = ReadApi::new(MatryoshkaStore::open(&db_path).unwrap(), repo_root);
    let card = read.read("src/auth/middleware.py").unwrap();
    assert_eq!(card.file.path, "src/auth/middleware.py");
    assert!(!card.symbols.is_empty());
    assert!(card.imports.external.is_some() || !card.imports.internal.is_empty());

    let bundle = read
        .read_bundle(
            "src/auth/middleware.py",
            &["src/config/env.py".to_string()],
            ReadPackMode::Edit,
            2,
        )
        .unwrap();
    assert_eq!(bundle.primary.file.path, "src/auth/middleware.py");
    assert_eq!(bundle.related.len(), 1);
    assert_eq!(bundle.related[0].file.path, "src/config/env.py");
    assert!(!bundle.primary.symbols.is_empty());
}

#[test]
fn incremental_update_refreshes_changed_entities_and_preserves_unaffected_cards() {
    let temp = tempfile::tempdir().unwrap();
    let repo_root = temp.path();
    fs::create_dir_all(repo_root.join("src")).unwrap();
    fs::write(
        repo_root.join("src/lib.rs"),
        "use crate::util::helper;\n\npub fn api() -> &'static str { helper() }\n",
    )
    .unwrap();
    fs::write(
        repo_root.join("src/util.rs"),
        "pub fn helper() -> &'static str { \"old behavior\" }\n",
    )
    .unwrap();

    let db_path = temp.path().join("index.db");
    let store = MatryoshkaStore::open(&db_path).unwrap();
    let indexer = FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default(), HeuristicChunkSummarizer);
    indexer.index_repo(repo_root).unwrap();

    let store = MatryoshkaStore::open(&db_path).unwrap();
    let before_unaffected = store.load_file_card("src/lib.rs").unwrap().unwrap();
    let before_changed_hash = store.load_file("src/util.rs").unwrap().unwrap().source_hash;

    fs::write(
        repo_root.join("src/util.rs"),
        "pub fn helper() -> &'static str { \"new behavior\" }\n\npub fn cache_key() -> &'static str { \"util-cache\" }\n",
    )
    .unwrap();

    let summary = indexer.update_repo(repo_root).unwrap();
    assert_eq!(summary.changed_files, 1);
    assert_eq!(summary.removed_files, 0);
    assert!(summary.changed_folders >= 1);

    let store = MatryoshkaStore::open(&db_path).unwrap();
    let after_unaffected = store.load_file_card("src/lib.rs").unwrap().unwrap();
    let after_changed = store.load_file("src/util.rs").unwrap().unwrap();
    assert_eq!(before_unaffected, after_unaffected);
    assert_ne!(before_changed_hash, after_changed.source_hash);

    let search = SearchEngine::new(store, DeterministicEmbedder::default());
    let hits = search.search("cache_key util cache", 5).unwrap();
    assert!(hits.iter().any(|hit| hit.path == "src/util.rs"));
    let fts_hits = MatryoshkaStore::open(&db_path)
        .unwrap()
        .search_semantic_fts("cache_key", 5)
        .unwrap();
    assert!(
        fts_hits
            .iter()
            .any(|hit| hit.record_id.contains("src/util.rs")),
        "{fts_hits:?}"
    );
}

#[test]
fn incremental_update_removes_deleted_files_from_fts_candidates() {
    let temp = tempfile::tempdir().unwrap();
    let repo_root = temp.path();
    fs::create_dir_all(repo_root.join("src")).unwrap();
    fs::write(
        repo_root.join("src/remove_me.rs"),
        "pub fn obsolete_unique_marker() -> &'static str { \"gone\" }\n",
    )
    .unwrap();
    fs::write(
        repo_root.join("src/keep.rs"),
        "pub fn stable_entrypoint() -> &'static str { \"keep\" }\n",
    )
    .unwrap();

    let db_path = temp.path().join("index.db");
    let store = MatryoshkaStore::open(&db_path).unwrap();
    let indexer = FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default(), HeuristicChunkSummarizer);
    indexer.index_repo(repo_root).unwrap();

    let before_hits = MatryoshkaStore::open(&db_path)
        .unwrap()
        .search_semantic_fts("obsolete_unique_marker", 8)
        .unwrap();
    assert!(
        before_hits
            .iter()
            .any(|hit| hit.record_id.contains("remove_me.rs")),
        "{before_hits:?}"
    );

    fs::remove_file(repo_root.join("src/remove_me.rs")).unwrap();
    let summary = indexer.update_repo(repo_root).unwrap();
    assert_eq!(summary.removed_files, 1);

    let after_store = MatryoshkaStore::open(&db_path).unwrap();
    let after_hits = after_store
        .search_semantic_fts("obsolete_unique_marker", 8)
        .unwrap();
    assert!(after_hits.is_empty(), "{after_hits:?}");

    let search = SearchEngine::new(after_store, DeterministicEmbedder::default());
    let hits = search.search("obsolete_unique_marker", 5).unwrap();
    assert!(
        hits.iter().all(|hit| !hit.path.contains("remove_me.rs")),
        "{hits:?}"
    );
}

#[test]
fn heuristic_parent_folder_cards_use_grounded_rollup_summaries() {
    let temp = tempfile::tempdir().unwrap();
    let repo_root = temp.path();
    fs::create_dir_all(repo_root.join("gateway/crates/service/src")).unwrap();
    fs::write(
        repo_root.join("gateway/crates/service/src/lib.rs"),
        "pub fn route_request() -> &'static str { \"routed\" }\n",
    )
    .unwrap();

    let db_path = temp.path().join("index.db");
    let store = MatryoshkaStore::open(&db_path).unwrap();
    let indexer = FullIndexer::new(
                store.clone(),
                HeuristicEnricher,
                DeterministicEmbedder::default(),
                HeuristicChunkSummarizer,
            );

    indexer.index_repo(repo_root).unwrap();

    let parent = store
        .load_folder_card("gateway/crates/service")
        .unwrap()
        .unwrap();
    assert!(
        parent.summary.contains("gateway/crates/service"),
        "{}",
        parent.summary
    );
    assert!(
        parent.summary.contains("gateway/crates/service/src"),
        "{}",
        parent.summary
    );
    assert!(
        !parent.summary.contains("central hub") && !parent.summary.contains("semantic"),
        "{}",
        parent.summary
    );
    assert!(
        parent.responsibility.contains("gateway/crates/service"),
        "{}",
        parent.responsibility
    );
    assert!(
        parent
            .subareas
            .iter()
            .any(|subarea| subarea.id == "gateway/crates/service/src"
                && subarea
                    .responsibility
                    .contains("gateway/crates/service/src/lib.rs")),
        "{:?}",
        parent.subareas
    );
}

#[test]
fn storage_heuristics_do_not_leak_matryoshka_internals() {
    let temp = tempfile::tempdir().unwrap();
    let repo_root = temp.path();
    fs::create_dir_all(repo_root.join("src/oauth")).unwrap();
    fs::write(
        repo_root.join("src/oauth/credential_store.rs"),
        "pub struct CredentialStore;\nimpl CredentialStore {\n    pub fn save_token(&self) {}\n}\n",
    )
    .unwrap();

    let db_path = temp.path().join("index.db");
    let store = MatryoshkaStore::open(&db_path).unwrap();
    let indexer = FullIndexer::new(
                store.clone(),
                HeuristicEnricher,
                DeterministicEmbedder::default(),
                HeuristicChunkSummarizer,
            );

    indexer.index_repo(repo_root).unwrap();

    let file_card = store
        .load_file_card("src/oauth/credential_store.rs")
        .unwrap()
        .unwrap();
    let folder_card = store.load_folder_card("src/oauth").unwrap().unwrap();
    let card_text = format!(
        "{}\n{}",
        serde_json::to_string(&file_card).unwrap(),
        serde_json::to_string(&folder_card).unwrap()
    );

    assert!(!card_text.contains("semantic records"), "{card_text}");
    assert!(!card_text.contains("facts, cards"), "{card_text}");
    assert!(
        !card_text.contains("persistent storage and retrieval behavior"),
        "{card_text}"
    );
}

#[test]
fn rebuild_semantic_recovers_search_without_full_reindex() {
    let repo_root = std::path::PathBuf::from("tests/fixtures/mini_repo");
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("index.db");
    let store = MatryoshkaStore::open(&db_path).unwrap();
    let indexer = FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default(), HeuristicChunkSummarizer);

    indexer.index_repo(&repo_root).unwrap();

    let store = MatryoshkaStore::open(&db_path).unwrap();
    store.replace_semantic_records(&[]).unwrap();
    assert!(store.load_all_semantic_records().unwrap().is_empty());

    let summary = indexer.rebuild_semantic_index(&repo_root).unwrap();
    assert!(summary.semantic_record_count >= 6);
    assert!(summary.file_card_record_count >= 2);
    assert!(summary.folder_card_record_count >= 1);
    assert_eq!(summary.repo_card_record_count, 1);

    let store = MatryoshkaStore::open(&db_path).unwrap();
    let records = store.load_all_semantic_records().unwrap();
    assert!(records.iter().any(|record| {
        record.record_id.starts_with("semantic:file_card:")
            || record.record_id.starts_with("semantic:folder_card:")
            || record.record_id.starts_with("semantic:repo_card:")
    }));
    assert!(records.iter().any(|record| {
        matches!(
            record.entity_type,
            matryoshka_core_ir::SemanticEntityType::Snippet
                | matryoshka_core_ir::SemanticEntityType::Symbol
        ) && record
            .embedding
            .as_ref()
            .is_some_and(|embedding| !embedding.is_empty())
    }));

    let search = SearchEngine::new(store, DeterministicEmbedder::default());
    let hits = search.search("repository architecture", 5).unwrap();
    assert!(hits.iter().any(|hit| {
        matches!(
            hit.entity_type,
            matryoshka_core_ir::SemanticEntityType::Repo
        )
    }));
}

#[test]
fn index_repo_with_progress_emits_real_events() {
    let repo_root = std::path::PathBuf::from("tests/fixtures/mini_repo");
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("index.db");
    let store = MatryoshkaStore::open(&db_path).unwrap();
    let indexer = FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default(), HeuristicChunkSummarizer);
    let mut events = Vec::new();

    let summary = indexer
        .index_repo_with_progress(&repo_root, |event| events.push(event))
        .unwrap();

    assert!(matches!(
        events.first(),
        Some(MatryoshkaProgressEvent::Started { .. })
    ));
    assert!(
        events
            .iter()
            .any(|event| { matches!(event, MatryoshkaProgressEvent::DiscoveringFiles) })
    );
    assert!(events.iter().any(|event| {
        matches!(
            event,
            MatryoshkaProgressEvent::FilesDiscovered { total_files: 2 }
        )
    }));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, MatryoshkaProgressEvent::ParsingFile { .. }))
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, MatryoshkaProgressEvent::ParsedFile { .. }))
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, MatryoshkaProgressEvent::EnrichingFile { .. }))
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, MatryoshkaProgressEvent::EnrichedFile { .. }))
            .count(),
        2
    );
    assert!(
        events
            .iter()
            .any(|event| { matches!(event, MatryoshkaProgressEvent::EmbeddingBatch { .. }) })
    );
    assert!(
        events
            .iter()
            .any(|event| { matches!(event, MatryoshkaProgressEvent::EmbeddedBatch { .. }) })
    );
    assert!(
        events
            .iter()
            .any(|event| { matches!(event, MatryoshkaProgressEvent::WritingDatabase { .. }) })
    );
    assert!(matches!(
        events.last(),
        Some(MatryoshkaProgressEvent::Completed {
            file_count,
            semantic_record_count,
            ..
        }) if *file_count == summary.file_count && *semantic_record_count == summary.semantic_record_count
    ));
}

#[test]
fn index_repo_with_progress_emits_failed_event() {
    let repo_root = std::env::temp_dir().join("matryoshka-missing-progress-fixture");
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("index.db");
    let store = MatryoshkaStore::open(&db_path).unwrap();
    let indexer = FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default(), HeuristicChunkSummarizer);
    let mut events = Vec::new();

    let err = indexer
        .index_repo_with_progress(&repo_root, |event| events.push(event))
        .unwrap_err();

    assert!(format!("{err:#}").contains("No such file"));
    assert!(matches!(
        events.first(),
        Some(MatryoshkaProgressEvent::Started { .. })
    ));
    assert!(
        events
            .iter()
            .any(|event| { matches!(event, MatryoshkaProgressEvent::DiscoveringFiles) })
    );
    assert!(matches!(
        events.last(),
        Some(MatryoshkaProgressEvent::Failed { stage, .. }) if stage == "parsing"
    ));
}

#[test]
fn rebuild_semantic_with_progress_emits_batch_events() {
    let repo_root = std::path::PathBuf::from("tests/fixtures/mini_repo");
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("index.db");
    let store = MatryoshkaStore::open(&db_path).unwrap();
    let indexer = FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default(), HeuristicChunkSummarizer);

    indexer.index_repo(&repo_root).unwrap();

    let store = MatryoshkaStore::open(&db_path).unwrap();
    store.replace_semantic_records(&[]).unwrap();

    let mut events = Vec::new();
    let summary = indexer
        .rebuild_semantic_index_with_progress(&repo_root, |event| events.push(event))
        .unwrap();

    assert!(matches!(
        events.first(),
        Some(MatryoshkaProgressEvent::Started { .. })
    ));
    assert!(
        events
            .iter()
            .any(|event| { matches!(event, MatryoshkaProgressEvent::EmbeddingBatch { .. }) })
    );
    assert!(
        events
            .iter()
            .any(|event| { matches!(event, MatryoshkaProgressEvent::EmbeddedBatch { .. }) })
    );
    assert!(
        events
            .iter()
            .any(|event| { matches!(event, MatryoshkaProgressEvent::WritingDatabase { .. }) })
    );
    assert!(matches!(
        events.last(),
        Some(MatryoshkaProgressEvent::Completed {
            semantic_record_count,
            ..
        }) if *semantic_record_count == summary.semantic_record_count
    ));
}

#[test]
fn update_repairs_missing_enriched_artifacts_after_interrupted_prewarm() {
    let repo_root = std::path::PathBuf::from("tests/fixtures/mini_repo");
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("index.db");
    let store = MatryoshkaStore::open(&db_path).unwrap();
    let indexer = FullIndexer::new(
                store.clone(),
                HeuristicEnricher,
                DeterministicEmbedder::default(),
                HeuristicChunkSummarizer,
            );

    indexer.index_repo(&repo_root).unwrap();
    store
        .connect()
        .unwrap()
        .execute_batch(
            "DELETE FROM file_cards; DELETE FROM folder_cards; DELETE FROM repo_cards; DELETE FROM semantic_records WHERE record_id LIKE 'semantic:%_card:%';",
        )
        .unwrap();

    assert_eq!(store.load_all_file_cards().unwrap().len(), 0);
    assert_eq!(store.load_all_folder_cards().unwrap().len(), 0);
    assert!(
        store
            .load_repo_card(&repo_root.to_string_lossy())
            .unwrap()
            .is_none()
    );

    let mut events = Vec::new();
    let summary = indexer
        .update_repo_with_progress(&repo_root, |event| events.push(event))
        .unwrap();

    assert_eq!(summary.changed_files, 0);
    assert!(summary.changed_folders > 0);
    assert!(summary.repo_card_updated);
    assert!(!store.load_all_file_cards().unwrap().is_empty());
    assert!(!store.load_all_folder_cards().unwrap().is_empty());
    assert!(
        store
            .load_repo_card(&repo_root.to_string_lossy())
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .load_all_semantic_records()
            .unwrap()
            .iter()
            .any(|record| record.record_id.starts_with("semantic:file_card:"))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaProgressEvent::EnrichingFile { .. }))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, MatryoshkaProgressEvent::EmbeddingBatch { .. }))
    );
    assert!(matches!(
        events.last(),
        Some(MatryoshkaProgressEvent::Completed {
            semantic_record_count,
            ..
        }) if *semantic_record_count == summary.semantic_record_count
    ));
}

#[test]
fn update_repairs_empty_non_heuristic_file_summaries() {
    let repo_root = std::path::PathBuf::from("tests/fixtures/mini_repo");
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("index.db");
    let store = MatryoshkaStore::open(&db_path).unwrap();
    let indexer = FullIndexer::new(
                store.clone(),
                HeuristicEnricher,
                DeterministicEmbedder::default(),
                HeuristicChunkSummarizer,
            );

    indexer.index_repo(&repo_root).unwrap();

    let mut card = store
        .load_file_card("src/config/env.py")
        .unwrap()
        .expect("expected fixture file card");
    card.summary.clear();
    card.role = "stale broken role".into();
    card.provenance.model = Some("mlx-empty-test".into());
    store.upsert_file_card(&card).unwrap();

    let mut events = Vec::new();
    let summary = indexer
        .update_repo_with_progress(&repo_root, |event| events.push(event))
        .unwrap();

    let repaired = store
        .load_file_card("src/config/env.py")
        .unwrap()
        .expect("expected repaired file card");
    assert_eq!(summary.changed_files, 0);
    assert_ne!(repaired.role, "stale broken role");
    assert_eq!(repaired.provenance.model.as_deref(), Some("heuristic"));
    assert!(
        events
            .iter()
            .any(|event| { matches!(event, MatryoshkaProgressEvent::ArtifactQuality { .. }) })
    );
}

#[test]
#[ignore = "requires a reachable local MLX endpoint and explicit env vars"]
fn real_mlx_progress_integration() {
    let base_url = std::env::var("MATRYOSHKA_MLX_BASE_URL")
        .expect("set MATRYOSHKA_MLX_BASE_URL to a reachable OpenAI-compatible MLX endpoint");
    let api_key = std::env::var("MATRYOSHKA_MLX_API_KEY")
        .expect("set MATRYOSHKA_MLX_API_KEY for the local MLX endpoint");
    let chat_model = std::env::var("MATRYOSHKA_MLX_CHAT_MODEL")
        .expect("set MATRYOSHKA_MLX_CHAT_MODEL to the enrichment model name");
    let embed_model = std::env::var("MATRYOSHKA_MLX_EMBED_MODEL")
        .expect("set MATRYOSHKA_MLX_EMBED_MODEL to the embedding model name");

    let repo_root = std::path::PathBuf::from("tests/fixtures/mini_repo");
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("index.db");
    let store = MatryoshkaStore::open(&db_path).unwrap();
    let indexer = FullIndexer::new(
        store,
        MlxChatEnricher::new(&base_url, &api_key).with_model(chat_model),
        EndpointEmbedder::new(&base_url, &api_key, embed_model),
                HeuristicChunkSummarizer,
    );
    let mut events = Vec::new();

    let summary = indexer
        .index_repo_with_progress(&repo_root, |event| events.push(event))
        .unwrap();

    assert!(summary.file_count >= 2);
    assert!(
        events
            .iter()
            .any(|event| { matches!(event, MatryoshkaProgressEvent::Completed { .. }) })
    );
}
