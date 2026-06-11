use matryoshka_core_ir::MatryoshkaProgressEvent;
use matryoshka_embed_client::DeterministicEmbedder;
use matryoshka_embed_client::EndpointEmbedder;
use matryoshka_enricher::HeuristicEnricher;
use matryoshka_enricher::MlxChatEnricher;
use matryoshka_indexer::FullIndexer;
use matryoshka_read_api::ReadApi;
use matryoshka_search::SearchEngine;
use matryoshka_store_sqlite::MatryoshkaStore;
use std::fs;

#[test]
fn indexes_searches_and_reads_file_cards() {
    let repo_root = std::path::PathBuf::from("tests/fixtures/mini_repo");
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("index.db");
    let store = MatryoshkaStore::open(&db_path).unwrap();
    let indexer = FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default());

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
    assert!(hits.iter().any(|hit| {
        hit.path.contains("env.py")
            && hit
                .summary
                .as_deref()
                .is_some_and(|summary| summary.contains("env.py"))
            && hit
                .description
                .as_deref()
                .is_some_and(|description| description.contains("Role:"))
            && !hit.key_behaviors.is_empty()
    }));

    let read = ReadApi::new(MatryoshkaStore::open(&db_path).unwrap(), repo_root);
    let card = read.read("src/auth/middleware.py").unwrap();
    assert_eq!(card.file.path, "src/auth/middleware.py");
    assert!(
        card.summary
            .as_deref()
            .unwrap_or_default()
            .contains("src/auth/middleware.py")
    );
    assert!(!card.symbols.is_empty());
    assert!(!card.imports.is_empty());
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
    let indexer = FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default());
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
}

#[test]
fn rebuild_semantic_recovers_search_without_full_reindex() {
    let repo_root = std::path::PathBuf::from("tests/fixtures/mini_repo");
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("index.db");
    let store = MatryoshkaStore::open(&db_path).unwrap();
    let indexer = FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default());

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
    let indexer = FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default());
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
    let indexer = FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default());
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
    let indexer = FullIndexer::new(store, HeuristicEnricher, DeterministicEmbedder::default());

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
