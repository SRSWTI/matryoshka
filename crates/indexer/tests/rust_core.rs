use matryoshka_embed_client::DeterministicEmbedder;
use matryoshka_enricher::HeuristicEnricher;
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

    let read = ReadApi::new(MatryoshkaStore::open(&db_path).unwrap(), repo_root);
    let card = read.read("src/auth/middleware.py").unwrap();
    assert_eq!(card.file.path, "src/auth/middleware.py");
    assert!(
        card.file_card
            .unwrap()
            .summary
            .contains("src/auth/middleware.py")
    );
    assert!(!card.imports.is_empty());

    let card_more = read.read_more("src/auth/middleware.py").unwrap();
    assert!(!card_more.symbol_blocks.is_empty());
    assert!(
        card_more
            .import_lines
            .iter()
            .any(|line| line.contains("src.config.env"))
    );
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
