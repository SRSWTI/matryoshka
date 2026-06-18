use anyhow::{Context, Result};
use matryoshka_core_ir::{
    EdgeFact, FileCard, FileFact, FolderCard, FolderFact, LateInteractionVector, RepoCard,
    RepositorySnapshot, SemanticRecord, SymbolFact,
};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct MatryoshkaStore {
    db_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticFtsHit {
    pub record_id: String,
    pub rank: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalIndexStats {
    pub semantic_records: usize,
    pub embedded_records: usize,
    pub fts_records: usize,
    pub late_vector_rows: usize,
    pub records_with_late_vectors: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CardSummaryRow {
    pub card_type: String,
    pub id: String,
    pub summary: String,
    pub is_empty: bool,
}

impl MatryoshkaStore {
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self> {
        let db_path = db_path.as_ref().to_path_buf();
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let store = Self { db_path };
        store.initialize()?;
        Ok(store)
    }

    pub fn connect(&self) -> Result<Connection> {
        let conn = Connection::open(&self.db_path)?;
        conn.execute_batch("PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;")?;
        Ok(conn)
    }

    pub fn initialize(&self) -> Result<()> {
        let conn = self.connect()?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS files (
                file_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS folders (
                folder_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS symbols (
                symbol_id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edges (
                edge_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS file_cards (
                file_id TEXT PRIMARY KEY,
                source_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS folder_cards (
                folder_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS repo_cards (
                repo_root TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS semantic_records (
                record_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                path TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS semantic_records_fts USING fts5(
                record_id UNINDEXED,
                title,
                path,
                content,
                metadata_text,
                tokenize = "unicode61 tokenchars '_./-'"
            );
            CREATE TABLE IF NOT EXISTS semantic_late_vectors (
                record_id TEXT NOT NULL,
                token TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                weight REAL NOT NULL,
                embedding_json TEXT NOT NULL,
                PRIMARY KEY(record_id, ordinal)
            );
            CREATE TABLE IF NOT EXISTS invalidation_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT NOT NULL,
                entity_kind TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_semantic_entity ON semantic_records(entity_id);
            CREATE INDEX IF NOT EXISTS idx_late_vectors_record ON semantic_late_vectors(record_id);
            "#,
        )?;
        Ok(())
    }

    pub fn replace_snapshot(&self, snapshot: &RepositorySnapshot) -> Result<()> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        tx.execute_batch(
            "DELETE FROM files; DELETE FROM folders; DELETE FROM symbols; DELETE FROM edges; DELETE FROM semantic_records; DELETE FROM semantic_records_fts; DELETE FROM semantic_late_vectors;",
        )?;
        tx.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('repo_root', ?1)",
            [snapshot.repo_root.as_str()],
        )?;
        tx.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('indexed_at', ?1)",
            [snapshot.indexed_at.to_rfc3339()],
        )?;

        for file in &snapshot.files {
            tx.execute(
                "INSERT OR REPLACE INTO files(file_id, path, source_hash, payload_json) VALUES(?1, ?2, ?3, ?4)",
                params![file.file_id, file.path, file.source_hash, to_json(file)?],
            )?;
        }
        for folder in &snapshot.folders {
            tx.execute(
                "INSERT OR REPLACE INTO folders(folder_id, path, payload_json) VALUES(?1, ?2, ?3)",
                params![folder.folder_id, folder.path, to_json(folder)?],
            )?;
        }
        for symbol in &snapshot.symbols {
            tx.execute(
                "INSERT OR REPLACE INTO symbols(symbol_id, file_id, name, path, payload_json) VALUES(?1, ?2, ?3, ?4, ?5)",
                params![symbol.symbol_id, symbol.file_id, symbol.name, symbol.path, to_json(symbol)?],
            )?;
        }
        for edge in &snapshot.edges {
            tx.execute(
                "INSERT OR REPLACE INTO edges(edge_id, source_id, target_id, kind, payload_json) VALUES(?1, ?2, ?3, ?4, ?5)",
                params![edge.edge_id, edge.source_id, edge.target_id, format!("{:?}", edge.kind), to_json(edge)?],
            )?;
        }
        for record in &snapshot.semantic_records {
            upsert_semantic_record_tx(&tx, record)?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn upsert_file_card(&self, card: &FileCard) -> Result<()> {
        let conn = self.connect()?;
        conn.execute(
            "INSERT OR REPLACE INTO file_cards(file_id, source_hash, payload_json) VALUES(?1, ?2, ?3)",
            params![card.file_id, card.provenance.source_hash, to_json(card)?],
        )?;
        Ok(())
    }

    pub fn upsert_folder_card(&self, card: &FolderCard) -> Result<()> {
        let conn = self.connect()?;
        conn.execute(
            "INSERT OR REPLACE INTO folder_cards(folder_id, payload_json) VALUES(?1, ?2)",
            params![card.folder_id, to_json(card)?],
        )?;
        Ok(())
    }

    pub fn upsert_repo_card(&self, card: &RepoCard) -> Result<()> {
        let conn = self.connect()?;
        conn.execute(
            "INSERT OR REPLACE INTO repo_cards(repo_root, payload_json) VALUES(?1, ?2)",
            params![card.repo_root, to_json(card)?],
        )?;
        Ok(())
    }

    pub fn upsert_semantic_records(&self, records: &[SemanticRecord]) -> Result<()> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        for record in records {
            upsert_semantic_record_tx(&tx, record)?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn replace_semantic_records(&self, records: &[SemanticRecord]) -> Result<()> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM semantic_records", [])?;
        tx.execute("DELETE FROM semantic_records_fts", [])?;
        tx.execute("DELETE FROM semantic_late_vectors", [])?;
        for record in records {
            upsert_semantic_record_tx(&tx, record)?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn load_file(&self, file_id: &str) -> Result<Option<FileFact>> {
        self.load_one("SELECT payload_json FROM files WHERE file_id = ?1", file_id)
    }

    pub fn load_file_card(&self, file_id: &str) -> Result<Option<FileCard>> {
        self.load_one(
            "SELECT payload_json FROM file_cards WHERE file_id = ?1",
            file_id,
        )
    }

    pub fn load_folder_card(&self, folder_id: &str) -> Result<Option<FolderCard>> {
        self.load_one(
            "SELECT payload_json FROM folder_cards WHERE folder_id = ?1",
            folder_id,
        )
    }

    pub fn load_repo_card(&self, repo_root: &str) -> Result<Option<RepoCard>> {
        self.load_one(
            "SELECT payload_json FROM repo_cards WHERE repo_root = ?1",
            repo_root,
        )
    }

    pub fn load_symbols_for_file(&self, file_id: &str) -> Result<Vec<SymbolFact>> {
        self.load_many(
            "SELECT payload_json FROM symbols WHERE file_id = ?1 ORDER BY symbol_id",
            file_id,
        )
    }

    pub fn load_edges_for_entity(&self, entity_id: &str) -> Result<(Vec<EdgeFact>, Vec<EdgeFact>)> {
        let conn = self.connect()?;
        let outgoing = query_json_many(
            &conn,
            "SELECT payload_json FROM edges WHERE source_id = ?1 ORDER BY target_id",
            entity_id,
        )?;
        let incoming = query_json_many(
            &conn,
            "SELECT payload_json FROM edges WHERE target_id = ?1 ORDER BY source_id",
            entity_id,
        )?;
        Ok((incoming, outgoing))
    }

    pub fn load_all_files(&self) -> Result<Vec<FileFact>> {
        self.load_all("SELECT payload_json FROM files ORDER BY path")
    }

    pub fn load_all_folders(&self) -> Result<Vec<FolderFact>> {
        self.load_all("SELECT payload_json FROM folders ORDER BY path")
    }

    pub fn load_all_symbols(&self) -> Result<Vec<SymbolFact>> {
        self.load_all("SELECT payload_json FROM symbols ORDER BY path, name")
    }

    pub fn load_all_edges(&self) -> Result<Vec<EdgeFact>> {
        self.load_all("SELECT payload_json FROM edges ORDER BY edge_id")
    }

    pub fn load_all_semantic_records(&self) -> Result<Vec<SemanticRecord>> {
        self.load_all("SELECT payload_json FROM semantic_records ORDER BY record_id")
    }

    pub fn load_semantic_records_by_ids(
        &self,
        record_ids: &[String],
    ) -> Result<Vec<SemanticRecord>> {
        let conn = self.connect()?;
        let mut records = Vec::new();
        let mut stmt =
            conn.prepare("SELECT payload_json FROM semantic_records WHERE record_id = ?1")?;
        for record_id in record_ids {
            let payload = stmt
                .query_row([record_id], |row| row.get::<_, String>(0))
                .optional()
                .with_context(|| format!("failed to load semantic record {record_id}"))?;
            if let Some(payload) = payload {
                records.push(from_json(&payload)?);
            }
        }
        Ok(records)
    }

    pub fn search_semantic_fts(&self, query: &str, limit: usize) -> Result<Vec<SemanticFtsHit>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let Some(match_query) = semantic_fts_query(query) else {
            return Ok(Vec::new());
        };
        let conn = self.connect()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT record_id, bm25(semantic_records_fts) AS rank
            FROM semantic_records_fts
            WHERE semantic_records_fts MATCH ?1
            ORDER BY rank
            LIMIT ?2
            "#,
        )?;
        let rows = stmt.query_map(params![match_query, limit as i64], |row| {
            Ok(SemanticFtsHit {
                record_id: row.get(0)?,
                rank: row.get::<_, f64>(1)? as f32,
            })
        })?;
        let mut hits = Vec::new();
        for row in rows {
            hits.push(row?);
        }
        Ok(hits)
    }

    pub fn rebuild_semantic_fts(&self) -> Result<usize> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM semantic_records_fts", [])?;
        let mut count = 0usize;
        {
            let mut stmt =
                tx.prepare("SELECT payload_json FROM semantic_records ORDER BY record_id")?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
            for row in rows {
                let record: SemanticRecord = from_json(&row?)?;
                upsert_semantic_fts_tx(&tx, &record)?;
                count += 1;
            }
        }
        tx.commit()?;
        Ok(count)
    }

    pub fn retrieval_index_stats(&self) -> Result<RetrievalIndexStats> {
        let conn = self.connect()?;
        let semantic_records = count_rows(&conn, "SELECT COUNT(*) FROM semantic_records")?;
        let fts_records = count_rows(&conn, "SELECT COUNT(*) FROM semantic_records_fts")?;
        let late_vector_rows = count_rows(&conn, "SELECT COUNT(*) FROM semantic_late_vectors")?;
        let records_with_late_vectors = count_rows(
            &conn,
            "SELECT COUNT(DISTINCT record_id) FROM semantic_late_vectors",
        )?;
        let embedded_records = self
            .load_all_semantic_records()?
            .iter()
            .filter(|record| record.embedding.is_some())
            .count();
        Ok(RetrievalIndexStats {
            semantic_records,
            embedded_records,
            fts_records,
            late_vector_rows,
            records_with_late_vectors,
        })
    }

    pub fn replace_late_interaction_vectors(
        &self,
        record_ids: &[String],
        vectors: &[LateInteractionVector],
    ) -> Result<()> {
        if record_ids.is_empty() && vectors.is_empty() {
            return Ok(());
        }
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        {
            let mut delete_stmt =
                tx.prepare("DELETE FROM semantic_late_vectors WHERE record_id = ?1")?;
            for record_id in record_ids {
                delete_stmt.execute([record_id])?;
            }
        }
        {
            let mut insert_stmt = tx.prepare(
                r#"
                INSERT OR REPLACE INTO semantic_late_vectors(record_id, token, ordinal, weight, embedding_json)
                VALUES(?1, ?2, ?3, ?4, ?5)
                "#,
            )?;
            for vector in vectors {
                insert_stmt.execute(params![
                    vector.record_id,
                    vector.token,
                    vector.ordinal as i64,
                    vector.weight,
                    to_json(&vector.embedding)?,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn load_late_interaction_vectors(
        &self,
        record_ids: &[String],
    ) -> Result<BTreeMap<String, Vec<LateInteractionVector>>> {
        if record_ids.is_empty() {
            return Ok(BTreeMap::new());
        }
        let conn = self.connect()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT token, ordinal, weight, embedding_json
            FROM semantic_late_vectors
            WHERE record_id = ?1
            ORDER BY ordinal
            "#,
        )?;
        let mut by_record = BTreeMap::<String, Vec<LateInteractionVector>>::new();
        for record_id in record_ids {
            let rows = stmt.query_map([record_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, f32>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })?;
            for row in rows {
                let (token, ordinal, weight, embedding_json) = row?;
                let embedding = from_json::<Vec<f32>>(&embedding_json)?;
                by_record
                    .entry(record_id.clone())
                    .or_default()
                    .push(LateInteractionVector {
                        record_id: record_id.clone(),
                        token,
                        ordinal: ordinal as usize,
                        weight,
                        embedding,
                    });
            }
        }
        Ok(by_record)
    }

    pub fn load_all_file_cards(&self) -> Result<Vec<FileCard>> {
        self.load_all("SELECT payload_json FROM file_cards ORDER BY file_id")
    }

    pub fn load_all_folder_cards(&self) -> Result<Vec<FolderCard>> {
        self.load_all("SELECT payload_json FROM folder_cards ORDER BY folder_id")
    }

    pub fn load_card_summaries(&self) -> Result<Vec<CardSummaryRow>> {
        let conn = self.connect()?;
        let mut rows = Vec::new();
        load_card_summary_rows(&conn, &mut rows, "file_cards", "file_id", "file")?;
        load_card_summary_rows(&conn, &mut rows, "folder_cards", "folder_id", "folder")?;
        load_card_summary_rows(&conn, &mut rows, "repo_cards", "repo_root", "repo")?;
        Ok(rows)
    }

    pub fn load_repo_root(&self) -> Result<Option<String>> {
        let conn = self.connect()?;
        conn.query_row(
            "SELECT value FROM meta WHERE key = 'repo_root'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .context("failed to load repo_root from sqlite")
    }

    pub fn mark_stale(&self, entity_kind: &str, entity_id: &str, reason: &str) -> Result<()> {
        let conn = self.connect()?;
        conn.execute(
            "INSERT INTO invalidation_queue(entity_id, entity_kind, reason) VALUES(?1, ?2, ?3)",
            params![entity_id, entity_kind, reason],
        )?;
        Ok(())
    }

    pub fn clear_invalidation_queue(&self) -> Result<()> {
        let conn = self.connect()?;
        conn.execute("DELETE FROM invalidation_queue", [])?;
        Ok(())
    }

    pub fn delete_files(&self, file_ids: &[String]) -> Result<()> {
        delete_many(
            &self.connect()?,
            "DELETE FROM files WHERE file_id = ?1",
            file_ids,
        )
    }

    pub fn delete_folders(&self, folder_ids: &[String]) -> Result<()> {
        delete_many(
            &self.connect()?,
            "DELETE FROM folders WHERE folder_id = ?1",
            folder_ids,
        )
    }

    pub fn delete_file_cards(&self, file_ids: &[String]) -> Result<()> {
        delete_many(
            &self.connect()?,
            "DELETE FROM file_cards WHERE file_id = ?1",
            file_ids,
        )
    }

    pub fn delete_folder_cards(&self, folder_ids: &[String]) -> Result<()> {
        delete_many(
            &self.connect()?,
            "DELETE FROM folder_cards WHERE folder_id = ?1",
            folder_ids,
        )
    }

    pub fn delete_symbols_for_files(&self, file_ids: &[String]) -> Result<()> {
        delete_many(
            &self.connect()?,
            "DELETE FROM symbols WHERE file_id = ?1",
            file_ids,
        )
    }

    pub fn delete_edges_for_entities(&self, entity_ids: &[String]) -> Result<()> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        {
            let mut stmt =
                tx.prepare("DELETE FROM edges WHERE source_id = ?1 OR target_id = ?1")?;
            for entity_id in entity_ids {
                stmt.execute([entity_id])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn delete_semantic_records_for_paths(&self, paths: &[String]) -> Result<()> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        {
            let mut late_stmt = tx.prepare(
                "DELETE FROM semantic_late_vectors WHERE record_id IN (SELECT record_id FROM semantic_records WHERE path = ?1)",
            )?;
            let mut record_stmt = tx.prepare("DELETE FROM semantic_records WHERE path = ?1")?;
            let mut fts_stmt = tx.prepare("DELETE FROM semantic_records_fts WHERE path = ?1")?;
            for path in paths {
                late_stmt.execute([path])?;
                record_stmt.execute([path])?;
                fts_stmt.execute([path])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn upsert_files(&self, files: &[FileFact]) -> Result<()> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO files(file_id, path, source_hash, payload_json) VALUES(?1, ?2, ?3, ?4)",
            )?;
            for file in files {
                stmt.execute(params![
                    file.file_id,
                    file.path,
                    file.source_hash,
                    to_json(file)?
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn upsert_folders(&self, folders: &[FolderFact]) -> Result<()> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO folders(folder_id, path, payload_json) VALUES(?1, ?2, ?3)",
            )?;
            for folder in folders {
                stmt.execute(params![folder.folder_id, folder.path, to_json(folder)?])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn upsert_symbols(&self, symbols: &[SymbolFact]) -> Result<()> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO symbols(symbol_id, file_id, name, path, payload_json) VALUES(?1, ?2, ?3, ?4, ?5)",
            )?;
            for symbol in symbols {
                stmt.execute(params![
                    symbol.symbol_id,
                    symbol.file_id,
                    symbol.name,
                    symbol.path,
                    to_json(symbol)?
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn upsert_edges(&self, edges: &[EdgeFact]) -> Result<()> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO edges(edge_id, source_id, target_id, kind, payload_json) VALUES(?1, ?2, ?3, ?4, ?5)",
            )?;
            for edge in edges {
                stmt.execute(params![
                    edge.edge_id,
                    edge.source_id,
                    edge.target_id,
                    format!("{:?}", edge.kind),
                    to_json(edge)?
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn load_one<T: DeserializeOwned>(&self, sql: &str, value: &str) -> Result<Option<T>> {
        let conn = self.connect()?;
        let payload: Option<String> = conn
            .query_row(sql, [value], |row| row.get(0))
            .optional()
            .with_context(|| format!("failed to query sqlite for {value}"))?;
        payload.map(|json| from_json(&json)).transpose()
    }

    fn load_many<T: DeserializeOwned>(&self, sql: &str, value: &str) -> Result<Vec<T>> {
        let conn = self.connect()?;
        query_json_many(&conn, sql, value)
    }

    fn load_all<T: DeserializeOwned>(&self, sql: &str) -> Result<Vec<T>> {
        let conn = self.connect()?;
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut values = Vec::new();
        for row in rows {
            values.push(from_json(&row?)?);
        }
        Ok(values)
    }
}

fn query_json_many<T: DeserializeOwned>(
    conn: &Connection,
    sql: &str,
    value: &str,
) -> Result<Vec<T>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map([value], |row| row.get::<_, String>(0))?;
    let mut values = Vec::new();
    for row in rows {
        values.push(from_json(&row?)?);
    }
    Ok(values)
}

fn load_card_summary_rows(
    conn: &Connection,
    rows: &mut Vec<CardSummaryRow>,
    table: &str,
    id_column: &str,
    card_type: &str,
) -> Result<()> {
    if !table_exists(conn, table)? {
        return Ok(());
    }
    let sql = format!(
        r#"
        SELECT {id_column}, COALESCE(json_extract(payload_json, '$.summary'), '')
        FROM {table}
        ORDER BY {id_column}
        "#
    );
    let mut stmt = conn.prepare(&sql)?;
    let mapped = stmt.query_map([], |row| {
        let id = row.get::<_, String>(0)?;
        let summary = row.get::<_, String>(1)?;
        Ok(CardSummaryRow {
            card_type: card_type.to_string(),
            id,
            is_empty: summary.trim().is_empty(),
            summary,
        })
    })?;
    for row in mapped {
        rows.push(row?);
    }
    Ok(())
}

fn table_exists(conn: &Connection, table: &str) -> Result<bool> {
    conn.query_row(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?1 LIMIT 1",
        [table],
        |_| Ok(()),
    )
    .optional()
    .map(|value| value.is_some())
    .context("failed to inspect sqlite schema")
}

fn to_json<T: Serialize>(value: &T) -> Result<String> {
    serde_json::to_string(value).context("failed to serialize sqlite payload")
}

fn from_json<T: DeserializeOwned>(value: &str) -> Result<T> {
    serde_json::from_str(value).context("failed to deserialize sqlite payload")
}

fn delete_many(conn: &Connection, sql: &str, values: &[String]) -> Result<()> {
    let mut stmt = conn.prepare(sql)?;
    for value in values {
        stmt.execute([value])?;
    }
    Ok(())
}

fn count_rows(conn: &Connection, sql: &str) -> Result<usize> {
    let count = conn.query_row(sql, [], |row| row.get::<_, i64>(0))?;
    Ok(count.max(0) as usize)
}

fn upsert_semantic_record_tx(
    tx: &rusqlite::Transaction<'_>,
    record: &SemanticRecord,
) -> Result<()> {
    tx.execute(
        "INSERT OR REPLACE INTO semantic_records(record_id, entity_id, entity_type, path, source_hash, payload_json) VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
        params![record.record_id, record.entity_id, format!("{:?}", record.entity_type), record.path, record.source_hash, to_json(record)?],
    )?;
    upsert_semantic_fts_tx(tx, record)?;
    Ok(())
}

fn upsert_semantic_fts_tx(tx: &rusqlite::Transaction<'_>, record: &SemanticRecord) -> Result<()> {
    tx.execute(
        "DELETE FROM semantic_records_fts WHERE record_id = ?1",
        params![record.record_id],
    )?;
    tx.execute(
        "INSERT INTO semantic_records_fts(record_id, title, path, content, metadata_text) VALUES(?1, ?2, ?3, ?4, ?5)",
        params![
            record.record_id,
            record.title,
            record.path,
            record.content,
            semantic_metadata_text(&record.metadata),
        ],
    )?;
    Ok(())
}

fn semantic_metadata_text(metadata: &std::collections::BTreeMap<String, Value>) -> String {
    fn collect(value: &Value, out: &mut Vec<String>) {
        match value {
            Value::String(text) => out.push(text.clone()),
            Value::Array(items) => {
                for item in items {
                    collect(item, out);
                }
            }
            Value::Object(map) => {
                for (key, value) in map {
                    out.push(key.clone());
                    collect(value, out);
                }
            }
            Value::Number(number) => out.push(number.to_string()),
            Value::Bool(value) => out.push(value.to_string()),
            Value::Null => {}
        }
    }

    let mut parts = Vec::new();
    for (key, value) in metadata {
        parts.push(key.clone());
        collect(value, &mut parts);
    }
    parts.join(" ")
}

fn semantic_fts_query(query: &str) -> Option<String> {
    let mut terms = Vec::new();
    for token in query.split(|ch: char| {
        !(ch.is_alphanumeric() || ch == '_' || ch == '/' || ch == '.' || ch == '-')
    }) {
        let token = token.trim_matches(|ch: char| {
            !(ch.is_alphanumeric() || ch == '_' || ch == '/' || ch == '.' || ch == '-')
        });
        if token.len() < 2 {
            continue;
        }
        let escaped = token.replace('"', "\"\"");
        terms.push(format!("\"{escaped}\""));
    }
    terms.sort();
    terms.dedup();
    (!terms.is_empty()).then(|| terms.join(" OR "))
}
