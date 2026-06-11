use anyhow::{Context, Result};
use matryoshka_core_ir::{
    EdgeFact, FileCard, FileFact, FolderCard, FolderFact, RepoCard, RepositorySnapshot,
    SemanticRecord, SymbolFact,
};
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct MatryoshkaStore {
    db_path: PathBuf,
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
            "#,
        )?;
        Ok(())
    }

    pub fn replace_snapshot(&self, snapshot: &RepositorySnapshot) -> Result<()> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        tx.execute_batch(
            "DELETE FROM files; DELETE FROM folders; DELETE FROM symbols; DELETE FROM edges; DELETE FROM semantic_records;",
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
            tx.execute(
                "INSERT OR REPLACE INTO semantic_records(record_id, entity_id, entity_type, path, source_hash, payload_json) VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
                params![record.record_id, record.entity_id, format!("{:?}", record.entity_type), record.path, record.source_hash, to_json(record)?],
            )?;
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
            tx.execute(
                "INSERT OR REPLACE INTO semantic_records(record_id, entity_id, entity_type, path, source_hash, payload_json) VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
                params![record.record_id, record.entity_id, format!("{:?}", record.entity_type), record.path, record.source_hash, to_json(record)?],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn replace_semantic_records(&self, records: &[SemanticRecord]) -> Result<()> {
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        tx.execute("DELETE FROM semantic_records", [])?;
        for record in records {
            tx.execute(
                "INSERT OR REPLACE INTO semantic_records(record_id, entity_id, entity_type, path, source_hash, payload_json) VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
                params![record.record_id, record.entity_id, format!("{:?}", record.entity_type), record.path, record.source_hash, to_json(record)?],
            )?;
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

    pub fn load_all_file_cards(&self) -> Result<Vec<FileCard>> {
        self.load_all("SELECT payload_json FROM file_cards ORDER BY file_id")
    }

    pub fn load_all_folder_cards(&self) -> Result<Vec<FolderCard>> {
        self.load_all("SELECT payload_json FROM folder_cards ORDER BY folder_id")
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
        delete_many(
            &self.connect()?,
            "DELETE FROM semantic_records WHERE path = ?1",
            paths,
        )
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
