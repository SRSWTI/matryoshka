from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from matryoshka.models import LabelResult


class LabelCache:
    def __init__(self, cache_path: str | Path, *, commit_interval: int = 1) -> None:
        self._path = Path(cache_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._commit_interval = max(commit_interval, 1)
        self._pending_writes = 0
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS label_cache (
                cache_key TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> LabelResult | None:
        row = self._conn.execute("SELECT payload_json FROM label_cache WHERE cache_key = ?", (key,)).fetchone()
        if row is None:
            return None
        return LabelResult(**json.loads(row[0]))

    def set(self, key: str, value: LabelResult) -> None:
        self._conn.execute(
            "INSERT INTO label_cache(cache_key, payload_json) VALUES(?, ?) ON CONFLICT(cache_key) DO UPDATE SET payload_json = excluded.payload_json",
            (key, json.dumps(value.to_dict(), sort_keys=True)),
        )
        self._pending_writes += 1
        if self._pending_writes >= self._commit_interval:
            self.save()

    def save(self) -> None:
        self._conn.commit()
        self._pending_writes = 0

    def build_key(self, scope: str, target_id: str, messages: list[dict[str, str]], model: str) -> str:
        fingerprint = hashlib.sha256()
        fingerprint.update(scope.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(target_id.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(model.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(json.dumps(messages, sort_keys=True).encode("utf-8"))
        return fingerprint.hexdigest()