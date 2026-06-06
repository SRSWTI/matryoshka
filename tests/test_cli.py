from __future__ import annotations

import json
import sqlite3
import sys
import hashlib
import re

import numpy as np

from matryoshka import cli


class FakeClient:
    def __init__(self, config):
        self.model = config.model

    def create_many_chat_completions(self, messages_list, *, temperature, max_tokens, progress_callback=None, result_callback=None):
        responses = []
        total = len(messages_list)
        for index, messages in enumerate(messages_list, start=1):
            response = self._respond(json.loads(messages[1]["content"]))
            responses.append(response)
            if result_callback is not None:
                result_callback(index - 1, response, index, total)
            if progress_callback is not None:
                progress_callback(index, total)
        return responses

    def create_chat_completion(self, messages, *, temperature, max_tokens):
        return self._respond(json.loads(messages[1]["content"]))

    def _respond(self, payload):
        if "file_packet" in payload:
            return {
                "summary": "Authentication middleware file.",
                "description": "Handles token verification.",
                "tags": ["auth", "jwt"],
                "categories": ["authentication"],
                "confidence": 0.9,
                "evidence": [payload["file_packet"]["path"]],
            }
        return {
            "summary": "Repository summary.",
            "description": "Groups analyzed files.",
            "tags": ["tooling"],
            "categories": ["developer-tooling"],
            "confidence": 0.85,
            "evidence": [payload.get("task", "")],
        }


class FakeEmbedder:
    def __init__(self, model_name="fake-embedder", batch_size=32, truncate_dim=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = truncate_dim or 48

    def encode(self, texts, *, show_progress_bar=False):
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9_]+", text.lower()):
                token_hash = hashlib.sha256(token.encode("utf-8")).digest()
                column = int.from_bytes(token_hash[:4], byteorder="little") % self.dimension
                vectors[row_index, column] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def test_cli_analyze_writes_output_and_summary(tmp_path, monkeypatch, capsys):
    auth_dir = tmp_path / "src" / "auth"
    auth_dir.mkdir(parents=True)
    (auth_dir / "middleware.py").write_text(
        """
import jwt

def verify_token(token: str) -> bool:
    return jwt.decode(token, "secret", algorithms=["HS256"]) is not None
""".strip(),
        encoding="utf-8",
    )

    output_path = tmp_path / "report.db"
    cache_path = tmp_path / "labels.db"
    monkeypatch.setattr(cli, "OpenAICompatibleClient", FakeClient)
    monkeypatch.setattr(
        cli,
        "main",
        cli.main,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matryoshka",
            "analyze",
            str(tmp_path),
            "--model",
            "fake-model",
            "--api-key",
            "2508",
            "--cache-path",
            str(cache_path),
            "--output",
            str(output_path),
            "--max-parallel-requests",
            "1",
            "--max-tokens",
            "120",
            "--thinking-budget",
            "0",
            "--max-files",
            "1",
        ],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()

    conn = sqlite3.connect(output_path)
    file_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE kind = 'file'").fetchone()[0]
    repo_summary = conn.execute("SELECT summary FROM nodes WHERE node_id = 'repo'").fetchone()[0]
    conn.close()

    assert exit_code == 0
    assert "progress: collected 1 source files" in captured.out
    assert f"database: {output_path}" in captured.out
    assert "files: 1" in captured.out
    assert file_count == 1
    assert repo_summary


def test_cli_analyze_excludes_paths_and_extensions(tmp_path, monkeypatch, capsys):
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "keep.py").write_text(
        """
def keep_me() -> bool:
    return True
""".strip(),
        encoding="utf-8",
    )

    excluded_dir = tmp_path / "tests"
    excluded_dir.mkdir(parents=True)
    (excluded_dir / "test_keep.py").write_text(
        """
def test_keep_me() -> None:
    assert True
""".strip(),
        encoding="utf-8",
    )

    (src_dir / "skip.ts").write_text(
        """
export function skipMe(): boolean {
  return true;
}
""".strip(),
        encoding="utf-8",
    )

    output_path = tmp_path / "report.db"
    cache_path = tmp_path / "labels.db"
    monkeypatch.setattr(cli, "OpenAICompatibleClient", FakeClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matryoshka",
            "analyze",
            str(tmp_path),
            "--model",
            "fake-model",
            "--api-key",
            "2508",
            "--cache-path",
            str(cache_path),
            "--output",
            str(output_path),
            "--max-parallel-requests",
            "1",
            "--max-tokens",
            "120",
            "--thinking-budget",
            "0",
            "--exclude-path",
            "tests",
            "--exclude-extension",
            ".ts",
        ],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()

    conn = sqlite3.connect(output_path)
    paths = {
        row[0]
        for row in conn.execute("SELECT path FROM nodes WHERE kind = 'file'").fetchall()
    }
    conn.close()

    assert exit_code == 0
    assert "progress: collected 1 source files" in captured.out
    assert paths == {"src/keep.py"}


def test_cli_analyze_uses_repo_named_default_db_path(tmp_path, monkeypatch, capsys):
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "main.py").write_text(
        """
def run() -> None:
    return None
""".strip(),
        encoding="utf-8",
    )

    cache_path = tmp_path / "labels.db"
    expected_db_path = tmp_path / ".matryoshka" / f"{tmp_path.name}.db"
    monkeypatch.setattr(cli, "OpenAICompatibleClient", FakeClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matryoshka",
            "analyze",
            str(tmp_path),
            "--model",
            "fake-model",
            "--api-key",
            "2508",
            "--cache-path",
            str(cache_path),
            "--max-parallel-requests",
            "1",
            "--max-tokens",
            "120",
            "--thinking-budget",
            "0",
            "--max-files",
            "1",
        ],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert expected_db_path.exists()
    assert f"database: {expected_db_path}" in captured.out


def test_cli_visualize_db_writes_markdown_report(tmp_path, monkeypatch, capsys):
    auth_dir = tmp_path / "src" / "auth"
    auth_dir.mkdir(parents=True)
    (auth_dir / "middleware.py").write_text(
        """
import jwt


def verify_token(token: str) -> bool:
    return jwt.decode(token, "secret", algorithms=["HS256"]) is not None
""".strip(),
        encoding="utf-8",
    )

    output_path = tmp_path / "report.db"
    cache_path = tmp_path / "labels.db"
    report_path = tmp_path / "report.md"
    monkeypatch.setattr(cli, "OpenAICompatibleClient", FakeClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matryoshka",
            "analyze",
            str(tmp_path),
            "--model",
            "fake-model",
            "--api-key",
            "2508",
            "--cache-path",
            str(cache_path),
            "--output",
            str(output_path),
            "--max-parallel-requests",
            "1",
            "--max-tokens",
            "120",
            "--thinking-budget",
            "0",
            "--max-files",
            "1",
        ],
    )
    assert cli.main() == 0
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matryoshka",
            "visualize-db",
            str(output_path),
            "--output",
            str(report_path),
            "--sample-limit",
            "5",
        ],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()
    report = report_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert f"visualization: {report_path}" in captured.out
    assert "# Matryoshka DB Visualization" in report
    assert "## Table Counts" in report
    assert "## SQL Schema" in report
    assert "## Sample Stored Rows" in report
    assert "### `nodes`" in report
    assert '"node_id"' in report
    assert "```mermaid" in report
    assert "| nodes |" in report


def test_cli_semantic_index_and_search(tmp_path, monkeypatch, capsys):
    auth_dir = tmp_path / "src" / "auth"
    auth_dir.mkdir(parents=True)
    (auth_dir / "middleware.py").write_text(
        """
import jwt


def verify_token(token: str) -> bool:
    return jwt.decode(token, "secret", algorithms=["HS256"]) is not None
""".strip(),
        encoding="utf-8",
    )

    output_path = tmp_path / "report.db"
    cache_path = tmp_path / "labels.db"
    index_dir = tmp_path / "semantic"
    monkeypatch.setattr(cli, "OpenAICompatibleClient", FakeClient)
    monkeypatch.setattr(cli, "build_text_embedder", lambda *args, **kwargs: FakeEmbedder())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matryoshka",
            "analyze",
            str(tmp_path),
            "--model",
            "fake-model",
            "--api-key",
            "2508",
            "--cache-path",
            str(cache_path),
            "--output",
            str(output_path),
            "--max-parallel-requests",
            "1",
            "--max-tokens",
            "120",
            "--thinking-budget",
            "0",
            "--max-files",
            "1",
        ],
    )
    assert cli.main() == 0
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matryoshka",
            "semantic-index",
            str(output_path),
            "--model",
            "fake-embedder",
            "--output-dir",
            str(index_dir),
        ],
    )
    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"semantic_index: {index_dir}" in captured.out

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matryoshka",
            "semantic-search",
            str(output_path),
            "where is verify_token implemented",
            "--index-dir",
            str(index_dir),
            "--limit",
            "3",
        ],
    )

    exit_code = cli.main()
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "src/auth/middleware.py" in captured.out


def test_cli_exact_search_commands(tmp_path, monkeypatch, capsys):
    auth_dir = tmp_path / "src" / "auth"
    shared_dir = tmp_path / "src" / "shared"
    providers_dir = tmp_path / "src" / "providers"
    auth_dir.mkdir(parents=True)
    shared_dir.mkdir(parents=True)
    providers_dir.mkdir(parents=True)

    (shared_dir / "crypto.py").write_text(
        """
def verify_sig(token: str) -> bool:
    return token.startswith("sig-")
""".strip(),
        encoding="utf-8",
    )
    (auth_dir / "middleware.py").write_text(
        """
from src.shared.crypto import verify_sig


def verify_token(token: str) -> bool:
    return verify_sig(token)
""".strip(),
        encoding="utf-8",
    )
    (providers_dir / "streaming.py").write_text(
        """
def stream_bedrock() -> str:
    return "ok"
""".strip(),
        encoding="utf-8",
    )
    (providers_dir / "adapter.py").write_text(
        """
from src.providers.streaming import stream_bedrock


def stream_bedrock_lazy() -> str:
    return stream_bedrock()
""".strip(),
        encoding="utf-8",
    )

    output_path = tmp_path / "report.db"
    cache_path = tmp_path / "labels.db"
    monkeypatch.setattr(cli, "OpenAICompatibleClient", FakeClient)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matryoshka",
            "analyze",
            str(tmp_path),
            "--model",
            "fake-model",
            "--api-key",
            "2508",
            "--cache-path",
            str(cache_path),
            "--output",
            str(output_path),
            "--max-parallel-requests",
            "1",
            "--max-tokens",
            "120",
            "--thinking-budget",
            "0",
            "--max-files",
            "4",
        ],
    )
    assert cli.main() == 0
    capsys.readouterr()

    monkeypatch.setattr(sys, "argv", ["matryoshka", "file-search", str(output_path), "middleware.py"])
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert "search_type: file" in captured.out
    assert "src/auth/middleware.py" in captured.out

    monkeypatch.setattr(sys, "argv", ["matryoshka", "symbol-search", str(output_path), "verify_sig"])
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert "search_type: symbol" in captured.out
    assert "verify_sig" in captured.out

    monkeypatch.setattr(sys, "argv", ["matryoshka", "import-search", str(output_path), "src.shared.crypto"])
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert "search_type: import" in captured.out
    assert "src/auth/middleware.py" in captured.out

    monkeypatch.setattr(sys, "argv", ["matryoshka", "call-search", str(output_path), "who calls stream_bedrock"])
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert "search_type: call" in captured.out
    assert "src/providers/adapter.py" in captured.out


def test_cli_question_command(tmp_path, monkeypatch, capsys):
    presentation_dir = tmp_path / "src" / "presentation"
    presentation_dir.mkdir(parents=True)
    (presentation_dir / "render.py").write_text(
        """
def decide_visible_output(user_role: str) -> str:
    return "admin panel" if user_role == "admin" else "basic panel"
""".strip(),
        encoding="utf-8",
    )

    output_path = tmp_path / "report.db"
    cache_path = tmp_path / "labels.db"
    index_dir = tmp_path / "semantic"
    monkeypatch.setattr(cli, "OpenAICompatibleClient", FakeClient)
    monkeypatch.setattr(cli, "build_text_embedder", lambda *args, **kwargs: FakeEmbedder())

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matryoshka",
            "analyze",
            str(tmp_path),
            "--model",
            "fake-model",
            "--api-key",
            "2508",
            "--cache-path",
            str(cache_path),
            "--output",
            str(output_path),
            "--max-parallel-requests",
            "1",
            "--max-tokens",
            "120",
            "--thinking-budget",
            "0",
            "--max-files",
            "1",
        ],
    )
    assert cli.main() == 0
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matryoshka",
            "semantic-index",
            str(output_path),
            "--model",
            "fake-embedder",
            "--output-dir",
            str(index_dir),
        ],
    )
    assert cli.main() == 0
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matryoshka",
            "question",
            str(output_path),
            "where does the system decide what to show the user",
            "--index-dir",
            str(index_dir),
        ],
    )
    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "src/presentation/render.py" in captured.out
    assert "decide_visible_output" in captured.out


def test_cli_visualize_focus_writes_symbol_neighborhood(tmp_path, monkeypatch, capsys):
    auth_dir = tmp_path / "src" / "auth"
    shared_dir = tmp_path / "src" / "shared"
    auth_dir.mkdir(parents=True)
    shared_dir.mkdir(parents=True)

    (shared_dir / "crypto.py").write_text(
        """
def verify_sig(token: str) -> bool:
    return token.startswith("sig-")
""".strip(),
        encoding="utf-8",
    )
    (auth_dir / "middleware.py").write_text(
        """
from src.shared.crypto import verify_sig


def verify_token(token: str) -> bool:
    return verify_sig(token)
""".strip(),
        encoding="utf-8",
    )

    output_path = tmp_path / "report.db"
    cache_path = tmp_path / "labels.db"
    report_path = tmp_path / "focus.md"
    monkeypatch.setattr(cli, "OpenAICompatibleClient", FakeClient)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matryoshka",
            "analyze",
            str(tmp_path),
            "--model",
            "fake-model",
            "--api-key",
            "2508",
            "--cache-path",
            str(cache_path),
            "--output",
            str(output_path),
            "--max-parallel-requests",
            "1",
            "--max-tokens",
            "120",
            "--thinking-budget",
            "0",
            "--max-files",
            "2",
        ],
    )
    assert cli.main() == 0
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "matryoshka",
            "visualize-focus",
            str(output_path),
            "verify_sig",
            "--kind",
            "symbol",
            "--output",
            str(report_path),
        ],
    )
    exit_code = cli.main()
    captured = capsys.readouterr()
    report = report_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert f"focus_visualization: {report_path}" in captured.out
    assert "## Symbol: `verify_sig`" in report
    assert "### Called By" in report
    assert "```mermaid" in report