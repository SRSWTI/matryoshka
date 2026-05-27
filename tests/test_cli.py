from __future__ import annotations

import json
import sqlite3
import sys

from cradle import cli


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
            "cradle",
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