from __future__ import annotations

import json
import sqlite3

from matryoshka.labeling import LabelingConfig, LabelingEngine
from matryoshka.pipeline import MatryoshkaPipeline, PipelineConfig
from matryoshka.retrieval import axe_retrieval
from matryoshka.storage import MatryoshkaDatabase


class FakeClient:
    def create_many_chat_completions(self, messages_list, *, temperature, max_tokens, progress_callback=None, result_callback=None):
        responses = []
        total = len(messages_list)
        for index, messages in enumerate(messages_list, start=1):
            response = self._respond(messages)
            responses.append(response)
            if result_callback is not None:
                result_callback(index - 1, response, index, total)
            if progress_callback is not None:
                progress_callback(index, total)
        return responses

    def create_chat_completion(self, messages, *, temperature, max_tokens):
        return self._respond(messages)

    def _respond(self, messages):
        payload = json.loads(messages[1]["content"])

        if "file_packet" in payload:
            path = payload["file_packet"]["path"]
            if "auth" in path:
                return {
                    "summary": "Authentication middleware and token handling.",
                    "description": "Defines request auth checks and token validation.",
                    "tags": ["auth", "jwt", "middleware"],
                    "categories": ["authentication", "middleware"],
                    "confidence": 0.93,
                    "evidence": [path],
                }
            if "env" in path:
                return {
                    "summary": "Environment API key loading utilities.",
                    "description": "Loads provider API keys from environment variables and helper functions.",
                    "tags": ["env", "api-key", "configuration"],
                    "categories": ["configuration", "shared-utils"],
                    "confidence": 0.91,
                    "evidence": [path],
                }
            return {
                "summary": "Shared crypto utilities.",
                "description": "Provides reusable verification helpers.",
                "tags": ["crypto", "shared"],
                "categories": ["shared-utils"],
                "confidence": 0.88,
                "evidence": [path],
            }

        node_packet = payload["node_packet"]
        node_id = node_packet["node_id"]
        if node_id == "repo":
            return {
                "summary": "Authentication and shared support codebase.",
                "description": "Contains auth flows plus reusable crypto utilities.",
                "tags": ["auth", "shared", "crypto"],
                "categories": ["authentication", "shared-utils"],
                "confidence": 0.95,
                "evidence": ["Folder labels indicate auth and shared utility domains"],
            }
        if node_id == "src/auth":
            return {
                "summary": "Authentication folder.",
                "description": "Groups middleware and token validation code.",
                "tags": ["auth", "jwt", "middleware"],
                "categories": ["authentication", "middleware"],
                "confidence": 0.91,
                "evidence": ["JWT imports", "auth child summaries"],
            }
        if node_id == "src/shared":
            return {
                "summary": "Shared utility folder.",
                "description": "Holds reusable crypto helpers.",
                "tags": ["shared", "crypto"],
                "categories": ["shared-utils"],
                "confidence": 0.87,
                "evidence": ["Utility-style symbol names"],
            }
        return {
            "summary": "Source root folder.",
            "description": "Combines auth and shared utility modules.",
            "tags": ["auth", "shared"],
            "categories": ["authentication", "shared-utils"],
            "confidence": 0.86,
            "evidence": ["Mixed child folder summaries"],
        }


def test_axe_retrieval_returns_implementation_and_references(tmp_path):
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
import jwt
from src.shared.crypto import verify_sig

def verify_token(token: str) -> bool:
    decoded = jwt.decode(token, "secret", algorithms=["HS256"])
    return verify_sig(decoded["sub"])
""".strip(),
        encoding="utf-8",
    )

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    pipeline = MatryoshkaPipeline(config=PipelineConfig(), labeling_engine=engine)
    graph = pipeline.analyze(tmp_path)

    db_path = tmp_path / "index.db"
    database = MatryoshkaDatabase(db_path)
    database.replace_graph(graph)

    conn = sqlite3.connect(db_path)
    repo_row = conn.execute("SELECT id, root_path, category FROM repos").fetchone()
    node_row = conn.execute(
        "SELECT repo_id, normalized_name, top_child_categories_json, top_dependency_tags_json FROM nodes WHERE node_id = 'src/auth/middleware.py'"
    ).fetchone()
    edge_types = {row[0] for row in conn.execute("SELECT DISTINCT edge_type FROM edges")}
    reference_count = conn.execute('SELECT COUNT(*) FROM "references"').fetchone()[0]
    conn.close()

    assert repo_row == (str(tmp_path), str(tmp_path), "authentication")
    assert node_row[0] == str(tmp_path)
    assert node_row[1] == "middleware.py"
    assert json.loads(node_row[2]) == []
    assert "shared" in json.loads(node_row[3])
    assert edge_types >= {"child", "import", "call"}
    assert reference_count > 0

    result = axe_retrieval(db_path, "where is verify_sig implemented", limit=3)

    assert result.node_hits
    assert result.node_hits[0].node.path == "src/shared/crypto.py"
    assert result.symbol_hits
    top_hit = result.symbol_hits[0]
    assert top_hit.symbol.name == "verify_sig"
    assert top_hit.symbol.path == "src/shared/crypto.py"
    assert top_hit.symbol.start_line == 1
    assert any(call.caller_node_id == "src/auth/middleware.py" and call.start_line == 6 for call in top_hit.called_by)
    assert any(reference.reference_kind == "call" and reference.source_node_id == "src/auth/middleware.py" for reference in top_hit.references)
    assert any(reference.reference_kind == "import" and reference.source_node_id == "src/auth/middleware.py" for reference in top_hit.references)


def test_axe_retrieval_prioritizes_exact_symbol_and_callers(tmp_path):
    providers_dir = tmp_path / "src" / "providers"
    providers_dir.mkdir(parents=True)

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

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    pipeline = MatryoshkaPipeline(config=PipelineConfig(), labeling_engine=engine)
    graph = pipeline.analyze(tmp_path)

    db_path = tmp_path / "index.db"
    MatryoshkaDatabase(db_path).replace_graph(graph)

    exact_result = axe_retrieval(db_path, "stream_bedrock", limit=3)

    assert exact_result.symbol_hits
    assert exact_result.symbol_hits[0].symbol.name == "stream_bedrock"
    assert exact_result.symbol_hits[0].symbol.path == "src/providers/streaming.py"

    caller_result = axe_retrieval(db_path, "who calls stream_bedrock", limit=3)

    assert caller_result.node_hits
    assert caller_result.node_hits[0].node.path == "src/providers/adapter.py"
    assert caller_result.symbol_hits
    assert caller_result.symbol_hits[0].symbol.name == "stream_bedrock"


def test_axe_retrieval_handles_natural_language_file_query(tmp_path):
    src_dir = tmp_path / "src"
    providers_dir = src_dir / "providers"
    providers_dir.mkdir(parents=True)

    (src_dir / "env_api_keys.py").write_text(
        """
def get_env_api_key(provider: str) -> str | None:
    return provider.upper()
""".strip(),
        encoding="utf-8",
    )
    (providers_dir / "openai.py").write_text(
        """
from src.env_api_keys import get_env_api_key


def load_openai_credentials() -> str | None:
    return get_env_api_key("openai")
""".strip(),
        encoding="utf-8",
    )

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    pipeline = MatryoshkaPipeline(config=PipelineConfig(), labeling_engine=engine)
    graph = pipeline.analyze(tmp_path)

    db_path = tmp_path / "index.db"
    MatryoshkaDatabase(db_path).replace_graph(graph)

    result = axe_retrieval(db_path, "how are api keys loaded from environment", limit=3)

    assert result.node_hits
    assert result.node_hits[0].node.path == "src/env_api_keys.py"
    assert result.symbol_hits
    assert result.symbol_hits[0].symbol.name == "get_env_api_key"