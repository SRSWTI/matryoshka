from __future__ import annotations

import hashlib
import json
import re
import sqlite3

import numpy as np

from matryoshka.hierarchical_search import axe_hierarchy_search
from matryoshka.labeling import LabelingConfig, LabelingEngine
from matryoshka.pipeline import MatryoshkaPipeline, PipelineConfig
from matryoshka.semantic_index import SemanticIndexBuilder, SemanticIndexStore
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
            if "presentation" in path:
                return {
                    "summary": "Decides what output to show the user and formats visible responses.",
                    "description": "Contains presentation routing and output formatting logic for user-facing responses.",
                    "tags": ["presentation", "output", "user"],
                    "categories": ["presentation"],
                    "confidence": 0.94,
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
                "summary": "Authentication middleware and token handling.",
                "description": "Defines request auth checks and token validation.",
                "tags": ["auth", "jwt", "middleware"],
                "categories": ["authentication"],
                "confidence": 0.9,
                "evidence": [path],
            }

        node_packet = payload["node_packet"]
        node_id = node_packet["node_id"]
        if node_id == "src/presentation":
            return {
                "summary": "Presentation and output formatting folder.",
                "description": "Groups the code that decides what to show the user and how visible responses are formatted.",
                "tags": ["presentation", "output", "user"],
                "categories": ["presentation"],
                "confidence": 0.93,
                "evidence": [node_id],
            }
        if node_id == "src/config":
            return {
                "summary": "Configuration folder.",
                "description": "Groups environment and credential loading code.",
                "tags": ["configuration", "env"],
                "categories": ["configuration"],
                "confidence": 0.91,
                "evidence": [node_id],
            }
        if node_id == "src":
            return {
                "summary": "Application source folder.",
                "description": "Contains presentation, auth, and configuration branches.",
                "tags": ["presentation", "auth", "configuration"],
                "categories": ["application-core"],
                "confidence": 0.9,
                "evidence": [node_id],
            }
        return {
            "summary": "Repository summary.",
            "description": "Contains application branches for presentation and configuration logic.",
            "tags": ["presentation", "configuration"],
            "categories": ["application-core"],
            "confidence": 0.88,
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


def _write_phase5_fixture(tmp_path):
    runtime_dir = tmp_path / "src" / "runtime"
    tokens_dir = tmp_path / "src" / "tokens"
    oauth_dir = tmp_path / "src" / "oauth"
    ui_dir = tmp_path / "src" / "ui"
    runtime_dir.mkdir(parents=True)
    tokens_dir.mkdir(parents=True)
    oauth_dir.mkdir(parents=True)
    ui_dir.mkdir(parents=True)

    (tokens_dir / "loader.py").write_text(
        """
from src.runtime.session import build_session


def load_token(token: str) -> str:
    return build_session(token)
""".strip(),
        encoding="utf-8",
    )
    (runtime_dir / "session.py").write_text(
        """
from src.tokens.loader import load_token


def build_session(token: str) -> str:
    return f"session:{load_token(token)}"
""".strip(),
        encoding="utf-8",
    )
    (oauth_dir / "device_flow.py").write_text(
        """
from src.runtime.session import build_session


def refresh_device_session(token: str) -> str:
    return build_session(token)
""".strip(),
        encoding="utf-8",
    )
    (oauth_dir / "refresh.py").write_text(
        """
from src.tokens.loader import load_token


def refresh_token(token: str) -> str:
    return load_token(token)
""".strip(),
        encoding="utf-8",
    )
    (ui_dir / "render.py").write_text(
        """
def render_screen() -> str:
    return "screen"
""".strip(),
        encoding="utf-8",
    )

    return {
        "target_paths": {
            "src/tokens/loader.py",
            "src/runtime/session.py",
            "src/oauth/device_flow.py",
            "src/oauth/refresh.py",
        }
    }


def test_hierarchy_search_routes_natural_language_query_through_folder_branch(tmp_path):
    presentation_dir = tmp_path / "src" / "presentation"
    auth_dir = tmp_path / "src" / "auth"
    config_dir = tmp_path / "src" / "config"
    presentation_dir.mkdir(parents=True)
    auth_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    (presentation_dir / "render.py").write_text(
        """
def decide_visible_output(user_role: str) -> str:
    return "admin panel" if user_role == "admin" else "basic panel"
""".strip(),
        encoding="utf-8",
    )
    (auth_dir / "middleware.py").write_text(
        """
def verify_token(token: str) -> bool:
    return token.startswith("sig-")
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "env_api_keys.py").write_text(
        """
def get_env_api_key(provider: str) -> str | None:
    return provider.upper()
""".strip(),
        encoding="utf-8",
    )

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    graph = MatryoshkaPipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    MatryoshkaDatabase(db_path).replace_graph(graph)
    SemanticIndexBuilder(db_path, embedder=FakeEmbedder()).build()

    result = axe_hierarchy_search(db_path, "where does the system decide what to show the user", embedder=FakeEmbedder(), limit=3)

    assert result.steps
    assert any(candidate.node.path == "src/presentation" for step in result.steps for candidate in step.candidates)
    assert result.node_hits
    assert result.node_hits[0].node.path == "src/presentation/render.py"
    assert result.symbol_hits
    assert result.symbol_hits[0].symbol.name == "decide_visible_output"


def test_hierarchy_search_keeps_symbol_lookup_inside_selected_branch(tmp_path):
    presentation_dir = tmp_path / "src" / "presentation"
    config_dir = tmp_path / "src" / "config"
    presentation_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    (presentation_dir / "render.py").write_text(
        """
def decide_visible_output(user_role: str) -> str:
    return user_role
""".strip(),
        encoding="utf-8",
    )
    (config_dir / "env_api_keys.py").write_text(
        """
def get_env_api_key(provider: str) -> str | None:
    return provider.upper()
""".strip(),
        encoding="utf-8",
    )

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    graph = MatryoshkaPipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    MatryoshkaDatabase(db_path).replace_graph(graph)
    SemanticIndexBuilder(db_path, embedder=FakeEmbedder()).build()

    result = axe_hierarchy_search(db_path, "how are api keys loaded from environment", embedder=FakeEmbedder(), limit=3)

    assert result.steps
    assert any(candidate.node.path == "src/config" for step in result.steps for candidate in step.candidates)
    assert result.node_hits
    assert result.node_hits[0].node.path == "src/config/env_api_keys.py"
    assert result.symbol_hits
    assert result.symbol_hits[0].symbol.name == "get_env_api_key"


def test_hierarchy_search_prefers_folder_branches_before_file_children(tmp_path):
    src_dir = tmp_path / "src"
    presentation_dir = src_dir / "presentation"
    presentation_dir.mkdir(parents=True)

    (src_dir / "types.py").write_text(
        """
USER_OUTPUT_LABEL = "visible-output"
""".strip(),
        encoding="utf-8",
    )
    (presentation_dir / "render.py").write_text(
        """
def decide_visible_output(user_role: str) -> str:
    return user_role
""".strip(),
        encoding="utf-8",
    )

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    graph = MatryoshkaPipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    MatryoshkaDatabase(db_path).replace_graph(graph)
    SemanticIndexBuilder(db_path, embedder=FakeEmbedder()).build()

    result = axe_hierarchy_search(db_path, "where does the system decide what to show the user", embedder=FakeEmbedder(), limit=3)

    assert result.steps
    assert result.steps[0].candidates[0].node.path == "src"
    assert any(candidate.node.path == "src/presentation" for candidate in result.steps[1].candidates)


def test_hierarchy_search_allows_direct_file_when_it_clearly_beats_folder_branch(tmp_path):
    providers_dir = tmp_path / "src" / "providers"
    images_dir = providers_dir / "images"
    images_dir.mkdir(parents=True)

    (providers_dir / "amazon_bedrock.py").write_text(
        """
def stream_bedrock() -> str:
    return "bedrock"
""".strip(),
        encoding="utf-8",
    )
    (images_dir / "openrouter.py").write_text(
        """
def render_image() -> str:
    return "image"
""".strip(),
        encoding="utf-8",
    )

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    graph = MatryoshkaPipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    MatryoshkaDatabase(db_path).replace_graph(graph)
    SemanticIndexBuilder(db_path, embedder=FakeEmbedder()).build()

    result = axe_hierarchy_search(db_path, "bedrock streaming provider implementation", embedder=FakeEmbedder(), limit=3)

    assert result.node_hits
    assert result.node_hits[0].node.path == "src/providers/amazon_bedrock.py"


def test_hierarchy_search_prefers_implementation_file_over_wrapper_module(tmp_path):
    src_dir = tmp_path / "src"
    providers_dir = src_dir / "providers"
    providers_dir.mkdir(parents=True)

    (src_dir / "bedrock_provider.py").write_text(
        """
bedrock_provider_module = {
    'stream': 'stream_bedrock',
}
""".strip(),
        encoding="utf-8",
    )
    (providers_dir / "amazon_bedrock.py").write_text(
        """
def stream_bedrock() -> str:
    return 'bedrock'
""".strip(),
        encoding="utf-8",
    )

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    graph = MatryoshkaPipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    MatryoshkaDatabase(db_path).replace_graph(graph)
    SemanticIndexBuilder(db_path, embedder=FakeEmbedder()).build()

    result = axe_hierarchy_search(db_path, "bedrock streaming provider implementation", embedder=FakeEmbedder(), limit=3)

    assert result.node_hits
    assert result.node_hits[0].node.path == "src/providers/amazon_bedrock.py"


def test_pipeline_persists_louvain_communities(tmp_path):
    fixture = _write_phase5_fixture(tmp_path)

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    graph = MatryoshkaPipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)

    community_nodes = [node for node in graph.nodes if node.kind == "community"]
    assert community_nodes
    assert graph.community_members
    assert fixture["target_paths"].issuperset({record.member_node_id for record in graph.community_members})

    db_path = tmp_path / "index.db"
    MatryoshkaDatabase(db_path).replace_graph(graph)

    with sqlite3.connect(db_path) as conn:
        community_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE kind = 'community'").fetchone()[0]
        membership_count = conn.execute("SELECT COUNT(*) FROM community_members").fetchone()[0]

    assert community_count >= 1
    assert membership_count >= 4


def test_semantic_index_builds_centroids_and_hierarchy_uses_community_branch(tmp_path):
    fixture = _write_phase5_fixture(tmp_path)

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    graph = MatryoshkaPipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    community_ids = [node.node_id for node in graph.nodes if node.kind == "community"]
    assert community_ids

    db_path = tmp_path / "index.db"
    MatryoshkaDatabase(db_path).replace_graph(graph)
    SemanticIndexBuilder(db_path, embedder=FakeEmbedder()).build()
    store = SemanticIndexStore(db_path)
    assert any(record.summary.startswith("Semantic routing summary") for record in store.centroid_records)

    query_vector = FakeEmbedder().encode(["token session refresh flow"])[0]
    centroid_hits = store.search_node_centroids(query_vector, community_ids, top_k=2)
    assert centroid_hits

    result = axe_hierarchy_search(db_path, "token session refresh flow", embedder=FakeEmbedder(), limit=3)

    assert any(candidate.node.kind == "community" for step in result.steps for candidate in step.candidates)
    assert result.node_hits
    assert result.node_hits[0].node.path in fixture["target_paths"]


def test_hierarchy_search_can_route_through_theme_domain(tmp_path):
    class ThemeClient(FakeClient):
        def _respond(self, messages):
            payload = json.loads(messages[1]["content"])
            if "file_packet" in payload:
                path = payload["file_packet"]["path"]
                if "security" in path or "identity" in path:
                    return {
                        "summary": "Authentication and identity management logic.",
                        "description": "Implements login checks, token validation, and session assembly.",
                        "tags": ["auth", "identity", "session"],
                        "categories": ["authentication"],
                        "confidence": 0.95,
                        "evidence": [path],
                    }
                return super()._respond(messages)

            node_packet = payload["node_packet"]
            node_id = node_packet["node_id"]
            if node_id in {"src/security", "src/identity"}:
                return {
                    "summary": "Authentication branch.",
                    "description": "Contains auth and identity code.",
                    "tags": ["auth", "identity"],
                    "categories": ["authentication"],
                    "confidence": 0.92,
                    "evidence": [node_id],
                }
            return super()._respond(messages)

    security_dir = tmp_path / "src" / "security"
    identity_dir = tmp_path / "src" / "identity"
    security_dir.mkdir(parents=True)
    identity_dir.mkdir(parents=True)

    (security_dir / "tokens.py").write_text(
        """
def validate_token(token: str) -> bool:
    return token.startswith("sig-")
""".strip(),
        encoding="utf-8",
    )
    (identity_dir / "session.py").write_text(
        """
from src.security.tokens import validate_token


def create_session(token: str) -> str:
    return token if validate_token(token) else "invalid"
""".strip(),
        encoding="utf-8",
    )

    engine = LabelingEngine(ThemeClient(), LabelingConfig())
    graph = MatryoshkaPipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    MatryoshkaDatabase(db_path).replace_graph(graph)
    SemanticIndexBuilder(db_path, embedder=FakeEmbedder()).build()

    result = axe_hierarchy_search(db_path, "auth login session", embedder=FakeEmbedder(), limit=3)

    assert result.steps
    assert any(candidate.node.node_id == "theme::authentication" for step in result.steps for candidate in step.candidates)
    assert result.node_hits
    assert result.node_hits[0].node.path in {"src/identity/session.py", "src/security/tokens.py"}