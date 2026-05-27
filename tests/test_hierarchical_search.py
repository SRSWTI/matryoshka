from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from cradle.hierarchical_search import axe_hierarchy_search
from cradle.labeling import LabelingConfig, LabelingEngine
from cradle.pipeline import CradlePipeline, PipelineConfig
from cradle.semantic_index import SemanticIndexBuilder
from cradle.storage import CradleDatabase


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
    graph = CradlePipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    CradleDatabase(db_path).replace_graph(graph)
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
    graph = CradlePipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    CradleDatabase(db_path).replace_graph(graph)
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
    graph = CradlePipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    CradleDatabase(db_path).replace_graph(graph)
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
    graph = CradlePipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    CradleDatabase(db_path).replace_graph(graph)
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
    graph = CradlePipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    CradleDatabase(db_path).replace_graph(graph)
    SemanticIndexBuilder(db_path, embedder=FakeEmbedder()).build()

    result = axe_hierarchy_search(db_path, "bedrock streaming provider implementation", embedder=FakeEmbedder(), limit=3)

    assert result.node_hits
    assert result.node_hits[0].node.path == "src/providers/amazon_bedrock.py"