from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from cradle.embeddings import DEFAULT_QUERY_TASK
from cradle.labeling import LabelingConfig, LabelingEngine
from cradle.pipeline import CradlePipeline, PipelineConfig
from cradle.semantic_index import SemanticIndexBuilder
from cradle.semantic_search import axe_semantic_search
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
                "summary": "Streaming provider helpers.",
                "description": "Contains stream wrappers and provider entry points.",
                "tags": ["stream", "provider"],
                "categories": ["llm-integration"],
                "confidence": 0.9,
                "evidence": [path],
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


def test_semantic_search_returns_implementation_file(tmp_path):
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
    graph = CradlePipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    CradleDatabase(db_path).replace_graph(graph)
    SemanticIndexBuilder(db_path, embedder=FakeEmbedder()).build()

    result = axe_semantic_search(db_path, "how are api keys loaded from environment", embedder=FakeEmbedder(), task=DEFAULT_QUERY_TASK, limit=3)

    assert result.node_hits
    assert result.node_hits[0].node.path == "src/env_api_keys.py"
    assert result.symbol_hits
    assert result.symbol_hits[0].symbol.name == "get_env_api_key"


def test_semantic_search_prioritizes_exact_symbol_and_callers(tmp_path):
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
    graph = CradlePipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    CradleDatabase(db_path).replace_graph(graph)
    SemanticIndexBuilder(db_path, embedder=FakeEmbedder()).build()

    exact = axe_semantic_search(db_path, "stream_bedrock", embedder=FakeEmbedder(), task=DEFAULT_QUERY_TASK, limit=3)
    callers = axe_semantic_search(db_path, "who calls stream_bedrock", embedder=FakeEmbedder(), task=DEFAULT_QUERY_TASK, limit=3)

    assert exact.symbol_hits
    assert exact.symbol_hits[0].symbol.name == "stream_bedrock"
    assert callers.node_hits
    assert callers.node_hits[0].node.path == "src/providers/adapter.py"


def test_semantic_search_prefers_implementation_over_register_wrapper(tmp_path):
    providers_dir = tmp_path / "src" / "providers"
    providers_dir.mkdir(parents=True)

    (providers_dir / "amazon_bedrock.py").write_text(
        """
def stream_bedrock() -> str:
    return "stream"


def stream_simple_bedrock() -> str:
    return stream_bedrock()
""".strip(),
        encoding="utf-8",
    )
    (providers_dir / "register_builtins.py").write_text(
        """
from src.providers.amazon_bedrock import stream_bedrock, stream_simple_bedrock


def load_bedrock_provider_module() -> tuple[object, object]:
    return stream_bedrock, stream_simple_bedrock


stream_bedrock_lazy = load_bedrock_provider_module
""".strip(),
        encoding="utf-8",
    )

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    graph = CradlePipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    CradleDatabase(db_path).replace_graph(graph)
    SemanticIndexBuilder(db_path, embedder=FakeEmbedder()).build()

    result = axe_semantic_search(
        db_path,
        "bedrock streaming provider implementation",
        embedder=FakeEmbedder(),
        task=DEFAULT_QUERY_TASK,
        limit=3,
    )

    assert result.node_hits
    assert result.node_hits[0].node.path == "src/providers/amazon_bedrock.py"


def test_semantic_search_prefers_provider_named_in_query(tmp_path):
    providers_dir = tmp_path / "src" / "providers"
    providers_dir.mkdir(parents=True)

    (providers_dir / "amazon_bedrock.py").write_text(
        """
def stream_bedrock() -> str:
    return "bedrock"
""".strip(),
        encoding="utf-8",
    )
    (providers_dir / "mistral.py").write_text(
        """
def stream_mistral() -> str:
    return "mistral"
""".strip(),
        encoding="utf-8",
    )

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    graph = CradlePipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    CradleDatabase(db_path).replace_graph(graph)
    SemanticIndexBuilder(db_path, embedder=FakeEmbedder()).build()

    result = axe_semantic_search(
        db_path,
        "bedrock streaming provider implementation",
        embedder=FakeEmbedder(),
        task=DEFAULT_QUERY_TASK,
        limit=3,
    )

    assert result.node_hits
    assert result.node_hits[0].node.path == "src/providers/amazon_bedrock.py"