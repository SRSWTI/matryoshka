from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from cradle.labeling import LabelingConfig, LabelingEngine
from cradle.pipeline import CradlePipeline, PipelineConfig
from cradle.question_answering import axe_question
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
            return {
                "summary": "Streaming provider helpers.",
                "description": "Contains provider entry points and stream wrappers.",
                "tags": ["provider", "stream"],
                "categories": ["llm-integration"],
                "confidence": 0.9,
                "evidence": [path],
            }

        node_id = payload["node_packet"]["node_id"]
        if node_id == "src/presentation":
            return {
                "summary": "Presentation and output formatting folder.",
                "description": "Groups the code that decides what to show the user.",
                "tags": ["presentation", "output", "user"],
                "categories": ["presentation"],
                "confidence": 0.93,
                "evidence": [node_id],
            }
        return {
            "summary": "Repository summary.",
            "description": "Contains presentation and provider branches.",
            "tags": ["presentation", "provider"],
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


def test_axe_question_answers_flow_query_with_hierarchy_and_excerpt(tmp_path):
    presentation_dir = tmp_path / "src" / "presentation"
    providers_dir = tmp_path / "src" / "providers"
    presentation_dir.mkdir(parents=True)
    providers_dir.mkdir(parents=True)

    (presentation_dir / "render.py").write_text(
        """
def decide_visible_output(user_role: str) -> str:
    return "admin panel" if user_role == "admin" else "basic panel"
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

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    graph = CradlePipeline(config=PipelineConfig(), labeling_engine=engine).analyze(tmp_path)
    db_path = tmp_path / "index.db"
    CradleDatabase(db_path).replace_graph(graph)
    SemanticIndexBuilder(db_path, embedder=FakeEmbedder()).build()

    result = axe_question(db_path, "where does the system decide what to show the user", embedder=FakeEmbedder())

    assert "src/presentation/render.py" in result.answer
    assert "decide_visible_output" in result.answer
    assert "Supporting import evidence" not in result.answer
    assert "Supporting reference evidence" not in result.answer
    assert result.excerpts
    assert "return \"admin panel\"" in result.excerpts[0].text


def test_axe_question_answers_caller_query_with_exact_call_evidence(tmp_path):
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

    result = axe_question(db_path, "who calls stream_bedrock", embedder=FakeEmbedder())

    assert "stream_bedrock is called from" in result.answer
    assert "stream_bedrock_lazy" in result.answer
    assert result.call_hits
    assert result.call_hits[0].caller_node is not None
    assert result.call_hits[0].caller_node.path == "src/providers/adapter.py"