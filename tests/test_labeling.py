from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from cradle.cache import LabelCache
from cradle.labeling import LabelingConfig, LabelingEngine
from cradle.models import FilePacket, NodePacket
from cradle.prompts import LABEL_RESPONSE_SCHEMA, build_file_label_messages, build_node_label_messages


class FakeClient:
    def __init__(self, responses):
        self._responses = responses
        self.model = "fake-model"
        self.calls = 0

    def create_many_chat_completions(self, messages_list, *, temperature, max_tokens, progress_callback=None, result_callback=None):
        self.calls += len(messages_list)
        responses = []
        total = len(messages_list)
        for index, messages in enumerate(messages_list, start=1):
            response = self._responses(messages)
            responses.append(response)
            if result_callback is not None:
                result_callback(index - 1, response, index, total)
            if progress_callback is not None:
                progress_callback(index, total)
        return responses

    def create_chat_completion(self, messages, *, temperature, max_tokens):
        self.calls += 1
        return self._responses(messages)


def test_prompt_builders_embed_schema_and_packets():
    file_packet = FilePacket(
        path="src/auth/middleware.py",
        language="python",
        summary_input="path: src/auth/middleware.py",
        imports_external=["jwt"],
        top_symbols=["def verify_token(token: str) -> Claims"],
    )
    node_packet = NodePacket(
        node_id="src/auth",
        path="src/auth",
        level="folder",
        child_summaries=["JWT validation middleware."],
    )

    file_messages = build_file_label_messages(file_packet)
    node_messages = build_node_label_messages(node_packet)

    file_payload = json.loads(file_messages[1]["content"])
    node_payload = json.loads(node_messages[1]["content"])

    assert file_payload["response_schema"] == LABEL_RESPONSE_SCHEMA
    assert file_payload["file_packet"]["path"] == "src/auth/middleware.py"
    assert node_payload["response_schema"] == LABEL_RESPONSE_SCHEMA
    assert node_payload["node_packet"]["node_id"] == "src/auth"


def test_prompt_builders_trim_large_packets():
    file_packet = FilePacket(
        path="src/huge/file.py",
        language="python",
        summary_input="x" * 5000,
        imports_external=[f"package_{index}" for index in range(40)],
        imports_internal=[f"src.module_{index}" for index in range(40)],
        imported_symbols=[f"symbol_{index}" for index in range(50)],
        top_symbols=["def example(value: str) -> str: ..." * 30 for _ in range(20)],
        docstrings=["d" * 1000 for _ in range(10)],
        call_hints=[f"callee_{index}" for index in range(50)],
        code_snippets=["line\n" * 500 for _ in range(10)],
        import_signature={f"pkg_{index}": index for index in range(30)},
        internal_signature={f"src.pkg_{index}": index for index in range(30)},
    )

    messages = build_file_label_messages(file_packet)
    payload = json.loads(messages[1]["content"])
    packet_payload = payload["file_packet"]

    assert len(packet_payload["summary_input"]) <= 1200
    assert len(packet_payload["imports_external"]) == 20
    assert len(packet_payload["imports_internal"]) == 20
    assert len(packet_payload["imported_symbols"]) == 25
    assert len(packet_payload["top_symbols"]) == 12
    assert len(packet_payload["code_snippets"]) == 3


def test_labeling_engine_filters_unknown_categories_and_compares_labels():
    def responder(messages):
        payload = json.loads(messages[1]["content"])
        if "file_packet" in payload:
            return {
                "summary": "JWT validation middleware.",
                "description": "Handles auth checks for incoming requests.",
                "tags": ["auth", "jwt", "middleware"],
                "categories": ["authentication", "made-up-category"],
                "confidence": 0.91,
                "evidence": ["Uses jwt", "Defines verify_token"],
            }
        return {
            "summary": "Authentication folder.",
            "description": "Groups the auth middleware and token code.",
            "tags": ["auth", "tokens"],
            "categories": ["authentication"],
            "confidence": 0.84,
            "evidence": ["Child summaries are auth-specific"],
        }

    engine = LabelingEngine(FakeClient(responder), LabelingConfig())
    file_packet = FilePacket(path="src/auth/middleware.py", language="python", summary_input="auth file")
    node_packet = NodePacket(node_id="src/auth", path="src/auth", level="folder")

    file_label = engine.label_files([file_packet])[file_packet.path]
    node_label = engine.label_node(node_packet)
    report = engine.compare_labels(file_label, node_label)

    assert file_label.categories == ["authentication"]
    assert node_label.categories == ["authentication"]
    assert report.agreed_categories == ["authentication"]
    assert report.bottom_up_only == []
    assert report.top_down_only == []


def test_labeling_engine_uses_persistent_cache(tmp_path):
    def responder(messages):
        return {
            "summary": "JWT validation middleware.",
            "description": "Handles auth checks for incoming requests.",
            "tags": ["auth", "jwt", "middleware"],
            "categories": ["authentication", "middleware"],
            "confidence": 0.91,
            "evidence": ["Uses jwt", "Defines verify_token"],
        }

    cache_path = Path(tmp_path) / "labels.db"
    client = FakeClient(responder)
    engine = LabelingEngine(client, LabelingConfig(), cache=LabelCache(cache_path))
    file_packet = FilePacket(path="src/auth/middleware.py", language="python", summary_input="auth file")

    first = engine.label_files([file_packet])[file_packet.path]

    conn = sqlite3.connect(cache_path)
    payload_json = conn.execute("SELECT payload_json FROM label_cache").fetchone()[0]
    conn.close()

    cached_engine = LabelingEngine(FakeClient(responder), LabelingConfig(), cache=LabelCache(cache_path))
    second_client = cached_engine._client
    second = cached_engine.label_files([file_packet])[file_packet.path]

    assert first.summary == second.summary
    assert client.calls == 1
    assert second_client.calls == 0
    assert "raw_response" not in json.loads(payload_json)


def test_labeling_engine_reports_progress_for_pending_requests():
    def responder(messages):
        return {
            "summary": "JWT validation middleware.",
            "description": "Handles auth checks for incoming requests.",
            "tags": ["auth", "jwt", "middleware"],
            "categories": ["authentication", "middleware"],
            "confidence": 0.91,
            "evidence": ["Uses jwt", "Defines verify_token"],
        }

    engine = LabelingEngine(FakeClient(responder), LabelingConfig())
    packets = [
        FilePacket(path="src/auth/a.py", language="python", summary_input="auth file"),
        FilePacket(path="src/auth/b.py", language="python", summary_input="auth file"),
    ]
    updates: list[str] = []

    labels = engine.label_files(packets, progress=updates.append)

    assert set(labels) == {"src/auth/a.py", "src/auth/b.py"}
    assert updates
    assert updates[-1] == "resolved file labels 2/2"


def test_labeling_engine_handles_non_object_model_response():
    def responder(messages):
        return ["unexpected", "list"]

    engine = LabelingEngine(FakeClient(responder), LabelingConfig())
    file_packet = FilePacket(path="src/auth/middleware.py", language="python", summary_input="auth file")

    label = engine.label_files([file_packet])[file_packet.path]

    assert label.confidence == 0.0
    assert label.categories == []
    assert label.summary == "Malformed model response for src/auth/middleware.py"