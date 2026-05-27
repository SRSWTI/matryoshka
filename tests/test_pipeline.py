from __future__ import annotations

import json

from cradle.labeling import LabelingConfig, LabelingEngine
from cradle.pipeline import CradlePipeline, PipelineConfig


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
        if payload.get("task", "").startswith("Confirm"):
            if node_id == "src/auth":
                return {
                    "summary": "Authentication folder.",
                    "description": "Aligned with the repository auth domain.",
                    "tags": ["auth", "jwt"],
                    "categories": ["authentication", "middleware"],
                    "confidence": 0.9,
                    "evidence": ["Repo context matches child summaries"],
                }
            return {
                "summary": "Shared support folder.",
                "description": "Aligned with shared repo utilities.",
                "tags": ["shared", "crypto"],
                "categories": ["shared-utils"],
                "confidence": 0.82,
                "evidence": ["Repo context preserves local specificity"],
            }

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


def test_pipeline_builds_graph_with_context_calls_and_references(tmp_path):
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
    pipeline = CradlePipeline(config=PipelineConfig(), labeling_engine=engine)

    graph = pipeline.analyze(tmp_path)

    nodes_by_id = {node.node_id: node for node in graph.nodes}
    assert set(nodes_by_id) >= {"src", "src/auth", "src/shared", "src/auth/middleware.py", "src/shared/crypto.py", "repo"}
    assert nodes_by_id["src/auth/middleware.py"].primary_category == "authentication"
    assert nodes_by_id["src/shared/crypto.py"].primary_category == "shared-utils"

    shared_import = next(
        record
        for record in graph.imports
        if record.importer_node_id == "src/auth/middleware.py" and record.target_node_id == "src/shared/crypto.py"
    )
    assert shared_import.strength_label == "medium"
    assert shared_import.names == ["verify_sig"]

    context = next(record for record in graph.node_context if record.node_id == "src/auth/middleware.py")
    assert context.source_node_id == "src/shared/crypto.py"
    assert context.inherited_category == "shared-utils"
    assert context.inherited_summary == "Shared crypto utilities."

    verify_sig_symbol = next(symbol for symbol in graph.symbols if symbol.name == "verify_sig")
    call = next(record for record in graph.calls if record.target_symbol_id == verify_sig_symbol.symbol_id)
    assert call.caller_node_id == "src/auth/middleware.py"
    assert call.start_line == 6

    references = [reference for reference in graph.references if reference.target_name == "verify_sig"]
    assert {reference.reference_kind for reference in references} >= {"call", "import"}