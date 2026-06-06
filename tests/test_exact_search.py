from __future__ import annotations

import json

from matryoshka.exact_search import axe_call_search, axe_file_search, axe_import_search, axe_module_search, axe_reference_search, axe_symbol_search
from matryoshka.labeling import LabelingConfig, LabelingEngine
from matryoshka.pipeline import MatryoshkaPipeline, PipelineConfig
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
            if "provider" in path or "stream" in path:
                return {
                    "summary": "Streaming provider helpers.",
                    "description": "Contains provider entry points and stream wrappers.",
                    "tags": ["provider", "stream"],
                    "categories": ["llm-integration"],
                    "confidence": 0.9,
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

        return {
            "summary": "Repository summary.",
            "description": "Groups analyzed files.",
            "tags": ["tooling"],
            "categories": ["developer-tooling"],
            "confidence": 0.85,
            "evidence": [payload.get("task", "")],
        }


def test_axe_exact_tools_cover_file_symbol_import_module_call_and_reference_queries(tmp_path):
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

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    pipeline = MatryoshkaPipeline(config=PipelineConfig(), labeling_engine=engine)
    graph = pipeline.analyze(tmp_path)

    db_path = tmp_path / "index.db"
    MatryoshkaDatabase(db_path).replace_graph(graph)

    file_result = axe_file_search(db_path, "middleware.py", limit=3)
    assert file_result.node_hits
    assert file_result.node_hits[0].node.path == "src/auth/middleware.py"

    symbol_result = axe_symbol_search(db_path, "verify_sig", limit=3)
    assert symbol_result.symbol_hits
    assert symbol_result.symbol_hits[0].symbol.name == "verify_sig"
    assert symbol_result.symbol_hits[0].symbol.path == "src/shared/crypto.py"

    import_result = axe_import_search(db_path, "src.shared.crypto", limit=3)
    assert import_result.import_hits
    assert import_result.import_hits[0].importer_node.path == "src/auth/middleware.py"
    assert import_result.import_hits[0].target_node is not None
    assert import_result.import_hits[0].target_node.path == "src/shared/crypto.py"

    module_result = axe_module_search(db_path, "src.providers.streaming", limit=3)
    assert module_result.node_hits
    assert module_result.node_hits[0].node.path == "src/providers/streaming.py"
    assert module_result.import_hits
    assert module_result.import_hits[0].importer_node.path == "src/providers/adapter.py"

    call_result = axe_call_search(db_path, "who calls stream_bedrock", limit=3)
    assert call_result.call_hits
    assert call_result.call_hits[0].caller_node is not None
    assert call_result.call_hits[0].caller_node.path == "src/providers/adapter.py"
    assert call_result.call_hits[0].target_symbol is not None
    assert call_result.call_hits[0].target_symbol.name == "stream_bedrock"

    reference_result = axe_reference_search(db_path, "verify_sig", limit=5)
    assert reference_result.reference_hits
    assert {hit.reference_record.reference_kind for hit in reference_result.reference_hits} >= {"call", "import"}
    assert all(hit.source_node is not None and hit.source_node.path == "src/auth/middleware.py" for hit in reference_result.reference_hits)