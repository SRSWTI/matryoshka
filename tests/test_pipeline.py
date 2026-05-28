from __future__ import annotations

import json

from cradle.labeling import LabelingConfig, LabelingEngine
from cradle.pipeline import CradlePipeline, PipelineConfig, RepositoryWalker


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


def test_repository_walker_ignores_nested_test_directories_by_default(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "keep.py").write_text("def keep() -> bool:\n    return True\n", encoding="utf-8")

    nested_tests_dir = tmp_path / "packages" / "feature" / "tests"
    nested_tests_dir.mkdir(parents=True)
    (nested_tests_dir / "test_feature.py").write_text("def test_feature() -> None:\n    assert True\n", encoding="utf-8")

    nested_test_dir = tmp_path / "pkg" / "test"
    nested_test_dir.mkdir(parents=True)
    (nested_test_dir / "test_other.py").write_text("def test_other() -> None:\n    assert True\n", encoding="utf-8")

    walker = RepositoryWalker(PipelineConfig())

    files = [path.relative_to(tmp_path).as_posix() for path in walker.collect_source_files(tmp_path)]

    assert files == ["src/keep.py"]


def test_repository_walker_honors_root_gitignore(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "keep.py").write_text("def keep() -> bool:\n    return True\n", encoding="utf-8")

    generated_dir = tmp_path / "generated"
    generated_dir.mkdir(parents=True)
    (generated_dir / "skip.py").write_text("def skip() -> bool:\n    return False\n", encoding="utf-8")

    (tmp_path / ".gitignore").write_text("generated/\nignored.py\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("def ignored() -> None:\n    return None\n", encoding="utf-8")

    walker = RepositoryWalker(PipelineConfig())

    files = [path.relative_to(tmp_path).as_posix() for path in walker.collect_source_files(tmp_path)]

    assert files == ["src/keep.py"]


def test_pipeline_persists_duplicate_tags_and_categories_without_conflict(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "dup.py").write_text("def dup() -> bool:\n    return True\n", encoding="utf-8")

    class DuplicateLabelClient(FakeClient):
        def _respond(self, messages):
            payload = json.loads(messages[1]["content"])
            if "file_packet" in payload:
                return {
                    "summary": "Duplicate label file.",
                    "description": "Returns repeated tags and categories.",
                    "tags": ["dup", "dup", "shared"],
                    "categories": ["repeat", "repeat", "mixed"],
                    "confidence": 0.9,
                    "evidence": [payload["file_packet"]["path"]],
                }
            return {
                "summary": "Duplicate label folder.",
                "description": "Returns repeated tags and categories.",
                "tags": ["dup", "dup", "shared"],
                "categories": ["repeat", "repeat", "mixed"],
                "confidence": 0.9,
                "evidence": [payload["node_packet"]["node_id"]],
            }

    engine = LabelingEngine(DuplicateLabelClient(), LabelingConfig())
    pipeline = CradlePipeline(config=PipelineConfig(), labeling_engine=engine)

    graph = pipeline.analyze(tmp_path)

    nodes = [node for node in graph.nodes if node.node_id == "src/dup.py"]
    assert nodes


def test_pipeline_builds_structural_communities_and_semantic_themes(tmp_path):
    auth_dir = tmp_path / "src" / "auth"
    auth_dir.mkdir(parents=True)

    (auth_dir / "tokens.py").write_text(
        """
def verify_token(token: str) -> bool:
    return token.startswith("sig-")
""".strip(),
        encoding="utf-8",
    )
    (auth_dir / "session.py").write_text(
        """
from src.auth.tokens import verify_token


def build_session(token: str) -> str:
    return token if verify_token(token) else "invalid"
""".strip(),
        encoding="utf-8",
    )

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    pipeline = CradlePipeline(config=PipelineConfig(), labeling_engine=engine)

    graph = pipeline.analyze(tmp_path)

    community_nodes = [node for node in graph.nodes if node.kind == "community"]
    theme_nodes = [node for node in graph.nodes if node.kind == "theme"]
    assert community_nodes
    assert any(node.summary.startswith("Structural import/call community") for node in community_nodes)
    assert any(node.node_id == "theme::authentication" for node in theme_nodes)
    assert graph.theme_members
    assert {record.member_node_id for record in graph.theme_members if record.theme_node_id == "theme::authentication"} == {
        "src/auth/session.py",
        "src/auth/tokens.py",
    }


def test_pipeline_marks_unresolvable_internal_imports_as_out_of_scope(tmp_path):
    """An internal import whose target does not exist in the analyzed root must
    be flagged is_out_of_scope=True so the UI can render it as a boundary edge
    rather than silently treating it as a missing/unresolved file."""
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)

    # This file imports from 'src.external_pkg.utils', which looks internal (the
    # first path segment 'src' exists in the repo) but the target file does NOT
    # exist anywhere under tmp_path.
    (src_dir / "service.py").write_text(
        "from src.external_pkg.utils import helper\n\ndef run(): helper()\n",
        encoding="utf-8",
    )
    # This file imports from a sibling that DOES exist — must NOT be out_of_scope.
    (src_dir / "utils.py").write_text("def helper(): pass\n", encoding="utf-8")
    (src_dir / "consumer.py").write_text(
        "from src.utils import helper\n\ndef consume(): helper()\n",
        encoding="utf-8",
    )

    engine = LabelingEngine(FakeClient(), LabelingConfig())
    pipeline = CradlePipeline(config=PipelineConfig(), labeling_engine=engine)

    graph = pipeline.analyze(tmp_path)

    # The import of the missing module must be flagged out-of-scope.
    oos = [
        record
        for record in graph.imports
        if record.importer_node_id == "src/service.py" and record.imported_module == "src.external_pkg.utils"
    ]
    assert oos, "expected an import record for src.external_pkg.utils"
    assert oos[0].is_out_of_scope is True
    assert oos[0].target_node_id is None

    # The resolvable sibling import must NOT be out-of-scope.
    resolved = [
        record
        for record in graph.imports
        if record.importer_node_id == "src/consumer.py" and record.imported_module == "src.utils"
    ]
    assert resolved, "expected an import record for src.utils"
    assert resolved[0].is_out_of_scope is False
    assert resolved[0].target_node_id is not None