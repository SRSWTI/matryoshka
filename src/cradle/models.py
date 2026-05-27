from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cradle.ast_extractor import FileExtraction


def _trim_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 3]}..."


def _trim_list(values: list[str], *, max_items: int, max_chars: int) -> list[str]:
    return [_trim_text(value, max_chars) for value in values[:max_items]]


def _trim_mapping(values: dict[str, int], *, max_items: int) -> dict[str, int]:
    items = sorted(values.items(), key=lambda item: (-item[1], item[0]))[:max_items]
    return dict(items)


@dataclass(slots=True)
class FilePacket:
    path: str
    language: str
    summary_input: str
    imports_external: list[str] = field(default_factory=list)
    imports_internal: list[str] = field(default_factory=list)
    imported_symbols: list[str] = field(default_factory=list)
    top_symbols: list[str] = field(default_factory=list)
    docstrings: list[str] = field(default_factory=list)
    call_hints: list[str] = field(default_factory=list)
    code_snippets: list[str] = field(default_factory=list)
    import_signature: dict[str, int] = field(default_factory=dict)
    internal_signature: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["summary_input"] = _trim_text(self.summary_input, 1_200)
        payload["imports_external"] = _trim_list(self.imports_external, max_items=20, max_chars=120)
        payload["imports_internal"] = _trim_list(self.imports_internal, max_items=20, max_chars=160)
        payload["imported_symbols"] = _trim_list(self.imported_symbols, max_items=25, max_chars=100)
        payload["top_symbols"] = _trim_list(self.top_symbols, max_items=12, max_chars=200)
        payload["docstrings"] = _trim_list(self.docstrings, max_items=6, max_chars=300)
        payload["call_hints"] = _trim_list(self.call_hints, max_items=20, max_chars=100)
        payload["code_snippets"] = _trim_list(self.code_snippets, max_items=3, max_chars=800)
        payload["import_signature"] = _trim_mapping(self.import_signature, max_items=20)
        payload["internal_signature"] = _trim_mapping(self.internal_signature, max_items=20)
        payload["metadata"] = {
            **self.metadata,
            "prompt_truncated": True,
        }
        return payload


@dataclass(slots=True)
class NodePacket:
    node_id: str
    path: str
    level: str
    child_ids: list[str] = field(default_factory=list)
    child_summaries: list[str] = field(default_factory=list)
    top_tags: list[str] = field(default_factory=list)
    top_external_packages: list[str] = field(default_factory=list)
    top_internal_modules: list[str] = field(default_factory=list)
    representative_symbols: list[str] = field(default_factory=list)
    representative_files: list[str] = field(default_factory=list)
    representative_snippets: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["child_ids"] = _trim_list(self.child_ids, max_items=40, max_chars=160)
        payload["child_summaries"] = _trim_list(self.child_summaries, max_items=20, max_chars=280)
        payload["top_tags"] = _trim_list(self.top_tags, max_items=20, max_chars=80)
        payload["top_external_packages"] = _trim_list(self.top_external_packages, max_items=20, max_chars=120)
        payload["top_internal_modules"] = _trim_list(self.top_internal_modules, max_items=20, max_chars=160)
        payload["representative_symbols"] = _trim_list(self.representative_symbols, max_items=15, max_chars=200)
        payload["representative_files"] = _trim_list(self.representative_files, max_items=15, max_chars=180)
        payload["representative_snippets"] = _trim_list(self.representative_snippets, max_items=4, max_chars=800)
        payload["metadata"] = {
            **self.metadata,
            "prompt_truncated": True,
        }
        return payload


@dataclass(slots=True)
class AnalyzedFile:
    packet: FilePacket
    extraction: FileExtraction
    absolute_path: str
    content_hash: str
    line_count: int


@dataclass(slots=True)
class LabelResult:
    scope: str
    target_id: str
    summary: str
    description: str
    tags: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def primary_category(self) -> str | None:
        return self.categories[0] if self.categories else None


@dataclass(slots=True)
class ConsistencyReport:
    target_id: str
    agreed_categories: list[str] = field(default_factory=list)
    bottom_up_only: list[str] = field(default_factory=list)
    top_down_only: list[str] = field(default_factory=list)
    confidence_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PipelineResult:
    files: dict[str, FilePacket] = field(default_factory=dict)
    nodes: dict[str, NodePacket] = field(default_factory=dict)
    file_labels: dict[str, LabelResult] = field(default_factory=dict)
    node_labels: dict[str, LabelResult] = field(default_factory=dict)
    repo_label: LabelResult | None = None
    consistency: dict[str, ConsistencyReport] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)