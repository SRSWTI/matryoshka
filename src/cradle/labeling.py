from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cradle.cache import LabelCache
from cradle.llm_client import OpenAICompatibleClient
from cradle.models import ConsistencyReport, FilePacket, LabelResult, NodePacket
from cradle.prompts import DEFAULT_TAXONOMY, build_confirmation_messages, build_file_label_messages, build_node_label_messages


@dataclass(slots=True)
class LabelingConfig:
    taxonomy: tuple[str, ...] = DEFAULT_TAXONOMY
    temperature: float = 0.0
    max_tokens: int = 600


class LabelingEngine:
    def __init__(self, client: OpenAICompatibleClient, config: LabelingConfig | None = None, cache: LabelCache | None = None) -> None:
        self._client = client
        self._config = config or LabelingConfig()
        self._cache = cache

    def label_files(self, packets: list[FilePacket], progress: Callable[[str], None] | None = None) -> dict[str, LabelResult]:
        requests_payload = [(packet, build_file_label_messages(packet, self._config.taxonomy)) for packet in packets]
        return self._resolve_file_labels(requests_payload, progress=progress)

    def label_node(self, packet: NodePacket) -> LabelResult:
        messages = build_node_label_messages(packet, self._config.taxonomy)
        cached = self._load_cached_result(packet.level, packet.node_id, messages)
        if cached is not None:
            return cached
        response = self._client.create_chat_completion(messages, temperature=self._config.temperature, max_tokens=self._config.max_tokens)
        result = _coerce_label_result(packet.level, packet.node_id, response, self._config.taxonomy)
        self._store_cached_result(packet.level, packet.node_id, messages, result)
        return result

    def label_nodes(self, packets: list[NodePacket], progress: Callable[[str], None] | None = None) -> dict[str, LabelResult]:
        requests_payload = [(packet, build_node_label_messages(packet, self._config.taxonomy)) for packet in packets]
        return self._resolve_node_labels(requests_payload, confirmation_suffix=None, progress=progress)

    def confirm_node(self, packet: NodePacket, repo_label: LabelResult) -> LabelResult:
        messages = build_confirmation_messages(packet, repo_label.summary, repo_label.categories, self._config.taxonomy)
        scope = f"{packet.level}-confirmation"
        cached = self._load_cached_result(scope, packet.node_id, messages)
        if cached is not None:
            return cached
        response = self._client.create_chat_completion(messages, temperature=self._config.temperature, max_tokens=self._config.max_tokens)
        result = _coerce_label_result(scope, packet.node_id, response, self._config.taxonomy)
        self._store_cached_result(scope, packet.node_id, messages, result)
        return result

    def confirm_nodes(self, packets: list[NodePacket], repo_label: LabelResult, progress: Callable[[str], None] | None = None) -> dict[str, LabelResult]:
        requests_payload = [
            (packet, build_confirmation_messages(packet, repo_label.summary, repo_label.categories, self._config.taxonomy))
            for packet in packets
        ]
        return self._resolve_node_labels(requests_payload, confirmation_suffix="confirmation", progress=progress)

    def compare_labels(self, bottom_up: LabelResult, top_down: LabelResult) -> ConsistencyReport:
        bottom = set(bottom_up.categories)
        top = set(top_down.categories)
        return ConsistencyReport(
            target_id=bottom_up.target_id,
            agreed_categories=sorted(bottom & top),
            bottom_up_only=sorted(bottom - top),
            top_down_only=sorted(top - bottom),
            confidence_delta=top_down.confidence - bottom_up.confidence,
        )

    def flush_cache(self) -> None:
        if self._cache is not None:
            self._cache.save()

    def _resolve_file_labels(
        self,
        requests_payload: list[tuple[FilePacket, list[dict[str, str]]]],
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, LabelResult]:
        resolved: dict[str, LabelResult] = {}
        pending_packets: list[FilePacket] = []
        pending_messages: list[list[dict[str, str]]] = []
        total_packets = len(requests_payload)
        for packet, messages in requests_payload:
            cached = self._load_cached_result("file", packet.path, messages)
            if cached is not None:
                resolved[packet.path] = cached
                continue
            pending_packets.append(packet)
            pending_messages.append(messages)

        notifier = _ProgressNotifier(progress, prefix="resolved file labels", offset=len(resolved), total=total_packets)
        notifier.emit_cached()

        if pending_messages:
            def store_result(index: int, response: dict[str, object], completed: int, total: int) -> None:
                packet = pending_packets[index]
                messages = pending_messages[index]
                result = _coerce_label_result("file", packet.path, response, self._config.taxonomy)
                resolved[packet.path] = result
                self._store_cached_result("file", packet.path, messages, result)

            self._client.create_many_chat_completions(
                pending_messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                progress_callback=notifier.callback,
                result_callback=store_result,
            )
        return resolved

    def _resolve_node_labels(
        self,
        requests_payload: list[tuple[NodePacket, list[dict[str, str]]]],
        confirmation_suffix: str | None,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, LabelResult]:
        resolved: dict[str, LabelResult] = {}
        pending_packets: list[NodePacket] = []
        pending_messages: list[list[dict[str, str]]] = []
        total_packets = len(requests_payload)
        for packet, messages in requests_payload:
            scope = f"{packet.level}-{confirmation_suffix}" if confirmation_suffix is not None else packet.level
            cached = self._load_cached_result(scope, packet.node_id, messages)
            if cached is not None:
                resolved[packet.node_id] = cached
                continue
            pending_packets.append(packet)
            pending_messages.append(messages)

        label_kind = "confirmed nodes" if confirmation_suffix is not None else "node labels"
        notifier = _ProgressNotifier(progress, prefix=f"resolved {label_kind}", offset=len(resolved), total=total_packets)
        notifier.emit_cached()

        if pending_messages:
            def store_result(index: int, response: dict[str, object], completed: int, total: int) -> None:
                packet = pending_packets[index]
                messages = pending_messages[index]
                scope = f"{packet.level}-{confirmation_suffix}" if confirmation_suffix is not None else packet.level
                result = _coerce_label_result(scope, packet.node_id, response, self._config.taxonomy)
                resolved[packet.node_id] = result
                self._store_cached_result(scope, packet.node_id, messages, result)

            self._client.create_many_chat_completions(
                pending_messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                progress_callback=notifier.callback,
                result_callback=store_result,
            )
        return resolved

    def _load_cached_result(self, scope: str, target_id: str, messages: list[dict[str, str]]) -> LabelResult | None:
        if self._cache is None:
            return None
        cache_key = self._cache.build_key(scope, target_id, messages, self._client.model)
        return self._cache.get(cache_key)

    def _store_cached_result(self, scope: str, target_id: str, messages: list[dict[str, str]], result: LabelResult) -> None:
        if self._cache is None:
            return
        cache_key = self._cache.build_key(scope, target_id, messages, self._client.model)
        self._cache.set(cache_key, result)


def _coerce_label_result(scope: str, target_id: str, response: dict[str, object], taxonomy: tuple[str, ...]) -> LabelResult:
    if not isinstance(response, dict):
        response = {
            "summary": f"Malformed model response for {target_id}",
            "description": "The model did not return a JSON object matching the expected schema.",
            "tags": [],
            "categories": [],
            "confidence": 0.0,
            "evidence": [f"response_type={type(response).__name__}"],
        }

    raw_tags = response.get("tags", [])
    raw_categories = response.get("categories", [])
    tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    categories = [str(category).strip() for category in raw_categories if str(category).strip()]
    normalized_categories = [_normalize_category(category, taxonomy) for category in categories]
    normalized_categories = [category for category in normalized_categories if category is not None]
    summary = str(response.get("summary", "")).strip()
    description = str(response.get("description", "")).strip()
    evidence_raw = response.get("evidence", [])
    evidence = [str(item).strip() for item in evidence_raw if str(item).strip()]

    if not summary:
        summary = f"Classification result for {target_id}"
    if not description:
        description = f"No detailed description was returned for {target_id}."

    confidence_raw = response.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    return LabelResult(
        scope=scope,
        target_id=target_id,
        summary=summary,
        description=description,
        tags=tags,
        categories=normalized_categories,
        confidence=confidence,
        evidence=evidence,
    )


def _normalize_category(category: str, taxonomy: tuple[str, ...]) -> str | None:
    aliases = {
        "tooling": "developer-tooling",
        "developer-tools": "developer-tooling",
        "developer-utilities": "developer-tooling",
        "code-intelligence": "code-analysis",
        "analysis": "code-analysis",
        "llm": "llm-integration",
        "ai-integration": "llm-integration",
        "cli": "cli-tooling",
        "command-line": "cli-tooling",
        "orchestration": "pipeline-orchestration",
        "data-models": "types-and-models",
        "models": "types-and-models",
        "cache": "caching",
    }
    candidate = aliases.get(category, category)
    if candidate in taxonomy:
        return candidate
    return None


class _ProgressNotifier:
    def __init__(self, emit: Callable[[str], None] | None, *, prefix: str, offset: int, total: int, step: int = 10) -> None:
        self._emit = emit
        self._prefix = prefix
        self._offset = offset
        self._total = total
        self._step = step
        self._last_reported = 0

    def emit_cached(self) -> None:
        if self._emit is None or self._offset == 0:
            return
        self._last_reported = self._offset
        self._emit(f"{self._prefix} {self._offset}/{self._total} (cache)")

    def callback(self, completed: int, total_pending: int) -> None:
        if self._emit is None:
            return
        current = self._offset + completed
        if current == self._total or current - self._last_reported >= self._step:
            self._last_reported = current
            self._emit(f"{self._prefix} {current}/{self._total}")