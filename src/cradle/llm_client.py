from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any
from urllib import error, request


class LLMClientError(RuntimeError):
    pass


@dataclass(slots=True)
class LLMClientConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 60.0
    max_parallel_requests: int = 4
    send_json_mode: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)


class OpenAICompatibleClient:
    def __init__(self, config: LLMClientConfig) -> None:
        self._config = config

    @property
    def model(self) -> str:
        return self._config.model

    def list_models(self) -> dict[str, Any]:
        return self._request_json("GET", "/v1/models")

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> dict[str, Any]:
        payload = self._build_payload(messages, temperature=temperature, max_tokens=max_tokens)

        try:
            response = self._request_completion_payload(payload)
            content = self._extract_message_content(response)
            return self._parse_json_content(content)
        except LLMClientError as exc:
            if "valid JSON" not in str(exc):
                raise

        retry_payload = self._build_payload(messages, temperature=temperature, max_tokens=max(max_tokens * 2, max_tokens + 256))
        response = self._request_completion_payload(retry_payload)
        content = self._extract_message_content(response)
        return self._parse_json_content(content)

    def _build_payload(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self._config.extra_body,
        }
        if self._config.send_json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _request_completion_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._config.send_json_mode:
            try:
                return self._request_json("POST", "/v1/chat/completions", payload)
            except LLMClientError as exc:
                if "response_format" not in str(exc):
                    raise
                payload_without_format = dict(payload)
                payload_without_format.pop("response_format", None)
                return self._request_json("POST", "/v1/chat/completions", payload_without_format)
        return self._request_json("POST", "/v1/chat/completions", payload)

    def create_many_chat_completions(
        self,
        requests_payload: list[list[dict[str, str]]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 600,
        progress_callback: Callable[[int, int], None] | None = None,
        result_callback: Callable[[int, dict[str, Any], int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        responses: list[dict[str, Any] | None] = [None] * len(requests_payload)
        with ThreadPoolExecutor(max_workers=self._config.max_parallel_requests) as executor:
            future_to_index = {
                executor.submit(
                    self.create_chat_completion,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ): index
                for index, messages in enumerate(requests_payload)
            }
            completed = 0
            total = len(requests_payload)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                response = future.result()
                responses[index] = response
                completed += 1
                if result_callback is not None:
                    result_callback(index, response, completed, total)
                if progress_callback is not None:
                    progress_callback(completed, total)
        return [response for response in responses if response is not None]

    def _request_json(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self._config.base_url.rstrip('/')}{endpoint}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **self._config.extra_headers}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        req = request.Request(url=url, method=method, data=body, headers=headers)
        try:
            with request.urlopen(req, timeout=self._config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"HTTP {exc.code} from LLM endpoint: {body_text}") from exc
        except error.URLError as exc:
            raise LLMClientError(f"Failed to reach LLM endpoint: {exc.reason}") from exc

    def _extract_message_content(self, response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(f"Unexpected completion payload: {response}") from exc
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        raise LLMClientError(f"Unsupported completion content type: {type(content)!r}")

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            extracted = _extract_first_json_object(content)
            if extracted is None:
                raise LLMClientError(f"Completion did not contain valid JSON: {content}")
            return json.loads(extracted)


def _extract_first_json_object(content: str) -> str | None:
    start = content.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]
    return None