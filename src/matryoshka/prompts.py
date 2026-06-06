from __future__ import annotations

import json
from typing import Any

from matryoshka.models import FilePacket, NodePacket

DEFAULT_TAXONOMY = (
    "authentication",
    "authorization",
    "token-management",
    "session-management",
    "database",
    "orm",
    "migrations",
    "api-routing",
    "middleware",
    "background-jobs",
    "caching",
    "configuration",
    "infra",
    "observability",
    "payments",
    "notifications",
    "shared-utils",
    "types-and-models",
    "developer-tooling",
    "code-analysis",
    "llm-integration",
    "cli-tooling",
    "pipeline-orchestration",
)


LABEL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "description",
        "tags",
        "categories",
        "confidence",
        "evidence",
    ],
    "properties": {
        "summary": {"type": "string"},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "categories": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
}


def build_file_label_messages(
    packet: FilePacket, taxonomy: tuple[str, ...] = DEFAULT_TAXONOMY
) -> list[dict[str, str]]:
    system_prompt = "\n".join(
        [
            "You are Matryoshka's file labeling engine.",
            "Your job is to classify a single source file using only the structured evidence provided.",
            "Prefer precise software capability tags over broad topic words.",
            "For developer tooling code, prefer categories like developer-tooling, code-analysis, llm-integration, cli-tooling, pipeline-orchestration, caching, and types-and-models over incidental implementation details.",
            "Do not classify a file as authentication just because it sends an Authorization header or handles API keys for another service.",
            "Choose categories from the allowed taxonomy when possible.",
            "Do not invent APIs, modules, or behavior that is not supported by the packet.",
            "Always populate every required field in the response schema.",
            "Return JSON only.",
        ]
    )

    user_payload = {
        "task": "Label this file for downstream code-intelligence routing.",
        "allowed_categories": list(taxonomy),
        "response_schema": LABEL_RESPONSE_SCHEMA,
        "file_packet": packet.to_prompt_dict(),
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, indent=2, sort_keys=True)},
    ]


def build_node_label_messages(
    packet: NodePacket, taxonomy: tuple[str, ...] = DEFAULT_TAXONOMY
) -> list[dict[str, str]]:
    system_prompt = "\n".join(
        [
            "You are Matryoshka's hierarchy labeling engine.",
            "Your job is to summarize a folder or repository node from aggregated evidence.",
            "Respect the hierarchy: summarize the dominant capabilities without flattening distinct subdomains.",
            "For repository infrastructure and developer tooling, prefer the tooling-oriented categories over generic shared-utils when the packet shows parsing, labeling, orchestration, CLI, or LLM integration.",
            "Prefer categories from the allowed taxonomy and keep tags concrete.",
            "Always populate every required field in the response schema.",
            "Return JSON only.",
        ]
    )

    user_payload = {
        "task": f"Label this {packet.level} node for hierarchical semantic retrieval.",
        "allowed_categories": list(taxonomy),
        "response_schema": LABEL_RESPONSE_SCHEMA,
        "node_packet": packet.to_prompt_dict(),
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, indent=2, sort_keys=True)},
    ]


def build_confirmation_messages(
    packet: NodePacket,
    repo_summary: str,
    repo_categories: list[str],
    taxonomy: tuple[str, ...] = DEFAULT_TAXONOMY,
) -> list[dict[str, str]]:
    system_prompt = "\n".join(
        [
            "You are Matryoshka's top-down consistency labeling engine.",
            "Check a child node against the repository's macro description without forcing false agreement.",
            "Preserve local specificity even when the repository-level summary is broader.",
            "Always populate every required field in the response schema.",
            "Return JSON only.",
        ]
    )

    user_payload = {
        "task": "Confirm or refine this child node label using the repository-level macro context.",
        "allowed_categories": list(taxonomy),
        "response_schema": LABEL_RESPONSE_SCHEMA,
        "repo_context": {
            "summary": repo_summary,
            "categories": repo_categories,
        },
        "node_packet": packet.to_prompt_dict(),
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, indent=2, sort_keys=True)},
    ]
