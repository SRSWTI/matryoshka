from __future__ import annotations

from pathlib import Path

from cradle.exact_search import axe_file_search, axe_symbol_search


def build_focus_visualization(db_path: str | Path, query: str, *, kind: str = "auto", limit: int = 8) -> str:
    if kind not in {"auto", "file", "symbol"}:
        raise ValueError(f"Unsupported focus visualization kind: {kind}")

    file_result = axe_file_search(db_path, query, limit=1) if kind in {"auto", "file"} else None
    symbol_result = axe_symbol_search(db_path, query, limit=1) if kind in {"auto", "symbol"} else None

    file_hit = file_result.node_hits[0] if file_result and file_result.node_hits else None
    symbol_hit = symbol_result.symbol_hits[0] if symbol_result and symbol_result.symbol_hits else None

    if kind == "symbol" or (kind == "auto" and symbol_hit is not None and (file_hit is None or symbol_hit.score >= file_hit.score)):
        if symbol_hit is None:
            return f"# Cradle Focus Visualization\n\nQuery: `{query}`\n\nNo matching symbol found."
        return _render_symbol_focus(query, symbol_hit, limit=limit)

    if file_hit is None:
        return f"# Cradle Focus Visualization\n\nQuery: `{query}`\n\nNo matching file found."
    return _render_file_focus(query, file_hit, limit=limit)


def _render_file_focus(query: str, hit, *, limit: int) -> str:
    node = hit.node
    lines = [
        "# Cradle Focus Visualization",
        "",
        f"Query: `{query}`",
        "",
        f"## File: `{node.path}`",
        "",
        f"Kind: `{node.kind}`",
        f"Category: `{node.primary_category or 'none'}`",
    ]
    if node.summary:
        lines.extend(["", "### Summary", "", node.summary])

    if hit.contexts:
        lines.extend(["", "### Context", ""])
        for context in hit.contexts[:limit]:
            lines.append(f"- `{context.source_node_id}` ({context.strength_label}): {context.inherited_summary}")

    if hit.imports:
        lines.extend(["", "### Imports", ""])
        for record in hit.imports[:limit]:
            target = record.target_node_id or "external"
            lines.append(f"- `{record.imported_module}` -> `{target}` ({record.strength_label})")

    lines.extend(["", "### Neighborhood", "", "```mermaid", "graph TD"])
    lines.append(f'    focus["{node.path}"]')
    for record in hit.imports[:limit]:
        target = record.target_node_id or record.imported_module
        edge_label = record.strength_label
        lines.append(f'    focus -->|{edge_label}| imp_{_mermaid_id(target)}["{target}"]')
    for context in hit.contexts[:limit]:
        lines.append(f'    ctx_{_mermaid_id(context.source_node_id)}["{context.source_node_id}"] -->|context| focus')
    lines.append("```")
    return "\n".join(lines)


def _render_symbol_focus(query: str, hit, *, limit: int) -> str:
    symbol = hit.symbol
    lines = [
        "# Cradle Focus Visualization",
        "",
        f"Query: `{query}`",
        "",
        f"## Symbol: `{symbol.qualified_name}`",
        "",
        f"Path: `{symbol.path}`",
        f"Kind: `{symbol.kind}`",
        f"Signature: `{symbol.signature}`",
    ]

    if hit.called_by:
        lines.extend(["", "### Called By", ""])
        for call in hit.called_by[:limit]:
            lines.append(f"- `{call.caller_node_id}:{call.start_line or 0}`")

    if hit.callees:
        lines.extend(["", "### Callees", ""])
        for call in hit.callees[:limit]:
            lines.append(f"- `{call.callee_name}`")

    if hit.references:
        lines.extend(["", "### References", ""])
        for reference in hit.references[:limit]:
            lines.append(f"- `{reference.source_node_id}:{reference.start_line or 0}` ({reference.reference_kind})")

    lines.extend(["", "### Neighborhood", "", "```mermaid", "graph TD"])
    lines.append(f'    focus["{symbol.qualified_name}"]')
    for call in hit.called_by[:limit]:
        lines.append(f'    caller_{_mermaid_id(call.caller_node_id)}["{call.caller_node_id}"] -->|calls| focus')
    for call in hit.callees[:limit]:
        target = call.target_symbol_id or call.callee_name
        lines.append(f'    focus -->|calls| callee_{_mermaid_id(target)}["{call.callee_name}"]')
    for reference in hit.references[:limit]:
        lines.append(f'    ref_{_mermaid_id(reference.source_node_id)}["{reference.source_node_id}"] -->|{reference.reference_kind}| focus')
    lines.append("```")
    return "\n".join(lines)


def _mermaid_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)