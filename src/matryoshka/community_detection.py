from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import networkx as nx

from matryoshka.graph_models import CallRecord, CodeNode, CommunityMemberRecord, ImportRecord, ThemeMemberRecord


def build_louvain_communities(
    nodes: list[CodeNode],
    imports: list[ImportRecord],
    calls: list[CallRecord] | None = None,
) -> tuple[list[CodeNode], list[CommunityMemberRecord]]:
    file_nodes = {node.node_id: node for node in nodes if node.kind == "file"}
    if len(file_nodes) < 2:
        return [], []

    graph = nx.Graph()
    graph.add_nodes_from(file_nodes)
    for record in imports:
        if not record.is_internal:
            continue
        if record.importer_node_id not in file_nodes or record.target_node_id not in file_nodes:
            continue
        if record.importer_node_id == record.target_node_id:
            continue
        weight = max(0.2, float(record.strength_weight))
        if graph.has_edge(record.importer_node_id, record.target_node_id):
            graph[record.importer_node_id][record.target_node_id]["weight"] += weight
        else:
            graph.add_edge(record.importer_node_id, record.target_node_id, weight=weight)

    for record in calls or []:
        if record.caller_node_id not in file_nodes or record.target_node_id not in file_nodes:
            continue
        if record.caller_node_id == record.target_node_id:
            continue
        weight = 0.35
        if graph.has_edge(record.caller_node_id, record.target_node_id):
            graph[record.caller_node_id][record.target_node_id]["weight"] += weight
        else:
            graph.add_edge(record.caller_node_id, record.target_node_id, weight=weight)

    if graph.number_of_edges() == 0:
        return [], []

    communities = [sorted(community) for community in nx.community.louvain_communities(graph, weight="weight", seed=0) if len(community) >= 2]
    if not communities:
        return [], []

    community_nodes: list[CodeNode] = []
    community_members: list[CommunityMemberRecord] = []
    for index, member_ids in enumerate(sorted(communities, key=lambda item: (-len(item), item[0])), start=1):
        members = [file_nodes[member_id] for member_id in member_ids]
        category_counter = Counter(node.primary_category for node in members if node.primary_category)
        tag_counter = Counter(tag for node in members for tag in node.tags)
        folder_counter = Counter(_folder_label(node.path) for node in members)
        dominant_category = category_counter.most_common(1)[0][0] if category_counter else None
        dominant_folders = [folder for folder, _ in folder_counter.most_common(3) if folder]
        representative_paths = sorted(member.path for member in members)[:3]
        slug = _slugify(dominant_category or (dominant_folders[0] if dominant_folders else f"community-{index:02d}"))
        community_id = f"community::{index:02d}::{slug}"
        description = _community_description(dominant_category, dominant_folders, representative_paths, len(members))
        summary = _community_summary(dominant_category, representative_paths, len(members))
        community_nodes.append(
            CodeNode(
                node_id=community_id,
                path=f"communities/{index:02d}-{slug}",
                name=f"community-{index:02d}",
                kind="community",
                parent_id="repo",
                summary=summary,
                description=description,
                primary_category=dominant_category,
                categories=[category for category, _ in category_counter.most_common(4)],
                tags=[tag for tag, _ in tag_counter.most_common(8)],
                confidence=0.65,
                symbol_count=sum(member.symbol_count for member in members),
                import_count=sum(member.import_count for member in members),
                file_count=len(members),
                folder_count=len({member.path.rsplit("/", 1)[0] if "/" in member.path else "" for member in members}),
            )
        )

        weights = _membership_weights(graph, member_ids)
        for rank, member_id in enumerate(sorted(member_ids, key=lambda item: (-weights[item], item)), start=1):
            community_members.append(
                CommunityMemberRecord(
                    community_node_id=community_id,
                    member_node_id=member_id,
                    membership_rank=rank,
                    membership_weight=weights[member_id],
                )
            )

    return community_nodes, community_members


def _membership_weights(graph: nx.Graph, member_ids: list[str]) -> dict[str, float]:
    member_set = set(member_ids)
    weights: dict[str, float] = {}
    for member_id in member_ids:
        weights[member_id] = sum(
            float(attributes.get("weight", 1.0))
            for neighbor_id, attributes in graph[member_id].items()
            if neighbor_id in member_set
        )
    return weights


def _folder_label(path: str) -> str:
    if "/" not in path:
        return ""
    parts = path.split("/")[:-1]
    return "/".join(parts[:2]) if len(parts) >= 2 else parts[0]


def _community_summary(dominant_category: str | None, representative_paths: list[str], member_count: int) -> str:
    if dominant_category and representative_paths:
        return f"Structural import/call community of {member_count} related files in {dominant_category}, including {', '.join(representative_paths)}."
    if representative_paths:
        return f"Structural import/call community of {member_count} related files, including {', '.join(representative_paths)}."
    return f"Structural import/call community of {member_count} related files."


def _community_description(
    dominant_category: str | None,
    dominant_folders: list[str],
    representative_paths: list[str],
    member_count: int,
) -> str:
    folder_text = ", ".join(dominant_folders) if dominant_folders else "multiple folders"
    representative_text = ", ".join(representative_paths) if representative_paths else "no representative files"
    if dominant_category:
        return (
            f"Structural community discovered from internal import and call coupling with {member_count} files clustered around {dominant_category}. "
            f"Dominant folders: {folder_text}. Representative files: {representative_text}."
        )
    return (
        f"Structural community discovered from internal import and call coupling with {member_count} related files. "
        f"Dominant folders: {folder_text}. Representative files: {representative_text}."
    )


def build_theme_domains(nodes: list[CodeNode]) -> tuple[list[CodeNode], list[ThemeMemberRecord]]:
    file_nodes = [node for node in nodes if node.kind == "file" and node.primary_category]
    if len(file_nodes) < 2:
        return [], []

    members_by_category: dict[str, list[CodeNode]] = defaultdict(list)
    for node in file_nodes:
        members_by_category[node.primary_category or ""].append(node)

    theme_nodes: list[CodeNode] = []
    theme_members: list[ThemeMemberRecord] = []
    for category, members in sorted(members_by_category.items()):
        if not category or len(members) < 2:
            continue
        tag_counter = Counter(tag for member in members for tag in member.tags)
        representative_paths = [member.path for member in sorted(members, key=lambda item: (-item.confidence, -item.symbol_count, item.path))[:4]]
        theme_id = f"theme::{_slugify(category)}"
        theme_nodes.append(
            CodeNode(
                node_id=theme_id,
                path=f"themes/{_slugify(category)}",
                name=category,
                kind="theme",
                parent_id="repo",
                summary=_theme_summary(category, representative_paths, len(members)),
                description=_theme_description(category, representative_paths, len(members), tag_counter),
                primary_category=category,
                categories=[category],
                tags=[tag for tag, _ in tag_counter.most_common(8)],
                confidence=round(sum(member.confidence for member in members) / len(members), 3),
                symbol_count=sum(member.symbol_count for member in members),
                import_count=sum(member.import_count for member in members),
                file_count=len(members),
                folder_count=len({member.path.rsplit("/", 1)[0] if "/" in member.path else "" for member in members}),
            )
        )
        ranked_members = sorted(members, key=lambda item: (-max(item.confidence, 0.1), -item.symbol_count, item.path))
        for rank, member in enumerate(ranked_members, start=1):
            theme_members.append(
                ThemeMemberRecord(
                    theme_node_id=theme_id,
                    member_node_id=member.node_id,
                    membership_rank=rank,
                    membership_weight=max(member.confidence, 0.1),
                )
            )

    return theme_nodes, theme_members


def _theme_summary(category: str, representative_paths: list[str], member_count: int) -> str:
    if representative_paths:
        return f"Semantic theme for {category} across {member_count} files, including {', '.join(representative_paths[:3])}."
    return f"Semantic theme for {category} across {member_count} files."


def _theme_description(category: str, representative_paths: list[str], member_count: int, tag_counter: Counter[str]) -> str:
    representatives = ", ".join(representative_paths[:4]) if representative_paths else "no representative files"
    top_tags = ", ".join(tag for tag, _ in tag_counter.most_common(6)) if tag_counter else "no dominant tags"
    return (
        f"Semantic theme/domain grouping for {category} built from labeled file categories. "
        f"It spans {member_count} files with representative files {representatives}. Dominant tags: {top_tags}."
    )


def _slugify(value: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if compact:
        return compact[:40]
    return "community"