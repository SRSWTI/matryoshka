from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CodeNode:
    node_id: str
    path: str
    name: str
    kind: str
    parent_id: str | None
    language: str | None = None
    summary: str = ""
    description: str = ""
    primary_category: str | None = None
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    start_line: int | None = None
    start_column: int | None = None
    end_line: int | None = None
    end_column: int | None = None
    symbol_count: int = 0
    import_count: int = 0
    file_count: int = 0
    folder_count: int = 0
    content_hash: str | None = None


@dataclass(slots=True)
class CodeSymbol:
    symbol_id: str
    node_id: str
    path: str
    name: str
    qualified_name: str
    normalized_name: str
    kind: str
    signature: str
    parent_name: str | None = None
    return_type: str | None = None
    docstring: str | None = None
    parameters: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    base_classes: list[str] = field(default_factory=list)
    start_line: int | None = None
    start_column: int | None = None
    end_line: int | None = None
    end_column: int | None = None


@dataclass(slots=True)
class ImportRecord:
    importer_node_id: str
    imported_module: str
    target_node_id: str | None
    is_internal: bool
    strength_label: str
    strength_weight: float
    # True when the import looks internal (same package / relative) but the
    # target file does not exist within the analyzed root.  The dependency is
    # real but lives outside the Cradle analysis scope.
    is_out_of_scope: bool = False
    names: list[str] = field(default_factory=list)
    start_line: int | None = None
    start_column: int | None = None
    end_line: int | None = None
    end_column: int | None = None


@dataclass(slots=True)
class CallRecord:
    caller_symbol_id: str
    caller_node_id: str
    callee_name: str
    target_symbol_id: str | None
    target_node_id: str | None
    start_line: int | None = None
    start_column: int | None = None
    end_line: int | None = None
    end_column: int | None = None


@dataclass(slots=True)
class SymbolReferenceRecord:
    target_symbol_id: str | None
    target_node_id: str | None
    target_name: str
    source_node_id: str
    source_symbol_id: str | None
    reference_kind: str
    start_line: int | None = None
    start_column: int | None = None
    end_line: int | None = None
    end_column: int | None = None


@dataclass(slots=True)
class NodeContextRecord:
    node_id: str
    source_node_id: str
    strength_label: str
    strength_weight: float
    inherited_summary: str
    inherited_category: str | None
    inherited_tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CommunityMemberRecord:
    community_node_id: str
    member_node_id: str
    membership_rank: int
    membership_weight: float


@dataclass(slots=True)
class ThemeMemberRecord:
    theme_node_id: str
    member_node_id: str
    membership_rank: int
    membership_weight: float


@dataclass(slots=True)
class RepositoryGraph:
    repo_root: str
    nodes: list[CodeNode] = field(default_factory=list)
    symbols: list[CodeSymbol] = field(default_factory=list)
    imports: list[ImportRecord] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    references: list[SymbolReferenceRecord] = field(default_factory=list)
    node_context: list[NodeContextRecord] = field(default_factory=list)
    community_members: list[CommunityMemberRecord] = field(default_factory=list)
    theme_members: list[ThemeMemberRecord] = field(default_factory=list)


@dataclass(slots=True)
class AnalysisSummary:
    repo_root: str
    file_count: int
    folder_count: int
    symbol_count: int
    import_count: int
    call_count: int
    reference_count: int
    repo_summary: str = ""
    repo_categories: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalNodeHit:
    score: float
    node: CodeNode
    contexts: list[NodeContextRecord] = field(default_factory=list)
    imports: list[ImportRecord] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalSymbolHit:
    score: float
    symbol: CodeSymbol
    references: list[SymbolReferenceRecord] = field(default_factory=list)
    callees: list[CallRecord] = field(default_factory=list)
    called_by: list[CallRecord] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalResult:
    query: str
    node_hits: list[RetrievalNodeHit] = field(default_factory=list)
    symbol_hits: list[RetrievalSymbolHit] = field(default_factory=list)


@dataclass(slots=True)
class ExactImportHit:
    score: float
    import_record: ImportRecord
    importer_node: CodeNode
    target_node: CodeNode | None = None


@dataclass(slots=True)
class ExactCallHit:
    score: float
    call_record: CallRecord
    caller_node: CodeNode | None = None
    caller_symbol: CodeSymbol | None = None
    target_node: CodeNode | None = None
    target_symbol: CodeSymbol | None = None


@dataclass(slots=True)
class ExactReferenceHit:
    score: float
    reference_record: SymbolReferenceRecord
    source_node: CodeNode | None = None
    source_symbol: CodeSymbol | None = None
    target_node: CodeNode | None = None
    target_symbol: CodeSymbol | None = None


@dataclass(slots=True)
class ExactSearchResult:
    query: str
    search_type: str
    node_hits: list[RetrievalNodeHit] = field(default_factory=list)
    symbol_hits: list[RetrievalSymbolHit] = field(default_factory=list)
    import_hits: list[ExactImportHit] = field(default_factory=list)
    call_hits: list[ExactCallHit] = field(default_factory=list)
    reference_hits: list[ExactReferenceHit] = field(default_factory=list)


@dataclass(slots=True)
class TraversalCandidate:
    score: float
    node: CodeNode


@dataclass(slots=True)
class TraversalStep:
    level: str
    parent_node_ids: list[str] = field(default_factory=list)
    candidates: list[TraversalCandidate] = field(default_factory=list)


@dataclass(slots=True)
class HierarchicalSearchResult:
    query: str
    steps: list[TraversalStep] = field(default_factory=list)
    node_hits: list[RetrievalNodeHit] = field(default_factory=list)
    symbol_hits: list[RetrievalSymbolHit] = field(default_factory=list)


@dataclass(slots=True)
class CodeExcerpt:
    path: str
    start_line: int
    end_line: int
    text: str


@dataclass(slots=True)
class QuestionResult:
    query: str
    answer: str
    traversal_steps: list[TraversalStep] = field(default_factory=list)
    node_hits: list[RetrievalNodeHit] = field(default_factory=list)
    symbol_hits: list[RetrievalSymbolHit] = field(default_factory=list)
    import_hits: list[ExactImportHit] = field(default_factory=list)
    call_hits: list[ExactCallHit] = field(default_factory=list)
    reference_hits: list[ExactReferenceHit] = field(default_factory=list)
    excerpts: list[CodeExcerpt] = field(default_factory=list)