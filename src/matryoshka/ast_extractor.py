from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import tree_sitter_python as tree_sitter_python
import tree_sitter_typescript as tree_sitter_typescript
from tree_sitter import Language, Node, Parser


@dataclass(slots=True)
class LineRange:
    start_line: int
    start_column: int
    end_line: int
    end_column: int


@dataclass(slots=True)
class ImportEdge:
    importer: str
    imported_module: str
    is_internal: bool
    alias: str | None = None
    names: list[str] = field(default_factory=list)
    line_range: LineRange | None = None


@dataclass(slots=True)
class CallSiteRecord:
    caller_name: str
    callee_name: str
    line_range: LineRange


@dataclass(slots=True)
class SymbolRecord:
    name: str
    kind: str
    signature: str
    line_range: LineRange
    parent: str | None = None
    return_type: str | None = None
    parameters: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    base_classes: list[str] = field(default_factory=list)
    docstring: str | None = None
    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)


class SymbolCapture(NamedTuple):
    record: SymbolRecord
    node: Node


@dataclass(slots=True)
class FileExtraction:
    language: str
    path: str
    symbols: list[SymbolRecord]
    import_edges: list[ImportEdge]
    external_packages: list[str]
    call_sites: list[CallSiteRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PYTHON_LANGUAGE = Language(tree_sitter_python.language())
TYPESCRIPT_LANGUAGE = Language(tree_sitter_typescript.language_typescript())

PARSERS = {
    ".py": ("python", Parser(PYTHON_LANGUAGE)),
    ".ts": ("typescript", Parser(TYPESCRIPT_LANGUAGE)),
    ".tsx": ("typescript", Parser(TYPESCRIPT_LANGUAGE)),
}


def extract_file(file_path: str | Path, source: str | None = None, repo_root: str | Path | None = None) -> FileExtraction:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in PARSERS:
        raise ValueError(f"Unsupported file type: {suffix}")

    language_name, parser = PARSERS[suffix]
    source_text = source if source is not None else path.read_text(encoding="utf-8")
    tree = parser.parse(source_text.encode("utf-8"))
    root = tree.root_node
    root_dir = Path(repo_root) if repo_root is not None else path.parent

    if language_name == "python":
        return _extract_python(path, source_text, root, root_dir)
    return _extract_typescript(path, source_text, root, root_dir)


def _extract_python(path: Path, source: str, root: Node, repo_root: Path) -> FileExtraction:
    captures: list[SymbolCapture] = []
    imports: list[ImportEdge] = []

    for child in root.children:
        if child.type == "import_statement":
            imports.extend(_python_import_statement(path, source, child, repo_root))
        elif child.type == "import_from_statement":
            imports.extend(_python_import_from_statement(path, source, child, repo_root))
        elif child.type == "decorated_definition":
            captures.extend(_python_decorated_symbols(source, child))
        elif child.type == "function_definition":
            captures.append(SymbolCapture(_python_function_symbol(source, child, parent=None), child))
        elif child.type == "class_definition":
            captures.extend(_python_class_symbols(source, child))
        elif child.type == "expression_statement":
            assignment = child.child(0)
            if assignment is not None and assignment.type == "assignment":
                symbol = _python_assignment_symbol(source, assignment)
                if symbol is not None:
                    captures.append(SymbolCapture(symbol, assignment))

    symbols, call_sites = _annotate_call_graph(source, captures, "python")
    external_packages = sorted({edge.imported_module.split(".", 1)[0] for edge in imports if not edge.is_internal})
    return FileExtraction("python", str(path), symbols, imports, external_packages, call_sites)


def _extract_typescript(path: Path, source: str, root: Node, repo_root: Path) -> FileExtraction:
    captures: list[SymbolCapture] = []
    imports: list[ImportEdge] = []

    for child in root.children:
        if child.type == "import_statement":
            edge = _typescript_import_statement(path, source, child, repo_root)
            if edge is not None:
                imports.append(edge)
        elif child.type == "export_statement":
            edge = _typescript_export_statement(path, source, child)
            if edge is not None:
                imports.append(edge)
            captures.extend(_typescript_export_symbols(source, child))
        elif child.type in {"function_declaration", "class_declaration", "interface_declaration", "enum_declaration", "type_alias_declaration"}:
            symbol = _typescript_declaration_symbol(source, child, parent=None)
            if symbol is not None:
                captures.append(SymbolCapture(symbol, child))
                if child.type == "class_declaration":
                    captures.extend(_typescript_class_members(source, child, symbol.name))
        elif child.type == "lexical_declaration":
            captures.extend(_typescript_variable_symbols(source, child))

    symbols, call_sites = _annotate_call_graph(source, captures, "typescript")
    external_packages = sorted({edge.imported_module.split("/", 1)[0] for edge in imports if not edge.is_internal})
    return FileExtraction("typescript", str(path), symbols, imports, external_packages, call_sites)


def _python_import_statement(path: Path, source: str, node: Node, repo_root: Path) -> list[ImportEdge]:
    edges: list[ImportEdge] = []
    for child in node.children:
        if child.type == "dotted_name":
            module = _text(source, child)
            edges.append(ImportEdge(str(path), module, _is_python_internal(module, repo_root), names=[module], line_range=_line_range(child)))
        elif child.type == "aliased_import":
            name_node = child.child_by_field_name("name") or child.child(0)
            alias_node = child.child_by_field_name("alias") or child.child(child.child_count - 1)
            if name_node is None:
                continue
            module = _text(source, name_node)
            alias = _text(source, alias_node) if alias_node is not None else None
            edges.append(
                ImportEdge(
                    str(path),
                    module,
                    _is_python_internal(module, repo_root),
                    alias=alias,
                    names=[module],
                    line_range=_line_range(name_node),
                )
            )
    return edges


def _python_import_from_statement(path: Path, source: str, node: Node, repo_root: Path) -> list[ImportEdge]:
    module_name = ""
    imported_names: list[str] = []
    for child in node.children:
        if child.type == "dotted_name" and not module_name:
            module_name = _text(source, child)
        elif child.type == "dotted_name":
            imported_names.append(_text(source, child))
        elif child.type == "wildcard_import":
            imported_names.append("*")
        elif child.type == "aliased_import":
            name_node = child.child_by_field_name("name") or child.child(0)
            if name_node is not None:
                imported_names.append(_text(source, name_node))
        elif child.type == "identifier":
            imported_names.append(_text(source, child))

    if not module_name:
        return []

    return [
        ImportEdge(
            importer=str(path),
            imported_module=module_name,
            is_internal=_is_python_internal(module_name, repo_root),
            names=imported_names,
            line_range=_line_range(node),
        )
    ]


def _python_decorated_symbols(source: str, node: Node) -> list[SymbolCapture]:
    definition = next((child for child in node.children if child.type in {"function_definition", "class_definition"}), None)
    if definition is None:
        return []
    if definition.type == "function_definition":
        return [SymbolCapture(_python_function_symbol(source, definition, parent=None), definition)]
    return _python_class_symbols(source, definition)


def _python_function_symbol(source: str, node: Node, parent: str | None) -> SymbolRecord:
    name_node = node.child_by_field_name("name")
    params_node = node.child_by_field_name("parameters")
    return_node = node.child_by_field_name("return_type")
    name = _text(source, name_node)
    parameters = _python_parameter_list(source, params_node)
    decorators = _python_decorators(source, node)
    signature = f"def {name}({', '.join(parameters)})"
    if return_node is not None:
        signature += f" -> {_text(source, return_node)}"

    return SymbolRecord(
        name=name,
        kind="method" if parent else "function",
        signature=signature,
        line_range=_line_range(node),
        parent=parent,
        return_type=_python_return_type(source, return_node),
        parameters=parameters,
        decorators=decorators,
        docstring=_python_docstring(source, node),
    )


def _python_class_symbols(source: str, node: Node) -> list[SymbolCapture]:
    name_node = node.child_by_field_name("name")
    superclasses_node = node.child_by_field_name("superclasses")
    class_name = _text(source, name_node)
    bases = []
    if superclasses_node is not None:
        bases = [_text(source, child) for child in superclasses_node.children if child.type not in {"(", ")", ","}]

    class_symbol = SymbolRecord(
        name=class_name,
        kind="class",
        signature=f"class {class_name}{_text(source, superclasses_node) if superclasses_node is not None else ''}",
        line_range=_line_range(node),
        base_classes=bases,
        decorators=_python_decorators(source, node),
        docstring=_python_docstring(source, node),
    )

    nested_symbols = [SymbolCapture(class_symbol, node)]
    body = node.child_by_field_name("body")
    if body is None:
        return nested_symbols

    for child in body.children:
        if child.type == "decorated_definition":
            nested_symbols.extend(_python_class_decorated_symbols(source, child, class_name))
        if child.type == "function_definition":
            nested_symbols.append(SymbolCapture(_python_function_symbol(source, child, parent=class_name), child))
        elif child.type == "expression_statement":
            assignment = child.child(0)
            if assignment is not None and assignment.type == "assignment":
                symbol = _python_assignment_symbol(source, assignment, parent=class_name)
                if symbol is not None:
                    nested_symbols.append(SymbolCapture(symbol, assignment))
    return nested_symbols


def _python_class_decorated_symbols(source: str, node: Node, parent: str) -> list[SymbolCapture]:
    definition = next((child for child in node.children if child.type in {"function_definition", "class_definition"}), None)
    if definition is None:
        return []
    if definition.type == "function_definition":
        return [SymbolCapture(_python_function_symbol(source, definition, parent=parent), definition)]
    return _python_class_symbols(source, definition)


def _python_assignment_symbol(source: str, node: Node, parent: str | None = None) -> SymbolRecord | None:
    left = node.child_by_field_name("left")
    right = node.child_by_field_name("right")
    if left is None or left.type != "identifier":
        return None
    name = _text(source, left)
    return SymbolRecord(
        name=name,
        kind="constant" if name.isupper() else "variable",
        signature=f"{name} = {_text(source, right) if right is not None else ''}".strip(),
        line_range=_line_range(node),
        parent=parent,
    )


def _python_parameter_list(source: str, node: Node | None) -> list[str]:
    if node is None:
        return []
    return [_text(source, child) for child in node.children if child.type not in {"(", ")", ","}]


def _python_decorators(source: str, node: Node) -> list[str]:
    decorators: list[str] = []
    current = node.prev_named_sibling
    while current is not None and current.type == "decorator":
        decorators.append(_text(source, current))
        current = current.prev_named_sibling
    decorators.reverse()
    return decorators


def _python_docstring(source: str, node: Node) -> str | None:
    body = node.child_by_field_name("body")
    if body is None or body.named_child_count == 0:
        return None
    first = body.named_child(0)
    if first is None or first.type != "expression_statement":
        return None
    literal = first.named_child(0)
    if literal is None or literal.type not in {"string", "concatenated_string"}:
        return None
    return _clean_string_literal(_text(source, literal))


def _typescript_import_statement(path: Path, source: str, node: Node, repo_root: Path) -> ImportEdge | None:
    source_node = node.child_by_field_name("source")
    if source_node is None:
        source_node = next((child for child in node.children if child.type == "string"), None)
    if source_node is None:
        return None
    module_name = _clean_string_literal(_text(source, source_node))
    names: list[str] = []
    import_clause = node.child_by_field_name("import_clause")
    if import_clause is None:
        import_clause = next((child for child in node.children if child.type == "import_clause"), None)
    if import_clause is not None:
        names.extend(_typescript_import_names(source, import_clause))
    return ImportEdge(
        importer=str(path),
        imported_module=module_name,
        is_internal=_is_typescript_internal(module_name),
        names=names,
        line_range=_line_range(node),
    )


def _typescript_export_statement(path: Path, source: str, node: Node) -> ImportEdge | None:
    source_node = node.child_by_field_name("source")
    if source_node is None:
        source_node = next((child for child in node.children if child.type == "string"), None)
    if source_node is None:
        return None
    module_name = _clean_string_literal(_text(source, source_node))
    names = _typescript_export_names(source, node)
    return ImportEdge(
        importer=str(path),
        imported_module=module_name,
        is_internal=_is_typescript_internal(module_name),
        names=names,
        line_range=_line_range(node),
    )


def _typescript_export_symbols(source: str, node: Node) -> list[SymbolCapture]:
    captures: list[SymbolCapture] = []
    declaration = next(
        (
            child
            for child in node.children
            if child.type in {"lexical_declaration", "function_declaration", "class_declaration", "interface_declaration", "enum_declaration", "type_alias_declaration"}
        ),
        None,
    )
    if declaration is None:
        return captures
    if declaration.type == "lexical_declaration":
        return _typescript_variable_symbols(source, declaration)

    symbol = _typescript_declaration_symbol(source, declaration, parent=None)
    if symbol is None:
        return captures
    captures.append(SymbolCapture(symbol, declaration))
    if declaration.type == "class_declaration":
        captures.extend(_typescript_class_members(source, declaration, symbol.name))
    return captures


def _typescript_import_names(source: str, node: Node) -> list[str]:
    names: list[str] = []
    for child in node.children:
        if child.type in {"identifier", "namespace_import"}:
            names.append(_text(source, child))
        elif child.type == "named_imports":
            for named_child in child.children:
                if named_child.type in {"import_specifier", "identifier"}:
                    names.append(_text(source, named_child))
    return names


def _typescript_export_names(source: str, node: Node) -> list[str]:
    export_clause = next((child for child in node.children if child.type == "export_clause"), None)
    if export_clause is None:
        if any(child.type == "*" for child in node.children):
            return ["*"]
        return []
    names: list[str] = []
    for child in export_clause.children:
        if child.type in {"export_specifier", "identifier"}:
            names.append(_text(source, child))
    return names


def _typescript_declaration_symbol(source: str, node: Node, parent: str | None) -> SymbolRecord | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    name = _text(source, name_node)
    kind_map = {
        "function_declaration": "function",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "type_alias_declaration": "type_alias",
    }
    kind = kind_map[node.type]
    signature = _single_line(_text(source, node).split("{", 1)[0].strip())
    return SymbolRecord(
        name=name,
        kind=kind,
        signature=signature,
        line_range=_line_range(node),
        parent=parent,
    )


def _typescript_class_members(source: str, node: Node, parent: str) -> list[SymbolCapture]:
    members: list[SymbolCapture] = []
    body = node.child_by_field_name("body")
    if body is None:
        return members
    for child in body.named_children:
        if child.type == "method_definition":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            name = _text(source, name_node)
            params_node = child.child_by_field_name("parameters")
            return_node = child.child_by_field_name("return_type")
            params = _typescript_parameter_list(source, params_node)
            signature = f"{name}({', '.join(params)})"
            normalized_return_type = _typescript_return_type(source, return_node)
            if normalized_return_type is not None:
                signature += f": {normalized_return_type}"
            members.append(
                SymbolCapture(
                    SymbolRecord(
                    name=name,
                    kind="method",
                    signature=signature,
                    line_range=_line_range(child),
                    parent=parent,
                    return_type=normalized_return_type,
                    parameters=params,
                    ),
                    child,
                )
            )
        elif child.type in {"public_field_definition", "property_signature"}:
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            name = _text(source, name_node)
            members.append(
                SymbolCapture(
                    SymbolRecord(
                    name=name,
                    kind="field",
                    signature=_single_line(_text(source, child)),
                    line_range=_line_range(child),
                    parent=parent,
                    ),
                    child,
                )
            )
    return members


def _typescript_variable_symbols(source: str, node: Node) -> list[SymbolCapture]:
    symbols: list[SymbolCapture] = []
    for child in node.named_children:
        if child.type != "variable_declarator":
            continue
        name_node = child.child_by_field_name("name")
        value_node = child.child_by_field_name("value")
        if name_node is None:
            continue
        name = _text(source, name_node)
        symbols.append(
            SymbolCapture(
                SymbolRecord(
                name=name,
                kind="constant" if name.isupper() else "variable",
                signature=f"{name} = {_typescript_variable_signature(source, value_node) if value_node is not None else ''}".strip(),
                line_range=_line_range(child),
                ),
                child,
            )
        )
    return symbols


def _typescript_variable_signature(source: str, value_node: Node) -> str:
    text = _single_line(_text(source, value_node))
    if value_node.type in {"arrow_function", "function_expression"}:
        body_node = value_node.child_by_field_name("body")
        if body_node is not None and body_node.type == "statement_block":
            return _single_line(_text(source, value_node).split("{", 1)[0].strip()) + " { ... }"
        return text
    if value_node.type == "object":
        return "{ ... }"
    if value_node.type == "array":
        return "[ ... ]"
    if len(text) > 160 and "{" in text:
        return text.split("{", 1)[0].rstrip() + " { ... }"
    return text


def _typescript_parameter_list(source: str, node: Node | None) -> list[str]:
    if node is None:
        return []
    return [_text(source, child) for child in node.named_children]


def _is_python_internal(module_name: str, repo_root: Path) -> bool:
    first_segment = module_name.split(".", 1)[0]
    dotted_path = repo_root / Path(*module_name.split("."))
    return (
        (repo_root / f"{first_segment}.py").exists()
        or (repo_root / first_segment).is_dir()
        or dotted_path.with_suffix(".py").exists()
        or dotted_path.is_dir()
    )


def _is_typescript_internal(module_name: str) -> bool:
    return module_name.startswith(("./", "../", "/", "@/"))


def _text(source: str, node: Node | None) -> str:
    if node is None:
        return ""
    return _source_bytes(source)[node.start_byte:node.end_byte].decode("utf-8")


@lru_cache(maxsize=128)
def _source_bytes(source: str) -> bytes:
    return source.encode("utf-8")


def _line_range(node: Node) -> LineRange:
    return LineRange(
        start_line=node.start_point[0] + 1,
        start_column=node.start_point[1] + 1,
        end_line=node.end_point[0] + 1,
        end_column=node.end_point[1] + 1,
    )


def _clean_string_literal(text: str) -> str:
    return text.strip().strip("\"'")


def _single_line(text: str) -> str:
    return " ".join(text.split())


def _python_return_type(source: str, node: Node | None) -> str | None:
    if node is None:
        return None
    return _text(source, node).removeprefix("->").strip()


def _typescript_return_type(source: str, node: Node | None) -> str | None:
    if node is None:
        return None
    return _text(source, node).removeprefix(":").strip()


def _annotate_call_graph(source: str, captures: list[SymbolCapture], language: str) -> tuple[list[SymbolRecord], list[CallSiteRecord]]:
    symbol_by_name = {capture.record.name: capture.record for capture in captures}
    incoming: dict[str, Counter[str]] = {name: Counter() for name in symbol_by_name}
    call_sites: list[CallSiteRecord] = []

    for capture in captures:
        if capture.record.kind not in {"function", "method"}:
            continue
        symbol_call_sites = _collect_call_sites(source, capture.node, language, capture.record.name)
        call_sites.extend(symbol_call_sites)
        call_counts = Counter(site.callee_name for site in symbol_call_sites)
        visible_counts = Counter({name: count for name, count in call_counts.items() if name != capture.record.name})
        capture.record.callees = [name for name, _ in visible_counts.most_common()]
        internal_counts = Counter({name: count for name, count in visible_counts.items() if name in symbol_by_name})
        for callee_name, count in internal_counts.items():
            incoming[callee_name][capture.record.name] += count

    for capture in captures:
        capture.record.callers = [name for name, _ in incoming[capture.record.name].most_common()]

    return [capture.record for capture in captures], call_sites


def _collect_call_sites(source: str, node: Node, language: str, caller_name: str) -> list[CallSiteRecord]:
    output: list[CallSiteRecord] = []
    stack = list(node.named_children)
    while stack:
        current = stack.pop()
        if language == "python" and current.type == "call":
            function_node = current.child_by_field_name("function")
            function_name = _call_target_name(source, function_node, language)
            if function_name:
                output.append(CallSiteRecord(caller_name=caller_name, callee_name=function_name, line_range=_line_range(current)))
        elif language == "typescript" and current.type == "call_expression":
            function_node = current.child_by_field_name("function")
            function_name = _call_target_name(source, function_node, language)
            if function_name:
                output.append(CallSiteRecord(caller_name=caller_name, callee_name=function_name, line_range=_line_range(current)))
        stack.extend(current.named_children)
    return output


def _call_target_name(source: str, node: Node | None, language: str) -> str | None:
    if node is None:
        return None
    if node.type in {"identifier", "property_identifier"}:
        return _text(source, node)
    if language == "python" and node.type == "attribute":
        attribute_node = node.child_by_field_name("attribute")
        if attribute_node is not None:
            return _text(source, attribute_node)
    if language == "typescript" and node.type == "member_expression":
        property_node = node.child_by_field_name("property")
        if property_node is not None:
            return _text(source, property_node)
    return None