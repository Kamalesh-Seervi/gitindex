"""
Extract code symbols (functions / classes / methods) with line ranges from a
source file, so PageIndex can build dir -> file -> symbol tree nodes.

- Python files use the standard library `ast` module (always available).
- Other languages use tree-sitter when `tree_sitter_languages` is installed;
  if it is not installed, those files simply stay file-level (no symbol split).

Each returned symbol is a dict:
    {
        "name": str,
        "kind": "function" | "class" | "method",
        "start_line": int,   # 1-indexed, inclusive
        "end_line": int,     # 1-indexed, inclusive
        "signature": str,    # first source line of the definition
        "docstring": str,    # short leading docstring/comment if available
        "children": [ ...symbols... ],   # e.g. methods of a class
    }
"""

import ast

# Extension -> tree-sitter language name (used only if tree_sitter_languages is present).
_TS_LANG_BY_EXT = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".php": "php",
    ".scala": "scala",
    ".kt": "kotlin",
    ".swift": "swift",
}

# tree-sitter node types that represent a function- or class-like definition.
_TS_DEF_TYPES = {
    "function_declaration",
    "function_definition",
    "method_declaration",
    "method_definition",
    "function_item",
    "class_declaration",
    "class_definition",
    "class_specifier",
    "struct_specifier",
    "struct_item",
    "interface_declaration",
    "impl_item",
    "trait_item",
    "module",
}

_TS_CLASS_TYPES = {
    "class_declaration",
    "class_definition",
    "class_specifier",
    "struct_specifier",
    "struct_item",
    "interface_declaration",
    "impl_item",
    "trait_item",
}


def extract_symbols(path: str, text: str) -> list[dict]:
    """Return top-level symbols (with nested children) for a source file."""
    if not text:
        return []
    ext = _ext(path)
    if ext == ".py":
        try:
            return _python_symbols(text)
        except SyntaxError:
            return []
    return _treesitter_symbols(ext, text)


# ── Python (stdlib ast) ───────────────────────────────────────────────────────
def _python_symbols(text: str) -> list[dict]:
    lines = text.splitlines()
    tree = ast.parse(text)
    return [_py_node(n, lines) for n in tree.body if _is_py_def(n)]


def _is_py_def(node) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


def _py_node(node, lines: list[str]) -> dict:
    start = node.lineno
    end = getattr(node, "end_lineno", None) or start
    is_class = isinstance(node, ast.ClassDef)
    children = []
    if is_class:
        children = [_py_node(n, lines) for n in node.body if _is_py_def(n)]
    return {
        "name": node.name,
        "kind": "class" if is_class else "function",
        "start_line": start,
        "end_line": end,
        "signature": _line(lines, start),
        "docstring": _first_sentence(ast.get_docstring(node) or ""),
        "children": [_relabel_method(c) for c in children],
    }


def _relabel_method(sym: dict) -> dict:
    if sym["kind"] == "function":
        sym["kind"] = "method"
    return sym


# ── tree-sitter (optional) ────────────────────────────────────────────────────
def _treesitter_symbols(ext: str, text: str) -> list[dict]:
    lang = _TS_LANG_BY_EXT.get(ext)
    if not lang:
        return []
    try:
        from tree_sitter_languages import get_parser  # type: ignore
    except Exception:
        return []
    try:
        parser = get_parser(lang)
        source = text.encode("utf-8", errors="ignore")
        tree = parser.parse(source)
    except Exception:
        return []

    lines = text.splitlines()

    def node_text(node) -> str:
        return source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

    def find_name(node) -> str:
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "field_identifier", "name", "constant"):
                return node_text(child)
        named = getattr(node, "child_by_field_name", lambda _: None)("name")
        return node_text(named) if named else ""

    def walk(node, depth=0) -> list[dict]:
        results = []
        for child in node.children:
            if child.type in _TS_DEF_TYPES:
                start = child.start_point[0] + 1
                end = child.end_point[0] + 1
                is_class = child.type in _TS_CLASS_TYPES
                if is_class:
                    kind = "class"
                elif depth > 0:
                    kind = "method"
                else:
                    kind = "function"
                sym = {
                    "name": find_name(child) or child.type,
                    "kind": kind,
                    "start_line": start,
                    "end_line": end,
                    "signature": _line(lines, start),
                    "docstring": "",
                    "children": walk(child, depth + 1) if is_class else [],
                }
                results.append(sym)
            else:
                # Descend into wrapper nodes (e.g. export statements, blocks).
                results.extend(walk(child, depth))
        return results

    return walk(tree.root_node)


# ── helpers ───────────────────────────────────────────────────────────────────
def _ext(path: str) -> str:
    i = path.rfind(".")
    return path[i:].lower() if i != -1 else ""


def _line(lines: list[str], lineno: int) -> str:
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def _first_sentence(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    for sep in (". ", "\n"):
        idx = text.find(sep)
        if 0 < idx < limit:
            return text[: idx + 1].strip()
    return text[:limit].strip()
