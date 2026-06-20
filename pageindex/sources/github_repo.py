"""
GitHub repository -> PageIndex tree (dir -> file -> symbol).

Reuses the vectorless PageIndex idea: build a hierarchical tree of the repo, give
every node a cheap summary, and let the retrieval agent reason over the tree.
Content is addressed by node line ranges (resolved against a per-file text store),
mirroring how the PDF/Markdown adapters address pages/lines.

Incremental refresh leverages git's content addressing: a file whose blob OID is
unchanged is not re-fetched, re-parsed, or re-summarized (Merkle diff).
"""

import os

try:
    from ..utils import write_node_id, format_structure, generate_doc_description, create_clean_structure_for_description
    from ..summarize import summarize_structure, SummaryCache
    from .github_graphql import GitHubGraphQL, parse_repo_url
    from .code_parser import extract_symbols
except ImportError:  # pragma: no cover
    from utils import write_node_id, format_structure, generate_doc_description, create_clean_structure_for_description
    from summarize import summarize_structure, SummaryCache
    from sources.github_graphql import GitHubGraphQL, parse_repo_url
    from sources.code_parser import extract_symbols


# Extensions that are worth parsing/retrieving as text.
_TEXT_EXTS = {
    ".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".java", ".rb",
    ".rs", ".c", ".h", ".cpp", ".cc", ".hpp", ".cs", ".php", ".scala", ".kt",
    ".swift", ".m", ".mm", ".sh", ".bash", ".sql", ".r", ".lua", ".pl",
    ".md", ".markdown", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".xml", ".html", ".css", ".scss", ".vue", ".svelte",
    ".gradle", ".dockerfile", ".tf", ".proto", ".graphql",
}
_NAME_WHITELIST = {"Dockerfile", "Makefile", "README", "LICENSE", ".gitignore"}

_OUTPUT_ORDER = [
    "title", "node_id", "kind", "path", "symbol_kind",
    "start_line", "end_line", "summary", "nodes",
]


def _is_text_path(path: str) -> bool:
    base = os.path.basename(path)
    if base in _NAME_WHITELIST:
        return True
    ext = os.path.splitext(path)[1].lower()
    return ext in _TEXT_EXTS


async def build_repo_doc(repo_url: str, opt, cache: SummaryCache = None, prev_doc: dict = None, gh: GitHubGraphQL = None) -> dict:
    owner, name = parse_repo_url(repo_url)
    gh = gh or GitHubGraphQL()

    meta = gh.repo_meta(owner, name)
    head_oid = meta["head_oid"]

    blobs = gh.walk_tree(owner, name, head_oid)
    candidates = [
        b for b in blobs
        if not b["isBinary"] and b["byteSize"] <= opt.github_max_file_bytes and _is_text_path(b["path"])
    ]

    prev_files = (prev_doc or {}).get("files", {})
    prev_oids = (prev_doc or {}).get("file_oids", {})
    prev_file_nodes = _index_file_nodes((prev_doc or {}).get("structure", []))

    # Merkle diff: only fetch blob text for files whose OID changed.
    to_fetch = [b["oid"] for b in candidates if prev_oids.get(b["path"]) != b["oid"]]
    fetched = gh.fetch_blob_texts(owner, name, to_fetch) if to_fetch else {}

    files: dict[str, str] = {}
    file_oids: dict[str, str] = {}
    reused_paths: set[str] = set()
    for b in candidates:
        path, oid = b["path"], b["oid"]
        if prev_oids.get(path) == oid and path in prev_files:
            files[path] = prev_files[path]
            reused_paths.add(path)
        else:
            files[path] = fetched.get(oid, "")
        file_oids[path] = oid

    structure = _build_path_tree(candidates)
    _attach_symbols(structure, files, reused_paths, prev_file_nodes)
    write_node_id(structure)

    await summarize_structure(
        structure,
        doc_kind="github_repo",
        files=files,
        cache=cache,
        model=opt.model,
        summary_model=(getattr(opt, "summary_model", "") or opt.model),
        leaf_mode=getattr(opt, "github_summarize_leaves", "extractive"),
        batch_size=getattr(opt, "github_summary_batch_size", 20),
    )

    structure = format_structure(structure, order=_OUTPUT_ORDER)
    doc_description = generate_doc_description(
        create_clean_structure_for_description(structure), model=opt.model
    )

    return {
        "type": "github_repo",
        "repo": f"{owner}/{name}",
        "doc_name": f"{owner}/{name}",
        "doc_description": doc_description,
        "default_branch": meta["default_branch"],
        "head_oid": head_oid,
        "pushed_at": meta["pushed_at"],
        "primary_language": meta["primary_language"],
        "file_count": len(files),
        "structure": structure,
        "files": files,
        "file_oids": file_oids,
    }


# ── tree building ─────────────────────────────────────────────────────────────
def _build_path_tree(candidates: list[dict]) -> list:
    """Turn a flat blob list into a nested dir/file node tree."""
    root: dict = {}
    for b in candidates:
        parts = b["path"].split("/")
        cur = root
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur.setdefault("__files__", []).append((parts[-1], b["path"], b["oid"]))
    return _dict_to_nodes(root)


def _dict_to_nodes(d: dict, prefix: str = "") -> list:
    nodes = []
    # Directories first (sorted), then files.
    for name in sorted(k for k in d.keys() if k != "__files__"):
        child_prefix = f"{prefix}{name}/"
        nodes.append(
            {
                "title": name,
                "kind": "dir",
                "path": child_prefix.rstrip("/"),
                "nodes": _dict_to_nodes(d[name], child_prefix),
            }
        )
    for fname, fpath, _oid in sorted(d.get("__files__", []), key=lambda x: x[0]):
        nodes.append({"title": fname, "kind": "file", "path": fpath, "start_line": 1, "end_line": 0})
    return nodes


def _attach_symbols(nodes: list, files: dict, reused_paths: set, prev_file_nodes: dict):
    for node in nodes:
        if node.get("kind") == "dir":
            _attach_symbols(node.get("nodes", []), files, reused_paths, prev_file_nodes)
            continue
        if node.get("kind") != "file":
            continue
        path = node["path"]
        text = files.get(path, "")
        node["end_line"] = max(1, len(text.splitlines()))
        # Reuse previously parsed symbol nodes when the file is unchanged.
        if path in reused_paths and path in prev_file_nodes:
            prev = prev_file_nodes[path]
            if prev.get("nodes"):
                node["nodes"] = _strip_ids(prev["nodes"])
            continue
        symbols = extract_symbols(path, text)
        if symbols:
            node["nodes"] = [_symbol_to_node(s, path) for s in symbols]


def _symbol_to_node(sym: dict, path: str) -> dict:
    node = {
        "title": sym["name"],
        "kind": "symbol",
        "symbol_kind": sym["kind"],
        "path": path,
        "start_line": sym["start_line"],
        "end_line": sym["end_line"],
        "signature": sym.get("signature", ""),
        "docstring": sym.get("docstring", ""),
    }
    children = sym.get("children") or []
    if children:
        node["nodes"] = [_symbol_to_node(c, path) for c in children]
    return node


def _index_file_nodes(structure: list) -> dict:
    """Map path -> file node, for incremental reuse."""
    out: dict[str, dict] = {}

    def walk(nodes):
        for n in nodes:
            if n.get("kind") == "file" and n.get("path"):
                out[n["path"]] = n
            if n.get("nodes"):
                walk(n["nodes"])

    walk(structure or [])
    return out


def _strip_ids(nodes: list) -> list:
    out = []
    for n in nodes:
        c = {k: v for k, v in n.items() if k != "node_id"}
        if c.get("nodes"):
            c["nodes"] = _strip_ids(c["nodes"])
        out.append(c)
    return out
