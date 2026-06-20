import json
import PyPDF2

try:
    from .utils import get_number_of_pages, remove_fields, create_node_mapping
except ImportError:
    from utils import get_number_of_pages, remove_fields, create_node_mapping


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_pages(pages: str) -> list[int]:
    """Parse a pages string like '5-7', '3,8', or '12' into a sorted list of ints."""
    result = []
    for part in pages.split(','):
        part = part.strip()
        if '-' in part:
            start, end = int(part.split('-', 1)[0].strip()), int(part.split('-', 1)[1].strip())
            if start > end:
                raise ValueError(f"Invalid range '{part}': start must be <= end")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def _count_pages(doc_info: dict) -> int:
    """Return total page count for a PDF document."""
    if doc_info.get('page_count'):
        return doc_info['page_count']
    if doc_info.get('pages'):
        return len(doc_info['pages'])
    return get_number_of_pages(doc_info['path'])


def _get_pdf_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    """Extract text for specific PDF pages (1-indexed). Prefer cached pages, fallback to PDF."""
    cached_pages = doc_info.get('pages')
    if cached_pages:
        page_map = {p['page']: p['content'] for p in cached_pages}
        return [
            {'page': p, 'content': page_map[p]}
            for p in page_nums if p in page_map
        ]
    path = doc_info['path']
    with open(path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total = len(pdf_reader.pages)
        valid_pages = [p for p in page_nums if 1 <= p <= total]
        return [
            {'page': p, 'content': pdf_reader.pages[p - 1].extract_text() or ''}
            for p in valid_pages
        ]


def _parse_node_ids(node_ids) -> list[str]:
    """Normalize a node-id argument ('0001', '0001,0005', or a list) to padded ids."""
    if isinstance(node_ids, (list, tuple)):
        raw = list(node_ids)
    else:
        raw = str(node_ids).replace(",", " ").split()
    out = []
    for r in raw:
        r = str(r).strip()
        if not r:
            continue
        out.append(r.zfill(4) if r.isdigit() else r)
    return out


def _node_content(doc_info: dict, node: dict) -> str:
    """Resolve the retrievable text for a GitHub tree node."""
    # Issue/PR leaves carry their own text.
    if node.get("text"):
        return node["text"]
    # Code nodes address a slice of a per-file text store.
    path = node.get("path")
    if path is not None:
        text = (doc_info.get("files") or {}).get(path, "")
        start, end = node.get("start_line"), node.get("end_line")
        if text and start and end:
            return "\n".join(text.splitlines()[start - 1 : end])
        return text
    # Org repo leaves point at a separately-indexed repo document.
    if node.get("ref_doc_id") or node.get("repo"):
        return json.dumps(
            {
                "repo": node.get("repo", ""),
                "ref_doc_id": node.get("ref_doc_id"),
                "summary": node.get("summary", ""),
                "hint": "Use this repo's ref_doc_id with get_document_structure to drill in; "
                "if ref_doc_id is null, index the repo first.",
            },
            ensure_ascii=False,
        )
    return node.get("summary", "")


def get_node_content(documents: dict, doc_id: str, node_ids) -> str:
    """
    Retrieve content for specific tree nodes (GitHub repo/tracker/org documents).

    node_ids: a single id ('0007'), comma-separated ids ('0007,0012'), or a list.
    Returns a JSON list of {'node_id', 'title', 'path', 'content'}.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({"error": f"Document {doc_id} not found"})
    mapping = create_node_mapping(doc_info.get("structure", []))
    results = []
    for nid in _parse_node_ids(node_ids):
        node = mapping.get(nid)
        if not node:
            results.append({"node_id": nid, "error": "node not found"})
            continue
        results.append(
            {
                "node_id": nid,
                "title": node.get("title", ""),
                "path": node.get("path"),
                "content": _node_content(doc_info, node),
            }
        )
    return json.dumps(results, ensure_ascii=False)


def _get_md_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    """
    For Markdown documents, 'pages' are line numbers.
    Find nodes whose line_num falls within [min(page_nums), max(page_nums)] and return their text.
    """
    min_line, max_line = min(page_nums), max(page_nums)
    results = []
    seen = set()

    def _traverse(nodes):
        for node in nodes:
            ln = node.get('line_num')
            if ln and min_line <= ln <= max_line and ln not in seen:
                seen.add(ln)
                results.append({'page': ln, 'content': node.get('text', '')})
            if node.get('nodes'):
                _traverse(node['nodes'])

    _traverse(doc_info.get('structure', []))
    results.sort(key=lambda x: x['page'])
    return results


# ── Tool functions ────────────────────────────────────────────────────────────

def get_document(documents: dict, doc_id: str) -> str:
    """Return JSON with document metadata: doc_id, doc_name, doc_description, type, status, page_count (PDF) or line_count (Markdown)."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    result = {
        'doc_id': doc_id,
        'doc_name': doc_info.get('doc_name', ''),
        'doc_description': doc_info.get('doc_description', ''),
        'type': doc_info.get('type', ''),
        'status': 'completed',
    }
    t = doc_info.get('type')
    if t == 'pdf':
        result['page_count'] = _count_pages(doc_info)
    elif t == 'github_repo':
        result['repo'] = doc_info.get('repo', '')
        result['default_branch'] = doc_info.get('default_branch', '')
        result['file_count'] = doc_info.get('file_count', 0)
        result['primary_language'] = doc_info.get('primary_language', '')
    elif t == 'github_tracker':
        result['repo'] = doc_info.get('repo', '')
        result['issue_count'] = doc_info.get('issue_count', 0)
        result['pr_count'] = doc_info.get('pr_count', 0)
        result['item_count'] = doc_info.get('item_count', 0)
        result['last_synced'] = doc_info.get('last_synced', '')
    elif t == 'github_org':
        result['login'] = doc_info.get('login', '')
        result['repo_count'] = doc_info.get('repo_count', 0)
        result['last_synced'] = doc_info.get('last_synced', '')
    else:
        result['line_count'] = doc_info.get('line_count', 0)
    if doc_info.get('index_usage'):
        result['index_usage'] = doc_info.get('index_usage')
    return json.dumps(result)


_STRUCT_FIELDS = ("title", "node_id", "kind", "path", "symbol_kind", "start_line", "end_line")


def _find_node_by_id(nodes: list, node_id: str) -> dict | None:
    """Depth-first search for a node by its node_id."""
    for n in nodes:
        if str(n.get("node_id")) == node_id:
            return n
        found = _find_node_by_id(n.get("nodes") or [], node_id)
        if found:
            return found
    return None


def _prune_structure(nodes: list, depth: int, budget: dict, summary_chars: int) -> list:
    """Return a shallow copy of the tree limited to `depth` levels and a global node
    `budget`. Un-expanded nodes are marked `expandable` with a `child_count` so the
    caller can drill in via node_id. This keeps tool output bounded for huge repos.
    """
    out = []
    for node in nodes:
        if budget["n"] <= 0:
            break
        budget["n"] -= 1
        item = {k: node[k] for k in _STRUCT_FIELDS if node.get(k) is not None}
        summary = node.get("summary")
        if summary:
            item["summary"] = summary if len(summary) <= summary_chars else summary[:summary_chars].rstrip() + "…"
        children = node.get("nodes") or []
        if children:
            if depth > 1 and budget["n"] > 0:
                item["nodes"] = _prune_structure(children, depth - 1, budget, summary_chars)
                if len(item["nodes"]) < len(children):
                    item["child_count"] = len(children)
                    item["expandable"] = True
            else:
                item["child_count"] = len(children)
                item["expandable"] = True
        out.append(item)
    return out


def get_document_structure(
    documents: dict,
    doc_id: str,
    node_id: str = None,
    depth: int = None,
    max_nodes: int = 1200,
    summary_chars: int = 180,
) -> str:
    """Return tree structure JSON with text fields removed (saves tokens).

    For large trees (e.g. big repos), pass `depth` to get a shallow view and a
    `node_id` to expand a specific subtree — nodes that were not fully expanded are
    flagged `"expandable": true` with a `"child_count"`. With no `depth`/`node_id`
    the full tree is returned (backward compatible for PDFs/Markdown).
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    structure = doc_info.get('structure', [])

    if depth is None and node_id is None:
        return json.dumps(remove_fields(structure, fields=['text']), ensure_ascii=False)

    eff_depth = 2 if depth is None else max(1, int(depth))
    budget = {"n": max_nodes}

    if node_id:
        nid = node_id.zfill(4) if str(node_id).isdigit() else str(node_id)
        node = _find_node_by_id(structure, nid)
        if node is None:
            return json.dumps({'error': f'node_id {node_id!r} not found'})
        pruned = _prune_structure([node], eff_depth, budget, summary_chars)
        return json.dumps(pruned[0] if pruned else {}, ensure_ascii=False)

    pruned = _prune_structure(structure, eff_depth, budget, summary_chars)
    payload = {"nodes": pruned, "total_top_level": len(structure)}
    if budget["n"] <= 0:
        payload["note"] = "Output truncated to fit context. Drill into an 'expandable' node_id to see more."
    return json.dumps(payload, ensure_ascii=False)


def _collect_matches(nodes: list, terms: list, match_all: bool, limit: int, out: list) -> bool:
    """Walk the tree, appending nodes that match the terms. Returns True if `limit` hit."""
    for n in nodes:
        haystack = " ".join(
            str(n.get(k, "")) for k in ("title", "path", "summary", "symbol_kind", "kind")
        ).lower()
        hit = all(t in haystack for t in terms) if match_all else any(t in haystack for t in terms)
        if hit:
            item = {k: n[k] for k in _STRUCT_FIELDS if n.get(k) is not None}
            s = n.get("summary")
            if s:
                item["summary"] = s if len(s) <= 160 else s[:160].rstrip() + "…"
            out.append(item)
            if len(out) >= limit:
                return True
        if n.get("nodes") and _collect_matches(n["nodes"], terms, match_all, limit, out):
            return True
    return False


def search_nodes(documents: dict, doc_id: str, query: str, limit: int = 60) -> str:
    """Search the whole tree for nodes matching `query`.

    Tries an exact all-words match first; if that finds nothing (and the query has
    multiple words) it broadens to an any-word match. Matches against each node's
    title, path, summary and symbol kind. Ideal for "find/how many X" questions over
    large repos without drilling the tree by hand.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    terms = (query or "").strip().lower().split()
    if not terms:
        return json.dumps({'error': 'empty query'})

    structure = doc_info.get("structure", [])
    matches: list[dict] = []
    truncated = _collect_matches(structure, terms, True, limit, matches)
    mode = "all"
    if not matches and len(terms) > 1:
        truncated = _collect_matches(structure, terms, False, limit, matches)
        mode = "any"
    return json.dumps(
        {"query": query, "match_mode": mode, "match_count": len(matches),
         "truncated": bool(truncated), "matches": matches},
        ensure_ascii=False,
    )


def get_page_content(documents: dict, doc_id: str, pages: str) -> str:
    """
    Retrieve page content for a document.

    pages format: '5-7', '3,8', or '12'
    For PDF: pages are physical page numbers (1-indexed).
    For Markdown: pages are line numbers corresponding to node headers.

    Returns JSON list of {'page': int, 'content': str}.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})

    # GitHub documents are addressed by node id, not page/line number.
    if str(doc_info.get('type', '')).startswith('github'):
        return get_node_content(documents, doc_id, pages)

    try:
        page_nums = _parse_pages(pages)
    except (ValueError, AttributeError) as e:
        return json.dumps({'error': f'Invalid pages format: {pages!r}. Use "5-7", "3,8", or "12". Error: {e}'})

    try:
        if doc_info.get('type') == 'pdf':
            content = _get_pdf_page_content(doc_info, page_nums)
        else:
            content = _get_md_page_content(doc_info, page_nums)
    except Exception as e:
        return json.dumps({'error': f'Failed to read page content: {e}'})

    return json.dumps(content, ensure_ascii=False)
