"""
GitHub issues + pull requests -> PageIndex tree.

The tracker is modeled as a small two-level tree:

    Issues
      ├─ Open
      │    ├─ #123 Title ...        (leaf: body + comments as content)
      │    └─ ...
      └─ Closed
    Pull Requests
      ├─ Open / Closed / Merged
      └─ ...

Leaves carry their full text (title + body + comments) so the retrieval agent can
read them via get_node_content. Summaries are extractive (title + first body line)
so indexing hundreds of items stays cheap.
"""

try:
    from ..utils import write_node_id, format_structure, generate_doc_description, create_clean_structure_for_description
    from ..summarize import summarize_structure, SummaryCache
    from .github_graphql import GitHubGraphQL, parse_repo_url
except ImportError:  # pragma: no cover
    from utils import write_node_id, format_structure, generate_doc_description, create_clean_structure_for_description
    from summarize import summarize_structure, SummaryCache
    from sources.github_graphql import GitHubGraphQL, parse_repo_url


_OUTPUT_ORDER = [
    "title", "node_id", "kind", "number", "state", "author",
    "labels", "created_at", "updated_at", "summary", "text", "nodes",
]


async def build_tracker_doc(
    repo_url: str,
    opt,
    cache: SummaryCache = None,
    include: str = "both",
    gh: GitHubGraphQL = None,
) -> dict:
    owner, name = parse_repo_url(repo_url)
    gh = gh or GitHubGraphQL()
    max_items = getattr(opt, "github_max_items", 300)

    roots = []
    issue_count = pr_count = 0
    if include in ("both", "issues"):
        issues = gh.fetch_issues(owner, name, max_items=max_items)
        issue_count = len(issues)
        if issues:
            roots.append(_group_records("Issues", issues, ["OPEN", "CLOSED"]))
    if include in ("both", "prs"):
        prs = gh.fetch_prs(owner, name, max_items=max_items)
        pr_count = len(prs)
        if prs:
            roots.append(_group_records("Pull Requests", prs, ["OPEN", "MERGED", "CLOSED"]))

    write_node_id(roots)

    await summarize_structure(
        roots,
        doc_kind="github_tracker",
        cache=cache,
        model=opt.model,
        summary_model=(getattr(opt, "summary_model", "") or opt.model),
    )

    last_synced = _max_updated(roots)
    structure = format_structure(roots, order=_OUTPUT_ORDER)
    doc_description = generate_doc_description(
        create_clean_structure_for_description(structure), model=opt.model
    )

    return {
        "type": "github_tracker",
        "repo": f"{owner}/{name}",
        "doc_name": f"{owner}/{name} (issues & PRs)",
        "doc_description": doc_description,
        "include": include,
        "issue_count": issue_count,
        "pr_count": pr_count,
        "item_count": issue_count + pr_count,
        "last_synced": last_synced,
        "structure": structure,
    }


def _group_records(group_title: str, records: list[dict], states: list[str]) -> dict:
    buckets: dict[str, list] = {s: [] for s in states}
    other: list = []
    for r in records:
        st = (r.get("state") or "").upper()
        (buckets.get(st, other)).append(_record_leaf(r))

    children = []
    for st in states:
        if buckets[st]:
            children.append({"title": st.capitalize(), "kind": "group", "nodes": buckets[st]})
    if other:
        children.append({"title": "Other", "kind": "group", "nodes": other})

    return {"title": group_title, "kind": "group", "nodes": children}


def _record_leaf(r: dict) -> dict:
    return {
        "title": f"#{r.get('number')} {r.get('title', '')}".strip(),
        "kind": r.get("kind", "issue"),
        "number": r.get("number"),
        "state": r.get("state", ""),
        "author": r.get("author", ""),
        "labels": r.get("labels", []),
        "created_at": r.get("created_at", ""),
        "updated_at": r.get("updated_at", ""),
        "text": _record_text(r),
    }


def _record_text(r: dict) -> str:
    lines = [
        f"Title: {r.get('title', '')}",
        f"Type: {r.get('kind', '')}   State: {r.get('state', '')}   Author: {r.get('author', '')}",
    ]
    if r.get("labels"):
        lines.append(f"Labels: {', '.join(r['labels'])}")
    lines.append("")
    lines.append(r.get("body", "") or "(no description)")
    comments = r.get("comments") or []
    if comments:
        lines.append("")
        lines.append("--- Comments ---")
        for c in comments:
            lines.append(f"@{c.get('author', '')}: {c.get('body', '')}")
    return "\n".join(lines)


def _max_updated(roots: list) -> str:
    latest = ""
    def walk(nodes):
        nonlocal latest
        for n in nodes:
            u = n.get("updated_at") or ""
            if u > latest:
                latest = u
            if n.get("nodes"):
                walk(n["nodes"])
    walk(roots)
    return latest
