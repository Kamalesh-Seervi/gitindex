"""
GitHub org / user -> corpus-level PageIndex tree ("PageIndex File System").

Builds a top-level tree of all repositories owned by an org or user, grouped by
primary language, with each repo as a leaf. This lets the retrieval agent reason
over an entire corpus (100+ repos) and pick which repo to drill into, instead of
loading every repo at once.

Each repo leaf can optionally be linked (ref_doc_id) to a separately-indexed repo
document, enabling two-phase navigation: org tree -> repo tree -> file/symbol.
Linking is lazy by default so indexing the org itself stays cheap.
"""

try:
    from ..utils import write_node_id, format_structure, generate_doc_description, create_clean_structure_for_description
    from ..summarize import summarize_structure, SummaryCache
    from .github_graphql import GitHubGraphQL
except ImportError:  # pragma: no cover
    from utils import write_node_id, format_structure, generate_doc_description, create_clean_structure_for_description
    from summarize import summarize_structure, SummaryCache
    from sources.github_graphql import GitHubGraphQL


_OUTPUT_ORDER = [
    "title", "node_id", "kind", "repo", "primary_language",
    "pushed_at", "ref_doc_id", "summary", "nodes",
]


async def build_org_doc(
    login: str,
    opt,
    cache: SummaryCache = None,
    gh: GitHubGraphQL = None,
    include_forks: bool = False,
    index_repo_fn=None,
) -> dict:
    gh = gh or GitHubGraphQL()
    repos = gh.list_owner_repos(
        login, max_items=getattr(opt, "github_max_items", 300), include_forks=include_forks
    )

    groups: dict[str, list] = {}
    for r in repos:
        lang = r.get("primary_language") or "Other"
        leaf = {
            "title": r["name"],
            "kind": "repo",
            "repo": f"{login}/{r['name']}",
            "description": r.get("description", ""),
            "primary_language": lang,
            "pushed_at": r.get("pushed_at", ""),
            "topics": r.get("topics", []),
        }
        # Optional eager linking to a per-repo indexed document.
        if index_repo_fn is not None:
            try:
                leaf["ref_doc_id"] = index_repo_fn(leaf["repo"])
            except Exception as e:  # keep org indexing resilient
                print(f"Warning: could not index {leaf['repo']}: {e}")
        groups.setdefault(lang, []).append(leaf)

    structure = [
        {"title": lang, "kind": "language", "nodes": sorted(items, key=lambda x: x["title"].lower())}
        for lang, items in sorted(groups.items())
    ]
    write_node_id(structure)

    await summarize_structure(
        structure,
        doc_kind="github_org",
        cache=cache,
        model=opt.model,
        summary_model=(getattr(opt, "summary_model", "") or opt.model),
    )

    structure = format_structure(structure, order=_OUTPUT_ORDER)
    doc_description = generate_doc_description(
        create_clean_structure_for_description(structure), model=opt.model
    )

    return {
        "type": "github_org",
        "login": login,
        "doc_name": f"{login} (GitHub org/user)",
        "doc_description": doc_description,
        "repo_count": len(repos),
        "last_synced": max((r.get("pushed_at", "") for r in repos), default=""),
        "structure": structure,
    }
