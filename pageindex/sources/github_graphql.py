"""
Minimal GitHub GraphQL (v4) client used by the PageIndex GitHub source adapters.

Auth: reads a token from the GITHUB_TOKEN (or GH_TOKEN) environment variable.
Only read queries are issued. Batched, aliased queries are used to walk a repo
tree and fetch blob contents in as few round-trips as possible.
"""

import os
import re
import time

import requests

GITHUB_GRAPHQL_ENDPOINT = "https://api.github.com/graphql"


class GitHubAuthError(RuntimeError):
    pass


class GitHubGraphQLError(RuntimeError):
    pass


def parse_repo_url(repo_url: str) -> tuple[str, str]:
    """Parse 'owner/name', a full GitHub URL, or 'git@github.com:owner/name.git'."""
    s = repo_url.strip()
    s = re.sub(r"^git@github\.com:", "", s)
    s = re.sub(r"^https?://github\.com/", "", s)
    s = re.sub(r"\.git$", "", s)
    s = s.strip("/")
    parts = s.split("/")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Could not parse owner/name from: {repo_url!r}")
    return parts[0], parts[1]


class GitHubGraphQL:
    def __init__(self, token: str = None, tree_batch: int = 40, blob_batch: int = 25):
        self.token = token or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if not self.token:
            raise GitHubAuthError(
                "No GitHub token found. Set GITHUB_TOKEN (or GH_TOKEN) in your environment."
            )
        self.tree_batch = tree_batch
        self.blob_batch = blob_batch
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "PageIndex-GitHub-Source",
            }
        )

    # ── core ──────────────────────────────────────────────────────────────────
    def query(self, query_str: str, variables: dict = None, max_retries: int = 6) -> dict:
        payload = {"query": query_str}
        if variables:
            payload["variables"] = variables

        last_err = None
        for attempt in range(max_retries):
            try:
                resp = self._session.post(GITHUB_GRAPHQL_ENDPOINT, json=payload, timeout=60)
            except requests.RequestException as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
                continue

            # Primary/secondary rate limits.
            if resp.status_code in (403, 429):
                retry_after = resp.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 60)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                time.sleep(min(2 ** attempt, 30))
                continue

            try:
                body = resp.json()
            except ValueError as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
                continue

            errors = body.get("errors")
            data = body.get("data")
            if errors:
                # If it's a rate-limit error, back off and retry; otherwise fail fast.
                if any(e.get("type") == "RATE_LIMITED" for e in errors):
                    time.sleep(min(2 ** attempt, 60))
                    continue
                if data is None:
                    raise GitHubGraphQLError(_format_errors(errors))
            if data is not None:
                return data
            last_err = GitHubGraphQLError(_format_errors(errors or [{"message": "empty response"}]))
            time.sleep(min(2 ** attempt, 30))

        raise GitHubGraphQLError(f"GraphQL request failed after {max_retries} retries: {last_err}")

    # ── repository metadata ───────────────────────────────────────────────────
    def repo_meta(self, owner: str, name: str) -> dict:
        q = """
        query($owner:String!, $name:String!) {
          repository(owner:$owner, name:$name) {
            description
            pushedAt
            primaryLanguage { name }
            defaultBranchRef { name target { oid } }
          }
        }
        """
        data = self.query(q, {"owner": owner, "name": name})
        repo = data.get("repository")
        if not repo:
            raise GitHubGraphQLError(f"Repository not found or inaccessible: {owner}/{name}")
        branch = repo.get("defaultBranchRef") or {}
        target = (branch.get("target") or {}) if branch else {}
        return {
            "description": repo.get("description") or "",
            "pushed_at": repo.get("pushedAt") or "",
            "primary_language": ((repo.get("primaryLanguage") or {}).get("name") or ""),
            "default_branch": branch.get("name") or "HEAD",
            "head_oid": target.get("oid") or "",
        }

    def _resolve_root_tree_oid(self, owner: str, name: str, oid: str) -> str:
        """Resolve a commit (or tree) OID to its root Tree OID.

        ``defaultBranchRef.target.oid`` is a *commit* OID; the tree walk needs the
        commit's root Tree. A tree OID resolves to itself.
        """
        q = (
            f'query {{ repository(owner:"{owner}", name:"{name}") {{ '
            f'object(oid:"{oid}") {{ __typename '
            f"... on Commit {{ tree {{ oid }} }} "
            f"... on Tree {{ oid }} }} }} }}"
        )
        data = self.query(q)
        obj = (data.get("repository") or {}).get("object") or {}
        if obj.get("__typename") == "Commit":
            return ((obj.get("tree") or {}).get("oid")) or ""
        return obj.get("oid") or oid

    # ── repo tree walk (batched BFS over tree objects) ────────────────────────
    def walk_tree(self, owner: str, name: str, head_oid: str, max_blobs: int = 5000) -> list[dict]:
        if not head_oid:
            return []
        root_tree_oid = self._resolve_root_tree_oid(owner, name, head_oid)
        if not root_tree_oid:
            return []
        blobs: list[dict] = []
        # BFS over tree OIDs, carrying each subtree's path prefix. TreeEntry.path is
        # null when a tree is fetched by raw OID, so full paths are rebuilt from the
        # parent prefix + entry name. No OID de-dup: git trees are acyclic (so BFS
        # terminates), and identical subtrees under different paths must each expand.
        queue: list[tuple[str, str]] = [(root_tree_oid, "")]

        while queue:
            batch = queue[: self.tree_batch]
            queue = queue[self.tree_batch :]
            aliases = [
                f't{i}: object(oid: "{oid}") {{ ... on Tree {{ entries {{ '
                f"name type oid object {{ __typename ... on Blob {{ byteSize isBinary }} }} }} }} }}"
                for i, (oid, _prefix) in enumerate(batch)
            ]
            q = f'query {{ repository(owner:"{owner}", name:"{name}") {{ {" ".join(aliases)} }} }}'
            data = self.query(q)
            repo = data.get("repository") or {}
            for i, (_oid, prefix) in enumerate(batch):
                obj = repo.get(f"t{i}")
                if not obj:
                    continue
                for e in obj.get("entries", []) or []:
                    ename = e.get("name") or ""
                    full_path = f"{prefix}/{ename}" if prefix else ename
                    etype = e.get("type")
                    if etype == "tree":
                        child_oid = e.get("oid")
                        if child_oid:
                            queue.append((child_oid, full_path))
                    elif etype == "blob":
                        o = e.get("object") or {}
                        blobs.append(
                            {
                                "path": full_path,
                                "oid": e.get("oid"),
                                "byteSize": o.get("byteSize", 0) or 0,
                                "isBinary": bool(o.get("isBinary")),
                            }
                        )
                        if len(blobs) >= max_blobs:
                            return blobs
        return blobs

    def fetch_blob_texts(self, owner: str, name: str, oids: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        uniq = list(dict.fromkeys(o for o in oids if o))
        for i in range(0, len(uniq), self.blob_batch):
            batch = uniq[i : i + self.blob_batch]
            aliases = [f'b{j}: object(oid:"{o}") {{ ... on Blob {{ text }} }}' for j, o in enumerate(batch)]
            q = f'query {{ repository(owner:"{owner}", name:"{name}") {{ {" ".join(aliases)} }} }}'
            data = self.query(q)
            repo = data.get("repository") or {}
            for j, o in enumerate(batch):
                obj = repo.get(f"b{j}") or {}
                out[o] = obj.get("text") or ""
        return out

    # ── issues / pull requests ────────────────────────────────────────────────
    def fetch_issues(self, owner: str, name: str, max_items: int = 300, comments_per_item: int = 20) -> list[dict]:
        return self._fetch_tracker(owner, name, "issues", max_items, comments_per_item)

    def fetch_prs(self, owner: str, name: str, max_items: int = 300, comments_per_item: int = 20) -> list[dict]:
        return self._fetch_tracker(owner, name, "pullRequests", max_items, comments_per_item)

    def _fetch_tracker(self, owner, name, field, max_items, comments_per_item) -> list[dict]:
        items: list[dict] = []
        cursor = None
        extra = "merged" if field == "pullRequests" else ""
        page = min(50, max_items)
        while len(items) < max_items:
            q = f"""
            query($owner:String!, $name:String!, $cursor:String, $page:Int!, $c:Int!) {{
              repository(owner:$owner, name:$name) {{
                {field}(first:$page, after:$cursor, orderBy:{{field:UPDATED_AT, direction:DESC}}) {{
                  pageInfo {{ hasNextPage endCursor }}
                  nodes {{
                    number title body state createdAt updatedAt {extra}
                    author {{ login }}
                    labels(first:10) {{ nodes {{ name }} }}
                    comments(first:$c) {{ nodes {{ author {{ login }} body }} }}
                  }}
                }}
              }}
            }}
            """
            data = self.query(q, {"owner": owner, "name": name, "cursor": cursor, "page": page, "c": comments_per_item})
            conn = ((data.get("repository") or {}).get(field)) or {}
            for n in conn.get("nodes", []) or []:
                items.append(self._normalize_tracker_node(n, field))
                if len(items) >= max_items:
                    break
            info = conn.get("pageInfo") or {}
            if not info.get("hasNextPage") or not conn.get("nodes"):
                break
            cursor = info.get("endCursor")
        return items

    @staticmethod
    def _normalize_tracker_node(n: dict, field: str) -> dict:
        labels = [l.get("name") for l in ((n.get("labels") or {}).get("nodes") or []) if l.get("name")]
        comments = [
            {"author": ((c.get("author") or {}).get("login") or ""), "body": c.get("body") or ""}
            for c in ((n.get("comments") or {}).get("nodes") or [])
        ]
        state = n.get("state") or ""
        if field == "pullRequests" and n.get("merged"):
            state = "MERGED"
        return {
            "kind": "pr" if field == "pullRequests" else "issue",
            "number": n.get("number"),
            "title": n.get("title") or "",
            "body": n.get("body") or "",
            "state": state,
            "author": ((n.get("author") or {}).get("login") or ""),
            "labels": labels,
            "created_at": n.get("createdAt") or "",
            "updated_at": n.get("updatedAt") or "",
            "comments": comments,
        }

    # ── org / user repositories ───────────────────────────────────────────────
    def list_owner_repos(self, login: str, max_items: int = 300, include_forks: bool = False) -> list[dict]:
        repos: list[dict] = []
        cursor = None
        page = min(50, max_items)
        while len(repos) < max_items:
            q = """
            query($login:String!, $cursor:String, $page:Int!) {
              repositoryOwner(login:$login) {
                repositories(first:$page, after:$cursor, orderBy:{field:PUSHED_AT, direction:DESC}) {
                  pageInfo { hasNextPage endCursor }
                  nodes {
                    name description pushedAt isArchived isFork
                    primaryLanguage { name }
                    repositoryTopics(first:10) { nodes { topic { name } } }
                  }
                }
              }
            }
            """
            data = self.query(q, {"login": login, "cursor": cursor, "page": page})
            owner = data.get("repositoryOwner")
            if not owner:
                raise GitHubGraphQLError(f"Owner not found or inaccessible: {login}")
            conn = owner.get("repositories") or {}
            for n in conn.get("nodes", []) or []:
                if n.get("isFork") and not include_forks:
                    continue
                topics = [
                    t.get("topic", {}).get("name")
                    for t in ((n.get("repositoryTopics") or {}).get("nodes") or [])
                    if t.get("topic")
                ]
                repos.append(
                    {
                        "name": n.get("name"),
                        "description": n.get("description") or "",
                        "pushed_at": n.get("pushedAt") or "",
                        "is_archived": bool(n.get("isArchived")),
                        "primary_language": ((n.get("primaryLanguage") or {}).get("name") or ""),
                        "topics": [t for t in topics if t],
                    }
                )
                if len(repos) >= max_items:
                    break
            info = conn.get("pageInfo") or {}
            if not info.get("hasNextPage") or not conn.get("nodes"):
                break
            cursor = info.get("endCursor")
        return repos


def _format_errors(errors: list) -> str:
    return "; ".join(e.get("message", str(e)) for e in errors)
