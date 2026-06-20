import os
import uuid
import json
import asyncio
import concurrent.futures
from pathlib import Path

import PyPDF2

from .page_index import page_index
from .page_index_md import md_to_tree
from .retrieve import get_document, get_document_structure, get_page_content, get_node_content, search_nodes
from .summarize import SummaryCache
from .utils import ConfigLoader, remove_fields, track_usage

META_INDEX = "_meta.json"


def _normalize_retrieve_model(model: str) -> str:
    """Preserve supported Agents SDK prefixes and route other provider paths via LiteLLM."""
    passthrough_prefixes = ("litellm/", "openai/")
    if not model or "/" not in model:
        return model
    if model.startswith(passthrough_prefixes):
        return model
    return f"litellm/{model}"


class PageIndexClient:
    """
    A client for indexing and retrieving document content.
    Flow: index() -> get_document() / get_document_structure() / get_page_content()

    For agent-based QA, see examples/agentic_vectorless_rag_demo.py.
    """
    def __init__(self, api_key: str = None, model: str = None, retrieve_model: str = None, workspace: str = None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")
        self.workspace = Path(workspace).expanduser() if workspace else None
        overrides = {}
        if model:
            overrides["model"] = model
        if retrieve_model:
            overrides["retrieve_model"] = retrieve_model
        opt = ConfigLoader().load(overrides or None)
        self.opt = opt
        self.model = opt.model
        self.retrieve_model = _normalize_retrieve_model(opt.retrieve_model or self.model)
        if self.workspace:
            self.workspace.mkdir(parents=True, exist_ok=True)
        self._cache = SummaryCache(str(self.workspace / "_summary_cache.json") if self.workspace else None)
        self.documents = {}
        if self.workspace:
            self._load_workspace()

    def index(self, file_path: str, mode: str = "auto") -> str:
        """Index a document. Returns a document_id."""
        # Persist a canonical absolute path so workspace reloads do not
        # reinterpret caller-relative paths against the workspace directory.
        file_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = str(uuid.uuid4())
        ext = os.path.splitext(file_path)[1].lower()

        is_pdf = ext == '.pdf'
        is_md = ext in ['.md', '.markdown']

        if mode == "pdf" or (mode == "auto" and is_pdf):
            print(f"Indexing PDF: {file_path}")
            result = page_index(
                doc=file_path,
                model=self.model,
                if_add_node_summary='yes',
                if_add_node_text='yes',
                if_add_node_id='yes',
                if_add_doc_description='yes'
            )
            # Extract per-page text so queries don't need the original PDF
            pages = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(pdf_reader.pages, 1):
                    pages.append({'page': i, 'content': page.extract_text() or ''})

            self.documents[doc_id] = {
                'id': doc_id,
                'type': 'pdf',
                'path': file_path,
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', ''),
                'page_count': len(pages),
                'structure': result['structure'],
                'pages': pages,
            }

        elif mode == "md" or (mode == "auto" and is_md):
            print(f"Indexing Markdown: {file_path}")
            coro = md_to_tree(
                md_path=file_path,
                if_thinning=False,
                if_add_node_summary='yes',
                summary_token_threshold=200,
                model=self.model,
                if_add_doc_description='yes',
                if_add_node_text='yes',
                if_add_node_id='yes'
            )
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, coro).result()
            except RuntimeError:
                result = asyncio.run(coro)
            self.documents[doc_id] = {
                'id': doc_id,
                'type': 'md',
                'path': file_path,
                'doc_name': result.get('doc_name', ''),
                'doc_description': result.get('doc_description', ''),
                'line_count': result.get('line_count', 0),
                'structure': result['structure'],
            }
        else:
            raise ValueError(f"Unsupported file format for: {file_path}")

        print(f"Indexing complete. Document ID: {doc_id}")
        if self.workspace:
            self._save_doc(doc_id)
        return doc_id

    @staticmethod
    def _make_meta_entry(doc: dict) -> dict:
        """Build a lightweight meta entry from a document dict."""
        entry = {
            'type': doc.get('type', ''),
            'doc_name': doc.get('doc_name', ''),
            'doc_description': doc.get('doc_description', ''),
            'path': doc.get('path', ''),
        }
        t = doc.get('type')
        if t == 'pdf':
            entry['page_count'] = doc.get('page_count')
        elif t == 'md':
            entry['line_count'] = doc.get('line_count')
        elif t == 'github_repo':
            for k in ('repo', 'default_branch', 'head_oid', 'primary_language', 'file_count', 'pushed_at'):
                entry[k] = doc.get(k)
        elif t == 'github_tracker':
            for k in ('repo', 'include', 'issue_count', 'pr_count', 'item_count', 'last_synced'):
                entry[k] = doc.get(k)
        elif t == 'github_org':
            for k in ('login', 'repo_count', 'last_synced'):
                entry[k] = doc.get(k)
        if doc.get('index_usage'):
            entry['index_usage'] = doc.get('index_usage')
        return entry

    @staticmethod
    def _read_json(path) -> dict | None:
        """Read a JSON file, returning None on any error."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: corrupt {Path(path).name}: {e}")
            return None

    def _save_doc(self, doc_id: str):
        doc = self.documents[doc_id].copy()
        # Strip text from structure nodes — redundant with pages (PDF only)
        if doc.get('structure') and doc.get('type') == 'pdf':
            doc['structure'] = remove_fields(doc['structure'], fields=['text'])
        path = self.workspace / f"{doc_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        self._save_meta(doc_id, self._make_meta_entry(doc))
        # Drop heavy fields; will lazy-load on demand
        for field in ('structure', 'pages', 'files', 'file_oids'):
            self.documents[doc_id].pop(field, None)

    def _rebuild_meta(self) -> dict:
        """Scan individual doc JSON files and return a meta dict."""
        meta = {}
        for path in self.workspace.glob("*.json"):
            if path.name == META_INDEX:
                continue
            doc = self._read_json(path)
            if doc and isinstance(doc, dict):
                meta[path.stem] = self._make_meta_entry(doc)
        return meta

    def _read_meta(self) -> dict | None:
        """Read and validate _meta.json, returning None on any corruption."""
        meta = self._read_json(self.workspace / META_INDEX)
        if meta is not None and not isinstance(meta, dict):
            print(f"Warning: {META_INDEX} is not a JSON object, ignoring")
            return None
        return meta

    def _save_meta(self, doc_id: str, entry: dict):
        meta = self._read_meta() or self._rebuild_meta()
        meta[doc_id] = entry
        meta_path = self.workspace / META_INDEX
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _load_workspace(self):
        meta = self._read_meta()
        if meta is None:
            meta = self._rebuild_meta()
            if meta:
                print(f"Loaded {len(meta)} document(s) from workspace (legacy mode).")
        for doc_id, entry in meta.items():
            doc = dict(entry, id=doc_id)
            if doc.get('path') and not os.path.isabs(doc['path']):
                doc['path'] = str((self.workspace / doc['path']).resolve())
            self.documents[doc_id] = doc

    def _ensure_doc_loaded(self, doc_id: str):
        """Load full document JSON on demand (structure, pages, etc.)."""
        doc = self.documents.get(doc_id)
        if not doc or doc.get('structure') is not None:
            return
        full = self._read_json(self.workspace / f"{doc_id}.json")
        if not full:
            return
        doc['structure'] = full.get('structure', [])
        for field in ('pages', 'files', 'file_oids'):
            if full.get(field) is not None:
                doc[field] = full[field]

    def get_document(self, doc_id: str) -> str:
        """Return document metadata JSON."""
        return get_document(self.documents, doc_id)

    def get_document_structure(self, doc_id: str, node_id: str = None, depth: int = None) -> str:
        """Return document tree structure JSON (without text fields).

        For large trees, pass `depth` for a shallow view and `node_id` to expand a
        specific subtree (nodes flagged 'expandable' carry a 'child_count').
        """
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_document_structure(self.documents, doc_id, node_id=node_id, depth=depth)

    def get_page_content(self, doc_id: str, pages: str) -> str:
        """Return page content for the given pages string (e.g. '5-7', '3,8', '12')."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_page_content(self.documents, doc_id, pages)

    def get_node_content(self, doc_id: str, node_ids) -> str:
        """Return content for specific tree nodes (GitHub docs). node_ids: '0007', '0007,0012', or a list."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return get_node_content(self.documents, doc_id, node_ids)

    def search_document(self, doc_id: str, query: str, limit: int = 60) -> str:
        """Search a document's tree for nodes matching `query` (all words, case-insensitive)."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        return search_nodes(self.documents, doc_id, query, limit=limit)

    # ── GitHub sources ────────────────────────────────────────────────────────
    @staticmethod
    def _run_async(coro):
        """Run a coroutine whether or not an event loop is already running."""
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)

    def _register_doc(self, doc: dict) -> str:
        doc_id = str(uuid.uuid4())
        doc['id'] = doc_id
        self.documents[doc_id] = doc
        print(f"Indexing complete. Document ID: {doc_id}")
        if self.workspace:
            self._save_doc(doc_id)
        return doc_id

    def index_github_repo(self, repo_url: str) -> str:
        """Index a GitHub repository (dir -> file -> symbol tree). Returns a document_id."""
        from .sources import build_repo_doc
        print(f"Indexing GitHub repo: {repo_url}")
        with track_usage() as usage:
            doc = self._run_async(build_repo_doc(repo_url, self.opt, cache=self._cache))
        doc["index_usage"] = usage
        self._print_usage("Indexing", usage)
        return self._register_doc(doc)

    def index_github_tracker(self, repo_url: str, include: str = "both") -> str:
        """Index a repo's issues and/or PRs ('both', 'issues', 'prs'). Returns a document_id."""
        from .sources import build_tracker_doc
        print(f"Indexing GitHub issues/PRs: {repo_url}")
        with track_usage() as usage:
            doc = self._run_async(build_tracker_doc(repo_url, self.opt, cache=self._cache, include=include))
        doc["index_usage"] = usage
        self._print_usage("Indexing", usage)
        return self._register_doc(doc)

    def index_github_org(self, login: str, index_repos: bool = False, include_forks: bool = False) -> str:
        """Index all repos of an org/user as a corpus tree. Returns a document_id.

        index_repos=True eagerly indexes each repo and links it (costly for 100+ repos);
        the default lazily lists repos so the corpus index stays cheap.
        """
        from .sources import build_org_doc
        print(f"Indexing GitHub org/user: {login}")
        index_fn = (lambda r: self.index_github_repo(r)) if index_repos else None
        with track_usage() as usage:
            doc = self._run_async(
                build_org_doc(login, self.opt, cache=self._cache, include_forks=include_forks, index_repo_fn=index_fn)
            )
        doc["index_usage"] = usage
        self._print_usage("Indexing", usage)
        return self._register_doc(doc)

    def index_github(self, target: str, mode: str = "auto", **kwargs) -> str:
        """Convenience dispatcher. mode: 'repo' | 'tracker' | 'org' | 'auto'."""
        if mode == "repo":
            return self.index_github_repo(target)
        if mode == "tracker":
            return self.index_github_tracker(target, include=kwargs.get("include", "both"))
        if mode == "org":
            return self.index_github_org(target, **kwargs)
        # auto: "owner/name" -> repo; single segment -> org/user.
        cleaned = target.strip().rstrip("/")
        cleaned = cleaned.split("github.com/", 1)[-1]
        segments = [s for s in cleaned.split("/") if s]
        if len(segments) >= 2:
            return self.index_github_repo(target)
        return self.index_github_org(segments[0] if segments else target)

    def refresh(self, doc_id: str) -> str:
        """Incrementally re-index a GitHub document, reusing cached summaries for unchanged content."""
        if self.workspace:
            self._ensure_doc_loaded(doc_id)
        doc = self.documents.get(doc_id)
        if not doc:
            raise KeyError(f"Document {doc_id} not found")
        t = doc.get("type")
        with track_usage() as usage:
            if t == "github_repo":
                from .sources import build_repo_doc
                new = self._run_async(build_repo_doc(doc["repo"], self.opt, cache=self._cache, prev_doc=doc))
            elif t == "github_tracker":
                from .sources import build_tracker_doc
                new = self._run_async(
                    build_tracker_doc(doc["repo"], self.opt, cache=self._cache, include=doc.get("include", "both"))
                )
            elif t == "github_org":
                from .sources import build_org_doc
                new = self._run_async(build_org_doc(doc["login"], self.opt, cache=self._cache))
            else:
                raise ValueError(f"refresh is not supported for document type {t!r}")
        new["id"] = doc_id
        new["index_usage"] = usage
        self._print_usage("Refresh", usage)
        self.documents[doc_id] = new
        if self.workspace:
            self._save_doc(doc_id)
        return doc_id

    @staticmethod
    def _print_usage(label: str, usage: dict):
        """Print a one-line token-usage summary."""
        cost = usage.get("cost_usd") or 0.0
        cost_str = f" ~${cost:.4f}" if cost else ""
        print(
            f"{label} token usage: {usage.get('total_tokens', 0):,} total "
            f"({usage.get('prompt_tokens', 0):,} in / {usage.get('completion_tokens', 0):,} out) "
            f"over {usage.get('calls', 0)} LLM call(s){cost_str}"
        )
