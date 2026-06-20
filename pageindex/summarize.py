"""
Cheap "summarize everything" engine for GitHub source trees.

Design (think Merkle tree + memoization):

  Tier 0 — extractive, FREE: leaf summaries come from signatures, docstrings,
           module headers, issue titles. No LLM call for most leaves.
  Tier 1 — memoized: every summary is cached by a content hash (SummaryCache).
           Git blob OIDs are themselves content hashes, so identical files /
           symbols across many repos are summarized exactly once.
  Tier 2 — batched LLM (optional): when leaf_mode == "llm", leaves needing an
           abstractive summary are bin-packed into a single LiteLLM call.
  Tier 3 — bottom-up reduction: a directory / group / repo summary is generated
           from its children's (short) summaries, not raw text, bounding cost.

Incremental refresh exploits the same cache: unchanged blob OIDs hit the cache,
so re-summarizing an updated corpus costs ~0 LLM calls for the unchanged parts.
"""

import asyncio
import hashlib
import json
from pathlib import Path

try:
    from .utils import llm_acompletion, extract_json
except ImportError:  # pragma: no cover - allow flat imports
    from utils import llm_acompletion, extract_json


def content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


class SummaryCache:
    """A persistent content-hash -> summary map. No-ops gracefully without a path."""

    def __init__(self, path: str = None):
        self.path = Path(path) if path else None
        self._data: dict[str, str] = {}
        self._dirty = False
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8")) or {}
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value: str):
        if value is None:
            return
        if self._data.get(key) != value:
            self._data[key] = value
            self._dirty = True

    def save(self):
        if not self.path or not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            self._dirty = False
        except OSError:
            pass


# ── public entry point ────────────────────────────────────────────────────────
async def summarize_structure(
    structure: list,
    *,
    doc_kind: str,
    files: dict = None,
    cache: SummaryCache = None,
    model: str = None,
    summary_model: str = None,
    leaf_mode: str = "extractive",
    batch_size: int = 20,
):
    """Populate a `summary` on every node in `structure`, cheaply."""
    files = files or {}
    sm = summary_model or model

    leaves, _ = _split_nodes(structure)

    # Tier 0/1/2 — leaf summaries (skip leaves already summarized, e.g. reused on refresh).
    if doc_kind == "github_repo":
        pending = []
        for node in leaves:
            if node.get("summary"):
                continue
            s = _extractive_code_leaf(node, files)
            if s and leaf_mode != "llm":
                node["summary"] = s
            else:
                pending.append(node)
        if pending:
            await _batched_llm_leaf_summaries(pending, files, cache, sm, batch_size)
    elif doc_kind in ("github_tracker", "github_org"):
        for node in leaves:
            if not node.get("summary"):
                node["summary"] = _extractive_record_leaf(node)
    else:
        for node in leaves:
            node.setdefault("summary", node.get("title", ""))

    # Tier 3 — bottom-up reduction for internal nodes (few, cached).
    await asyncio.gather(*[_summarize_subtree(node, cache, sm) for node in structure])

    if cache:
        cache.save()
    return structure


# ── internal: leaves ──────────────────────────────────────────────────────────
def _split_nodes(structure: list) -> tuple[list, list]:
    leaves, internals = [], []

    def walk(nodes):
        for n in nodes:
            if n.get("nodes"):
                internals.append(n)
                walk(n["nodes"])
            else:
                leaves.append(n)

    walk(structure)
    return leaves, internals


def _node_source(node: dict, files: dict) -> str:
    """Resolve the raw text a code node points at (file slice by line range)."""
    path = node.get("path")
    text = files.get(path, "") if path else ""
    if not text:
        return ""
    start = node.get("start_line")
    end = node.get("end_line")
    if start and end:
        return "\n".join(text.splitlines()[start - 1 : end])
    return text


def _extractive_code_leaf(node: dict, files: dict) -> str:
    """Free, deterministic summary for a code leaf (signature/docstring/header)."""
    sig = (node.get("signature") or "").strip()
    doc = (node.get("docstring") or "").strip()
    if node.get("kind") == "symbol":
        parts = [p for p in (sig, doc) if p]
        return " — ".join(parts) if parts else (node.get("title") or "")
    # File leaf with no parsed symbols: use a module docstring / first real lines.
    src = _node_source(node, files)
    header = _leading_doc(src)
    if header:
        return header
    return f"File `{node.get('path', node.get('title', ''))}`"


def _extractive_record_leaf(node: dict) -> str:
    """Free summary for issue/PR/repo leaves."""
    if node.get("kind") in ("issue", "pr"):
        state = node.get("state", "")
        title = node.get("title", "")
        first = _leading_doc(node.get("text", ""), max_lines=2)
        head = f"[{state}] {title}".strip()
        return f"{head} — {first}" if first else head
    if node.get("kind") == "repo":
        desc = (node.get("description") or "").strip()
        lang = node.get("primary_language") or ""
        prefix = f"({lang}) " if lang else ""
        return f"{prefix}{desc}" if desc else (node.get("title") or "")
    return node.get("summary") or node.get("title", "")


async def _batched_llm_leaf_summaries(nodes, files, cache, model, batch_size):
    # Resolve source + hash; serve cache hits for free.
    todo = []
    for node in nodes:
        src = _node_source(node, files)
        key = content_hash(src) if src else content_hash(node.get("title", ""))
        node["_hash"] = key
        cached = cache.get(key) if cache else None
        if cached is not None:
            node["summary"] = cached
        elif src:
            todo.append(node)
        else:
            node["summary"] = node.get("title", "")

    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    await asyncio.gather(*[_run_leaf_batch(b, files, cache, model) for b in batches])
    for node in nodes:
        node.pop("_hash", None)


async def _run_leaf_batch(batch, files, cache, model):
    snippets = []
    for i, node in enumerate(batch):
        src = _node_source(node, files)
        snippets.append(f"### Snippet {i + 1}: {node.get('path', '')} {node.get('title', '')}\n{src[:1500]}")
    prompt = (
        "Summarize each numbered code snippet in ONE concise sentence describing what it does. "
        "Return ONLY a JSON array of strings, in the same order, with exactly "
        f"{len(batch)} items.\n\n" + "\n\n".join(snippets)
    )
    resp = await llm_acompletion(model, prompt)
    summaries = _coerce_list(resp, len(batch))
    for node, summary in zip(batch, summaries):
        summary = (summary or node.get("title", "")).strip()
        node["summary"] = summary
        if cache and node.get("_hash"):
            cache.set(node["_hash"], summary)


# ── internal: bottom-up reduction ─────────────────────────────────────────────
async def _summarize_subtree(node: dict, cache, model):
    children = node.get("nodes")
    if not children:
        return
    await asyncio.gather(*[_summarize_subtree(c, cache, model) for c in children])
    await _reduce_node(node, cache, model)


async def _reduce_node(node: dict, cache, model):
    child_lines = []
    for c in node["nodes"]:
        s = c.get("summary") or c.get("prefix_summary") or ""
        child_lines.append(f"- {c.get('title', '')}: {s}")
    joined = "\n".join(child_lines)
    key = content_hash(f"{node.get('kind', '')}|{node.get('title', '')}|{joined}")
    if cache:
        cached = cache.get(key)
        if cached is not None:
            node["summary"] = cached
            return
    label = node.get("kind", "section")
    prompt = (
        f"You are summarizing a {label} named '{node.get('title', '')}' in a code/project tree. "
        "Given its direct children below, write ONE concise sentence describing what this "
        f"{label} contains or is responsible for. Return only the sentence.\n\n{joined}"
    )
    summary = (await llm_acompletion(model, prompt) or node.get("title", "")).strip()
    node["summary"] = summary
    if cache:
        cache.set(key, summary)


# ── small helpers ─────────────────────────────────────────────────────────────
def _coerce_list(resp: str, n: int) -> list:
    try:
        data = extract_json(resp)
        if isinstance(data, list):
            return (list(data) + [""] * n)[:n]
    except Exception:
        pass
    # Fall back: split lines.
    lines = [l.strip("-• ").strip() for l in (resp or "").splitlines() if l.strip()]
    return (lines + [""] * n)[:n]


def _leading_doc(text: str, max_lines: int = 3, limit: int = 240) -> str:
    """Pull the first meaningful line(s): docstring, comment, or heading."""
    if not text:
        return ""
    out = []
    for raw in text.splitlines():
        line = raw.strip().strip("#/*\"'` ").strip()
        if not line:
            if out:
                break
            continue
        out.append(line)
        if len(out) >= max_lines:
            break
    return " ".join(out)[:limit]
