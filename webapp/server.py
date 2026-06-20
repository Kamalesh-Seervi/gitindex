"""
Starlette backend for the PageIndex GitHub chat UI.

Endpoints
  GET  /api/docs            -> list indexed GitHub sources
  POST /api/index           -> index a repo/tracker/org   body: {target, mode, include}
  GET  /api/chat  (SSE)     -> stream an agent answer      query: ?doc_id=&message=
  GET  /                    -> static chat UI (webapp/static/index.html)

Run
  GITHUB_TOKEN=... python webapp/server.py      # -> http://127.0.0.1:8000

Indexing/summaries and the agent both use the model from pageindex/config.yaml
(e.g. github_copilot/...), so a GitHub Copilot subscription powers everything.
"""

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))   # for `import pageindex`
sys.path.insert(0, str(_HERE))   # for `import agent`

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from pageindex import PageIndexClient

from agent import build_agent, stream_agent

WORKSPACE = Path(os.getenv("PAGEINDEX_WORKSPACE", str(_ROOT / "examples" / "workspace")))
client = PageIndexClient(workspace=str(WORKSPACE))

# In-memory conversation history per document: doc_id -> [{role, content}, ...]
SESSIONS: dict[str, list[dict]] = {}


def _doc_summary(doc_id: str, doc: dict) -> dict:
    return {
        "doc_id": doc_id,
        "name": doc.get("doc_name") or doc.get("repo") or doc.get("login") or doc_id,
        "type": doc.get("type", "unknown"),
        "file_count": doc.get("file_count"),
        "item_count": doc.get("item_count"),
        "repo_count": doc.get("repo_count"),
        "index_usage": doc.get("index_usage"),
    }


def list_docs(request):
    # De-duplicate re-indexed sources (each index creates a new doc_id); keep the
    # most recent entry per (name, type).
    by_key: dict[tuple, dict] = {}
    for doc_id, doc in client.documents.items():
        if not str(doc.get("type", "")).startswith("github"):
            continue
        summary = _doc_summary(doc_id, doc)
        by_key[(summary["name"], summary["type"])] = summary
    docs = sorted(by_key.values(), key=lambda d: d["name"].lower())
    return JSONResponse({"docs": docs})


async def index_source(request):
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    target = (body.get("target") or "").strip()
    mode = body.get("mode") or "auto"
    include = body.get("include") or "both"

    if not target:
        return JSONResponse({"error": "Provide a repo/org, e.g. owner/name."}, status_code=400)
    if not (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")):
        return JSONResponse(
            {"error": "GITHUB_TOKEN (or GH_TOKEN) is not set on the server."}, status_code=400
        )

    try:
        if mode == "tracker":
            doc_id = await run_in_threadpool(client.index_github, target, "tracker", include=include)
        else:
            doc_id = await run_in_threadpool(client.index_github, target, mode)
    except Exception as exc:  # surface GraphQL/auth/LLM errors to the UI
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    SESSIONS.pop(doc_id, None)
    return JSONResponse({"doc_id": doc_id, "meta": json.loads(client.get_document(doc_id))})


def chat(request):
    doc_id = request.query_params.get("doc_id", "")
    message = (request.query_params.get("message") or "").strip()

    if doc_id not in client.documents:
        return JSONResponse({"error": "Unknown doc_id. Index a source first."}, status_code=400)
    if not message:
        return JSONResponse({"error": "Empty message."}, status_code=400)

    history = SESSIONS.setdefault(doc_id, [])
    history.append({"role": "user", "content": message})
    agent = build_agent(client, doc_id)

    async def events():
        parts: list[str] = []
        try:
            async for kind, payload in stream_agent(agent, list(history)):
                if kind == "text":
                    parts.append(payload)
                    yield {"event": "token", "data": json.dumps(payload)}
                elif kind == "usage":
                    yield {"event": "usage", "data": json.dumps(payload)}
                else:
                    yield {"event": "tool", "data": json.dumps(payload)}
        except Exception as exc:
            yield {"event": "error", "data": json.dumps(f"{type(exc).__name__}: {exc}")}
            return
        history.append({"role": "assistant", "content": "".join(parts)})
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(events())


routes = [
    Route("/api/docs", list_docs, methods=["GET"]),
    Route("/api/index", index_source, methods=["POST"]),
    Route("/api/chat", chat, methods=["GET"]),
    Mount("/", app=StaticFiles(directory=str(_HERE / "static"), html=True)),
]

app = Starlette(routes=routes)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"PageIndex chat UI -> http://{host}:{port}   (workspace: {WORKSPACE})")
    if not (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")):
        print("WARNING: GITHUB_TOKEN is not set; indexing will fail until you set it.")
    uvicorn.run(app, host=host, port=port)
