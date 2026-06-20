"""
Agentic Vectorless RAG over a GitHub repo / issues / org — Demo

Same vectorless, reasoning-based retrieval as the PDF demo, but the index is built
from GitHub via the GraphQL API instead of a document. PageIndex turns a repo into a
dir -> file -> symbol tree (or issues/PRs, or a whole org) and the agent reasons over
that tree to fetch only the nodes it needs — no vectors, no chunking.

Agent tools:
  - get_document()           — metadata (repo, file/item count, ...)
  - get_document_structure() — the tree index (titles + summaries, no code)
  - get_node_content()       — fetch code/text for specific node_ids from the tree

Setup:
  export GITHUB_TOKEN=...        # required (GraphQL API + private repos)
  export OPENAI_API_KEY=...      # only if retrieve_model is a bare OpenAI model (e.g. gpt-4o)
  pip install "openai-agents[litellm]" requests

LLM provider:
  The agent uses `retrieve_model` from config.yaml. A bare id (e.g. "gpt-4o") runs on
  OpenAI directly; a provider-prefixed id (e.g. "github_copilot/claude-opus-4-8",
  "anthropic/...") is routed through LiteLLM, so a GitHub Copilot subscription works
  with no OPENAI_API_KEY (a one-time `github.com/login/device` prompt appears).

Usage:
  python3 examples/github_vectorless_rag_demo.py                 # indexes a repo
  MODE=tracker REPO=owner/name python3 examples/github_vectorless_rag_demo.py
  MODE=org ORG=some-org python3 examples/github_vectorless_rag_demo.py
"""
import os
import sys
import json
import asyncio
import concurrent.futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, ModelSettings, Runner, function_tool, set_tracing_disabled
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from openai.types.responses import ResponseTextDeltaEvent, ResponseReasoningSummaryTextDeltaEvent

from pageindex import PageIndexClient
import pageindex.utils as utils

WORKSPACE = Path(__file__).parent / "workspace"

MODE = os.getenv("MODE", "repo")            # "repo" | "tracker" | "org"
REPO = os.getenv("REPO", "VectifyAI/PageIndex")
ORG = os.getenv("ORG", "VectifyAI")
QUESTION = os.getenv(
    "QUESTION",
    "How does PageIndex perform vectorless retrieval? Point to the relevant code.",
)

AGENT_SYSTEM_PROMPT = """
You are PageIndex, a codebase/issue QA assistant for GitHub sources.
The source is a TREE and can be huge (thousands of files) — never try to load it all.
TOOL USE:
- Call get_document() first to confirm the source type and size.
- Call get_document_structure() (no args) to see the TOP level of the tree.
- Nodes marked "expandable" have a "child_count"; drill in with get_document_structure(node_ids="0042").
- Use summaries to decide where to go; keep depth small (1-2) and follow node_ids.
- Once you've found the right leaves, call get_node_content(node_ids="0007,0012") for code.
- Before each tool call, output one short sentence explaining the reason.
Answer based only on tool output. Cite node titles / file paths. Be concise.
"""


# GitHub Copilot's API rejects requests that lack IDE-auth headers. LiteLLM injects
# these automatically — but only when no extra_headers are passed. The OpenAI Agents
# SDK always sends extra_headers (its default User-Agent), which suppresses LiteLLM's
# Copilot defaults, so we must supply the IDE headers ourselves for that path.
_COPILOT_IDE_HEADERS = {
    "editor-version": "vscode/1.95.0",
    "editor-plugin-version": "copilot/1.155.0",
    "copilot-integration-id": "vscode-chat",
}


def _resolve_agent_model(model: str):
    """Make a config model usable by the OpenAI Agents SDK.

    Bare OpenAI ids (e.g. "gpt-4o") are passed through and handled by the SDK's
    default OpenAI client. Provider-prefixed ids (e.g. "github_copilot/...",
    "anthropic/...") are routed through LiteLLM so any LiteLLM provider — including
    a GitHub Copilot subscription — drives the agent loop, matching the LiteLLM path
    already used for indexing/summaries.
    """
    name = model.removeprefix("litellm/")
    if "/" not in name:
        return name
    try:
        from agents.extensions.models.litellm_model import LitellmModel
    except ImportError as exc:
        raise SystemExit(
            f"retrieve_model '{model}' needs LiteLLM routing for the agent. "
            "Install it with: pip install 'openai-agents[litellm]'"
        ) from exc
    return LitellmModel(model=name)


def _agent_model_settings(model: str) -> ModelSettings:
    """Per-model agent settings; adds Copilot IDE-auth headers for github_copilot models."""
    if model.removeprefix("litellm/").startswith("github_copilot/"):
        return ModelSettings(extra_headers=dict(_COPILOT_IDE_HEADERS))
    return ModelSettings()


def query_agent(client: PageIndexClient, doc_id: str, prompt: str, verbose: bool = False) -> str:
    @function_tool
    def get_document() -> str:
        """Get source metadata: type, repo/org, file or item count, name, description."""
        return client.get_document(doc_id)

    @function_tool
    def get_document_structure(node_id: str = "", depth: int = 2) -> str:
        """Browse the tree one level at a time (it can be huge).

        No node_id returns the top level; pass an "expandable" node_id to expand that
        subtree; depth controls how many levels to reveal (default 2, keep small).
        """
        return client.get_document_structure(doc_id, node_id=node_id or None, depth=depth)

    @function_tool
    def get_node_content(node_ids: str) -> str:
        """
        Get the code/text for specific tree nodes.
        node_ids: one id ('0007'), several ('0007,0012'), from the structure's node_id fields.
        """
        return client.get_node_content(doc_id, node_ids)

    @function_tool
    def search_repo(query: str, limit: int = 60) -> str:
        """Search the whole tree for nodes whose title/path/summary match all words in
        `query` (case-insensitive). Best for "find/how many X" questions over a large repo.
        Returns matching node_ids to pass to get_node_content.
        """
        return client.search_document(doc_id, query, limit=limit)

    agent = Agent(
        name="PageIndex-GitHub",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=[get_document, get_document_structure, get_node_content, search_repo],
        model=_resolve_agent_model(client.retrieve_model),
        model_settings=_agent_model_settings(client.retrieve_model),
    )

    async def _run():
        streamed_run = Runner.run_streamed(agent, prompt, max_turns=30)
        current = None
        async for event in streamed_run.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                if isinstance(event.data, ResponseReasoningSummaryTextDeltaEvent):
                    if current != "reasoning":
                        print("\n[reasoning]: ", end="", flush=True)
                    print(event.data.delta, end="", flush=True)
                    current = "reasoning"
                elif isinstance(event.data, ResponseTextDeltaEvent):
                    if current != "text":
                        print("\n[text]: ", end="", flush=True)
                    print(event.data.delta, end="", flush=True)
                    current = "text"
            elif isinstance(event, RunItemStreamEvent):
                item = event.item
                if item.type == "tool_call_item":
                    raw = item.raw_item
                    args = f"({getattr(raw, 'arguments', '{}')})" if verbose else ""
                    print(f"\n[tool call]: {raw.name}{args}", flush=True)
                    current = None
                elif item.type == "tool_call_output_item" and verbose:
                    output = str(item.output)
                    preview = output[:200] + "..." if len(output) > 200 else output
                    print(f"\n[tool call output]: {preview}", flush=True)
                    current = None
        if current is not None:
            print()
        return "" if not streamed_run.final_output else str(streamed_run.final_output)

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run()).result()
    except RuntimeError:
        return asyncio.run(_run())


def index_source(client: PageIndexClient) -> str:
    if MODE == "tracker":
        print(f"Indexing issues & PRs for {REPO} ...")
        return client.index_github_tracker(REPO)
    if MODE == "org":
        print(f"Indexing org/user {ORG} ...")
        return client.index_github_org(ORG)
    print(f"Indexing repo {REPO} ...")
    return client.index_github_repo(REPO)


if __name__ == "__main__":
    if not (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")):
        sys.exit("Please set GITHUB_TOKEN (or GH_TOKEN) in your environment.")

    set_tracing_disabled(True)
    client = PageIndexClient(workspace=WORKSPACE)

    print("=" * 60)
    print(f"Step 1: Index GitHub source (mode={MODE}) and view tree")
    print("=" * 60)
    doc_id = index_source(client)
    print(f"\nIndexed. doc_id: {doc_id}")
    print("\nTree Structure:")
    structure = json.loads(client.get_document_structure(doc_id))
    utils.print_tree(structure)

    print("\n" + "=" * 60)
    print("Step 2: View source metadata")
    print("=" * 60)
    print(f"\n{client.get_document(doc_id)}")

    print("\n" + "=" * 60)
    print("Step 3: Agent Query (auto tool-use)")
    print("=" * 60)
    print(f"\nQuestion: '{QUESTION}'")
    query_agent(client, doc_id, QUESTION, verbose=True)
