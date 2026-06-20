"""
Reusable agent runtime for the PageIndex GitHub chat UI.

Wraps the same vectorless retrieval tools used by the CLI demo
(get_document, get_document_structure, get_node_content) in an OpenAI Agents SDK
agent, routing the model through LiteLLM so a GitHub Copilot subscription (or any
LiteLLM provider) can drive the reasoning loop.
"""

from agents import Agent, ModelSettings, Runner, function_tool, set_tracing_disabled
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from openai.types.responses import ResponseTextDeltaEvent

set_tracing_disabled(True)

# GitHub Copilot's API rejects requests that lack IDE-auth headers. LiteLLM injects
# these automatically, but only when no extra_headers are passed. The OpenAI Agents
# SDK always sends extra_headers (its default User-Agent), which suppresses LiteLLM's
# Copilot defaults, so we must supply the IDE headers ourselves for that path.
_COPILOT_IDE_HEADERS = {
    "editor-version": "vscode/1.95.0",
    "editor-plugin-version": "copilot/1.155.0",
    "copilot-integration-id": "vscode-chat",
}

AGENT_SYSTEM_PROMPT = """
You are PageIndex, a codebase/issue QA assistant for GitHub sources.
The source is a TREE and can be huge (thousands of files) — never try to load it all.
TOOL USE:
- Call get_document() first to confirm the source type and size.
- To find or COUNT things across a big repo (e.g. "how many cycode sast tasks"),
  use search_repo(query="cycode sast") — it scans the whole tree in one call and
  returns matching node_ids. Prefer this over browsing for find/count questions.
- To explore, call get_document_structure() (no args) for the TOP level, then drill
  into nodes marked "expandable" with get_document_structure(node_id="0042").
- Once you've found the right leaves, call get_node_content(node_ids="0007,0012").
- Before each tool call, output one short sentence explaining the reason.
Answer based only on tool output. Cite node titles / file paths. Be concise.
"""


def _resolve_agent_model(model: str):
    """Make a config model usable by the OpenAI Agents SDK.

    Bare OpenAI ids (e.g. "gpt-4o") are handled by the SDK's default OpenAI client.
    Provider-prefixed ids (e.g. "github_copilot/...", "anthropic/...") are routed
    through LiteLLM so any LiteLLM provider drives the agent loop.
    """
    name = model.removeprefix("litellm/")
    if "/" not in name:
        return name
    try:
        from agents.extensions.models.litellm_model import LitellmModel
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"retrieve_model '{model}' needs LiteLLM routing for the agent. "
            "Install it with: pip install 'openai-agents[litellm]'"
        ) from exc
    return LitellmModel(model=name)


def _agent_model_settings(model: str) -> ModelSettings:
    """Per-model agent settings; adds Copilot IDE-auth headers for github_copilot models.

    ``include_usage=True`` asks LiteLLM for a final usage chunk during streaming so
    the SDK can report token counts.
    """
    if model.removeprefix("litellm/").startswith("github_copilot/"):
        return ModelSettings(extra_headers=dict(_COPILOT_IDE_HEADERS), include_usage=True)
    return ModelSettings(include_usage=True)


def build_agent(client, doc_id: str) -> Agent:
    """Build an agent whose tools are bound to a single indexed document."""

    @function_tool
    def get_document() -> str:
        """Get source metadata: type, repo/org, file or item count, name, description."""
        return client.get_document(doc_id)

    @function_tool
    def get_document_structure(node_id: str = "", depth: int = 2) -> str:
        """Browse the source tree one level at a time (it can be huge).

        - No node_id: returns the top level of the tree.
        - node_id: expands that subtree (use a node_id flagged "expandable").
        - depth: how many levels to reveal at once (default 2; keep it small).
        Read summaries to decide where to drill, then use get_node_content for code.
        """
        return client.get_document_structure(doc_id, node_id=node_id or None, depth=depth)

    @function_tool
    def get_node_content(node_ids: str) -> str:
        """Get the code/text for specific tree nodes, e.g. "0007" or "0007,0012"."""
        return client.get_node_content(doc_id, node_ids)

    @function_tool
    def search_repo(query: str, limit: int = 60) -> str:
        """Search the whole tree for nodes whose title/path/summary match all words in
        `query` (case-insensitive). Best for "find/how many X" questions over a large
        repo. Returns matching node_ids to pass to get_node_content.
        """
        return client.search_document(doc_id, query, limit=limit)

    return Agent(
        name="PageIndex-GitHub",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=[get_document, get_document_structure, get_node_content, search_repo],
        model=_resolve_agent_model(client.retrieve_model),
        model_settings=_agent_model_settings(client.retrieve_model),
    )


async def stream_agent(agent: Agent, input_items):
    """Async-generate ``(kind, text)`` events while the agent runs.

    kind is "text" for answer tokens, "tool" for a tool-call name, or "usage" for
    a final dict of token counts. ``input_items`` is a list of ``{"role", "content"}``
    dicts (the running conversation).
    """
    streamed = Runner.run_streamed(agent, input_items, max_turns=30)
    async for event in streamed.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            if isinstance(event.data, ResponseTextDeltaEvent):
                yield ("text", event.data.delta)
        elif isinstance(event, RunItemStreamEvent):
            if event.item.type == "tool_call_item":
                yield ("tool", getattr(event.item.raw_item, "name", "tool"))
    usage = getattr(getattr(streamed, "context_wrapper", None), "usage", None)
    if usage is not None:
        yield ("usage", {
            "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            "calls": getattr(usage, "requests", 0) or 0,
        })
