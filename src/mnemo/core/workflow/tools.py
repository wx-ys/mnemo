"""Tool library — registry of pydantic-ai Tools available to AgentSteps.

Tools are registered by name and resolved at agent creation time.
When an AgentStep declares ``tools = ["search_kb", "get_file_context"]``,
the ToolLibrary resolves those names to actual pydantic-ai :class:`Tool`
instances that the Agent can call.

Built-in tools:

============  ====================================================
Tool          Description
============  ====================================================
search_kb     Search the knowledge base (hybrid ANN + BM25 + graph)
get_context   Get a file's full context (md + wiki + entities)
chunk_text    Split text into chunks using the configured chunker
extract_kw    Extract keywords from text via LLM
============  ====================================================

Usage::

    from mnemo.core.workflow.tools import ToolLibrary

    @ToolLibrary.register("my_tool")
    async def my_tool(ctx: WorkflowDeps, query: str) -> str: ...

    tool = ToolLibrary.resolve("my_tool")  # returns pydantic_ai.Tool
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from typing import Any


class ToolLibrary:
    """Registry of named tool functions.

    Registered tools are plain Python functions (sync or async) that
    take LLM-callable parameters.  They are NOT wrapped in
    ``pydantic_ai.Tool`` until they are passed to an Agent — this
    avoids premature schema generation issues with complex types.

    Thread-safe for reads after registration.
    """

    _tools: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator — register a tool function under *name*.

        Usage::

            @ToolLibrary.register("search_kb")
            def search_kb(query: str, mode: str = "hybrid", limit: int = 5) -> list[dict]:
                ...
        """
        def decorator(fn: Callable) -> Callable:
            cls._tools[name] = fn
            return fn
        return decorator

    @classmethod
    def resolve(cls, tool_name: str) -> Callable[..., Any] | None:
        """Look up a registered tool function by name.

        Returns the raw function — pydantic-ai wrapping happens
        inside ``AgentRegistry`` when the Agent is created.
        """
        return cls._tools.get(tool_name)

    @classmethod
    def resolve_many(cls, tool_names: list[str]) -> list[Callable[..., Any]]:
        """Resolve multiple tool names at once.

        Unknown names are silently skipped (with a debug log).
        Returns plain functions, not pydantic-ai Tool wrappers.
        """
        funcs: list[Callable[..., Any]] = []
        for name in tool_names:
            fn = cls.resolve(name)
            if fn is not None:
                funcs.append(fn)
            else:
                import logging
                logging.getLogger("mnemo.workflow").debug(
                    "Tool '%s' not found in ToolLibrary — skipping", name,
                )
        return funcs

    @classmethod
    def list_tools(cls) -> list[str]:
        """Return all registered tool names."""
        return list(cls._tools.keys())


# ============================================================================
# Built-in tool implementations
# ============================================================================
#
# Tools are plain functions that take simple types (str, int, dict)
# as parameters — the LLM fills these via tool calling.  Access to
# the KnowledgeBase happens through _get_kb() which resolves the
# currently active instance.
# ============================================================================


def _get_kb() -> Any:
    """Resolve the active KnowledgeBase instance, or return None."""
    try:
        from mnemo.core.agent_manager import AgentManager
        am = AgentManager.get_instance()
        if hasattr(am, '_kb_ref') and am._kb_ref is not None:
            return am._kb_ref
    except Exception:
        pass
    return None


@ToolLibrary.register("search_kb")
def search_kb_tool(
    query: str,
    mode: str = "hybrid",
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search the knowledge base for relevant content.

    Parameters
    ----------
    query : Search query string.
    mode : Search mode: "hybrid", "vector", or "keyword".
    limit : Maximum number of results (1-20).
    """
    kb = _get_kb()
    if kb is None:
        return [{"error": "KnowledgeBase not available"}]

    try:
        from mnemo.api.types import SearchMode

        mode_map = {
            "hybrid": SearchMode.HYBRID,
            "vector": SearchMode.VECTOR,
            "keyword": SearchMode.KEYWORD,
        }
        results = kb.search(
            query,
            mode=mode_map.get(mode, SearchMode.HYBRID),
            limit=min(limit, 20),
        )
        return [
            {
                "id": r.id,
                "snippet": r.snippet,
                "score": r.score,
                "file_type": r.file_type,
            }
            for r in results
        ]
    except Exception as exc:
        return [{"error": f"Search failed: {exc}"}]


@ToolLibrary.register("get_file_context")
def get_file_context_tool(
    file_id: str,
) -> dict[str, Any]:
    """Get full context (markdown, wiki, entities) for a file.

    Parameters
    ----------
    file_id : File UUID or filename to look up.
    """
    kb = _get_kb()
    if kb is None:
        return {"error": "KnowledgeBase not available"}

    try:
        context = kb.get_context(file_id)
        return {
            "file_id": context.file_id,
            "file_type": context.file_type,
            "filename": context.filename,
            "category": context.category,
            "tags": context.tags,
            "keys": context.keys,
            "md_content": context.md_content[:2000] if context.md_content else "",
            "wiki_content": context.wiki_content[:2000] if context.wiki_content else "",
            "entities": [
                e.get("name", str(e)) if isinstance(e, dict) else str(e)
                for e in context.entities[:30]
            ] if context.entities else [],
        }
    except Exception as exc:
        return {"error": f"Failed to get context: {exc}"}


@ToolLibrary.register("chunk_text")
def chunk_text_tool(
    text: str,
    strategy: str = "paragraph",
) -> dict[str, Any]:
    """Split text into chunks.

    Parameters
    ----------
    text : Text content to split.
    strategy : Chunking strategy name (e.g. "paragraph", "token").
    """
    try:
        from mnemo.core.plugin_base import PluginHub
        from mnemo.core.interfaces import IChunker
        chunker = PluginHub.get(IChunker, strategy)
        cfg = getattr(chunker, "config_schema", {})
        chunks = chunker.chunk(text, {k: v.default for k, v in cfg.items()
                                       if hasattr(v, 'default')})
        return {
            "count": len(chunks),
            "chunks": [c.text for c in chunks[:20]],
        }
    except Exception as exc:
        return {"error": f"Chunking failed: {exc}"}


@ToolLibrary.register("resolve_file_ref")
def resolve_file_ref_tool(
    ref: str,
) -> dict[str, Any]:
    """Resolve a file reference (UUID or filename) to a file ID.

    Parameters
    ----------
    ref : File UUID or filename to resolve.
    """
    kb = _get_kb()
    if kb is None:
        return {"error": "KnowledgeBase not available"}

    try:
        file_id = kb.resolve_file_ref(ref)
        return {"file_id": file_id, "found": file_id is not None}
    except Exception as exc:
        return {"error": str(exc), "file_id": None, "found": False}
