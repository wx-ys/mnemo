"""Mnemo MCP (Model Context Protocol) server.

Provides Mnemo as a tool provider for AI agents (Claude Desktop, Cursor,
etc.) via the standard MCP stdio transport.

Usage::

    mnemo mcp  # starts the MCP server on stdio

Claude Desktop config (``claude_desktop_config.json``)::

    {
      "mcpServers": {
        "mnemo": {
          "command": "uv",
          "args": ["run", "mnemo", "mcp", "-d", "/path/to/kb"]
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mnemo.api import MnemoAPI, SearchMode

logger = logging.getLogger("mnemo.mcp")

# ---------------------------------------------------------------------------
# Server definition
# ---------------------------------------------------------------------------

server = Server("mnemo")

# Global API instance (set by run())
_api: MnemoAPI | None = None


def _get_api() -> MnemoAPI:
    """Get the global API instance, creating it if needed."""
    global _api
    if _api is None:
        _api = MnemoAPI()
    return _api


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOLS = [
    Tool(
        name="mnemo_search",
        description=(
            "Semantic search over the Mnemo knowledge base. "
            "Supports hybrid (vector + keyword + graph), vector-only, "
            "and keyword-only search modes."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text",
                },
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "vector", "keyword"],
                    "description": "Search mode. Default: hybrid",
                    "default": "hybrid",
                },
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Limit to these key scopes (e.g. ['research::nlp'])",
                },
                "file_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Limit to these file extensions (e.g. ['pdf', 'py'])",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results. Default 10, max 50.",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="mnemo_ask",
        description=(
            "Ask a question and get a knowledge-base-grounded answer with "
            "inline citations to source files. Uses RAG pipeline: search → "
            "rerank → context assembly → LLM answer."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to answer from the knowledge base",
                },
                "grounded": {
                    "type": "boolean",
                    "description": "If true, answer is strictly grounded in KB content. Default: true",
                    "default": True,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum source chunks to retrieve. Default 10, max 20.",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["question"],
        },
    ),
    Tool(
        name="mnemo_list",
        description=(
            "List files in the knowledge base with optional filters. "
            "Returns file metadata: ID, filename, type, category, size, keys, tags."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_type": {
                    "type": "string",
                    "description": "Filter by file extension, e.g. 'pdf', 'py'",
                },
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by hierarchical keys (AND logic)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by flat tags (AND logic)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results. Default 50, max 200.",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 200,
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset. Default 0.",
                    "default": 0,
                    "minimum": 0,
                },
            },
        },
    ),
    Tool(
        name="mnemo_get",
        description=(
            "Get full file context for AI agent consumption. Returns "
            "metadata, markdown content, wiki summary, user notes, and "
            "linked graph entities for a file."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "File UUID or filename",
                },
            },
            "required": ["file_id"],
        },
    ),
    Tool(
        name="mnemo_stats",
        description=(
            "Get aggregate knowledge base statistics: total files, "
            "total size, type breakdown, embedding count, key count."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


# ---------------------------------------------------------------------------
# Handler registration
# ---------------------------------------------------------------------------

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """Return the list of available MCP tools."""
    return _TOOLS


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any],
) -> list[TextContent]:
    """Dispatch tool calls to the appropriate handler."""
    api = _get_api()

    try:
        if name == "mnemo_search":
            return await _handle_search(api, arguments)
        elif name == "mnemo_ask":
            return await _handle_ask(api, arguments)
        elif name == "mnemo_list":
            return await _handle_list(api, arguments)
        elif name == "mnemo_get":
            return await _handle_get(api, arguments)
        elif name == "mnemo_stats":
            return await _handle_stats(api)
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"}),
            )]
    except Exception as exc:
        logger.exception("Tool '%s' failed", name)
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(exc)}, ensure_ascii=False),
        )]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def _handle_search(api: MnemoAPI, args: dict) -> list[TextContent]:
    """Execute mnemo_search."""
    mode_map = {
        "hybrid": SearchMode.HYBRID,
        "vector": SearchMode.VECTOR,
        "keyword": SearchMode.KEYWORD,
    }
    mode = mode_map.get(args.get("mode", "hybrid"), SearchMode.HYBRID)

    results = api.search(
        query=args["query"],
        mode=mode,
        keys=args.get("keys"),
        file_types=args.get("file_types"),
        limit=min(args.get("limit", 10), 50),
    )

    return [TextContent(
        type="text",
        text=json.dumps(
            [r.model_dump() for r in results],
            ensure_ascii=False, indent=2,
        ),
    )]


async def _handle_ask(api: MnemoAPI, args: dict) -> list[TextContent]:
    """Execute mnemo_ask — RAG Q&A with citations."""
    response = api.ask(
        question=args["question"],
        grounded=args.get("grounded", True),
        limit=min(args.get("limit", 10), 20),
    )

    return [TextContent(
        type="text",
        text=json.dumps(
            response.model_dump(),
            ensure_ascii=False, indent=2,
        ),
    )]


async def _handle_list(api: MnemoAPI, args: dict) -> list[TextContent]:
    """Execute mnemo_list."""
    results = api.list_files(
        file_type=args.get("file_type"),
        keys=args.get("keys"),
        tags=args.get("tags"),
        limit=min(args.get("limit", 50), 200),
        offset=args.get("offset", 0),
    )

    return [TextContent(
        type="text",
        text=json.dumps(
            [r.model_dump() for r in results],
            ensure_ascii=False, indent=2,
        ),
    )]


async def _handle_get(api: MnemoAPI, args: dict) -> list[TextContent]:
    """Execute mnemo_get."""
    file_id = args["file_id"]

    # Try direct ID first, then resolve by filename
    resolved = api.resolve_file_ref(file_id)
    if resolved is None:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"File not found: {file_id}"}),
        )]
    file_id = resolved

    info = api.get(file_id)
    context = api.get_context(file_id)

    # Merge info + context into one response
    output = {
        **info.model_dump(),
        **context.model_dump(),
    }

    return [TextContent(
        type="text",
        text=json.dumps(output, ensure_ascii=False, indent=2),
    )]


async def _handle_stats(api: MnemoAPI) -> list[TextContent]:
    """Execute mnemo_stats."""
    return [TextContent(
        type="text",
        text=json.dumps(
            api.stats().model_dump(),
            ensure_ascii=False, indent=2,
        ),
    )]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run(data_dir: str | None = None) -> None:
    """Start the MCP stdio server.

    Parameters
    ----------
    data_dir : str, optional
        Knowledge base data directory. If None, uses the default
        (~/mnemo-data).
    """
    global _api

    if data_dir:
        _api = MnemoAPI(data_dir)
    else:
        _api = MnemoAPI()

    logger.info("Mnemo MCP server starting (data_dir=%s)", _api.data_dir)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
