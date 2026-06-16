"""Mnemo public API — stable, documented entry points for all interfaces.

``MnemoAPI`` is the single facade for Python API, CLI, REST, and MCP
interfaces.  All public types are re-exported from this package.

Usage::

    from mnemo.api import MnemoAPI, FileInfo, SearchResult

    with MnemoAPI("~/my-kb") as api:
        file = api.add("paper.pdf", keys=["research::nlp"])
        results = api.search("attention mechanism")
"""

from mnemo.api.ask_types import AskResponse, Citation
from mnemo.api.client import MnemoAPI
from mnemo.api.types import (
    CheckReport,
    EmbedBackend,
    FileContext,
    FileInfo,
    FileStatus,
    ImportReport,
    KnowledgeBaseStats,
    SearchMode,
    SearchResult,
    SyncReport,
)

__all__ = [
    # Main facade
    "MnemoAPI",
    # Enums
    "EmbedBackend",
    "FileStatus",
    "SearchMode",
    # Data types
    "AskResponse",
    "CheckReport",
    "Citation",
    "FileContext",
    "FileInfo",
    "ImportReport",
    "KnowledgeBaseStats",
    "SearchResult",
    "SyncReport",
]
