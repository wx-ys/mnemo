"""Mnemo public API type definitions — pydantic models.

These types form the external contract for all interfaces (CLI, REST,
MCP, Python API).  All types are pydantic ``BaseModel`` subclasses,
providing automatic validation, serialization (``model_dump()``), and
JSON Schema generation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SearchMode(str, Enum):
    """Search mode selector."""
    HYBRID = "hybrid"       # vector + keyword combined
    VECTOR = "vector"       # vector-only ANN
    KEYWORD = "keyword"     # keyword-only BM25


class FileStatus(str, Enum):
    """File processing status."""
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class EmbedBackend(str, Enum):
    """Embedding backend identifier."""
    LOCAL = "local"
    OPENAI = "openai"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class FileInfo(BaseModel):
    """File information returned by the public API."""

    id: str = ""
    file_type: str = ""
    filename: str = ""
    file_size: int = 0
    file_hash: str = ""

    raw_path: str = ""
    metadata_path: str = ""
    md_path: str = ""
    wiki_path: str = ""

    md_status: FileStatus = FileStatus.PENDING
    wiki_status: FileStatus = FileStatus.PENDING
    embed_status: FileStatus = FileStatus.PENDING

    category: str = ""
    tags: list[str] = []
    keys: list[str] = []
    keywords: list[str] = []

    added_at: str = ""
    updated_at: str = ""

    source_path: str = ""
    source_kb: str = ""

    version: int = 1
    related_files: list[str] = []

    custom: dict[str, Any] = {}

    # Detailed per-step processing info for CLI display
    processing_detail: dict[str, Any] = {}

    # Internal marker (not serialized)
    _duplicate: bool = False


class SearchResult(BaseModel):
    """Search result returned by the public API."""

    id: str = ""
    file_path: str = ""
    score: float = 0.0
    snippet: str = ""
    match_source: str = ""
    file_type: str = ""
    wiki_summary: str | None = None
    match_count: int = 1
    all_snippets: list[str] = []


class FileContext(BaseModel):
    """Full file context for agent consumption."""

    file_id: str = ""
    file_type: str = ""
    filename: str = ""
    category: str = ""
    tags: list[str] = []
    keys: list[str] = []
    md_content: str = ""
    wiki_content: str = ""
    metadata_content: str = ""
    user_notes: str = ""
    entities: list[dict] = []


class KnowledgeBaseStats(BaseModel):
    """Aggregate knowledge base statistics."""

    total_files: int = 0
    total_size: int = 0                # bytes
    type_breakdown: dict[str, int] = {}
    embed_count: int = 0
    key_count: int = 0
    last_sync: str = ""


class CheckReport(BaseModel):
    """Integrity check report."""

    status: str = "ok"                 # 'ok' | 'warning' | 'error'
    issues: list[dict] = []
    suggestions: list[str] = []


class ImportReport(BaseModel):
    """Import operation report."""

    imported: int = 0
    skipped: int = 0
    errors: list[str] = []


class SyncReport(BaseModel):
    """Sync operation report."""

    direction: str = ""                # 'push' | 'pull'
    synced: int = 0
    skipped: int = 0
    errors: list[str] = []
    timestamp: str = ""
