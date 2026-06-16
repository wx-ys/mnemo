"""Core data types shared across all interfaces.

These are the internal types used by plugins and the KB engine.
Public API types (BaseModel-based) live in ``mnemo.api.types``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FileMeta(BaseModel):
    """Metadata for a single file in the knowledge base.

    Parameters
    ----------
    id : str
        Unique file identifier (UUID).
    file_type : str
        File extension without dot, e.g. 'pdf', 'csv'.
    filename : str
        Original filename.
    file_hash : str
        Content hash in 'algorithm:hex' format, e.g. 'sha256:abc123...'.
    file_size : int
        File size in bytes.
    source_path : str
        Original source path before ingestion.
    raw_path : str
        Relative path to raw file under data_dir.
    metadata_path : str
        Relative path to metadata .md file.
    md_path : str
        Relative path to markdown .md file.
    wiki_path : str
        Relative path to wiki .md file.
    md_status : str
        Processing status: 'pending' | 'done' | 'failed' | 'skipped'.
    wiki_status : str
        Processing status: 'pending' | 'done' | 'failed' | 'skipped'.
    embed_status : str
        Processing status: 'pending' | 'done' | 'failed' | 'skipped'.
    category : str
        Content category, e.g. 'paper', 'dataset'.
    tags : list[str]
        Flat tags for filtering.
    keywords : list[str]
        Extracted or user-provided keywords.
    added_at : str
        ISO 8601 timestamp of ingestion.
    updated_at : str
        ISO 8601 timestamp of last update.
    custom : dict
        User-defined custom fields (importance, status, etc.).
    source_kb : str
        Source knowledge base identifier when imported from another Mnemo instance.
    """

    model_config = {"extra": "allow"}

    id: str
    file_type: str
    filename: str
    file_hash: str
    file_size: int
    source_path: str

    raw_path: str = ""
    metadata_path: str = ""
    md_path: str = ""
    wiki_path: str = ""

    md_status: str = "pending"
    wiki_status: str = "pending"
    embed_status: str = "pending"

    category: str = ""
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    added_at: str = ""
    updated_at: str = ""
    deleted_at: str = ""

    custom: dict[str, Any] = Field(default_factory=dict)
    source_kb: str = ""


class SearchResult(BaseModel):
    """A single search result before deduplication.

    Parameters
    ----------
    id : str
        File identifier.
    file_path : str
        Path to the matching file.
    score : float
        Relevance score (0.0 to 1.0).
    snippet : str
        Matching text snippet with surrounding context.
    match_source : str
        Which index produced this match: 'wiki' | 'md' | 'metadata' | 'keyword'.
    file_type : str
        File extension.
    wiki_summary : str or None
        Wiki summary text if available.
    """

    model_config = {"extra": "allow"}

    id: str
    file_path: str
    score: float
    snippet: str
    match_source: str
    file_type: str = ""
    wiki_summary: str | None = None


class GroupedResult(BaseModel):
    """Search result after deduplication by file.

    When a long file is split into multiple chunks and several chunks
    match the same query, they are merged into one GroupedResult.

    Parameters
    ----------
    file_id : str
        File identifier.
    score : float
        Highest score among the merged chunks.
    top_snippet : str
        Snippet from the highest-scoring chunk.
    match_count : int
        Number of chunks that matched.
    all_snippets : list[str]
        Snippets from all matching chunks (expandable).
    file_type : str
        File extension.
    wiki_summary : str or None
        Wiki summary text if available.
    """

    model_config = {"extra": "allow"}

    file_id: str
    score: float
    top_snippet: str
    match_count: int
    all_snippets: list[str] = Field(default_factory=list)
    file_type: str = ""
    wiki_summary: str | None = None


class ChunkInfo(BaseModel):
    """A single embedding chunk extracted from a document.

    Parameters
    ----------
    text : str
        Chunk text content.
    chunk_index : int
        Zero-based index of this chunk within the document.
    start_char : int
        Character offset where this chunk starts in the source text.
    end_char : int
        Character offset where this chunk ends in the source text.
    metadata : dict
        Arbitrary metadata: ``section_header``, ``header_path``,
        ``parent_id``, ``chunk_level``, ``language``, etc.
    """

    model_config = {"extra": "allow"}

    text: str
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
