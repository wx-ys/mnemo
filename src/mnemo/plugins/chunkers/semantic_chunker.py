"""Semantic chunking — LLM-free embedding-based topic boundary detection.

Uses ``langchain-experimental`` SemanticChunker with Mnemo's configured
embedder to detect natural topic boundaries via embedding similarity.

Only activated when user explicitly sets ``chunker = "semantic"``
in their file category config (default is ``LangChainChunker``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mnemo.core.interfaces.chunker import IChunker
from mnemo.core.interfaces.param_spec import Param

if TYPE_CHECKING:
    from mnemo.core.interfaces.types import ChunkInfo


class _EmbeddingsAdapter:
    """Adapt pydantic-ai Embedder to langchain's Embeddings interface."""

    def __init__(self, embedder):
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = self._embedder.embed_documents_sync(texts)
        return [list(v) for v in result.embeddings]

    def embed_query(self, text: str) -> list[float]:
        result = self._embedder.embed_query_sync(text)
        return list(result.embeddings[0])


class SemanticChunker(IChunker):
    """Semantic boundary chunking via embedding similarity.

    Splits text into sentences, embeds each sentence, and detects
    topic boundaries where cosine distance between adjacent sentences
    exceeds a percentile threshold.  No LLM is involved — only the
    configured embedder is used.

    **Cost:** ~N embedding calls per document (N = number of sentences).
    Use selectively for content where semantic coherence matters.
    """

    __plugin_impl__ = True
    name = "semantic"

    config_schema = {
        "breakpoint_threshold": Param(
            type="float", default=95.0, desc="Percentile threshold for breakpoint detection (0-100)"),
        "min_chunk_size": Param(
            type="int", default=100, desc="Minimum characters per semantic chunk"),
        "buffer_size": Param(
            type="int", default=1, desc="Sliding window size for sentence grouping"),
    }

    # ── IChunker ───────────────────────────────────────────────────────

    def chunk(
        self, text: str, config: dict | None = None,
    ) -> list[ChunkInfo]:
        """Split *text* at semantic boundaries.

        Parameters
        ----------
        text : str
            Text to split.
        config : dict, optional
            Resolved config with ``breakpoint_threshold``, ``min_chunk_size``,
            ``buffer_size``, and optionally ``_embedder`` (injected by
            ``_resolve_chunker``).

        Returns
        -------
        list of ChunkInfo
        """
        from mnemo.core.interfaces.types import ChunkInfo

        cfg = config or {}
        embedder = cfg.get("_embedder")

        # Fall back to paragraph chunking if embedder unavailable
        if embedder is None:
            from mnemo.plugins.chunkers.paragraph_chunker import ParagraphChunker
            return ParagraphChunker().chunk(text, config)

        threshold = float(cfg.get("breakpoint_threshold", 95.0))
        min_size = int(cfg.get("min_chunk_size", 100))
        buffer_size = int(cfg.get("buffer_size", 1))

        try:
            from langchain_experimental.text_splitter import (
                SemanticChunker as LCSemantic,
            )
            splitter = LCSemantic(
                embeddings=_EmbeddingsAdapter(embedder),
                buffer_size=buffer_size,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=threshold,
                min_chunk_size=min_size,
                add_start_index=True,
            )
            docs = splitter.create_documents([text])
        except Exception:
            from mnemo.plugins.chunkers.paragraph_chunker import ParagraphChunker
            return ParagraphChunker().chunk(text, config)

        chunks: list[ChunkInfo] = []
        offset = 0
        for i, doc in enumerate(docs):
            content = doc.page_content
            start = text.find(content, max(0, offset - len(content)))
            if start < 0:
                start = offset
            end = start + len(content)
            offset = end

            meta = dict(doc.metadata) if doc.metadata else {}
            chunks.append(ChunkInfo(
                text=content,
                chunk_index=i,
                start_char=start,
                end_char=end,
                metadata=meta,
            ))

        return chunks
