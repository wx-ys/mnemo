"""Paragraph-based chunking — splits on ``\\n\\n`` paragraph boundaries.

This is the default chunker and preserves the original behaviour that
was previously implemented in ``BaseParser.chunk()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mnemo.core.interfaces.chunker import IChunker

if TYPE_CHECKING:
    from mnemo.core.interfaces.types import ChunkInfo


class ParagraphChunker(IChunker):
    """Paragraph-based text chunking.

    Splits markdown text on double-newline (``\\n\\n``) boundaries.
    Paragraphs are accumulated into chunks up to ``max_chunk_size``
    characters.  Individual paragraphs that exceed the limit
    are kept intact (no mid-paragraph splitting).
    """

    __plugin_impl__ = True
    name = "paragraph"

    # All defaults match IChunker — no own config_schema needed.
    # MRO merging in _get_merged_schema() automatically inherits
    # max_chunk_size=8000, max_chunk_count=200, overlap_size=0.

    # ── IChunker interface ─────────────────────────────────────────────

    def chunk(self, text: str, config: dict | None = None) -> list[ChunkInfo]:
        """Split *text* on paragraph boundaries.

        Parameters
        ----------
        text : str
            Markdown-formatted text to split.
        config : dict, optional
            Resolved config dict.  Reads ``max_chunk_size`` and
            ``max_chunk_count``.  Default values from ``config_schema``
            are used when keys are missing.

        Returns
        -------
        list of ChunkInfo
        """
        from mnemo.core.interfaces.types import ChunkInfo

        cfg = config or {}
        max_chunk_size = int(cfg.get("max_chunk_size", 8000))
        max_chunk_count = int(cfg.get("max_chunk_count", 200))

        chunks: list[ChunkInfo] = []
        paragraphs = text.split("\n\n")
        current_text = ""
        chunk_idx = 0
        offset = 0

        for para in paragraphs:
            if len(current_text) + len(para) < max_chunk_size:
                current_text += para + "\n\n"
            else:
                if current_text.strip():
                    chunk_text = current_text.strip()
                    start = text.find(chunk_text, max(0, offset - len(chunk_text)))
                    if start < 0:
                        start = offset
                    end = start + len(chunk_text)
                    chunks.append(ChunkInfo(
                        text=chunk_text,
                        chunk_index=chunk_idx,
                        start_char=start,
                        end_char=end,
                    ))
                    offset = end
                    chunk_idx += 1
                    if max_chunk_count > 0 and chunk_idx >= max_chunk_count:
                        return chunks
                current_text = para + "\n\n"

        if current_text.strip():
            chunk_text = current_text.strip()
            start = text.find(chunk_text, max(0, offset - len(chunk_text)))
            if start < 0:
                start = offset
            end = start + len(chunk_text)
            chunks.append(ChunkInfo(
                text=chunk_text,
                chunk_index=chunk_idx,
                start_char=start,
                end_char=end,
            ))

        return chunks
