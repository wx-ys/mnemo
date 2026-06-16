"""Fixed-size character chunking — strict character-count splitting.

Each chunk is exactly ``max_chunk_size`` characters (except the last).
No attempt is made to respect paragraph or sentence boundaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mnemo.core.interfaces.chunker import IChunker
from mnemo.core.interfaces.param_spec import Param

if TYPE_CHECKING:
    from mnemo.core.interfaces.types import ChunkInfo


class FixedSizeChunker(IChunker):
    """Fixed-size character-count chunking.

    Splits text into chunks of exactly ``max_chunk_size`` characters.
    Use when you need predictable, uniform chunk sizes regardless of
    document structure.
    """

    __plugin_impl__ = True
    name = "fixed_size"

    # Only override fields where fixed_size differs from IChunker defaults.
    config_schema = {
        "max_chunk_size": Param(type="int", default=2048, desc="Characters per chunk"),
        "max_chunk_count": Param(type="int", default=0, desc="Maximum chunks per document (0 = unlimited)"),
    }

    # ── IChunker interface ─────────────────────────────────────────────

    def chunk(self, text: str, config: dict | None = None) -> list[ChunkInfo]:
        """Split *text* into fixed-size character chunks.

        Parameters
        ----------
        text : str
            Text to split.
        config : dict, optional
            Resolved config with ``max_chunk_size`` and ``max_chunk_count``.

        Returns
        -------
        list of ChunkInfo
        """
        from mnemo.core.interfaces.types import ChunkInfo

        cfg = config or {}
        chunk_size = int(cfg.get("max_chunk_size", 2048))
        max_chunk_count = int(cfg.get("max_chunk_count", 0))

        chunks: list[ChunkInfo] = []
        chunk_idx = 0

        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            if not chunk_text.strip():
                continue
            chunks.append(ChunkInfo(
                text=chunk_text,
                chunk_index=chunk_idx,
                start_char=i,
                end_char=min(i + chunk_size, len(text)),
            ))
            chunk_idx += 1
            if max_chunk_count > 0 and chunk_idx >= max_chunk_count:
                break

        return chunks
