"""Token-based chunking — splits text by token count using tiktoken.

Useful for LLM-oriented workflows where API token limits matter more
than raw character counts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mnemo.core.interfaces.chunker import IChunker
from mnemo.core.interfaces.param_spec import Param

if TYPE_CHECKING:
    from mnemo.core.interfaces.types import ChunkInfo


class TokenChunker(IChunker):
    """Token-count-based chunking via tiktoken.

    Uses a tiktoken encoding to count tokens and split text
    when the token budget is exceeded.  Prefers paragraph
    boundaries as split points.
    """

    __plugin_impl__ = True
    name = "token"

    # Only declare fields added or overridden vs IChunker.
    # tokenizer, max_tokens, overlap_tokens are new;
    # max_chunk_size is overridden (8000 → 512 tokens).
    config_schema = {
        "tokenizer": Param(
            type="str", default="cl100k_base",
            desc="tiktoken encoding name (cl100k_base, o200k_base, etc.)"
        ),
        "max_tokens": Param(
            type="int", default=512,
            desc="Maximum tokens per chunk (overrides max_chunk_size)"
        ),
        "overlap_tokens": Param(
            type="int", default=50,
            desc="Token overlap between adjacent chunks"
        ),
    }

    # ── IChunker interface ─────────────────────────────────────────────

    def chunk(self, text: str, config: dict | None = None) -> list[ChunkInfo]:
        """Split *text* by token count using tiktoken.

        Parameters
        ----------
        text : str
            Text to split.
        config : dict, optional
            Resolved config with ``tokenizer``, ``max_tokens``,
            ``overlap_tokens``, ``max_chunk_count``.

        Returns
        -------
        list of ChunkInfo
        """
        from mnemo.core.interfaces.types import ChunkInfo

        cfg = config or {}
        tokenizer_name = cfg.get("tokenizer", "cl100k_base")
        max_tokens = int(cfg.get("max_tokens", 512))
        overlap = int(cfg.get("overlap_tokens", 50))
        max_chunk_count = int(cfg.get("max_chunk_count", 200))

        try:
            import tiktoken
            enc = tiktoken.get_encoding(tokenizer_name)
        except Exception:
            # Fall back to paragraph chunking if tiktoken is unavailable
            from mnemo.plugins.chunkers.paragraph_chunker import ParagraphChunker
            return ParagraphChunker().chunk(text, {
                "max_chunk_size": max_tokens * 4,  # rough char estimate
                "max_chunk_count": max_chunk_count,
            })

        # Split into paragraphs first, then group by token count
        paragraphs = text.split("\n\n")
        chunks: list[ChunkInfo] = []
        current_paras: list[str] = []
        current_tokens = 0
        chunk_idx = 0

        for para in paragraphs:
            para_tokens = len(enc.encode(para))
            if current_tokens + para_tokens <= max_tokens:
                current_paras.append(para)
                current_tokens += para_tokens
            else:
                # Flush current chunk
                if current_paras:
                    chunk_text = "\n\n".join(current_paras)
                    chunks.append(ChunkInfo(start_char=0, end_char=0,
                        text=chunk_text,
                        chunk_index=chunk_idx,
                    ))
                    chunk_idx += 1
                    if max_chunk_count > 0 and chunk_idx >= max_chunk_count:
                        return chunks
                # Start new chunk with overlap from previous
                current_paras = [para]
                current_tokens = para_tokens
                # Add overlap: take last `overlap` tokens' worth from prev chunk
                if overlap > 0 and chunks:
                    # Rough overlap: prepend last paragraph from previous chunk
                    prev = chunks[-1].text.split("\n\n")
                    if prev:
                        overlap_text = prev[-1]
                        if len(enc.encode(overlap_text)) <= overlap:
                            current_paras.insert(0, overlap_text)
                            current_tokens += len(enc.encode(overlap_text))

        # Flush final chunk
        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunks.append(ChunkInfo(start_char=0, end_char=0,
                text=chunk_text,
                chunk_index=chunk_idx,
            ))

        return chunks
