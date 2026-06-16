"""Small-to-Big retrieval chunking.

Produces two levels of chunks:
- **Parent chunks** (large, fewer): stored for context expansion at retrieval
- **Child chunks** (small, more): embedded and indexed for precise ANN search

At search time, the small child chunks enable precise retrieval and the
parent chunks provide the broader context for LLM generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mnemo.core.interfaces.chunker import IChunker
from mnemo.core.interfaces.param_spec import Param

if TYPE_CHECKING:
    from mnemo.core.interfaces.types import ChunkInfo


class SmallToBigChunker(IChunker):
    """Two-level chunking for Small-to-Big retrieval.

    Returns child chunks (for embedding/search).  Parent chunks
    are stored separately in ``chunk()``'s return metadata — the
    caller (``kb.py``) stores them in LanceDB's parents table.
    """

    __plugin_impl__ = True
    name = "small_to_big"

    config_schema = {
        "parent_chunk_size": Param(
            type="int", default=2000, desc="Target characters per parent chunk"),
        "child_chunk_size": Param(
            type="int", default=500, desc="Target characters per child chunk"),
        "parent_overlap": Param(
            type="int", default=200, desc="Overlap between parent chunks"),
        "child_overlap": Param(
            type="int", default=100, desc="Overlap between child chunks"),
    }

    # ── IChunker ───────────────────────────────────────────────────────

    def chunk(
        self, text: str, config: dict | None = None,
    ) -> list[ChunkInfo]:
        """Split *text* into child chunks with parent references.

        The returned ChunkInfo list contains **child chunks only**.
        Each child's ``metadata["parent_id"]`` references its parent.
        The caller is responsible for storing parent chunks.

        Parameters
        ----------
        text : str
            Text to split.
        config : dict, optional
            Resolved config.

        Returns
        -------
        list of ChunkInfo
            Child chunks (for embedding).
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        from mnemo.core.interfaces.types import ChunkInfo

        cfg = config or {}
        parent_size = int(cfg.get("parent_chunk_size", 2000))
        child_size = int(cfg.get("child_chunk_size", 500))
        parent_overlap = int(cfg.get("parent_overlap", 200))
        child_overlap = int(cfg.get("child_overlap", 100))
        file_id = cfg.get("file_id", "unknown")

        # Create parent chunks
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
        )
        parent_docs = parent_splitter.create_documents([text])

        # Create child chunks from each parent
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
        )

        chunks: list[ChunkInfo] = []
        child_idx = 0

        for pi, pdoc in enumerate(parent_docs):
            parent_id = f"{file_id}::parent_{pi}"
            parent_text = pdoc.page_content

            # Store parent info as metadata on this splitter instance
            # (caller reads these from chunk metadata to store parents)
            child_docs = child_splitter.create_documents([parent_text])
            for cdoc in child_docs:
                start = text.find(cdoc.page_content)
                if start < 0:
                    start = 0
                end = start + len(cdoc.page_content)

                chunks.append(ChunkInfo(
                    text=cdoc.page_content,
                    chunk_index=child_idx,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "parent_id": parent_id,
                        "parent_text": parent_text[:500],  # truncated for metadata
                        "chunk_level": "child",
                    },
                ))
                child_idx += 1

        return chunks
