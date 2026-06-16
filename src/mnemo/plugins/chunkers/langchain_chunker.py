"""LangChain-powered chunker — file-type-aware text splitting.

Wraps ``langchain-text-splitters`` to provide 22 language-specific
code splitters, Markdown header splitting, HTML splitting, and
recursive character splitting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mnemo.core.interfaces.chunker import IChunker
from mnemo.core.interfaces.param_spec import Param

if TYPE_CHECKING:
    from mnemo.core.interfaces.types import ChunkInfo


# Map file extensions to langchain Language enum values
_EXT_TO_LANGUAGE: dict[str, str] = {
    "py": "PYTHON", "js": "JS", "ts": "TS", "jsx": "JS",
    "tsx": "TS", "java": "JAVA", "c": "C", "cpp": "CPP", "h": "C",
    "hpp": "CPP", "cs": "CSHARP", "go": "GO", "rs": "RUST",
    "rb": "RUBY", "php": "PHP", "swift": "SWIFT", "kt": "KOTLIN",
    "scala": "SCALA", "lua": "LUA", "pl": "PERL", "sh": "BASH",
    "bash": "BASH", "sql": "SQL", "r": "R", "jl": "JULIA",
    "html": "HTML", "htm": "HTML", "md": "MARKDOWN", "rst": "RST",
    "tex": "LATEX", "proto": "PROTO",
}


class LangChainChunker(IChunker):
    """LangChain-powered chunker with file-type-aware splitter selection.

    Automatically selects the best splitter for each file type:
    - Code files → language-specific recursive splitter
    - Markdown → Markdown header + recursive splitter
    - HTML → HTML header + recursive splitter
    - Default → recursive character splitter

    This is the **default** chunker (replaces ``ParagraphChunker``).
    """

    __plugin_impl__ = True
    name = "default"

    config_schema = {
        "chunk_size": Param(
            type="int", default=512,
            desc="Target chunk size in characters"
        ),
        "chunk_overlap": Param(
            type="int", default=64,
            desc="Character overlap between adjacent chunks"
        ),
        "max_chunk_count": Param(
            type="int", default=0,
            desc="Maximum chunks per document (0 = unlimited)"
        ),
        "md_headers": Param(
            type="str", default="#,##,###",
            desc="Comma-separated Markdown headers to split on"
        ),
    }

    # ── IChunker ───────────────────────────────────────────────────────

    def chunk(
        self, text: str, config: dict | None = None,
    ) -> list[ChunkInfo]:
        """Split *text* using the appropriate langchain splitter.

        Parameters
        ----------
        text : str
            Text to split.
        config : dict, optional
            Resolved config dict.  Supports ``file_type`` key for
            language-specific splitting.

        Returns
        -------
        list of ChunkInfo
        """
        from mnemo.core.interfaces.types import ChunkInfo

        cfg = config or {}
        chunk_size = int(cfg.get("chunk_size", 512))
        chunk_overlap = int(cfg.get("chunk_overlap", 64))
        max_chunk_count = int(cfg.get("max_chunk_count", 0))
        file_type = cfg.get("file_type", "")

        docs = self._split(text, file_type, chunk_size, chunk_overlap, cfg)

        chunks: list[ChunkInfo] = []
        offset = 0
        for i, doc in enumerate(docs):
            if max_chunk_count > 0 and i >= max_chunk_count:
                break
            content = doc.page_content
            start = text.find(content, offset) if offset < len(text) else offset
            end = start + len(content) if start >= 0 else offset + len(content)
            offset = end

            meta = dict(doc.metadata) if doc.metadata else {}
            chunks.append(ChunkInfo(
                text=content,
                chunk_index=i,
                start_char=max(0, start),
                end_char=end,
                metadata=meta,
            ))

        return chunks

    def _split(
        self, text: str, file_type: str,
        chunk_size: int, chunk_overlap: int, cfg: dict,
    ) -> list:
        """Select and apply the appropriate langchain splitter."""
        ext = file_type.lower().lstrip(".")

        # 1. Markdown: header-aware splitting
        if ext in ("md", "markdown", "rst"):
            return self._split_markdown(text, chunk_size, chunk_overlap, cfg)

        # 2. HTML: header-aware splitting
        if ext in ("html", "htm"):
            return self._split_html(text, chunk_size, chunk_overlap, cfg)

        # 3. Code: language-specific recursive splitter
        lang = _EXT_TO_LANGUAGE.get(ext, "")
        if lang:
            return self._split_code(text, lang, chunk_size, chunk_overlap)

        # 4. Default: recursive character splitter
        return self._split_recursive(text, chunk_size, chunk_overlap)

    # ── Splitter helpers ───────────────────────────────────────────────

    @staticmethod
    def _split_recursive(text: str, chunk_size: int, overlap: int) -> list:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
        )
        return splitter.create_documents([text])

    @staticmethod
    def _split_code(text: str, language: str, chunk_size: int, overlap: int) -> list:
        from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
        lang_enum = getattr(Language, language, None)
        if lang_enum is None:
            return LangChainChunker._split_recursive(text, chunk_size, overlap)
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_enum,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        return splitter.create_documents([text])

    @staticmethod
    def _split_markdown(text: str, chunk_size: int, overlap: int, cfg: dict) -> list:
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )
        headers_str = cfg.get("md_headers", "#,##,###")
        headers = [
            (h.strip(), h.strip().lstrip("#").strip() or f"H{len(h.strip())}")
            for h in headers_str.split(",") if h.strip()
        ]
        # First split by headers to get sections with metadata
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers,
            strip_headers=False,
        )
        try:
            header_docs = md_splitter.split_text(text)
        except Exception:
            # Fallback to recursive if header parsing fails
            return LangChainChunker._split_recursive(text, chunk_size, overlap)

        # Then recursively split large sections
        rc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        return rc_splitter.split_documents(header_docs)

    @staticmethod
    def _split_html(text: str, chunk_size: int, overlap: int, cfg: dict) -> list:
        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,
        )
        # HTML header splitting requires lxml — fall back to recursive
        try:
            from langchain_text_splitters import HTMLHeaderTextSplitter
            headers_str = cfg.get("md_headers", "h1,h2,h3")
            headers = [
                (h.strip(), h.strip())
                for h in headers_str.split(",") if h.strip()
            ]
            html_splitter = HTMLHeaderTextSplitter(
                headers_to_split_on=headers,
            )
            header_docs = html_splitter.split_text(text)
            rc_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
            )
            return rc_splitter.split_documents(header_docs)
        except Exception:
            return LangChainChunker._split_recursive(text, chunk_size, overlap)
