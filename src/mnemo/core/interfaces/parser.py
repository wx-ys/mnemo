"""File parser interface (IParser)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from mnemo.core.interfaces.param_spec import Param
from mnemo.core.interfaces.types import ChunkInfo
from mnemo.core.plugin_base import PluginBase, PluginHub


class IParser(PluginBase, ABC):
    """Interface for converting raw files to Markdown.

    A parser can target one or more specific file types, or serve
    as the default for an entire category.

    Resolution priority: type-level -> category-level -> fallback (text).

    Notes
    -----
    Subclasses must be registered via ``__plugin_impl__ = True`` marker.
    """

    __plugin_interface__ = True
    name: ClassVar[str] = "parser"
    plugin_path: ClassVar[str] = "parsers"

    # ── Config schema ──────────────────────────────────────────────────

    config_schema: dict[str, Param] = {
        "default_auto_md": Param(
            type="bool", default=True,
            desc="Global default: auto-generate markdown on add",
        ),
    }

    # ── Abstract interface ─────────────────────────────────────────────

    @property
    @abstractmethod
    def category(self) -> str:
        """Parent category used for directory layout and fallback chain.

        Examples: 'documents', 'data', 'code'.
        """
        ...

    @property
    @abstractmethod
    def supported_types(self) -> list[str]:
        """File extensions this parser handles, e.g. ['pdf', 'PDF']."""
        ...

    @property
    @abstractmethod
    def default_enable_md(self) -> bool:
        """Whether to generate markdown by default for this type."""
        ...

    @property
    @abstractmethod
    def default_enable_wiki(self) -> bool:
        """Whether to generate wiki by default for this type."""
        ...

    @property
    @abstractmethod
    def default_enable_embed(self) -> bool:
        """Whether to generate embeddings by default for this type."""
        ...

    @abstractmethod
    def parse(self, file_path: Path) -> str:
        """Convert a raw file into Markdown text.

        Parameters
        ----------
        file_path : Path
            Absolute path to the raw file.

        Returns
        -------
        str
            Markdown representation of the file content.

        Raises
        ------
        ParseFailedError
            If the file cannot be parsed (corrupted, unsupported variant, etc.).
        """
        ...

    @abstractmethod
    def chunk(self, md_text: str, max_chunk_size: int = 8000) -> list[ChunkInfo]:
        """Split Markdown text into embedding-ready chunks.

        .. deprecated::
            Use :class:`IChunker` plugins via ``PluginHub``
            instead.  This method is kept for backward compatibility with
            existing parsers that override it.  The default implementation
            in :class:`BaseParser` now delegates to
            ``PluginHub.get(IChunker, "paragraph").chunk()``.

        Parameters
        ----------
        md_text : str
            The full Markdown text to split.
        max_chunk_size : int, optional
            Maximum characters per chunk. Default is 8000.

        Returns
        -------
        list of ChunkInfo
            Ordered list of chunks.
        """
        ...

    @classmethod
    def resolve(cls, file_extension: str, category: str) -> "IParser":
        """Find the best parser for a file extension and category.

        Priority: type-level -> category-level -> 'text' fallback.
        """
        inst = PluginHub.get_instance_for_type(cls, file_extension)
        if inst is not None:
            return inst
        inst = PluginHub.get_instance_for_category(cls, category)
        if inst is not None:
            return inst
        if PluginHub.has(cls, "text"):
            return PluginHub.get(cls, "text")
        raise KeyError(
            f"No parser found for '{file_extension}' (category: {category}), "
            f"and fallback 'text' parser is not registered."
        )
