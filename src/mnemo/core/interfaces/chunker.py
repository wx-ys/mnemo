"""Text chunking interface (IChunker).

Chunking is an independent concern — decoupled from parsing so that
different chunking strategies (paragraph, token, semantic, etc.) can
be selected per file category or globally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from mnemo.core.interfaces.param_spec import Param
from mnemo.core.plugin_base import PluginBase, PluginHub

if TYPE_CHECKING:
    from mnemo.core.interfaces.types import ChunkInfo


class IChunker(PluginBase, ABC):
    """Interface for text chunking strategies.

    Each chunker is a plugin registered via ``__plugin_impl__ = True``.
    The chunker to use is resolved per file category from
    ``file_categories.toml``, falling back to the global default
    in ``config.toml``.

    Notes
    -----
    Subclasses must provide a unique ``name`` class attribute.
    """

    __plugin_interface__ = True
    name: ClassVar[str] = "chunker"
    plugin_path: ClassVar[str] = "chunkers"

    # ── Config schema ──────────────────────────────────────────────────

    config_schema: dict[str, Param] = {
        "default_plugin": Param(type="str", default="default", desc="Default chunker plugin (default = langchain, paragraph, token, fixed_size, semantic)"),
        "max_chunk_size": Param(type="int", default=8000, desc="Maximum characters per chunk"),
        "max_chunk_count": Param(type="int", default=200, desc="Maximum chunks per document (0 = unlimited)"),
        "overlap_size": Param(type="int", default=0, desc="Character overlap between adjacent chunks"),
    }

    # ── Abstract interface ─────────────────────────────────────────────

    @abstractmethod
    def chunk(self, text: str, config: dict | None = None) -> list[ChunkInfo]:
        """Split *text* into a list of :class:`ChunkInfo`.

        Parameters
        ----------
        text : str
            The text to split (typically markdown content).
        config : dict, optional
            Resolved configuration dict.  If omitted, the chunker uses
            its own ``config_schema`` defaults.

        Returns
        -------
        list of ChunkInfo
            Ordered chunks with index and metadata.
        """
        ...
