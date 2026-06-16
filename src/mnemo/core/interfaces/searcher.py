"""Search engine interface (ISearcher).

Searchers are resolved from ``PluginHub`` and self-resolve
their dependencies.  Each searcher declares which ingestion capabilities
it requires via :meth:`required_capabilities` — the ingestion pipeline
uses this to skip unnecessary stages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from mnemo.core.interfaces.param_spec import Param
from mnemo.core.plugin_base import PluginBase, PluginHub

if TYPE_CHECKING:
    from mnemo.core.interfaces.types import GroupedResult, SearchResult


class ISearcher(PluginBase, ABC):
    """Interface for search engines.

    Subclasses must be registered via ``__plugin_impl__ = True`` marker
    and provide a unique ``name`` class attribute.  Dependencies
    (embedder, vector store, graph store, etc.) are self-resolved
    from their respective registries in ``__init__``.

    Notes
    -----
    Searchers declare :meth:`required_capabilities` so the ingestion
    pipeline can prune unnecessary stages.  For example, a keyword-only
    searcher returns ``{'markdown_content'}`` and the ingestion pipeline
    skips embedding and entity extraction.
    """

    __plugin_interface__ = True
    name: ClassVar[str] = "searcher"
    plugin_path: ClassVar[str] = "searchers"

    # ── Config schema ──────────────────────────────────────────────────

    config_schema: dict[str, Param] = {
        "default_plugin": Param(
            type="str", default="default", desc="Default search plugin to use (default=LightRAG, keyword=BM25 only, simple=grep)"),
        "default_mode": Param(
            type="str", default="hybrid", desc="Default search mode: 'hybrid', 'vector', or 'keyword'"),
        "default_limit": Param(
            type="int", default=10, desc="Default max search results"),
        "max_limit": Param(
            type="int", default=100, desc="Hard limit on search results"),
        "rrf_k": Param(
            type="int", default=60, desc="RRF rank fusion constant"),
    }

    # ── Lifecycle (optional) ───────────────────────────────────────────

    def init(self, data_dir: Path) -> None:
        """Bind to a data directory.  Called by KnowledgeBase after
        the searcher is resolved from the registry.

        The default implementation is a no-op.  Searchers that need
        a data directory (for keyword index, etc.) should override.

        Parameters
        ----------
        data_dir : Path
            Knowledge base root directory.
        """
        pass

    # ── Abstract interface ─────────────────────────────────────────────

    @property
    def required_capabilities(self) -> set[str]:
        """Capability tags this searcher needs from the ingestion pipeline.

        Tags: ``'embeddings'``, ``'graph_entities'``, ``'markdown_content'``.

        Override in subclasses.  The default (empty set) means no
        ingestion stages are skipped.
        """
        return set()

    @abstractmethod
    def search(
        self,
        query: str,
        mode: str = "hybrid",
        candidate_ids: list[str] | None = None,
        limit: int = 10,
        file_types: list[str] | None = None,
        with_metadata: bool = True,
        on_progress: Callable[[str, str], None] | None = None,
        diagnose: bool = False,
        diagnostic_ctx: Any = None,
    ) -> list[SearchResult]:
        """Execute a search query.

        Parameters
        ----------
        query : str
            Search query string.
        mode : str, optional
            Search mode: 'hybrid', 'vector', or 'keyword'. Default is 'hybrid'.
        candidate_ids : list of str, optional
            Pre-filter: only search among these file IDs (from key expansion).
        limit : int, optional
            Maximum number of results. Default is 10.
        file_types : list of str, optional
            Restrict to specific file extensions.
        with_metadata : bool, optional
            Whether to include the metadata vector store. Default is True.
        on_progress : callable, optional
            Callback ``(stage, status)`` for progress reporting.
            Stages: ``"vector"``, ``"keyword"``, ``"graph"``, ``"fuse"``.
            Statuses: ``"in_progress"``, ``"done"``, ``"skipped"``.
        diagnose : bool, optional
            If True, emit detailed diagnostic data at each pipeline stage.
        diagnostic_ctx : DiagnosticContext, optional
            Carries trace file path and preview configs when diagnosing.

        Returns
        -------
        list of SearchResult
            Results sorted by relevance score descending.
        """
        ...

    @abstractmethod
    def dedup_by_file(self, results: list[SearchResult]) -> list[GroupedResult]:
        """Merge multi-chunk results by file.

        Parameters
        ----------
        results : list of SearchResult
            Raw results (possibly multiple per file).

        Returns
        -------
        list of GroupedResult
            One result per file, with merged snippets.
        """
        ...
