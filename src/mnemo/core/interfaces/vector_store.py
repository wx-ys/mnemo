"""Vector store interface (IVectorStore).

Decoupled from IEmbedder and ISearcher — IEmbedder generates vectors,
IVectorStore stores and retrieves them, ISearcher fuses results.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from mnemo.core.interfaces.param_spec import Param
from mnemo.core.plugin_base import PluginBase, PluginHub


class IVectorStore(PluginBase, ABC):
    """Interface for vector storage and ANN retrieval.

    Notes
    -----
    Subclasses must be registered via ``__plugin_impl__ = True`` marker.
    Default implementation: ``LanceDBStore``.
    """

    __plugin_interface__ = True
    name: ClassVar[str] = "vector_store"
    plugin_path: ClassVar[str] = "vector_stores"

    # ── Config schema ──────────────────────────────────────────────────

    config_schema: dict[str, Param] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────

    def init(self, data_dir: Path) -> None:
        """Bind to a data directory. Called by KnowledgeBase after creation.

        The default implementation is a no-op.  Concrete stores should
        override to set up the database connection.
        """
        pass

    # ── Abstract interface ─────────────────────────────────────────────

    @abstractmethod
    def init_tables(self) -> None:
        """Create tables if they do not exist. Idempotent."""
        ...

    @abstractmethod
    def add_vectors(
        self,
        table: str,
        ids: list[str],
        vectors: list[list[float]],
        metadata: dict[str, list] | None = None,
    ) -> None:
        """Batch-insert vectors into a table.

        Parameters
        ----------
        table : str
            Table name, e.g. 'raw_md', 'raw_wiki', 'metadata'.
        ids : list of str
            File IDs (one per vector).
        vectors : list of list of float
            Embedding vectors.
        metadata : dict of str → list, optional
            Additional columns, e.g. {'file_type': ['pdf', 'txt'],
            'chunk_index': [0, 0], 'model': ['dashscope', 'dashscope']}.
        """
        ...

    @abstractmethod
    def search(
        self,
        table: str,
        query_vector: list[float],
        candidate_ids: list[str] | None = None,
        limit: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[dict]:
        """ANN search in a table.

        Parameters
        ----------
        table : str
            Table name.
        query_vector : list of float
            Query embedding.
        candidate_ids : list of str, optional
            Pre-filter: only search among these IDs.
        limit : int, optional
            Maximum results. Default is 10.
        filters : dict, optional
            Additional column filters, e.g. {'model': 'dashscope'}.

        Returns
        -------
        list of dict
            Each dict has: 'id', 'score', plus metadata columns.
        """
        ...

    @abstractmethod
    def delete_vectors(self, table: str, ids: list[str]) -> None:
        """Delete vectors by file ID.

        Parameters
        ----------
        table : str
            Table name.
        ids : list of str
            File IDs to remove.
        """
        ...

    @abstractmethod
    def table_info(self, table: str) -> dict:
        """Get table metadata.

        Parameters
        ----------
        table : str
            Table name.

        Returns
        -------
        dict
            Keys: 'name', 'num_rows', 'dimension', 'model'.
        """
        ...
