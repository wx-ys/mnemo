"""Graph store interface (IGraphStore).

Stores entities, relations, and file-entity links for LightRAG graph enhancement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from mnemo.core.interfaces.param_spec import Param
from mnemo.core.plugin_base import PluginBase, PluginHub


class IGraphStore(PluginBase, ABC):
    """Interface for entity/relation graph storage.

    Notes
    -----
    Subclasses must be registered via ``__plugin_impl__ = True`` marker.
    Default implementation: ``SQLiteGraphStore`` (shares index.db).
    """

    __plugin_interface__ = True
    name: ClassVar[str] = "graph_store"
    plugin_path: ClassVar[str] = "graph_stores"

    # ── Config schema ──────────────────────────────────────────────────

    config_schema: dict[str, Param] = {
        "enabled": Param(type="bool", default=True, desc="Enable graph-based search enhancement"),
        "max_hops": Param(type="int", default=2, desc="Max graph traversal hops from search entities"),
        "max_expanded_docs": Param(type="int", default=20, desc="Max documents expanded via graph traversal"),
        "entity_weight": Param(type="float", default=0.3, desc="Weight of graph entity score in RRF fusion"),
    }

    # ── Lifecycle ──────────────────────────────────────────────────────

    def init(self, data_dir: Path) -> None:
        """Bind to a data directory. Called by KnowledgeBase after creation."""
        pass

    # ── Abstract interface ─────────────────────────────────────────────

    @abstractmethod
    def init_tables(self) -> None:
        """Create graph tables if they do not exist. Idempotent."""
        ...

    @abstractmethod
    def upsert_entities(self, entities: list[dict]) -> list[int]:
        """Insert or update entities.

        Parameters
        ----------
        entities : list of dict
            Each dict has: 'name', 'type', 'description'.

        Returns
        -------
        list of int
            Entity IDs (new or existing).
        """
        ...

    @abstractmethod
    def add_relations(self, relations: list[dict]) -> None:
        """Add relations between entities.

        Parameters
        ----------
        relations : list of dict
            Each dict has: 'source_id', 'target_id', 'relation', 'description'.
        """
        ...

    @abstractmethod
    def link_file_entities(
        self, file_id: str, entity_ids: list[int], scores: list[float]
    ) -> None:
        """Link entities to a file with relevance scores.

        Parameters
        ----------
        file_id : str
        entity_ids : list of int
        scores : list of float
            Relevance score for each entity (0.0-1.0).
        """
        ...

    @abstractmethod
    def search_entities(self, name_pattern: str, limit: int = 10) -> list[dict]:
        """Find entities by name pattern (LIKE match).

        Parameters
        ----------
        name_pattern : str
            SQL LIKE pattern, e.g. '%transformer%'.
        limit : int

        Returns
        -------
        list of dict
        """
        ...

    @abstractmethod
    def expand_from_entities(
        self, entity_ids: list[int], hops: int = 2
    ) -> list[str]:
        """Graph traversal: find file IDs related to given entities.

        Parameters
        ----------
        entity_ids : list of int
            Starting entity IDs.
        hops : int, optional
            Maximum traversal depth. Default is 2.

        Returns
        -------
        list of str
            Related file IDs, ordered by relevance (closer = higher).
        """
        ...

    @abstractmethod
    def get_file_entities(self, file_id: str) -> list[dict]:
        """Get all entities linked to a file.

        Parameters
        ----------
        file_id : str

        Returns
        -------
        list of dict
        """
        ...

    @abstractmethod
    def delete_file_entities(self, file_id: str) -> int:
        """Remove all entity links for a file, then clean up orphaned entities
        and relations.

        Parameters
        ----------
        file_id : str

        Returns
        -------
        int
            Number of entity links removed.
        """
        ...
