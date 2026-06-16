"""Graph expansion for search — extracted from LightRAGSearcher.

Resolves entity extraction + graph traversal independently so the
searcher doesn't act as its own DI container.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("mnemo")


class GraphExpander:
    """Graph-based query expansion for LightRAG search.

    Self-resolves dependencies from registries and provides a single
    ``expand()`` method that returns file IDs linked to query entities.

    Parameters
    ----------
    max_hops : int
        Maximum graph traversal hops from matched entities.
    """

    def __init__(self, max_hops: int = 2) -> None:
        from mnemo.core.plugin_base import PluginHub
        from mnemo.core.interfaces import IEntityExtractor, IGraphStore

        self._max_hops = max_hops
        self._graph_store: Any = None
        self._entity_extractor: Any = None

        try:
            self._graph_store = PluginHub.get(IGraphStore, "sqlite")
            self._entity_extractor = PluginHub.get(IEntityExtractor, "llm")
            self._enabled = True
        except KeyError:
            self._enabled = False
            logger.debug("GraphExpander: graph subsystem not available")

    @property
    def enabled(self) -> bool:
        """Whether graph expansion is available."""
        return self._enabled

    def expand(self, query: str) -> list[str]:
        """Expand a query into file IDs via entity-graph traversal.

        Returns an empty list if graph expansion is unavailable or
        no entities are matched.
        """
        if not self._enabled:
            return []

        try:
            entities = self._entity_extractor.extract_from_query(query)
            if not entities:
                return []
            entity_ids = self._graph_store.upsert_entities(entities)
            return self._graph_store.expand_from_entities(
                entity_ids, hops=self._max_hops,
            )
        except Exception:
            logger.warning(
                "Graph expansion failed, skipping graph channel",
                exc_info=True,
            )
            return []
