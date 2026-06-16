"""Entity extractor interface (IEntityExtractor).

Extracts entities and relations from text for LightRAG graph construction.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from mnemo.core.interfaces.param_spec import Param
from mnemo.core.plugin_base import PluginBase, PluginHub


class IEntityExtractor(PluginBase, ABC):
    """Interface for entity and relation extraction from text.

    Notes
    -----
    Subclasses must be registered via ``__plugin_impl__ = True`` marker.
    Default implementation: LLM-based extractor (reuses wiki template LLM config).
    """

    __plugin_interface__ = True
    name: ClassVar[str] = "entity_extractor"
    plugin_path: ClassVar[str] = "entity_extractors"

    # ── Config schema ──────────────────────────────────────────────────

    config_schema: dict[str, Param] = {
        "enabled_on_add": Param(type="bool", default=True, desc="Run entity extraction automatically on file add"),
        "max_entities_per_file": Param(type="int", default=20, desc="Max entities to extract per file"),
        "temperature": Param(type="float", default=0.1, desc="LLM temperature for entity extraction"),
    }

    # ── Abstract interface ─────────────────────────────────────────────

    @abstractmethod
    def extract(self, text: str) -> tuple[list[dict], list[dict]]:
        """Extract entities and relations from a document.

        Parameters
        ----------
        text : str
            Document text (markdown).

        Returns
        -------
        tuple[list[dict], list[dict]]
            (entities, relations).
            Each entity: {'name': str, 'type': str, 'description': str}.
            Each relation: {'source': str, 'target': str, 'relation': str,
            'description': str}.
        """
        ...

    @abstractmethod
    def extract_from_query(self, query: str) -> list[dict]:
        """Extract entities from a search query (lightweight).

        Parameters
        ----------
        query : str
            Search query.

        Returns
        -------
        list of dict
            Each entity: {'name': str, 'type': str}.
        """
        ...
