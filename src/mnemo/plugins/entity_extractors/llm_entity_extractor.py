"""LLM-based entity extractor for LightRAG.

Uses the same LLM configuration as wiki generation to extract
entities and relations from document text.
"""

from __future__ import annotations

from mnemo.core.interfaces import IEntityExtractor


class LLMEntityExtractor(IEntityExtractor):
    """Extract entities and relations from text using an LLM.

    Falls back to keyword-based extraction when LLM is unavailable.

    Parameters
    ----------
    llm_config : dict or None
        LLM configuration (model, api_key, etc.).
    """

    __plugin_impl__ = True
    name = "llm"

    def __init__(self, llm_config: dict | None = None):
        self._config = llm_config or {}

    # -- IEntityExtractor -----------------------------------------------------

    def extract(self, text: str) -> tuple[list[dict], list[dict]]:
        """Extract entities and relations from document text.

        Uses LLM if configured, otherwise falls back to keyword extraction.

        Parameters
        ----------
        text : str

        Returns
        -------
        tuple[list[dict], list[dict]]
            (entities, relations).
        """
        if self._has_llm():
            return self._extract_with_llm(text)
        return self._extract_keywords(text)

    def extract_from_query(self, query: str) -> list[dict]:
        """Extract entities from a short search query.

        Uses simple keyword matching — no LLM call needed for queries.

        Parameters
        ----------
        query : str

        Returns
        -------
        list of dict
            Each: {'name': str, 'type': str}.
        """
        # Simple heuristic: split by spaces and punctuation,
        # treat each substantial word as a potential entity
        import re
        words = re.findall(r'[\w一-鿿]+', query.lower())
        entities = []
        for w in words:
            if len(w) >= 2:
                entities.append({"name": w, "type": "concept"})
        return entities

    # -- Internal -------------------------------------------------------------

    def _has_llm(self) -> bool:
        key = self._config.get("api_key", "")
        return bool(key) and len(key) > 10 and not key.startswith("${")

    def _extract_with_llm(self, text: str) -> tuple[list[dict], list[dict]]:
        """Call LLM via pydantic-ai Agent directly.

        Uses structured output with ``EntityOutput`` model first,
        falling back to plain text + JSON parsing.

        Parameters
        ----------
        text : str
            Document text (truncated to first ~3000 chars).

        Returns
        -------
        tuple[list[dict], list[dict]]
            (entities, relations).
        """
        import json

        from pydantic import BaseModel, Field

        from mnemo.core.agent_manager import AgentManager

        truncated = text[:3000] if len(text) > 3000 else text

        system_prompt = (
            "You are an entity extraction assistant. "
            "Extract key entities and their relations from the text. "
            "Entity types: concept, person, model, method, dataset, tool, language, field."
        )

        agent_name = self._config.get("agent_name", "default")
        am = AgentManager.get_instance()
        if not am._initialized:
            return self._extract_keywords(text)

        # Try structured output
        try:
            class EntityOutput(BaseModel):
                entities: list[dict] = Field(default_factory=list)
                relations: list[dict] = Field(default_factory=list)

            agent = am.get_agent(agent_name, output_type=EntityOutput)
            result = agent.run_sync(truncated, instructions=system_prompt)
            if isinstance(result.output, BaseModel):
                return (result.output.entities, result.output.relations)
        except Exception:
            pass  # fall through to JSON parsing

        # Fallback: plain text + JSON parsing
        try:
            agent = am.get_agent(agent_name, output_type=str)
            result = agent.run_sync(truncated, instructions=system_prompt)
            response = result.output.strip() if result.output else ""
        except Exception:
            return self._extract_keywords(text)

        if response and not response.startswith("[LLM"):
            try:
                cleaned = response.strip().removeprefix("```json").removesuffix("```")
                data = json.loads(cleaned)
                entities = data.get("entities", [])
                relations = data.get("relations", [])
                return (entities, relations)
            except (json.JSONDecodeError, KeyError):
                pass

        return self._extract_keywords(text)

    @staticmethod
    def _extract_keywords(text: str) -> tuple[list[dict], list[dict]]:
        """Fallback: keyword-based extraction.

        Parameters
        ----------
        text : str

        Returns
        -------
        tuple[list[dict], list[dict]]
        """
        import re
        # Extract capitalized words and Chinese noun phrases as entities
        words = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[A-Z]{2,}', text)
        chinese = re.findall(r'[一-鿿]{2,8}', text)

        entities: list[dict] = []
        seen: set[str] = set()

        for w in words[:10]:
            if w.lower() not in seen:
                seen.add(w.lower())
                entities.append({"name": w, "type": "concept",
                                 "description": "Extracted from text"})

        for w in chinese[:10]:
            if w not in seen:
                seen.add(w)
                entities.append({"name": w, "type": "concept",
                                 "description": ""})

        return (entities, [])
