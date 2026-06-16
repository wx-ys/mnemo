"""Agent manager — pure pydantic-ai Agent factory.

Reads ``[agent.xxx]`` config sections and creates :class:`pydantic_ai.Agent`
instances on demand.  Callers use the Agent directly — this manager only
handles config-driven instantiation, lifecycle, and thread-safe caching.

Usage::

    am = AgentManager.get_instance()
    am.init(config)  # once, from KB.__init__

    agent = am.get_agent("default")
    result = agent.run_sync("What is AI?")

    # Structured output:
    agent = am.get_agent("default", output_type=WikiOutput)
    wiki: WikiOutput = agent.run_sync("Summarize this paper...").output
"""

from __future__ import annotations

import os
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from mnemo.core.interfaces.param_spec import Param

# ---------------------------------------------------------------------------
# Config schema for each [agent.<name>] section
# ---------------------------------------------------------------------------

AGENT_CONFIG_SCHEMA: dict[str, Param] = {
    "model": Param(
        type="str", default="deepseek-v4-flash",
        desc="LLM model name",
    ),
    "base_url": Param(
        type="str", default="https://api.deepseek.com/v1",
        desc="API base URL",
    ),
    "api_key": Param(
        type="str", env_var="LLM_API_KEY",
        desc="API key (reads from LLM_API_KEY env var)",
    ),
    "temperature": Param(
        type="float", default=0.3,
        desc="Generation temperature (0.0–2.0)",
    ),
    "max_tokens": Param(
        type="int", default=2048,
        desc="Max output tokens",
    ),
    "timeout": Param(
        type="int", default=60,
        desc="Request timeout in seconds",
    ),
    "max_retries": Param(
        type="int", default=3,
        desc="Max retry attempts on transient errors",
    ),
    "system_prompt": Param(
        type="str", default="",
        desc="Optional default system prompt (callers override this)",
    ),
}


# ---------------------------------------------------------------------------
# AgentManager singleton
# ---------------------------------------------------------------------------

class AgentManager:
    """Singleton manager for :class:`pydantic_ai.Agent` instances.

    Each named agent config (``[agent.default]``, ``[agent.code_review]``,
    etc.) produces a lazily-created Agent.  The manager is a **pure factory**
    — it does not wrap ``run_sync`` or add error handling.  Callers use
    the Agent directly.

    Thread-safe: Agent creation is cached per (name, output_type) key.
    """

    _instance: AgentManager | None = None

    def __new__(cls) -> AgentManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        # Cache: (agent_name, output_type.__name__) → Agent
        self._agents: dict[tuple[str, str], Agent[Any]] = {}
        self._configs: dict[str, dict[str, Any]] = {}
        self._default_name: str = "default"
        self._initialized: bool = False

    # -- Public API -----------------------------------------------------------

    def init(self, config: dict[str, Any]) -> None:
        """Parse ``[agent.*]`` sections from the raw config dict.

        Must be called once after config is loaded (by ``KB.__init__``).
        """
        if self._initialized:
            return

        agent_section = config.get("agent", {})
        if not isinstance(agent_section, dict) or not agent_section:
            agent_section = {"default": {}}

        for name, raw_cfg in agent_section.items():
            if not isinstance(raw_cfg, dict):
                raw_cfg = {}
            self._configs[name] = self._resolve_config(raw_cfg)

        global_cfg = config.get("global", {})
        if isinstance(global_cfg, dict):
            self._default_name = global_cfg.get("default_agent", "default")

        if self._default_name not in self._configs:
            self._configs[self._default_name] = self._resolve_config({})

        self._initialized = True

    @property
    def default_agent_name(self) -> str:
        """The default agent name (from ``[global].default_agent``)."""
        return self._default_name

    def list_agent_names(self) -> list[str]:
        """Return all configured agent names."""
        return list(self._configs.keys())

    def get_agent(
        self,
        agent_name: str | None = None,
        *,
        output_type: type[Any] = str,
        config_overrides: dict[str, Any] | None = None,
    ) -> Agent[Any]:
        """Return a configured :class:`pydantic_ai.Agent` for *agent_name*.

        The Agent is lazily created and cached.  *output_type* affects
        caching — an agent created with ``output_type=str`` is separate
        from one created with ``output_type=WikiOutput``.

        Parameters
        ----------
        agent_name : str, optional
            Which ``[agent.<name>]`` config to use.  Defaults to the
            configured default agent.
        output_type : type, optional
            The Agent's output type (``str`` for plain text, or a
            :class:`pydantic.BaseModel` subclass for structured output).
        config_overrides : dict, optional
            Per-call overrides merged on top of the resolved config.

        Returns
        -------
        pydantic_ai.Agent
        """
        name = agent_name or self._default_name
        cache_key = (name, output_type.__name__)

        if cache_key in self._agents:
            return self._agents[cache_key]

        cfg = dict(self._configs.get(name, self._configs.get(self._default_name, {})))
        if config_overrides:
            cfg = {**cfg, **config_overrides}

        api_key = cfg.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        base_url = cfg.get("base_url", "") or None

        provider = OpenAIProvider(
            base_url=base_url,
            api_key=api_key or "dummy-key",  # placeholder to satisfy client init
        )
        model = OpenAIChatModel(str(cfg.get("model", "deepseek-v4-flash")), provider=provider)

        agent = Agent(
            model=model,
            output_type=output_type,
            system_prompt=str(cfg.get("system_prompt", "")),
            model_settings={
                "temperature": float(cfg.get("temperature", 0.3)),
                "max_tokens": int(cfg.get("max_tokens", 2048)),
            },
            retries=int(cfg.get("max_retries", 3)),
            defer_model_check=True,
        )
        self._agents[cache_key] = agent
        return agent

    @classmethod
    def get_instance(cls) -> AgentManager:
        """Return (and create if needed) the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- Internal -------------------------------------------------------------

    def _resolve_config(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Merge a raw TOML ``[agent.<name>]`` dict with schema defaults."""
        resolved: dict[str, Any] = {}
        for key, param in AGENT_CONFIG_SCHEMA.items():
            if key in raw and raw[key] != "":
                resolved[key] = raw[key]
            elif param.env_var:
                resolved[key] = os.environ.get(param.env_var, param.default)
            else:
                resolved[key] = param.default
        return resolved
