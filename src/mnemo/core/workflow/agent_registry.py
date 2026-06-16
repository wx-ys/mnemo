"""Agent registry — pydantic-ai Agent factory with Tool + Deps support.

Upgrades the existing ``AgentManager`` singleton to support:
- Tool resolution from ``ToolLibrary`` (tools by name → pydantic-ai Tool)
- Structured output via pydantic BaseModel (output_type)
- WorkflowDeps injection for AgentSteps
- Backward compatibility: AgentManager delegates to AgentRegistry

Usage::

    from mnemo.core.workflow.agent_registry import AgentRegistry

    # Initialize once (delegates to AgentManager.init)
    AgentRegistry.init(config)

    # Get an agent with tools
    agent = AgentRegistry.get_agent_with_tools(
        "default",
        tools=["search_kb", "get_file_context"],
        output_type=WikiOutput,
    )

    # Run with WorkflowDeps
    result = agent.run_sync(prompt, deps=WorkflowDeps(kb=kb_instance))
"""

from __future__ import annotations

import os
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from mnemo.core.workflow.tools import ToolLibrary


# Re-export the config schema from AgentManager for consistency
from mnemo.core.agent_manager import AGENT_CONFIG_SCHEMA  # noqa: F401


class AgentRegistry:
    """Central registry for pydantic-ai Agent instances.

    Singleton that creates and caches Agent instances per (name, output_type)
    key.  Adds tool resolution and WorkflowDeps support on top of the
    existing AgentManager infrastructure.

    This class mostly delegates configuration parsing to AgentManager,
    then adds the tool/system_prompt resolution layer on top.
    """

    _instance: AgentRegistry | None = None

    def __new__(cls) -> AgentRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_ready") and self._ready:
            return
        # Cache: (agent_name, output_type_name) → Agent
        self._agents: dict[tuple[str, str], Agent[Any]] = {}
        self._ready: bool = False

    # -- initialization -------------------------------------------------------

    @classmethod
    def init(cls, config: dict[str, Any]) -> None:
        """Ensure the AgentManager is initialized (delegates).

        Must be called once after config is loaded.
        """
        from mnemo.core.agent_manager import AgentManager
        am = AgentManager.get_instance()
        if not am._initialized:
            am.init(config)
        # Mark ourself as ready
        instance = cls._get()
        instance._ready = True

    @classmethod
    def _get(cls) -> AgentRegistry:
        """Get (or create) the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- public API -----------------------------------------------------------

    def get_agent(
        self,
        agent_name: str | None = None,
        *,
        output_type: type[Any] = str,
        config_overrides: dict[str, Any] | None = None,
    ) -> Agent[Any]:
        """Return a pydantic-ai Agent without extra tools.

        Delegates directly to ``AgentManager.get_agent`` for backward
        compatibility.
        """
        from mnemo.core.agent_manager import AgentManager
        return AgentManager.get_instance().get_agent(
            agent_name=agent_name,
            output_type=output_type,
            config_overrides=config_overrides,
        )

    def get_agent_with_tools(
        self,
        agent_name: str | None = None,
        *,
        tools: list[str] | None = None,
        output_type: type[Any] = str,
        system_prompt: str | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> Agent[Any]:
        """Return a pydantic-ai Agent with ToolLibrary tools attached.

        Parameters
        ----------
        agent_name : str, optional
            Which ``[agent.<name>]`` config to use.
        tools : list of str, optional
            Tool names from ToolLibrary to attach.
        output_type : type
            Pydantic BaseModel for structured output, or str.
        system_prompt : str, optional
            Override the default system prompt.
        config_overrides : dict, optional
            Per-call config overrides.

        Returns
        -------
        pydantic_ai.Agent
        """
        from mnemo.core.agent_manager import AgentManager

        name = agent_name or AgentManager.get_instance().default_agent_name
        cache_key = (name, output_type.__name__, tuple(sorted(tools or [])))

        if cache_key in self._agents:
            agent = self._agents[cache_key]
            # NOTE: system_prompt should be passed via instructions=
            # at call time (e.g., agent.run_sync(prompt, instructions=...)).
            # agent._system_prompt is a legacy pattern from pydantic-ai <1.x
            # and does not exist in v1.x.
            return agent

        # Resolve tool names → plain functions (Agent auto-wraps them)
        tool_funcs = ToolLibrary.resolve_many(tools or [])

        # Get base agent config from AgentManager
        am = AgentManager.get_instance()
        base_agent = am.get_agent(
            agent_name=name,
            output_type=output_type,
            config_overrides=config_overrides,
        )

        # Re-create agent with tools if tools are requested
        if tool_funcs:
            cfg = self._get_config(name, config_overrides)
            agent = self._build_agent(
                cfg, output_type, system_prompt, tool_funcs,
            )
        else:
            agent = base_agent

        # NOTE: system_prompt should be passed via instructions=
        # at call time (see AgentStep._run for the pattern).
        self._agents[cache_key] = agent
        return agent

    def list_agents(self) -> list[str]:
        """Return all configured agent names."""
        from mnemo.core.agent_manager import AgentManager
        return AgentManager.get_instance().list_agent_names()

    # -- internal -------------------------------------------------------------

    def _get_config(
        self, name: str, overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get the resolved config dict for an agent."""
        from mnemo.core.agent_manager import AgentManager
        am = AgentManager.get_instance()
        cfg = dict(am._configs.get(name, am._configs.get(am._default_name, {})))
        if overrides:
            cfg = {**cfg, **overrides}
        return cfg

    def _build_agent(
        self,
        cfg: dict[str, Any],
        output_type: type[Any],
        system_prompt: str | None,
        tools: list[Any],
    ) -> Agent[Any]:
        """Build a pydantic-ai Agent from config + tools."""
        api_key = cfg.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        base_url = cfg.get("base_url", "") or None

        provider = OpenAIProvider(
            base_url=base_url,
            api_key=api_key or "dummy-key",
        )
        model = OpenAIChatModel(
            str(cfg.get("model", "deepseek-v4-flash")),
            provider=provider,
        )

        agent = Agent(
            model=model,
            output_type=output_type,
            system_prompt=system_prompt or str(cfg.get("system_prompt", "")),
            model_settings={
                "temperature": float(cfg.get("temperature", 0.3)),
                "max_tokens": int(cfg.get("max_tokens", 2048)),
            },
            tools=tools if tools else None,
            retries=int(cfg.get("max_retries", 3)),
            defer_model_check=True,
        )
        return agent
