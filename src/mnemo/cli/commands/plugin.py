"""mnemo plugin command — list registered plugins."""

from __future__ import annotations

import rich_click as click

from mnemo.cli.formatter import plugin_table


def run_list(ctx: click.Context):
    """List all registered parser and template plugins."""
    from mnemo.core.plugin_base import PluginHub
    from mnemo.core.interfaces import IParser, ITemplate

    plugins = []

    for name, impl_cls in PluginHub.iter_impls(IParser):
        inst = PluginHub.get(IParser, name)
        plugins.append({"type": "Parser", "name": name,
                        "category": getattr(inst, "category", "")})

    for name, impl_cls in PluginHub.iter_impls(ITemplate):
        inst = PluginHub.get(ITemplate, name)
        plugins.append({"type": "Template", "name": name,
                        "category": getattr(inst, "category", "")})

    # Agent & Embedder configs (not plugin-based, read from config)
    try:
        from mnemo.core.agent_manager import AgentManager
        am = AgentManager.get_instance()
        if am._initialized:
            for name in am.list_agent_names():
                plugins.append({"type": "Agent", "name": name, "category": ""})
    except Exception:
        pass

    plugin_table(plugins)
