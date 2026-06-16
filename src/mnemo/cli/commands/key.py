"""mnemo key command — manage hierarchical keys."""

from __future__ import annotations

import rich_click as click

from mnemo.cli.formatter import console, key_table, key_tree, success


def _get_kb(ctx: click.Context):
    """Get a KnowledgeBase instance from CLI context."""
    from mnemo.api import MnemoAPI
    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    return MnemoAPI(data_dir if data_dir else "~/mnemo-data")


def run_list(ctx: click.Context):
    """List all registered keys with usage stats."""
    kb = _get_kb(ctx)
    stats = kb.kb.key_manager.get_key_stats()
    key_table(stats)


def run_add(ctx: click.Context, key_path: str, description: str | None):
    """Register a new key."""
    kb = _get_kb(ctx)
    kb.kb.key_manager.register_key(key_path, description or "")
    success(f"Key registered: [bold]{key_path}[/bold]")


def run_remove(ctx: click.Context, key_path: str):
    """Remove a key and its file associations."""
    kb = _get_kb(ctx)
    kb.kb.key_manager.remove_key(key_path)
    success(f"Key removed: [bold]{key_path}[/bold]")


def run_tree(ctx: click.Context, root_key: str | None):
    """Display the key hierarchy as a tree."""
    kb = _get_kb(ctx)
    tree_data = kb.kb.key_manager.get_key_tree(root_key)
    stats = kb.kb.key_manager.get_key_stats()

    if not tree_data:
        console.print("\n  [dim]No keys registered.[/dim]\n")
        return

    tree = key_tree(tree_data, stats)
    console.print()
    console.print(tree)
    console.print()
