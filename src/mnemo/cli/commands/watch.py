"""mnemo watch command — start the file watcher daemon."""

from __future__ import annotations

import rich_click as click

from mnemo.cli.formatter import console


def run(ctx: click.Context, interval: int):
    """Monitor the knowledge base for file changes."""
    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")

    console.print(f"\n  👁️  [bold]Watching[/bold]: {kb.kb.data_dir}")
    console.print(f"  [dim]Interval: {interval}s | Press Ctrl+C to stop[/dim]\n")
    kb.kb.watch(interval=interval)
