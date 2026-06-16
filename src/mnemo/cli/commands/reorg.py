"""mnemo reorg command — reorganize chunk directories."""

import rich_click as click

from mnemo.cli.formatter import console


def run(ctx: click.Context, file_type: str | None, dry_run: bool, confirm: bool):
    """Reorganize files by chunk strategy (not yet implemented)."""
    console.print("\n  ⏳ [dim]Reorg is not yet implemented.[/dim]\n")
