"""mnemo info command — show detailed file information."""

from __future__ import annotations

import rich_click as click

from mnemo.cli.formatter import error, file_info_panel


def run(ctx: click.Context, file_ref: str):
    """Display detailed information about a file (by ID or filename)."""
    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")

    # Try as UUID first, then as filename
    file_id = kb.resolve_file_ref(file_ref)

    if file_id is None:
        error(f"File not found: [bold]{file_ref}[/bold]")
        return

    try:
        info = kb.get_info(file_id)
    except KeyError:
        error(f"File not found: [bold]{file_ref}[/bold]")
        return

    # Best-effort context
    context = None
    try:
        context = kb.get_context(file_id)
    except Exception:
        pass

    file_info_panel(info, context)
