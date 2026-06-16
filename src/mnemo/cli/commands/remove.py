"""mnemo remove command — soft-delete a file from the knowledge base."""

from __future__ import annotations

import rich_click as click

from mnemo.cli.formatter import console, error, success


def run(ctx: click.Context, file_id: str, force: bool):
    """Remove a file from the knowledge base (soft-delete to trash)."""
    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")

    # Support filename lookup
    resolved = kb.resolve_file_ref(file_id)
    if resolved is None:
        error(f"File not found: [bold]{file_id}[/bold]")
        return
    file_id = resolved

    try:
        info = kb.get_info(file_id)
    except KeyError:
        error(f"File not found: {file_id}")
        return

    if not force:
        console.print(f"\n  About to remove: [bold]{info.filename}[/bold]")
        console.print(f"  Type: {info.file_type} | Category: {info.category}")
        if not click.confirm("\n  Proceed?"):
            return

    result = kb.remove(file_id)
    success(f"Removed: [bold]{result['filename']}[/bold]")
    console.print("  Trash:  .mnemo/trash/ (files + DB)")
    console.print(f"  [dim](Use 'mnemo trash restore {result['file_id'][:8]}' to recover within 30 days)[/dim]")
