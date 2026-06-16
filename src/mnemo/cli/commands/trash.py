"""mnemo trash command — manage the recycle bin.

Trash is a complete mirror of the main knowledge base structure
under ``.mnemo/trash/`` with its own SQLite + LanceDB.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import rich_click as click
from rich import box
from rich.table import Table

from mnemo.cli.formatter import console, error, human_size, success


def _get_trash_and_api(ctx: click.Context):
    """Get TrashStore and MnemoAPI from CLI context."""
    from mnemo.api import MnemoAPI
    from mnemo.core.trash_store import TrashStore

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    api = MnemoAPI(data_dir if data_dir else "~/mnemo-data")
    trash = TrashStore(api.kb.data_dir)
    return trash, api


def run_list(ctx: click.Context):
    """List files in the trash (soft-deleted)."""
    trash, _api = _get_trash_and_api(ctx)

    items, total = trash.list_trash()

    if not items:
        console.print("\n  [dim]🗑️  Trash is empty.[/dim]\n")
        trash.close()
        return

    table = Table(
        title=f"\n🗑️  [bold]Trash[/bold] ({total} items)",
        box=box.ROUNDED, border_style="yellow",
    )
    table.add_column("#", style="bold cyan", width=4, justify="right")
    table.add_column("Filename", style="white", max_width=40)
    table.add_column("Type", style="yellow", width=10)
    table.add_column("Size", style="green", width=10)
    table.add_column("Deleted", style="dim", width=20)

    for i, item in enumerate(items, 1):
        table.add_row(
            str(i),
            item["filename"],
            item["file_type"],
            human_size(item["file_size"]),
            item["deleted_at"][:19] if item["deleted_at"] else "?",
        )

    console.print(table)
    console.print("  [dim]Use[/dim] [bold]mnemo trash restore <#>[/bold] [dim]or[/dim] [bold]mnemo trash restore <name>[/bold]")
    console.print()
    trash.close()


def run_restore(ctx: click.Context, file_ref: str):
    """Restore a file from trash by number (#), UUID, or filename.

    Examples:
        mnemo trash restore 1          restore item #1 from list
        mnemo trash restore paper      restore by filename match
        mnemo trash restore abc123...  restore by file ID
    """
    trash, api = _get_trash_and_api(ctx)
    kb = api.kb

    items, _ = trash.list_trash(limit=500)

    if not items:
        console.print("\n  [dim]Trash is empty.[/dim]\n")
        trash.close()
        return

    # Resolve file_ref: try number → index → exact ID → filename match
    resolved_id: str | None = None

    # Try number
    try:
        idx = int(file_ref) - 1
        if 0 <= idx < len(items):
            resolved_id = items[idx]["file_id"]
    except ValueError:
        pass

    # Try exact file ID match
    if resolved_id is None:
        for item in items:
            if item["file_id"] == file_ref or item["file_id"].startswith(file_ref):
                resolved_id = item["file_id"]
                break

    # Try filename match (contains)
    if resolved_id is None:
        matches = [item for item in items if file_ref.lower() in item["filename"].lower()]
        if len(matches) == 1:
            resolved_id = matches[0]["file_id"]
        elif len(matches) > 1:
            console.print(f"\n  [yellow]Multiple matches for '{file_ref}':[/yellow]")
            for i, m in enumerate(matches, 1):
                console.print(f"    [{i}] {m['filename']} ({m['file_type']}) — {m['file_id'][:8]}...")
            console.print("\n  [dim]Use a number or full ID to select.[/dim]")
            trash.close()
            return

    if resolved_id is None:
        error(f"No file matching '{file_ref}' found in trash.")
        console.print("  [dim]Use[/dim] [bold]mnemo trash list[/bold] [dim]to see items.[/dim]")
        trash.close()
        return

    # Resolve the match name for confirmation
    match_item = next((item for item in items if item["file_id"] == resolved_id), None)
    if match_item is None:
        error(f"File not found in trash: {file_ref}")
        trash.close()
        return

    console.print(f"\n  Restoring: [bold]{match_item['filename']}[/bold]")
    console.print(f"  Type: {match_item['file_type']} | "
                  f"Deleted: {match_item['deleted_at'][:19] if match_item['deleted_at'] else '?'}")

    if not click.confirm("\n  Proceed?"):
        trash.close()
        return

    result = trash.restore_file(resolved_id, kb.indexer, kb.key_manager)

    if result is None:
        error(f"Failed to restore: {resolved_id}")
    else:
        success(f"Restored: [bold]{result['filename']}[/bold]")
        console.print(
            f"  [dim]File is back in the knowledge base. "
            f"Run[/dim] mnemo reindex --file {resolved_id[:8]} [dim]to re-embed.[/dim]"
        )

    trash.close()


def run_clean(ctx: click.Context, force: bool):
    """Permanently delete files older than 30 days from trash."""
    trash, _api = _get_trash_and_api(ctx)

    items, total = trash.list_trash(limit=10000)

    if not items:
        console.print("\n  [dim]Trash is empty.[/dim]\n")
        trash.close()
        return

    cutoff = datetime.now(UTC) - timedelta(days=30)
    old_items = [
        item for item in items
        if item["deleted_at"] and item["deleted_at"] < cutoff.isoformat()
    ]

    if not old_items:
        console.print("\n  [dim]No items older than 30 days.[/dim]\n")
        trash.close()
        return

    if not force:
        console.print(f"\n  [yellow]{len(old_items)} item(s)[/yellow] older than 30 days:")
        for i, item in enumerate(old_items, 1):
            console.print(
                f"    [{i}] {item['filename']} "
                f"([dim]{item['file_type']}, deleted {item['deleted_at'][:10]}[/dim])"
            )
        if not click.confirm("\n  Proceed with permanent deletion?"):
            trash.close()
            return

    count = trash.clean_expired()
    success(f"Permanently deleted {count} item(s).")
    trash.close()
