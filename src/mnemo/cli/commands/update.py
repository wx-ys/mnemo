"""mnemo update command — update file metadata."""

from __future__ import annotations

import rich_click as click

from mnemo.cli.formatter import console, error, success


def run(
    ctx: click.Context,
    file_id: str,
    keys: str | None,
    add_keys: str | None,
    remove_keys: str | None,
    tags: str | None,
    note: str | None,
):
    """Update file metadata: keys, tags, and notes."""
    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")

    try:
        current = kb.get_info(file_id)
    except KeyError:
        error(f"File not found: {file_id}")
        return

    # Resolve key operations
    final_keys = None
    if keys is not None:
        final_keys = [k.strip() for k in keys.split(",") if k.strip()]
    else:
        current_keys = list(current.keys)
        if add_keys:
            to_add = [k.strip() for k in add_keys.split(",") if k.strip()]
            for k in to_add:
                if k not in current_keys:
                    current_keys.append(k)
        if remove_keys:
            to_remove = set(k.strip() for k in remove_keys.split(",") if k.strip())
            current_keys = [k for k in current_keys if k not in to_remove]
        if add_keys or remove_keys:
            final_keys = current_keys

    # Resolve tags
    final_tags = None
    if tags is not None:
        final_tags = [t.strip() for t in tags.split(",") if t.strip()]

    updated = kb.update(file_id, keys=final_keys, tags=final_tags, note=note)

    success(f"Updated: [bold]{updated.filename}[/bold]")
    console.print(f"  Keys:  {', '.join(updated.keys) if updated.keys else '(none)'}")
    console.print(f"  Tags:  {', '.join(updated.tags) if updated.tags else '(none)'}")
