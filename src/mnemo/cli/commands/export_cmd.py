"""mnemo export command — export the knowledge base."""

from __future__ import annotations

import rich_click as click

from mnemo.cli.formatter import (
    ProgressDisplay,
    _icon,
    banner,
    success,
)


def run(
    ctx: click.Context,
    dest: str,
    file_type: str | None,
    keys: str | None,
    after: str | None,
):
    """Export the knowledge base to a tar.gz archive."""
    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")

    key_list = [k.strip() for k in keys.split(",") if k.strip()] if keys else None

    banner(f"Exporting to {dest}...", icon_key="export")

    with ProgressDisplay(f"Exporting to {dest}...") as progress:
        archive_path = kb.export_kb(dest=dest, file_type=file_type, keys=key_list, after=after)
        progress.update(f"{_icon('ok')} Export complete")

    success(f"Exported: [bold]{archive_path}[/bold]")
