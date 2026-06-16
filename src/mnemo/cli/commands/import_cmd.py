"""mnemo import command — import an external knowledge base."""

from __future__ import annotations

import rich_click as click

from mnemo.cli.formatter import (
    ProgressDisplay,
    _icon,
    banner,
    console,
    error,
    summary,
)


def run(ctx: click.Context, source: str, dry_run: bool):
    """Import a tar.gz archive or directory into the knowledge base."""
    from pathlib import Path

    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")

    src_path = Path(source)
    if not src_path.exists():
        error(f"Source not found: {source}")
        return

    label = "Preview import" if dry_run else "Importing"
    banner(f"{label} from {source}...", icon_key="import")

    with ProgressDisplay(f"{label} from {src_path.name}...") as progress:
        report = kb.import_kb(src_path, dry_run=dry_run)
        progress.update(f"{_icon('ok')} {label} complete")

    summary(
        added=report.imported, skipped=report.skipped,
        errors=len(report.errors),
        label="Import preview" if dry_run else "Import complete",
    )
    for err in report.errors:
        console.print(f"    [dim]- {err}[/dim]")
