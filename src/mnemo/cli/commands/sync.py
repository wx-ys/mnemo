"""mnemo sync command — remote synchronization via rclone."""

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


def _get_remote(kb) -> str:
    """Get configured remote, or complain."""
    remote = kb.kb.config_loader.get("sync.remote", "")
    if not remote:
        error("No sync.remote configured. Set it in config.toml.")
    return remote


def run_push(ctx: click.Context):
    """Push local data to the configured remote."""
    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")

    remote = _get_remote(kb)
    if not remote:
        return

    banner(f"Pushing to {remote}...", icon_key="sync")

    with ProgressDisplay(f"Pushing to {remote}...") as progress:
        report = kb.sync_push()
        progress.update(f"{_icon('ok')} Sync push complete")

    summary(
        added=report.synced, skipped=report.skipped,
        errors=len(report.errors), label="Sync push complete",
    )
    for err in report.errors:
        console.print(f"    [dim]- {err}[/dim]")


def run_pull(ctx: click.Context):
    """Pull remote data to local."""
    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")

    remote = _get_remote(kb)
    if not remote:
        return

    banner(f"Pulling from {remote}...", icon_key="sync")

    with ProgressDisplay(f"Pulling from {remote}...") as progress:
        report = kb.sync_pull()
        progress.update(f"{_icon('ok')} Sync pull complete")

    summary(
        added=report.synced, skipped=report.skipped,
        errors=len(report.errors), label="Sync pull complete",
    )
    for err in report.errors:
        console.print(f"    [dim]- {err}[/dim]")


def run_status(ctx: click.Context):
    """Check sync status."""
    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")

    status = kb.kb.syncer.status()
    console.print("\n  ☁️  [bold]Sync Status[/bold]\n")
    console.print(f"  Last push:  {status.get('last_push', 'N/A')}")
    console.print(f"  Last pull:  {status.get('last_pull', 'N/A')}")
    console.print(f"  Pending:    {status.get('pending_changes', 'N/A')}")
    console.print()
