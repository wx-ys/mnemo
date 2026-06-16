"""mnemo check command — check knowledge base integrity."""

from __future__ import annotations

import rich_click as click

from mnemo.cli.formatter import check_report


def run(ctx: click.Context, fix: bool):
    """Check knowledge base integrity and optionally repair issues."""
    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")

    report = kb.check(fix=fix)
    check_report(report)
