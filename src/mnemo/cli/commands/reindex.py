"""mnemo reindex command — rebuild embeddings and/or index."""

from __future__ import annotations

import rich_click as click

from mnemo.cli.formatter import (
    ProgressDisplay,
    _icon,
    banner,
    error,
    summary,
)


def run(
    ctx: click.Context,
    file_id: str | None,
    file_type: str | None,
    all_files: bool,
    meta_only: bool,
):
    """Rebuild embeddings for selected files."""
    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")

    if not any([file_id, file_type, all_files]):
        error("Specify --file, --type, or --all to select files for reindexing.")
        return

    mode_desc = (
        f"file {file_id}" if file_id
        else f"type {file_type}" if file_type
        else "all files"
    )

    banner(f"Reindexing {mode_desc}...", icon_key="reindex")

    with ProgressDisplay(f"Reindexing {mode_desc}...") as progress:
        result = kb.reindex(
            file_id=file_id, file_type=file_type,
            all_files=all_files, meta_only=meta_only,
        )
        progress.update(f"{_icon('ok')} Reindex complete")

    summary(
        reindexed=result["reindexed"],
        skipped=result["skipped"],
        failed=result["failed"],
        label="Reindex complete",
    )
