"""mnemo list command — list files in the knowledge base."""

from __future__ import annotations

import json

import rich_click as click
from rich.console import Console as _Unused  # noqa

from mnemo.cli.formatter import console, file_table, human_size


def run(
    ctx: click.Context,
    file_type: str | None,
    tags: str | None,
    keys: str | None,
    sort_by: str,
    limit: int,
    offset: int,
    output_format: str,
):
    """List files with optional filters and pagination."""
    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    key_list = [k.strip() for k in keys.split(",") if k.strip()] if keys else None

    results = kb.list_files(
        file_type=file_type,
        keys=key_list,
        tags=tag_list,
        sort_by=sort_by,
        limit=limit,
        offset=offset,
    )

    if output_format == "json":
        console.print_json(json.dumps(
            [{"id": r.id, "filename": r.filename, "file_type": r.file_type,
              "category": r.category, "file_size": r.file_size,
              "file_size_human": human_size(r.file_size),
              "added_at": r.added_at, "keys": r.keys, "tags": r.tags}
             for r in results],
            ensure_ascii=False, indent=2,
        ))
    elif output_format == "csv":
        click.echo("id,filename,file_type,category,file_size,added_at,keys")
        for r in results:
            keys_str = ";".join(r.keys) if r.keys else ""
            click.echo(f'{r.id},{r.filename},{r.file_type},{r.category},{r.file_size},{r.added_at},"{keys_str}"')
    else:
        file_table(results)
