"""mnemo add command — ingest files into the knowledge base."""

from __future__ import annotations

from pathlib import Path

import rich_click as click

from mnemo.cli.formatter import (
    ProgressDisplay,
    _icon,
    add_result,
    console,
    error,
    summary,
    warn,
)


def run(
    ctx: click.Context,
    source: str | None,
    file_path: str | None,
    dir_path: str | None,
    url: str | None,
    move: bool,
    keys: str | None,
    tags: str | None,
    note: str | None,
    no_md: bool,
    no_wiki: bool,
    no_embed: bool,
    overwrite: bool,
    diagnose: bool = False,
    verbose: bool = False,
):
    """Add files, directories, or URLs to the knowledge base."""
    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")

    # Parse keys and tags
    key_list = [k.strip() for k in keys.split(",") if k.strip()] if keys else None
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    # Resolve source: positional arg is auto-detected, explicit flags override
    sources: list[str] = []

    if url:
        try:
            result = kb.add_url(
                url, move=False, keys=key_list, tags=tag_list, note=note or "",
                auto_md=not no_md, auto_wiki=not no_wiki,
                auto_embed=not no_embed, overwrite=overwrite,
            )
            add_result(result)
        except NotImplementedError:
            error("URL ingestion is not yet implemented. Download the file first, then use the path.")
        return

    # --dir flag
    if dir_path:
        dir_p = Path(dir_path)
        if dir_p.is_dir():
            sources.extend(str(f) for f in dir_p.rglob("*") if f.is_file())
        else:
            error(f"Not a directory: {dir_path}\n  Use [bold]mnemo add <file>[/bold] for files, or --dir for directories.")
            return

    # --file flag (explicit)
    if file_path:
        sources.append(file_path)

    # Positional argument (auto-detect)
    if source and not file_path and not dir_path:
        src_path = Path(source)
        if not src_path.exists():
            error(f"Source not found: {source}")
            return
        if src_path.is_dir():
            error(
                f"[bold]{source}[/bold] is a directory.\n"
                f"  To add a directory: [bold]mnemo add --dir {source}[/bold]"
            )
            return
        sources.append(source)

    if not sources:
        warn("No source specified.")
        console.print(
            "  [dim]Usage:[/dim]\n"
            "    mnemo add [bold]<file>[/bold]              add a file\n"
            "    mnemo add --dir [bold]<dir>[/bold]         add all files in a directory\n"
            "    mnemo add --url [bold]<url>[/bold]         add from a URL"
        )
        return

    # -- Process each source --------------------------------------------------
    added_count = 0
    error_count = 0

    def on_progress(step: str, status: str):
        step_label = step.replace("_", " ").title()
        step_icon = _icon(f"step.{step}") if f"step.{step}" in {
            "validate", "copy", "metadata", "convert_md", "generate_wiki",
            "extract_entities", "generate_embedding",
        } else _icon("running")

        # Thinking/reasoning content streams in real-time during LLM steps.
        # Use update_message() so the stage timer keeps counting (not reset
        # on every thinking delta).  Status format: "thinking:<text>"
        if status.startswith("thinking:"):
            thinking = status[len("thinking:"):]
            # Truncate very long thinking chunks for spinner readability
            display = thinking if len(thinking) <= 120 else f"{thinking[:117]}..."
            progress.update_message(
                f"{step_icon} {step_label}  🤔 {display}"
            )
        elif status == "done":
            progress.update(f"{step_icon} {step_label} — done")
        elif status == "skipped":
            progress.update(f"{step_icon} {step_label} — skipped")
        else:
            progress.update(f"{step_icon} {step_label}...")

    for i, src in enumerate(sources, 1):
        fname = Path(src).name
        label = f"[{i}/{len(sources)}] {fname}"

        with ProgressDisplay(f"Adding {label}") as progress:
            try:
                result = kb.add(
                    src,
                    move=move,
                    keys=key_list,
                    tags=tag_list,
                    note=note or "",
                    auto_md=not no_md,
                    auto_wiki=not no_wiki,
                    auto_embed=not no_embed,
                    overwrite=overwrite,
                    on_progress=on_progress,
                    diagnose=diagnose,
                    verbose=verbose,
                )
                # Check if it was a duplicate (identical hash)
                if getattr(result, '_duplicate', False):
                    console.print(f"  ⏭️  [dim]{fname}: skipped (identical file already exists)[/dim]")
                    console.print(f"       ID: {result.id}")
                    continue
                added_count += 1
            except FileExistsError as e:
                console.print(f"  ⚠️  [yellow]{fname}: name conflict[/yellow]")
                console.print(f"       [dim]{e}[/dim]")
                error_count += 1
                continue
            except FileNotFoundError as e:
                console.print(f"  ✗ [red]{fname}: {e}[/red]")
                error_count += 1
                continue
            except Exception as e:
                console.print(f"  ✗ [red]{fname}: {e}[/red]")
                error_count += 1
                continue

        add_result(result, i)

    summary(added=added_count, errors=error_count)
