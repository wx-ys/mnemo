"""mnemo ask command — RAG question answering with citations."""

from __future__ import annotations

import rich_click as click

from mnemo.cli.formatter import (
    ProgressDisplay,
    _icon,
    console,
)


def run(
    ctx: click.Context,
    question: str,
    grounded: bool,
    limit: int,
):
    """Ask a question and get a knowledge-base-grounded answer.

    The RAG pipeline: search → rerank → context assembly → LLM answer.
    """
    from mnemo.api import MnemoAPI
    from mnemo.core.kb_ask import AskPipeline

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    api = MnemoAPI(data_dir if data_dir else "~/mnemo-data")
    pipeline = AskPipeline(api)

    # Track channel stats for display
    channel_stats: dict[str, int] = {}

    def on_progress(stage: str, status: str):
        stage_icon = _icon("running")
        stage_label = stage.title()

        if status.startswith("done:"):
            detail = status.split(":", 1)[1] if ":" in status else "?"
            channel_stats[stage] = detail
            progress.update(f"{stage_icon} {stage_label} — {detail}")
        elif status == "skipped":
            progress.update(f"{stage_icon} {stage_label} — skipped")
        elif status.startswith("in_progress"):
            progress.update(f"{stage_icon} {stage_label}...")
        else:
            progress.update(f"{stage_icon} {stage_label}...")

    with ProgressDisplay(f"Asking: {question[:60]}") as progress:
        response = pipeline.ask(
            question=question,
            grounded=grounded,
            limit=limit,
            on_progress=on_progress,
        )
        progress.update(f"{_icon('ok')} Answer ready")

    # -- Display answer --------------------------------------------------------
    console.print()
    console.print(f"  [bold]Q:[/bold] {question}")
    console.print()

    # Render answer with highlighted citations
    answer = response.answer
    # Colorize citation markers
    import re
    answer = re.sub(
        r'\[(\d+)\]',
        r'[bold cyan][\1][/bold cyan]',
        answer,
    )
    console.print(f"  {answer}")
    console.print()

    # Render citations
    if response.citations:
        console.print(f"  [dim]Sources ({len(response.citations)}):[/dim]")
        for i, c in enumerate(response.citations, 1):
            snippet_display = c.snippet[:100].replace("\n", " ")
            console.print(
                f"    [bold cyan][{i}][/bold cyan] "
                f"[dim]{c.file_id[:8]}...[/dim] "
                f"[italic]{snippet_display}...[/italic]"
            )
        console.print()

    # Footer
    console.print(
        f"  [dim]Model: {response.model} | "
        f"Grounded: {response.grounded}[/dim]"
    )
    console.print()
