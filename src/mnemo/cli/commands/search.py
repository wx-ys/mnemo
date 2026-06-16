"""mnemo search command — search the knowledge base."""

from __future__ import annotations

import json
from pathlib import Path

import rich_click as click

from mnemo.cli.formatter import (
    ProgressDisplay,
    _icon,
    console,
    search_table,
    success,
    warn,
)


def run(
    ctx: click.Context,
    query: str,
    mode: str,
    keys: str | None,
    file_type: str | None,
    limit: int,
    with_meta: bool,
    output_format: str,
    expand_chunks: bool,
    output: str | None,
    diagnose: bool = False,
    verbose: bool = False,
):
    """Search the knowledge base and display results."""
    from mnemo.api import MnemoAPI, SearchMode

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")

    # Gather searcher info for progress and summary
    _kb = kb.kb
    searcher_name = getattr(_kb.searcher, 'name', '?')
    searcher_caps = getattr(_kb.searcher, 'required_capabilities', set())
    searcher_type = type(_kb.searcher).__name__

    # Gather embedder info from global singleton (core/embedder.py)
    try:
        from mnemo.core.embedder import get_model_name, get_dimension
        embedder_name = "embedder"
        embedder_model = get_model_name() or "?"
        embedder_dim = str(get_dimension())
    except RuntimeError:
        embedder_name = "embedder"
        embedder_model = "?"
        embedder_dim = "?"

    # Check capability compatibility and warn if needed
    if mode in ("hybrid", "vector") and "embeddings" not in searcher_caps:
        warn(
            f"Active searcher '{searcher_name}' does not support "
            f"[bold]{mode}[/bold] mode (no embedding capability). "
            f"Falling back to keyword search."
        )
    if mode == "hybrid" and "graph_entities" not in searcher_caps:
        console.print(
            f"  [dim]Note: graph-enhanced search unavailable "
            f"(searcher '{searcher_name}' lacks graph capability)[/dim]"
        )

    key_list = [k.strip() for k in keys.split(",") if k.strip()] if keys else None
    file_types = [file_type] if file_type else None

    mode_map = {
        "hybrid": SearchMode.HYBRID,
        "vector": SearchMode.VECTOR,
        "keyword": SearchMode.KEYWORD,
    }
    search_mode = mode_map.get(mode, SearchMode.HYBRID)

    # Channel stats collected by on_progress
    channel_stats: dict[str, int] = {}
    _interactive = output_format not in ("json", "csv")

    def on_progress(stage: str, status: str):
        if not _interactive:
            return
        stage_icon = _icon(f"step.{stage}") if f"step.{stage}" in {
            "vector", "keyword", "graph", "fuse", "grep",
        } else _icon("running")
        stage_label = stage.replace("_", " ").title()

        if status.startswith("done:"):
            count = status.split(":")[1] if ":" in status else "?"
            channel_stats[stage] = int(count) if count.isdigit() else 0
            progress.update(
                f"{stage_icon} {stage_label} — {count} hits"
            )
        elif status == "skipped":
            progress.update(f"{stage_icon} {stage_label} — skipped")
        elif status.startswith("in_progress:"):
            # Plugin info embedded in status: "in_progress:IEmbedder[openai/model]"
            plugin_info = status.split(":", 1)[1] if ":" in status else ""
            progress.update(
                f"{stage_icon} {stage_label} [dim]via {plugin_info}[/dim]..."
            )
        else:
            progress.update(f"{stage_icon} {stage_label}...")

    if _interactive:
        ctx_mgr = ProgressDisplay(f"Searching: {query[:60]}")
    else:
        from contextlib import nullcontext
        ctx_mgr = nullcontext()
    progress = None  # type: ignore[assignment]

    with ctx_mgr as progress:
        results = kb.search(
            query=query,
            mode=search_mode,
            keys=key_list,
            file_types=file_types,
            limit=limit,
            with_metadata=with_meta,
            expand_chunks=expand_chunks,
            on_progress=on_progress,
            diagnose=diagnose,
            verbose=verbose,
        )

    if _interactive and progress is not None:
        progress.update(
            f"{_icon('ok')} Search complete — {len(results)} results"
        )

    # -- Search pipeline info (interactive only) --------------------------
    if _interactive:
        # Line 1: searcher + channels
        info_parts = [f"[bold]{searcher_name}[/bold]"]
        if searcher_type != "LightRAGSearcher":
            info_parts.append(f"({searcher_type})")
        ch_parts = []
        for ch in ("vector", "keyword", "graph", "grep"):
            if ch in channel_stats:
                ch_parts.append(f"{ch}:{channel_stats[ch]}")
        if ch_parts:
            info_parts.append("[dim]" + " ".join(ch_parts) + "[/dim]")
        console.print(f"  {_icon('search')} {' · '.join(info_parts)}")

        # Line 2: embedder details (what embedded the query)
        embed_parts = [
            f"[dim]embed:[/dim] {embedder_name}",
            f"[dim]model:[/dim] {embedder_model}",
            f"[dim]dim:[/dim] {embedder_dim}",
            f"[dim]mode:[/dim] {mode}",
        ]
        console.print(f"     {'  '.join(embed_parts)}")

    # -- Serialize output ------------------------------------------------
    if output:
        out_path = Path(output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append(f"# Search: {query}")
        lines.append(f"# Mode: {mode} | Searcher: {searcher_name} | Results: {len(results)}")
        lines.append("")
        for r in results:
            lines.append(f"## [{r.score:.4f}] {r.file_type or ''} — {r.match_source or ''}")
            lines.append(f"  ID: {r.id}")
            lines.append(f"  Snippet: {r.snippet}")
            if r.match_count:
                lines.append(f"  Match count: {r.match_count}")
            lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        success(f"Results saved to: {out_path}")

    # -- Terminal output -------------------------------------------------
    if output_format == "json":
        console.print_json(json.dumps(
            [{"id": r.id, "score": round(r.score, 4), "snippet": r.snippet,
              "file_type": r.file_type, "match_source": r.match_source,
              "match_count": r.match_count}
             for r in results],
            ensure_ascii=False, indent=2,
        ))
    elif output_format == "csv":
        click.echo("id,score,snippet,file_type,match_source,match_count")
        for r in results:
            snippet = r.snippet.replace('"', '""')
            click.echo(f'{r.id},{r.score:.4f},"{snippet}",{r.file_type},{r.match_source},{r.match_count}')
    else:
        search_table(results)
