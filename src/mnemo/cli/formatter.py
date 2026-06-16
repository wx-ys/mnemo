"""CLI output formatting — centralized styling with Rich.

Every command uses helpers from this module for consistent
colors, icons, tables, panels, and progress display across the entire CLI.
"""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree as RichTree

# ---------------------------------------------------------------------------
# Shared console
# ---------------------------------------------------------------------------

console = Console()

# ---------------------------------------------------------------------------
# Progress display — dynamic multi-stage spinner for CLI operations
# ---------------------------------------------------------------------------


class ProgressDisplay:
    """Dynamic progress display with real-time elapsed time counter.

    Shows TWO timers that auto-refresh via a background thread:
    - **Stage time**: elapsed since last ``update()`` call (resets per stage)
    - **Total time**: elapsed since ``__enter__``

    Usage::

        with ProgressDisplay("Adding files...") as progress:
            progress.update("Finding files...")    # stage: 0.0s  total: 0.0s
            # ... do work ...
            progress.update("Copying file.txt")     # stage: 0.0s  total: 1.2s
            # ... do work ...
    """

    def __init__(self, message: str = "", spinner: str = "dots"):
        import time as _time
        self._message = message
        self._spinner = spinner
        self._status: Any = None
        self._total_start = _time.monotonic()
        self._stage_start = self._total_start
        self._stage_msg = message
        self._running = False

    def __enter__(self) -> ProgressDisplay:
        self._status = console.status(
            self._make_line(self._message),
            spinner=self._spinner,
        )
        self._status.__enter__()
        self._running = True
        self._start_refresh_thread()
        return self

    def __exit__(self, *args: Any) -> None:
        self._running = False
        if self._status is not None:
            self._status.__exit__(*args)

    def update(self, message: str) -> None:
        """Transition to a new stage — resets the stage timer."""
        import time as _time
        self._stage_start = _time.monotonic()
        self._stage_msg = message
        if self._status is not None:
            self._status.update(self._make_line(message))

    def update_message(self, message: str) -> None:
        """Update the status message without resetting the stage timer.

        Useful for streaming content (e.g., LLM thinking) that should
        appear inline in the progress bar but not reset the stage clock.
        """
        self._stage_msg = message
        if self._status is not None:
            self._status.update(self._make_line(message))

    def _make_line(self, message: str) -> str:
        """Build the status line: message + [stage_time / total_time]."""
        import time as _time
        now = _time.monotonic()
        stage_elapsed = now - self._stage_start
        total_elapsed = now - self._total_start
        return (
            f"[bold blue]{message}[/bold blue]  "
            f"[dim]{stage_elapsed:.1f}s / {total_elapsed:.1f}s[/dim]"
        )

    def _start_refresh_thread(self) -> None:
        """Background thread that refreshes the time display."""
        import threading
        import time as _time

        def _refresh_loop():
            while self._running:
                _time.sleep(0.15)
                if self._status is not None and self._running:
                    try:
                        self._status.update(self._make_line(self._stage_msg))
                    except Exception:
                        pass

        t = threading.Thread(target=_refresh_loop, daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Icons
# ---------------------------------------------------------------------------

ICON = {
    # Operations
    "init":       "🛠️",
    "add":        "📥",
    "search":     "🔍",
    "list":       "📋",
    "info":       "📄",
    "update":     "✏️",
    "remove":     "🗑️",
    "reindex":    "🔄",
    "check":      "🔬",
    "watch":      "👁️",
    "export":     "📦",
    "import":     "📥",
    "sync":       "☁️",
    "config":     "⚙️",
    "key":        "🔑",
    "plugin":     "🧩",
    "trash":      "🗑️",
    # Status
    "ok":         "✅",
    "done":       "✅",
    "skip":       "⏭️",
    "fail":       "❌",
    "warn":       "⚠️",
    "pending":    "⏳",
    "running":    "🚀",
    "in_progress":"🔄",
    # File types
    "docs":       "📝",
    "data":       "📊",
    "code":       "💻",
    "img":        "🖼",
    "audio":      "🎵",
    "video":      "🎬",
    "web":        "🌐",
    "other":      "📎",
    # Steps
    "step.validate":        "🔍",
    "step.copy":            "📋",
    "step.metadata":        "🏷️",
    "step.convert_md":      "📝",
    "step.generate_wiki":   "🤖",
    "step.extract_entities":"🧠",
    "step.generate_embedding":"🧮",
    # Misc
    "hash":       "🔒",
    "size":       "📏",
    "clock":      "🕐",
    "source":     "📁",
    "keys":       "🔑",
    "tags":       "🏷️",
    "note":       "📝",
    "sparkles":   "✨",
    "rocket":     "🚀",
    "package":    "📦",
    "link":       "🔗",
    "pin":        "📌",
}


def human_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    for unit in ("KB", "MB", "GB", "TB"):
        size_bytes /= 1024
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
    return f"{size_bytes:.1f} PB"


def _icon(key: str) -> str:
    return ICON.get(key, "•")


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def status_color(status: str) -> str:
    """Map a status string to a Rich color name."""
    return {
        "done": "green", "ok": "green", "healthy": "green",
        "pending": "yellow", "in_progress": "yellow", "incomplete": "yellow",
        "failed": "red", "error": "red", "missing": "red",
        "skipped": "dim", "todo": "dim",
    }.get(status.lower(), "white")


def status_badge(status: str) -> str:
    """Return a colored icon+label string for a status."""
    color = status_color(status)
    icon_map = {
        "done": "✅", "ok": "✅", "healthy": "✅",
        "pending": "⏳", "in_progress": "🔄", "incomplete": "⏳",
        "failed": "❌", "error": "❌", "missing": "❌",
        "skipped": "⏭️", "todo": "⏭️",
    }
    icon = icon_map.get(status.lower(), "•")
    return f"[{color}]{icon} {status}[/{color}]"


# ---------------------------------------------------------------------------
# Headers & banners
# ---------------------------------------------------------------------------

def header(title: str, icon_key: str = "", subtitle: str = "") -> None:
    """Print a section header."""
    icon = _icon(icon_key) if icon_key else ""
    text = f"{icon} [bold]{title}[/bold]" if icon else f"[bold]{title}[/bold]"
    if subtitle:
        text += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(text, box=box.ROUNDED, border_style="blue"))


def banner(title: str, icon_key: str = "sparkles") -> None:
    """Print a larger banner for startup messages."""
    icon = _icon(icon_key)
    console.print(f"\n[bold blue]{icon} {title}[/bold blue]\n")


def success(msg: str) -> None:
    """Print a success message."""
    console.print(f"  {_icon('ok')} [green]{msg}[/green]")


def warn(msg: str) -> None:
    """Print a warning message."""
    console.print(f"  {_icon('warn')} [yellow]{msg}[/yellow]")


def error(msg: str) -> None:
    """Print an error message."""
    console.print(f"  {_icon('fail')} [red]{msg}[/red]")


def info(msg: str) -> None:
    """Print an info message."""
    console.print(f"  {_icon('info')} [dim]{msg}[/dim]")


def progress_msg(step: str, status: str) -> None:
    """Print a step progress line."""
    icon = _icon(f"step.{step}") if f"step.{step}" in ICON else _icon("running")
    badge = status_badge(status)
    step_label = step.replace("_", " ").title()
    console.print(f"  {icon} {step_label:<22} {badge}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def search_table(results: list) -> None:
    """Render search results as a Rich table."""
    if not results:
        console.print(f"\n  {_icon('info')} [dim]No results found.[/dim]\n")
        return

    table = Table(
        title=f"\n{_icon('search')} [bold]Search Results[/bold] ({len(results)} found)",
        box=box.ROUNDED,
        border_style="blue",
        show_lines=False,
    )
    table.add_column("Score", style="cyan", width=8, justify="right")
    table.add_column("Type", style="green", width=6)
    table.add_column("Source", style="magenta", width=10)
    table.add_column("Snippet", style="white", max_width=70)

    for r in results:
        snippet = (r.snippet or "")[:90].replace("\n", " ")
        score_str = f"{r.score:.4f}"
        score_color = "green" if r.score > 0.5 else "yellow" if r.score > 0.3 else "dim"
        table.add_row(
            f"[{score_color}]{score_str}[/{score_color}]",
            r.file_type or "",
            r.match_source or "",
            snippet,
        )

    console.print(table)
    console.print()


def file_table(results: list, title: str = "Files") -> None:
    """Render file list as a Rich table."""
    if not results:
        console.print(f"\n  {_icon('info')} [dim]No files found.[/dim]\n")
        return

    table = Table(
        title=f"\n{_icon('list')} [bold]{title}[/bold] ({len(results)} shown)",
        box=box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Filename", style="bold white", max_width=30)
    table.add_column("Type", style="green", width=8)
    table.add_column("Category", style="cyan", width=12)
    table.add_column("Size", style="yellow", width=10, justify="right")
    table.add_column("Added", style="dim", width=12)

    for r in results:
        filename = (r.filename or "")[:29]
        cat_icon = _icon(r.category) if r.category else ""
        cat_display = f"{cat_icon} {r.category}" if cat_icon else (r.category or "")
        added = r.added_at[:10] if r.added_at else ""
        table.add_row(
            filename,
            r.file_type or "",
            cat_display,
            human_size(r.file_size) if r.file_size else "",
            added,
        )

    console.print(table)
    console.print()


def key_table(stats: dict[str, int]) -> None:
    """Render key stats as a Rich table."""
    if not stats:
        console.print(f"\n  {_icon('info')} [dim]No keys registered.[/dim]\n")
        return

    table = Table(
        title=f"\n{_icon('keys')} [bold]Key Registry[/bold] ({len(stats)} keys)",
        box=box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Key Path", style="bold cyan")
    table.add_column("Files", style="green", width=8, justify="right")

    for key_path, count in sorted(stats.items()):
        table.add_row(key_path, str(count))

    console.print(table)
    console.print()


def plugin_table(plugins: list[dict]) -> None:
    """Render registered plugins as a Rich table."""
    if not plugins:
        console.print(f"\n  {_icon('info')} [dim]No plugins registered.[/dim]\n")
        return

    table = Table(
        title=f"\n{_icon('plugin')} [bold]Registered Plugins[/bold]",
        box=box.ROUNDED,
        border_style="blue",
    )
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="bold white")
    table.add_column("Category", style="green")

    for p in plugins:
        table.add_row(
            p.get("type", ""),
            p.get("name", ""),
            p.get("category", p.get("file_type", "")),
        )

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Info panel
# ---------------------------------------------------------------------------

def file_info_panel(file_info, context=None) -> None:
    """Render file info as a Rich panel with sections.

    Parameters
    ----------
    file_info : FileInfo
        File metadata from KB.get_info().
    context : FileContext or None
        File context from KB.get_context() (best-effort).
    """
    cat_icon = _icon(file_info.category) if file_info.category else ""

    # -- Basic info panel --------------------------------------------------
    basic_table = Table(box=None, show_header=False, padding=(0, 1))
    basic_table.add_column("key", style="dim", width=14)
    basic_table.add_column("value", style="white")
    basic_table.add_row("Filename", f"[bold]{file_info.filename}[/bold]")
    basic_table.add_row("ID", file_info.id)
    basic_table.add_row("Type", file_info.file_type)
    basic_table.add_row("Category", f"{cat_icon} {file_info.category}" if cat_icon else file_info.category)
    basic_table.add_row("Size", f"{file_info.file_size:,} bytes")
    basic_table.add_row("Hash", file_info.file_hash[:40] + "..." if len(file_info.file_hash) > 40 else file_info.file_hash)

    console.print(Panel(
        basic_table,
        title=f"[bold]{_icon('info')} File Info[/bold]",
        border_style="blue",
    ))

    # -- Status panel ------------------------------------------------------
    status_table = Table(box=None, show_header=False, padding=(0, 1))
    status_table.add_column("key", style="dim", width=14)
    status_table.add_column("value")
    status_table.add_row("Markdown", status_badge(file_info.md_status.value))
    status_table.add_row("Wiki", status_badge(file_info.wiki_status.value))
    status_table.add_row("Embedding", status_badge(file_info.embed_status.value))

    console.print(Panel(
        status_table,
        title=f"[bold]{_icon('running')} Processing Status[/bold]",
        border_style="blue",
    ))

    # -- Metadata panel ----------------------------------------------------
    meta_table = Table(box=None, show_header=False, padding=(0, 1))
    meta_table.add_column("key", style="dim", width=14)
    meta_table.add_column("value")

    keys_str = ", ".join(file_info.keys) if file_info.keys else "(none)"
    tags_str = ", ".join(file_info.tags) if file_info.tags else "(none)"
    meta_table.add_row(f"{_icon('keys')} Keys", keys_str)
    meta_table.add_row(f"{_icon('tags')} Tags", tags_str)

    if file_info.source_path:
        meta_table.add_row(f"{_icon('source')} Source", file_info.source_path)
    meta_table.add_row(f"{_icon('package')} Raw", file_info.raw_path)
    if file_info.md_path:
        meta_table.add_row(f"{_icon('info')} Markdown", file_info.md_path)
    if file_info.wiki_path:
        meta_table.add_row(f"{_icon('info')} Wiki", file_info.wiki_path)

    console.print(Panel(
        meta_table,
        title=f"[bold]{_icon('tags')} Metadata[/bold]",
        border_style="blue",
    ))

    # -- Timestamps --------------------------------------------------------
    time_table = Table(box=None, show_header=False, padding=(0, 1))
    time_table.add_column("key", style="dim", width=14)
    time_table.add_column("value")
    time_table.add_row(f"{_icon('clock')} Added", file_info.added_at or "")
    time_table.add_row(f"{_icon('clock')} Updated", file_info.updated_at or "")

    console.print(Panel(
        time_table,
        title=f"[bold]{_icon('clock')} Timestamps[/bold]",
        border_style="blue",
    ))

    # -- Content preview ---------------------------------------------------
    if context:
        parts = []
        if context.user_notes:
            parts.append(f"[bold]{_icon('note')} User Notes:[/bold]\n{context.user_notes}")
        if context.md_content:
            preview = context.md_content[:300].replace("\n", " ")
            parts.append(f"[bold]{_icon('info')} Markdown Preview:[/bold]\n{preview}...")
        if context.entities:
            # entities is list[dict], extract name field
            entity_names = []
            for e in context.entities[:15]:
                if isinstance(e, dict):
                    entity_names.append(e.get("name", e.get("id", str(e))))
                else:
                    entity_names.append(str(e))
            entity_str = ", ".join(entity_names)
            parts.append(f"[bold]{_icon('link')} Entities:[/bold]\n{entity_str}")
        if parts:
            console.print(Panel(
                "\n\n".join(parts),
                title=f"[bold]{_icon('info')} Content Preview[/bold]",
                border_style="cyan",
            ))

    console.print()


# ---------------------------------------------------------------------------
# Key tree
# ---------------------------------------------------------------------------

def key_tree(node: dict, stats: dict[str, int] | None = None, path_prefix: str = "") -> RichTree:
    """Render a key hierarchy as a Rich Tree.

    Parameters
    ----------
    node : dict
        Nested key→subtree mapping.
    stats : dict or None
        Per-key file counts.
    path_prefix : str
        Accumulated key path.

    Returns
    -------
    RichTree
    """
    if stats is None:
        stats = {}

    root_label = f"[bold cyan]{_icon('keys')} Key Hierarchy[/bold cyan]"
    tree = RichTree(root_label)

    def _add_branch(parent_tree, subtree, prefix):
        for key_name, children in sorted(subtree.items()):
            full_path = f"{prefix}::{key_name}" if prefix else key_name
            count = stats.get(full_path, 0)
            label = f"[bold]{key_name}[/bold]"
            if count:
                label += f" [dim]({count} files)[/dim]"
            branch = parent_tree.add(label)
            if isinstance(children, dict) and children:
                _add_branch(branch, children, full_path)

    _add_branch(tree, node, path_prefix)
    return tree


# ---------------------------------------------------------------------------
# Init output
# ---------------------------------------------------------------------------

def init_output(results: dict) -> None:
    """Render the init completion output with per-item status.

    Parameters
    ----------
    results : dict
        Keys are item names (``"data_dir"``, ``"config"``, ``"file_categories"``,
        ``"env"``, ``"index"``, ``"vectors"``).  Each value is a dict with::

            path (str)   — absolute path to the item
            status (str) — "created", "exists", or "skipped"
    """
    status_badges = {
        "created": "[green]created[/green]",
        "exists": "[dim]already exists[/dim]",
        "skipped": "[yellow]skipped[/yellow]",
    }

    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column("key", style="dim", width=14)
    table.add_column("value", style="white")
    table.add_column("status", style="white", width=18)

    for item_key, label, icon in [
        ("data_dir", "Data Dir", _icon("package")),
        ("config", "Config", _icon("config")),
        ("file_categories", "File Categories", _icon("config")),
        ("env", "Env", _icon("config")),
        ("index", "Index", _icon("info")),
        ("vectors", "Vectors", _icon("running")),
    ]:
        info = results.get(item_key)
        if info is None:
            continue
        badge = status_badges.get(info.get("status", ""), "")
        table.add_row(f"{icon} {label}", info.get("path", ""), badge)

    console.print(Panel(
        table,
        title=f"[bold green]{_icon('init')} Knowledge Base Initialized![/bold green]",
        border_style="green",
    ))
    console.print(f"  {_icon('rocket')} [dim]Ready to add files with[/dim] [bold]mnemo add[/bold]\n")


# ---------------------------------------------------------------------------
# Add result
# ---------------------------------------------------------------------------

def add_result(result, index: int = 1) -> None:
    """Render a single file add result with detailed processing info."""
    cat_icon = _icon(result.category) if result.category else ""
    cat_display = f"{cat_icon} {result.category}" if cat_icon else result.category

    pd = getattr(result, "processing_detail", {}) or {}

    # -- Basic info ----------------------------------------------------------
    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column("key", style="dim", width=14)
    table.add_column("value")
    table.add_row(f"{_icon('info')} Filename", f"[bold]{result.filename}[/bold]")
    table.add_row(f"{_icon('info')} ID", result.id)
    table.add_row(f"{_icon('info')} Type", result.file_type)
    table.add_row(f"{_icon('info')} Category", cat_display)
    keys_str = ", ".join(result.keys) if result.keys else "(none)"
    table.add_row(f"{_icon('keys')} Keys", keys_str)
    table.add_row(f"{_icon('info')} Markdown", status_badge(result.md_status.value))

    # -- Wiki detail ---------------------------------------------------------
    wiki_pd = pd.get("wiki", {})
    wiki_status = result.wiki_status.value
    wiki_badge = status_badge(wiki_status)
    wiki_parts = [wiki_badge]
    if wiki_pd:
        model = wiki_pd.get("model", "")
        base_url = wiki_pd.get("base_url", "")
        if model:
            wiki_parts.append(f"[dim]model:[/dim] {model}")
        if base_url:
            wiki_parts.append(f"[dim]url:[/dim] {base_url}")
        chars = wiki_pd.get("chars", 0)
        if chars:
            wiki_parts.append(f"[dim]chars:[/dim] {chars}")
        # Token breakdown: show total, or input/output detail if available
        tokens_input = wiki_pd.get("tokens_input", 0)
        tokens_output = wiki_pd.get("tokens_output", 0)
        tokens_total = wiki_pd.get("tokens_used", 0) or (tokens_input + tokens_output)
        if tokens_total:
            if tokens_input or tokens_output:
                wiki_parts.append(
                    f"[dim]tokens:[/dim] {tokens_total} "
                    f"[dim](in:[/dim]{tokens_input} "
                    f"[dim]out:[/dim]{tokens_output}[dim])[/dim]"
                )
            else:
                wiki_parts.append(f"[dim]tokens:[/dim] {tokens_total}")
        error_msg = wiki_pd.get("error", "")
        if error_msg and wiki_status == "failed":
            wiki_parts.append("[red]✗[/red]")
        elif error_msg and wiki_status == "pending":
            wiki_parts.append(f"[dim]({error_msg})[/dim]")
    table.add_row(f"{_icon('info')} Wiki", "  ".join(wiki_parts))

    # -- Embedding detail ----------------------------------------------------
    embed_pd = pd.get("embedding", {})
    embed_status = result.embed_status.value
    embed_badge = status_badge(embed_status)
    embed_parts = [embed_badge]
    if embed_pd:
        model = embed_pd.get("model", "")
        dimension = embed_pd.get("dimension", 0)
        base_url = embed_pd.get("base_url", "")
        chunks = embed_pd.get("chunks", 0)
        if model:
            embed_parts.append(f"[dim]model:[/dim] {model}")
        if base_url:
            embed_parts.append(f"[dim]url:[/dim] {base_url}")
        if dimension:
            embed_parts.append(f"[dim]dim:[/dim] {dimension}")
        if chunks:
            embed_parts.append(f"[dim]chunks:[/dim] {chunks}")
        error_msg = embed_pd.get("error", "")
        if error_msg and embed_status == "failed":
            embed_parts.append("[red]✗[/red]")
    table.add_row(f"{_icon('info')} Embedding", "  ".join(embed_parts))

    # -- Full error display (below the table, no truncation) ------------------
    wiki_error = (pd.get("wiki", {}).get("error", "")
                  if result.wiki_status.value == "failed" else "")
    embed_error = (pd.get("embedding", {}).get("error", "")
                   if result.embed_status.value == "failed" else "")

    if wiki_error:
        console.print(f"  [red]Wiki error:[/red] {wiki_error}")
    if embed_error:
        console.print(f"  [red]Embedding error:[/red] {embed_error}")

    console.print(Panel(
        table,
        title=f"[bold green]{_icon('add')} Added #{index}[/bold green]",
        border_style="green",
    ))
    console.print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summary(added: int = 0, errors: int = 0, skipped: int = 0, reindexed: int = 0,
            failed: int = 0, total: int = 0, label: str = "operation") -> None:
    """Print a summary line with counts."""
    parts = []
    if added:
        parts.append(f"[green]{added} added[/green]")
    if skipped:
        parts.append(f"[dim]{skipped} skipped[/dim]")
    if errors:
        parts.append(f"[red]{errors} errors[/red]")
    if reindexed:
        parts.append(f"[green]{reindexed} reindexed[/green]")
    if failed:
        parts.append(f"[red]{failed} failed[/red]")
    if total:
        parts.append(f"{total} total")

    console.print(f"\n  {_icon('sparkles')} {label}: {' | '.join(parts)}\n")


# ---------------------------------------------------------------------------
# Check report
# ---------------------------------------------------------------------------

def check_report(report) -> None:
    """Render KB check/consistency report."""
    status = report.status
    color = status_color(status)
    issues = getattr(report, "issues", [])
    suggestions = getattr(report, "suggestions", [])

    console.print(Panel(
        f"[{color}]{status_badge(status)}[/{color}]",
        title=f"[bold]{_icon('check')} Integrity Check[/bold]",
        border_style=color,
    ))

    if issues:
        console.print(f"\n  {_icon('warn')} [bold yellow]Issues:[/bold yellow]")
        for i, issue in enumerate(issues, 1):
            console.print(f"    {i}. [dim]{issue}[/dim]")
        console.print()

    if suggestions:
        console.print(f"  {_icon('info')} [dim]Suggestions:[/dim]")
        for s in suggestions:
            console.print(f"    • {s}")
        console.print()

    if not issues:
        console.print(f"\n  {_icon('ok')} [green]All files are consistent.[/green]\n")
