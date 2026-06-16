"""
Mnemo CLI entry point.

Built with Click. All subcommands use lazy loading for fast startup.
"""

from __future__ import annotations

from pathlib import Path

import rich_click as click

from mnemo import __version__

# Configure rich-click
click.rich_click.TEXT_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS_HELP_ORDER = True
click.rich_click.STYLE_OPTION = "bold yellow"
click.rich_click.STYLE_SWITCH = "bold green"
click.rich_click.STYLE_ARGUMENT = "bold cyan"
click.rich_click.STYLE_COMMAND = "bold cyan"
click.rich_click.STYLE_METAVAR = "dim yellow"
click.rich_click.STYLE_HEADER = "bold blue"
click.rich_click.STYLE_USAGE = "dim"


def _resolve_data_dir(explicit: str | None) -> str:
    """Resolve the data directory from explicit flag, global config, or default."""
    if explicit:
        return explicit

    # Try global config
    global_config = Path.home() / ".config" / "mnemo" / "config.toml"
    if global_config.exists():
        import tomllib
        with open(global_config, "rb") as f:
            cfg = tomllib.load(f)
        default = cfg.get("global", {}).get("default_data_dir", "")
        if default and Path(default).exists():
            return default

    return str(Path.home() / "mnemo-data")


@click.group()
@click.version_option(__version__, prog_name="mnemo")
@click.option(
    "--data-dir", "-d",
    default=None,
    help="Knowledge base data directory (default: auto-detect or ~/mnemo-data)",
    type=click.Path(),
)
@click.pass_context
def main(ctx: click.Context, data_dir: str | None):
    """
    Mnemo — AI-agent-friendly personal knowledge base.

    Manage information carriers: ingest, classify, LLM wiki,
    embed, and search.
    """
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = _resolve_data_dir(data_dir)


# ──────────────────────────────────────────────────────────────────────
# Subcommand registration (lazy loading)
# ──────────────────────────────────────────────────────────────────────

@main.command("init")
@click.argument("directory", type=click.Path(), required=False)
@click.option("--force", is_flag=True, help="Force initialization (overwrite existing config)")
@click.pass_context
def init_cmd(ctx: click.Context, directory: str | None, force: bool):
    """Initialize a Mnemo knowledge base directory"""
    from mnemo.cli.commands.init_cmd import run
    run(ctx, directory, force)


@main.command("add")
@click.argument("source", type=click.Path(exists=True), required=False)
@click.option("--file", "-f", "file_path", type=click.Path(exists=True), help="File to add")
@click.option("--dir", "dir_path", type=click.Path(exists=True), help="Directory to add (batch)")
@click.option("--url", help="Add from URL")
@click.option("--move", is_flag=True, help="Move instead of copy")
@click.option("--keys", "-k", help="Hierarchical keys (comma-separated)")
@click.option("--tags", "-t", help="Flat tags (comma-separated)")
@click.option("--note", help="Initial note")
@click.option("--no-md", is_flag=True, help="Skip markdown conversion")
@click.option("--no-wiki", is_flag=True, help="Skip wiki generation")
@click.option("--no-embed", is_flag=True, help="Skip embedding generation")
@click.option("--overwrite", is_flag=True, help="Overwrite duplicate files")
@click.option("--diagnose", is_flag=True, help="Write detailed pipeline diagnostics to .mnemo/diagnostics/")
@click.option("--verbose", "-v", is_flag=True, help="Print diagnostic summaries to terminal (with --diagnose)")
@click.pass_context
def add_cmd(ctx: click.Context, source, file_path, dir_path, url, move, keys, tags, note,
            no_md, no_wiki, no_embed, overwrite, diagnose, verbose):
    """Add an information carrier to the knowledge base

    SOURCE can be a file or directory path. If omitted, use --file, --dir, or --url.
    """
    from mnemo.cli.commands.add import run
    run(ctx, source, file_path, dir_path, url, move, keys, tags, note,
        no_md, no_wiki, no_embed, overwrite, diagnose, verbose)


@main.command("search")
@click.argument("query")
@click.option("--mode", default="hybrid", type=click.Choice(["hybrid", "vector", "keyword"]))
@click.option("--keys", "-k", help="Limit to key scope")
@click.option("--type", "file_type", help="Limit to file type")
@click.option("--limit", "-n", default=10, type=int)
@click.option("--with-meta/--no-meta", default=True)
@click.option("--format", "output_format", default="table",
              type=click.Choice(["table", "json", "csv"]))
@click.option("--expand-chunks/--no-expand", default=False)
@click.option("--output", "-o", type=click.Path(), help="Save results to file")
@click.option("--diagnose", is_flag=True, help="Write detailed pipeline diagnostics to .mnemo/diagnostics/")
@click.option("--verbose", "-v", is_flag=True, help="Print diagnostic summaries to terminal (with --diagnose)")
@click.pass_context
def search_cmd(ctx, query, mode, keys, file_type, limit, with_meta,
               output_format, expand_chunks, output, diagnose, verbose):
    """Search the knowledge base"""
    from mnemo.cli.commands.search import run
    run(ctx, query, mode, keys, file_type, limit, with_meta, output_format,
        expand_chunks, output, diagnose, verbose)


@main.command("list")
@click.option("--type", "file_type", help="Filter by file type")
@click.option("--tags", help="Filter by tags")
@click.option("--keys", help="Filter by keys")
@click.option("--sort-by", default="added_at",
              type=click.Choice(["added_at", "updated_at", "file_type",
                                 "filename", "file_size", "id"]),
              help="Sort column")
@click.option("--limit", "-n", default=50, type=int)
@click.option("--offset", default=0, type=int)
@click.option("--format", "output_format", default="table",
              type=click.Choice(["table", "json", "csv"]))
@click.pass_context
def list_cmd(ctx, file_type, tags, keys, sort_by, limit, offset, output_format):
    """List files in the knowledge base"""
    from mnemo.cli.commands.list_cmd import run
    run(ctx, file_type, tags, keys, sort_by, limit, offset, output_format)


@main.command("info")
@click.argument("file_ref")
@click.pass_context
def info_cmd(ctx, file_ref):
    """Show detailed file information (by ID or filename)"""
    from mnemo.cli.commands.info import run
    run(ctx, file_ref)


@main.command("update")
@click.argument("file_id")
@click.option("--keys", "-k", help="Set keys (replace all)")
@click.option("--add-keys", help="Add keys")
@click.option("--remove-keys", help="Remove keys")
@click.option("--tags", "-t", help="Set tags")
@click.option("--note", help="Update note")
@click.pass_context
def update_cmd(ctx, file_id, keys, add_keys, remove_keys, tags, note):
    """Update file information"""
    from mnemo.cli.commands.update import run
    run(ctx, file_id, keys, add_keys, remove_keys, tags, note)


@main.command("reindex")
@click.option("--file", "file_id", help="Reindex a specific file")
@click.option("--type", "file_type", help="Reindex by type")
@click.option("--all", "all_files", is_flag=True, help="Full index rebuild")
@click.option("--meta-only", is_flag=True, help="Only rebuild metadata vector store")
@click.pass_context
def reindex_cmd(ctx, file_id, file_type, all_files, meta_only):
    """Rebuild index / embeddings"""
    from mnemo.cli.commands.reindex import run
    run(ctx, file_id, file_type, all_files, meta_only)


@main.command("reorg")
@click.option("--type", "file_type", help="Restrict reorganization to this type")
@click.option("--dry-run", is_flag=True, help="Preview migration plan")
@click.option("--confirm", is_flag=True, help="Confirm execution")
@click.pass_context
def reorg_cmd(ctx, file_type, dry_run, confirm):
    """Reorganize files by new chunk strategy"""
    from mnemo.cli.commands.reorg import run
    run(ctx, file_type, dry_run, confirm)


@main.command("remove")
@click.argument("file_id")
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.pass_context
def remove_cmd(ctx, file_id, force):
    """Remove (soft-delete) a file from the knowledge base"""
    from mnemo.cli.commands.remove import run
    run(ctx, file_id, force)


@main.command("watch")
@click.option("--interval", "-i", default=30, type=int, help="Polling interval (seconds)")
@click.pass_context
def watch_cmd(ctx, interval):
    """Start the file watcher daemon"""
    from mnemo.cli.commands.watch import run
    run(ctx, interval)


@main.command("status")
@click.pass_context
def status_cmd(ctx):
    """Show knowledge base status (location, size, counts)"""
    from mnemo.cli.commands.status_cmd import run
    run(ctx)


@main.command("check")
@click.option("--fix", is_flag=True, help="Auto-repair issues")
@click.pass_context
def check_cmd(ctx, fix):
    """Check knowledge base integrity"""
    from mnemo.cli.commands.check import run
    run(ctx, fix)


@main.command("config")
@click.argument("action", type=click.Choice(["show", "get", "set", "example"]))
@click.argument("key", required=False)
@click.argument("value", required=False)
@click.option("--output", "-o", type=click.Path(), help="Write example to file")
@click.option("--file-categories", "-f", is_flag=True,
              help="Operate on file_categories.toml instead of config.toml")
@click.pass_context
def config_cmd(ctx, action, key, value, output, file_categories):
    """Show / modify / export configuration"""
    from mnemo.cli.commands.config import run
    run(ctx, action, key, value, output, file_categories)


# ── Key management ───────────────────────────────────────────────────
@main.group("key")
def key_group():
    """Manage hierarchical keys"""
    pass


@key_group.command("list")
@click.pass_context
def key_list(ctx):
    from mnemo.cli.commands.key import run_list
    run_list(ctx)


@key_group.command("add")
@click.argument("key_path")
@click.option("--description", "-d", help="Key description")
@click.pass_context
def key_add(ctx, key_path, description):
    from mnemo.cli.commands.key import run_add
    run_add(ctx, key_path, description)


@key_group.command("remove")
@click.argument("key_path")
@click.pass_context
def key_remove(ctx, key_path):
    from mnemo.cli.commands.key import run_remove
    run_remove(ctx, key_path)


@key_group.command("tree")
@click.argument("root_key", required=False)
@click.pass_context
def key_tree(ctx, root_key):
    from mnemo.cli.commands.key import run_tree
    run_tree(ctx, root_key)


# ── Plugin management ────────────────────────────────────────────────
@main.group("plugin")
def plugin_group():
    """Manage parser and template plugins"""
    pass


@plugin_group.command("list")
@click.pass_context
def plugin_list(ctx):
    from mnemo.cli.commands.plugin import run_list
    run_list(ctx)


# ── Trash management ─────────────────────────────────────────────────
@main.group("trash")
def trash_group():
    """Manage trash / recycle bin"""
    pass


@trash_group.command("list")
@click.pass_context
def trash_list(ctx):
    from mnemo.cli.commands.trash import run_list
    run_list(ctx)


@trash_group.command("restore")
@click.argument("file_id")
@click.pass_context
def trash_restore(ctx, file_id):
    from mnemo.cli.commands.trash import run_restore
    run_restore(ctx, file_id)


@trash_group.command("clean")
@click.option("--force", is_flag=True)
@click.pass_context
def trash_clean(ctx, force):
    from mnemo.cli.commands.trash import run_clean
    run_clean(ctx, force)


# ── Import / Export / Sync ───────────────────────────────────────────
@main.command("export")
@click.argument("dest", type=click.Path())
@click.option("--type", "file_type", help="Export by type")
@click.option("--keys", help="Export by key scope")
@click.option("--after", help="Export files added after this date")
@click.pass_context
def export_cmd(ctx, dest, file_type, keys, after):
    """Export the knowledge base"""
    from mnemo.cli.commands.export_cmd import run
    run(ctx, dest, file_type, keys, after)


@main.command("import")
@click.argument("source", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Preview import without changes")
@click.pass_context
def import_cmd(ctx, source, dry_run):
    """Import an external knowledge base"""
    from mnemo.cli.commands.import_cmd import run
    run(ctx, source, dry_run)


@main.group("sync")
def sync_group():
    """Remote synchronization"""
    pass


@sync_group.command("push")
@click.pass_context
def sync_push(ctx):
    from mnemo.cli.commands.sync import run_push
    run_push(ctx)


@sync_group.command("pull")
@click.pass_context
def sync_pull(ctx):
    from mnemo.cli.commands.sync import run_pull
    run_pull(ctx)


@sync_group.command("status")
@click.pass_context
def sync_status(ctx):
    from mnemo.cli.commands.sync import run_status
    run_status(ctx)


# ── DB Debug ───────────────────────────────────────────────────────────
@main.command("db")
@click.pass_context
def db_cmd(ctx: click.Context):
    """Show vector DB and graph DB internals (debug)"""
    from mnemo.cli.commands.db_cmd import run
    run(ctx)


# ── RAG Ask ────────────────────────────────────────────────────────────
@main.command("ask")
@click.argument("question")
@click.option("--grounded/--no-grounded", default=True,
              help="Strictly ground answer in KB content (default: yes)")
@click.option("--limit", "-n", default=10, type=int,
              help="Maximum source chunks (default: 10)")
@click.pass_context
def ask_cmd(ctx: click.Context, question: str, grounded: bool, limit: int):
    """Ask a question and get a KB-grounded answer with citations

    The RAG pipeline: query expansion → multi-source search →
    rerank → context assembly → LLM answer with [N] citations.

    \b
    Example:
        mnemo ask "What is the key contribution of this paper?"
        mnemo ask --no-grounded "Tell me about attention mechanisms"
    """
    from mnemo.cli.commands.ask import run
    run(ctx, question, grounded, limit)


# ── MCP Server ─────────────────────────────────────────────────────────
@main.command("mcp")
@click.option("--data-dir", "-d", default=None,
              help="Knowledge base data directory")
@click.pass_context
def mcp_cmd(ctx: click.Context, data_dir: str | None):
    """Start MCP server (stdio) for AI agent integration

    Add this to your Claude Desktop config to use Mnemo:

    \b
    {
      "mcpServers": {
        "mnemo": {
          "command": "uv",
          "args": ["run", "mnemo", "mcp", "-d", "/path/to/kb"]
        }
      }
    }
    """
    import asyncio

    from mnemo.api.mcp_server import run
    resolved = data_dir or (ctx.obj.get("data_dir") if ctx.obj else None)
    asyncio.run(run(resolved))


# ── REST API Server ────────────────────────────────────────────────────
@main.command("serve")
@click.option("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
@click.option("--port", default=8765, type=int, help="Bind port (default: 8765)")
@click.option("--reload", is_flag=True, help="Enable auto-reload (development)")
@click.option("--data-dir", "-d", default=None,
              help="Knowledge base data directory")
@click.pass_context
def serve_cmd(ctx: click.Context, host: str, port: int, reload: bool,
              data_dir: str | None):
    """Start REST API server (FastAPI + OpenAPI docs)"""
    resolved = data_dir or (ctx.obj.get("data_dir") if ctx.obj else None)
    import os

    import uvicorn
    if resolved:
        os.environ["MNEMO_DATA_DIR"] = resolved
    uvicorn.run(
        "mnemo.api.server:app",
        host=host, port=port, reload=reload,
    )


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
