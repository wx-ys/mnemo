"""mnemo db command — inspect vector DB and graph DB internals."""

from __future__ import annotations

import rich_click as click
from rich.panel import Panel

from mnemo.cli.formatter import console


def run(ctx: click.Context):
    """Show detailed vector DB and graph DB information for debugging."""
    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    api = MnemoAPI(data_dir if data_dir else "~/mnemo-data")
    kb = api.kb

    # ── Vector DB ────────────────────────────────────────────────────────
    console.print()
    console.print(Panel("[bold]🧮 Vector DB (LanceDB)[/bold]", border_style="cyan"))

    try:
        from mnemo.core.plugin_base import PluginHub
        from mnemo.core.interfaces import IVectorStore
        vs = PluginHub.get(IVectorStore, "lancedb")
        if hasattr(vs, '_db'):
            table_names = vs._db.table_names()
            console.print(f"  Tables: {', '.join(table_names) if table_names else '(none)'}")
            console.print()

            for tbl_name in sorted(table_names):
                try:
                    tbl = vs._db.open_table(tbl_name)
                    rows = tbl.count_rows()
                    schema = tbl.schema
                    console.print(f"  [bold]{tbl_name}[/bold]: {rows} rows")
                    # Show column names
                    col_names = [f.name for f in schema]
                    console.print(f"    Columns: {', '.join(col_names[:10])}")
                    # Show a sample row
                    try:
                        sample = tbl.to_lance().head(1)
                        if sample.num_rows > 0:
                            console.print(f"    Sample: id={sample.column('id')[0].as_py()}")
                    except Exception:
                        pass
                    console.print()
                except Exception as e:
                    console.print(f"  [bold]{tbl_name}[/bold]: [red]error: {e}[/red]")
                    console.print()
    except Exception as e:
        console.print(f"  [red]Vector DB unavailable: {e}[/red]")

    # ── Graph DB ─────────────────────────────────────────────────────────
    console.print(Panel("[bold]🕸️  Graph DB (SQLite)[/bold]", border_style="cyan"))

    try:
        from mnemo.core.plugin_base import PluginHub
        from mnemo.core.interfaces import IGraphStore
        gs = PluginHub.get(IGraphStore, "sqlite")
        if hasattr(gs, '_get_conn'):
            conn = gs._get_conn()
            # List tables
            tbl_rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            ).fetchall()
            console.print(f"  Tables: {', '.join(r[0] for r in tbl_rows) if tbl_rows else '(none)'}")
            console.print()

            # Entity count
            try:
                ec = conn.execute("SELECT COUNT(*) FROM graph_entities").fetchone()[0]
                rc = conn.execute("SELECT COUNT(*) FROM graph_relations").fetchone()[0]
                console.print(f"  [bold]entities[/bold]: {ec} rows")
                console.print(f"  [bold]relations[/bold]: {rc} rows")

                # Sample entities
                if ec > 0:
                    samples = conn.execute(
                        "SELECT name, type FROM graph_entities LIMIT 10",
                    ).fetchall()
                    console.print("    Sample entities:")
                    for s in samples:
                        console.print(f"      - {s['name']} [{s['type']}]")
                console.print()
            except Exception as e:
                console.print(f"  [red]error: {e}[/red]")
    except Exception as e:
        console.print(f"  [red]Graph DB unavailable: {e}[/red]")

    # ── SQLite Index ─────────────────────────────────────────────────────
    console.print(Panel("[bold]🗄️  SQLite Index[/bold]", border_style="cyan"))

    try:
        from mnemo.core.plugin_base import PluginHub
        from mnemo.core.interfaces import IIndexer
        idx = PluginHub.get(IIndexer, "sqlite")
        if hasattr(idx, '_get_conn'):
            conn = idx._get_conn()
            # File counts
            total = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM files WHERE deleted_at = ''",
            ).fetchone()[0]
            deleted = conn.execute(
                "SELECT COUNT(*) FROM files WHERE deleted_at != ''",
            ).fetchone()[0]
            key_count = conn.execute(
                "SELECT COUNT(*) FROM file_keys",
            ).fetchone()[0]

            console.print(f"  Files: [bold]{active}[/bold] active + {deleted} deleted = {total} total")
            console.print(f"  File-key mappings: {key_count}")
            console.print()

            # Status breakdown
            if active > 0:
                md = conn.execute(
                    "SELECT md_status, COUNT(*) FROM files WHERE deleted_at = '' GROUP BY md_status",
                ).fetchall()
                wiki = conn.execute(
                    "SELECT wiki_status, COUNT(*) FROM files WHERE deleted_at = '' GROUP BY wiki_status",
                ).fetchall()
                embed = conn.execute(
                    "SELECT embed_status, COUNT(*) FROM files WHERE deleted_at = '' GROUP BY embed_status",
                ).fetchall()
                console.print("  Status breakdown:")
                console.print(f"    md:      {', '.join(f'{r[0]}={r[1]}' for r in md)}")
                console.print(f"    wiki:    {', '.join(f'{r[0]}={r[1]}' for r in wiki)}")
                console.print(f"    embed:   {', '.join(f'{r[0]}={r[1]}' for r in embed)}")
            console.print()
    except Exception as e:
        console.print(f"  [red]Index DB unavailable: {e}[/red]")

    console.print()
