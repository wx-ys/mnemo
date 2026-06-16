"""mnemo status command — show knowledge base overview."""

from __future__ import annotations

import rich_click as click
from rich.panel import Panel
from rich.table import Table

from mnemo.cli.formatter import console, human_size


def run(ctx: click.Context):
    """Show knowledge base location, size, and statistics."""
    from mnemo.api import MnemoAPI

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb = MnemoAPI(data_dir if data_dir else "~/mnemo-data")
    # Access internal KnowledgeBase for indexer/raw stats access
    _kb = kb.kb

    # Gather all data first (this is fast — just DB queries)
    stats = _kb.indexer.get_stats()
    total_files = stats.get("total_files", 0)
    total_size = stats.get("total_size", 0)

    all_files = _kb.indexer.list_files(limit=10000)

    by_category: dict[str, dict] = {}  # cat -> {count, size}
    by_type: dict[str, dict] = {}      # type -> {count, size}
    md_done = embed_done = wiki_done = 0

    for f in all_files:
        cat = f.category or "other"
        if cat not in by_category:
            by_category[cat] = {"count": 0, "size": 0}
        by_category[cat]["count"] += 1
        by_category[cat]["size"] += (f.file_size or 0)

        ft = f.file_type or "?"
        if ft not in by_type:
            by_type[ft] = {"count": 0, "size": 0}
        by_type[ft]["count"] += 1
        by_type[ft]["size"] += (f.file_size or 0)

        if f.md_status == "done":
            md_done += 1
        if f.embed_status == "done":
            embed_done += 1
        if f.wiki_status == "done":
            wiki_done += 1

    # Gather embedding config (global singleton, no plugin system)
    import os
    from mnemo.core.param_config import get_global_config

    emb_cfg = _kb.config.get("embedder", {})
    if not isinstance(emb_cfg, dict):
        emb_cfg = {}
    embed_model = emb_cfg.get("model", "?")
    embed_dim = str(get_global_config().get("dimension", "?"))
    embed_base_url = emb_cfg.get("base_url", "") or ""
    embed_api_key = emb_cfg.get("api_key", "") or os.environ.get("DASHSCOPE_API_KEY", "")
    embed_api_key_ok = bool(embed_api_key and len(embed_api_key) > 5)

    # Gather LLM config via AgentManager (config-driven)
    from mnemo.core.agent_manager import AgentManager
    from mnemo.core.param_config import resolve_agent_config

    am = AgentManager.get_instance()
    llm_active = am.default_agent_name if am._initialized else "default"
    llm_model = "?"
    llm_base_url = ""
    llm_api_key_ok = False
    try:
        if am._initialized:
            agent_cfg = resolve_agent_config()
            llm_model = agent_cfg.get("model", "?")
            llm_base_url = agent_cfg.get("base_url", "") or ""
            llm_api_key = agent_cfg.get("api_key", "")
            llm_api_key_ok = bool(llm_api_key and len(llm_api_key) > 5)
    except Exception:
        pass

    # Estimate DB sizes
    index_size = 0
    index_path = _kb.data_dir / ".mnemo" / "index.db"
    if index_path.exists():
        index_size = index_path.stat().st_size

    vector_size = 0
    vector_dir = _kb.data_dir / "embedding"
    if vector_dir.exists():
        for f in vector_dir.rglob("*"):
            if f.is_file():
                vector_size += f.stat().st_size

    graph_size = 0
    graph_path = _kb.data_dir / ".mnemo" / "graph.db"
    if graph_path.exists():
        graph_size = graph_path.stat().st_size

    # -- Render ---------------------------------------------------------------
    def pct_str(done: int) -> str:
        if total_files == 0:
            return "—"
        pct = done * 100 // total_files
        color = "green" if pct == 100 else "yellow" if pct > 0 else "dim"
        return f"[{color}]{done}/{total_files} ({pct}%)[/{color}]"

    # Overview
    overview = Table(box=None, show_header=False, padding=(0, 1))
    overview.add_column("k", style="dim", width=18)
    overview.add_column("v", style="white")
    overview.add_row("📁 Data Directory", str(_kb.data_dir))
    overview.add_row("📊 Total Files", f"[bold]{total_files}[/bold]")
    overview.add_row("💾 Total Size", human_size(total_size) if total_size else "N/A")
    overview.add_row("🗄️  Index DB", f"{human_size(index_size)}" if index_size else "N/A")
    overview.add_row("🧮 Vector DB", f"{human_size(vector_size)}" if vector_size else "N/A")
    overview.add_row("🕸️  Graph DB", f"{human_size(graph_size)}" if graph_size else "N/A")

    console.print(Panel(overview, title="[bold]📋 Knowledge Base Status[/bold]", border_style="blue"))

    # Processing status with sizes
    status_table = Table(box=None, show_header=False, padding=(0, 1))
    status_table.add_column("k", style="dim", width=18)
    status_table.add_column("v")
    status_table.add_row("📝 Markdown", pct_str(md_done))
    status_table.add_row("🤖 Wiki (LLM)", pct_str(wiki_done))
    status_table.add_row("🧮 Embedding", pct_str(embed_done))

    console.print(Panel(status_table, title="[bold]Processing Status[/bold]", border_style="blue"))

    # Embedding info (global singleton, no plugin system)
    embed_table = Table(box=None, show_header=False, padding=(0, 1))
    embed_table.add_column("k", style="dim", width=18)
    embed_table.add_column("v")
    if embed_base_url:
        url_display = str(embed_base_url)[:55]
        embed_table.add_row("🌐 Base URL", url_display)
    embed_table.add_row("🧠 Model", str(embed_model))
    embed_table.add_row("📐 Dimension", str(embed_dim))
    embed_table.add_row("🔑 API Key", "[green]configured[/green]" if embed_api_key_ok else "[red]not set[/red]")
    embed_table.add_row("💾 DB Size", human_size(vector_size) if vector_size else "N/A")

    console.print(Panel(embed_table, title="[bold]🧮 Embedding[/bold]", border_style="cyan"))

    # LLM info (agent-driven, no plugin system)
    llm_table = Table(box=None, show_header=False, padding=(0, 1))
    llm_table.add_column("k", style="dim", width=18)
    llm_table.add_column("v")
    llm_table.add_row("🤖 Agent", f"[bold]{llm_active}[/bold]")
    if llm_base_url:
        url_display = str(llm_base_url)[:55]
        llm_table.add_row("🌐 Base URL", url_display)
    llm_table.add_row("🧠 Model", str(llm_model))
    llm_table.add_row("🔑 API Key", "[green]configured[/green]" if llm_api_key_ok else "[red]not set[/red]")

    console.print(Panel(llm_table, title="[bold]🤖 LLM[/bold]", border_style="cyan"))

    # -- Vector DB info --------------------------------------------------------
    vec_table = Table(box=None, show_header=False, padding=(0, 1))
    vec_table.add_column("k", style="dim", width=24)
    vec_table.add_column("v")
    try:
        from mnemo.core.plugin_base import PluginHub
        from mnemo.core.interfaces import IVectorStore
        vs = PluginHub.get(IVectorStore, "lancedb")
        if hasattr(vs, 'table_info'):
            for tbl_name in ("raw_md", "raw_wiki", "raw_metadata", "raw_md_parents"):
                try:
                    info = vs.table_info(tbl_name)
                    if info.get("num_rows", 0) > 0:
                        vec_table.add_row(
                            f"📊 {tbl_name}",
                            f"[bold]{info['num_rows']}[/bold] rows · {info.get('num_small_files', '?')} files",
                        )
                except Exception:
                    pass
    except Exception:
        pass
    if vec_table.rows:
        vec_table.add_row("💾 DB Size", human_size(vector_size) if vector_size else "N/A")
        console.print(Panel(vec_table, title="[bold]🧮 Vector DB[/bold]", border_style="cyan"))

    # -- Graph DB info ---------------------------------------------------------
    graph_table = Table(box=None, show_header=False, padding=(0, 1))
    graph_table.add_column("k", style="dim", width=24)
    graph_table.add_column("v")
    try:
        from mnemo.core.plugin_base import PluginHub
        from mnemo.core.interfaces import IGraphStore
        gs = PluginHub.get(IGraphStore, "sqlite")
        if hasattr(gs, '_get_conn'):
            conn = gs._get_conn()
            entity_count = conn.execute(
                "SELECT COUNT(*) FROM graph_entities",
            ).fetchone()[0]
            relation_count = conn.execute(
                "SELECT COUNT(*) FROM graph_relations",
            ).fetchone()[0]
            graph_table.add_row("🕸️  Entities", str(entity_count))
            graph_table.add_row("🔗 Relations", str(relation_count))
        elif hasattr(gs, 'get_stats'):
            gs_stats = gs.get_stats()
            graph_table.add_row("🕸️  Entities", str(gs_stats.get("entity_count", "?")))
            graph_table.add_row("🔗 Relations", str(gs_stats.get("relation_count", "?")))
    except Exception:
        pass
    if graph_table.rows:
        graph_table.add_row("💾 DB Size", human_size(graph_size) if graph_size else "N/A")
        console.print(Panel(graph_table, title="[bold]🕸️  Graph DB[/bold]", border_style="cyan"))

    # By category with sizes
    if by_category:
        cat_table = Table(box=None, show_header=False, padding=(0, 1))
        cat_table.add_column("k", style="dim", width=18)
        cat_table.add_column("v")
        icon_map = {
            "docs": "📝", "data": "📊", "code": "💻",
            "img": "🖼", "audio": "🎵", "video": "🎬",
            "web": "🌐", "other": "📎",
        }
        for cat, info in sorted(by_category.items(), key=lambda x: -x[1]["count"]):
            icon = icon_map.get(cat, "•")
            count = info["count"]
            size = info["size"]
            cat_table.add_row(
                f"{icon} {cat}",
                f"[bold]{count}[/bold] files · {human_size(size)}"
            )

        console.print(Panel(cat_table, title="[bold]By Category[/bold]", border_style="blue"))

    # By type
    if len(by_type) > 1:
        type_str = " · ".join(
            f"[bold]{t}[/bold]:{info['count']}"
            for t, info in sorted(by_type.items(), key=lambda x: -x[1]["count"])[:15]
        )
        console.print(Panel(type_str, title="[bold]By Type[/bold]", border_style="cyan"))

    console.print()
