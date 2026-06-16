"""mnemo init command — initialize a knowledge base directory.

Auto-generates ``config.toml`` from registered plugin ``config_schema``
declarations via :func:`mnemo.core.param_config.generate_toml_template`.
Static sections are merged with auto-generated plugin sections.
"""

from __future__ import annotations

from pathlib import Path

import rich_click as click

from mnemo.cli.formatter import (
    _icon,
    banner,
    console,
    init_output,
    warn,
)

# ── Static config sections (not plugin-derived) ──────────────────────────

_STATIC_CONFIG: dict[str, dict] = {
    "global": {
        k: p.default
        for k, p in __import__(
            "mnemo.core.param_config", fromlist=["GLOBAL_CONFIG_SCHEMA"],
        ).GLOBAL_CONFIG_SCHEMA.items()
    },
}


def _save_global_default(target: Path) -> None:
    """Persist the KB path as the default in the global config.

    Writes or updates ``~/.config/mnemo/config.toml`` so that future
    ``mnemo`` invocations without ``-d`` will use this KB.
    """
    global_config_dir = Path.home() / ".config" / "mnemo"
    global_config_dir.mkdir(parents=True, exist_ok=True)
    config_path = global_config_dir / "config.toml"

    # Read existing if present
    config: dict = {}
    if config_path.exists():
        import tomllib
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

    config.setdefault("global", {})["default_data_dir"] = str(target)

    import tomli_w
    with open(config_path, "wb") as f:
        tomli_w.dump(config, f)


def run(ctx: click.Context, directory: str | None, force: bool):
    """Create the full directory structure, default config, and empty index."""
    target = Path(directory).expanduser().resolve() if directory else Path.home() / "mnemo-data"

    # Track per-item status for final summary
    results: dict[str, dict] = {}

    # -- Data dir ------------------------------------------------------------
    data_dir_existed = target.exists()
    target.mkdir(parents=True, exist_ok=True)
    results["data_dir"] = {
        "path": str(target),
        "status": "exists" if data_dir_existed else "created",
    }

    if data_dir_existed:
        console.print(f"  {_icon('package')} [dim]Data directory already exists:[/dim] {target}")
    else:
        console.print(f"  {_icon('package')} [green]Created data directory:[/green] {target}")

    banner(f"Initializing Mnemo knowledge base at {target}")

    # -- Bootstrap plugins first (so we can read their schemas) -------------
    from mnemo.core.kb import _bootstrap_builtins
    _bootstrap_builtins()

    # -- Create directory structure from file categories --------------------
    from mnemo.core.plugin_base import PluginHub
    from mnemo.core.interfaces import IFileCategory

    created_dirs = 0
    existing_dirs = 0
    dirs: list[Path] = [
        target / "plugins",
        target / ".mnemo" / "logs",
        target / ".mnemo" / "transactions",
        target / ".mnemo" / "trash",
        target / "embedding",
    ]
    for fc_name, impl_cls in PluginHub.iter_impls(IFileCategory):
        inst = PluginHub.get(IFileCategory, fc_name)
        dir_path = getattr(inst, "dir_path", fc_name)
        dirs += [
            target / "raw" / dir_path,
            target / "raw_md" / dir_path,
            target / "raw_wiki" / dir_path,
            target / "raw_metadata" / dir_path,
        ]
    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created_dirs += 1
        else:
            existing_dirs += 1

    if created_dirs or existing_dirs:
        parts = []
        if created_dirs:
            parts.append(f"[green]{created_dirs} created[/green]")
        if existing_dirs:
            parts.append(f"[dim]{existing_dirs} already exist[/dim]")
        console.print(f"  📁 Directories: {' | '.join(parts)}")

    # -- Build config.toml from static + interface/plugin schemas -----------
    config_path = target / ".mnemo" / "config.toml"
    if config_path.exists() and not force:
        results["config"] = {"path": str(config_path), "status": "exists"}
        warn(f"Config already exists at {config_path} (use --force to overwrite)")
    else:
        from mnemo.core.param_config import ParamConfig

        static = dict(_STATIC_CONFIG)
        static["global"]["name"] = target.name

        pc = ParamConfig({})
        toml_content = pc.generate_toml_template(static_config=static)

        config_path.parent.mkdir(parents=True, exist_ok=True)
        existed = config_path.exists()
        config_path.write_text(toml_content, encoding="utf-8")
        results["config"] = {
            "path": str(config_path),
            "status": "overwritten" if existed and force else "created",
        }
        if existed and force:
            console.print(f"  {_icon('config')} [green]Config overwritten:[/green] {config_path}")
        else:
            console.print(f"  {_icon('config')} [green]Config generated from interface & plugin schemas[/green]")

    # -- Build file_categories.toml -----------------------------------------
    fc_path = target / ".mnemo" / "file_categories.toml"
    if fc_path.exists() and not force:
        results["file_categories"] = {"path": str(fc_path), "status": "exists"}
        console.print(f"  {_icon('config')} [dim]File categories config already exists[/dim]")
    else:
        from mnemo.core.param_config import ParamConfig
        pc = ParamConfig({})
        fc_content = pc.generate_file_categories_toml()
        fc_existed = fc_path.exists()
        fc_path.write_text(fc_content, encoding="utf-8")
        results["file_categories"] = {
            "path": str(fc_path),
            "status": "overwritten" if fc_existed and force else "created",
        }
        if fc_existed and force:
            console.print(f"  {_icon('config')} [green]File categories config overwritten[/green]")
        else:
            console.print(f"  {_icon('config')} [green]File categories config generated[/green]")

    # -- Initialize databases ------------------------------------------------
    from mnemo.core.plugin_base import PluginHub
    from mnemo.core.interfaces import IIndexer
    indexer = PluginHub.get(IIndexer, "sqlite")

    index_db_path = target / ".mnemo" / "index.db"
    index_existed = index_db_path.exists()
    indexer.init(target)
    results["index"] = {
        "path": str(index_db_path),
        "status": "exists" if index_existed else "created",
    }
    if index_existed:
        console.print(f"  {_icon('info')} [dim]Index database already exists:[/dim] {index_db_path}")
    else:
        console.print(f"  {_icon('info')} [green]Index database created:[/green] {index_db_path}")

    from mnemo.plugins.vector_stores.lancedb_store import LanceDBStore
    vector_store = LanceDBStore(target)

    vectors_path = target / "embedding"
    # Check BEFORE init_tables() — the embedding/ dir may have been created
    # by the directory structure step above, so look for actual LanceDB data
    vectors_existed = vectors_path.exists() and any(
        f.suffix in (".lance", ".manifest") for f in vectors_path.iterdir()
    ) if vectors_path.exists() else False
    vector_store.init_tables()
    results["vectors"] = {
        "path": str(vectors_path),
        "status": "exists" if vectors_existed else "created",
    }
    if vectors_existed:
        console.print(f"  {_icon('running')} [dim]Vector store already exists:[/dim] {vectors_path}")
    else:
        console.print(f"  {_icon('running')} [green]Vector store initialized:[/green] {vectors_path}")

    # -- Create .env template ------------------------------------------------
    env_path = target / ".mnemo" / ".env"
    if env_path.exists() and not force:
        results["env"] = {"path": str(env_path), "status": "exists"}
        console.print(f"  {_icon('config')} [dim].env template already exists[/dim]")
    else:
        try:
            from importlib.resources import files
            template = files("mnemo.prompts").joinpath("env.template").read_text(encoding="utf-8")
            env_existed = env_path.exists()
            env_path.write_text(template, encoding="utf-8")
            results["env"] = {
                "path": str(env_path),
                "status": "overwritten" if env_existed and force else "created",
            }
            if env_existed and force:
                console.print(f"  {_icon('config')} [green].env template overwritten[/green]")
            else:
                console.print(f"  {_icon('config')} [green]Created .env template (edit to add your API keys)[/green]")
        except Exception:
            results["env"] = {"path": str(env_path), "status": "skipped"}
            console.print(f"  {_icon('config')} [dim].env template not available — skipped[/dim]")

    # -- Persist KB path in global config ------------------------------------
    _save_global_default(target)

    init_output(results)
