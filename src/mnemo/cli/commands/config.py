"""mnemo config command — view / export / modify configuration."""

from __future__ import annotations

import json
from pathlib import Path

import rich_click as click

from mnemo.cli.formatter import console, error, success


def run(
    ctx: click.Context,
    action: str,
    key: str | None,
    value: str | None,
    output: str | None,
    file_categories: bool = False,
):
    """Manage Mnemo configuration."""
    from mnemo.core.kb import _bootstrap_builtins

    data_dir = ctx.obj.get("data_dir") if ctx.obj else None
    kb_dir = Path(data_dir) if data_dir else Path.home() / "mnemo-data"

    # Determine which config file to operate on
    if file_categories:
        config_path = kb_dir / ".mnemo" / "file_categories.toml"
        config_label = "file_categories.toml"
    else:
        config_path = kb_dir / ".mnemo" / "config.toml"
        config_label = "config.toml"

    # -- `show` ------------------------------------------------------------
    if action == "show":
        # Display the raw TOML file (with ENV:: placeholders intact
        # for security — never expose resolved environment variable values).
        if config_path.exists():
            click.echo(config_path.read_text("utf-8"))
        else:
            error(f"No config file found at {config_path}")
            console.print(
                "  [dim]Run[/dim] [bold]mnemo init[/bold] [dim]to create one.[/dim]"
            )

    # -- `get` -------------------------------------------------------------
    elif action == "get":
        if not key:
            error("Usage: mnemo config get <key>")
            console.print("  [dim]Example: mnemo config get embedder.openai.model[/dim]")
            console.print(
                "  [dim]File categories:[/dim] mnemo config get [bold]-f[/bold] file_category.code"
            )
            return

        # Read from raw TOML file (never expose resolved ENV::VALUE keys
        # — always show the placeholder so API keys are not leaked).
        import tomllib
        try:
            raw = tomllib.loads(config_path.read_text("utf-8"))
        except Exception:
            error(f"Failed to parse {config_label}")
            return

        val = raw
        for part in key.split("."):
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                error(f"Key '{key}' not found in {config_label}")
                return

        click.echo(json.dumps(val, ensure_ascii=False, indent=2)
                   if isinstance(val, (dict, list))
                   else str(val))

    # -- `set` -------------------------------------------------------------
    elif action == "set":
        error("config set is not yet implemented.")
        console.print(
            f"  [dim]Edit[/dim] {config_path} [dim]directly.[/dim]"
        )

    # -- `example` ---------------------------------------------------------
    elif action == "example":
        _bootstrap_builtins()

        from mnemo.cli.commands.init_cmd import _STATIC_CONFIG
        from mnemo.core.param_config import ParamConfig

        static = dict(_STATIC_CONFIG)
        static["global"]["name"] = "my-knowledge-base"

        pc = ParamConfig({})
        if file_categories:
            content = pc.generate_file_categories_toml()
        else:
            content = pc.generate_toml_template(static_config=static)

        if output:
            Path(output).write_text(content, encoding="utf-8")
            success(f"Config example written to: {output}")
        else:
            click.echo(content)
