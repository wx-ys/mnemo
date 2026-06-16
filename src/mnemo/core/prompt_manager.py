"""Centralized prompt management.

All LLM prompts are stored in ``src/mnemo/prompts/builtin.toml``
for easy maintenance, i18n, and user customization.

Override hierarchy (later overrides earlier):
    1. Built-in: ``src/mnemo/prompts/builtin.toml`` (package data)
    2. Global:   ``~/.config/mnemo/prompts.toml``
    3. Project:  ``{data_dir}/.mnemo/prompts.toml``
    4. Runtime:  ``PromptManager.register_prompt()`` (plugins)

Language variants:
    Prompts are keyed as ``{name}`` (default, English) or
    ``{name}-zh`` (Chinese variant). The caller may request a
    specific language; if not found, falls back to the default.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from mnemo.prompts import _BUILTIN_PROMPTS_PATH

# ---------------------------------------------------------------------------
# Prompt Manager
# ---------------------------------------------------------------------------

class PromptManager:
    """Central registry for LLM prompts.

    Loads prompts from YAML files with a priority-based merge.
    Users can override any prompt by placing a file in their
    config directory or calling ``register_prompt()`` at runtime.

    Parameters
    ----------
    user_prompts_paths : list of Path or None
        Additional YAML files to merge. Project and global config
        paths are appended automatically.
    """

    def __init__(self, user_prompts_paths: list[Path] | None = None):
        self._prompts: dict[str, dict[str, str]] = {}

        # 1. Load built-in prompts from package YAML
        self._load_toml(_BUILTIN_PROMPTS_PATH)

        # 2. Load global user prompts
        global_path = Path.home() / ".config" / "mnemo" / "prompts.toml"
        if global_path.exists():
            self._load_toml(global_path)

        # 3. Load project/user-provided prompts
        if user_prompts_paths:
            for p in user_prompts_paths:
                if p.exists():
                    self._load_toml(p)

    # -- Public API -----------------------------------------------------------

    def get_system_prompt(self, name: str) -> str:
        """Get the system prompt for a named template.

        Parameters
        ----------
        name : str
            Prompt name, e.g. ``'wiki.paper'``, ``'entity_extraction'``.

        Returns
        -------
        str
            The system prompt. Falls back to ``'wiki.default'`` if
            *name* is not found.
        """
        entry = self._prompts.get(name, self._prompts.get("wiki.default", {}))
        return entry.get("system_prompt", "")

    def get_user_prompt_template(self, name: str) -> str:
        """Get the user prompt template for a named template.

        Parameters
        ----------
        name : str
            Prompt name, e.g. ``'wiki.paper'``.

        Returns
        -------
        str
            The user prompt template (may contain ``{placeholder}`` vars).
            Falls back to ``'{content}'`` if not found.
        """
        entry = self._prompts.get(name, self._prompts.get("wiki.default", {}))
        return entry.get("user_prompt_template", "{content}")

    def register_prompt(
        self, name: str, system_prompt: str, user_prompt_template: str
    ) -> None:
        """Register or override a prompt at runtime.

        This is the primary extension point for user plugins.
        Runtime registrations always take precedence over file-loaded prompts.

        Parameters
        ----------
        name : str
            Prompt name, e.g. ``'wiki.my_custom'``.
        system_prompt : str
        user_prompt_template : str
        """
        self._prompts[name] = {
            "system_prompt": system_prompt,
            "user_prompt_template": user_prompt_template,
        }

    def list_prompts(self) -> list[str]:
        """Return all registered prompt names.

        Returns
        -------
        list of str
        """
        return sorted(self._prompts.keys())

    def export(self) -> dict[str, dict[str, str]]:
        """Export all prompts as a dict (for serialization / debugging).

        Returns
        -------
        dict
        """
        return dict(self._prompts)

    # -- Internal -------------------------------------------------------------

    def _load_toml(self, path: Path) -> None:
        """Load and merge prompts from a TOML file.

        Entries in *path* overwrite previously loaded prompts
        with the same name (last-write-wins).

        Parameters
        ----------
        path : Path
            Path to the TOML file.
        """
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            if isinstance(data, dict):
                for name, entry in data.items():
                    if isinstance(entry, dict):
                        self._prompts[name] = {
                            "system_prompt": str(entry.get("system_prompt", "")),
                            "user_prompt_template": str(
                                entry.get("user_prompt_template", "{content}")
                            ),
                        }
        except Exception:
            import logging
            logging.getLogger("mnemo").warning(
                "Failed to load prompts from %s (best-effort, skipping)", path,
            )
