"""Mnemo configuration loader.

Loads TOML config with deep merge from multiple sources.
Priority: env vars > project > global > hardcoded defaults.
"""

from __future__ import annotations

import os
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mnemo.core.interfaces import IConfigLoader

# ---------------------------------------------------------------------------
# Hardcoded defaults
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "global": {
        "mode": "auto",
        "name": "",
        "debug": False,
    },
    "sources": [],
}


# ---------------------------------------------------------------------------
# Config loader implementation
# ---------------------------------------------------------------------------

class ConfigLoader(IConfigLoader):
    """TOML configuration loader with deep merge.

    Load order:
        1. Hardcoded defaults (DEFAULT_CONFIG)
        2. Global config (~/.config/mnemo/config.toml)
        3. Project config ({data_dir}/.mnemo/config.toml)
        4. Environment variable overrides (MNEMO_* prefix)
    """

    __plugin_impl__ = True
    name = "toml"

    def __init__(self):
        self._config: dict[str, Any] = {}
        self._data_dir: Path | None = None

    def load(self, data_dir: Path | None = None) -> dict:
        """Load and merge configuration from all sources."""
        self._data_dir = data_dir
        self._load_dotenv(data_dir)

        config = deepcopy(DEFAULT_CONFIG)

        # 1. Merge global config
        global_path = Path.home() / ".config" / "mnemo" / "config.toml"
        if global_path.exists():
            config = self._deep_merge(config, self._load_toml(global_path))

        # 2. Merge project config
        if data_dir:
            project_path = data_dir / ".mnemo" / "config.toml"
        else:
            project_path = self._find_project_config()
        if project_path and project_path.exists():
            config = self._deep_merge(config, self._load_toml(project_path))

        # 3. Apply environment variable overrides
        config = self._apply_env_overrides(config)

        # 4. Resolve ${VAR_NAME} and legacy ENV::VAR_NAME placeholders
        #    (new code uses Param.env_var in config_schema instead)
        config = self._resolve_env_vars(config)

        self._config = config
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot-separated key path."""
        keys = key.split(".")
        value: Any = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    @property
    def config(self) -> dict:
        return self._config

    @staticmethod
    def _load_dotenv(data_dir: Path | None) -> None:
        import logging
        logger = logging.getLogger("mnemo")

        paths: list[Path] = []
        candidates = [
            ("global", Path.home() / ".config" / "mnemo" / ".env"),
        ]
        if data_dir:
            candidates += [
                ("project", data_dir / ".mnemo" / ".env"),
                ("data_dir", data_dir / ".env"),
            ]
        candidates.append(("cwd", Path.cwd() / ".env"))

        for label, p in candidates:
            if p.exists():
                paths.append((label, p))
                logger.debug("Loading .env from %s: %s", label, p)
            else:
                logger.debug("No .env at %s: %s", label, p)

        for _label, p in paths:
            load_dotenv(p, override=True)

    @staticmethod
    def _load_toml(path: Path) -> dict:
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except Exception:
            return {}

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        result = deepcopy(base)
        for key, value in override.items():
            if (key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    @staticmethod
    def _find_project_config() -> Path | None:
        current = Path.cwd()
        for parent in [current, *current.parents]:
            for ext in (".toml", ".yaml"):
                candidate = parent / ".mnemo" / f"config{ext}"
                if candidate.exists():
                    return candidate
        return None

    @staticmethod
    def _apply_env_overrides(config: dict) -> dict:
        for var_name, var_value in os.environ.items():
            if not var_name.startswith("MNEMO_"):
                continue
            key = var_name[6:].lower().replace("_", ".")
            _set_nested(config, key, var_value)
        return config

    @staticmethod
    def _resolve_env_vars(config: dict) -> dict:
        """Resolve ``ENV::`` and ``${}`` placeholders (legacy compat).

        New plugins should use ``Param.env_var`` instead — see
        :class:`mnemo.core.interfaces.param_spec.Param`.
        """
        if isinstance(config, dict):
            return {k: ConfigLoader._resolve_env_vars(v) for k, v in config.items()}
        if isinstance(config, list):
            return [ConfigLoader._resolve_env_vars(item) for item in config]
        if isinstance(config, str):
            if config.startswith("ENV::"):
                return os.environ.get(config[5:], "")
            if config.startswith("${") and config.endswith("}"):
                return os.environ.get(config[2:-1], "")
        return config


def _set_nested(d: dict, key_path: str, value: Any) -> None:
    keys = key_path.split(".")
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value
