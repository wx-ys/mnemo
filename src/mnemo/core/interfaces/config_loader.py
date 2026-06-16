"""Configuration loader interface (IConfigLoader)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from mnemo.core.plugin_base import PluginBase, PluginHub


class IConfigLoader(PluginBase, ABC):
    """Interface for loading and resolving configuration.

    Merges config from multiple sources with priority:
    env vars > project-level > global > hardcoded defaults.

    Notes
    -----
    Subclasses must be registered via ``__plugin_impl__ = True`` marker.
    """

    __plugin_interface__ = True
    name: ClassVar[str] = "config_loader"
    plugin_path: ClassVar[str] = "config_loaders"

    @abstractmethod
    def load(self, data_dir: Path | None = None) -> dict:
        """Load and merge configuration from all sources.

        Parameters
        ----------
        data_dir : Path, optional
            Project data directory for project-level config.

        Returns
        -------
        dict
            Fully resolved configuration dictionary.
        """
        ...

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated path.

        Parameters
        ----------
        key : str
            Dot-separated key path, e.g. 'search.default_limit'.
        default : Any, optional
            Value to return if key is not found.

        Returns
        -------
        Any
            Configuration value, or *default* if not found.
        """
        ...
