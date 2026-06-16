"""Import interface (IImporter)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from mnemo.core.plugin_base import PluginBase, PluginHub


class IImporter(PluginBase, ABC):
    """Interface for importing external knowledge bases.

    Supports importing from tar.gz archives or raw directories.
    Hash-based deduplication is applied automatically.

    Notes
    -----
    Subclasses must be registered via ``__plugin_impl__ = True`` marker.
    """

    __plugin_interface__ = True
    name: ClassVar[str] = "importer"
    plugin_path: ClassVar[str] = "importers"

    def init(self, data_dir: Path) -> None:
        """Bind to a data directory. Called by KnowledgeBase after creation."""
        pass

    @abstractmethod
    def import_from(self, source: Path, dry_run: bool = False) -> dict:
        """Import files from an external source.

        Parameters
        ----------
        source : Path
            Path to a tar.gz file or a directory.
        dry_run : bool, optional
            If True, preview the import without making changes.

        Returns
        -------
        dict
            Report with keys: 'imported', 'skipped', 'errors'.
        """
        ...
