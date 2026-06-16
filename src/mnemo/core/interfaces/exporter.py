"""Export interface (IExporter)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from mnemo.core.plugin_base import PluginBase, PluginHub


class IExporter(PluginBase, ABC):
    """Interface for exporting the knowledge base.

    Produces self-contained tar.gz archives that can be
    imported into another Mnemo instance.

    Notes
    -----
    Subclasses must be registered via ``__plugin_impl__ = True`` marker.
    """

    __plugin_interface__ = True
    name: ClassVar[str] = "exporter"
    plugin_path: ClassVar[str] = "exporters"

    def init(self, data_dir: Path) -> None:
        """Bind to a data directory. Called by KnowledgeBase after creation."""
        pass

    @abstractmethod
    def export_to(
        self,
        dest: Path,
        file_type: str | None = None,
        keys: list[str] | None = None,
        after: str | None = None,
    ) -> Path:
        """Export the knowledge base (or a subset) to a file.

        Parameters
        ----------
        dest : Path
            Destination file path (should end with .tar.gz).
        file_type : str, optional
            Export only this file type.
        keys : list of str, optional
            Export only files matching these keys.
        after : str, optional
            ISO 8601 date — only export files added after this date.

        Returns
        -------
        Path
            Path to the created archive.
        """
        ...
