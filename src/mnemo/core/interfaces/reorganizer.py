"""Chunk reorganizer interface (IReorganizer)."""

from abc import ABC, abstractmethod
from typing import ClassVar

from mnemo.core.plugin_base import PluginBase, PluginHub


class IReorganizer(PluginBase, ABC):
    """Interface for reorganizing chunk directories.

    When the user changes the chunk strategy (interval_days or max_files),
    this component replans and migrates files to new chunk directories.

    Notes
    -----
    Subclasses must be registered via ``__plugin_impl__ = True`` marker.
    """

    __plugin_interface__ = True
    name: ClassVar[str] = "reorganizer"
    plugin_path: ClassVar[str] = "reorganizers"

    @abstractmethod
    def plan(self, file_type: str | None = None) -> list[dict]:
        """Generate a migration plan without executing it.

        Parameters
        ----------
        file_type : str, optional
            Restrict planning to this file type.

        Returns
        -------
        list of dict
            Each item: {'old_path': ..., 'new_path': ..., 'file_id': ...}.
        """
        ...

    @abstractmethod
    def execute(self, file_type: str | None = None) -> dict:
        """Execute the migration plan.

        Parameters
        ----------
        file_type : str, optional
            Restrict execution to this file type.

        Returns
        -------
        dict
            Report with keys: 'moved', 'skipped', 'errors'.
        """
        ...
