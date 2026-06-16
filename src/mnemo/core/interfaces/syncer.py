"""Remote sync interface (ISyncer)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from mnemo.core.interfaces.param_spec import Param
from mnemo.core.plugin_base import PluginBase, PluginHub


class ISyncer(PluginBase, ABC):
    """Interface for remote synchronization.

    Delegates to external tools (rclone, rsync, s3cmd) for actual
    data transfer. Mnemo only manages the sync workflow.

    Notes
    -----
    Subclasses must be registered via ``__plugin_impl__ = True`` marker.
    """

    __plugin_interface__ = True
    name: ClassVar[str] = "syncer"
    plugin_path: ClassVar[str] = "syncers"

    def init(self, data_dir: Path) -> None:
        """Bind to a data directory. Called by KnowledgeBase after creation."""
        pass

    # ── Config schema ──────────────────────────────────────────────────

    config_schema: dict[str, Param] = {
        "remote": Param(
            type="str", default="",
            desc="Remote path/URI for sync (e.g., 's3://bucket', 'rclone:remote:path')",
        ),
        "method": Param(
            type="str", default="rclone",
            desc="Sync method: 'rclone', 'rsync', or 's3'",
        ),
        "auto_sync": Param(
            type="bool", default=False,
            desc="Auto-sync after each add/remove operation",
        ),
    }

    # ── Abstract interface ─────────────────────────────────────────────

    @abstractmethod
    def push(self) -> dict:
        """Push local data to the remote.

        Returns
        -------
        dict
            Sync report with keys: 'synced', 'skipped', 'errors', 'timestamp'.
        """
        ...

    @abstractmethod
    def pull(self) -> dict:
        """Pull remote data to local.

        Returns
        -------
        dict
            Sync report with keys: 'synced', 'skipped', 'errors', 'timestamp'.
        """
        ...

    @abstractmethod
    def status(self) -> dict:
        """Check sync status.

        Returns
        -------
        dict
            Status with keys: 'last_push', 'last_pull', 'pending_changes'.
        """
        ...
