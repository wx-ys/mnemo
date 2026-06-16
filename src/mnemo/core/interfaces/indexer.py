"""Global index interface (IIndexer)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from mnemo.core.interfaces.param_spec import Param
from mnemo.core.plugin_base import PluginBase, PluginHub

if TYPE_CHECKING:
    from mnemo.core.interfaces.types import FileMeta


class IIndexer(PluginBase, ABC):
    """Interface for the SQLite global index.

    Manages file metadata CRUD, operation logging, and
    consistency checks between the index and the filesystem.

    Notes
    -----
    Subclasses must be registered via ``__plugin_impl__ = True`` marker.
    """

    __plugin_interface__ = True
    name: ClassVar[str] = "indexer"
    plugin_path: ClassVar[str] = "indexers"

    # ── Config schema ──────────────────────────────────────────────────

    config_schema: dict[str, Param] = {
        "chunk_interval_days": Param(type="int", default=30, desc="Days per chunk directory before rolling"),
        "chunk_max_files": Param(type="int", default=50, desc="Max files per chunk directory"),
        "chunk_strategy": Param(type="str", default="time_and_count", desc="Chunk strategy: 'time', 'count', or 'time_and_count'"),
        "on_duplicate": Param(type="str", default="skip", desc="Behavior when duplicate file hash detected: 'skip' or 'overwrite'"),
    }

    # ── Lifecycle ──────────────────────────────────────────────────────

    def init(self, data_dir: "Path | None" = None) -> None:
        """Bind to a data directory. Called by KnowledgeBase after creation.

        The default implementation is a no-op.  Concrete indexers should
        override to set up database connections.
        """
        pass

    # ── Abstract interface ─────────────────────────────────────────────

    @abstractmethod
    def init_db(self) -> None:
        """Initialize database tables.

        Must be idempotent (safe to call on an existing database).
        Creates tables: files, key_registry, file_keys, operation_log, plugins.
        """
        ...

    @abstractmethod
    def insert_file(self, meta: "FileMeta") -> None:
        """Insert a new file record.

        Parameters
        ----------
        meta : FileMeta
            File metadata to insert.
        """
        ...

    @abstractmethod
    def update_file(self, file_id: str, **kwargs) -> None:
        """Update fields of an existing file record.

        Parameters
        ----------
        file_id : str
            File identifier.
        **kwargs
            Field names and new values to update.
        """
        ...

    @abstractmethod
    def delete_file(self, file_id: str) -> None:
        """Remove a file record from the index.

        Parameters
        ----------
        file_id : str
            File identifier.
        """
        ...

    @abstractmethod
    def get_file(self, file_id: str) -> "FileMeta | None":
        """Retrieve a single file record.

        Parameters
        ----------
        file_id : str
            File identifier.

        Returns
        -------
        FileMeta or None
            The file metadata, or None if not found.
        """
        ...

    @abstractmethod
    def list_files(
        self,
        file_type: str | None = None,
        tags: list[str] | None = None,
        keys: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        sort_by: str = "added_at",
        limit: int = 50,
        offset: int = 0,
    ) -> list["FileMeta"]:
        """List files with optional filters and pagination.

        Parameters
        ----------
        file_type : str, optional
            Filter by file extension.
        tags : list of str, optional
            Filter by tags (AND logic).
        keys : list of str, optional
            Filter by keys (AND logic).
        date_from : str, optional
            ISO 8601 lower bound on added_at.
        date_to : str, optional
            ISO 8601 upper bound on added_at.
        sort_by : str, optional
            Column to sort by. Default is 'added_at'.
        limit : int, optional
            Maximum results. Default is 50.
        offset : int, optional
            Pagination offset. Default is 0.

        Returns
        -------
        list of FileMeta
            Matching file records.
        """
        ...

    @abstractmethod
    def file_exists_by_hash(self, file_hash: str) -> str | None:
        """Check if a file with the given hash already exists.

        Parameters
        ----------
        file_hash : str
            Content hash in 'algorithm:hex' format.

        Returns
        -------
        str or None
            Existing file_id if found, None otherwise.
        """
        ...

    @abstractmethod
    def get_stats(self) -> dict:
        """Get aggregate statistics.

        Returns
        -------
        dict
            Keys: total_files, total_size, type_breakdown, embed_count, etc.
        """
        ...

    @abstractmethod
    def check_consistency(self) -> list[dict]:
        """Check index-filesystem consistency.

        Returns
        -------
        list of dict
            Inconsistency items, each with 'type', 'file_id', and 'detail' keys.
            Empty list means everything is consistent.
        """
        ...

    @abstractmethod
    def trash_file(self, file_id: str, deleted_at: str) -> None:
        """Soft-delete a file by marking it as trashed.

        Parameters
        ----------
        file_id : str
            File identifier.
        deleted_at : str
            ISO 8601 timestamp of deletion.
        """
        ...

    @abstractmethod
    def restore_file(self, file_id: str) -> None:
        """Restore a soft-deleted file from trash.

        Parameters
        ----------
        file_id : str
            File identifier.
        """
        ...

    @abstractmethod
    def list_trash(self, limit: int = 50, offset: int = 0) -> tuple[list, int]:
        """List soft-deleted files in trash.

        Parameters
        ----------
        limit : int
            Maximum results. Default 50.
        offset : int
            Pagination offset. Default 0.

        Returns
        -------
        tuple[list, int]
            (file_meta_list, total_count).
        """
        ...
