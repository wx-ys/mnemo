"""Key hierarchy manager interface (IKeyManager)."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from mnemo.core.plugin_base import PluginBase, PluginHub


class IKeyManager(PluginBase, ABC):
    """Interface for managing hierarchical keys and file-key associations.

    Keys use ``::`` as the hierarchy separator, e.g. ``astronomy::galaxy::spiral``.
    Searching a parent key automatically expands to all descendant keys.

    Notes
    -----
    Subclasses must be registered via ``__plugin_impl__ = True`` marker.
    Key data lives in SQLite; updating keys does NOT trigger re-embedding.
    """

    # ── Lifecycle ──────────────────────────────────────────────────────

    def init(self, data_dir: Path) -> None:
        """Bind to a data directory. Called by KnowledgeBase after creation."""
        pass

    __plugin_interface__ = True
    name: ClassVar[str] = "key_manager"
    plugin_path: ClassVar[str] = "key_managers"

    @abstractmethod
    def register_key(self, key_path: str, description: str = "") -> None:
        """Register a new key in the hierarchy.

        Parameters
        ----------
        key_path : str
            Full key path, e.g. 'research::paper::nlp'.
        description : str, optional
            Human-readable description of the key.
        """
        ...

    @abstractmethod
    def remove_key(self, key_path: str) -> None:
        """Remove a key and its file associations.

        Parameters
        ----------
        key_path : str
            Key to remove.
        """
        ...

    @abstractmethod
    def rename_key(self, old_path: str, new_path: str) -> None:
        """Rename a key, cascading to all associated files.

        Parameters
        ----------
        old_path : str
            Current key path.
        new_path : str
            New key path.
        """
        ...

    @abstractmethod
    def expand_keys(self, key_path: str) -> list[str]:
        """Expand a key to include itself and all descendants.

        Parameters
        ----------
        key_path : str
            Root key to expand.

        Returns
        -------
        list of str
            The key itself plus all descendant keys.
        """
        ...

    @abstractmethod
    def expand_keys_multi(self, key_paths: list[str]) -> list[str]:
        """Expand multiple keys and deduplicate.

        Parameters
        ----------
        key_paths : list of str
            Keys to expand.

        Returns
        -------
        list of str
            Deduplicated union of all expanded keys.
        """
        ...

    @abstractmethod
    def add_file_keys(self, file_id: str, key_paths: list[str]) -> None:
        """Associate keys with a file (additive).

        Parameters
        ----------
        file_id : str
            File identifier.
        key_paths : list of str
            Keys to associate.
        """
        ...

    @abstractmethod
    def remove_file_keys(self, file_id: str, key_paths: list[str]) -> None:
        """Remove specific key associations from a file.

        Parameters
        ----------
        file_id : str
            File identifier.
        key_paths : list of str
            Keys to disassociate.
        """
        ...

    @abstractmethod
    def set_file_keys(self, file_id: str, key_paths: list[str]) -> None:
        """Replace all key associations for a file.

        Parameters
        ----------
        file_id : str
            File identifier.
        key_paths : list of str
            New complete set of keys.
        """
        ...

    @abstractmethod
    def get_file_keys(self, file_id: str) -> list[str]:
        """Get all keys associated with a file.

        Parameters
        ----------
        file_id : str
            File identifier.

        Returns
        -------
        list of str
            Key paths.
        """
        ...

    @abstractmethod
    def get_files_by_keys(
        self, key_paths: list[str], mode: str = "and"
    ) -> list[str]:
        """Find files matching given keys.

        Parameters
        ----------
        key_paths : list of str
            Keys to match.
        mode : str, optional
            'and': file must have ALL keys. 'or': file must have ANY key.
            Default is 'and'.

        Returns
        -------
        list of str
            Matching file IDs.
        """
        ...

    @abstractmethod
    def get_key_tree(self, root_key: str | None = None) -> dict:
        """Get the key hierarchy as a nested dict.

        Parameters
        ----------
        root_key : str, optional
            Root to start from. None returns the full tree.

        Returns
        -------
        dict
            Nested dict representing the key tree.
        """
        ...

    @abstractmethod
    def get_key_stats(self) -> dict:
        """Get per-key usage statistics.

        Returns
        -------
        dict
            Mapping of key_path -> file_count.
        """
        ...
