"""File watcher interface (IWatcher)."""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class IWatcher(Protocol):
    """Protocol for file system monitoring.

    Watches the data directory for changes and triggers
    appropriate re-index / re-embed operations.

    Notes
    -----
    Implemented as a Protocol rather than ABC to allow
    any callable object to satisfy the interface.
    """

    def start(self) -> None:
        """Start the watcher daemon."""
        ...

    def stop(self) -> None:
        """Stop the watcher daemon."""
        ...

    def on_file_changed(self, path: Path) -> None:
        """Callback invoked when a watched file changes.

        Parameters
        ----------
        path : Path
            Absolute path to the changed file.
        """
        ...
