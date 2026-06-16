"""File watcher implementation using watchdog.

Monitors the knowledge base directory for changes and triggers
re-index / re-embed operations as needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("mnemo.watcher")


class _ChangeHandler(FileSystemEventHandler):
    """Watchdog event handler that forwards changes to a callback."""

    def __init__(self, callback):
        self._callback = callback
        super().__init__()

    def on_modified(self, event):
        if not event.is_directory:
            self._callback("modified", Path(event.src_path))

    def on_created(self, event):
        if not event.is_directory:
            self._callback("created", Path(event.src_path))

    def on_deleted(self, event):
        if not event.is_directory:
            self._callback("deleted", Path(event.src_path))


class FileWatcher:
    """Watchdog-based file system monitor for Mnemo.

    Watches the ``raw/`` and ``raw_metadata/`` directories and
    triggers reindexing operations when files change.

    Parameters
    ----------
    data_dir : Path
        Root data directory to watch.
    on_change : callable
        Callback ``(action: str, path: Path)`` invoked on file events.
    """

    def __init__(self, data_dir: Path, on_change=None):
        self._data_dir = data_dir
        self._on_change = on_change or (lambda action, path: None)
        self._observer: Observer | None = None
        self._running = False

    def start(self) -> None:
        """Start the file watcher daemon.

        Monitors ``raw/`` and ``raw_metadata/`` directories recursively.
        """
        if self._running:
            return

        self._observer = Observer()
        handler = _ChangeHandler(self._on_change)

        # Watch raw and raw_metadata directories if they exist
        for subdir in ["raw", "raw_metadata"]:
            watch_dir = self._data_dir / subdir
            if watch_dir.exists():
                self._observer.schedule(handler, str(watch_dir), recursive=True)
                logger.info("Watching: %s", watch_dir)

        self._observer.start()
        self._running = True
        logger.info("File watcher started on %s", self._data_dir)

    def stop(self) -> None:
        """Stop the file watcher."""
        if self._observer and self._running:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._running = False
            logger.info("File watcher stopped")

    @property
    def is_running(self) -> bool:
        """Check if the watcher is currently running."""
        return self._running
