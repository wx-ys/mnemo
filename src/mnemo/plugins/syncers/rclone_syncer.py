"""Rclone-based remote sync implementation."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from mnemo.core.interfaces import ISyncer


class RcloneSyncer(ISyncer):
    """Sync the knowledge base via rclone.

    Wraps the ``rclone`` CLI for push/pull operations.

    Parameters
    ----------
    kb : KnowledgeBase or None
        The KB instance. Set via ``init()``.
    """

    __plugin_impl__ = True
    name = "default"

    def __init__(self):
        self._kb = None
        self._remote: str = ""

    def init(self, kb) -> None:
        """Bind to a KnowledgeBase instance.

        Parameters
        ----------
        kb : KnowledgeBase
        """
        self._kb = kb
        self._remote = kb.config_loader.get("sync.remote", "")

    def push(self) -> dict:
        """Push local data to the remote.

        Returns
        -------
        dict
            Sync report.
        """
        return self._run_rclone("sync", f"{self._kb.data_dir}", self._remote)

    def pull(self) -> dict:
        """Pull remote data to local.

        Returns
        -------
        dict
            Sync report.
        """
        return self._run_rclone("sync", self._remote, f"{self._kb.data_dir}")

    def status(self) -> dict:
        """Check sync status.

        Returns
        -------
        dict
            Status with keys: last_push, last_pull, pending_changes.
        """
        if not self._remote:
            return {"last_push": "", "last_pull": "", "pending_changes": "No remote configured"}

        try:
            result = subprocess.run(
                ["rclone", "check", str(self._kb.data_dir), self._remote],
                capture_output=True, text=True, timeout=30,
            )
            return {
                "last_push": "",
                "last_pull": "",
                "pending_changes": result.stdout.strip() or "unknown",
            }
        except FileNotFoundError:
            return {"last_push": "", "last_pull": "", "pending_changes": "rclone not installed"}
        except Exception as e:
            return {"last_push": "", "last_pull": "", "pending_changes": str(e)}

    def _run_rclone(self, command: str, source: str, dest: str) -> dict:
        """Execute an rclone command.

        Parameters
        ----------
        command : str
            rclone subcommand (e.g. 'sync').
        source : str
            Source path or remote.
        dest : str
            Destination path or remote.

        Returns
        -------
        dict
            Sync report.
        """
        if not self._remote:
            return {
                "direction": command,
                "synced": 0,
                "skipped": 0,
                "errors": ["No sync.remote configured in config.yaml"],
                "timestamp": datetime.now(UTC).isoformat(),
            }

        try:
            result = subprocess.run(
                ["rclone", command, source, dest, "--progress"],
                capture_output=True, text=True, timeout=300,
            )
            synced = 0
            if result.returncode == 0:
                synced = 1  # simplified — real impl would parse rclone output

            return {
                "direction": command,
                "synced": synced,
                "skipped": 0,
                "errors": [result.stderr.strip()] if result.returncode != 0 else [],
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except FileNotFoundError:
            return {
                "direction": command,
                "synced": 0,
                "skipped": 0,
                "errors": ["rclone is not installed. Install it: https://rclone.org/"],
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            return {
                "direction": command,
                "synced": 0,
                "skipped": 0,
                "errors": [str(e)],
                "timestamp": datetime.now(UTC).isoformat(),
            }
