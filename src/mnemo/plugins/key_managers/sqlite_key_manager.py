"""SQLite implementation of the key hierarchy manager (IKeyManager).

Uses recursive CTE for key expansion and shares ``index.db`` with the Indexer.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mnemo.core.interfaces import IKeyManager


class SQLiteKeyManager(IKeyManager):
    """SQLite-backed key hierarchy manager.

    Shares ``{data_dir}/.mnemo/index.db`` with ``SQLiteIndexer``.
    Tables ``key_registry`` and ``file_keys`` are created by the Indexer's
    ``_ensure_tables`` — this class assumes they already exist.

    Parameters
    ----------
    None (data_dir is bound via ``init()``).
    """

    __plugin_impl__ = True
    name = "sqlite"

    def __init__(self):
        self._data_dir: Path | None = None
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def init(self, data_dir: Path) -> None:
        """Bind the key manager to a data directory.

        Must be called before any operations. If the data directory
        changes, the old connection is closed.

        Parameters
        ----------
        data_dir : Path
            Root data directory.
        """
        if self._data_dir is not None and self._data_dir != data_dir:
            self.close()
        self._data_dir = data_dir
        # Force connection creation (ensures tables exist via Indexer init)
        self._get_conn()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create the database connection.

        Returns
        -------
        sqlite3.Connection

        Raises
        ------
        RuntimeError
            If ``init()`` has not been called.
        """
        if self._data_dir is None:
            raise RuntimeError(
                "SQLiteKeyManager.init(data_dir) must be called before any operations"
            )

        if self._conn is not None:
            return self._conn

        db_path = self._data_dir / ".mnemo" / "index.db"
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    @staticmethod
    def _parse_parent(key_path: str) -> tuple[str, int]:
        """Extract parent key and depth from a key path.

        Parameters
        ----------
        key_path : str
            Full key path, e.g. ``'a::b::c'``.

        Returns
        -------
        tuple[str, int]
            (parent_key, depth). For ``'a::b::c'`` → ``('a::b', 2)``.
        """
        parts = key_path.split("::")
        if len(parts) <= 1:
            return ("", 0)
        parent = "::".join(parts[:-1])
        return (parent, len(parts) - 1)

    # ------------------------------------------------------------------
    # Key registration
    # ------------------------------------------------------------------

    def register_key(self, key_path: str, description: str = "") -> None:
        """Register a new key in the hierarchy.

        Automatically determines ``parent_key`` and ``depth`` from
        the ``::``-separated path.

        Parameters
        ----------
        key_path : str
            Full key path, e.g. ``'research::paper::nlp'``.
        description : str, optional
            Human-readable description.
        """
        conn = self._get_conn()
        parent_key, depth = self._parse_parent(key_path)
        now = datetime.now(UTC).isoformat()

        conn.execute(
            """INSERT OR IGNORE INTO key_registry (key_path, parent_key, depth, description, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (key_path, parent_key, depth, description, now),
        )
        conn.commit()

    def remove_key(self, key_path: str) -> None:
        """Remove a key and its file associations.

        Cascading delete via foreign keys removes ``file_keys`` rows.

        Parameters
        ----------
        key_path : str
            Key to remove.
        """
        conn = self._get_conn()
        conn.execute("DELETE FROM key_registry WHERE key_path = ?", (key_path,))
        conn.commit()

    def rename_key(self, old_path: str, new_path: str) -> None:
        """Rename a key, cascading to all child keys and file associations.

        Uses deferred foreign key checks so that both ``key_registry``
        and ``file_keys`` can be updated within a single transaction.

        Parameters
        ----------
        old_path : str
            Current key path.
        new_path : str
            New key path.
        """
        conn = self._get_conn()

        # Temporarily disable FK checks — we update both key_registry
        # and file_keys atomically within a single transaction.
        conn.execute("PRAGMA foreign_keys = OFF")

        # Find all affected keys: the renamed key + its descendants
        descendants = self._expand_keys_internal(conn, old_path)
        descendants.append(old_path)

        for old in descendants:
            new = old.replace(old_path, new_path, 1)
            parent, depth = self._parse_parent(new)

            # Update file_keys first (FK references key_registry.key_path)
            conn.execute(
                "UPDATE file_keys SET key_path = ? WHERE key_path = ?",
                (new, old),
            )
            conn.execute(
                "UPDATE key_registry SET key_path = ?, parent_key = ?, depth = ? "
                "WHERE key_path = ?",
                (new, parent, depth, old),
            )
        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")

    # ------------------------------------------------------------------
    # Key expansion
    # ------------------------------------------------------------------

    def expand_keys(self, key_path: str) -> list[str]:
        """Expand a key to include itself and all descendants.

        Uses recursive CTE for efficient hierarchy traversal.

        Parameters
        ----------
        key_path : str
            Root key to expand.

        Returns
        -------
        list of str
        """
        conn = self._get_conn()
        return self._expand_keys_internal(conn, key_path)

    @staticmethod
    def _expand_keys_internal(conn: sqlite3.Connection, key_path: str) -> list[str]:
        """Internal CTE-based key expansion.

        Parameters
        ----------
        conn : sqlite3.Connection
        key_path : str

        Returns
        -------
        list of str
        """
        rows = conn.execute(
            """WITH RECURSIVE key_tree AS (
                SELECT key_path FROM key_registry WHERE key_path = ?
                UNION ALL
                SELECT k.key_path
                FROM key_registry k
                JOIN key_tree kt ON k.parent_key = kt.key_path
            )
            SELECT key_path FROM key_tree""",
            (key_path,),
        ).fetchall()
        return [row["key_path"] for row in rows]

    def expand_keys_multi(self, key_paths: list[str]) -> list[str]:
        """Expand multiple keys with a single recursive CTE.

        Parameters
        ----------
        key_paths : list of str

        Returns
        -------
        list of str
        """
        if not key_paths:
            return []
        conn = self._get_conn()
        placeholders = ", ".join("?" for _ in key_paths)
        rows = conn.execute(
            f"""WITH RECURSIVE key_tree AS (
                SELECT key_path FROM key_registry
                WHERE key_path IN ({placeholders})
                UNION ALL
                SELECT k.key_path FROM key_registry k
                JOIN key_tree kt ON k.parent_key = kt.key_path
            )
            SELECT DISTINCT key_path FROM key_tree""",
            key_paths,
        ).fetchall()
        return [r["key_path"] for r in rows]

    # ------------------------------------------------------------------
    # File-key associations
    # ------------------------------------------------------------------

    def add_file_keys(self, file_id: str, key_paths: list[str]) -> None:
        """Associate keys with a file (additive).

        Parameters
        ----------
        file_id : str
        key_paths : list of str
        """
        conn = self._get_conn()
        conn.executemany(
            "INSERT OR IGNORE INTO file_keys (file_id, key_path) VALUES (?, ?)",
            [(file_id, kp) for kp in key_paths],
        )
        conn.commit()

    def remove_file_keys(self, file_id: str, key_paths: list[str]) -> None:
        """Remove specific key associations from a file.

        Parameters
        ----------
        file_id : str
        key_paths : list of str
        """
        conn = self._get_conn()
        conn.executemany(
            "DELETE FROM file_keys WHERE file_id = ? AND key_path = ?",
            [(file_id, kp) for kp in key_paths],
        )
        conn.commit()

    def set_file_keys(self, file_id: str, key_paths: list[str]) -> None:
        """Replace all key associations for a file.

        Parameters
        ----------
        file_id : str
        key_paths : list of str
        """
        conn = self._get_conn()
        conn.execute("DELETE FROM file_keys WHERE file_id = ?", (file_id,))
        conn.executemany(
            "INSERT INTO file_keys (file_id, key_path) VALUES (?, ?)",
            [(file_id, kp) for kp in key_paths],
        )
        conn.commit()

    def get_file_keys(self, file_id: str) -> list[str]:
        """Get all keys associated with a file.

        Parameters
        ----------
        file_id : str

        Returns
        -------
        list of str
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT key_path FROM file_keys WHERE file_id = ?", (file_id,)
        ).fetchall()
        return [r["key_path"] for r in rows]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_files_by_keys(
        self, key_paths: list[str], mode: str = "and"
    ) -> list[str]:
        """Find files matching given keys.

        Parameters
        ----------
        key_paths : list of str
        mode : str, optional
            ``'and'``: file must have ALL keys. ``'or'``: ANY key.
            Default is ``'and'``.

        Returns
        -------
        list of str
        """
        conn = self._get_conn()
        if not key_paths:
            return []

        if mode == "or":
            placeholders = ", ".join("?" for _ in key_paths)
            rows = conn.execute(
                f"SELECT DISTINCT file_id FROM file_keys "
                f"WHERE key_path IN ({placeholders})",
                key_paths,
            ).fetchall()
            return [r["file_id"] for r in rows]

        # AND mode: INTERSECT
        # Build: SELECT file_id FROM file_keys WHERE key_path = ?1
        #        INTERSECT SELECT file_id FROM file_keys WHERE key_path = ?2 ...
        subqueries = "\nINTERSECT\n".join(
            ["SELECT file_id FROM file_keys WHERE key_path = ?"]
            * len(key_paths)
        )
        rows = conn.execute(subqueries, key_paths).fetchall()
        return [r["file_id"] for r in rows]

    # ------------------------------------------------------------------
    # Tree & Stats
    # ------------------------------------------------------------------

    def get_key_tree(self, root_key: str | None = None) -> dict[str, Any]:
        """Get the key hierarchy as a nested dict.

        Parameters
        ----------
        root_key : str, optional
            Root to start from. ``None`` returns the full tree.

        Returns
        -------
        dict
            Nested dict of ``{key_name: {subtree}}``.
        """
        conn = self._get_conn()

        if root_key is not None:
            rows = conn.execute(
                """SELECT key_path FROM key_registry
                   WHERE key_path = ? OR parent_key LIKE ? || '%'
                   ORDER BY depth, key_path""",
                (root_key, root_key),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT key_path FROM key_registry ORDER BY depth, key_path"
            ).fetchall()

        tree: dict[str, Any] = {}
        for row in rows:
            parts = row["key_path"].split("::")
            node = tree
            for part in parts:
                if part not in node:
                    node[part] = {}
                node = node[part]
        return tree

    def get_key_stats(self) -> dict[str, int]:
        """Get per-key usage statistics.

        Returns
        -------
        dict[str, int]
            Mapping of ``key_path`` → file count.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT k.key_path, COUNT(fk.file_id) AS cnt
               FROM key_registry k
               LEFT JOIN file_keys fk ON k.key_path = fk.key_path
               GROUP BY k.key_path
               ORDER BY cnt DESC"""
        ).fetchall()
        return {r["key_path"]: r["cnt"] for r in rows}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
