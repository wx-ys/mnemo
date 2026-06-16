"""SQLite implementation of the global index (IIndexer).

Manages the ``files`` table with WAL-mode SQLite for concurrent read access.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from mnemo.core.interfaces import FileMeta, IIndexer


class SQLiteIndexer(IIndexer):
    """SQLite-backed global index.

    Uses a single ``index.db`` file in ``{data_dir}/.mnemo/``.
    All SQL uses parameterized queries; WAL mode is enabled for
    concurrent read support.

    Parameters
    ----------
    data_dir : Path
        Root data directory. The database is stored at
        ``data_dir/.mnemo/index.db``.
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
        """Bind the indexer to a data directory.

        Must be called before any CRUD operations. If the data directory
        changes, the old connection is closed.

        Parameters
        ----------
        data_dir : Path
            Root data directory.
        """
        if self._data_dir is not None and self._data_dir != data_dir:
            self.close()
        self._data_dir = data_dir
        self._ensure_tables(self._get_conn())

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create the database connection.

        Returns
        -------
        sqlite3.Connection

        Raises
        ------
        RuntimeError
            If ``init()`` has not been called yet.
        """
        if self._data_dir is None:
            raise RuntimeError(
                "SQLiteIndexer.init(data_dir) must be called before any operations"
            )

        if self._conn is not None:
            return self._conn

        db_dir = self._data_dir / ".mnemo"
        db_dir.mkdir(parents=True, exist_ok=True)

        db_path = db_dir / "index.db"
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    @staticmethod
    def _row_to_filemeta(row: sqlite3.Row) -> FileMeta:
        """Convert a ``sqlite3.Row`` to a ``FileMeta``.

        Parameters
        ----------
        row : sqlite3.Row
            Database row.

        Returns
        -------
        FileMeta
        """
        def _parse_json(val: str | None) -> list[str]:
            if val is None:
                return []
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return []

        def _parse_json_dict(val: str | None) -> dict[str, Any]:
            if val is None:
                return {}
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return {}

        return FileMeta(
            id=row["id"],
            file_type=row["file_type"],
            filename=row["filename"],
            file_hash=row["file_hash"],
            file_size=row["file_size"],
            source_path=row["source_path"] or "",
            raw_path=row["raw_path"] or "",
            metadata_path=row["metadata_path"] or "",
            md_path=row["md_path"] or "",
            wiki_path=row["wiki_path"] or "",
            md_status=row["md_status"] or "pending",
            wiki_status=row["wiki_status"] or "pending",
            embed_status=row["embed_status"] or "pending",
            category=row["category"] or "",
            tags=_parse_json(row["tags"]),
            keywords=_parse_json(row["keywords"]),
            added_at=row["added_at"] or "",
            updated_at=row["updated_at"] or "",
            deleted_at=row["deleted_at"] or "",
            custom=_parse_json_dict(row["custom"]),
            source_kb=row["source_kb"] or "",
        )

    # ------------------------------------------------------------------
    # IIndexer implementation
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Create database tables if they do not exist.

        Tables: files, key_registry, file_keys, operation_log, plugins.
        Idempotent — safe to call on an existing database.
        """
        # We need a data_dir to init; use a no-op if called without one
        # (the real data_dir is passed via KB, stored when first accessed)
        pass  # Tables are created lazily on first insert/get.

    def _ensure_tables(self, conn: sqlite3.Connection) -> None:
        """Create all tables if they do not already exist.

        Parameters
        ----------
        conn : sqlite3.Connection
            Active database connection.
        """
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                id              TEXT PRIMARY KEY,
                file_type       TEXT    NOT NULL,
                filename        TEXT    NOT NULL,
                file_hash       TEXT    NOT NULL,
                file_size       INTEGER NOT NULL DEFAULT 0,
                source_path     TEXT    NOT NULL DEFAULT '',
                raw_path        TEXT    NOT NULL DEFAULT '',
                metadata_path   TEXT    NOT NULL DEFAULT '',
                md_path         TEXT    NOT NULL DEFAULT '',
                wiki_path       TEXT    NOT NULL DEFAULT '',
                md_status       TEXT    NOT NULL DEFAULT 'pending',
                wiki_status     TEXT    NOT NULL DEFAULT 'pending',
                embed_status    TEXT    NOT NULL DEFAULT 'pending',
                category        TEXT    NOT NULL DEFAULT '',
                tags            TEXT    NOT NULL DEFAULT '[]',
                keywords        TEXT    NOT NULL DEFAULT '[]',
                added_at        TEXT    NOT NULL DEFAULT '',
                updated_at      TEXT    NOT NULL DEFAULT '',
                deleted_at      TEXT    NOT NULL DEFAULT '',
                custom          TEXT    NOT NULL DEFAULT '{}',
                source_kb       TEXT    NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS key_registry (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                key_path    TEXT UNIQUE NOT NULL,
                parent_key  TEXT    NOT NULL DEFAULT '',
                depth       INTEGER NOT NULL DEFAULT 0,
                description TEXT    NOT NULL DEFAULT '',
                created_at  TEXT    NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS file_keys (
                file_id  TEXT NOT NULL,
                key_path TEXT NOT NULL,
                PRIMARY KEY (file_id, key_path),
                FOREIGN KEY (file_id)   REFERENCES files(id)        ON DELETE CASCADE,
                FOREIGN KEY (key_path)  REFERENCES key_registry(key_path) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS operation_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL DEFAULT '',
                action      TEXT    NOT NULL DEFAULT '',
                file_id     TEXT    NOT NULL DEFAULT '',
                detail      TEXT    NOT NULL DEFAULT '',
                status      TEXT    NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS plugins (
                name          TEXT PRIMARY KEY,
                type          TEXT    NOT NULL DEFAULT '',
                version       TEXT    NOT NULL DEFAULT '',
                file_path     TEXT    NOT NULL DEFAULT '',
                enabled       INTEGER NOT NULL DEFAULT 1,
                registered_at TEXT    NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_files_type  ON files(file_type);
            CREATE INDEX IF NOT EXISTS idx_files_hash  ON files(file_hash);
            CREATE INDEX IF NOT EXISTS idx_files_added ON files(added_at);
            CREATE INDEX IF NOT EXISTS idx_key_parent  ON key_registry(parent_key);
            CREATE INDEX IF NOT EXISTS idx_key_depth   ON key_registry(depth);
            CREATE INDEX IF NOT EXISTS idx_fk_file     ON file_keys(file_id);
            CREATE INDEX IF NOT EXISTS idx_fk_key      ON file_keys(key_path);
        """)

        # Migration: add deleted_at column to existing databases
        try:
            conn.execute("ALTER TABLE files ADD COLUMN deleted_at TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists

    # -- CRUD ---------------------------------------------------------------

    def insert_file(self, meta: FileMeta) -> None:
        """Insert a new file record.

        Parameters
        ----------
        meta : FileMeta
            File metadata to insert.
        """
        conn = self._get_conn()
        self._ensure_tables(conn)

        conn.execute(
            """INSERT INTO files (
                id, file_type, filename, file_hash, file_size, source_path,
                raw_path, metadata_path, md_path, wiki_path,
                md_status, wiki_status, embed_status,
                category, tags, keywords, added_at, updated_at, custom, source_kb
            ) VALUES (
                :id, :file_type, :filename, :file_hash, :file_size, :source_path,
                :raw_path, :metadata_path, :md_path, :wiki_path,
                :md_status, :wiki_status, :embed_status,
                :category, :tags, :keywords, :added_at, :updated_at, :custom, :source_kb
            )""",
            {
                "id": meta.id,
                "file_type": meta.file_type,
                "filename": meta.filename,
                "file_hash": meta.file_hash,
                "file_size": meta.file_size,
                "source_path": meta.source_path,
                "raw_path": meta.raw_path,
                "metadata_path": meta.metadata_path,
                "md_path": meta.md_path,
                "wiki_path": meta.wiki_path,
                "md_status": meta.md_status,
                "wiki_status": meta.wiki_status,
                "embed_status": meta.embed_status,
                "category": meta.category,
                "tags": json.dumps(meta.tags),
                "keywords": json.dumps(meta.keywords),
                "added_at": meta.added_at,
                "updated_at": meta.updated_at,
                "custom": json.dumps(meta.custom),
                "source_kb": meta.source_kb,
            },
        )
        conn.commit()

    def get_file(self, file_id: str) -> FileMeta | None:
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
        conn = self._get_conn()
        self._ensure_tables(conn)

        row = conn.execute(
            "SELECT * FROM files WHERE id = ?", (file_id,)
        ).fetchone()

        if row is None:
            return None
        return self._row_to_filemeta(row)

    def update_file(self, file_id: str, **kwargs: Any) -> None:
        """Update fields of an existing file record.

        Parameters
        ----------
        file_id : str
            File identifier.
        **kwargs
            Field names and values to update.
        """
        conn = self._get_conn()
        if not kwargs:
            return

        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values())
        values.append(file_id)
        conn.execute(
            f"UPDATE files SET {set_clause} WHERE id = ?", values
        )
        conn.commit()

    def delete_file(self, file_id: str) -> None:
        """Remove a file record from the index.

        Parameters
        ----------
        file_id : str
            File identifier.
        """
        conn = self._get_conn()
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.commit()

    # -- Trash ----------------------------------------------------------------

    def trash_file(self, file_id: str, deleted_at: str) -> None:
        """Mark a file as deleted (soft-delete → trash).

        Sets ``deleted_at`` and moves ``raw_path`` / ``md_path`` /
        ``wiki_path`` / ``metadata_path`` to a trash prefix.

        Parameters
        ----------
        file_id : str
            File identifier.
        deleted_at : str
            ISO 8601 timestamp of deletion.
        """
        conn = self._get_conn()
        conn.execute(
            "UPDATE files SET deleted_at = ?, embed_status = 'pending' WHERE id = ?",
            (deleted_at, file_id),
        )
        conn.execute("DELETE FROM file_keys WHERE file_id = ?", (file_id,))
        conn.commit()

    def restore_file(self, file_id: str) -> FileMeta | None:
        """Restore a file from trash (undo soft-delete).

        Clears ``deleted_at``.

        Parameters
        ----------
        file_id : str
            File identifier.

        Returns
        -------
        FileMeta or None
            The restored file metadata, or None if not found.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM files WHERE id = ?", (file_id,),
        ).fetchone()
        if row is None:
            return None

        conn.execute(
            "UPDATE files SET deleted_at = '' WHERE id = ?", (file_id,),
        )
        conn.commit()

        return self._row_to_filemeta(row)

    def list_trash(self, limit: int = 50, offset: int = 0) -> tuple[list[FileMeta], int]:
        """List files in trash (soft-deleted).

        Parameters
        ----------
        limit : int
            Max results.
        offset : int
            Pagination offset.

        Returns
        -------
        tuple[list[FileMeta], int]
            (results, total_count).
        """
        conn = self._get_conn()
        self._ensure_tables(conn)

        total = conn.execute(
            "SELECT COUNT(*) FROM files WHERE deleted_at != ''",
        ).fetchone()[0]

        rows = conn.execute(
            "SELECT * FROM files WHERE deleted_at != '' "
            "ORDER BY deleted_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        return [self._row_to_filemeta(r) for r in rows], total

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
    ) -> list[FileMeta]:
        """List files with optional filters and pagination.

        Parameters
        ----------
        file_type : str, optional
            Filter by file extension.
        tags : list of str, optional
            Filter by tags (AND logic — file must have all tags).
        keys : list of str, optional
            Filter by keys via file_keys table (AND logic).
        date_from : str, optional
            ISO 8601 lower bound on ``added_at``.
        date_to : str, optional
            ISO 8601 upper bound on ``added_at``.
        sort_by : str, optional
            Column to sort by. Default is ``'added_at'``.
        limit : int, optional
            Maximum number of results. Default is 50.
        offset : int, optional
            Pagination offset. Default is 0.

        Returns
        -------
        list of FileMeta
        """
        conn = self._get_conn()
        self._ensure_tables(conn)

        # Whitelist sort columns to prevent injection
        _allowed_sort = {
            "added_at", "updated_at", "file_type", "filename",
            "file_size", "id",
        }
        if sort_by not in _allowed_sort:
            sort_by = "added_at"

        where_parts: list[str] = []
        params: list[Any] = []

        # Always exclude deleted (trashed) files from normal listing
        where_parts.append("deleted_at = ''")

        if file_type:
            where_parts.append("file_type = ?")
            params.append(file_type)

        if date_from:
            where_parts.append("added_at >= ?")
            params.append(date_from)

        if date_to:
            where_parts.append("added_at <= ?")
            params.append(date_to)

        if tags:
            for tag in tags:
                # Use SQLite JSON1 extension for correct array matching
                # (avoids false positives like "nlp" matching "nlp-pro")
                where_parts.append(
                    "EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)"
                )
                params.append(tag)

        if keys:
            placeholders = ", ".join("?" for _ in keys)
            where_parts.append(
                f"id IN (SELECT file_id FROM file_keys WHERE key_path IN ({placeholders}))"
            )
            params.extend(keys)

        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)

        query = (
            f"SELECT * FROM files {where_clause} "
            f"ORDER BY {sort_by} DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_filemeta(r) for r in rows]

    # -- Hash lookup --------------------------------------------------------

    def file_exists_by_hash(self, file_hash: str) -> str | None:
        """Check if a file with the given hash already exists.

        Parameters
        ----------
        file_hash : str
            Content hash in ``'algorithm:hex'`` format.

        Returns
        -------
        str or None
            Existing ``file_id`` if found, ``None`` otherwise.
        """
        conn = self._get_conn()
        self._ensure_tables(conn)

        row = conn.execute(
            "SELECT id FROM files WHERE file_hash = ? AND deleted_at = ''", (file_hash,)
        ).fetchone()
        return row["id"] if row else None

    # -- Stats --------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get aggregate statistics.

        Returns
        -------
        dict
            Keys: ``total_files``, ``total_size``, ``type_breakdown``,
            ``embed_count``, ``key_count``.
        """
        conn = self._get_conn()
        self._ensure_tables(conn)

        total = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(file_size), 0) AS s FROM files WHERE deleted_at = ''"
        ).fetchone()

        type_rows = conn.execute(
            "SELECT file_type, COUNT(*) AS n FROM files WHERE deleted_at = '' GROUP BY file_type"
        ).fetchall()

        embed_count = conn.execute(
            "SELECT COUNT(*) FROM files WHERE embed_status = 'done' AND deleted_at = ''"
        ).fetchone()[0]

        key_count = conn.execute(
            "SELECT COUNT(DISTINCT key_path) FROM file_keys"
        ).fetchone()[0]

        return {
            "total_files": total["n"],
            "total_size": total["s"],
            "type_breakdown": {r["file_type"]: r["n"] for r in type_rows},
            "embed_count": embed_count,
            "key_count": key_count,
        }

    # -- Consistency check --------------------------------------------------

    def check_consistency(self) -> list[dict[str, Any]]:
        """Check index-filesystem consistency.

        Verifies that every file record in SQLite has a corresponding
        file on disk.

        Returns
        -------
        list of dict
            Each item has ``type``, ``file_id``, and ``detail`` keys.
            An empty list means everything is consistent.
        """
        conn = self._get_conn()
        self._ensure_tables(conn)

        assert self._data_dir is not None  # _get_conn() ensures this
        issues: list[dict[str, Any]] = []

        rows = conn.execute("SELECT id, raw_path FROM files WHERE deleted_at = ''").fetchall()
        for row in rows:
            raw_path = self._data_dir / row["raw_path"]
            if not raw_path.exists():
                issues.append({
                    "type": "missing_file",
                    "file_id": row["id"],
                    "detail": f"File not found on disk: {raw_path}",
                })

        return issues

    # -- Lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
