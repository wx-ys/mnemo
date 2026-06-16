"""Trash store — independent SQLite + LanceDB for soft-deleted files.

Mirrors the main knowledge base structure under ``.mnemo/trash/``:

::

    .mnemo/trash/
    ├── raw/                    # mirror of main raw/
    ├── raw_md/                 # mirror of main raw_md/
    ├── raw_wiki/               # mirror of main raw_wiki/
    ├── raw_metadata/           # mirror of main raw_metadata/
    ├── embedding/              # LanceDB (trash vectors)
    │   └── raw_md.lance/
    └── index.db                # SQLite (files + file_keys)

All trash data is fully independent — restore is a move + re-insert.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


class TrashStore:
    """Independent trash storage with its own SQLite + LanceDB.

    Parameters
    ----------
    data_dir : Path
        Knowledge base root directory.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._trash_dir = data_dir / ".mnemo" / "trash"
        self._db_path = self._trash_dir / "index.db"
        self._conn: sqlite3.Connection | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    def _ensure_dirs_and_schema(self) -> sqlite3.Connection:
        """Create trash directories + tables. Returns a connection."""
        for sub in ("raw", "raw_md", "raw_wiki", "raw_metadata", "embedding"):
            (self._trash_dir / sub).mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
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
            CREATE TABLE IF NOT EXISTS file_keys (
                file_id  TEXT NOT NULL,
                key_path TEXT NOT NULL,
                PRIMARY KEY (file_id, key_path)
            );
            CREATE TABLE IF NOT EXISTS entities (
                file_id  TEXT NOT NULL,
                name     TEXT NOT NULL,
                type     TEXT NOT NULL DEFAULT 'concept',
                description TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (file_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_trash_files_deleted
                ON files(deleted_at);
        """)
        return conn

    def ensure_init(self) -> None:
        """Create trash directories and tables (idempotent, public API)."""
        self._get_conn()  # triggers _ensure_dirs_and_schema if needed

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = self._ensure_dirs_and_schema()
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Trash ────────────────────────────────────────────────────────────

    def trash_file(
        self,
        meta: Any,            # FileMeta
        file_keys: list[str],
        entities: list[dict],
    ) -> dict:
        """Move a file and all its artifacts to trash.

        Vectors are NOT exported to trash (LanceDB row-level export is
        unreliable).  Callers should re-embed on restore.

        Parameters
        ----------
        meta : FileMeta
            File metadata from the main index.
        file_keys : list of str
            Keys associated with the file.
        entities : list of dict
            Graph entities associated with the file.

        Returns
        -------
        dict
            Summary with ``file_id``, ``trashed_files`` count.
        """
        self.ensure_init()
        now = datetime.now(UTC).isoformat()
        data_dir = self._data_dir
        moved = 0

        # 1. Move file artifacts (mirror original paths)
        for attr in ("raw_path", "md_path", "wiki_path", "metadata_path"):
            src_rel = getattr(meta, attr, "")
            if not src_rel:
                continue
            src = data_dir / src_rel
            if not src.exists():
                continue
            dest = self._trash_dir / src_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            moved += 1

        # 2. Insert file record into trash SQLite
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO files (
                id, file_type, filename, file_hash, file_size, source_path,
                raw_path, metadata_path, md_path, wiki_path,
                md_status, wiki_status, embed_status,
                category, tags, keywords, added_at, updated_at,
                deleted_at, custom, source_kb
            ) VALUES (
                :id, :file_type, :filename, :file_hash, :file_size, :source_path,
                :raw_path, :metadata_path, :md_path, :wiki_path,
                :md_status, :wiki_status, :embed_status,
                :category, :tags, :keywords, :added_at, :updated_at,
                :deleted_at, :custom, :source_kb
            )""",
            {
                "id": meta.id, "file_type": meta.file_type,
                "filename": meta.filename, "file_hash": meta.file_hash,
                "file_size": meta.file_size, "source_path": meta.source_path,
                "raw_path": meta.raw_path, "metadata_path": meta.metadata_path,
                "md_path": meta.md_path, "wiki_path": meta.wiki_path,
                "md_status": meta.md_status, "wiki_status": meta.wiki_status,
                "embed_status": meta.embed_status,
                "category": meta.category,
                "tags": json.dumps(meta.tags),
                "keywords": json.dumps(meta.keywords),
                "added_at": meta.added_at, "updated_at": meta.updated_at,
                "deleted_at": now,
                "custom": json.dumps(meta.custom),
                "source_kb": meta.source_kb,
            },
        )

        # 3. Insert file_keys into trash
        for key in file_keys:
            conn.execute(
                "INSERT OR REPLACE INTO file_keys (file_id, key_path) VALUES (?, ?)",
                (meta.id, key),
            )

        # 4. Insert entities into trash
        for ent in entities:
            conn.execute(
                "INSERT OR REPLACE INTO entities (file_id, name, type, description) "
                "VALUES (?, ?, ?, ?)",
                (meta.id, ent.get("name", ""), ent.get("type", "concept"),
                 ent.get("description", "")),
            )

        conn.commit()

        return {
            "file_id": meta.id,
            "trashed_files": moved,
        }

    # ── Restore ──────────────────────────────────────────────────────────

    def restore_file(
        self, file_id: str, main_indexer: Any, main_key_manager: Any,
    ) -> dict | None:
        """Restore a file from trash.

        Moves file artifacts back, re-inserts into main index, and
        re-registers keys. Does NOT re-embed (caller should trigger
        reindex if needed).

        Parameters
        ----------
        file_id : str
            File identifier.
        main_indexer : IIndexer
            Main SQLite indexer.
        main_key_manager : IKeyManager
            Main key manager.

        Returns
        -------
        dict or None
            Restored file summary, or None if not in trash.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM files WHERE id = ?", (file_id,),
        ).fetchone()
        if row is None:
            return None

        data_dir = self._data_dir
        restored_files = 0

        # 1. Move file artifacts back to original paths
        for attr in ("raw_path", "md_path", "wiki_path", "metadata_path"):
            src_rel = row[attr] or ""
            if not src_rel:
                continue
            src = self._trash_dir / src_rel
            if not src.exists():
                continue
            dest = data_dir / src_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            restored_files += 1

        # 2. Restore in main index — clear deleted_at on existing record
        #    (the record still exists; remove() only sets deleted_at)
        main_indexer.restore_file(file_id)

        # 3. Re-register keys
        key_rows = conn.execute(
            "SELECT key_path FROM file_keys WHERE file_id = ?", (file_id,),
        ).fetchall()
        for kr in key_rows:
            key = kr["key_path"]
            main_key_manager.register_key(key)
            main_key_manager.add_file_keys(file_id, [key])

        # 4. Delete from trash
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        conn.execute("DELETE FROM file_keys WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM entities WHERE file_id = ?", (file_id,))
        conn.commit()

        return {
            "file_id": file_id,
            "filename": row["filename"],
            "restored_files": restored_files,
        }

    # ── List ─────────────────────────────────────────────────────────────

    def list_trash(
        self, limit: int = 200, offset: int = 0,
    ) -> tuple[list[dict], int]:
        """List files in trash.

        Returns (items, total_count). Each item is a dict with
        ``file_id``, ``filename``, ``file_type``, ``deleted_at``.
        """
        conn = self._get_conn()
        total = conn.execute(
            "SELECT COUNT(*) FROM files WHERE deleted_at != ''",
        ).fetchone()[0]

        rows = conn.execute(
            "SELECT id, filename, file_type, file_size, deleted_at "
            "FROM files WHERE deleted_at != '' "
            "ORDER BY deleted_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        items = [
            {
                "file_id": r["id"],
                "filename": r["filename"],
                "file_type": r["file_type"],
                "file_size": r["file_size"],
                "deleted_at": r["deleted_at"],
            }
            for r in rows
        ]
        return items, total

    def get_trash_file(self, file_id: str) -> dict | None:
        """Get a single trash file metadata."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM files WHERE id = ?", (file_id,),
        ).fetchone()
        if row is None:
            return None

        def _parse_json(val):
            try:
                return json.loads(val) if val else []
            except Exception:
                return []

        return {
            "file_id": row["id"],
            "filename": row["filename"],
            "file_type": row["file_type"],
            "file_size": row["file_size"],
            "file_hash": row["file_hash"],
            "category": row["category"],
            "tags": _parse_json(row["tags"]),
            "raw_path": row["raw_path"],
            "md_path": row["md_path"],
            "wiki_path": row["wiki_path"],
            "metadata_path": row["metadata_path"],
            "added_at": row["added_at"],
            "deleted_at": row["deleted_at"],
            "md_status": row["md_status"],
            "wiki_status": row["wiki_status"],
            "embed_status": row["embed_status"],
        }

    # ── Clean (permanent delete) ─────────────────────────────────────────

    def clean_expired(self, days: int = 30) -> int:
        """Permanently delete trash files older than *days*.

        Removes file artifacts from disk, vectors from trash LanceDB,
        and records from trash SQLite.

        Returns count of permanently deleted files.
        """
        conn = self._get_conn()
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()

        rows = conn.execute(
            "SELECT id, raw_path, md_path, wiki_path, metadata_path "
            "FROM files WHERE deleted_at < ?",
            (cutoff,),
        ).fetchall()

        count = 0
        for row in rows:
            file_id = row["id"]

            # Delete file artifacts from trash filesystem
            for col in ("raw_path", "md_path", "wiki_path", "metadata_path"):
                rel = row[col]
                if rel:
                    f = self._trash_dir / rel
                    if f.exists():
                        if f.is_dir():
                            shutil.rmtree(f)
                        else:
                            f.unlink()

            # Delete from trash SQLite
            conn.execute("DELETE FROM file_keys WHERE file_id = ?", (file_id,))
            conn.execute("DELETE FROM entities WHERE file_id = ?", (file_id,))
            conn.execute("DELETE FROM files WHERE id = ?", (file_id,))

            count += 1

        conn.commit()
        return count
