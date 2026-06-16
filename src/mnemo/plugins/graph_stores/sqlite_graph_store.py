"""SQLite implementation of IGraphStore.

Stores entities, relations, and file-entity links for LightRAG.
Uses a separate ``graph.db`` in ``.mnemo/``.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC
from pathlib import Path
from typing import Any

from mnemo.core.interfaces import IGraphStore


class SQLiteGraphStore(IGraphStore):
    """SQLite-backed entity/relation graph store.

    Uses ``{data_dir}/.mnemo/graph.db`` — independent from ``index.db``
    so the graph can be rebuilt without touching the global index.

    Parameters
    ----------
    data_dir : Path or None
        Root data directory. Set via ``init()``.
    """

    __plugin_impl__ = True
    name = "sqlite"

    def __init__(self):
        self._data_dir: Path | None = None
        self._conn: sqlite3.Connection | None = None

    def init(self, data_dir: Path) -> None:
        """Bind to a data directory and ensure tables exist.

        Parameters
        ----------
        data_dir : Path
        """
        if self._data_dir is not None and self._data_dir != data_dir:
            self.close()
        self._data_dir = data_dir
        self._init_tables()

    def _get_conn(self) -> sqlite3.Connection:
        if self._data_dir is None:
            raise RuntimeError("SQLiteGraphStore.init(data_dir) must be called first")
        if self._conn is not None:
            return self._conn
        db_dir = self._data_dir / ".mnemo"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / "graph.db"
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    # -- IGraphStore ----------------------------------------------------------

    def init_tables(self) -> None:
        """Create tables — called automatically by init()."""
        self._init_tables()

    def _init_tables(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS graph_entities (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE,
                type        TEXT    NOT NULL DEFAULT 'concept',
                description TEXT    NOT NULL DEFAULT '',
                created_at  TEXT    NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS graph_relations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id   INTEGER NOT NULL,
                target_id   INTEGER NOT NULL,
                relation    TEXT    NOT NULL DEFAULT 'related_to',
                description TEXT    NOT NULL DEFAULT '',
                weight      REAL    NOT NULL DEFAULT 1.0,
                created_at  TEXT    NOT NULL DEFAULT '',
                FOREIGN KEY (source_id) REFERENCES graph_entities(id),
                FOREIGN KEY (target_id) REFERENCES graph_entities(id)
            );
            CREATE TABLE IF NOT EXISTS file_entities (
                file_id     TEXT    NOT NULL,
                entity_id   INTEGER NOT NULL,
                relevance   REAL    NOT NULL DEFAULT 1.0,
                PRIMARY KEY (file_id, entity_id),
                FOREIGN KEY (entity_id) REFERENCES graph_entities(id)
            );
            CREATE INDEX IF NOT EXISTS idx_gr_source ON graph_relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_gr_target ON graph_relations(target_id);
            CREATE INDEX IF NOT EXISTS idx_fe_file   ON file_entities(file_id);
            CREATE INDEX IF NOT EXISTS idx_fe_entity ON file_entities(entity_id);
        """)
        conn.commit()

    def upsert_entities(self, entities: list[dict]) -> list[int]:
        """Insert or update entities by name.

        Parameters
        ----------
        entities : list of dict
            Each with keys: name, type, description.

        Returns
        -------
        list of int
            Entity IDs.
        """
        from datetime import datetime
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()
        ids: list[int] = []

        for e in entities:
            row = conn.execute(
                "SELECT id FROM graph_entities WHERE name = ?",
                (e["name"],),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE graph_entities SET type=?, description=? WHERE id=?",
                    (e.get("type", "concept"), e.get("description", ""), row["id"]),
                )
                ids.append(row["id"])
            else:
                cur = conn.execute(
                    "INSERT INTO graph_entities(name, type, description, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (e["name"], e.get("type", "concept"),
                     e.get("description", ""), now),
                )
                ids.append(cur.lastrowid)
        conn.commit()
        return ids

    def add_relations(self, relations: list[dict]) -> None:
        """Add relations between entities.

        Parameters
        ----------
        relations : list of dict
            Each with: source_id, target_id, relation, description.
        """
        from datetime import datetime
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()

        for r in relations:
            conn.execute(
                "INSERT OR IGNORE INTO graph_relations "
                "(source_id, target_id, relation, description, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (r["source_id"], r["target_id"],
                 r.get("relation", "related_to"),
                 r.get("description", ""), now),
            )
        conn.commit()

    def link_file_entities(
        self, file_id: str, entity_ids: list[int], scores: list[float]
    ) -> None:
        """Link entities to a file.

        Parameters
        ----------
        file_id : str
        entity_ids : list of int
        scores : list of float
        """
        conn = self._get_conn()
        conn.execute("DELETE FROM file_entities WHERE file_id = ?", (file_id,))
        conn.executemany(
            "INSERT INTO file_entities(file_id, entity_id, relevance) VALUES (?, ?, ?)",
            [(file_id, eid, scores[i] if i < len(scores) else 1.0)
             for i, eid in enumerate(entity_ids)],
        )
        conn.commit()

    def search_entities(self, name_pattern: str, limit: int = 10) -> list[dict]:
        """Find entities by name LIKE match.

        Parameters
        ----------
        name_pattern : str
            SQL LIKE pattern.
        limit : int

        Returns
        -------
        list of dict
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, name, type, description FROM graph_entities "
            "WHERE name LIKE ? LIMIT ?",
            (name_pattern, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def expand_from_entities(
        self, entity_ids: list[int], hops: int = 2
    ) -> list[str]:
        """Graph traversal: find file IDs related to given entities.

        Uses recursive CTE for N-hop graph expansion.

        Parameters
        ----------
        entity_ids : list of int
        hops : int

        Returns
        -------
        list of str
            File IDs ordered by proximity.
        """
        if not entity_ids:
            return []

        conn = self._get_conn()
        placeholders = ", ".join("?" for _ in entity_ids)

        rows = conn.execute(
            f"""WITH RECURSIVE graph_walk AS (
                -- Hop 0: direct neighbors of seed entities
                SELECT r.target_id AS entity_id, r.source_id AS seed, 0 AS hop
                FROM graph_relations r
                WHERE r.source_id IN ({placeholders})
                UNION
                SELECT r.source_id AS entity_id, r.target_id AS seed, 0 AS hop
                FROM graph_relations r
                WHERE r.target_id IN ({placeholders})
                UNION ALL
                -- Hop 1..N: continue traversal
                SELECT r.target_id, g.seed, g.hop + 1
                FROM graph_relations r
                JOIN graph_walk g ON r.source_id = g.entity_id
                WHERE g.hop < ?
            )
            SELECT DISTINCT fe.file_id, MIN(gw.hop) AS min_hop
            FROM graph_walk gw
            JOIN file_entities fe ON fe.entity_id = gw.entity_id
            WHERE gw.hop <= ?
            GROUP BY fe.file_id
            ORDER BY min_hop
            LIMIT 20""",
            entity_ids + entity_ids + [hops, hops],
        ).fetchall()

        return [r["file_id"] for r in rows]

    def get_file_entities(self, file_id: str) -> list[dict[str, Any]]:
        """Get all entities linked to a file.

        Parameters
        ----------
        file_id : str

        Returns
        -------
        list of dict
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT e.id, e.name, e.type, e.description, fe.relevance
               FROM file_entities fe
               JOIN graph_entities e ON fe.entity_id = e.id
               WHERE fe.file_id = ?
               ORDER BY fe.relevance DESC""",
            (file_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_file_entities(self, file_id: str) -> int:
        """Remove all entity links for a file, then clean up orphaned entities
        and relations.

        Parameters
        ----------
        file_id : str

        Returns
        -------
        int
            Number of entity links removed.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM file_entities WHERE file_id = ?", (file_id,),
        )
        removed = cursor.rowcount
        conn.commit()
        # Clean up entities and relations that are now orphaned
        self.cleanup_orphans()
        return removed

    def cleanup_orphans(self) -> int:
        """Remove entities and relations with no file links.

        Returns
        -------
        int
            Number of entities removed.
        """
        conn = self._get_conn()
        # Remove orphan relations (either endpoint gone)
        conn.execute(
            """DELETE FROM graph_relations
               WHERE source_id NOT IN (SELECT id FROM graph_entities)
                  OR target_id NOT IN (SELECT id FROM graph_entities)"""
        )
        # Remove entities with no file links and no relations
        cursor = conn.execute(
            """DELETE FROM graph_entities
               WHERE id NOT IN (SELECT DISTINCT entity_id FROM file_entities)
                 AND id NOT IN (SELECT DISTINCT source_id FROM graph_relations)
                 AND id NOT IN (SELECT DISTINCT target_id FROM graph_relations)"""
        )
        conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
