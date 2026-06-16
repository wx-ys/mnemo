"""LanceDB implementation of IVectorStore.

File-based vector storage — zero config, just a directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from mnemo.core.interfaces import IVectorStore
from mnemo.core.interfaces.param_spec import Param


def _get_vector_dimension() -> int:
    """Get the default vector dimension from the global config.

    Reads ``[global].dimension`` via :func:`get_global_config`.  Falls
    back to the GLOBAL_CONFIG_SCHEMA class-level default (1024) on error.

    This function exists for module-level use (e.g. schema generation
    before a ``KnowledgeBase`` is fully initialized).
    """
    try:
        from mnemo.core.param_config import GLOBAL_CONFIG_SCHEMA, get_global_config
        dim = get_global_config().get("dimension", 0)
        if dim:
            return int(dim)
        # Ultimate fallback: class-level default from the global schema
        return int(GLOBAL_CONFIG_SCHEMA.get("dimension", Param(type="int", default=1024)).default)
    except Exception:
        import logging
        logging.getLogger("mnemo").debug(
            "Failed to resolve vector dimension from global config; falling back to 1024",
        )
        return 1024


def _make_arrow_schema(dimension: int) -> pa.Schema:
    """Create an Arrow schema with the given vector dimension."""
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dimension)),
        pa.field("file_type", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("model", pa.string()),
        pa.field("start_char", pa.int32()),
        pa.field("end_char", pa.int32()),
        pa.field("section_header", pa.string()),
        pa.field("parent_id", pa.string()),
    ])


class LanceDBStore(IVectorStore):
    """LanceDB-backed vector store.

    Each table corresponds to a different content type
    (``raw_md``, ``raw_wiki``, ``metadata``).

    ANN index is created automatically on first use.  Index parameters
    are configurable via ``config_schema``.

    Parameters
    ----------
    data_dir : Path
        Root data directory. LanceDB data is stored at
        ``data_dir/embedding/``.
    """

    __plugin_impl__ = True
    name = "lancedb"

    config_schema: dict[str, Param] = {
        "index_type": Param(
            type="str", default="auto",
            desc="ANN index: 'auto', 'ivf_pq', 'ivf_hnsw_sw', or 'none'"
        ),
        "index_metric": Param(
            type="str", default="cosine",
            desc="Distance metric: 'cosine', 'l2', or 'dot'"
        ),
        "index_num_partitions": Param(
            type="int", default=0,
            desc="IVF partitions (0 = auto: sqrt(n_rows))"
        ),
    }

    def __init__(self, data_dir: Path | None = None):
        self._data_dir = data_dir
        self._embed_dir: Path | None = None
        self._db: Any = None
        self._indexed: set[str] = set()  # tables that have been indexed
        if data_dir is not None:
            self.init(data_dir)

    def init(self, data_dir: Path) -> None:
        """Bind to a data directory.

        Parameters
        ----------
        data_dir : Path
        """
        self._data_dir = data_dir
        self._embed_dir = data_dir / "embedding"
        self._embed_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._embed_dir))

    def _read_config(self) -> dict:
        """Read resolved config from param_config (or use defaults)."""
        try:
            from mnemo.core.param_config import get_config
            return get_config(self.__class__)
        except Exception:
            return {}

    def _get_dimension(self) -> int:
        """Resolve vector dimension from the global config.

        The ``dimension`` setting lives in ``[global]`` (single source of
        truth shared by embedder and vector store).  Falls back to the
        module-level helper (IEmbedder class-level default) on error.
        """
        from mnemo.core.param_config import get_global_config
        dim = get_global_config().get("dimension", 0)
        if dim:
            return int(dim)
        return _get_vector_dimension()

    # -- IVectorStore ---------------------------------------------------------

    def init_tables(self, dimension: int | None = None) -> None:
        """Create tables and ANN indexes if they do not exist. Idempotent.

        Creates: ``raw_md``, ``raw_wiki``, ``metadata``.
        Creates IVF_PQ ANN index on the ``vector`` column for fast search.

        Parameters
        ----------
        dimension : int, optional
            Vector dimension. If None, tables are only created if they
            don't exist (existing tables are left as-is).
        """
        if dimension is None:
            # Default mode: create missing tables only, don't touch existing
            existing = self._db.table_names()  # noqa
            schema = _make_arrow_schema(self._get_dimension())
            for name in ("raw_md", "raw_wiki", "metadata"):
                if name not in existing:
                    try:
                        self._db.create_table(name, schema=schema, mode="create")
                    except Exception:
                        import logging
                        logging.getLogger("mnemo").debug(
                            "Failed to create LanceDB table '%s' (may already exist)", name,
                        )
                self._ensure_index(name)
        else:
            # Explicit dimension: ensure table matches
            for name in ("raw_md", "raw_wiki", "metadata"):
                self._ensure_table(name, dimension)
                self._ensure_index(name)

    def _ensure_index(self, table_name: str) -> None:
        """Create an ANN index on *table_name* if not already indexed."""
        cfg = self._read_config()
        index_type = cfg.get("index_type", "auto")
        if index_type == "none":
            return
        if table_name in self._indexed:
            return
        existing = self._db.table_names()  # noqa
        if table_name not in existing:
            return
        try:
            tbl = self._db.open_table(table_name)
            metric = cfg.get("index_metric", "cosine")
            num_partitions = int(cfg.get("index_num_partitions", 0))
            # Default: IVF_PQ with auto partitions
            if index_type == "auto":
                tbl.create_index(metric=metric)
            elif index_type == "ivf_pq":
                kwargs = {"metric": metric}
                if num_partitions > 0:
                    kwargs["num_partitions"] = num_partitions
                tbl.create_index(**kwargs)
            elif index_type == "ivf_hnsw_sw":
                kwargs = {"metric": metric, "index_type": "IVF_HNSW_SW"}
                if num_partitions > 0:
                    kwargs["num_partitions"] = num_partitions
                tbl.create_index(**kwargs)
            self._indexed.add(table_name)
        except Exception as exc:
            import logging
            logging.getLogger("mnemo").debug(
                "Failed to create ANN index on table '%s' (best-effort): %s",
                table_name, exc,
            )

    def _ensure_table(self, name: str, dimension: int) -> None:
        """Ensure table *name* exists with the given vector dimension.

        If a table with the same name exists but has a different vector
        dimension (or no vector column), it is dropped and recreated.
        """
        existing = self._db.table_names()  # noqa: table_names is still the stable API
        if name in existing:
            try:
                tbl = self._db.open_table(name)
                vec_field = tbl.schema.field("vector")
                existing_dim = vec_field.type.list_size
                if existing_dim == dimension:
                    return  # Already correct
            except Exception:
                pass  # Can't inspect schema — will drop and recreate

            # Drop existing table (wrong dimension or no vector column)
            try:
                self._db.drop_table(name)
            except Exception:
                pass

        # Create fresh table
        schema = _make_arrow_schema(dimension)
        try:
            self._db.create_table(name, schema=schema, mode="create")
        except Exception:
            pass  # Race condition

    def add_vectors(
        self,
        table: str,
        ids: list[str],
        vectors: list[list[float]],
        metadata: dict[str, list] | None = None,
    ) -> None:
        """Batch-insert vectors.

        Parameters
        ----------
        table : str
            Table name.
        ids : list of str
        vectors : list of list of float
        metadata : dict, optional
            Extra columns as {col_name: [values]}.
        """
        # Detect dimension from actual vectors and ensure table exists
        dim = len(vectors[0]) if vectors else self._get_dimension()
        self.init_tables()  # Ensure base tables exist
        self._ensure_table(table, dim)  # Ensure this specific table matches dimension

        tbl = self._db.open_table(table)
        rows: list[dict[str, Any]] = []
        for i, vec in enumerate(vectors):
            row: dict[str, Any] = {
                "id": ids[i],
                "vector": [float(v) for v in vec],
                "file_type": "",
                "chunk_index": 0,
                "model": "dashscope",
                "start_char": 0,
                "end_char": 0,
                "section_header": "",
                "parent_id": "",
            }
            if metadata:
                for col, vals in metadata.items():
                    if col in row:
                        row[col] = vals[i] if i < len(vals) else row[col]
            rows.append(row)
        tbl.add(rows)
        # Ensure ANN index exists after adding vectors
        self._ensure_index(table)

    def _verify_table(self, name: str, expected_dim: int) -> bool:
        """Check that table *name* exists and has a vector column of the right dim.

        Non-destructive — never drops or recreates tables.
        Returns True if the table is ready for search, False otherwise.
        """
        existing = self._db.table_names()  # noqa
        if name not in existing:
            import logging
            logging.getLogger("mnemo").warning(
                "Vector table '%s' does not exist. Run 'mnemo add' first.", name,
            )
            return False
        try:
            tbl = self._db.open_table(name)
            vec_field = tbl.schema.field("vector")
            existing_dim = vec_field.type.list_size
            if existing_dim == expected_dim:
                return True
            import logging
            logging.getLogger("mnemo").warning(
                "Vector table '%s' dimension mismatch: expected %d, got %d. "
                "Re-add files to rebuild the table.",
                name, expected_dim, existing_dim,
            )
            return False
        except Exception:
            import logging
            logging.getLogger("mnemo").warning(
                "Vector table '%s' has no 'vector' column. "
                "Re-add files to rebuild the table.", name,
            )
            return False

    def search(
        self,
        table: str,
        query_vector: list[float],
        candidate_ids: list[str] | None = None,
        limit: int = 10,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search with optional ID pre-filter.

        Parameters
        ----------
        table : str
        query_vector : list of float
        candidate_ids : list of str, optional
        limit : int
        filters : dict, optional

        Returns
        -------
        list of dict
        """
        dim = len(query_vector)
        self.init_tables()
        if not self._verify_table(table, dim):
            return []  # table not ready — return empty, caller should handle
        self._ensure_index(table)  # ensure ANN index exists before search

        tbl = self._db.open_table(table)

        # Build query
        q = tbl.search([float(v) for v in query_vector])

        # Apply ID pre-filter (if candidate_ids provided)
        if candidate_ids:
            # LanceDB where clause for IN filter
            q = q.where(f"id IN ({','.join(repr(c) for c in candidate_ids)})")

        # Apply extra filters
        if filters:
            for col, val in filters.items():
                q = q.where(f"{col} = '{val}'")

        results = q.limit(limit).to_list()
        return results

    def delete_vectors(self, table: str, ids: list[str]) -> None:
        """Delete vectors by file ID.

        Parameters
        ----------
        table : str
        ids : list of str
        """
        self.init_tables()
        tbl = self._db.open_table(table)
        for fid in ids:
            tbl.delete(f"id = '{fid}'")

    def table_info(self, table: str) -> dict[str, Any]:
        """Get table metadata.

        Parameters
        ----------
        table : str

        Returns
        -------
        dict
        """
        self.init_tables()
        tbl = self._db.open_table(table)
        return {
            "name": table,
            "num_rows": tbl.count_rows(),
            "dimension": self._get_dimension(),
            "model": "dashscope",
        }
