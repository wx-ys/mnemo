"""Mnemo public Python API — the single entry point for all interfaces.

``MnemoAPI`` is a lightweight facade over ``KnowledgeBase``.  It provides
a stable, documented, context-manager-based lifecycle.  All CLI commands,
the REST server, and the MCP server delegate to this class.

Usage::

    # Context manager (recommended)
    with MnemoAPI("~/my-kb") as api:
        api.add("paper.pdf", keys=["research::nlp"])
        results = api.search("attention mechanism")
        for r in results:
            print(r.snippet)

    # Manual
    api = MnemoAPI("~/my-kb")
    try:
        ...
    finally:
        api.close()
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mnemo.api.types import (
    CheckReport,
    FileContext,
    FileInfo,
    ImportReport,
    KnowledgeBaseStats,
    SearchMode,
    SearchResult,
    SyncReport,
)

if TYPE_CHECKING:
    from mnemo.api.ask_types import AskResponse


class MnemoAPI:
    """Public Python API for Mnemo — a file-based personal knowledge base.

    Parameters
    ----------
    data_dir : str or Path
        Root directory for the knowledge base data.
        Default: ``~/mnemo-data``.
    config_path : str or Path, optional
        Path to an additional config file to merge.
    """

    def __init__(
        self,
        data_dir: str | Path = "~/mnemo-data",
        config_path: str | Path | None = None,
    ) -> None:
        self._data_dir: str = str(data_dir)
        self._config_path: str | None = str(config_path) if config_path else None
        self._kb: Any = None

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> MnemoAPI:
        self._ensure_kb()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release resources. The API instance is reusable after close."""
        self._kb = None

    # =====================================================================
    # Ingestion
    # =====================================================================

    def add(
        self,
        source: str | Path,
        *,
        move: bool = False,
        keys: list[str] | None = None,
        tags: list[str] | None = None,
        note: str = "",
        category: str = "",
        auto_md: bool | None = None,
        auto_wiki: bool | None = None,
        auto_embed: bool | None = None,
        overwrite: bool = False,
        on_progress: Callable[[str, str], None] | None = None,
        diagnose: bool = False,
        verbose: bool = False,
    ) -> FileInfo:
        """Add a file to the knowledge base.

        Pipeline: validate → copy → metadata → markdown → wiki → embed.

        Parameters
        ----------
        source : str or Path
            Path to the source file.
        move : bool
            Move the file instead of copying. Default False.
        keys : list of str, optional
            Hierarchical keys, e.g. ``['research::paper']``.
        tags : list of str, optional
            Flat tags for filtering.
        note : str, optional
            Initial user note.
        category : str, optional
            Content category override (auto-detected if empty).
        auto_md : bool, optional
            Override default markdown generation.
        auto_wiki : bool, optional
            Override default wiki generation.
        auto_embed : bool, optional
            Override default embedding generation.
        overwrite : bool
            Overwrite when a hash collision is detected.
        on_progress : callable, optional
            Callback ``(step_name: str, status: str)`` for progress reporting.
        diagnose : bool, optional
            Write detailed pipeline diagnostic traces to disk.
        verbose : bool, optional
            Also print diagnostic summaries to terminal (with diagnose=True).

        Returns
        -------
        FileInfo
        """
        return self.kb.add(
            source=source,
            move=move,
            keys=keys,
            tags=tags,
            note=note,
            category=category,
            auto_md=auto_md,
            auto_wiki=auto_wiki,
            auto_embed=auto_embed,
            overwrite=overwrite,
            on_progress=on_progress,
            diagnose=diagnose,
            verbose=verbose,
        )

    def add_batch(
        self, sources: list[str | Path], **kwargs: Any,
    ) -> list[FileInfo]:
        """Add multiple files.  Failed files are skipped (errors logged).

        Parameters
        ----------
        sources : list of str or Path
            Source file paths.
        **kwargs
            Passed to :meth:`add` for each file.

        Returns
        -------
        list of FileInfo
        """
        return self.kb.add_batch(sources, **kwargs)

    # =====================================================================
    # Retrieval
    # =====================================================================

    def search(
        self,
        query: str,
        *,
        mode: SearchMode = SearchMode.HYBRID,
        keys: list[str] | None = None,
        file_types: list[str] | None = None,
        limit: int = 10,
        with_metadata: bool = True,
        expand_chunks: bool = False,
        on_progress: Callable[[str, str], None] | None = None,
        diagnose: bool = False,
        verbose: bool = False,
    ) -> list[SearchResult]:
        """Search the knowledge base.

        Parameters
        ----------
        query : str
            Search query.
        mode : SearchMode
            'hybrid', 'vector', or 'keyword'. Default hybrid.
        keys : list of str, optional
            Restrict search to these key scopes (hierarchy auto-expanded).
        file_types : list of str, optional
            Restrict to these file extensions.
        limit : int
            Maximum results. Default 10.
        with_metadata : bool
            Include metadata vector store. Default True.
        expand_chunks : bool
            Return per-chunk results (rather than per-file). Default False.
        on_progress : callable, optional
            Callback ``(stage: str, status: str)`` forwarded to the searcher.
        diagnose : bool, optional
            Write detailed pipeline diagnostic traces to disk.
        verbose : bool, optional
            Also print diagnostic summaries to terminal (with diagnose=True).

        Returns
        -------
        list of SearchResult
        """
        return self.kb.search(
            query=query,
            mode=mode,
            keys=keys,
            file_types=file_types,
            limit=limit,
            with_metadata=with_metadata,
            expand_chunks=expand_chunks,
            on_progress=on_progress,
            diagnose=diagnose,
            verbose=verbose,
        )

    def ask(
        self,
        question: str,
        *,
        grounded: bool = True,
        limit: int = 10,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> AskResponse:
        """Ask a question and get a knowledge-base-grounded answer with citations.

        RAG pipeline: query rewrite → multi-source search → rerank →
        context assembly → LLM answer with inline citations.

        Parameters
        ----------
        question : str
            Natural language question.
        grounded : bool
            If True, answer is strictly grounded in KB content.
            Default True.
        limit : int
            Maximum source chunks to retrieve. Default 10.
        on_progress : callable, optional
            Callback for progress reporting.

        Returns
        -------
        AskResponse
        """
        from mnemo.core.kb_ask import AskPipeline
        pipeline = AskPipeline(self)
        return pipeline.ask(
            question, grounded=grounded, limit=limit,
            on_progress=on_progress,
        )

    # =====================================================================
    # CRUD
    # =====================================================================

    def get(self, file_id: str) -> FileInfo:
        """Get detailed file information.

        Parameters
        ----------
        file_id : str
            File identifier (UUID or filename).

        Returns
        -------
        FileInfo
        """
        return self.kb.get_info(file_id)

    def get_info(self, file_id: str) -> FileInfo:
        """Alias for :meth:`get` — backward compatible."""
        return self.get(file_id)

    def get_context(self, file_id: str) -> FileContext:
        """Get full file context for agent consumption.

        Returns metadata, markdown content, wiki summary, user notes,
        and linked graph entities.

        Parameters
        ----------
        file_id : str
            File identifier.

        Returns
        -------
        FileContext
        """
        return self.kb.get_context(file_id)

    def list_files(
        self,
        *,
        file_type: str | None = None,
        keys: list[str] | None = None,
        tags: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        sort_by: str = "added_at",
        limit: int = 50,
        offset: int = 0,
    ) -> list[FileInfo]:
        """List files with filters and pagination.

        Parameters
        ----------
        file_type : str, optional
            Filter by file extension.
        keys : list of str, optional
            Filter by keys (AND logic).
        tags : list of str, optional
            Filter by tags (AND logic).
        date_from : str, optional
            ISO 8601 lower bound on added_at.
        date_to : str, optional
            ISO 8601 upper bound on added_at.
        sort_by : str
            Column to sort by. Default ``'added_at'``.
        limit : int
            Maximum results. Default 50.
        offset : int
            Pagination offset. Default 0.

        Returns
        -------
        list of FileInfo
        """
        return self.kb.list_files(
            file_type=file_type,
            keys=keys,
            tags=tags,
            date_from=date_from,
            date_to=date_to,
            sort_by=sort_by,
            limit=limit,
            offset=offset,
        )

    def resolve_file_ref(self, ref: str) -> str | None:
        """Resolve a file reference (UUID or filename) to a file ID.

        Parameters
        ----------
        ref : str
            File UUID or filename.

        Returns
        -------
        str or None
            File ID if found, None otherwise.
        """
        return self.kb.resolve_file_ref(ref)

    def update(
        self,
        file_id: str,
        *,
        keys: list[str] | None = None,
        tags: list[str] | None = None,
        note: str | None = None,
        category: str | None = None,
    ) -> FileInfo:
        """Update file metadata.

        Parameters
        ----------
        file_id : str
            File identifier.
        keys : list of str, optional
            Replace all keys.
        tags : list of str, optional
            Replace all tags.
        note : str, optional
            Update user note.
        category : str, optional
            Update category.

        Returns
        -------
        FileInfo
        """
        return self.kb.update(
            file_id=file_id,
            keys=keys,
            tags=tags,
            note=note,
            category=category,
        )

    def remove(self, file_id: str) -> dict:
        """Soft-delete a file (move to .mnemo/trash/).

        Parameters
        ----------
        file_id : str
            File identifier.

        Returns
        -------
        dict
            Result with ``file_id``, ``filename``, ``trash_path``.
        """
        return self.kb.remove(file_id)

    # =====================================================================
    # Management
    # =====================================================================

    def stats(self) -> KnowledgeBaseStats:
        """Get aggregate knowledge base statistics.

        Returns
        -------
        KnowledgeBaseStats
        """
        return self.kb.stats()

    def check(self, fix: bool = False) -> CheckReport:
        """Check knowledge base integrity.

        Parameters
        ----------
        fix : bool
            If True, automatically repair issues.

        Returns
        -------
        CheckReport
        """
        return self.kb.check(fix=fix)

    def reindex(
        self,
        *,
        file_id: str | None = None,
        file_type: str | None = None,
        all_files: bool = False,
        meta_only: bool = False,
    ) -> dict[str, Any]:
        """Rebuild embeddings and/or index.

        Parameters
        ----------
        file_id : str, optional
            Reindex a single file.
        file_type : str, optional
            Reindex all files of this type.
        all_files : bool
            Reindex everything.
        meta_only : bool
            Only rebuild the metadata vector store.

        Returns
        -------
        dict
            Result summary with ``reindexed``, ``skipped``, ``failed``.
        """
        return self.kb.reindex(
            file_id=file_id,
            file_type=file_type,
            all_files=all_files,
            meta_only=meta_only,
        )

    def export_kb(
        self,
        dest: str | Path,
        *,
        file_type: str | None = None,
        keys: list[str] | None = None,
        after: str | None = None,
    ) -> Path:
        """Export the knowledge base.

        Parameters
        ----------
        dest : str or Path
            Destination path (should end with .tar.gz).
        file_type : str, optional
            Export only this type.
        keys : list of str, optional
            Export only files matching these keys.
        after : str, optional
            ISO 8601 date — only files added after this date.

        Returns
        -------
        Path
            Path to the created archive.
        """
        return self.kb.export_kb(
            dest=Path(dest),
            file_type=file_type,
            keys=keys,
            after=after,
        )

    def import_kb(
        self, source: str | Path, *, dry_run: bool = False,
    ) -> ImportReport:
        """Import an external knowledge base.

        Parameters
        ----------
        source : str or Path
            Path to tar.gz or directory.
        dry_run : bool
            If True, preview without importing.

        Returns
        -------
        ImportReport
        """
        return self.kb.import_kb(source=Path(source), dry_run=dry_run)

    def sync_push(self) -> SyncReport:
        """Push local data to remote.

        Returns
        -------
        SyncReport
        """
        return self.kb.sync_push()

    def sync_pull(self) -> SyncReport:
        """Pull remote data to local.

        Returns
        -------
        SyncReport
        """
        return self.kb.sync_pull()

    # =====================================================================
    # Internals
    # =====================================================================

    def _ensure_kb(self) -> None:
        """Lazy-init the underlying KnowledgeBase."""
        if self._kb is None:
            from mnemo.core.kb import KnowledgeBase
            self._kb = KnowledgeBase(self._data_dir, self._config_path)

    @property
    def kb(self) -> Any:
        """Access the underlying KnowledgeBase (for advanced use)."""
        self._ensure_kb()
        return self._kb

    @property
    def data_dir(self) -> str:
        """The resolved data directory path."""
        return self._data_dir

    def __repr__(self) -> str:
        return f"MnemoAPI(data_dir='{self._data_dir}')"
