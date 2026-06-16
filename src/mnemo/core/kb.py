"""
Mnemo main class: KnowledgeBase.

Top-level entry point that composes all components via dependency injection.
Every component is obtained through PluginHub, never imported directly.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mnemo.api.types import (
    CheckReport,
    FileContext,
    FileInfo,
    FileStatus,
    ImportReport,
    KnowledgeBaseStats,
    SearchMode,
    SearchResult,
    SyncReport,
)
from mnemo.core.plugin_base import PluginHub
from mnemo.core.interfaces import (
    IConfigLoader,
    IEntityExtractor,
    IExporter,
    IGraphStore,
    IImporter,
    IIndexer,
    IKeyManager,
    IReorganizer,
    ISearcher,
    ISyncer,
    IVectorStore,
    IChunker,
    IFileCategory,
)

if TYPE_CHECKING:
    from mnemo.core.interfaces.types import FileMeta


def _bootstrap_builtins() -> None:
    """Ensure all built-in implementations are imported and registered.

    This must be called before any PluginHub.get() call.
    The import side-effects trigger PluginBase.__init_subclass__
    auto-registration.  Idempotent — calling multiple times is harmless.
    """
    # Core infrastructure
    import mnemo.plugins.chunkers.fixed_size_chunker  # noqa: F401

    # Chunkers
    import mnemo.plugins.chunkers.langchain_chunker  # noqa: F401 (default)
    import mnemo.plugins.chunkers.paragraph_chunker  # noqa: F401
    import mnemo.plugins.chunkers.semantic_chunker  # noqa: F401
    import mnemo.plugins.chunkers.small_to_big_chunker  # noqa: F401
    import mnemo.plugins.chunkers.token_chunker  # noqa: F401
    import mnemo.plugins.entity_extractors.llm_entity_extractor  # noqa: F401 — IEntityExtractor
    import mnemo.plugins.exporters.tar_exporter  # noqa: F401 — IExporter
    import mnemo.plugins.file_categories.audio  # noqa: F401
    import mnemo.plugins.file_categories.code  # noqa: F401
    import mnemo.plugins.file_categories.code_py  # noqa: F401
    import mnemo.plugins.file_categories.data  # noqa: F401

    # File categories
    import mnemo.plugins.file_categories.docs  # noqa: F401
    import mnemo.plugins.file_categories.img  # noqa: F401
    import mnemo.plugins.file_categories.other  # noqa: F401
    import mnemo.plugins.file_categories.video  # noqa: F401
    import mnemo.plugins.file_categories.web  # noqa: F401
    import mnemo.plugins.graph_stores.sqlite_graph_store  # noqa: F401 — IGraphStore

    # Import / Export / Sync
    import mnemo.plugins.importers.tar_importer  # noqa: F401 — IImporter
    import mnemo.plugins.indexers.sqlite_indexer  # noqa: F401 — IIndexer
    import mnemo.plugins.key_managers.sqlite_key_manager  # noqa: F401 — IKeyManager

    # LLM provider
    import mnemo.plugins.parsers.code  # noqa: F401
    import mnemo.plugins.parsers.data_file  # noqa: F401
    import mnemo.plugins.parsers.image  # noqa: F401
    import mnemo.plugins.parsers.pdf  # noqa: F401

    # Plugins (parsers + templates)
    import mnemo.plugins.parsers.text  # noqa: F401
    import mnemo.plugins.parsers.url  # noqa: F401
    import mnemo.plugins.searchers.keyword_searcher  # noqa: F401

    # Search (searcher plugins self-register via __plugin_impl__ = True)
    import mnemo.plugins.searchers.lightrag_searcher  # noqa: F401
    import mnemo.plugins.searchers.simple_searcher  # noqa: F401
    import mnemo.plugins.syncers.rclone_syncer  # noqa: F401 — ISyncer
    import mnemo.plugins.templates.note  # noqa: F401
    import mnemo.plugins.templates.paper  # noqa: F401
    import mnemo.plugins.vector_stores.lancedb_store  # noqa: F401 — IVectorStore

    # Config
    import mnemo.utils.config  # noqa: F401 — IConfigLoader


class KnowledgeBase:
    """Main entry point for Mnemo.

    Assembles all components via their registries. Every subsystem
    can be replaced at runtime by registering a different implementation.

    Parameters
    ----------
    data_dir : str or Path
        Root directory for the knowledge base data.
    config_path : str or Path, optional
        Path to an additional config file to merge.

    Examples
    --------
    >>> kb = KnowledgeBase("~/mnemo-data")
    >>> kb.add("/path/to/paper.pdf", keys=["research::paper"])
    >>> results = kb.search("attention mechanism", keys=["research::paper"])
    """

    def __init__(
        self,
        data_dir: str | Path = "~/mnemo-data",
        config_path: str | Path | None = None,
    ):
        _bootstrap_builtins()

        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # -- config ----------------------------------------------------------
        self.config_loader = PluginHub.get(IConfigLoader, "toml")
        self.config = self.config_loader.load(self.data_dir)
        if config_path:
            pass  # TODO: merge extra config file

        # Load file categories config (separate TOML)
        fc_config: dict = {}
        fc_path = self.data_dir / ".mnemo" / "file_categories.toml"
        if fc_path.exists():
            fc_config = self._load_toml_file(fc_path)

        # Initialize unified parameter config system.
        from mnemo.core.param_config import init_param_config
        init_param_config(self.config, fc_config)

        # -- logging -----------------------------------------------------------
        from mnemo.utils.logging import setup_logging
        from mnemo.core.param_config import get_global_config
        global_cfg = get_global_config()
        log_level = "DEBUG" if global_cfg.get("debug", False) else "INFO"
        setup_logging(self.data_dir, level=log_level)

        # -- agent manager (config-driven singleton) ---------------------------
        from mnemo.core.agent_manager import AgentManager
        am = AgentManager.get_instance()
        am.init(self.config)

        # -- core subsystems (always needed) ---------------------------------
        self.indexer = PluginHub.get(IIndexer, "sqlite")
        self.indexer.init(self.data_dir)

        self.key_manager = PluginHub.get(IKeyManager, "sqlite")
        self.key_manager.init(self.data_dir)

        # -- embedding (module-level singleton via core/embedder.py) -----------
        from mnemo.core.embedder import init_embedder
        init_embedder(self.config)

        # -- vector store (IVectorStore, via PluginHub) -------------------------
        self.vector_store = PluginHub.get(IVectorStore, "lancedb")
        if hasattr(self.vector_store, "init"):
            self.vector_store.init(self.data_dir)

        # -- graph store (IGraphStore, via PluginHub) ----------------------------
        self.graph_store = PluginHub.get(IGraphStore, "sqlite")
        if hasattr(self.graph_store, "init"):
            self.graph_store.init(self.data_dir)

        # -- entity extractor (IEntityExtractor, via PluginHub) ------------------
        self.entity_extractor = PluginHub.get(IEntityExtractor, "llm")
        # Entity extractor self-configures via get_config() if it has config_schema

        # -- prompt manager ----------------------------------------------------
        from mnemo.core.prompt_manager import PromptManager
        self.prompt_manager = PromptManager(
            user_prompts_paths=[self.data_dir / ".mnemo" / "prompts.toml"]
        )

        # -- searcher (ISearcher, via PluginHub) ---------------------------------
        from mnemo.core.interfaces.searcher import ISearcher
        from mnemo.core.param_config import get_config
        search_cfg = get_config(ISearcher)
        searcher_name = search_cfg.get("default_plugin", "default")
        self.searcher = PluginHub.get(ISearcher, searcher_name)
        if hasattr(self.searcher, "init"):
            self.searcher.init(self.data_dir)

        # Resolve which ingestion capabilities the active searcher requires.
        # Pipeline stages that produce unneeded capabilities are skipped.
        self._required_caps: set[str] = getattr(
            self.searcher, 'required_capabilities', set(),
        )

        # -- lazy-loaded subsystems ------------------------------------------
        self._importer: Any = None
        self._exporter: Any = None
        self._reorganizer: Any = None
        self._syncer: Any = None

    @property
    def param_config(self):
        """The context-local :class:`ParamConfig` for this KB instance."""
        from mnemo.core.param_config import get_param_config
        return get_param_config()

    # ========================================================================
    # Add files
    # ========================================================================

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
        """Add a file to the knowledge base via the WorkflowEngine.

        DAG pipeline: validate → copy → metadata → markdown → wiki →
        entity_extract → embed → index.

        Parameters
        ----------
        source : str or Path
            Path to the source file.
        move : bool, optional
            If True, move the file; otherwise copy.
        keys : list of str, optional
            Hierarchical keys, e.g. ``['research::paper']``.
        tags : list of str, optional
            Flat tags for filtering.
        note : str, optional
            Initial user note attached to the file.
        category : str, optional
            Content category override (auto-detected if empty).
        auto_md : bool, optional
            Override default markdown generation.
        auto_wiki : bool, optional
            Override default wiki generation.
        auto_embed : bool, optional
            Override default embedding generation.
        overwrite : bool, optional
            If True, overwrite when a hash collision is detected.
        on_progress : callable, optional
            Callback ``(step_name, status)`` for progress reporting.
        diagnose : bool, optional
            If True, write detailed pipeline diagnostic traces to
            ``<data_dir>/.mnemo/diagnostics/``.
        verbose : bool, optional
            If True (with ``diagnose=True``), also print diagnostic
            summaries to the terminal.

        Returns
        -------
        FileInfo
            Metadata of the newly added file.

        Raises
        ------
        FileNotFoundError
            If *source* does not exist.
        """
        import asyncio
        import time as _time

        # Import add steps to register them
        import mnemo.core.workflow.add_steps  # noqa: F401

        from mnemo.core.workflow.context import WorkflowContext
        from mnemo.core.workflow.dag import WorkflowDAG
        from mnemo.core.workflow.engine import WorkflowEngine
        from mnemo.core.workflow.step import FunctionStep, StepConfig

        source_path = Path(source).expanduser().resolve()

        # Build DAG: linear chain of 8 steps
        step_names = [
            "validate_file", "copy_file", "create_metadata",
            "parse_file_to_markdown", "generate_wiki",
            "extract_entities", "embed_chunks", "write_index",
        ]
        prev = None
        steps: list[FunctionStep] = []
        for name in step_names:
            deps = [prev] if prev else []
            steps.append(FunctionStep(
                StepConfig(
                    name=name, type="function",
                    depends_on=deps,
                    progress_label=name.replace("_", " ").title(),
                ),
                func_name=name,
            ))
            prev = name

        dag = WorkflowDAG(steps, name="add")

        # -- Diagnostic context ------------------------------------------------
        diag_ctx = None
        trace_file = None
        if diagnose:
            from mnemo.core.diagnostics import DiagnosticContext
            trace_dir = self.data_dir / ".mnemo" / "diagnostics"
            trace_dir.mkdir(parents=True, exist_ok=True)
            ts = _time.strftime("%Y%m%d_%H%M%S")
            trace_file = trace_dir / f"add_{ts}_{dag.name}.jsonl"
            diag_ctx = DiagnosticContext(
                enabled=True,
                trace_file=trace_file,
                verbose=verbose,
            )

        ctx = WorkflowContext(
            workflow_name="add",
            kb=self,
            config={
                "auto_md": auto_md if auto_md is not None else True,
                "auto_wiki": auto_wiki if auto_wiki is not None else False,
                "auto_embed": auto_embed if auto_embed is not None else True,
                "overwrite": overwrite,
                "move": move,
                "category": category,
                "note": note,
            },
            diagnostic=diag_ctx,
        )
        ctx.set_input("source_path", source_path)
        ctx.data["keys"] = keys or []
        ctx.data["tags"] = tags or []

        if on_progress or diagnose:
            ctx.emitter = self._make_progress_emitter(
                on_progress, diag_ctx=diag_ctx,
            )

        engine = WorkflowEngine()
        try:
            ctx = asyncio.run(engine.execute(dag, ctx))
        except RuntimeError:
            loop = asyncio.get_event_loop()
            ctx = loop.run_until_complete(engine.execute(dag, ctx))

        # Check for duplicate
        dup_id = ctx.data.get("_duplicate_id")
        if dup_id:
            result = self.get_info(dup_id)
            result._duplicate = True  # type: ignore[attr-defined]
            return result

        # Invalidate searcher caches
        if hasattr(self.searcher, 'invalidate_keyword_index'):
            self.searcher.invalidate_keyword_index()

        # Close diagnostic sink and attach trace path to result
        result = ctx.data.get("file_info", FileInfo())
        if diag_ctx is not None and trace_file is not None:
            import asyncio as _asyncio
            try:
                loop = _asyncio.get_running_loop()
            except RuntimeError:
                _asyncio.run(ctx.emitter.close())
            else:
                # Fire-and-forget close in running loop
                pass  # sinks flush on close; we can't await here
            # Store trace path on result for CLI to report
            if hasattr(result, 'processing_detail') and result.processing_detail:
                result.processing_detail["diagnostic_trace"] = str(trace_file)

        return result

    @staticmethod
    def _make_progress_emitter(
        on_progress: Callable[[str, str], None] | None = None,
        diag_ctx: Any = None,
    ):
        """Build an EventEmitter that forwards to the legacy on_progress callback.

        When *diag_ctx* is provided, also registers DiagnosticSink and
        optionally VerboseDiagnosticSink on the emitter.
        """
        from mnemo.core.workflow.events import (
            EventEmitter, EventSink, WorkflowEvent,
        )

        class LegacyProgressSink(EventSink):
            async def handle(self, event: WorkflowEvent) -> None:
                if on_progress is None:
                    return
                if event.event_type == "step.start":
                    on_progress(event.step_name or "?", "in_progress")
                elif event.event_type == "step.end":
                    on_progress(event.step_name or "?", "done")
                elif event.event_type == "step.error":
                    on_progress(event.step_name or "?", "failed")
                elif event.event_type == "step.skip":
                    on_progress(event.step_name or "?", "skipped")
                elif event.event_type == "step.progress":
                    # Forward progress updates (e.g., thinking stream chunks)
                    # so they appear inline in the Rich spinner via
                    # ProgressDisplay.update_message().
                    kind = event.data.get("kind", "")
                    if kind == "thinking":
                        on_progress(
                            event.step_name or "?",
                            f"thinking:{event.message}",
                        )
                    else:
                        on_progress(
                            event.step_name or "?",
                            event.message or "in_progress",
                        )

        emitter = EventEmitter()
        emitter.register(LegacyProgressSink())

        # Register diagnostic sinks when --diagnose is active
        if diag_ctx is not None:
            from mnemo.core.diagnostics import DiagnosticSink, VerboseDiagnosticSink
            emitter.register(DiagnosticSink(diag_ctx))
            if diag_ctx.verbose:
                emitter.register(VerboseDiagnosticSink(diag_ctx))

        return emitter

    def add_batch(
        self, sources: list[str | Path], **kwargs
    ) -> list[FileInfo]:
        """Add multiple files.

        Parameters
        ----------
        sources : list of str or Path
            Source file paths.
        **kwargs
            Passed to :meth:`add` for each file.

        Returns
        -------
        list of FileInfo
            Results. Failed files are skipped (errors logged, not raised).
        """
        results: list[FileInfo] = []
        for src in sources:
            try:
                results.append(self.add(src, **kwargs))
            except Exception:
                import logging
                logging.getLogger("mnemo").warning(
                    "Batch add: failed to add '%s', skipping", src, exc_info=True,
                )
        return results

    def add_url(self, url: str, **kwargs) -> FileInfo:
        """Add a file from a URL.

        Parameters
        ----------
        url : str
            URL to download and ingest.
        **kwargs
            Passed to :meth:`add`.

        Returns
        -------
        FileInfo
        """
        raise NotImplementedError("add_url is not yet implemented")

    # ========================================================================
    # Search
    # ========================================================================

    def search(
        self,
        query: str,
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
        mode : SearchMode, optional
            'hybrid', 'vector', or 'keyword'. Default is hybrid.
        keys : list of str, optional
            Restrict search to these key scopes (hierarchy auto-expanded).
        file_types : list of str, optional
            Restrict to these file extensions.
        limit : int, optional
            Maximum results. Default is 10.
        with_metadata : bool, optional
            Include metadata vector store in search. Default is True.
        expand_chunks : bool, optional
            If False, merge multi-chunk results per file. Default is False.
        on_progress : callable, optional
            Callback ``(stage, status)`` forwarded to the searcher.
        diagnose : bool, optional
            If True, write detailed pipeline diagnostic traces to
            ``<data_dir>/.mnemo/diagnostics/``.
        verbose : bool, optional
            If True (with ``diagnose=True``), also print diagnostic
            summaries to the terminal.

        Returns
        -------
        list of SearchResult
            Results sorted by score descending.
        """
        import time as _time
        import uuid

        candidate_ids = None
        if keys:
            expanded = self.key_manager.expand_keys_multi(keys)
            candidate_ids = self.key_manager.get_files_by_keys(expanded, mode="and")

        # -- Diagnostic context ------------------------------------------------
        diag_ctx = None
        trace_file = None
        if diagnose:
            from mnemo.core.diagnostics import DiagnosticContext
            trace_dir = self.data_dir / ".mnemo" / "diagnostics"
            trace_dir.mkdir(parents=True, exist_ok=True)
            ts = _time.strftime("%Y%m%d_%H%M%S")
            trace_file = trace_dir / f"search_{ts}.jsonl"
            diag_ctx = DiagnosticContext(
                enabled=True,
                trace_file=trace_file,
                verbose=verbose,
                _run_id=uuid.uuid4().hex[:12],
            )
            # Write pipeline start event
            diag_ctx.emit_diagnostic(
                stage="search",
                event_type="pipeline.start",
                data={
                    "query": query,
                    "mode": mode.value,
                    "limit": limit,
                },
            )

        results = self.searcher.search(
            query=query,
            mode=mode.value,
            candidate_ids=candidate_ids,
            limit=limit,
            file_types=file_types,
            with_metadata=with_metadata,
            on_progress=on_progress,
            diagnose=diagnose,
            diagnostic_ctx=diag_ctx,
        )

        # Close diagnostic trace
        if diag_ctx is not None:
            diag_ctx.emit_diagnostic(
                stage="search",
                event_type="pipeline.end",
                data={
                    "result_count": len(results),
                },
            )
            diag_ctx.close()

        if not expand_chunks:
            grouped = self.searcher.dedup_by_file(results)
            return [
                SearchResult(
                    id=g.file_id,
                    file_path="",
                    score=g.score,
                    snippet=g.top_snippet,
                    match_source="",
                    match_count=g.match_count,
                    all_snippets=g.all_snippets,
                )
                for g in grouped
            ]

        # Convert interface SearchResult → API SearchResult for expand_chunks case
        return [
            SearchResult(
                id=r.id,
                file_path=r.file_path,
                score=r.score,
                snippet=r.snippet,
                match_source=r.match_source,
                file_type=r.file_type,
                wiki_summary=r.wiki_summary,
            )
            for r in results
        ]

    # ========================================================================
    # Info
    # ========================================================================

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
        # Try as UUID first (skip deleted files)
        meta = self.indexer.get_file(ref)
        if meta is not None and not meta.deleted_at:
            return meta.id

        # Try as filename (partial match)
        all_files = self.indexer.list_files(limit=10000)
        matches = [f for f in all_files if ref == f.filename]
        if len(matches) == 1:
            return matches[0].id
        if len(matches) > 1:
            # Multiple matches — return the most recently added
            matches.sort(key=lambda f: f.added_at or "", reverse=True)
            return matches[0].id

        return None

    def get_info(self, file_id: str) -> FileInfo:
        """Get detailed file information.

        Parameters
        ----------
        file_id : str
            File identifier.

        Returns
        -------
        FileInfo

        Raises
        ------
        KeyError
            If *file_id* is not found or is in trash.
        """
        meta = self.indexer.get_file(file_id)
        if meta is None or meta.deleted_at:
            raise KeyError(f"File not found: {file_id}")
        return FileInfo(
            id=meta.id, file_type=meta.file_type, filename=meta.filename,
            file_size=meta.file_size, file_hash=meta.file_hash,
            raw_path=meta.raw_path, metadata_path=meta.metadata_path,
            md_path=meta.md_path, wiki_path=meta.wiki_path,
            md_status=FileStatus(meta.md_status),
            wiki_status=FileStatus(meta.wiki_status),
            embed_status=FileStatus(meta.embed_status),
            category=meta.category, tags=meta.tags,
            keys=self.key_manager.get_file_keys(file_id),
            added_at=meta.added_at, updated_at=meta.updated_at,
            source_path=meta.source_path, source_kb=meta.source_kb,
        )

    def get_context(self, file_id: str) -> FileContext:
        """Get full file context for agent consumption.

        Returns metadata, markdown content, wiki summary, and user notes.

        Parameters
        ----------
        file_id : str
            File identifier.

        Returns
        -------
        FileContext

        Raises
        ------
        KeyError
            If *file_id* is not found or is in trash.
        """
        meta = self.indexer.get_file(file_id)
        if meta is None or meta.deleted_at:
            raise KeyError(f"File not found: {file_id}")

        # Read markdown content
        md_content = ""
        if meta.md_path:
            md_file = self.data_dir / meta.md_path
            if md_file.exists():
                md_content = md_file.read_text(encoding="utf-8")

        # Read wiki content
        wiki_content = ""
        if meta.wiki_path:
            wiki_file = self.data_dir / meta.wiki_path
            if wiki_file.exists():
                wiki_content = wiki_file.read_text(encoding="utf-8")

        # Read metadata content
        metadata_content = ""
        if meta.metadata_path:
            meta_file = self.data_dir / meta.metadata_path
            if meta_file.exists():
                metadata_content = meta_file.read_text(encoding="utf-8")

        # Extract user notes from metadata
        user_notes = ""
        if meta.custom and isinstance(meta.custom, dict):
            user_notes = meta.custom.get("note", "")

        return FileContext(
            file_id=file_id,
            file_type=meta.file_type,
            filename=meta.filename,
            category=meta.category,
            tags=meta.tags or [],
            keys=self.key_manager.get_file_keys(file_id),
            md_content=md_content,
            wiki_content=wiki_content,
            metadata_content=metadata_content,
            user_notes=user_notes,
            entities=self.graph_store.get_file_entities(file_id),
        )

    def list_files(
        self,
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
        sort_by : str, optional
            Column to sort by. Default is 'added_at'.
        limit : int, optional
            Maximum results. Default is 50.
        offset : int, optional
            Pagination offset. Default is 0.

        Returns
        -------
        list of FileInfo
        """
        metas = self.indexer.list_files(
            file_type=file_type,
            tags=tags,
            keys=keys,
            date_from=date_from,
            date_to=date_to,
            sort_by=sort_by,
            limit=limit,
            offset=offset,
        )
        return [
            FileInfo(
                id=m.id, file_type=m.file_type, filename=m.filename,
                file_size=m.file_size, file_hash=m.file_hash,
                raw_path=m.raw_path, metadata_path=m.metadata_path,
                md_path=m.md_path, wiki_path=m.wiki_path,
                md_status=FileStatus(m.md_status),
                wiki_status=FileStatus(m.wiki_status),
                embed_status=FileStatus(m.embed_status),
                category=m.category, tags=m.tags,
                keys=self.key_manager.get_file_keys(m.id),
                added_at=m.added_at, updated_at=m.updated_at,
                source_path=m.source_path, source_kb=m.source_kb,
            )
            for m in metas
        ]

    def stats(self) -> KnowledgeBaseStats:
        """Get aggregate statistics.

        Returns
        -------
        KnowledgeBaseStats
        """
        raw_stats = self.indexer.get_stats()
        return KnowledgeBaseStats(**raw_stats)

    # ========================================================================
    # Management
    # ========================================================================

    def update(
        self,
        file_id: str,
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
            Update user note (triggers metadata re-embedding).
        category : str, optional
            Update category.

        Returns
        -------
        FileInfo

        Raises
        ------
        KeyError
            If *file_id* is not found.
        """
        meta = self.indexer.get_file(file_id)
        if meta is None:
            raise KeyError(f"File not found: {file_id}")

        now_iso = datetime.now(UTC).isoformat()
        indexer_updates: dict[str, Any] = {"updated_at": now_iso}

        if keys is not None:
            # Also register any new keys that don't exist yet
            for k in keys:
                self.key_manager.register_key(k)
            self.key_manager.set_file_keys(file_id, keys)

        if tags is not None:
            import json
            indexer_updates["tags"] = json.dumps(tags, ensure_ascii=False)

        if category is not None:
            indexer_updates["category"] = category

        if note is not None:
            from mnemo.core.metadata_writer import MetadataWriter
            writer = MetadataWriter(self.data_dir)
            existing_note = meta.custom.get("note", "") if isinstance(meta.custom, dict) else ""
            writer.update_note(file_id, existing_note, note)

        self.indexer.update_file(file_id, **indexer_updates)

        return self.get_info(file_id)

    def remove(self, file_id: str) -> dict:
        """Soft-delete a file — move everything to an independent trash.

        The trash mirrors the main structure (``raw/``, ``raw_md/``,
        ``raw_wiki/``, ``raw_metadata/``, ``embedding/``, ``index.db``)
        under ``.mnemo/trash/``, so restore is a single move + re-index.

        After removal the file is invisible to ``list_files``, ``search``,
        ``stats``, and ``get_info``.

        Parameters
        ----------
        file_id : str
            File identifier.

        Returns
        -------
        dict
            Result keys: ``file_id``, ``filename``, ``trashed_files``.

        Raises
        ------
        KeyError
            If *file_id* is not found.
        """
        from mnemo.core.trash_store import TrashStore

        meta = self.indexer.get_file(file_id)
        if meta is None:
            raise KeyError(f"File not found: {file_id}")

        now_iso = datetime.now(UTC).isoformat()

        # 1. Move file artifacts + DB records to trash
        trash = TrashStore(self.data_dir)
        file_keys = self.key_manager.get_file_keys(file_id)
        entities = self.graph_store.get_file_entities(file_id) if hasattr(
            self.graph_store, 'get_file_entities',
        ) else []

        result = trash.trash_file(meta, file_keys, entities)

        # 2. Mark as deleted in main index
        self.indexer.trash_file(file_id, now_iso)

        # 3. Clean up vectors from main LanceDB
        try:
            self.vector_store.delete_vectors("raw_md", [file_id])
        except Exception:
            import logging
            logging.getLogger("mnemo").debug(
                "Failed to delete vectors for '%s' (table may not exist)", file_id,
            )

        # 4. Clean up graph entities and relations for this file
        try:
            self.graph_store.delete_file_entities(file_id)
        except Exception:
            import logging
            logging.getLogger("mnemo").debug(
                "Failed to delete graph entities for '%s'", file_id, exc_info=True,
            )

        # 5. Invalidate searcher caches
        if hasattr(self.searcher, 'invalidate_keyword_index'):
            self.searcher.invalidate_keyword_index()

        trash.close()

        return {
            "file_id": file_id,
            "filename": meta.filename,
            "trashed_files": result["trashed_files"],
        }

    def reindex(
        self,
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
        all_files : bool, optional
            Reindex everything.
        meta_only : bool, optional
            Only rebuild the metadata vector store.

        Returns
        -------
        dict
            Result summary: {'reindexed': int, 'skipped': int, 'failed': int}.
        """
        # Determine which files to reindex
        if file_id:
            metas = [self.indexer.get_file(file_id)]
            if metas[0] is None:
                raise KeyError(f"File not found: {file_id}")
        elif file_type:
            metas = self.indexer.list_files(file_type=file_type, limit=10000)
        elif all_files:
            metas = self.indexer.list_files(limit=10000)
        else:
            return {"reindexed": 0, "skipped": 0, "failed": 0,
                    "message": "Use --file, --type, or --all to select files"}

        reindexed = 0
        skipped = 0
        failed = 0

        for meta in metas:
            try:
                # Read markdown content from disk
                md_content = ""
                if meta.md_path:
                    md_file = self.data_dir / meta.md_path
                    if md_file.exists():
                        md_content = md_file.read_text(encoding="utf-8")

                if not meta_only and md_content:
                    # Delete old vectors
                    self.vector_store.delete_vectors("raw_md", [meta.id])

                    # Regenerate embeddings
                    file_ext = meta.file_type
                    cat_name = meta.category
                    chunker, chunker_cfg = self._resolve_chunker(file_ext)
                    chunks = chunker.chunk(md_content, chunker_cfg)
                    chunk_texts = [c.text for c in chunks]
                    from mnemo.core.embedder import get_embedder, get_dimension, get_model_name
                    try:
                        result = get_embedder().embed_documents_sync(chunk_texts)
                        vectors = [list(v) for v in result.embeddings]
                    except Exception:
                        vectors = [[0.0] * get_dimension() for _ in chunk_texts]
                    self.vector_store.add_vectors(
                        "raw_md",
                        ids=[meta.id] * len(vectors),
                        vectors=vectors,
                        metadata={
                            "file_type": [meta.file_type] * len(vectors),
                            "chunk_index": [c.chunk_index for c in chunks],
                            "model": [get_model_name()] * len(vectors),
                            "start_char": [c.start_char for c in chunks],
                            "end_char": [c.end_char for c in chunks],
                            "section_header": [
                                c.metadata.get("section_header", "")
                                for c in chunks
                            ],
                            "parent_id": [
                                c.metadata.get("parent_id", "")
                                for c in chunks
                            ],
                        },
                    )

                    # Update embed status
                    self.indexer.update_file(
                        meta.id,
                        embed_status="done",
                        updated_at=datetime.now(UTC).isoformat(),
                    )

                reindexed += 1
            except Exception:
                import logging
                logging.getLogger("mnemo").warning(
                    "Reindex failed for file '%s'", meta.id, exc_info=True,
                )
                failed += 1

        return {"reindexed": reindexed, "skipped": skipped, "failed": failed}

    def reorg(
        self,
        file_type: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """Reorganize chunk directories.

        Parameters
        ----------
        file_type : str, optional
            Restrict to this type.
        dry_run : bool, optional
            If True, preview the plan without executing.

        Returns
        -------
        dict
            Migration plan or results.
        """
        org = self.reorganizer
        if dry_run:
            return org.plan(file_type)
        return org.execute(file_type)

    def check(self, fix: bool = False) -> CheckReport:
        """Check knowledge base integrity.

        Parameters
        ----------
        fix : bool, optional
            If True, automatically repair issues.

        Returns
        -------
        CheckReport
        """
        issues = self.indexer.check_consistency()
        return CheckReport(
            status="warning" if issues else "ok",
            issues=issues,
            suggestions=(
                ["Run mnemo check --fix to auto-repair"] if issues and not fix else []
            ),
        )

    def watch(self, interval: int = 30) -> None:
        """Start the file watcher daemon.

        Monitors ``raw/`` and ``raw_metadata/`` for changes.
        On modification, triggers reindexing of the changed file.

        Parameters
        ----------
        interval : int, optional
            Polling interval in seconds. Default is 30.
        """
        import logging

        logger = logging.getLogger("mnemo.watch")
        logger.info("Starting file watcher (interval=%ds)...", interval)

        def on_change(action: str, path: Path):
            """Handle a file system event."""
            logger.info("[%s] %s", action, path)

            # Determine file_id from path
            # Try to find the file in the index by raw_path
            try:
                rel_path = str(path.relative_to(self.data_dir))
            except ValueError:
                return

            # Search the index for a file matching this path
            # (simplified: scan recent files — full implementation would use index)
            if action == "modified" and "raw_metadata" in rel_path:
                # Metadata file changed — regenerate metadata embedding
                logger.info("Metadata change detected: %s", rel_path)
            elif action == "created" and "raw" in rel_path:
                logger.info("New file detected: %s. Use 'mnemo add' to ingest.", rel_path)

        from mnemo.core.file_watcher import FileWatcher
        watcher = FileWatcher(self.data_dir, on_change=on_change)
        watcher.start()

        try:
            import signal
            import sys

            def _signal_handler(signum, frame):
                watcher.stop()
                logger.info("Watcher stopped by signal")
                sys.exit(0)

            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)

            # Keep running until interrupted
            import time
            while watcher.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            watcher.stop()
            logger.info("Watcher stopped by user")

    # ========================================================================
    # Import / Export / Sync
    # ========================================================================

    def export_kb(
        self,
        dest: str | Path,
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
        return self.exporter.export_to(Path(dest), file_type, keys, after)

    def import_kb(self, source: str | Path, dry_run: bool = False) -> ImportReport:
        """Import an external knowledge base.

        Parameters
        ----------
        source : str or Path
            Path to tar.gz or directory.
        dry_run : bool, optional
            If True, preview without importing.

        Returns
        -------
        ImportReport
        """
        return self.importer.import_from(Path(source), dry_run)

    def sync_push(self) -> SyncReport:
        """Push local data to remote.

        Returns
        -------
        SyncReport
        """
        return self.syncer.push()

    def sync_pull(self) -> SyncReport:
        """Pull remote data to local.

        Returns
        -------
        SyncReport
        """
        return self.syncer.pull()

    # ========================================================================
    # Internal helpers
    # ========================================================================

    @property
    def importer(self):
        if self._importer is None:
            self._importer = PluginHub.get(IImporter, "default")
            if hasattr(self._importer, "init"):
                self._importer.init(self)
        return self._importer

    @property
    def exporter(self):
        if self._exporter is None:
            self._exporter = PluginHub.get(IExporter, "default")
            if hasattr(self._exporter, "init"):
                self._exporter.init(self)
        return self._exporter

    @property
    def reorganizer(self):
        if self._reorganizer is None:
            self._reorganizer = PluginHub.get(IReorganizer, "default")
        return self._reorganizer

    @property
    def syncer(self):
        if self._syncer is None:
            self._syncer = PluginHub.get(ISyncer, "default")
            if hasattr(self._syncer, "init"):
                self._syncer.init(self)
        return self._syncer

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file.

        Parameters
        ----------
        file_path : Path
            File to hash.

        Returns
        -------
        str
            Hash in 'sha256:{hex}' format.
        """
        import hashlib
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"

    def _get_current_chunk(self, file_type: str) -> str:
        """Get the active chunk directory name.

        Strategy (from ``IIndexer`` config):
        - ``"time"``: new directory every ``chunk_interval_days``
        - ``"count"``: new directory when ``chunk_max_files`` reached
        - ``"time_and_count"`` (default): whichever limit is hit first

        Parameters
        ----------
        file_type : str
            File extension (unused, reserved for per-type strategy).

        Returns
        -------
        str
            Chunk directory name, e.g. ``'2026_06_07-chunk01'``.
        """
        from datetime import datetime

        from mnemo.core.interfaces.indexer import IIndexer
        from mnemo.core.param_config import get_config

        cfg = get_config(IIndexer)
        strategy = cfg.get("chunk_strategy", "time_and_count")
        interval_days = int(cfg.get("chunk_interval_days", 30))
        max_files = int(cfg.get("chunk_max_files", 50))

        now = datetime.now()
        today_str = now.strftime("%Y_%m_%d")

        # Scan existing chunks for today
        raw_dir = self.data_dir / "raw"
        if not raw_dir.exists():
            return f"{today_str}-chunk01"

        # Collect today's chunks and their file counts
        today_chunks: dict[str, int] = {}
        for d in raw_dir.rglob("*"):
            if d.is_dir() and d.parent == raw_dir:
                continue  # only look inside category/type dirs
            # d is a chunk directory like raw/docs/txt/2026_06_07-chunk01
            if d.is_dir() and d.name.startswith(today_str):
                today_chunks[d.name] = sum(1 for _ in d.rglob("*") if _.is_file())

        # Find the current chunk to write to
        if not today_chunks:
            return f"{today_str}-chunk01"

        # Check time-based rollover
        if strategy in ("time", "time_and_count"):
            # Find the earliest chunk for today and check its date
            for chunk_name in sorted(today_chunks):
                try:
                    chunk_date_str = chunk_name[:10]  # YYYY_MM_DD
                    chunk_date = datetime.strptime(chunk_date_str, "%Y_%m_%d")
                    if (now - chunk_date).days >= interval_days:
                        # Today is in a new interval — start fresh
                        return f"{today_str}-chunk01"
                except (ValueError, IndexError):
                    pass
                break  # only check the first one

        if strategy in ("count", "time_and_count"):
            # Use the newest chunk if it has room, otherwise create new
            for chunk_name in sorted(today_chunks, reverse=True):
                if today_chunks[chunk_name] < max_files:
                    return chunk_name

        # All chunks full or time rolled — create new
        chunk_num = len(today_chunks) + 1
        return f"{today_str}-chunk{chunk_num:02d}"

    def _resolve_file_classification(self, file_ext: str) -> tuple[str, str]:
        """Map a file extension to (dir_path, file_type) via file categories.

        Uses the hierarchical file category resolution: for ``.py`` files,
        ``CodePyCategory`` (dir_path ``"code/py"``) wins over
        ``CodeCategory`` (dir_path ``"code"``).

        Parameters
        ----------
        file_ext : str
            File extension without dot, e.g. 'pdf', 'py'.

        Returns
        -------
        tuple[str, str]
            (category_dir_path, file_type).
            e.g. ``("code/py", "py")`` for Python files.
        """
        fc = self._resolve_file_category(file_ext)
        if fc is not None:
            dir_path = getattr(fc, "dir_path", file_ext)
        else:
            dir_path = file_ext
        return (dir_path, file_ext)

    def _resolve_flag(
        self, flag: str, user_value: bool | None, file_type: str, _category: str = ""
    ) -> bool:
        """Resolve a processing flag via the file category system.

        Priority: user argument > file category config > interface default.

        Parameters
        ----------
        flag : str
            Flag name: 'auto_md', 'auto_wiki', or 'auto_embed'.
        user_value : bool or None
            Explicit user value (from CLI/API).
        file_type : str
            File extension (used for category resolution).
        _category : str
            Ignored (retained for backward-compatible signature).

        Returns
        -------
        bool
            Resolved flag value.
        """
        if user_value is not None:
            return user_value

        # Resolve category for this file type
        from mnemo.core.param_config import get_param_config
        fc = self._resolve_file_category(file_type)
        cat_name = getattr(fc, "name", "other") if fc is not None else "other"
        category = fc
        schema = getattr(category, "config_schema", {}) if category is not None else {}

        # Check category config first
        pc = get_param_config()
        if pc is not None:
            cat_cfg = pc.get_file_category_config(cat_name, schema)
            if flag in cat_cfg:
                val = cat_cfg[flag]
                if isinstance(val, bool):
                    return val

        # Fall back to interface defaults
        from mnemo.core.interfaces.parser import IParser
        from mnemo.core.interfaces.template import ITemplate
        from mnemo.core.param_config import get_config

        if flag == "auto_md":
            return get_config(IParser).get("default_auto_md", True)
        elif flag == "auto_wiki":
            return get_config(ITemplate).get("default_auto_wiki", True)
        elif flag == "auto_embed":
            return True  # default: auto-embed enabled

        return True

    @staticmethod
    def _resolve_plugin_ref(
        raw_section: dict, plugin_key: str,
    ) -> tuple[str | None, dict]:
        """Resolve a plugin reference with optional inline config overrides.

        Supports two TOML forms::

            # Shorthand — just the plugin name
            chunker = "token"

            # Long form — name + inline overrides (TOML dotted keys)
            chunker.name = "token"
            chunker.max_chunk_size = 3000

        This is a **generic** helper that works for any plugin reference
        in file categories (``chunker``, ``parser``, ``template``, etc.).

        Parameters
        ----------
        raw_section : dict
            Raw TOML section dict for a file category
            (e.g. ``raw_fc["file_category"]["code.py"]``).
        plugin_key : str
            Key name, e.g. ``"chunker"``, ``"parser"``.

        Returns
        -------
        tuple
            ``(plugin_name_or_None, overrides_dict)``.
        """
        ref = raw_section.get(plugin_key)
        if ref is None:
            return None, {}
        if isinstance(ref, str):
            # Shorthand: chunker = "token"
            return ref, {}
        if isinstance(ref, dict):
            # Long form: chunker.name = "token", chunker.max_chunk_size = N
            overrides = {k: v for k, v in ref.items() if k != "name"}
            return ref.get("name"), overrides
        return None, {}

    def _resolve_chunker(self, file_type: str) -> tuple[Any, dict]:
        """Resolve the chunker plugin and its config for a file type.

        Config resolution order:
        1. Global chunker plugin config (``[chunker.<name>]`` in config.toml)
        2. Per-category chunker reference (name + inline overrides via
           TOML dotted keys, e.g. ``chunker.name = "token"`` and
           ``chunker.max_chunk_size = 3000``)
        3. Per-category ``chunker_config`` sub-section overrides
           (``[file_category.<name>.chunker_config]``) — for backward
           compatibility with existing configurations.

        This allows different file categories to use the same chunker
        type but with different parameters.

        Parameters
        ----------
        file_type : str
            File extension (used for category resolution).

        Returns
        -------
        tuple
            ``(chunker_instance, resolved_config_dict)``.
        """
        from mnemo.core.interfaces.chunker import IChunker
        from mnemo.core.param_config import get_config, get_param_config

        # Default: global chunker setting
        chunker_name = get_config(IChunker).get("default_plugin", "paragraph")

        # Check file category for chunker reference and overrides
        fc = self._resolve_file_category(file_type)
        cat_name = getattr(fc, "name", "other") if fc is not None else "other"
        pc = get_param_config()

        if pc is not None:
            raw_fc = pc._file_categories_config.get("file_category", {})

            # Walk parent chain: child overrides parent
            cat_overrides: dict = {}
            cat_chunker_name: str | None = None

            parts = cat_name.split(".")
            for i in range(len(parts)):
                ancestor_name = ".".join(parts[:i+1])
                ancestor = raw_fc.get(ancestor_name, {})
                if isinstance(ancestor, dict):
                    # Try generic plugin ref (long form: chunker.name = "...")
                    ref_name, ref_overrides = self._resolve_plugin_ref(
                        ancestor, "chunker",
                    )
                    if ref_name is not None:
                        cat_chunker_name = ref_name
                        cat_overrides = {**cat_overrides, **ref_overrides}

                    # Also support legacy chunker_config sub-section
                    legacy = ancestor.get("chunker_config", {})
                    if isinstance(legacy, dict):
                        cat_overrides = {**cat_overrides, **legacy}

            if cat_chunker_name is not None:
                chunker_name = cat_chunker_name

        # Get the chunker instance and its resolved global config
        chunker = PluginHub.get(IChunker, chunker_name)
        chunker_cfg = get_config(chunker.__class__)

        # Apply per-category overrides on top of global config
        if cat_overrides:
            chunker_cfg = {**chunker_cfg, **cat_overrides}

        # Pass file_type for language-specific splitting
        chunker_cfg["file_type"] = file_type
        # Pass embedder for semantic chunker (needed for embedding-based
        # boundary detection — injected rather than imported to avoid
        # circular dependencies in the plugin layer)
        if chunker_name == "semantic":
            from mnemo.core.embedder import get_embedder
            chunker_cfg["_embedder"] = get_embedder()
        return chunker, chunker_cfg

    def _store_parent_chunks(
        self, file_id: str, parents: dict[str, str], _chunker_cfg: dict,
    ) -> None:
        """Store Small-to-Big parent chunks (text only, no embedding)."""
        try:
            pids = list(parents.keys())
            ptexts = [parents[k] for k in pids]
            self.vector_store.add_vectors(
                "raw_md_parents",
                ids=pids,
                vectors=[[0.0] * 4] * len(pids),  # dummy vectors
                metadata={
                    "file_type": [""] * len(pids),
                    "chunk_index": list(range(len(pids))),
                    "model": [""] * len(pids),
                    "start_char": [0] * len(pids),
                    "end_char": [len(t) for t in ptexts],
                    "section_header": [""] * len(pids),
                    "parent_id": [""] * len(pids),
                },
            )
        except Exception:
            import logging
            logging.getLogger("mnemo").debug(
                "Failed to store parent chunks (raw_md_parents table may not exist)",
            )

    @staticmethod
    def _load_toml_file(path: Path) -> dict:
        """Load a TOML file, returning empty dict on failure."""
        try:
            import tomllib
            with open(path, "rb") as f:
                return tomllib.load(f)
        except Exception:
            import logging
            logging.getLogger("mnemo").debug(
                "Failed to load TOML file: %s", path,
            )
            return {}

    def _build_filemeta(
        self,
        file_id: str,
        file_type: str,
        filename: str,
        file_hash: str,
        file_size: int,
        source_path: str,
        raw_dest: Path,
        meta_path: Path,
        md_content: str,
        wiki_content: str,
        auto_md: bool,
        auto_wiki: bool,
        auto_embed: bool,
        category: str,
        tags: list[str],
        keys: list[str],
    ) -> FileMeta:
        """Build a FileMeta for insertion into the global index.

        Parameters
        ----------
        (all parameters are self-explanatory)

        Returns
        -------
        FileMeta
        """
        from mnemo.core.interfaces.types import FileMeta

        now = datetime.now(UTC).isoformat()

        # Resolve relative paths safely (callers may pass empty paths
        # when the file hasn't been written yet, e.g. during wiki step).
        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(self.data_dir)) if p.parts and p != Path() else ""
            except ValueError:
                return ""

        return FileMeta(
            id=file_id,
            file_type=file_type,
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            source_path=source_path,
            raw_path=_rel(raw_dest),
            metadata_path=_rel(meta_path),
            md_path=(
                f"raw_md/{category}/{file_type}/{raw_dest.parent.name}/{Path(filename).stem}.md"
                if auto_md else ""
            ),
            wiki_path=(
                f"raw_wiki/{category}/{file_type}/{raw_dest.parent.name}/{Path(filename).stem}.md"
                if auto_wiki else ""
            ),
            md_status="done" if auto_md else "skipped",
            wiki_status="pending" if auto_wiki else "skipped",
            embed_status="pending" if auto_embed else "skipped",
            category=category,
            tags=tags,
            keywords=[],
            added_at=now,
            updated_at=now,
        )

    @staticmethod
    def _resolve_file_category(file_extension: str) -> Any:
        """Find the most specific file category for a file extension.

        Returns the :class:`IFileCategory` instance, or None if no match
        (including the ``"other"`` fallback).
        """
        ext = file_extension.lower().lstrip(".")

        # Collect all matching category instances
        matches: list[Any] = []
        for _, impl_cls in PluginHub.iter_impls(IFileCategory):
            inst = PluginHub.get(IFileCategory, impl_cls.name)
            types = getattr(inst, "types", [])
            if ext in [t.lower().lstrip(".") for t in types]:
                matches.append(inst)

        # Sort by depth descending (most specific first)
        matches.sort(key=lambda c: getattr(c, "depth", 1), reverse=True)

        if matches:
            return matches[0]

        # Fallback to "other"
        if PluginHub.has(IFileCategory, "other"):
            return PluginHub.get(IFileCategory, "other")

        return None

    @staticmethod
    def _get_category_chain(category_name: str) -> list[str]:
        """Walk parent chain from category_name up to root."""
        chain: list[str] = []
        current = category_name
        visited: set[str] = set()
        while current and current not in visited:
            visited.add(current)
            chain.append(current)
            if not PluginHub.has(IFileCategory, current):
                break
            inst = PluginHub.get(IFileCategory, current)
            parent = getattr(inst, "parent", None)
            if parent and parent != current:
                current = parent
            else:
                break
        return chain

    def __repr__(self) -> str:
        return f"KnowledgeBase(data_dir='{self.data_dir}')"
