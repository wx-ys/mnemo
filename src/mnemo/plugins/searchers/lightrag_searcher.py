"""LightRAG searcher: vector + entity graph + BM25 hybrid.

Self-resolves all dependencies from registries — no manual DI needed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mnemo.core.interfaces import ISearcher
from mnemo.core.interfaces.param_spec import Param
from mnemo.core.interfaces.types import GroupedResult, SearchResult
from mnemo.plugins.searchers.keyword_searcher import KeywordSearcher


class LightRAGSearcher(ISearcher):
    """LightRAG: vector ANN + entity graph expansion + BM25 keyword.

    Three retrieval channels are fused via Reciprocal Rank Fusion (RRF).
    All dependencies are self-resolved from registries in ``__init__``.
    Config is read from the unified parameter config system.
    """

    __plugin_impl__ = True
    name = "default"

    # ── Config schema ──────────────────────────────────────────────────

    config_schema: dict[str, Param] = {
        "vector_store_plugin": Param(
            type="str", default="lancedb",
            desc="Vector store plugin name"
        ),
        "graph_enabled": Param(
            type="bool", default=True,
            desc="Enable graph-enhanced entity search"
        ),
        "weights.vector": Param(
            type="float", default=0.4,
            desc="RRF weight for vector channel"
        ),
        "weights.keyword": Param(
            type="float", default=0.3,
            desc="RRF weight for keyword (BM25) channel"
        ),
        "weights.graph": Param(
            type="float", default=0.3,
            desc="RRF weight for graph (entity) channel"
        ),
        "graph_max_hops": Param(
            type="int", default=2,
            desc="Max graph traversal hops for entity expansion"
        ),
    }

    # ── Capability declaration ─────────────────────────────────────────

    @property
    def required_capabilities(self) -> set[str]:
        """LightRAG needs embeddings and markdown for all modes;
        graph entities are needed for hybrid mode (optional).
        """
        caps = {'embeddings', 'markdown_content'}
        if self._graph_enabled:
            # Check actual availability via the expander
            expander = self._get_graph_expander()
            if expander.enabled:
                caps.add('graph_entities')
        return caps

    # ── Constructor (self-resolves DI) ─────────────────────────────────

    def __init__(self):
        from mnemo.core.param_config import get_config
        from mnemo.core.plugin_base import PluginHub
        from mnemo.core.interfaces import IVectorStore, IIndexer, IKeyManager

        cfg = get_config(self.__class__)

        # Embedder is retrieved from the global singleton (core/embedder.py).
        # No injection needed — call get_embedder() at search time.

        vector_store_name = cfg.get("vector_store_plugin", "lancedb")
        self._vector_store = PluginHub.get(IVectorStore, vector_store_name)

        # Graph subsystem via GraphExpander (handles DI internally)
        self._graph_enabled = bool(cfg.get("graph_enabled", True))

        # Resolve indexer and key manager
        self._indexer = PluginHub.get(IIndexer, "sqlite")
        self._key_manager = PluginHub.get(IKeyManager, "sqlite")

        # data_dir is injected via init() by KnowledgeBase.
        # Fallback for legacy usage (searcher created without KB).
        self._data_dir: Path | None = None

        # Keyword searcher (lazy, not a registry plugin)
        self._keyword_searcher: KeywordSearcher | None = None

        # Graph expander (lazy — self-resolves DI internally)
        self._graph_expander: Any = None

        # RRF parameters — resolved from own config_schema
        self._RRF_K = int(cfg.get("rrf_k", 60))
        self._WEIGHT_VECTOR = float(cfg.get("weights.vector", 0.4))
        self._WEIGHT_KEYWORD = float(cfg.get("weights.keyword", 0.3))
        self._WEIGHT_GRAPH = float(cfg.get("weights.graph", 0.3))
        self._GRAPH_MAX_HOPS = int(cfg.get("graph_max_hops", 2))

    # ── Lifecycle (ISearcher.init) ─────────────────────────────────────

    def init(self, data_dir: Any) -> None:
        """Bind to a data directory (called by KnowledgeBase after creation).

        Overrides the no-op default from ISearcher.
        """
        self._data_dir = Path(data_dir)

    # ── ISearcher ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        candidate_ids: list[str] | None = None,
        limit: int = 10,
        file_types: list[str] | None = None,
        with_metadata: bool = True,
        on_progress: Callable[[str, str], None] | None = None,
        diagnose: bool = False,
        diagnostic_ctx: Any = None,
    ) -> list[SearchResult]:
        """Execute LightRAG search.

        Parameters
        ----------
        query : str
        mode : str
            'hybrid' (all channels), 'vector', 'keyword'.
        candidate_ids : list of str, optional
        limit : int
        file_types : list of str, optional
        with_metadata : bool
        on_progress : callable, optional
            ``(stage, status)`` callback for progress display.
        diagnose : bool, optional
            If True, emit detailed diagnostic data.
        diagnostic_ctx : DiagnosticContext, optional
            Carries trace file path and preview configs.

        Returns
        -------
        list of SearchResult
        """

        vector_results: list[dict] = []
        keyword_results: list[dict] = []
        graph_file_ids: list[str] = []

        # -- Build plugin info labels for progress --------------------------
        from mnemo.core.embedder import get_embedder, get_model_name, get_dimension
        emb_model = get_model_name()

        # 1. Vector ANN search
        if mode in ("hybrid", "vector"):
            if on_progress:
                on_progress(
                    "vector",
                    f"in_progress:Embedder[{emb_model}]",
                )
            try:
                result = get_embedder().embed_query_sync(query)
                query_vec = list(result.embeddings[0])
            except Exception:
                query_vec = [0.0] * get_dimension()

            # -- Diagnostic: query embedding -------------------------------
            if diagnostic_ctx is not None:
                from mnemo.core.diagnostics import truncate_vector
                diagnostic_ctx.emit_diagnostic(
                    stage="query_embedding",
                    data={
                        "query": query,
                        "model": emb_model,
                        "dimension": get_dimension(),
                        "vector_preview": truncate_vector(
                            query_vec, diagnostic_ctx.max_vector_preview_dims,
                        ),
                    },
                )

            tables = ["raw_md", "raw_wiki"] if with_metadata else ["raw_md"]
            for tbl_name in tables:
                try:
                    tbl_results = self._vector_store.search(
                        tbl_name, query_vec,
                        candidate_ids=candidate_ids,
                        limit=limit * 2,
                        filters=None,  # no model filter — all vectors use same embedder
                    )
                    for r in tbl_results:
                        r["match_source"] = f"vector:{tbl_name}"
                    vector_results.extend(tbl_results)

                    # -- Diagnostic: vector ANN per table -----------------
                    if diagnostic_ctx is not None:
                        from mnemo.core.diagnostics import compute_distance_stats
                        distances = [
                            r.get("_distance", 0.0) for r in tbl_results
                        ]
                        per_file = [
                            {"file_id": r.get("id", ""),
                             "distance": r.get("_distance", 0.0),
                             "score": 1.0 - float(r.get("_distance", 0.0))}
                            for r in tbl_results[:20]
                        ]
                        diagnostic_ctx.emit_diagnostic(
                            stage="vector_ann",
                            data={
                                "table": tbl_name,
                                "result_count": len(tbl_results),
                                "distance_stats": compute_distance_stats(distances),
                                "per_file": per_file,
                            },
                        )

                except Exception as e:
                    import logging
                    logging.getLogger("mnemo").warning(
                        "Vector search on '%s' failed: %s", tbl_name, e,
                    )
            # Enrich vector results with snippets (LanceDB rows don't include text)
            self._enrich_vector_results(vector_results, preserve_distance=diagnose)
            if on_progress:
                on_progress("vector", f"done:{len(vector_results)}")

        # 2. BM25 keyword search
        if mode in ("hybrid", "keyword"):
            if on_progress:
                on_progress("keyword", "in_progress:BM25[jieba]")
            kw = self._get_keyword_searcher()
            keyword_results = kw.search(query, candidate_ids=candidate_ids,
                                         limit=limit * 2)

            # -- Diagnostic: keyword BM25 ------------------------------------
            if diagnostic_ctx is not None:
                top_scores = [r.get("score", 0.0) for r in keyword_results[:5]]
                diagnostic_ctx.emit_diagnostic(
                    stage="keyword_bm25",
                    data={
                        "result_count": len(keyword_results),
                        "top_scores": top_scores,
                    },
                )

            if on_progress:
                on_progress("keyword", f"done:{len(keyword_results)}")

        # 3. Graph expansion (LightRAG enhancement)
        if mode == "hybrid" and self._graph_enabled:
            if on_progress:
                on_progress(
                    "graph",
                    f"in_progress:IGraph[hops={self._GRAPH_MAX_HOPS}]",
                )
            graph_file_ids = self._get_graph_expander().expand(query)

            # -- Diagnostic: graph expansion --------------------------------
            if diagnostic_ctx is not None:
                diagnostic_ctx.emit_diagnostic(
                    stage="graph_expand",
                    data={
                        "file_ids": graph_file_ids[:50],
                        "count": len(graph_file_ids),
                        "max_hops": self._GRAPH_MAX_HOPS,
                    },
                )

            if on_progress:
                on_progress("graph", f"done:{len(graph_file_ids)}")

        # 4. RRF fusion
        if on_progress:
            on_progress("fuse", "in_progress")
        merged = self._rrf_fuse(
            vector_results, keyword_results, graph_file_ids,
            limit=limit,
            diagnose=diagnose, diagnostic_ctx=diagnostic_ctx,
        )
        if on_progress:
            on_progress("fuse", f"done:{len(merged)}")

        # 5. Apply file_type filter
        if file_types:
            merged = [r for r in merged if r.get("file_type", "") in file_types]

        return [
            SearchResult(
                id=r.get("id", ""),
                file_path="",
                score=r.get("score", 0.0),
                snippet=r.get("snippet", ""),
                match_source=r.get("match_source", ""),
                file_type=r.get("file_type", ""),
            )
            for r in merged[:limit]
        ]

    def dedup_by_file(self, results: list[SearchResult]) -> list[GroupedResult]:
        """Merge multi-chunk results by file."""
        grouped: dict[str, GroupedResult] = {}
        for r in results:
            if r.id not in grouped:
                grouped[r.id] = GroupedResult(
                    file_id=r.id,
                    score=r.score,
                    top_snippet=r.snippet,
                    match_count=1,
                    all_snippets=[r.snippet],
                    file_type=r.file_type,
                    wiki_summary=r.wiki_summary,
                )
            else:
                g = grouped[r.id]
                g.score = max(g.score, r.score)
                g.match_count += 1
                g.all_snippets.append(r.snippet)
        return sorted(grouped.values(), key=lambda g: g.score, reverse=True)

    # ── Internal ───────────────────────────────────────────────────────

    @staticmethod
    def _dedup_by_id(results: list[dict]) -> list[dict]:
        """Keep only the best (earliest) occurrence of each file ID.

        Prevents multi-chunk files from dominating RRF fusion by
        contributing one RRF point per chunk.
        """
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in results:
            fid = r.get("id", "")
            if fid and fid not in seen:
                seen.add(fid)
                deduped.append(r)
        return deduped

    def _rrf_fuse(
        self,
        vector_results: list[dict],
        keyword_results: list[dict],
        graph_file_ids: list[str],
        limit: int = 10,
        k: int | None = None,
        diagnose: bool = False,
        diagnostic_ctx: Any = None,
    ) -> list[dict[str, Any]]:
        """Reciprocal Rank Fusion across retrieval channels.

        Each channel is deduplicated by file ID before RRF so that
        multi-chunk files don't accumulate outsized weight.
        """
        rrf_k = k if k is not None else self._RRF_K

        scores: dict[str, float] = {}
        snippets: dict[str, str] = {}
        sources: dict[str, list[str]] = {}

        # For diagnostics: track per-channel scores
        channel_scores: dict[str, dict[str, float]] = {
            "vector": {}, "keyword": {}, "graph": {},
        }

        # Vector channel — deduplicate so each file appears once
        for rank, r in enumerate(self._dedup_by_id(vector_results)):
            fid = r["id"]
            ch_score = self._WEIGHT_VECTOR / (rrf_k + rank + 1)
            scores[fid] = scores.get(fid, 0.0) + ch_score
            channel_scores["vector"][fid] = round(ch_score, 6)
            if fid not in snippets:
                snippets[fid] = r.get("snippet", "")
            sources.setdefault(fid, []).append(r.get("match_source", "vector"))

        # Keyword channel — deduplicate so each file appears once
        for rank, r in enumerate(self._dedup_by_id(keyword_results)):
            fid = r["id"]
            ch_score = self._WEIGHT_KEYWORD / (rrf_k + rank + 1)
            scores[fid] = scores.get(fid, 0.0) + ch_score
            channel_scores["keyword"][fid] = round(ch_score, 6)
            if fid not in snippets:
                snippets[fid] = r.get("snippet", "")
            sources.setdefault(fid, []).append("keyword")

        # Graph channel — file IDs are already unique
        for rank, fid in enumerate(graph_file_ids):
            ch_score = self._WEIGHT_GRAPH / (rrf_k + rank + 1)
            scores[fid] = scores.get(fid, 0.0) + ch_score
            channel_scores["graph"][fid] = round(ch_score, 6)
            sources.setdefault(fid, []).append("graph")

        # Sort by score descending
        sorted_ids = sorted(scores, key=lambda fid: scores[fid], reverse=True)

        # -- Diagnostic: RRF fusion ------------------------------------------
        if diagnostic_ctx is not None:
            # Cap per-channel scores to top 50 file IDs
            capped_channel_scores = {}
            for ch_name, ch_dict in channel_scores.items():
                sorted_ch = sorted(ch_dict.items(), key=lambda x: x[1], reverse=True)
                capped_channel_scores[ch_name] = dict(sorted_ch[:50])

            diagnostic_ctx.emit_diagnostic(
                stage="rrf_fuse",
                data={
                    "rrf_k": rrf_k,
                    "weights": {
                        "vector": self._WEIGHT_VECTOR,
                        "keyword": self._WEIGHT_KEYWORD,
                        "graph": self._WEIGHT_GRAPH,
                    },
                    "per_channel_scores": capped_channel_scores,
                    "fused_scores": {
                        fid: round(min(scores[fid], 1.0), 6)
                        for fid in sorted_ids[:limit]
                    },
                    "result_count": min(len(sorted_ids), limit),
                },
            )

        return [
            {
                "id": fid,
                "score": min(scores[fid], 1.0),
                "snippet": snippets.get(fid, ""),
                "match_source": ", ".join(sources.get(fid, [])),
            }
            for fid in sorted_ids[:limit]
        ]

    def _enrich_vector_results(
        self, results: list[dict], preserve_distance: bool = False,
    ) -> None:
        """Enrich raw LanceDB vector results with snippets from markdown files.

        LanceDB rows have ``start_char`` / ``end_char`` but no ``text`` column.
        This method reads the corresponding markdown file for each result and
        extracts a snippet.  Results are mutated in place.

        Parameters
        ----------
        results : list of dict
            Raw LanceDB search results to enrich in place.
        preserve_distance : bool
            If True, keep the ``_distance`` field (for diagnostics).
            Otherwise it is popped and converted to a 0-1 ``score``.
        """
        if not results:
            return

        # Group by file_id to minimize file reads
        by_file: dict[str, list[dict]] = {}
        for r in results:
            fid = r.get("id", "")
            if fid:
                by_file.setdefault(fid, []).append(r)

        for fid, chunk_results in by_file.items():
            # Get markdown path from indexer
            try:
                file_meta = self._indexer.get_file(fid)
                md_path_str = getattr(file_meta, "md_path", "") if file_meta else ""
            except Exception:
                md_path_str = ""

            if not md_path_str or self._data_dir is None:
                continue

            md_file = self._data_dir / md_path_str
            if not md_file.exists():
                continue

            try:
                md_text = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            for r in chunk_results:
                start = r.get("start_char", 0)
                end = r.get("end_char", len(md_text))
                if start < end and start < len(md_text):
                    snippet = md_text[start:min(end, len(md_text))]
                    r["snippet"] = snippet.strip()[:300]
                else:
                    r["snippet"] = md_text[:200]  # fallback: first 200 chars

                # Convert LanceDB _distance to a 0-1 score (cosine: lower = better)
                if preserve_distance:
                    distance = r.get("_distance", None)
                else:
                    distance = r.pop("_distance", None)
                if distance is not None:
                    r["score"] = max(0.0, min(1.0, 1.0 - float(distance)))

    def _get_keyword_searcher(self) -> KeywordSearcher:
        """Lazy-init the keyword searcher; rebuild index only once."""
        if self._keyword_searcher is None:
            data_dir = self._data_dir
            if data_dir is None:
                # Legacy fallback: searcher created without init() call
                from mnemo.core.param_config import get_param_config
                pc = get_param_config()
                if pc is not None and hasattr(pc, '_config'):
                    global_cfg = pc._config.get("global", {})
                    data_dir_str = global_cfg.get("data_dir", "~/mnemo-data")
                else:
                    data_dir_str = "~/mnemo-data"
                data_dir = Path(data_dir_str).expanduser()
            self._keyword_searcher = KeywordSearcher(data_dir)
            self._keyword_searcher.build_index()
        return self._keyword_searcher

    def _get_graph_expander(self) -> Any:
        """Lazy-init the graph expander (self-resolves DI internally)."""
        if self._graph_expander is None:
            from mnemo.plugins.searchers.graph_expander import GraphExpander
            self._graph_expander = GraphExpander(max_hops=self._GRAPH_MAX_HOPS)
        return self._graph_expander

    def invalidate_keyword_index(self) -> None:
        """Force rebuild of the BM25 index on next search (call after add/remove)."""
        self._keyword_searcher = None
