"""BM25 keyword search with jieba Chinese tokenization.

KeywordSearcher implements ISearcher and is registered as a
standalone searcher plugin — no embedder or graph dependency.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from rank_bm25 import BM25Okapi

from mnemo.core.interfaces import ISearcher
from mnemo.core.interfaces.param_spec import Param
from mnemo.core.interfaces.types import GroupedResult, SearchResult


def _get_jieba():
    """Import jieba with its noisy model-loading output suppressed.

    jieba logs model-loading messages at INFO level to its own logger.
    We silence them so the CLI stays clean.
    """
    import jieba
    jieba_logger = logging.getLogger("jieba")
    jieba_logger.setLevel(logging.WARNING)
    return jieba


class KeywordSearcher(ISearcher):
    """BM25-based keyword search over markdown/wiki/metadata files.

    Self-resolves ``data_dir`` from the unified param config.
    Does NOT require embeddings or graph entities — ingestion
    can skip those stages when this searcher is active.

    Builds an in-memory BM25 index from all ``.md`` files under
    the search directories. Supports ID pre-filtering.
    """

    __plugin_impl__ = True
    name = "keyword"

    config_schema = {
        "default_mode": Param(
            type="str", default="keyword",
            desc="Default search mode (keyword-only for this searcher)"
        ),
        "default_limit": Param(
            type="int", default=10, desc="Default max search results"
        ),
    }

    @property
    def required_capabilities(self) -> set[str]:
        """Keyword searcher only needs markdown content — no embeddings."""
        return {'markdown_content'}

    def __init__(self, data_dir: Path | None = None):
        # Self-resolve data_dir from param config if not provided
        if data_dir is not None:
            self._data_dir = data_dir
        else:
            from mnemo.core.param_config import get_param_config
            pc = get_param_config()
            if pc is not None and hasattr(pc, '_config'):
                global_cfg = pc._config.get("global", {})
                data_dir_str = global_cfg.get("data_dir", "~/mnemo-data")
            else:
                data_dir_str = "~/mnemo-data"
            self._data_dir = Path(data_dir_str).expanduser()

        self._index: BM25Okapi | None = None
        self._file_ids: list[str] = []        # parallel to _corpus
        self._file_texts: list[str] = []       # original text for snippet extraction
        self._corpus: list[list[str]] = []     # tokenized
        self._mtime_cache: dict[str, float] = {}  # file_path -> mtime

    # -- Build ---------------------------------------------------------------

    def build_index(self, force: bool = False) -> None:
        """Build (or incrementally update) the BM25 index.

        On first call, scans all ``.md`` files.  On subsequent calls,
        only re-indexes files whose modification time has changed.

        Parameters
        ----------
        force : bool
            If True, force a full rebuild even if nothing changed.
        """
        jieba = _get_jieba()

        if force:
            self._file_ids.clear()
            self._file_texts.clear()
            self._corpus.clear()
            self._mtime_cache.clear()

        search_dirs = [
            self._data_dir / "raw_md",
            self._data_dir / "raw_wiki",
            self._data_dir / "raw_metadata",
        ]

        # Collect current files and their mtimes
        current_files: dict[str, Path] = {}  # file_id -> path
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for md_file in search_dir.rglob("*.md"):
                file_id = md_file.stem
                current_files[file_id] = md_file

        # Remove deleted files from index
        removed = [fid for fid in self._file_ids if fid not in current_files]
        if removed:
            removed_set = set(removed)
            keep_indices = [
                i for i, fid in enumerate(self._file_ids)
                if fid not in removed_set
            ]
            self._file_ids = [self._file_ids[i] for i in keep_indices]
            self._file_texts = [self._file_texts[i] for i in keep_indices]
            self._corpus = [self._corpus[i] for i in keep_indices]

        # Add or update changed files
        for file_id, md_file in current_files.items():
            mtime = md_file.stat().st_mtime
            cached = self._mtime_cache.get(file_id, 0)
            if mtime <= cached and file_id in self._file_ids:
                continue  # Unchanged — skip
            try:
                text = md_file.read_text(encoding="utf-8")
                tokens = [t for t in jieba.cut(text) if t.strip()]
                if file_id in self._file_ids:
                    idx = self._file_ids.index(file_id)
                    self._file_texts[idx] = text
                    self._corpus[idx] = tokens
                else:
                    self._file_ids.append(file_id)
                    self._file_texts.append(text)
                    self._corpus.append(tokens)
                self._mtime_cache[file_id] = mtime
            except Exception:
                import logging
                logging.getLogger("mnemo").debug(
                    "Skipping unreadable file during keyword index build: %s", md_file,
                )

        if self._corpus:
            self._index = BM25Okapi(self._corpus)
        else:
            self._index = None

    # -- Search --------------------------------------------------------------

    def search(
        self,
        query: str,
        mode: str = "keyword",
        candidate_ids: list[str] | None = None,
        limit: int = 10,
        file_types: list[str] | None = None,
        with_metadata: bool = True,
        on_progress: Callable[[str, str], None] | None = None,
        diagnose: bool = False,
        diagnostic_ctx: Any = None,
    ) -> list:
        """BM25 keyword search.

        Parameters
        ----------
        query : str
            Search query.
        candidate_ids : list of str, optional
            Pre-filter: only search among these file IDs.
        limit : int
            Maximum number of results.

        Returns
        -------
        list of dict
            Each dict: ``{'id': str, 'score': float, 'snippet': str}``.
        """
        if on_progress:
            on_progress("keyword", "in_progress")
        if self._index is None:
            if on_progress:
                on_progress("keyword", "done:0")
            return []

        jieba = _get_jieba()
        tokens = [t for t in jieba.cut(query) if t.strip()]
        scores = self._index.get_scores(tokens)

        # Build (idx, score) pairs, filter by candidate_ids if needed
        results: list[tuple[int, float]] = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            fid = self._file_ids[idx]
            if candidate_ids and fid not in candidate_ids:
                continue
            results.append((idx, score))

        # Sort descending, take top K
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:limit]

        if on_progress:
            on_progress("keyword", f"done:{len(results)}")

        return [
            {
                "id": self._file_ids[idx],
                "score": float(score),
                "snippet": self._extract_snippet(self._file_texts[idx], query, 150),
                "match_source": "keyword",
            }
            for idx, score in results
        ]

    def dedup_by_file(self, results: list[SearchResult]) -> list[GroupedResult]:
        """Merge multi-chunk results by file."""
        grouped: dict[str, GroupedResult] = {}
        for r in results:
            if r.id not in grouped:
                grouped[r.id] = GroupedResult(
                    file_id=r.id, score=r.score, top_snippet=r.snippet,
                    match_count=1, all_snippets=[r.snippet],
                    file_type=r.file_type, wiki_summary=r.wiki_summary,
                )
            else:
                g = grouped[r.id]
                g.score = max(g.score, r.score)
                g.match_count += 1
                g.all_snippets.append(r.snippet)
        return sorted(grouped.values(), key=lambda g: g.score, reverse=True)

    @staticmethod
    def _extract_snippet(text: str, query: str, context_len: int = 150) -> str:
        """Find the query in text and return surrounding context.

        Parameters
        ----------
        text : str
        query : str
        context_len : int

        Returns
        -------
        str
        """
        if not query:
            return text[:context_len]

        jieba = _get_jieba()
        tokens = [t for t in jieba.cut(query) if t.strip()]
        # Try exact match first
        pos = text.lower().find(query.lower())
        if pos >= 0:
            start = max(0, pos - context_len // 2)
            end = min(len(text), pos + len(query) + context_len // 2)
            snippet = text[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet += "..."
            return snippet

        # Try token match
        for token in tokens:
            if len(token) < 2:
                continue
            pos = text.lower().find(token.lower())
            if pos >= 0:
                start = max(0, pos - context_len // 2)
                end = min(len(text), pos + len(token) + context_len // 2)
                st = "..." if start > 0 else ""
                en = "..." if end < len(text) else ""
                return st + text[start:end] + en

        return text[:context_len] + "..."
