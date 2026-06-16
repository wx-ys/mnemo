"""Simple searcher — substring / grep-like search over markdown files.

No embedding, no vector store, no graph — purely scans ``raw_md/``
files for substring matches.  Fast, zero-config, and works offline.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from mnemo.core.interfaces import ISearcher
from mnemo.core.interfaces.param_spec import Param
from mnemo.core.interfaces.types import GroupedResult, SearchResult


class SimpleSearcher(ISearcher):
    """Grep-like substring search over markdown content.

    Scans ``raw_md/`` directory for ``.md`` files containing the
    query as a case-insensitive substring.  No embedding model,
    vector store, or graph store required.

    Use this when:
    - You want fast, deterministic substring matching
    - You don't have an embedding API key configured
    - You want minimal dependencies
    """

    __plugin_impl__ = True
    name = "simple"

    config_schema = {
        "default_mode": Param(
            type="str", default="keyword",
            desc="Default search mode (keyword-only for this searcher)"
        ),
        "default_limit": Param(
            type="int", default=10,
            desc="Default max search results"
        ),
        "case_sensitive": Param(
            type="bool", default=False,
            desc="Enable case-sensitive matching"
        ),
    }

    @property
    def required_capabilities(self) -> set[str]:
        """Simple searcher only needs markdown files on disk — nothing else."""
        return {'markdown_content'}

    def __init__(self):
        from mnemo.core.param_config import get_config, get_param_config

        cfg = get_config(self.__class__)
        self._case_sensitive = bool(cfg.get("case_sensitive", False))

        # Resolve data_dir from param config
        pc = get_param_config()
        if pc is not None and hasattr(pc, '_config'):
            global_cfg = pc._config.get("global", {})
            data_dir_str = global_cfg.get("data_dir", "~/mnemo-data")
        else:
            data_dir_str = "~/mnemo-data"
        self._data_dir = Path(data_dir_str).expanduser()

    # ── ISearcher ──────────────────────────────────────────────────────

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
    ) -> list[SearchResult]:
        """Search markdown files for substring matches.

        Parameters
        ----------
        query : str
            Search query (substring match).
        mode : str
            Ignored — this searcher only does keyword/grep.
        candidate_ids : list of str, optional
        limit : int
        file_types : list of str, optional
        with_metadata : bool
            If True, also scan raw_wiki/ and raw_metadata/.

        Returns
        -------
        list of SearchResult
        """
        if on_progress:
            on_progress("grep", "in_progress")
        q = query if self._case_sensitive else query.lower()

        search_dirs = [self._data_dir / "raw_md"]
        if with_metadata:
            search_dirs.append(self._data_dir / "raw_wiki")
            search_dirs.append(self._data_dir / "raw_metadata")

        results: list[SearchResult] = []

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for md_file in search_dir.rglob("*.md"):
                file_id = md_file.stem
                # Pre-filter by candidate_ids
                if candidate_ids and file_id not in candidate_ids:
                    continue
                # Pre-filter by file type (check parent dir name)
                if file_types:
                    # parent dir is the type, e.g. raw_md/docs/txt/file.md → type=txt
                    ftype = md_file.parent.name
                    if ftype not in file_types:
                        continue
                try:
                    text = md_file.read_text(encoding="utf-8")
                except Exception:
                    continue

                target = text if self._case_sensitive else text.lower()
                pos = target.find(q)
                if pos < 0:
                    continue

                # Extract snippet around match
                start = max(0, pos - 75)
                end = min(len(text), pos + len(query) + 75)
                snippet = text[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(text):
                    snippet += "..."

                results.append(SearchResult(
                    id=file_id,
                    file_path=str(md_file.relative_to(self._data_dir)),
                    score=1.0,  # exact substring match = max score
                    snippet=snippet,
                    match_source="grep",
                    file_type=md_file.parent.name,
                ))

            # Stop early if we have enough results
            if len(results) >= limit:
                break

        if on_progress:
            on_progress("grep", f"done:{len(results)}")
        return results[:limit]

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
