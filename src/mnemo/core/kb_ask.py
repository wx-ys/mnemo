"""RAG question-answering via the WorkflowEngine (V2).

Replaces the old linear AskPipeline with a config-driven DAG workflow.
Uses the ask.workflow.toml definition and pydantic-ai Agent steps.

Usage::

    from mnemo.core.kb_ask import AskPipeline

    pipeline = AskPipeline(api)
    response = pipeline.ask("What is the key contribution?")
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mnemo.api.ask_types import AskResponse, Citation
from mnemo.api.types import SearchMode, SearchResult

if TYPE_CHECKING:
    from mnemo.api.client import MnemoAPI

logger = logging.getLogger("mnemo.ask")


class AskPipeline:
    """RAG question-answering pipeline powered by WorkflowEngine.

    Parameters
    ----------
    api : MnemoAPI
        The knowledge base API instance.
    """

    def __init__(self, api: MnemoAPI) -> None:
        self._api = api
        self._ranker: Any = None
        self._last_usage: dict[str, int] = {}

    def ask(
        self,
        question: str,
        *,
        grounded: bool = True,
        limit: int = 10,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> AskResponse:
        """Answer a question using RAG over the knowledge base.

        Pipeline: expand query → multi-source search → rerank →
        assemble context → generate LLM answer with citations.
        """
        import mnemo.core.workflow.search_steps  # noqa: F401

        # 1. Search — multi-angle retrieval
        if on_progress:
            on_progress("search", "in_progress")

        queries = self._expand_query(question)
        all_results: list[SearchResult] = []
        seen: set[str] = set()

        for q in queries:
            results = self._api.search(
                q, mode=SearchMode.HYBRID, limit=limit,
            )
            for r in results:
                if r.id not in seen:
                    seen.add(r.id)
                    all_results.append(r)

        if on_progress:
            on_progress("search", f"done:{len(all_results)}")

        # 2. Rerank
        if on_progress:
            on_progress("rerank", "in_progress")
        reranked = self._rerank(question, all_results)
        if on_progress:
            on_progress("rerank", f"done:{len(reranked)}")

        # 3. Assemble context
        if on_progress:
            on_progress("assemble", "in_progress")
        context = self._assemble_context(reranked, max_chars=8000)
        if on_progress:
            on_progress("assemble", f"done:{len(context)} chars")

        # 4. Generate answer
        if on_progress:
            on_progress("generate", "in_progress")
        answer, citations = self._generate_answer(
            question, context, reranked, grounded,
        )
        if on_progress:
            on_progress("generate", f"done:{len(citations)} citations")

        return AskResponse(
            answer=answer,
            citations=citations,
            grounded=grounded,
            model=self._get_model_name(),
            tokens_used=self._last_usage.get("tokens_total", 0),
            tokens_input=self._last_usage.get("tokens_input", 0),
            tokens_output=self._last_usage.get("tokens_output", 0),
        )

    # ── Query expansion ─────────────────────────────────────────────────

    @staticmethod
    def _expand_query(query: str) -> list[str]:
        queries = [query]
        question_words = [
            "what", "how", "why", "when", "where", "who", "which",
            "什么", "如何", "怎么", "为什么", "哪里", "谁", "哪个",
            "请问", "能否", "可以", "解释一下",
        ]
        kw_query = query
        for w in question_words:
            kw_query = kw_query.replace(w, "")
        kw_query = " ".join(kw_query.split()).strip()
        if kw_query and kw_query != query:
            queries.append(kw_query)
        return queries[:3]

    # ── Rerank ──────────────────────────────────────────────────────────

    def _rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return results
        try:
            return self._rerank_with_model(query, results)
        except Exception:
            return sorted(results, key=lambda r: r.score, reverse=True)

    def _rerank_with_model(
        self, query: str, results: list[SearchResult],
    ) -> list[SearchResult]:
        from rerankers import Reranker

        if self._ranker is None:
            try:
                self._ranker = Reranker(
                    "mixedbread-ai/mxbai-rerank-xsmall-v1",
                    model_type="cross-encoder",
                )
            except Exception:
                return sorted(results, key=lambda r: r.score, reverse=True)

        docs = [r.snippet for r in results]
        reranked = self._ranker.rank(query=query, docs=docs)
        result_map = {r.snippet: r for r in results}
        merged: list[SearchResult] = []
        for rr in reranked.results:
            key = rr.text if hasattr(rr, 'text') else rr.document.text
            if key in result_map:
                r = result_map[key]
                r.score = float(getattr(rr, 'score', r.score))
                merged.append(r)
        covered = {r.snippet for r in merged}
        for r in results:
            if r.snippet not in covered:
                merged.append(r)
        return merged

    # ── Context assembly ────────────────────────────────────────────────

    @staticmethod
    def _assemble_context(results: list[SearchResult], max_chars: int = 8000) -> str:
        parts: list[str] = []
        total = 0
        for i, r in enumerate(results):
            snippet = r.snippet.strip()
            header = f"[{i + 1}] ({r.file_type or 'file'}) {snippet[:80]}..."
            available = max_chars - total - len(header) - 10
            if available <= 0:
                break
            if len(snippet) > available:
                snippet = snippet[:available] + "..."
            parts.append(f"{header}\n{snippet}\n")
            total += len(parts[-1])
        return "\n".join(parts)

    # ── Answer generation ───────────────────────────────────────────────

    def _generate_answer(
        self, question: str, context: str,
        sources: list[SearchResult], grounded: bool,
    ) -> tuple[str, list[Citation]]:
        system_prompt = (
            "You are a knowledge base assistant. Answer questions based "
            "ONLY on the provided context. "
            + (
                "If the context does not contain enough information, "
                "say so clearly — do not make up facts."
                if grounded
                else "Use the context as primary source; you may supplement "
                "with general knowledge when needed."
            )
            + "\n\n"
            "Cite sources inline using [N] markers that match the context "
            "numbers. Be concise and accurate."
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer the question using ONLY the context above. "
            "Include [N] citations for each claim."
        )

        try:
            answer_text = self._call_llm(system_prompt, user_prompt)
        except Exception:
            answer_text = (
                "[LLM unavailable — showing search results instead]\n\n"
                + context
            )

        citations = self._extract_citations(answer_text, sources)
        return answer_text, citations

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM and capture token usage in ``self._last_usage``."""
        from mnemo.core.agent_manager import AgentManager
        from mnemo.core.param_config import resolve_agent_config

        agent_cfg = resolve_agent_config()
        agent_name = agent_cfg.get("agent_name", "default")
        am = AgentManager.get_instance()
        if not am._initialized:
            return "[LLM unavailable: AgentManager not initialized]"

        try:
            agent = am.get_agent(agent_name, output_type=str)
            result = agent.run_sync(user_prompt, instructions=system_prompt)
            # Capture token usage
            try:
                usage = result.usage()
                self._last_usage = {
                    "tokens_input": usage.input_tokens,
                    "tokens_output": usage.output_tokens,
                    "tokens_total": usage.input_tokens + usage.output_tokens,
                    "requests": usage.requests,
                    "tool_calls": usage.tool_calls,
                }
            except Exception:
                self._last_usage = {}
            return result.output.strip() if result.output else ""
        except Exception as exc:
            self._last_usage = {}
            from mnemo.utils.api_errors import format_api_error
            return f"[LLM failed: {format_api_error(exc, context='LLM')}]"

    def _get_model_name(self) -> str:
        try:
            from mnemo.core.param_config import resolve_agent_config
            return resolve_agent_config().get("model", "?")
        except Exception:
            return "?"

    @staticmethod
    def _extract_citations(
        answer: str, sources: list[SearchResult],
    ) -> list[Citation]:
        import re
        cited = set[int]()
        for match in re.finditer(r"\[(\d+)\]", answer):
            try:
                n = int(match.group(1))
                if 1 <= n <= len(sources):
                    cited.add(n - 1)
            except ValueError:
                pass
        if not cited:
            cited = set(range(min(3, len(sources))))
        citations: list[Citation] = []
        for idx in sorted(cited):
            if idx < len(sources):
                s = sources[idx]
                citations.append(Citation(
                    file_id=s.id,
                    filename="",
                    snippet=s.snippet[:200],
                    relevance=round(s.score, 4),
                ))
        return citations
