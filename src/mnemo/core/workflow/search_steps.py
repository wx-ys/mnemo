"""Search workflow step implementations — real KB integration.

Replaces the stubs in ``compat.py`` with actual search logic using
the KnowledgeBase's searcher, vector store, and graph store.

Registered functions:
- plan_search_strategy
- run_vector_search
- run_keyword_search
- run_graph_search
- fuse_search_results
- rerank_results
- filter_and_format_results
"""

from __future__ import annotations

import logging
from typing import Any

from mnemo.api.types import SearchMode, SearchResult
from mnemo.core.workflow.context import WorkflowContext
from mnemo.core.workflow.step import StepRegistry

logger = logging.getLogger("mnemo.workflow")


def _get_kb(ctx: WorkflowContext) -> Any:
    if ctx.kb is None:
        raise RuntimeError("KnowledgeBase not available in WorkflowContext")
    return ctx.kb


# ============================================================================
# Plan search strategy
# ============================================================================


@StepRegistry.register_function("plan_search_strategy")
async def plan_search_strategy(ctx: WorkflowContext) -> dict[str, Any]:
    """Analyze query judgment and rewrites → search plan.

    Reads from context data:
    - rewritten_queries (from rewrite_query step)
    - query_judgment (from judge_query step)
    - config (workflow config flags)

    Produces a plan dict consumed by the parallel search steps.
    """
    kb = _get_kb(ctx)
    config = ctx.config

    # Determine which search modes to enable
    modes: list[str] = config.get("search_modes", ["vector", "keyword"])

    # Graph search is off by default (more expensive)
    if config.get("search_graph_enabled", False):
        modes.append("graph")

    # Default weights (can be adjusted by query_judgment)
    weights = {
        "vector": float(config.get("fusion_vector_weight", 0.5)),
        "keyword": float(config.get("fusion_keyword_weight", 0.3)),
        "graph": float(config.get("fusion_graph_weight", 0.2)),
    }

    # Normalize weights for enabled modes only
    active_weights = {m: weights.get(m, 0.0) for m in modes}
    total = sum(active_weights.values()) or 1.0
    active_weights = {k: v / total for k, v in active_weights.items()}

    # Extract queries from rewritten_queries data
    queries: list[str] = [ctx.get_input("query", "")]
    rewritten = ctx.get_output("rewritten_queries", "")
    if isinstance(rewritten, str) and rewritten:
        # Try to parse LLM output as list of queries
        import json
        try:
            parsed = json.loads(rewritten)
            if isinstance(parsed, list):
                queries = [q for q in parsed if isinstance(q, str)]
            elif isinstance(parsed, dict):
                queries = [q for q in parsed.values() if isinstance(q, str)]
        except (json.JSONDecodeError, TypeError):
            queries = [q.strip() for q in rewritten.split("\n") if q.strip()]

    limit = int(config.get("search_limit", 10))
    file_types = ctx.get_input("file_types")
    keys = ctx.get_input("keys")

    return {
        "modes": modes,
        "weights": active_weights,
        "queries": queries,
        "limit": limit,
        "file_types": file_types,
        "keys": keys,
    }


# ============================================================================
# Parallel search channels
# ============================================================================


@StepRegistry.register_function("run_vector_search")
async def run_vector_search(ctx: WorkflowContext) -> dict[str, Any]:
    """Vector ANN search via the KB searcher."""
    kb = _get_kb(ctx)
    plan = ctx.get_output("plan_search_strategy", {})
    queries = plan.get("queries", [])
    limit = plan.get("limit", 10)
    file_types = plan.get("file_types")
    keys = plan.get("keys")

    all_results: list[SearchResult] = []
    for query in queries:
        try:
            results = kb.search(
                query, mode=SearchMode.VECTOR, limit=limit,
                file_types=file_types, keys=keys,
            )
            all_results.extend(results)
        except Exception:
            logger.debug("Vector search failed for query: %s", query, exc_info=True)

    return {"results": all_results, "source": "vector", "count": len(all_results)}


@StepRegistry.register_function("run_keyword_search")
async def run_keyword_search(ctx: WorkflowContext) -> dict[str, Any]:
    """BM25 keyword search via the KB searcher."""
    kb = _get_kb(ctx)
    plan = ctx.get_output("plan_search_strategy", {})
    queries = plan.get("queries", [])
    limit = plan.get("limit", 10)
    file_types = plan.get("file_types")
    keys = plan.get("keys")

    all_results: list[SearchResult] = []
    for query in queries:
        try:
            results = kb.search(
                query, mode=SearchMode.KEYWORD, limit=limit,
                file_types=file_types, keys=keys,
            )
            all_results.extend(results)
        except Exception:
            logger.debug("Keyword search failed for query: %s", query, exc_info=True)

    return {"results": all_results, "source": "keyword", "count": len(all_results)}


@StepRegistry.register_function("run_graph_search")
async def run_graph_search(ctx: WorkflowContext) -> dict[str, Any]:
    """Knowledge graph traversal search."""
    kb = _get_kb(ctx)
    plan = ctx.get_output("plan_search_strategy", {})

    # Graph search is optional — only run if graph store is available
    try:
        if hasattr(kb, 'graph_store') and kb.graph_store is not None:
            # Use the configured searcher's graph-aware search
            if hasattr(kb.searcher, 'graph_search'):
                results = kb.searcher.graph_search(
                    query=plan.get("queries", [""])[0],
                    limit=plan.get("limit", 10),
                )
                return {"results": results, "source": "graph", "count": len(results)}
    except Exception:
        logger.debug("Graph search failed", exc_info=True)

    return {"results": [], "source": "graph", "count": 0}


# ============================================================================
# Fuse results (RRF)
# ============================================================================


@StepRegistry.register_function("fuse_search_results")
async def fuse_search_results(ctx: WorkflowContext) -> dict[str, Any]:
    """Reciprocal Rank Fusion of multi-source results."""
    plan = ctx.get_output("plan_search_strategy", {})
    weights = plan.get("weights", {"vector": 0.6, "keyword": 0.4})

    # Collect results from each channel
    sources: dict[str, list[SearchResult]] = {}
    for step_name in ("run_vector_search", "run_keyword_search", "run_graph_search"):
        step_output = ctx.get_output(step_name)
        if isinstance(step_output, dict):
            results = step_output.get("results", [])
            source = step_output.get("source", step_name)
            if results:
                sources[source] = results

    if not sources:
        return {"fused": [], "count": 0}

    # RRF scoring
    k = 60  # RRF constant
    fused_scores: dict[str, tuple[float, SearchResult]] = {}

    for source_name, results in sources.items():
        weight = weights.get(source_name, 0.3)
        for rank, result in enumerate(results):
            rrf = weight / (k + rank + 1)
            if result.id in fused_scores:
                prev_score, prev_result = fused_scores[result.id]
                fused_scores[result.id] = (prev_score + rrf, prev_result)
            else:
                fused_scores[result.id] = (rrf, result)

    # Sort by fused score descending
    sorted_results = sorted(
        fused_scores.values(), key=lambda x: x[0], reverse=True,
    )
    fused = [
        SearchResult(
            id=r.id,
            file_path=r.file_path,
            score=score,
            snippet=r.snippet,
            match_source=r.match_source,
            match_count=getattr(r, 'match_count', 1),
            all_snippets=getattr(r, 'all_snippets', [r.snippet]),
            file_type=getattr(r, 'file_type', ''),
        )
        for score, r in sorted_results
    ]

    return {"fused": fused, "count": len(fused)}


# ============================================================================
# Rerank
# ============================================================================


@StepRegistry.register_function("rerank_results")
async def rerank_results(ctx: WorkflowContext) -> dict[str, Any]:
    """Rerank fused results using cross-encoder or LLM."""
    fused_output = ctx.get_output("fuse_search_results", {})
    fused = fused_output.get("fused", [])

    if not fused:
        return {"reranked": [], "count": 0}

    config = ctx.config
    method = config.get("rerank_method", "score")

    if method == "none":
        return {"reranked": fused, "count": len(fused)}

    if method == "cross-encoder":
        try:
            from rerankers import Reranker
            query = ctx.get_input("query", "")

            ranker = Reranker(
                "mixedbread-ai/mxbai-rerank-xsmall-v1",
                model_type="cross-encoder",
            )
            docs = [r.snippet for r in fused]
            reranked = ranker.rank(query=query, docs=docs)

            result_map = {r.snippet: r for r in fused}
            merged: list = []
            for rr in reranked.results:
                key = rr.text if hasattr(rr, 'text') else str(rr.document.text)
                if key in result_map:
                    r = result_map[key]
                    r.score = float(getattr(rr, 'score', r.score))
                    merged.append(r)

            # Append any not covered
            covered = {r.snippet for r in merged}
            for r in fused:
                if r.snippet not in covered:
                    merged.append(r)

            return {"reranked": merged, "count": len(merged)}
        except Exception:
            logger.debug("Cross-encoder rerank failed, using score sort", exc_info=True)

    # Default: score-based sort
    sorted_results = sorted(fused, key=lambda r: r.score, reverse=True)
    return {"reranked": sorted_results, "count": len(sorted_results)}


# ============================================================================
# Filter & Format
# ============================================================================


@StepRegistry.register_function("filter_and_format_results")
async def filter_and_format_results(ctx: WorkflowContext) -> dict[str, Any]:
    """Threshold filter and final format of search results."""
    rerank_output = ctx.get_output("rerank_results", {})
    results = rerank_output.get("reranked", [])

    config = ctx.config
    min_score = float(config.get("search_min_score", 0.0))
    max_results = int(config.get("search_limit", 10))

    # Apply threshold and limit
    filtered = [r for r in results if r.score >= min_score]
    limited = filtered[:max_results]

    return {
        "results": limited,
        "count": len(limited),
        "total_found": len(results),
    }


# ============================================================================
# Ask steps
# ============================================================================


@StepRegistry.register_function("assemble_rag_context")
async def assemble_rag_context(ctx: WorkflowContext) -> dict[str, Any]:
    """Assemble context window from search results for RAG."""
    search_output = ctx.get_output("filter_and_format_results", {})
    results = search_output.get("results", [])

    max_chars = int(ctx.config.get("max_context_chars", 8000))
    parts: list[str] = []
    total = 0

    for i, r in enumerate(results):
        snippet = (r.snippet or "").strip()
        header = f"[{i + 1}] ({getattr(r, 'file_type', 'file') or 'file'})"

        available = max_chars - total - len(header) - 10
        if available <= 0:
            break

        if len(snippet) > available:
            snippet = snippet[:available] + "..."

        parts.append(f"{header}\n{snippet}\n")
        total += len(parts[-1])

    context_text = "\n".join(parts)
    return {
        "context_text": context_text,
        "source_count": len(parts),
        "total_chars": len(context_text),
    }
