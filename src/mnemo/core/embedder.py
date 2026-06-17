"""Embedder module — global pydantic-ai Embedder singleton.

One knowledge base uses one embedding model / dimension.  This module
provides a single entry point for the entire library::

    from mnemo.core.embedder import get_embedder, init_embedder

    # Once, in KB.__init__:
    init_embedder(config)

    # Anywhere:
    embedder = get_embedder()
    result = embedder.embed_documents_sync(["text1", "text2"])
    vectors = [list(v) for v in result.embeddings]
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.embeddings import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider

from mnemo.core.interfaces.param_spec import Param
from mnemo.core.param_config import resolve_api_key

# ---------------------------------------------------------------------------
# Embedder config schema — single source of truth for [embedder] section
# ---------------------------------------------------------------------------

EMBEDDER_CONFIG_SCHEMA: dict[str, Param] = {
    "model": Param(
        type="str", default="text-embedding-v4",
        desc="Embedding model name",
    ),
    "base_url": Param(
        type="str",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        desc="API base URL",
    ),
    "api_key": Param(
        type="str", default="EMBED_API_KEY", env_var="EMBED_API_KEY",
        desc="API key — env var name or literal key (e.g. EMBED_API_KEY or sk-xxx)",
    ),
    "batch_size": Param(
        type="int", default=10,
        desc="Batch size for embedding texts",
    ),
}

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_embedder: Embedder[Any] | None = None
_dimension: int = 1024
_model_name: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_embedder(config: dict[str, Any]) -> Embedder[Any]:
    """Initialize the global Embedder singleton from config.

    Reads the ``[embedder]`` section and ``[global].dimension``.
    Must be called once (by ``KB.__init__``).

    Parameters
    ----------
    config : dict
        Full merged config dict.

    Returns
    -------
    pydantic_ai.embeddings.Embedder
    """
    global _embedder, _dimension, _model_name

    emb_cfg = config.get("embedder", {})
    if not isinstance(emb_cfg, dict):
        emb_cfg = {}

    # Resolve api_key via the unified helper: TOML value is env var name
    # (e.g. "EMBED_API_KEY") or a literal key; falls back to env var if empty
    api_key = resolve_api_key(
        emb_cfg.get("api_key", ""),
        EMBEDDER_CONFIG_SCHEMA.get("api_key"),
    )
    base_url = emb_cfg.get("base_url", "") or None
    _model_name = str(emb_cfg.get("model", "text-embedding-v4"))

    global_cfg = config.get("global", {})
    if isinstance(global_cfg, dict):
        _dimension = int(global_cfg.get("dimension", 1024))

    provider = OpenAIProvider(base_url=base_url, api_key=api_key or "dummy-key")
    model = OpenAIEmbeddingModel(_model_name, provider=provider)
    _embedder = Embedder(model=model, defer_model_check=True)
    return _embedder


def get_embedder() -> Embedder[Any]:
    """Return the global pydantic-ai Embedder singleton.

    Raises
    ------
    RuntimeError
        If :func:`init_embedder` has not been called yet.
    """
    if _embedder is None:
        raise RuntimeError(
            "Embedder not initialized — call init_embedder() first"
        )
    return _embedder


def get_dimension() -> int:
    """Return the configured embedding vector dimension."""
    return _dimension


def get_model_name() -> str:
    """Return the configured embedding model name."""
    return _model_name
