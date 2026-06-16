"""Typed parameter specification — replaces raw dict config_schema entries.

``Param`` replaces the anonymous ``{"type": "str", "default": "...",
"desc": "..."}`` dicts with a typed, self-documenting pydantic model.
The ``env_var`` field eliminates the ``ENV::`` magic-string placeholder
pattern — a parameter that reads from an environment variable simply
declares ``env_var="THE_VAR"`` instead of ``default="ENV::THE_VAR"``.

Usage::

    from mnemo.core.interfaces.param_spec import Param

    config_schema: dict[str, Param] = {
        "api_key": Param(type="str", env_var="EMBED_API_KEY",
                         desc="API key"),
        "model":   Param(type="str", default="text-embedding-3-small",
                         desc="Model name"),
    }
    # NOTE: Shared settings like 'dimension' now live in [global]
    # (see GLOBAL_CONFIG_SCHEMA in param_config.py)

Resolution priority (highest first):
1. ``MNEMO_*`` runtime env-var override (bypasses schema)
2. TOML explicit value (user set in config file)
3. ``env_var`` environment variable (if declared)
4. ``default`` schema default
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Param(BaseModel):
    """Typed specification for a single configurable parameter.

    Attributes
    ----------
    type : str
        Value type: ``"str"``, ``"int"``, ``"float"``, ``"bool"``.
    default : Any
        Default value when no TOML override or env var is present.
    desc : str
        Human-readable description (shown as TOML comment).
    env_var : str or None
        Environment variable name that backs this parameter,
        e.g. ``"EMBED_API_KEY"``.  When set, the resolution pipeline
        reads from ``os.environ`` as a fallback between the TOML
        override and the schema default.
    """

    type: str = Field(
        default="str",
        description="Value type: 'str', 'int', 'float', or 'bool'",
    )
    default: Any = Field(
        default="",
        description="Default value when no override is present",
    )
    desc: str = Field(
        default="",
        description="Human-readable description (shown as TOML comment)",
    )
    env_var: str | None = Field(
        default=None,
        description="Environment variable name that backs this parameter",
    )
