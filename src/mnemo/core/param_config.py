"""
Unified parameter configuration system for Mnemo.

Provides a single entry point for both interface-level and plugin-level
configuration resolution:

- ``get_config(cls)`` — resolved dict with defaults, TOML overrides,
  ``ENV::`` resolution, and type casting.
- ``generate_toml_template()`` — auto-generate ``config.toml``
- ``generate_file_categories_toml()`` — auto-generate ``file_categories.toml``

Usage::

    from mnemo.core.param_config import get_config, init_param_config

    # Initialize once (done in KnowledgeBase.__init__)
    init_param_config(loaded_config_dict)

    # Interface-level config (e.g., IEmbedder → [embedder] section)
    cfg = get_config(IEmbedder)

    # Plugin-level config (e.g., OpenAIEmbedder → [embedder.openai] section)
    cfg = get_config(OpenAIEmbedder)
"""

from __future__ import annotations

import inspect
import os
import re
from typing import Any

from mnemo.core.interfaces.param_spec import Param
from mnemo.core.plugin_base import PluginHub


# ---------------------------------------------------------------------------
# Unified API key resolution
# ---------------------------------------------------------------------------


def resolve_api_key(value: str, param: Param | None = None) -> str:
    """Resolve an API key from a config value + environment variable.

    Resolution:
    1. *value* is non-empty → try ``os.environ[value]`` first (treat as
       env var name), fall back to the literal *value*
    2. *value* is empty + *param* has ``env_var`` → ``os.environ[param.env_var]``
    3. Otherwise → ``""``

    This means users can set any of these in their TOML:

    .. code-block:: toml

        api_key = "LLM_API_KEY"   # reads from $LLM_API_KEY
        api_key = "sk-abc123"     # literal key (no env var named "sk-abc123")
        # api_key not set         # falls back to param.env_var (e.g. "LLM_API_KEY")

    Parameters
    ----------
    value : str
        The config value (e.g. ``cfg.get("api_key", "")``).
    param : Param or None
        The parameter spec with optional ``env_var`` field.

    Returns
    -------
    str
        The resolved API key (may be empty string if not found).
    """
    key = value or ""
    if key:
        # Try as env var name first; fall back to literal value
        return os.environ.get(key, key)
    if param is not None and param.env_var:
        return os.environ.get(param.env_var, "")
    return ""

# ---------------------------------------------------------------------------
# Global config schema — single source of truth for [global] settings.
# ---------------------------------------------------------------------------

GLOBAL_CONFIG_SCHEMA: dict[str, Param] = {
    "name": Param(
        type="str", default="",
        desc="Human-readable name for this knowledge base",
    ),
    "mode": Param(
        type="str", default="auto",
        desc="Processing mode: 'auto' (all steps) or 'manual' (user-triggered)",
    ),
    "debug": Param(
        type="bool", default=False,
        desc="Enable debug logging",
    ),
    "dimension": Param(
        type="int", default=1024,
        desc="Default vector dimension for both embedding and vector storage (must be consistent!)",
    ),
    "default_agent": Param(
        type="str", default="default",
        desc="Default agent name (references [agent.<name>] section)",
    ),
}

# ---------------------------------------------------------------------------
# Section naming
# ---------------------------------------------------------------------------

def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case.

    ``"LLMProvider"`` → ``"llm_provider"``,
    ``"Embedder"`` → ``"embedder"``.
    """
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _registry_name_to_section(registry_name: str) -> str:
    """Convert a registry's human-readable name to a TOML section prefix.

    ``"Embedder"`` → ``"embedder"``,
    ``"LLMProvider"`` → ``"llm_provider"``.
    """
    return _camel_to_snake(registry_name)


# Cache: class → section name
_SECTION_CACHE: dict[type, str] = {}


def _get_section_for_class(cls: type) -> str:
    """Determine the TOML ``[section]`` prefix for a class.

    Walks registries to find which interface *cls* implements, then
    converts the registry name to snake_case.

    Returns the section prefix (e.g. ``"embedder"``), **not** the
    full ``[section.plugin_name]`` key.
    """
    if cls in _SECTION_CACHE:
        return _SECTION_CACHE[cls]

    for iface_name, iface in PluginHub.iter_interfaces():
        if issubclass(cls, iface):
            section = _registry_name_to_section(iface_name)
            _SECTION_CACHE[cls] = section
            return section

    # Fallback: derive from class name
    fallback = _camel_to_snake(cls.__name__)
    _SECTION_CACHE[cls] = fallback
    return fallback


def _is_interface_class(cls: type) -> bool:
    """Check if *cls* is an interface ABC (not a concrete plugin).

    Returns True for abstract base classes like ``IEmbedder``,
    ``ISearcher``, etc. Returns False for concrete plugin classes
    like ``OpenAIEmbedder``, ``OpenAIProvider``.
    """
    return inspect.isabstract(cls)


def _to_param(entry: Any) -> Param:
    """Normalize a schema entry to a :class:`Param`.

    Accepts both old-style ``dict`` entries (backward compat) and
    ``Param`` instances.  Old-style dicts are wrapped as
    ``Param(**entry)`` — any extra keys are ignored.
    """
    if isinstance(entry, Param):
        return entry
    if isinstance(entry, dict):
        return Param(
            type=entry.get("type", "str"),
            default=entry.get("default", ""),
            desc=entry.get("desc", ""),
            env_var=None,  # old style doesn't have env_var; ENV:: handled via BC path
        )
    return Param()


def _get_merged_schema(cls: type) -> dict[str, Param]:
    """Merge ``config_schema`` from *cls* and all its ancestor classes.

    Walks the MRO in reverse (bases first, then subclasses) so that
    subclass fields override parent fields.  Each value is normalized
    to a :class:`Param` via :func:`_to_param`.
    """
    merged: dict[str, Param] = {}
    for base in reversed(cls.__mro__):
        if base is object:
            continue
        schema = getattr(base, "config_schema", None)
        if schema and isinstance(schema, dict):
            merged |= {k: _to_param(v) for k, v in schema.items()}
    return merged


def _get_own_schema(cls: type) -> dict[str, Param]:
    """Get ``config_schema`` fields defined directly on *cls*.

    Only returns fields that are declared in *cls*'s own ``__dict__``
    (not inherited from parent classes).
    """
    own = cls.__dict__.get("config_schema", None)
    if not own:
        return {}
    return {k: _to_param(v) for k, v in own.items()}


# ---------------------------------------------------------------------------
# ParamConfig
# ---------------------------------------------------------------------------

class ParamConfig:
    """Unified parameter configuration resolver.

    Wraps the loaded config dict and provides per-class config resolution
    using each class's ``config_schema`` for defaults, types, and docs.

    Supports both:
    - **Interface-level** config: ``get_config(IEmbedder)`` → reads from
      ``[embedder]`` section, returning interface-wide params.
    - **Plugin-level** config: ``get_config(OpenAIEmbedder)`` → reads from
      ``[embedder.openai]`` section, returning plugin-specific params.

    Parameters
    ----------
    config : dict
        The fully-loaded config dict (from ConfigLoader).
    file_categories_config : dict, optional
        Separate config dict for file categories (from file_categories.toml).
    """

    def __init__(
        self,
        config: dict[str, Any],
        file_categories_config: dict[str, Any] | None = None,
    ) -> None:
        self._config = config
        self._file_categories_config = file_categories_config or {}

    # -- main API ------------------------------------------------------------

    def get_config(self, cls: type) -> dict[str, Any]:
        """Get resolved configuration for an interface or plugin class.

        Automatically merges ``config_schema`` from all ancestor classes
        (MRO order). Subclass fields override parent fields.

        - If *cls* is an **interface ABC** (e.g. ``IEmbedder``):
          resolves from ``[section]`` in TOML.
        - If *cls* is a **plugin class** (e.g. ``OpenAIEmbedder``):
          resolves from ``[section.plugin_name]`` in TOML, with
          ``[section]`` interface-level values as fallback.

        Parameters
        ----------
        cls : type
            An interface ABC or plugin class with ``config_schema``.

        Returns
        -------
        dict
            Resolved parameter dict. Empty dict if *cls* has no schema.
        """
        merged_schema = _get_merged_schema(cls)
        if not merged_schema:
            return {}

        section = _get_section_for_class(cls)
        plugin_name = getattr(cls, "name", None)

        # Interface ABC: resolve from [section] directly
        if _is_interface_class(cls) or not plugin_name:
            return self._resolve_config(section, merged_schema)

        # Plugin class: merge [section] + [section.plugin_name] + schema
        return self._resolve_plugin_config(section, plugin_name, merged_schema)

    def _resolve_config(
        self,
        section: str,
        schema: dict[str, Param],
    ) -> dict[str, Any]:
        """Resolve config from a flat ``[section]`` with *schema*."""
        result, toml_keys = self._apply_defaults_and_toml(section, schema)

        # Layer: plugin subsection (none for interface-level)
        self._apply_env_params(result, toml_keys, schema)
        return self._cast_types(result, schema)

    def _resolve_plugin_config(
        self,
        section: str,
        plugin_name: str,
        schema: dict[str, Param],
    ) -> dict[str, Any]:
        """Resolve config for a plugin: [section] + [section.name] layers."""
        result, toml_keys = self._apply_defaults_and_toml(section, schema)

        # Layer 2: [section.plugin_name] plugin-level values (override)
        toml_section: dict = self._config.get(section, {})
        if isinstance(toml_section, dict):
            toml_plugin = toml_section.get(plugin_name, {})
            if isinstance(toml_plugin, dict):
                for key, value in toml_plugin.items():
                    if key in result:
                        result[key] = self._resolve_toml_value(value)
                        toml_keys.add(key)

        self._apply_env_params(result, toml_keys, schema)
        return self._cast_types(result, schema)

    def _get_interface_config(
        self,
        section: str,
        schema: dict[str, Param],
    ) -> dict[str, Any]:
        """Resolve interface-level config from ``[section]``."""
        result, toml_keys = self._apply_defaults_and_toml(section, schema)
        self._apply_env_params(result, toml_keys, schema)
        return self._cast_types(result, schema)

    # -- Shared helpers -------------------------------------------------------

    def _apply_defaults_and_toml(
        self,
        section: str,
        schema: dict[str, Param],
    ) -> tuple[dict[str, Any], set[str]]:
        """Build the initial config dict from schema defaults + TOML overrides.

        Returns ``(result, toml_keys)`` where *toml_keys* tracks which
        keys were explicitly set in TOML (so env-var fallback can skip them).
        """
        result: dict[str, Any] = {}
        for key, param in schema.items():
            result[key] = param.default

        toml_keys: set[str] = set()
        toml_section: dict = self._config.get(section, {})
        if isinstance(toml_section, dict):
            for key, value in toml_section.items():
                if key in result and not isinstance(value, dict):
                    result[key] = self._resolve_toml_value(value)
                    toml_keys.add(key)

        return result, toml_keys

    @staticmethod
    def _resolve_toml_value(value: Any) -> Any:
        """Resolve a TOML value, handling legacy ``ENV::`` placeholders.

        If *value* is a string starting with ``ENV::``, the env var is
        read from ``os.environ``.  This provides backward compatibility
        with existing TOML files that still use the old placeholder format.
        """
        if isinstance(value, str) and value.startswith("ENV::"):
            return os.environ.get(value[5:], "")
        return value

    @staticmethod
    def _apply_env_params(
        result: dict[str, Any],
        toml_keys: set[str],
        schema: dict[str, Param],
    ) -> None:
        """Apply ``Param.env_var`` fallback for keys NOT overridden in TOML.

        Mutates *result* in place.
        """
        for key, param in schema.items():
            if key in toml_keys:
                continue  # user explicitly set this in TOML
            if param.env_var is None:
                continue
            env_val = os.environ.get(param.env_var)
            if env_val is not None:
                result[key] = env_val

    def get_config_by_name(
        self,
        section: str,
        plugin_name: str,
        schema: dict[str, Param],
    ) -> dict[str, Any]:
        """Get resolved config for a plugin identified by section + name.

        Supports **external / unregistered plugins**: callers can supply
        a *schema* directly without the plugin class being registered.

        Parameters
        ----------
        section : str
            TOML section prefix, e.g. ``"embedder"``.
        plugin_name : str
            Plugin name, e.g. ``"openai"``.
        schema : dict[str, Param]
            Schema dict (values normalized to :class:`Param`).

        Returns
        -------
        dict
            Resolved parameter dict.
        """
        return self._resolve_plugin_config(
            section, plugin_name,
            {k: _to_param(v) for k, v in schema.items()},
        )

    # -- File category config -----------------------------------------------

    def get_file_category_config(
        self,
        category_name: str,
        schema: dict[str, Param] | None = None,
    ) -> dict[str, Any]:
        """Get resolved config for a file category.

        Reads from ``self._file_categories_config`` (file_categories.toml).
        Falls back to schema defaults for missing keys.

        Parameters
        ----------
        category_name : str
            Category name, e.g. ``"code"`` or ``"code.py"``.
        schema : dict[str, Param], optional
            Schema from the category plugin. If None, an empty dict
            is used (TOML-only resolution).

        Returns
        -------
        dict
            Resolved parameter dict.
        """
        norm_schema = {} if schema is None else {k: _to_param(v) for k, v in schema.items()}

        # 1. Schema defaults
        result: dict[str, Any] = {}
        for key, param in norm_schema.items():
            result[key] = param.default

        # 2. Override with TOML values (parent hierarchy)
        toml_keys: set[str] = set()
        fc_section = self._file_categories_config.get("file_category", {})
        if isinstance(fc_section, dict):
            parts = category_name.split(".")
            for i in range(len(parts)):
                ancestor = ".".join(parts[:i+1])
                ancestor_values = fc_section.get(ancestor, {})
                if isinstance(ancestor_values, dict):
                    for key, value in ancestor_values.items():
                        if key in result:
                            result[key] = self._resolve_toml_value(value)
                            toml_keys.add(key)

        # 3. Apply env_var fallback
        self._apply_env_params(result, toml_keys, norm_schema)

        # 4. Cast types
        return self._cast_types(result, norm_schema)

    # -- TOML template generation -------------------------------------------

    def generate_toml_template(
        self,
        static_config: dict[str, Any] | None = None,
        existing_config: dict[str, Any] | None = None,
    ) -> str:
        """Generate a complete ``config.toml`` template.

        Combines static (app-level) sections with auto-generated
        interface-level and plugin-level sections.

        Parameters
        ----------
        static_config : dict, optional
            App-level config (global section).
        existing_config : dict, optional
            Existing config to preserve unknown sections.

        Returns
        -------
        str
            Complete TOML document.
        """
        lines: list[str] = []

        # Header
        lines.extend([
            "# " + "=" * 77,
            "# Mnemo Configuration",
            "# " + "=" * 77,
            "# Merged at runtime (later overrides earlier):",
            "#   1. config_schema defaults (on interface & plugin classes)",
            "#   2. ~/.config/mnemo/config.toml         (global)",
            "#   3. {data_dir}/.mnemo/config.toml         (project)",
            "#   4. MNEMO_* env vars                     (highest priority)",
            "#",
            "# Parameters backed by env vars are commented out below (reads from $VAR).",
            "# Override by uncommenting and setting a literal value.",
            "# File type processing params are in file_categories.toml.",
            "# " + "=" * 77,
            "",
        ])

        # -- Global section (from GLOBAL_CONFIG_SCHEMA) -----------------------
        lines.append(
            "# ── App Settings "
            + "─" * 55
        )
        lines.append("")
        lines.append("[global]")
        for param_name, param in GLOBAL_CONFIG_SCHEMA.items():
            if param.desc:
                lines.append(f"# {param.desc}")
            if param.env_var:
                lines.append(
                    f"# {param_name} = {_toml_value(param.default)}"
                    f"  # reads from ${param.env_var}"
                )
            else:
                lines.append(f"{param_name} = {_toml_value(param.default)}")
        lines.append("")
        if static_config:
            # Merge any extra static config keys not in the schema
            extra = {
                k: v for k, v in static_config.get("global", {}).items()
                if k not in GLOBAL_CONFIG_SCHEMA
            }
            if extra:
                for k, v in extra.items():
                    lines.append(f"{k} = {_toml_value(v)}")
                lines.append("")

        # -- Interface-level sections ----------------------------------------
        # Collect interface config_schema from all registries' interface_type.
        # Skip FileCategoryRegistry — file categories have their own
        # file_categories.toml to avoid duplication.
        from mnemo.core.interfaces import IFileCategory
        interface_schemas: dict[str, dict] = {}  # section → schema
        seen_interfaces: set[type] = set()
        for iface_name, iface in PluginHub.iter_interfaces():
            if iface is IFileCategory:
                continue  # file_categories.toml handles this
            if iface in seen_interfaces:
                continue
            seen_interfaces.add(iface)
            schema = getattr(iface, "config_schema", None)
            if schema:
                section = _registry_name_to_section(iface_name)
                interface_schemas[section] = schema

        if interface_schemas:
            lines.append(
                "# ── Interface Configuration "
                + "─" * 49
            )
            lines.append("")
            lines.append(
                "# Each interface section defines global defaults for that subsystem."
            )
            lines.append(
                "# Plugins of that interface have their own [interface.plugin] sections."
            )
            lines.append("")

            for section in sorted(interface_schemas.keys()):
                schema = interface_schemas[section]
                norm = {k: _to_param(v) for k, v in schema.items()}
                lines.append(f"[{section}]")
                for param_name in sorted(norm.keys()):
                    param = norm[param_name]
                    if param.desc:
                        lines.append(f"# {param.desc}")
                    if param.env_var:
                        lines.append(
                            f"# {param_name} = {_toml_value(param.default)}"
                            f"  # reads from ${param.env_var}"
                        )
                    else:
                        lines.append(f"{param_name} = {_toml_value(param.default)}")
                lines.append("")

        # -- Plugin-level sections -------------------------------------------
        # Collect all registered plugins — each gets a section even if
        # it has no own fields (users can add overrides there).
        # Skip FileCategoryRegistry — file categories have their own
        # file_categories.toml to avoid duplication.
        collected: dict[str, dict[str, dict]] = {}  # section → {plugin → own_schema}
        for iface_name, iface in PluginHub.iter_interfaces():
            if iface is IFileCategory:
                continue  # file_categories.toml handles this
            section = _registry_name_to_section(iface_name)
            for plugin_name, impl_cls in PluginHub.iter_impls(iface):
                inst = PluginHub.get(iface, plugin_name)
                own_schema = _get_own_schema(inst.__class__)
                collected.setdefault(section, {})[plugin_name] = own_schema

        if collected:
            lines.append(
                "# ── Plugin Configuration "
                + "─" * 51
            )
            lines.append("")
            lines.append(
                "# Each plugin section: [interface_name.plugin_name]"
            )
            lines.append("")

            for section_name in sorted(collected.keys()):
                plugins = collected[section_name]
                for plugin_name in sorted(plugins.keys()):
                    schema = plugins[plugin_name]
                    norm = {k: _to_param(v) for k, v in schema.items()}
                    section_header = f"{section_name}.{plugin_name}"
                    lines.append(f"[{section_header}]")

                    if norm:
                        for param_name in sorted(norm.keys()):
                            param = norm[param_name]
                            if param.desc:
                                lines.append(f"# {param.desc}")
                            if param.env_var:
                                lines.append(
                                    f"# {param_name} = {_toml_value(param.default)}"
                                    f"  # reads from ${param.env_var}"
                                )
                            else:
                                lines.append(f"{param_name} = {_toml_value(param.default)}")
                    else:
                        lines.append("# All settings inherited from parent interface")
                    lines.append("")

        # -- Agent & Embedder definitions (config-driven, no plugin classes) --
        from mnemo.core.agent_manager import AGENT_CONFIG_SCHEMA as _AGENT_SCHEMA

        lines.append(
            "# ── Agent Definitions "
            + "─" * 55
        )
        lines.append("")
        lines.append(
            "# Define named LLM agent profiles here. Each [agent.<name>]"
        )
        lines.append(
            "# section creates an independent pydantic-ai Agent instance."
        )
        lines.append(
            "# Use different agents for different file types via file_categories.toml."
        )
        lines.append("")
        lines.append("[agent.default]")
        for param_name in sorted(_AGENT_SCHEMA.keys()):
            param = _AGENT_SCHEMA[param_name]
            if param.desc:
                lines.append(f"# {param.desc}")
            if param.env_var:
                # api_key: uncommented so user can edit the env var name
                lines.append(
                    f"{param_name} = {_toml_value(param.default)}"
                    f"  # reads from ${param.env_var}"
                )
            else:
                lines.append(f"{param_name} = {_toml_value(param.default)}")
        lines.append("")

        # Embedder — single section (one KB, one embedder)
        from mnemo.core.embedder import EMBEDDER_CONFIG_SCHEMA
        lines.append(
            "# ── Embedder "
            + "─" * 59
        )
        lines.append("")
        lines.append(
            "# One knowledge base uses one embedding model / dimension."
        )
        lines.append("")
        lines.append("[embedder]")
        for param_name in sorted(EMBEDDER_CONFIG_SCHEMA.keys()):
            param = EMBEDDER_CONFIG_SCHEMA[param_name]
            if param.desc:
                lines.append(f"# {param.desc}")
            if param.env_var:
                # api_key: uncommented so user can edit the env var name
                lines.append(
                    f"{param_name} = {_toml_value(param.default)}"
                    f"  # reads from ${param.env_var}"
                )
            else:
                lines.append(f"{param_name} = {_toml_value(param.default)}")
        lines.append("")

        return "\n".join(lines) + "\n"

    def generate_file_categories_toml(self) -> str:
        """Generate a ``file_categories.toml`` template.

        Builds sections for all registered file category plugins
        from ``FileCategoryRegistry``.

        Returns
        -------
        str
            Complete TOML document for file categories.
        """
        from mnemo.core.plugin_base import PluginHub
        from mnemo.core.interfaces import IFileCategory

        lines: list[str] = [
            "# " + "=" * 77,
            "# Mnemo File Category Configuration",
            "# " + "=" * 77,
            "# Each [file_category.<name>] section defines how a category of",
            "# file types is processed: parsing, template, wiki, embedding, etc.",
            "#",
            "# Categories use dot-separated names for hierarchy:",
            "#   [file_category.code]       — all code files",
            "#   [file_category.code.py]    — Python-specific overrides",
            "#",
            "# Child categories inherit from parents and can override any setting.",
            "# Users can add custom categories here (even without a plugin class).",
            "# " + "=" * 77,
            "",
        ]

        # Collect own (non-inherited) schemas from FileCategoryRegistry
        categories: dict[str, dict] = {}  # name → own_schema (or {} for all-inherited)
        for name, impl_cls in PluginHub.iter_impls(IFileCategory):
            inst = PluginHub.get(IFileCategory, name)
            own_schema = _get_own_schema(inst.__class__)
            # Always include registered categories, even with no own fields
            categories[name] = own_schema

        # Also include external categories from loaded config
        fc_section = self._file_categories_config.get("file_category", {})
        if isinstance(fc_section, dict):
            for cat_name, cat_data in fc_section.items():
                if cat_name not in categories and isinstance(cat_data, dict):
                    schema: dict[str, dict] = {}
                    for k, v in cat_data.items():
                        if not isinstance(v, dict):
                            schema[k] = {
                                "type": _infer_type_name(v),
                                "default": v,
                                "desc": "",
                            }
                    if schema:
                        categories[cat_name] = schema

        # Sort by depth then alphabetically
        sorted_cats = sorted(categories.keys(),
                           key=lambda n: (n.count("."), n))

        for cat_name in sorted_cats:
            schema = categories[cat_name]
            norm = {k: _to_param(v) for k, v in schema.items()}
            section_header = f"file_category.{cat_name}"
            lines.append(f"[{section_header}]")

            if norm:
                for param_name in sorted(norm.keys()):
                    param = norm[param_name]
                    if param.desc:
                        lines.append(f"# {param.desc}")
                    if param.env_var:
                        lines.append(
                            f"# {param_name} = {_toml_value(param.default)}"
                            f"  # reads from ${param.env_var}"
                        )
                    else:
                        lines.append(f"{param_name} = {_toml_value(param.default)}")
            else:
                lines.append("# All settings inherited from parent category")
            lines.append("")

        return "\n".join(lines) + "\n"

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _resolve_env_vars(config: dict) -> dict:
        """Backward-compat: resolve legacy ``ENV::VAR_NAME`` strings.

        This handles raw config dicts that haven't been through the
        schema-driven pipeline (e.g. from ``ConfigLoader`` directly).
        New code should use :meth:`_apply_env_params` instead.
        """
        result: dict[str, Any] = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("ENV::"):
                result[key] = os.environ.get(value[5:], "")
            else:
                result[key] = value
        return result

    @staticmethod
    def _cast_types(
        config: dict[str, Any],
        schema: dict[str, Param],
    ) -> dict[str, Any]:
        """Cast config values to their declared types (int, float, bool)."""
        result: dict[str, Any] = {}
        for key, value in config.items():
            param = schema.get(key)
            type_name = param.type if param else "str"
            try:
                if type_name == "int":
                    result[key] = int(value)
                elif type_name == "float":
                    result[key] = float(value)
                elif type_name == "bool":
                    if isinstance(value, bool):
                        result[key] = value
                    elif isinstance(value, str):
                        result[key] = value.lower() in ("true", "1", "yes")
                    else:
                        result[key] = bool(value)
                else:
                    result[key] = value
            except (ValueError, TypeError):
                result[key] = value
        return result

    @staticmethod
    def _dict_to_toml_lines(
        d: dict[str, Any],
        prefix: str = "",
    ) -> list[str]:
        """Convert a nested dict to TOML lines."""
        lines: list[str] = []

        for key, value in d.items():
            if str(key).startswith("#"):
                continue

            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                if _is_leaf_dict(value):
                    lines.append(f"[{full_key}]")
                    for sub_key, sub_value in value.items():
                        if str(sub_key).startswith("#"):
                            continue
                        lines.append(f"{sub_key} = {_toml_value(sub_value)}")
                    lines.append("")
                else:
                    lines.append(f"[{full_key}]")
                    for sub_key, sub_value in value.items():
                        if str(sub_key).startswith("#"):
                            continue
                        if isinstance(sub_value, dict):
                            lines.extend(
                                ParamConfig._dict_to_toml_lines(
                                    {sub_key: sub_value}, prefix=full_key
                                )
                            )
                        else:
                            lines.append(f"{sub_key} = {_toml_value(sub_value)}")
                    lines.append("")
            elif isinstance(value, list):
                lines.append(f"{key} = {_toml_value(value)}")
            elif isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, (int, float)):
                lines.append(f"{key} = {value}")
            elif isinstance(value, str):
                lines.append(f"{key} = {_toml_value(value)}")
            elif value is None:
                pass

        return lines


# ---------------------------------------------------------------------------
# Context-local config (thread-safe & async-safe via contextvars)
# ---------------------------------------------------------------------------

import contextvars

_param_config_ctx: contextvars.ContextVar[ParamConfig | None] = (
    contextvars.ContextVar("mnemo_param_config", default=None)
)


def init_param_config(
    config: dict[str, Any],
    file_categories_config: dict[str, Any] | None = None,
) -> ParamConfig:
    """Initialize the context-local parameter config.

    Each ``KnowledgeBase`` instance sets its own config via this function.
    The config is stored in a :class:`contextvars.ContextVar`, making it
    safe for concurrent KB instances (threads / asyncio tasks).

    Parameters
    ----------
    config : dict
        The fully-loaded config dict.
    file_categories_config : dict, optional
        Separate file categories config dict.

    Returns
    -------
    ParamConfig
        The initialized instance.
    """
    pc = ParamConfig(config, file_categories_config)
    _param_config_ctx.set(pc)
    return pc


def get_config(cls: type) -> dict[str, Any]:
    """Get resolved configuration for an interface or plugin class.

    Reads from the context-local config set by :func:`init_param_config`.
    Returns an empty dict if no config has been initialized in this context.

    Parameters
    ----------
    cls : type
        An interface ABC or plugin class that defines ``config_schema``.

    Returns
    -------
    dict
        Resolved parameter dict. Empty dict if not initialized.
    """
    pc = _param_config_ctx.get()
    if pc is None:
        return {}
    return pc.get_config(cls)


def get_param_config() -> ParamConfig | None:
    """Return the current context-local ParamConfig, or None."""
    return _param_config_ctx.get()


def reset_param_config() -> None:
    """Reset the context-local config (useful in tests)."""
    _param_config_ctx.set(None)


def get_global_config() -> dict[str, Any]:
    """Get resolved global configuration (``[global]`` section).

    Merges the schema defaults from :data:`GLOBAL_CONFIG_SCHEMA` with
    TOML overrides from the context-local config.  This is the single
    source of truth for settings that span multiple subsystems
    (e.g. ``dimension`` shared by embedder and vector store).

    Returns the schema defaults if no config has been initialized.
    """
    pc = _param_config_ctx.get()
    if pc is None:
        # No config initialized — return bare defaults
        return {k: p.default for k, p in GLOBAL_CONFIG_SCHEMA.items()}

    # Resolve using the same pipeline as plugin configs
    schema = GLOBAL_CONFIG_SCHEMA
    result, toml_keys = pc._apply_defaults_and_toml("global", schema)
    pc._apply_env_params(result, toml_keys, schema)
    return pc._cast_types(result, schema)


def resolve_agent_config(agent_name: str | None = None) -> dict[str, Any]:
    """Resolve configuration for a named agent from ``[agent.<name>]`` sections.

    Merges agent-level TOML values with schema defaults and env-var
    fallbacks.  Compatible with both named agents and inline overrides.

    Parameters
    ----------
    agent_name : str, optional
        Which ``[agent.<name>]`` config to resolve.  If None, uses the
        default agent (from ``[global].default_agent`` or ``"default"``).

    Returns
    -------
    dict
        Resolved config dict with keys from ``AGENT_CONFIG_SCHEMA``.
    """
    from mnemo.core.agent_manager import AGENT_CONFIG_SCHEMA, AgentManager

    am = AgentManager.get_instance()
    if not am._initialized:
        # AgentManager not yet initialized — return bare schema defaults
        return {k: p.default for k, p in AGENT_CONFIG_SCHEMA.items()}

    target = agent_name or am.default_agent_name
    cfg = am._configs.get(target, am._configs.get(am.default_agent_name, {}))
    if cfg:
        return dict(cfg)

    # Fallback: schema defaults
    return {k: p.default for k, p in AGENT_CONFIG_SCHEMA.items()}


# ---------------------------------------------------------------------------
# TOML formatting helpers
# ---------------------------------------------------------------------------

def _toml_value(value: Any) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        items = ", ".join(_toml_value(v) for v in value)
        return f"[{items}]"
    if isinstance(value, (int, float)):
        return str(value)
    return f'"{value}"'


def _is_leaf_dict(d: dict) -> bool:
    """Check if a dict is a 'leaf' — all values are simple (non-dict)."""
    if not d:
        return True
    for v in d.values():
        if isinstance(v, dict):
            return False
    return True


def _infer_type_name(value: Any) -> str:
    """Infer a schema type name from a Python value."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


# ============================================================================
# .env template generation — scans all config schemas for env_var declarations
# ============================================================================


def generate_env_template() -> str:
    """Generate a ``.env`` template by scanning all config schemas.

    Walks all known ``Param`` schemas in the codebase, collects those
    with a non-None ``env_var`` field, deduplicates, and writes a
    commented template.

    Returns
    -------
    str
        Complete ``.env`` file content.
    """
    from mnemo.core.agent_manager import AGENT_CONFIG_SCHEMA
    from mnemo.core.embedder import EMBEDDER_CONFIG_SCHEMA
    from mnemo.core.plugin_base import PluginHub

    # Collect all schemas: interface-level, plugin-level, global, and special
    all_schemas: list[tuple[str, dict[str, Param]]] = [
        ("Global", GLOBAL_CONFIG_SCHEMA),
        ("Agent (all [agent.xxx] sections)", AGENT_CONFIG_SCHEMA),
        ("Embedder ([embedder])", EMBEDDER_CONFIG_SCHEMA),
    ]

    # Walk all registered plugin interfaces for config_schema
    for iface_name, iface_type in PluginHub.iter_interfaces():
        schema = getattr(iface_type, "config_schema", None)
        if schema:
            all_schemas.append((f"Interface: {iface_name}", schema))

    # Walk all registered plugin implementations for config_schema
    for iface_name, iface_type in PluginHub.iter_interfaces():
        for impl_name, impl_cls in PluginHub.iter_impls(iface_type):
            schema = getattr(impl_cls, "config_schema", None)
            if schema:
                all_schemas.append((f"Plugin: {impl_name}", schema))

    # Collect unique env_var entries
    seen: set[str] = set()
    lines: list[str] = [
        "# " + "=" * 77,
        "# Mnemo Environment Variables",
        "# " + "=" * 77,
        "#",
        "# Copy this file to .env and fill in your API keys.",
        "# The key names match the default ``api_key`` values in config.toml.",
        "#",
        "# Example:",
        "#   LLM_API_KEY=sk-your-key-here",
        "#   EMBED_API_KEY=sk-your-key-here",
        "#",
        "# Generated by: mnemo init",
        "#",
        "",
    ]

    for _source_name, schema in all_schemas:
        for param_name, param in schema.items():
            env_var = param.env_var
            if not env_var or env_var in seen:
                continue
            seen.add(env_var)
            if param.desc:
                lines.append(f"# {param.desc}")
            lines.append(f"{env_var}=")
            lines.append("")

    return "\n".join(lines) + "\n"
