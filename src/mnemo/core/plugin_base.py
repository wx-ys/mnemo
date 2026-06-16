"""Unified plugin system — auto-registration via __init_subclass__ hooks.

Replaces 15 separate ``Registry[T]`` singletons with a single
:class:`PluginHub` and auto-registration via :class:`PluginBase`.

Usage::

    # Define an interface
    class IChunker(PluginBase, ABC):
        __plugin_interface__ = True
        name: ClassVar[str] = "chunker"
        plugin_path: ClassVar[str] = "chunkers"

    # Define a plugin (no decorator needed)
    class ParagraphChunker(IChunker):
        __plugin_impl__ = True
        name = "paragraph"

    # Lookup — returns IChunker, not Any
    chunker = PluginHub.get(IChunker, "paragraph")
"""

from __future__ import annotations

import importlib.util
from abc import ABC
from pathlib import Path
from typing import Any, ClassVar, TypeVar, cast

T = TypeVar("T")


# ============================================================================
# Validation helpers
# ============================================================================


def _validate_interface(cls: type) -> None:
    """Ensure a plugin interface has the required attributes."""
    cls_name = cls.__name__
    name_val = getattr(cls, "name", None)
    if not isinstance(name_val, str) or not name_val:
        raise TypeError(
            f"Plugin interface {cls_name} must define "
            f"'name: str' as a class variable (e.g., name = 'chunker')."
        )
    pp_val = getattr(cls, "plugin_path", None)
    if not isinstance(pp_val, str) or not pp_val:
        raise TypeError(
            f"Plugin interface {cls_name} must define "
            f"'plugin_path: str' (e.g., plugin_path = 'chunkers')."
        )
    # Check ABC inheritance — walk __bases__ and __mro__
    found_abc = False
    for base in getattr(cls, "__mro__", ()):
        if base is ABC:
            found_abc = True
            break
    if not found_abc:
        raise TypeError(
            f"Plugin interface {cls_name} must also inherit from ABC "
            f"(e.g., class {cls_name}(PluginBase, ABC): ...)"
        )


def _resolve_plugin_interface(cls: type) -> type | None:
    """Walk MRO to find the first parent class marked ``__plugin_interface__``."""
    for base in cls.__mro__:
        if base.__dict__.get("__plugin_interface__", False):
            return base
    return None


def _validate_impl(cls: type) -> None:
    """Ensure a plugin implementation has the required attributes."""
    cls_name = cls.__name__
    name_val = getattr(cls, "name", None)
    if not isinstance(name_val, str) or not name_val:
        raise TypeError(
            f"Plugin {cls_name} must define 'name: str' as a class variable."
        )
    iface = _resolve_plugin_interface(cls)
    if iface is None:
        raise TypeError(
            f"Plugin {cls_name} must inherit from a class marked "
            f"with __plugin_interface__ = True."
        )
    # Store the interface type — use plain attribute assignment
    cls._plugin_interface_type = iface  # type: ignore[attr-defined]


def _import_modules_from_dir(directory: Path) -> int:
    """Import all .py files from *directory* (non-recursive).

    Returns the count of modules imported.
    """
    count = 0
    if not directory.is_dir():
        return 0
    for py_file in sorted(directory.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = py_file.stem
        spec = importlib.util.spec_from_file_location(module_name, str(py_file))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            count += 1
        except Exception:
            import logging

            logging.getLogger("mnemo").warning(
                "Failed to load plugin module '%s': %s",
                py_file,
                exc_info=True,
            )
    return count


# ============================================================================
# PluginBase — auto-registration via __init_subclass__
# ============================================================================


class PluginBase:
    """Base class for all Mnemo plugin interfaces and implementations.

    Uses ``__init_subclass__`` to auto-register subclasses into
    :class:`PluginHub`.  No manual decorators needed.

    Two markers control registration:

    - ``__plugin_interface__ = True`` — this class IS a plugin interface.
      Must also inherit from ``ABC`` and define ``name: str`` +
      ``plugin_path: str``.

    - ``__plugin_impl__ = True`` — this class IS a concrete plugin.
      Must define ``name: str``.  The interface it implements is
      detected automatically from the parent class MRO.

    Intermediate utility classes set neither marker and are ignored.
    """

    _plugin_interface_type: ClassVar[type | None] = None
    """The interface this plugin implements. Set automatically for impls."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Only process classes that explicitly opt in via markers.
        # Use cls.__dict__ (not getattr) to avoid inheriting markers
        # from parent classes.
        is_interface = cls.__dict__.get("__plugin_interface__", False)
        is_impl = cls.__dict__.get("__plugin_impl__", False)

        if is_interface:
            _validate_interface(cls)
            PluginHub._register_interface(cls)
        elif is_impl:
            _validate_impl(cls)
            PluginHub._register_impl(cls)
        # else: intermediate base, test stub — skip registration


# ============================================================================
# PluginHub — unified registry with generics
# ============================================================================


class PluginHub:
    """Unified registry for all Mnemo plugin interfaces and implementations.

    Single global hub — replaces 15 separate ``Registry[T]`` singletons.
    All registration happens automatically via
    :class:`PluginBase.__init_subclass__`.
    """

    _interfaces: ClassVar[dict[str, type]] = {}
    """interface_name (e.g. 'chunker') -> interface type (e.g. IChunker)"""

    _impls: ClassVar[dict[type, dict[str, type]]] = {}
    """interface_type -> {plugin_name -> impl_class}"""

    _discovery_roots: ClassVar[list[Path]] = []
    """Root directories for plugin discovery."""

    # -- registration (called by PluginBase.__init_subclass__) ----------------

    @classmethod
    def _register_interface(cls, iface: type) -> None:
        """Register a new plugin interface type."""
        iface_name: str = getattr(iface, "name", "")
        if iface_name in cls._interfaces:
            existing = cls._interfaces[iface_name]
            raise TypeError(
                f"Duplicate plugin interface name '{iface_name}': "
                f"{iface.__name__} conflicts with {existing.__name__}."
            )
        cls._interfaces[iface_name] = iface
        cls._impls.setdefault(iface, {})

    @classmethod
    def _register_impl(cls, impl: type) -> None:
        """Register a concrete plugin implementation."""
        iface: type = cast(type, getattr(impl, "_plugin_interface_type", None))
        name: str = getattr(impl, "name", "")
        plugins = cls._impls.get(iface, {})
        if name in plugins:
            existing = plugins[name]
            raise TypeError(
                f"Duplicate plugin name '{name}' for interface "
                f"'{iface.__name__}': {impl.__name__} conflicts with "
                f"{existing.__name__}."
            )
        plugins[name] = impl

    # -- typed lookup ---------------------------------------------------------

    @classmethod
    def get(cls, interface: type[T], name: str) -> T:
        """Look up a plugin instance by interface and name.

        Returns a properly-typed instance.  Lazily instantiates on first
        access and caches the instance on the implementation class.
        """
        impl_cls = cls.get_class(interface, name)
        cache_attr = "_plugin_instance"
        if not hasattr(impl_cls, cache_attr):
            setattr(impl_cls, cache_attr, impl_cls())
        return cast(T, getattr(impl_cls, cache_attr))

    @classmethod
    def get_class(cls, interface: type[T], name: str) -> type[T]:
        """Look up a plugin class by interface and name (no instantiation)."""
        if interface not in cls._impls:
            raise KeyError(
                f"Unknown plugin interface '{interface.__name__}'. "
                f"Known interfaces: {list(cls._interfaces.keys())}"
            )
        plugins = cls._impls[interface]
        if name not in plugins:
            raise KeyError(
                f"No plugin named '{name}' for interface "
                f"'{interface.__name__}'. Available: {list(plugins.keys())}"
            )
        return plugins[name]

    @classmethod
    def get_interface(cls, name: str) -> type:
        """Look up a plugin interface type by name."""
        if name not in cls._interfaces:
            raise KeyError(
                f"Unknown plugin interface '{name}'. "
                f"Known: {list(cls._interfaces.keys())}"
            )
        return cls._interfaces[name]

    @classmethod
    def has(cls, interface: type, name: str) -> bool:
        """Check if a plugin is registered for the given interface and name."""
        return name in cls._impls.get(interface, {})

    @classmethod
    def has_interface(cls, name: str) -> bool:
        """Check if an interface is registered with the given name."""
        return name in cls._interfaces

    # -- introspection --------------------------------------------------------

    @classmethod
    def iter_impls(cls, interface: type[T]) -> list[tuple[str, type[T]]]:
        """List all (name, class) pairs for an interface."""
        return list(cls._impls.get(interface, {}).items())

    @classmethod
    def list_names(cls, interface: type) -> list[str]:
        """List all plugin names for an interface."""
        return list(cls._impls.get(interface, {}).keys())

    @classmethod
    def iter_interfaces(cls) -> list[tuple[str, type]]:
        """List all (name, interface_type) pairs."""
        return list(cls._interfaces.items())

    # -- type-based and category-based lookup ---------------------------------

    @classmethod
    def get_instance_for_type(
        cls, interface: type[T], file_extension: str,
    ) -> T | None:
        """Find the first plugin whose ``supported_types`` includes *ext*."""
        ext = file_extension.lower().lstrip(".")
        for impl_name, _impl_cls in cls.iter_impls(interface):
            inst = cls.get(interface, impl_name)
            supported: list[str] = getattr(inst, "supported_types", [])
            if isinstance(supported, list):
                normalized = [t.lower().lstrip(".") for t in supported]
                if ext in normalized or file_extension.lower() in normalized:
                    return inst
        return None

    @classmethod
    def get_instance_for_category(
        cls, interface: type[T], category: str,
    ) -> T | None:
        """Find the first plugin whose ``category`` matches."""
        for impl_name, _impl_cls in cls.iter_impls(interface):
            inst = cls.get(interface, impl_name)
            if getattr(inst, "category", "") == category:
                return inst
        return None

    # -- discovery ------------------------------------------------------------

    @classmethod
    def add_discovery_root(cls, path: Path) -> None:
        """Register a directory to scan for plugins."""
        cls._discovery_roots.append(Path(path))

    @classmethod
    def discover(cls) -> int:
        """Scan all discovery roots for plugins and import them.

        For each registered interface, scans ``<root>/<plugin_path>/``.
        Returns the count of newly loaded plugin modules.
        """
        total = 0
        for root in cls._discovery_roots:
            for iface in list(cls._impls.keys()):
                plugin_path_val = getattr(iface, "plugin_path", None)
                if not plugin_path_val:
                    continue
                scan_dir = root / plugin_path_val
                if scan_dir.is_dir():
                    total += _import_modules_from_dir(scan_dir)
        return total
