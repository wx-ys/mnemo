"""File category plugin interface.

File categories form a hierarchical tree that controls the full
processing pipeline for each file type: identification, markdown
conversion, wiki generation, embedding, and storage layout.

Categories are organized with dot-separated names creating a hierarchy:
``"code"`` → ``"code.py"`` → ``"code.py.django"``. A child category
inherits and can override any parent setting.

When a file is added, the most specific matching category is resolved
and used to determine all processing parameters.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar

from mnemo.core.interfaces.param_spec import Param
from mnemo.core.plugin_base import PluginBase, PluginHub


class IFileCategory(PluginBase, ABC):
    """Interface for file category plugins.

    Each subclass represents one file category (e.g. ``"docs"``,
    ``"code.py"``). The ``name`` uses dots to form a hierarchy:
    deeper names are more specific and override parent settings.

    .. note::

        The ``name`` class variable serves a dual purpose:
        (1) Plugin registration key via ``PluginHub`` —
            the interface itself uses ``"file_category"``.
        (2) Category name for concrete subclasses (e.g. ``"code.py"``) —
            concrete categories override ``name`` and set
            ``__plugin_impl__ = True`` to auto-register.

    Class Attributes
    ----------------
    name : str
        Unique category name. Dots create hierarchy
        (e.g. ``"code.py"`` is a child of ``"code"``).
        The interface itself uses ``"file_category"`` as the plugin
        registration key; concrete categories override this with their
        own category name.
    parent : str or None
        Parent category name, or ``None`` for root categories.
    types : list of str
        File extensions (without dot) that this category matches.
    config_schema : dict
        Declared config parameters with defaults, types, and descriptions.
    """

    __plugin_interface__ = True
    name: ClassVar[str] = "file_category"     # plugin registration name
    plugin_path: ClassVar[str] = "file_categories"

    # ── Registration metadata (override in subclasses) ──────────────────

    # NOTE: Concrete categories override ``name`` with their own category
    # name (e.g. ``name = "code"``) and set ``__plugin_impl__ = True``.
    # The ``parent`` and ``types`` fields below are category hierarchy
    # metadata, not plugin registration metadata.
    parent: str | None = None
    types: list[str] = []

    # ── Config schema ────────────────────────────────────────────────────

    config_schema: dict[str, Param] = {
        "parser": Param(
            type="str",
            default="text",
            desc="Parser plugin name for markdown conversion",
        ),
        "template": Param(
            type="str",
            default="note",
            desc="Template plugin name for wiki generation",
        ),
        "auto_md": Param(
            type="bool",
            default=True,
            desc="Auto-generate markdown on add",
        ),
        "auto_wiki": Param(
            type="bool",
            default=True,
            desc="Auto-generate wiki on add",
        ),
        "auto_embed": Param(
            type="bool",
            default=True,
            desc="Auto-generate embedding on add",
        ),
        "chunker": Param(
            type="str",
            default="paragraph",
            desc="Chunker plugin name for text splitting strategy",
        ),
    }

    # ── Derived properties ───────────────────────────────────────────────

    @property
    def dir_path(self) -> str:
        """Storage directory path derived from category name.

        ``"code.py"`` → ``"code/py"``,
        ``"documents"`` → ``"documents"``.
        """
        return self.name.replace(".", "/")

    @property
    def depth(self) -> int:
        """Hierarchy depth (number of dot-separated segments)."""
        return self.name.count(".") + 1 if self.name else 1
