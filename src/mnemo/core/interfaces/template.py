"""LLM template interface (ITemplate)."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from mnemo.core.interfaces.param_spec import Param
from mnemo.core.plugin_base import PluginBase, PluginHub

if TYPE_CHECKING:
    from mnemo.core.interfaces.types import FileMeta


@runtime_checkable
class WikiResultProtocol(Protocol):
    """Structural protocol for wiki generation results.

    Any object with these attributes satisfies the interface,
    avoiding a hard dependency on ``plugins.base.WikiResult``.
    """

    content: str
    tokens_input: int
    tokens_output: int
    requests: int
    tool_calls: int

    @property
    def total_tokens(self) -> int: ...


class ITemplate(PluginBase, ABC):
    """Interface for LLM-powered wiki generation.

    Maps a file's Markdown content to a structured wiki summary using
    configurable system/user prompts.

    Resolution priority: type-level -> category-level -> fallback (note).

    Notes
    -----
    Subclasses must be registered via ``__plugin_impl__ = True`` marker.
    """

    __plugin_interface__ = True
    name: ClassVar[str] = "template"
    plugin_path: ClassVar[str] = "templates"

    # ── Config schema ──────────────────────────────────────────────────

    config_schema: dict[str, Param] = {
        "default_auto_wiki": Param(
            type="bool", default=True,
            desc="Global default: auto-generate wiki on add"
        ),
        "wiki_chunk_size": Param(
            type="int", default=8000,
            desc="Max input characters for wiki generation"
        ),
        "wiki_chunk_overlap": Param(
            type="int", default=200,
            desc="Overlap between wiki chunks"
        ),
        "wiki_temperature": Param(
            type="float", default=0.3, desc="LLM temperature for wiki generation"
        ),
        "wiki_max_tokens": Param(
            type="int", default=2048, desc="Max output tokens for wiki generation"
        ),
        "wiki_max_input_chars": Param(
            type="int", default=16000, desc="Max input characters per wiki generation call"
        ),
    }

    # ── Abstract interface ─────────────────────────────────────────────

    @property
    @abstractmethod
    def category(self) -> str:
        """Parent category for fallback chain.

        Empty string indicates a fallback template (not tied to a category).
        """
        ...

    @property
    @abstractmethod
    def supported_types(self) -> list[str]:
        """File types this template applies to."""
        ...

    @abstractmethod
    def generate_wiki(
        self, md_content: str, metadata: "FileMeta", model_config: dict,
    ) -> Any:
        """Generate a wiki summary via LLM.

        Parameters
        ----------
        md_content : str
            The file's Markdown content.
        metadata : FileMeta
            File metadata (filename, type, source, etc.).
        model_config : dict
            LLM configuration: model name, temperature, max_tokens, etc.

        Returns
        -------
        WikiResult or str
            Wiki-formatted Markdown summary.  Implementations should
            return an object with ``content``, ``tokens_input``,
            ``tokens_output``, ``total_tokens`` attributes.

        Raises
        ------
        ModelAPIError
            If the LLM API call fails (network, auth, quota, etc.).
        """
        ...

    @classmethod
    def resolve(cls, file_extension: str, category: str) -> "ITemplate":
        """Find the best template for a file extension and category.

        Priority: type-level -> category-level -> 'note' fallback.
        """
        inst = PluginHub.get_instance_for_type(cls, file_extension)
        if inst is not None:
            return inst
        inst = PluginHub.get_instance_for_category(cls, category)
        if inst is not None:
            return inst
        if PluginHub.has(cls, "note"):
            return PluginHub.get(cls, "note")
        raise KeyError(
            f"No template found for '{file_extension}' (category: {category}), "
            f"and fallback 'note' template is not registered."
        )
