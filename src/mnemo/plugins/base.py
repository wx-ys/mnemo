"""Base classes for Mnemo built-in plugins.

Provides default implementations for parser and template interfaces.
Concrete plugins inherit from these and override only what they need.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

from mnemo.core.interfaces import ChunkInfo, FileMeta, IParser, ITemplate

# ============================================================================
# BaseParser
# ============================================================================

class BaseParser(IParser):
    """Default parser implementation with sensible defaults.

    Subclasses must override at minimum:
        ``name``, ``category``, ``supported_types``, ``parse()``.

    Optional overrides:
        ``chunk()``, ``default_enable_*`` properties.
    """

    @property
    def default_enable_md(self) -> bool:
        return True

    @property
    def default_enable_wiki(self) -> bool:
        return True

    @property
    def default_enable_embed(self) -> bool:
        return True

    def chunk(self, md_text: str, max_chunk_size: int = 8000) -> list[ChunkInfo]:
        """Default paragraph-based chunking (deprecated — delegates to IChunker).

        .. deprecated::
            Use :class:`PluginHub` and :class:`IChunker` plugins
            directly.  This shim delegates to ``PluginHub.get(IChunker, "paragraph")``
            for backward compatibility with existing parsers.

        Splits on double-newline boundaries, keeping each chunk
        under ``max_chunk_size`` characters.

        Parameters
        ----------
        md_text : str
            Markdown text to split.
        max_chunk_size : int, optional
            Maximum characters per chunk. Default is 8000.

        Returns
        -------
        list of ChunkInfo
        """
        import warnings
        warnings.warn(
            "BaseParser.chunk() is deprecated. Use ChunkerRegistry.get() instead.",
            DeprecationWarning, stacklevel=2,
        )
        from mnemo.core.plugin_base import PluginHub
        from mnemo.core.interfaces import IChunker
        chunker = PluginHub.get(IChunker, "paragraph")
        return chunker.chunk(md_text, {"max_chunk_size": max_chunk_size})


# ============================================================================
# WikiResult — structured return from LLM calls (content + token usage)
# ============================================================================


class WikiResult:
    """Result of a wiki generation LLM call.

    Attributes
    ----------
    content : str
        The generated wiki Markdown text.
    tokens_input : int
        Number of input/prompt tokens consumed.
    tokens_output : int
        Number of output/completion tokens generated.
    requests : int
        Number of API requests made.
    tool_calls : int
        Number of tool calls made (always 0 for wiki generation).
    """

    __slots__ = ("content", "tokens_input", "tokens_output",
                  "requests", "tool_calls")

    def __init__(
        self,
        content: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        requests: int = 0,
        tool_calls: int = 0,
    ) -> None:
        self.content = content
        self.tokens_input = tokens_input
        self.tokens_output = tokens_output
        self.requests = requests
        self.tool_calls = tool_calls

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.tokens_input + self.tokens_output


def _extract_usage(result) -> tuple[int, int, int, int]:
    """Extract token usage from a pydantic-ai AgentRunResult."""
    try:
        usage = result.usage()
        return (
            usage.input_tokens,
            usage.output_tokens,
            usage.requests,
            usage.tool_calls,
        )
    except Exception:
        return (0, 0, 0, 0)


# -- Raw stream event type checks -------------------------------------------
# These helpers inspect ModelResponseStreamEvent objects to detect
# thinking/reasoning parts during real-time streaming.  Using isinstance
# is more robust than string comparison on part_kind/part_delta_kind.


def _is_text_part(part: Any) -> bool:
    """True if *part* is a ``TextPart`` with non-empty content."""
    from pydantic_ai.messages import TextPart
    return isinstance(part, TextPart) and bool(part.content)


def _is_thinking_part(part: Any) -> bool:
    """True if *part* is a ``ThinkingPart``."""
    from pydantic_ai.messages import ThinkingPart
    return isinstance(part, ThinkingPart)


def _is_text_delta(event: Any) -> bool:
    """True if *event* is a ``PartDeltaEvent`` with a ``TextPartDelta``."""
    from pydantic_ai.messages import PartDeltaEvent, TextPartDelta
    return (isinstance(event, PartDeltaEvent)
            and isinstance(event.delta, TextPartDelta)
            and bool(event.delta.content_delta))


def _is_thinking_delta(event: Any) -> bool:
    """True if *event* is a ``PartDeltaEvent`` with a ``ThinkingPartDelta``."""
    from pydantic_ai.messages import PartDeltaEvent, ThinkingPartDelta
    return (isinstance(event, PartDeltaEvent)
            and isinstance(event.delta, ThinkingPartDelta)
            and bool(event.delta.content_delta))


def _is_thinking_start(event: Any) -> bool:
    """True if *event* is a ``PartStartEvent`` with a ``ThinkingPart``.

    NOTE: The initial ThinkingPart may have empty content (when
    created from ``<think>`` tag detection).  We accept both empty
    and non-empty ThinkingPart starts — the real content arrives
    via subsequent ``ThinkingPartDelta`` events.
    """
    from pydantic_ai.messages import PartStartEvent, ThinkingPart
    return (isinstance(event, PartStartEvent)
            and isinstance(event.part, ThinkingPart))


def _extract_thinking_from_messages(
    messages: list[Any],
    on_thinking: Callable[[str], None],
) -> int:
    """Post-hoc fallback: extract ``ThinkingPart`` objects from message list.

    Iterates all messages looking for ``ThinkingPart`` objects and
    calls *on_thinking* for each one with non-empty content.  This
    handles models that return thinking only in the final response
    (not as stream deltas) and cases where the pydantic-ai stream
    is consumed internally before application code gets the
    ``StreamedRunResult``.

    Returns the count of thinking parts found.
    """
    from pydantic_ai.messages import ThinkingPart
    count = 0
    try:
        for msg in messages:
            for part in getattr(msg, 'parts', []):
                if isinstance(part, ThinkingPart) and part.content and part.content.strip():
                    on_thinking(part.content)
                    count += 1
    except Exception:
        import logging
        logging.getLogger("mnemo.llm").debug(
            "Failed to extract thinking from messages", exc_info=True,
        )
    return count


# ============================================================================
# BaseTemplate
# ============================================================================

class BaseTemplate(ITemplate):
    """Default template implementation.

    Subclasses must override at minimum:
        ``name``, ``supported_types``, ``system_prompt``, ``user_prompt_template``.

    The ``generate_wiki()`` method has a default implementation that
    calls an LLM API — subclasses may override ``_call_llm()`` for
    different backends.

    Two call paths are available:

    * **Sync** — ``generate_wiki()`` → ``_call_llm()`` (uses
      ``agent.run_sync()``).  Returns a ``WikiResult`` with token usage.
    * **Async streaming** — ``generate_wiki_stream()`` → ``_call_llm_stream()``
      (uses ``agent.run_stream()``).  Emits stream chunks via a callback
      and returns a ``WikiResult`` with token usage.
    """

    @property
    def category(self) -> str:
        """Category for fallback chain. Empty string = fallback template."""
        return ""

    system_prompt: str = (
        "You are an information organization assistant. "
        "Generate a concise summary based on the provided content."
    )
    user_prompt_template: str = "{content}"

    # -- Sync API ----------------------------------------------------------

    def generate_wiki(
        self, md_content: str, metadata: FileMeta, model_config: dict
    ) -> WikiResult:
        """Generate a wiki summary via LLM (synchronous).

        Parameters
        ----------
        md_content : str
            The file's Markdown content.
        metadata : FileMeta
            File metadata.
        model_config : dict
            LLM configuration (model, temperature, max_tokens, etc.).

        Returns
        -------
        WikiResult
            Generated content and token usage.
        """
        prompt = self.user_prompt_template.format(
            content=md_content,
            filename=metadata.filename,
            file_type=metadata.file_type,
            source=metadata.source_path,
        )

        return self._call_llm(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            config=model_config,
        )

    def _call_llm(
        self, system_prompt: str, user_prompt: str, config: dict
    ) -> WikiResult:
        """Call the LLM via pydantic-ai Agent synchronously.

        Parameters
        ----------
        system_prompt : str
        user_prompt : str
        config : dict
            LLM configuration.  Key fields:
            ``agent_name`` — which ``[agent.<name>]`` config to use.

        Returns
        -------
        WikiResult
            LLM response text and token usage.
        """
        from mnemo.core.agent_manager import AgentManager

        agent_name = config.get("agent_name", "default")
        am = AgentManager.get_instance()
        if not am._initialized:
            return WikiResult("[LLM skipped: AgentManager not initialized]")

        # Try structured output for wiki generation
        try:
            from pydantic import BaseModel

            class WikiOutput(BaseModel):
                summary: str
                key_points: list[str] = []
                tags: list[str] = []

            agent = am.get_agent(agent_name, output_type=WikiOutput)
            result = agent.run_sync(user_prompt, instructions=system_prompt)
            if isinstance(result.output, WikiOutput) and isinstance(result.output.summary, str):
                ti, to, req, tc = _extract_usage(result)
                return WikiResult(result.output.summary, ti, to, req, tc)
        except Exception:
            pass  # fall through to plain chat

        # Fallback: plain text
        try:
            agent = am.get_agent(agent_name, output_type=str)
            result = agent.run_sync(user_prompt, instructions=system_prompt)
            content = result.output.strip() if result.output else ""
            ti, to, req, tc = _extract_usage(result)
            return WikiResult(content, ti, to, req, tc)
        except Exception as exc:
            import logging
            logging.getLogger("mnemo").error(
                "Wiki LLM call failed: %s", exc, exc_info=True,
            )
            return WikiResult(
                f"[LLM failed: {exc}]",
            )

    # -- Async streaming API -----------------------------------------------

    async def generate_wiki_stream(
        self,
        md_content: str,
        metadata: FileMeta,
        model_config: dict,
        *,
        on_chunk: "Callable[[str], None] | None" = None,
        on_thinking: "Callable[[str], None] | None" = None,
        on_stream_end: "Callable[[], None] | None" = None,
    ) -> WikiResult:
        """Generate a wiki summary via LLM with streaming (async).

        Parameters
        ----------
        md_content : str
            The file's Markdown content.
        metadata : FileMeta
            File metadata.
        model_config : dict
            LLM configuration.
        on_chunk : callable, optional
            Called with each text delta as it arrives from the LLM.
        on_thinking : callable, optional
            Called with each thinking/reasoning delta in real-time
            as it streams from the LLM (via ThinkingPartDelta events).
        on_stream_end : callable, optional
            Called when streaming is complete.

        Returns
        -------
        WikiResult
            Generated content and token usage.
        """
        prompt = self.user_prompt_template.format(
            content=md_content,
            filename=metadata.filename,
            file_type=metadata.file_type,
            source=metadata.source_path,
        )

        return await self._call_llm_stream(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            config=model_config,
            on_chunk=on_chunk,
            on_thinking=on_thinking,
            on_stream_end=on_stream_end,
        )

    async def _call_llm_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        config: dict,
        *,
        on_chunk: "Callable[[str], None] | None" = None,
        on_thinking: "Callable[[str], None] | None" = None,
        on_stream_end: "Callable[[], None] | None" = None,
    ) -> WikiResult:
        """Call the LLM via pydantic-ai Agent with streaming (async).

        Iterates raw :class:`ModelResponseStreamEvent` events from the
        agent stream so that **both** text deltas and thinking/reasoning
        deltas are captured in real-time.  After the stream completes,
        token usage is captured from ``streamed.usage()``.

        Parameters
        ----------
        system_prompt : str
        user_prompt : str
        config : dict
        on_chunk : callable, optional
            Called with each text delta as it arrives.
        on_thinking : callable, optional
            Called with each thinking delta in real-time as it arrives
            (via ``ThinkingPartDelta.content_delta`` events).
        on_stream_end : callable, optional
            Called when the stream completes.

        Returns
        -------
        WikiResult
            LLM response text and token usage.
        """
        from mnemo.core.agent_manager import AgentManager

        agent_name = config.get("agent_name", "default")
        am = AgentManager.get_instance()
        if not am._initialized:
            return WikiResult("[LLM skipped: AgentManager not initialized]")

        # Plain text streaming (structured output streaming is more complex
        # and rarely needed for wiki generation — the primary goal is
        # real-time display of generated text and thinking).
        try:
            agent = am.get_agent(agent_name, output_type=str)
        except Exception as exc:
            import logging
            logging.getLogger("mnemo").error(
                "Cannot create streaming agent: %s", exc, exc_info=True,
            )
            return WikiResult(
                f"[LLM failed: cannot create agent — {exc}]"
            )

        accumulated: list[str] = []
        tokens_input = 0
        tokens_output = 0
        requests = 0
        tool_calls = 0
        thinking_count = 0

        try:
            async with agent.run_stream(
                user_prompt, instructions=system_prompt,
            ) as streamed:
                # -- Pre-fetch any parts already accumulated ---------------
                # pydantic-ai may consume the raw stream internally before
                # yielding StreamedRunResult.  In that case the response
                # parts are already populated.  Extract them first so we
                # don't lose already-accumulated text and thinking.
                if streamed._stream_response is not None:
                    initial_response = streamed._stream_response.response
                    for part in initial_response.parts:
                        if _is_text_part(part):
                            if part.content:
                                accumulated.append(part.content)
                                if on_chunk:
                                    on_chunk(part.content)
                        elif _is_thinking_part(part):
                            if part.content and part.content.strip():
                                if on_thinking:
                                    on_thinking(part.content)
                                thinking_count += 1

                # -- Iterate raw stream events for real-time deltas --------
                # Iterate raw ModelResponseStreamEvent events instead of
                # using stream_text(delta=True).  This gives us BOTH
                # TextPartDelta AND ThinkingPartDelta events in real-time.
                # stream_text() only yields text deltas — thinking deltas
                # go through a separate channel and would otherwise be
                # invisible until all_messages() is called at the end.
                if streamed._stream_response is not None:
                    async for event in streamed._stream_response:
                        if _is_text_delta(event):
                            delta = event.delta.content_delta
                            accumulated.append(delta)
                            if on_chunk:
                                on_chunk(delta)
                        elif _is_thinking_delta(event):
                            delta = event.delta.content_delta
                            if delta and on_thinking:
                                on_thinking(delta)
                                thinking_count += 1
                        elif _is_thinking_start(event):
                            # Initial ThinkingPart — may have content
                            # (reasoning_content API field) or be empty
                            # (<think> tag detection starts with empty part).
                            content = getattr(event.part, 'content', '')
                            if content and on_thinking:
                                on_thinking(content)
                                thinking_count += 1

                # Capture usage after streaming completes
                usage = streamed.usage()
                tokens_input = usage.input_tokens
                tokens_output = usage.output_tokens
                requests = usage.requests
                tool_calls = usage.tool_calls

                # Log a diagnostic if no thinking content was found
                if on_thinking and thinking_count == 0:
                    import logging
                    logging.getLogger("mnemo.llm").info(
                        "No thinking deltas captured during streaming; "
                        "checking all_messages() for post-hoc ThinkingPart "
                        "objects (may indicate model returned thinking in "
                        "final response only, or stream was pre-consumed)."
                    )
                    # Fallback: extract thinking from message history.
                    # This handles two cases:
                    # 1. pydantic-ai consumes the raw stream internally
                    #    before yielding StreamedRunResult (the "consumed
                    #    stream" issue), so our __aiter__ loop above found
                    #    zero events.
                    # 2. The model returns thinking only in the final
                    #    response, not as stream deltas.
                    thinking_count = _extract_thinking_from_messages(
                        streamed.all_messages(), on_thinking,
                    )
                    if thinking_count == 0:
                        import logging
                        logging.getLogger("mnemo.llm").info(
                            "No thinking/reasoning content in LLM response "
                            "(model may not support reasoning mode — "
                            "thinking display only works with models like "
                            "deepseek-reasoner)"
                        )
        except Exception as exc:
            import logging
            logging.getLogger("mnemo").error(
                "Wiki LLM streaming call failed: %s", exc, exc_info=True,
            )
            return WikiResult(
                f"[LLM failed: {exc}]",
            )
        finally:
            if on_stream_end:
                on_stream_end()

        content = "".join(accumulated).strip()
        return WikiResult(content, tokens_input, tokens_output, requests, tool_calls)

# -- Static helpers ----------------------------------------------------
