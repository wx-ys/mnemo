"""Step type system — the atomic unit of a workflow.

Three step flavours:

* **FunctionStep** — calls a registered Python function (deterministic).
* **AgentStep** — runs a pydantic-ai Agent with Tools (LLM-powered).
* **PipelineStep** — nests another workflow as a sub-step.

Each step is configured via TOML (``[workflow.steps.<name>]``) and
registered in the ``StepRegistry`` so the execution engine can look
them up by name.

Usage::

    from mnemo.core.workflow.step import (
        Step, StepConfig, FunctionStep, AgentStep, PipelineStep,
        StepRegistry,
    )

    @StepRegistry.register_function("validate_file")
    async def validate_file(ctx: WorkflowContext) -> dict: ...

    config = StepConfig(
        name="validate",
        type="function",
        func_name="validate_file",
        depends_on=[],
    )
    step = FunctionStep(config)
    result = await step.execute(ctx)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from mnemo.core.workflow.context import WorkflowContext

# ---------------------------------------------------------------------------
# StepConfig — TOML-sourced configuration for a single step
# ---------------------------------------------------------------------------


class StepConfig(BaseModel):
    """Configuration for one step, deserialized from TOML.

    Example TOML::

        [workflow.steps.judge]
        type = "agent"
        agent_name = "default"
        system_prompt = "prompts://judge_file"
        tools = ["judge_content_type", "judge_language"]
        output_type = "FileJudgment"
        depends_on = ["copy"]
        condition = "ctx.data.config.auto_wiki"
        output_key = "judgment"
        progress_label = "Analyzing file with LLM"
        timeout_seconds = 30
        retry = 2
    """

    name: str
    """Unique name within the workflow (matches TOML key)."""

    type: Literal["function", "agent", "pipeline"] = "function"
    """Step flavour."""

    description: str = ""

    # -- execution ------------------------------------------------------------

    retry: int = 0
    """Number of retries on failure (0 = no retry)."""

    retry_delay_seconds: float = 1.0
    """Seconds between retries (doubled each attempt)."""

    timeout_seconds: float | None = None
    """Max seconds before the step is cancelled."""

    # -- conditional ----------------------------------------------------------

    condition: str | None = None
    """Jinja2 / simple expression evaluated against ctx.data.
    If it evaluates to falsy, the step is skipped.
    Examples: ``"ctx.data.config.auto_wiki"``,
    ``"ctx.data.parsed_md != ''"``."""

    # -- dependencies ---------------------------------------------------------

    depends_on: list[str] = Field(default_factory=list)
    """Step names this step must wait for."""

    # -- output ---------------------------------------------------------------

    output_key: str | None = None
    """Key to store the result in ``ctx.data`` (defaults to step name)."""

    # -- progress -------------------------------------------------------------

    progress_label: str = ""
    """Human-readable label for CLI progress display."""


# ---------------------------------------------------------------------------
# Step — abstract base
# ---------------------------------------------------------------------------


class Step(ABC):
    """Abstract step — the atomic unit of a workflow DAG.

    Subclasses implement :meth:`_run` which receives the workflow
    context and returns an arbitrary result.  The engine calls
    :meth:`execute` which wraps ``_run`` with retry, timeout,
    condition checking, and event emission.
    """

    config: StepConfig

    def __init__(self, config: StepConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def output_key(self) -> str:
        return self.config.output_key or self.config.name

    # -- public API -----------------------------------------------------------

    async def execute(self, ctx: WorkflowContext) -> Any:
        """Run the step with retry, timeout, and events.

        Called by the execution engine.  Subclasses should override
        ``_run``, not this method.
        """
        result = await self._run(ctx)
        if self.output_key:
            ctx.set_output(self.output_key, result)
        # If result is a dict, spread its keys into ctx.data for
        # downstream steps to discover via direct key access.
        if isinstance(result, dict):
            for key, value in result.items():
                if key not in ctx.data:
                    ctx.data[key] = value
        return result

    # -- subclasses override --------------------------------------------------

    @abstractmethod
    async def _run(self, ctx: WorkflowContext) -> Any:
        """Core step logic — override in subclasses."""

    def should_run(self, ctx: WorkflowContext) -> bool:
        """Check the ``condition`` expression against current context.

        Returns True if the step should execute.
        """
        if not self.config.condition:
            return True
        return _eval_condition(self.config.condition, ctx)


# ---------------------------------------------------------------------------
# FunctionStep
# ---------------------------------------------------------------------------


class FunctionStep(Step):
    """Deterministic step — calls a registered Python async function.

    The function is looked up via ``StepRegistry.get_function(func_name)``.

    Example TOML::

        [workflow.steps.parse_to_md]
        type = "function"
        func_name = "parse_file_to_markdown"
        depends_on = ["judge"]
    """

    def __init__(self, config: StepConfig, func_name: str = "",
                 kwargs: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.func_name = func_name or config.name
        self.kwargs = kwargs or {}

    async def _run(self, ctx: WorkflowContext) -> Any:
        fn = StepRegistry.get_function(self.func_name)
        if fn is None:
            raise KeyError(
                f"Function '{self.func_name}' is not registered. "
                f"Use @StepRegistry.register_function('{self.func_name}') "
                f"to register it."
            )
        import inspect
        if inspect.iscoroutinefunction(fn):
            return await fn(ctx, **self.kwargs)
        else:
            return fn(ctx, **self.kwargs)


# ---------------------------------------------------------------------------
# AgentStep
# ---------------------------------------------------------------------------


class AgentStep(Step):
    """LLM-powered step — runs a pydantic-ai Agent with optional Tools.

    The agent is obtained from ``AgentRegistry`` (upgraded from the
    current ``AgentManager`` singleton).

    Example TOML::

        [workflow.steps.judge]
        type = "agent"
        agent_name = "default"
        system_prompt = "prompts://judge_file"
        tools = ["judge_content_type", "judge_language"]
        output_type = "FileJudgment"
        timeout_seconds = 30
        retry = 2
    """

    def __init__(
        self,
        config: StepConfig,
        agent_name: str = "default",
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        output_type: str = "str",
        stream: bool = False,
    ) -> None:
        super().__init__(config)
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.output_type_name = output_type
        self.stream_enabled = stream

    async def _run(self, ctx: WorkflowContext) -> Any:
        """Execute the pydantic-ai agent.

        Looks up the agent from AgentRegistry, resolves system prompt
        through PromptManager if it starts with ``prompts://``, and
        runs the agent with the current context as Deps.

        System prompt is passed via ``instructions=`` parameter
        (pydantic-ai v1.x API — ``agent._system_prompt`` does not
        exist and never did in v1.x).
        """
        # Resolve system prompt from PromptManager if needed
        resolved_prompt = self._resolve_system_prompt(ctx)

        # Build user prompt from context
        user_prompt = self._build_user_prompt(ctx)

        # Obtain the agent
        agent = self._get_agent()

        # Run the agent
        deps = ctx.to_deps()
        run_kwargs: dict[str, Any] = {"deps": deps}
        if resolved_prompt:
            run_kwargs["instructions"] = resolved_prompt

        if self.stream_enabled:
            # Streaming mode — collect text chunks and emit progress.
            # agent.run_stream() returns an async context manager that
            # yields a StreamedRunResult with .stream_text(delta=True).
            chunks: list[str] = []
            async with agent.run_stream(user_prompt, **run_kwargs) as streamed:
                async for chunk_text in streamed.stream_text(delta=True):
                    if chunk_text:
                        chunks.append(chunk_text)
                        ctx.emit(
                            event_type="step.progress",
                            step_name=self.name,
                            data={"chunk": chunk_text},
                        )
            return "".join(chunks)
        else:
            result = agent.run_sync(user_prompt, **run_kwargs)
            return result.output if hasattr(result, 'output') else result

    def _resolve_system_prompt(self, ctx: WorkflowContext) -> str:
        """Resolve a ``prompts://`` reference through PromptManager."""
        prompt_ref = self.system_prompt or ""
        if prompt_ref.startswith("prompts://"):
            prompt_name = prompt_ref[len("prompts://"):]
            # Support templated names like "wiki.{{ ctx.data.judgment.template_name }}"
            if "{{" in prompt_name:
                prompt_name = _resolve_template(prompt_name, ctx)
            try:
                # Use PromptManager if KB is available
                if ctx.kb and hasattr(ctx.kb, 'prompt_manager'):
                    return ctx.kb.prompt_manager.get_system_prompt(prompt_name)
            except Exception:
                import logging
                logging.getLogger("mnemo.workflow").debug(
                    "Failed to resolve prompt '%s' via PromptManager", prompt_name,
                )
            return ""
        return prompt_ref

    def _build_user_prompt(self, ctx: WorkflowContext) -> str:
        """Build the user prompt from context inputs."""
        # Default: use the primary input key as the user prompt
        user_input = ctx.get_input("user_prompt", "")
        if not user_input:
            # Fallback: stringify the data dict
            import json
            user_input = json.dumps(ctx.data, ensure_ascii=False, default=str)
        return user_input

    def _get_agent(self):
        """Obtain the pydantic-ai Agent from AgentRegistry."""
        from mnemo.core.agent_manager import AgentManager
        am = AgentManager.get_instance()
        if not am._initialized:
            raise RuntimeError(
                "AgentManager is not initialized. "
                "Call AgentManager.get_instance().init(config) first."
            )
        # For now, use the existing AgentManager.
        # Phase 2 will upgrade this to AgentRegistry with Tool support.
        return am.get_agent(self.agent_name, output_type=str)


# ---------------------------------------------------------------------------
# PipelineStep
# ---------------------------------------------------------------------------


class PipelineStep(Step):
    """Nests another workflow as a sub-step.

    Example TOML::

        [workflow.steps.search_main]
        type = "pipeline"
        workflow_name = "search"
        depends_on = ["analyze_question"]
    """

    def __init__(self, config: StepConfig, workflow_name: str = "") -> None:
        super().__init__(config)
        self.workflow_name = workflow_name

    async def _run(self, ctx: WorkflowContext) -> Any:
        """Execute the nested workflow within the current context.

        Creates a sub-context, runs the child workflow engine,
        and merges results back.
        """
        from mnemo.core.workflow.engine import WorkflowEngine

        sub_ctx = WorkflowContext(
            workflow_name=self.workflow_name,
            kb=ctx.kb,
            config=ctx.config,
            emitter=ctx.emitter,
        )
        # Propagate relevant data into sub-context
        sub_ctx.data = dict(ctx.data)
        sub_ctx.inputs = dict(ctx.inputs)

        engine = WorkflowEngine()
        result_ctx = await engine.execute_by_name(self.workflow_name, sub_ctx)

        # Merge results back
        ctx.data.update(result_ctx.data)
        return result_ctx


# ---------------------------------------------------------------------------
# StepRegistry
# ---------------------------------------------------------------------------


class StepRegistry:
    """Registry for function implementations and step type factories.

    Functions registered via ``register_function`` are callable by name
    from ``FunctionStep``.  This decouples workflow definitions (TOML)
    from their implementations (Python).

    Usage::

        @StepRegistry.register_function("validate_file")
        async def validate_file(ctx: WorkflowContext) -> dict:
            ...
    """

    _functions: dict[str, Callable[..., Any]] = {}
    _step_classes: dict[str, type[Step]] = {}

    # -- function registry ---------------------------------------------------

    @classmethod
    def register_function(cls, name: str) -> Callable:
        """Decorator — register a function under *name*.

        Usage::

            @StepRegistry.register_function("validate_file")
            async def validate_file(ctx: WorkflowContext) -> dict: ...
        """
        def decorator(fn: Callable) -> Callable:
            cls._functions[name] = fn
            return fn
        return decorator

    @classmethod
    def get_function(cls, name: str) -> Callable[..., Any] | None:
        """Look up a registered function by name."""
        return cls._functions.get(name)

    @classmethod
    def list_functions(cls) -> list[str]:
        """Return all registered function names."""
        return list(cls._functions.keys())

    # -- step class registry (for type = "agent" / "function" / ...) --------

    @classmethod
    def register_step_class(cls, type_name: str, step_cls: type[Step]) -> None:
        """Register a step class for a given type name."""
        cls._step_classes[type_name] = step_cls

    @classmethod
    def get_step_class(cls, type_name: str) -> type[Step] | None:
        """Look up a step class by type name."""
        return cls._step_classes.get(type_name)

    @classmethod
    def create_step(cls, config: StepConfig) -> Step:
        """Factory — create a Step instance from its config."""
        if config.type == "function":
            return FunctionStep(config, func_name=config.name)
        elif config.type == "agent":
            return AgentStep(config)
        elif config.type == "pipeline":
            return PipelineStep(config)
        else:
            raise ValueError(f"Unknown step type: {config.type}")


# ---------------------------------------------------------------------------
# Condition evaluator
# ---------------------------------------------------------------------------


def _eval_condition(condition: str, ctx: WorkflowContext) -> bool:
    """Evaluate a simple condition expression against the context.

    Supports basic patterns:
    - ``ctx.data.key`` — truthiness check
    - ``ctx.data.key == 'value'`` — equality
    - ``ctx.data.key != 'value'`` — inequality
    - ``ctx.data.key and ctx.data.other`` — boolean operators
    """
    # Simple truthiness: "ctx.data.xxx" or "ctx.data.xxx.yyy"
    if re.match(r'^ctx\.data\.[a-zA-Z_][a-zA-Z0-9_.]*$', condition):
        # Strip "ctx.data." prefix, then resolve from ctx.data
        path = condition[len("ctx.data."):]
        current: Any = ctx.data
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return False
        return bool(current)

    # Equality: ctx.data.key == 'value'
    eq_match = re.match(
        r"^ctx\.data\.([a-zA-Z_][a-zA-Z0-9_.]*)"
        r"\s*==\s*'([^']*)'$", condition,
    )
    if eq_match:
        path, expected = eq_match.groups()
        actual = ctx.resolve_ref(path)
        return str(actual) == expected

    # Not-equal: ctx.data.key != 'value'
    ne_match = re.match(
        r"^ctx\.data\.([a-zA-Z_][a-zA-Z0-9_.]*)"
        r"\s*!=\s*'([^']*)'$", condition,
    )
    if ne_match:
        path, expected = ne_match.groups()
        actual = ctx.resolve_ref(path)
        return str(actual) != expected

    # Boolean: a and b
    if " and " in condition:
        return all(
            _eval_condition(sub.strip(), ctx)
            for sub in condition.split(" and ")
        )

    # Boolean: a or b
    if " or " in condition:
        return any(
            _eval_condition(sub.strip(), ctx)
            for sub in condition.split(" or ")
        )

    # Fallback: use Python eval (sandbox: only ctx is available)
    import logging
    logging.getLogger("mnemo.workflow").warning(
        "Could not evaluate condition: %s — defaulting to True", condition,
    )
    return True


def _resolve_template(template: str, ctx: WorkflowContext) -> str:
    """Resolve ``{{ ctx.data.xxx }}`` patterns in a template string."""
    def _replacer(match: re.Match) -> str:
        path = match.group(1).strip()
        if path.startswith("ctx.data."):
            ref = path[len("ctx.data."):]
            val = ctx.resolve_ref(ref)
            return str(val) if val is not None else ""
        return match.group(0)
    return re.sub(r'\{\{\s*([^}]+)\s*\}\}', _replacer, template)
