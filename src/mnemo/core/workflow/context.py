"""Workflow context — typed state carrier and pydantic-ai Deps adapter.

``WorkflowContext`` flows through every step in a workflow DAG.
Steps read inputs from it and write outputs back into it.
For ``AgentStep``, it also serves as the pydantic-ai dependency
injection source (via ``WorkflowDeps``).

Usage::

    from mnemo.core.workflow.context import WorkflowContext, WorkflowDeps

    ctx = WorkflowContext(
        workflow_name="add",
        emitter=emitter,
        kb=kb_instance,
        config={"auto_wiki": True},
    )
    ctx.set_input("source_path", Path("/data/paper.pdf"))

    # In AgentStep: extract Deps from context
    deps = ctx.to_deps()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from mnemo.core.workflow.events import EventEmitter, NullSink

if TYPE_CHECKING:
    from mnemo.core.kb import KnowledgeBase


# ---------------------------------------------------------------------------
# WorkflowDeps — pydantic-ai dependency injection container
# ---------------------------------------------------------------------------


@dataclass
class WorkflowDeps:
    """Dependencies injected into pydantic-ai Agent runs.

    Pass this to ``agent.run_sync(prompt, deps=...)`` to give the
    agent access to the knowledge base, config, and event bus.
    """

    kb: Any = None  # KnowledgeBase (typed as Any to avoid pydantic schema issues)
    """The KnowledgeBase instance (for search_kb tool, etc.)."""

    config: dict[str, Any] = field(default_factory=dict)
    """Resolved workflow configuration."""

    emitter: EventEmitter = field(default_factory=lambda: EventEmitter())
    """Event bus for progress / logging."""


# ---------------------------------------------------------------------------
# WorkflowContext
# ---------------------------------------------------------------------------


class WorkflowContext(BaseModel):
    """Typed context propagated through the workflow DAG.

    Dual role:
    1. **Data carrier**: steps read inputs via ``self.data`` and write
       outputs to ``self.data[key]``.
    2. **Deps adapter**: ``to_deps()`` extracts kb/config/emitter for
       pydantic-ai Agent injection.

    Parameters
    ----------
    workflow_name : str
        Name of the workflow definition (e.g. 'add', 'search').
    emitter : EventEmitter, optional
        Event bus.  A NullSink emitter is used if none given.
    kb : KnowledgeBase, optional
        The owning KB instance.  Needed by AgentSteps that call
        KB operations as tools.
    config : dict, optional
        Resolved configuration for this workflow run.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # -- identity ------------------------------------------------------------

    workflow_name: str
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])

    # -- DI (pydantic-ai Deps) -----------------------------------------------

    kb: Any = None
    """KnowledgeBase instance (typed as Any to avoid circular import)."""

    config: dict[str, Any] = Field(default_factory=dict)

    emitter: Any = Field(default_factory=NullSink)
    """EventEmitter (typed as Any — actual emitter is set by the engine)."""

    # -- data flow -----------------------------------------------------------

    data: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary step outputs keyed by step name or output_key."""

    # -- diagnostics -----------------------------------------------------------

    diagnostic: Any = None
    """DiagnosticContext for pipeline observability (None when disabled)."""

    # -- input ----------------------------------------------------------------

    inputs: dict[str, Any] = Field(default_factory=dict)
    """Workflow-level inputs set before execution begins."""

    def set_input(self, key: str, value: Any) -> None:
        """Set a workflow-level input before execution begins."""
        self.inputs[key] = value

    def get_input(self, key: str, default: Any = None) -> Any:
        """Read a workflow-level input."""
        return self.inputs.get(key, default)

    # -- data helpers ---------------------------------------------------------

    def set_output(self, key: str, value: Any) -> None:
        """Store a step output in the shared data namespace."""
        self.data[key] = value

    def get_output(self, key: str, default: Any = None) -> Any:
        """Read a previously stored step output."""
        return self.data.get(key, default)

    def resolve_ref(self, ref: str) -> Any:
        """Resolve a `data.some_key.nested` reference string.

        Used for dynamic system_prompt paths like
        ``"data.judgment.template_name"``.
        """
        parts = ref.split(".")
        current: Any = self.data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current

    # -- deps adapter ---------------------------------------------------------

    def to_deps(self) -> WorkflowDeps:
        """Extract a pydantic-ai Deps container from this context."""
        return WorkflowDeps(
            kb=self.kb,
            config=self.config,
            emitter=self.emitter,
        )

    # -- event helpers --------------------------------------------------------

    def emit(self, event_type: str, step_name: str | None = None,
             message: str = "", data: dict[str, Any] | None = None) -> None:
        """Emit a workflow event through the bus (sync, fire-and-forget)."""
        from mnemo.core.workflow.events import WorkflowEvent

        wf_event = WorkflowEvent(
            event_type=event_type,  # type: ignore[arg-type]
            run_id=self.run_id,
            workflow_name=self.workflow_name,
            step_name=step_name,
            message=message,
            data=data or {},
        )
        try:
            self.emitter.emit_sync(wf_event)
        except Exception:
            import logging
            logging.getLogger("mnemo.workflow").debug(
                "Failed to emit event %s", event_type, exc_info=True,
            )
