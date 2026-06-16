"""EventBus — unified event system for workflow progress, logging, and metrics.

All workflow components emit events through the EventBus.  Sinks
(CLI, file logger, metrics collector) consume events independently,
enabling clean separation of concerns.

Usage::

    from mnemo.core.workflow.events import (
        EventEmitter, WorkflowEvent, CLISink, LogSink,
    )

    emitter = EventEmitter()
    emitter.register(CLISink())
    emitter.register(LogSink(data_dir))

    emitter.emit(WorkflowEvent(
        event_type="step.start",
        step_name="validate",
        message="Validating file...",
    ))
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

EventType = Literal[
    # Workflow lifecycle
    "workflow.start",
    "workflow.end",
    "workflow.error",
    # Step lifecycle
    "step.start",
    "step.progress",
    "step.end",
    "step.error",
    "step.skip",
    # Stream events (real-time LLM output)
    "stream.chunk",
    "stream.end",
    # Log events
    "log.info",
    "log.warning",
    "log.error",
    "log.debug",
    # Metric events
    "metric.token",
    "metric.latency",
    "metric.embedding",
]


class StepStatus(str, Enum):
    """Standard step outcome."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# WorkflowEvent
# ---------------------------------------------------------------------------


class WorkflowEvent(BaseModel):
    """A single event emitted during workflow execution.

    Lightweight, serializable, and self-describing.  Sinks decide
    how to render or store it.
    """

    event_type: EventType
    """What kind of event this is."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    """When the event was created (UTC)."""

    run_id: str = ""
    """Unique identifier for the workflow run."""

    workflow_name: str = ""
    """Name of the workflow (e.g. 'add', 'search', 'ask')."""

    step_name: str | None = None
    """Step within the workflow, if applicable."""

    message: str = ""
    """Human-readable description."""

    data: dict[str, Any] = Field(default_factory=dict)
    """Arbitrary payload (metrics, chunk text, config, etc.)."""

    # -- convenience helpers --------------------------------------------------

    @classmethod
    def step_start(
        cls, run_id: str, workflow: str, step: str, message: str = "",
    ) -> WorkflowEvent:
        return cls(
            event_type="step.start",
            run_id=run_id,
            workflow_name=workflow,
            step_name=step,
            message=message or f"Starting {step}",
        )

    @classmethod
    def step_end(
        cls, run_id: str, workflow: str, step: str, message: str = "",
        data: dict[str, Any] | None = None,
    ) -> WorkflowEvent:
        return cls(
            event_type="step.end",
            run_id=run_id,
            workflow_name=workflow,
            step_name=step,
            message=message or f"Completed {step}",
            data=data or {},
        )

    @classmethod
    def step_error(
        cls, run_id: str, workflow: str, step: str, error: str,
        data: dict[str, Any] | None = None,
    ) -> WorkflowEvent:
        return cls(
            event_type="step.error",
            run_id=run_id,
            workflow_name=workflow,
            step_name=step,
            message=error,
            data=data or {},
        )

    @classmethod
    def step_progress(
        cls, run_id: str, workflow: str, step: str, message: str = "",
        data: dict[str, Any] | None = None,
    ) -> WorkflowEvent:
        return cls(
            event_type="step.progress",
            run_id=run_id,
            workflow_name=workflow,
            step_name=step,
            message=message,
            data=data or {},
        )

    @classmethod
    def step_skip(
        cls, run_id: str, workflow: str, step: str, reason: str = "",
    ) -> WorkflowEvent:
        return cls(
            event_type="step.skip",
            run_id=run_id,
            workflow_name=workflow,
            step_name=step,
            message=reason or f"Skipped {step}",
        )

    @classmethod
    def stream_chunk(
        cls, run_id: str, workflow: str, step: str,
        chunk: str, is_first: bool = False,
    ) -> WorkflowEvent:
        """A single chunk of streaming text from an LLM step."""
        return cls(
            event_type="stream.chunk",
            run_id=run_id,
            workflow_name=workflow,
            step_name=step,
            message=chunk,
            data={"is_first": is_first},
        )

    @classmethod
    def stream_end(
        cls, run_id: str, workflow: str, step: str,
    ) -> WorkflowEvent:
        """Signal end of streaming text from an LLM step."""
        return cls(
            event_type="stream.end",
            run_id=run_id,
            workflow_name=workflow,
            step_name=step,
            message="",
        )


# ---------------------------------------------------------------------------
# EventSink — abstract base
# ---------------------------------------------------------------------------


class EventSink(ABC):
    """Consumes workflow events.

    Subclass and override :meth:`handle` to produce side effects
    (print to terminal, write to file, send to telemetry, etc.).
    """

    @abstractmethod
    async def handle(self, event: WorkflowEvent) -> None:
        """Process a single event."""

    async def close(self) -> None:
        """Optional cleanup (flush buffers, close files, etc.)."""


# ---------------------------------------------------------------------------
# EventEmitter
# ---------------------------------------------------------------------------


class EventEmitter:
    """Fan-out event bus.

    Sinks register via :meth:`register`.  When :meth:`emit` is called
    the event is delivered to every sink.  If a sink raises an exception
    it is caught and logged — one bad sink does not take down the bus.

    Usage::

        emitter = EventEmitter()
        emitter.register(CLISink())
        emitter.register(LogSink(data_dir))

        emitter.emit(WorkflowEvent.step_start(
            run_id="r1", workflow="add", step="validate",
        ))
    """

    def __init__(self) -> None:
        self._sinks: list[EventSink] = []
        self.run_id: str = uuid.uuid4().hex[:12]
        self._start_time: float = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since this emitter was created."""
        return time.monotonic() - self._start_time

    # -- sink management ------------------------------------------------------

    def register(self, sink: EventSink) -> None:
        """Add a sink to receive future events."""
        self._sinks.append(sink)

    def remove(self, sink: EventSink) -> None:
        """Remove a previously registered sink."""
        try:
            self._sinks.remove(sink)
        except ValueError:
            pass

    # -- emit ----------------------------------------------------------------

    async def emit(self, event: WorkflowEvent) -> None:
        """Deliver *event* to all registered sinks.

        Automatically stamps the ``run_id`` if not already set.
        Sink errors are caught and logged — they never propagate
        to the caller.
        """
        # Stamp run_id if not set
        if not event.run_id:
            event.run_id = self.run_id

        for sink in self._sinks:
            try:
                await sink.handle(event)
            except Exception:
                import logging
                logging.getLogger("mnemo.workflow").warning(
                    "Sink %s failed to handle event %s",
                    type(sink).__name__, event.event_type, exc_info=True,
                )

    def emit_sync(self, event: WorkflowEvent) -> None:
        """Synchronous wrapper that works both inside and outside an event loop.

        - No running loop: uses ``asyncio.run()``.
        - Inside a running loop: schedules via ``create_task``
          (fire-and-forget — does NOT block; events are best-effort).
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.emit(event))
        else:
            # Already inside a running loop on the same thread —
            # create a task instead of blocking (avoids deadlock).
            loop.create_task(self.emit(event))

    async def close(self) -> None:
        """Close all sinks (flush buffers, etc.)."""
        for sink in self._sinks:
            try:
                await sink.close()
            except Exception:
                import logging
                logging.getLogger("mnemo.workflow").warning(
                    "Sink %s failed to close", type(sink).__name__, exc_info=True,
                )


# ---------------------------------------------------------------------------
# Built-in sinks
# ---------------------------------------------------------------------------


class NullSink(EventSink):
    """Discards all events — useful for testing or quiet mode."""

    async def handle(self, event: WorkflowEvent) -> None:
        pass


class LogSink(EventSink):
    """Writes structured JSON events to a log file.

    Stream chunks (``stream.chunk``) are written incrementally and flushed
    so the log file stays current even while generation is in progress.

    Parameters
    ----------
    log_path : Path
        File to append JSON-lines events to.
    """

    def __init__(self, log_path: str | Path) -> None:
        from pathlib import Path
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")

    async def handle(self, event: WorkflowEvent) -> None:
        import json
        self._file.write(event.model_dump_json() + "\n")
        # Flush stream chunks immediately so the log file stays current
        # even during long-running LLM generation.
        if event.event_type in ("stream.chunk", "stream.end", "metric.token"):
            self._file.flush()

    async def close(self) -> None:
        self._file.close()


class MetricsSink(EventSink):
    """Aggregates metrics from workflow runs.

    Tracks: total tokens (input/output), step latencies, embedding counts.
    """

    def __init__(self) -> None:
        self.total_tokens_input: int = 0
        self.total_tokens_output: int = 0
        self.total_embeddings: int = 0
        self.step_latencies: dict[str, list[float]] = {}

    async def handle(self, event: WorkflowEvent) -> None:
        if event.event_type == "metric.token":
            self.total_tokens_input += event.data.get("tokens_input", 0)
            self.total_tokens_output += event.data.get("tokens_output", 0)
        elif event.event_type == "metric.embedding":
            self.total_embeddings += event.data.get("count", 0)
        elif event.event_type == "metric.latency":
            step = event.step_name or "unknown"
            elapsed = event.data.get("elapsed_seconds", 0)
            self.step_latencies.setdefault(step, []).append(elapsed)

    def summary(self) -> dict[str, Any]:
        """Return aggregated metrics."""
        return {
            "tokens_input": self.total_tokens_input,
            "tokens_output": self.total_tokens_output,
            "embeddings": self.total_embeddings,
            "step_latencies": {
                k: {
                    "count": len(v),
                    "total_seconds": round(sum(v), 3),
                    "avg_seconds": round(sum(v) / len(v), 3) if v else 0,
                }
                for k, v in self.step_latencies.items()
            },
        }


class CLISink(EventSink):
    """Workflow event handler for CLI context.

    Logs step transitions and stream events via the Python logging
    subsystem (file handlers only — no console output).  Real-time
    user-facing display (thinking content, progress) is handled
    exclusively through the ``on_progress`` callback pipeline
    (``LegacyProgressSink`` → ``ProgressDisplay`` Rich spinner).
    """

    def __init__(self) -> None:
        import logging
        self._logger = logging.getLogger("mnemo.cli")

    async def handle(self, event: WorkflowEvent) -> None:
        """Log step transitions and stream events to file handlers."""
        if event.event_type in ("step.start", "step.end", "step.error",
                                "step.skip", "step.progress"):
            icon = {
                "step.start": "▶",
                "step.end": "✅",
                "step.error": "❌",
                "step.skip": "⏭️",
                "step.progress": "⏳",
            }.get(event.event_type, "•")
            step = event.step_name or "?"
            msg = event.message or ""
            if event.data.get("kind") == "thinking":
                self._logger.debug("🤔 [thinking] %s: %s", step, msg)
            elif msg:
                self._logger.info("%s %s: %s", icon, step, msg)
            else:
                self._logger.info("%s %s", icon, step)

        elif event.event_type == "stream.chunk":
            # Log text chunks at DEBUG level (file only — no terminal output).
            # User-facing display for thinking content goes through
            # LegacyProgressSink → on_progress → ProgressDisplay instead.
            step = event.step_name or "?"
            is_first = event.data.get("is_first")
            if is_first:
                self._logger.debug("▶ %s: [stream start]", step)
            self._logger.debug("[stream] %s", event.message)

        elif event.event_type == "stream.end":
            step = event.step_name or "?"
            self._logger.debug("▶ %s: [stream end]", step)
