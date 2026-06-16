"""Workflow Engine — execute a WorkflowDAG with retry, timeout, conditions, and events.

The engine takes a ``WorkflowDAG`` and a ``WorkflowContext``, then
executes each level of steps concurrently (via ``asyncio.gather``).
Per-step behaviour:

* **Condition check** — skip if the step's ``condition`` evaluates to falsy.
* **Retry** — on failure, retry up to ``retry`` times with exponential backoff.
* **Timeout** — if ``timeout_seconds`` is set, cancel the step after that duration.
* **Events** — emit ``step.start``, ``step.end``, ``step.error``, ``step.skip``
  events through the context's emitter.
* **Error isolation** — one step failing does not block sibling steps in the same level.

Usage::

    from mnemo.core.workflow.engine import WorkflowEngine
    from mnemo.core.workflow.dag import WorkflowDAG
    from mnemo.core.workflow.context import WorkflowContext

    dag = WorkflowDAG.from_config(toml_config)
    ctx = WorkflowContext(workflow_name="add", kb=kb_instance)

    engine = WorkflowEngine()
    result_ctx = await engine.execute(dag, ctx)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mnemo.core.workflow.context import WorkflowContext
from mnemo.core.workflow.dag import WorkflowDAG
from mnemo.core.workflow.events import WorkflowEvent
from mnemo.core.workflow.step import Step

logger = logging.getLogger("mnemo.workflow")


class WorkflowEngine:
    """Executes a WorkflowDAG level by level.

    Parameters
    ----------
    fail_fast : bool
        If True, stop the entire workflow on the first step failure.
        Default is False (error isolation).
    """

    def __init__(self, fail_fast: bool = False) -> None:
        self.fail_fast = fail_fast

    # -- public API ------------------------------------------------------------

    async def execute(
        self,
        dag: WorkflowDAG,
        ctx: WorkflowContext,
    ) -> WorkflowContext:
        """Execute all steps in *dag* against *ctx*.

        Parameters
        ----------
        dag : WorkflowDAG
            The workflow to execute.
        ctx : WorkflowContext
            Initial context (inputs, kb reference, emitter, etc.).

        Returns
        -------
        WorkflowContext
            Updated context with all step outputs stored in ``ctx.data``.
        """
        # Emit workflow start
        ctx.emit(
            event_type="workflow.start",
            message=f"Starting workflow '{dag.name}'",
            data={"step_count": len(dag)},
        )

        levels = dag.topological_levels()
        logger.info(
            "Workflow '%s' starting: %d steps in %d levels",
            dag.name, len(dag), len(levels),
        )

        for level_idx, level_names in enumerate(levels):
            logger.debug(
                "Level %d/%d: %s", level_idx + 1, len(levels), level_names,
            )

            # Gather all steps in this level (they can run in parallel)
            tasks = [
                self._execute_step(dag.get_step(name), ctx)
                for name in level_names
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for failures
            for name, result in zip(level_names, results):
                if isinstance(result, (FileNotFoundError, FileExistsError)):
                    raise result
                if isinstance(result, Exception):
                    logger.error(
                        "Step '%s' in workflow '%s' failed: %s",
                        name, dag.name, result,
                    )
                    ctx.emit(
                        event_type="step.error",
                        step_name=name,
                        message=str(result),
                    )
                    if self.fail_fast:
                        ctx.emit(
                            event_type="workflow.error",
                            message=f"Workflow aborted at step '{name}': {result}",
                        )
                        return ctx

        # Emit workflow end
        ctx.emit(
            event_type="workflow.end",
            message=f"Workflow '{dag.name}' completed",
            data={"step_count": len(dag)},
        )
        logger.info("Workflow '%s' completed successfully", dag.name)
        return ctx

    async def execute_by_name(
        self, workflow_name: str, ctx: WorkflowContext,
    ) -> WorkflowContext:
        """Execute a workflow by name (loaded from config).

        This is used by ``PipelineStep`` to nest workflows.
        """
        from mnemo.core.workflow.dag import WorkflowDAG
        # Load the workflow config — for now this is a stub
        # Phase 2 will add proper WorkflowConfigLoader
        raise NotImplementedError(
            "execute_by_name requires WorkflowConfigLoader (Phase 2)"
        )

    # -- internal -------------------------------------------------------------

    async def _execute_step(self, step: Step, ctx: WorkflowContext) -> None:
        """Execute one step with retry, timeout, condition, and events."""
        # Condition check
        if not step.should_run(ctx):
            reason = f"condition '{step.config.condition}' evaluated to False"
            logger.debug("Step '%s' skipped: %s", step.name, reason)
            ctx.emit(
                event_type="step.skip",
                step_name=step.name,
                message=reason,
            )
            return

        # Emit step start
        ctx.emit(
            event_type="step.start",
            step_name=step.name,
            message=step.config.progress_label or f"Running {step.name}",
        )
        start_time = asyncio.get_event_loop().time()

        # Retry loop
        last_error: Exception | None = None
        max_attempts = step.config.retry + 1

        for attempt in range(max_attempts):
            try:
                if step.config.timeout_seconds:
                    result = await asyncio.wait_for(
                        step.execute(ctx),
                        timeout=step.config.timeout_seconds,
                    )
                else:
                    result = await step.execute(ctx)

                # Success — emit step end with metrics
                elapsed = asyncio.get_event_loop().time() - start_time
                ctx.emit(
                    event_type="step.end",
                    step_name=step.name,
                    message=f"Completed {step.name}",
                    data={
                        "elapsed_seconds": round(elapsed, 3),
                        "attempts": attempt + 1,
                    },
                )
                ctx.emit(
                    event_type="metric.latency",
                    step_name=step.name,
                    data={"elapsed_seconds": round(elapsed, 3)},
                )
                return

            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Step '{step.name}' timed out after "
                    f"{step.config.timeout_seconds}s"
                )
                logger.warning("Step '%s' timed out (attempt %d/%d)",
                               step.name, attempt + 1, max_attempts)

            except (FileNotFoundError, FileExistsError):
                # Validation errors must propagate immediately
                raise
            except Exception as exc:
                last_error = exc
                logger.warning("Step '%s' failed (attempt %d/%d): %s",
                               step.name, attempt + 1, max_attempts, exc)

            # Wait before retry (exponential backoff)
            if attempt < max_attempts - 1:
                delay = step.config.retry_delay_seconds * (2 ** attempt)
                await asyncio.sleep(delay)

        # All retries exhausted — emit error event
        total_elapsed = asyncio.get_event_loop().time() - start_time
        error_msg = str(last_error) if last_error else "Unknown error"
        ctx.emit(
            event_type="step.error",
            step_name=step.name,
            message=error_msg,
            data={
                "elapsed_seconds": round(total_elapsed, 3),
                "attempts": max_attempts,
            },
        )

        if self.fail_fast:
            raise last_error  # type: ignore[misc]
