"""DAG Builder — TOML step definitions → executable DAG via graphlib.

Uses Python 3.9+ standard library ``graphlib.TopologicalSorter`` for
topological sorting, parallel-level detection, and cycle detection.
This module adds Mnemo-specific semantics on top:
TOML parsing, StepConfig validation, and Step instance creation.

Key design decision (see UPGRADE_PLAN.md §0):
- graphlib handles the *graph math* (topo sort, ready-node detection, cycles)
- This module handles the *domain mapping* (TOML → graphlib graph)

Usage::

    from mnemo.core.workflow.dag import WorkflowDAG

    toml_config = load_toml("add.workflow.toml")
    dag = WorkflowDAG.from_config(toml_config)

    for level in dag.topological_levels():
        print(f"Parallel steps: {level}")  # ['B','C'] can run together
"""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from typing import Any

from mnemo.core.workflow.step import Step, StepConfig, StepRegistry


class WorkflowDAGError(Exception):
    """Raised when the DAG has structural issues (cycles, missing deps)."""


class WorkflowDAG:
    """A declarative, acyclic workflow built from TOML step definitions.

    Wraps ``graphlib.TopologicalSorter`` — the standard library handles
    topological ordering, cycle detection, and "ready node" parallelism
    detection.  This class adds:

    * TOML config parsing → StepConfig + Step instances
    * Validation (duplicate names, missing dependencies)
    * Iteration over parallel-executable levels

    Parameters
    ----------
    steps : list[Step]
        All steps in the workflow, each with ``depends_on`` declared.
    name : str
        Workflow name for logging / debugging.
    """

    def __init__(self, steps: list[Step], name: str = "") -> None:
        self.name = name
        self._steps: dict[str, Step] = {}
        self._sorter: TopologicalSorter | None = None

        # Index steps by name
        for step in steps:
            if step.name in self._steps:
                raise WorkflowDAGError(
                    f"Duplicate step name '{step.name}' in workflow '{name}'"
                )
            self._steps[step.name] = step

        # Build the graphlib predecessor map.
        # graphlib convention: graph[node] = {predecessors}
        # i.e. predecessors must complete BEFORE node can run.
        self._graph: dict[str, set[str]] = {}
        for step in steps:
            deps = set(step.config.depends_on)
            # Validate all deps exist
            missing = deps - set(self._steps.keys())
            if missing:
                raise WorkflowDAGError(
                    f"Step '{step.name}' depends on unknown step(s): {missing}"
                )
            self._graph[step.name] = deps

        # Detect cycles via graphlib
        try:
            TopologicalSorter(self._graph).prepare()
        except CycleError as exc:
            raise WorkflowDAGError(
                f"Cycle detected in workflow '{name}': {exc}"
            ) from exc

    # -- factory --------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict[str, Any], name: str = "") -> WorkflowDAG:
        """Build a DAG from a parsed TOML workflow definition.

        Expects::

            {
                "workflow": {
                    "name": "add",
                    "steps": {
                        "validate": {
                            "type": "function",
                            "depends_on": [],
                            ...
                        },
                        "copy": {
                            "type": "function",
                            "func_name": "copy_file",
                            "depends_on": ["validate"],
                            ...
                        },
                        ...
                    }
                }
            }

        Parameters
        ----------
        config : dict
            Parsed TOML dict with a ``[workflow]`` section.
        name : str
            Workflow name fallback if not in config.

        Returns
        -------
        WorkflowDAG

        Raises
        ------
        WorkflowDAGError
            If the config is invalid or the graph has cycles.
        """
        wf_section = config.get("workflow", config)
        wf_name = wf_section.get("name", name)

        raw_steps = wf_section.get("steps", {})
        if not raw_steps:
            raise WorkflowDAGError(
                f"Workflow '{wf_name}' has no [workflow.steps] defined"
            )

        steps: list[Step] = []
        for step_name, raw_cfg in raw_steps.items():
            if not isinstance(raw_cfg, dict):
                raise WorkflowDAGError(
                    f"Step '{step_name}' must be a TOML table"
                )

            # Build StepConfig
            step_config = StepConfig(
                name=step_name,
                type=raw_cfg.get("type", "function"),
                description=raw_cfg.get("description", ""),
                retry=raw_cfg.get("retry", 0),
                retry_delay_seconds=raw_cfg.get("retry_delay_seconds", 1.0),
                timeout_seconds=raw_cfg.get("timeout_seconds"),
                condition=raw_cfg.get("condition"),
                depends_on=raw_cfg.get("depends_on", []),
                output_key=raw_cfg.get("output_key"),
                progress_label=raw_cfg.get("progress_label", ""),
            )

            # Create the step instance via StepRegistry factory
            step = StepRegistry.create_step(step_config)

            # Apply type-specific config fields
            if step_config.type == "function":
                step.func_name = raw_cfg.get("func_name", step_name)
                step.kwargs = raw_cfg.get("kwargs", {})
            elif step_config.type == "agent":
                step.agent_name = raw_cfg.get("agent_name", "default")
                step.system_prompt = raw_cfg.get("system_prompt")
                step.tools = raw_cfg.get("tools", [])
                step.output_type_name = raw_cfg.get("output_type", "str")
                step.stream_enabled = raw_cfg.get("stream", False)
            elif step_config.type == "pipeline":
                step.workflow_name = raw_cfg.get("workflow_name", "")

            steps.append(step)

        return cls(steps, name=wf_name)

    # -- graph traversal ------------------------------------------------------

    @property
    def steps(self) -> dict[str, Step]:
        """All steps keyed by name."""
        return dict(self._steps)

    def get_step(self, name: str) -> Step:
        """Get a single step by name.

        Raises
        ------
        KeyError
            If the step doesn't exist.
        """
        if name not in self._steps:
            raise KeyError(
                f"Step '{name}' not found in workflow '{self.name}'"
            )
        return self._steps[name]

    def topological_levels(self) -> list[list[str]]:
        """Yield groups of parallel-executable step names.

        Each group (level) contains steps whose dependencies are all
        satisfied.  Steps within a level can run concurrently.
        Levels are returned in execution order.

        Delegates to ``TopologicalSorter.get_ready()`` which returns
        all nodes whose predecessor count has reached zero.

        Returns
        -------
        list[list[str]]
            Ordered levels.  e.g. ``[['A'], ['B', 'C'], ['D']]``
            means A runs first, then B and C in parallel, then D.
        """
        self._sorter = TopologicalSorter(self._graph)
        self._sorter.prepare()

        levels: list[list[str]] = []
        while self._sorter.is_active():
            ready = list(self._sorter.get_ready())
            if ready:
                levels.append(sorted(ready))
                for node in ready:
                    self._sorter.done(node)

        return levels

    def predecessors_of(self, step_name: str) -> set[str]:
        """Return the set of steps that must complete before *step_name*."""
        return self._graph.get(step_name, set())

    def successors_of(self, step_name: str) -> set[str]:
        """Return the set of steps that depend on *step_name*."""
        return {
            name
            for name, deps in self._graph.items()
            if step_name in deps
        }

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"WorkflowDAG(name='{self.name}', steps={len(self._steps)})"
