"""Workflow configuration loader — load and merge TOML workflow definitions.

Resolution order (later overrides earlier):
1. Builtin workflow definitions in ``src/mnemo/builtin/workflows/``
2. User overrides in ``{data_dir}/.mnemo/workflows/``
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


class WorkflowConfigLoader:
    """Loads and merges workflow TOML definitions."""

    def __init__(self, builtin_dir: str | Path | None = None) -> None:
        if builtin_dir is None:
            builtin_dir = (
                Path(__file__).resolve().parent.parent.parent
                / "builtin" / "workflows"
            )
        self._builtin_dir = Path(builtin_dir)
        self._cache: dict[str, dict[str, Any]] = {}

    def load(
        self, workflow_name: str, data_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Load a workflow definition by name with optional user overrides."""
        cache_key = f"{workflow_name}:{data_dir}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        builtin = self._load_builtin(workflow_name)

        if data_dir:
            user = self._load_user_override(workflow_name, Path(data_dir))
            if user:
                builtin = self._deep_merge(builtin, user)

        self._cache[cache_key] = builtin
        return builtin

    def list_builtin_workflows(self) -> list[str]:
        """List available builtin workflow names."""
        if not self._builtin_dir.exists():
            return []
        return [
            p.stem.replace(".workflow", "")
            for p in self._builtin_dir.glob("*.workflow.toml")
        ]

    def validate(self, workflow_name: str) -> list[str]:
        """Validate a workflow definition, returning a list of issues."""
        issues: list[str] = []
        try:
            config = self.load(workflow_name)
        except Exception as exc:
            return [f"Failed to load: {exc}"]

        wf = config.get("workflow", config)
        steps = wf.get("steps", {})
        if not steps:
            issues.append("No [workflow.steps] defined")

        names = set(steps.keys())
        for name, raw in steps.items():
            if isinstance(raw, dict):
                for dep in raw.get("depends_on", []):
                    if dep not in names:
                        issues.append(
                            f"Step '{name}' depends on unknown step '{dep}'"
                        )
            if isinstance(raw, dict):
                t = raw.get("type", "function")
                if t not in ("function", "agent", "pipeline"):
                    issues.append(f"Step '{name}': unknown type '{t}'")

        return issues

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- internal -------------------------------------------------------------

    def _load_builtin(self, workflow_name: str) -> dict[str, Any]:
        path = self._builtin_dir / f"{workflow_name}.workflow.toml"
        if not path.exists():
            raise FileNotFoundError(
                f"Builtin workflow '{workflow_name}' not found at {path}. "
                f"Available: {self.list_builtin_workflows()}"
            )
        with open(path, "rb") as f:
            return tomllib.load(f)

    @staticmethod
    def _load_user_override(
        workflow_name: str, data_dir: Path,
    ) -> dict[str, Any] | None:
        user_path = (
            data_dir / ".mnemo" / "workflows" / f"{workflow_name}.workflow.toml"
        )
        if not user_path.exists():
            return None
        with open(user_path, "rb") as f:
            return tomllib.load(f)

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        result = dict(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = WorkflowConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
