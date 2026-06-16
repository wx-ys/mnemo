"""Pipeline diagnostics — structured trace capture for search quality optimization.

Provides a ``DiagnosticSink`` (EventSink) that writes JSONL trace files and
a ``DiagnosticContext`` that carries timing/config through the call chain.

Usage::

    from mnemo.core.diagnostics import DiagnosticContext, DiagnosticSink

    # Create context and sink when --diagnose is set
    diag_ctx = DiagnosticContext(
        trace_file=Path("/path/to/trace.jsonl"),
        verbose=True,  # also print to terminal
    )

    # Register on the event emitter
    emitter.register(DiagnosticSink(diag_ctx))

    # Steps check ctx.diagnostic.enabled and emit data:
    if ctx.diagnostic and ctx.diagnostic.enabled:
        ctx.emit("step.progress", step_name="my_step",
                 data={"_diagnostic": {"substage": "chunker", ...}})
"""

from __future__ import annotations

import json
import math
import time as _time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mnemo.core.workflow.events import EventSink, WorkflowEvent


# ============================================================================
# Helpers
# ============================================================================


def truncate_vector(vector: list[float], max_dims: int = 8) -> dict[str, Any]:
    """Return a preview of a vector: first N dims + total dimension count."""
    return {
        "preview": [round(v, 6) for v in vector[:max_dims]],
        "total_dims": len(vector),
    }


def truncate_text(text: str, max_chars: int = 200) -> dict[str, Any]:
    """Return a preview of text: first N chars + total char count."""
    return {
        "preview": text[:max_chars],
        "total_chars": len(text),
    }


def compute_distance_stats(distances: list[float]) -> dict[str, float]:
    """Compute min, max, mean, std for a list of distance values."""
    if not distances:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "count": 0}
    n = len(distances)
    mean = sum(distances) / n
    variance = sum((d - mean) ** 2 for d in distances) / n
    return {
        "min": round(min(distances), 6),
        "max": round(max(distances), 6),
        "mean": round(mean, 6),
        "std": round(math.sqrt(variance), 6),
        "count": n,
    }


# ============================================================================
# DiagnosticContext
# ============================================================================


@dataclass
class DiagnosticContext:
    """Lightweight context carried through a workflow run for diagnostics.

    Attributes
    ----------
    enabled : bool
        Whether diagnostics are active.
    trace_file : Path or None
        Path to the JSONL trace file.
    verbose : bool
        If True, also print diagnostic summaries to terminal.
    max_vector_preview_dims : int
        How many vector dimensions to include in previews.
    max_text_preview_chars : int
        How many characters of text to include in previews.
    """

    enabled: bool = False
    trace_file: Path | None = None
    verbose: bool = False
    max_vector_preview_dims: int = 8
    max_text_preview_chars: int = 200
    _stage_timers: dict[str, float] = field(default_factory=dict)
    _file: Any = None  # file handle for direct writes (search path)
    _run_id: str = ""  # set by KB before passing to searcher

    def start_stage(self, stage_name: str) -> None:
        """Record the start time of a pipeline stage."""
        self._stage_timers[stage_name] = _time.monotonic()

    def elapsed(self, stage_name: str) -> float:
        """Return elapsed seconds since ``start_stage(stage_name)``."""
        start = self._stage_timers.get(stage_name)
        if start is None:
            return 0.0
        return round(_time.monotonic() - start, 4)

    def stop_stage(self, stage_name: str) -> float:
        """Stop timing and return elapsed seconds."""
        elapsed = self.elapsed(stage_name)
        self._stage_timers.pop(stage_name, None)
        return elapsed

    def emit_diagnostic(
        self, stage: str, data: dict[str, Any],
        event_type: str = "stage.end",
        message: str = "",
    ) -> None:
        """Write a diagnostic event directly to the trace file.

        Used by the search pipeline (which doesn't go through EventBus).
        Formats and writes a JSONL line immediately.

        Parameters
        ----------
        stage : str
            Pipeline stage name (e.g. 'query_embedding', 'vector_ann').
        data : dict
            Diagnostic payload to serialize.
        event_type : str
            Event type label. Default 'stage.end'.
        message : str
            Optional human-readable message.
        """
        if not self.enabled or self.trace_file is None:
            return

        # Ensure file is open
        if self._file is None:
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.trace_file, "a", encoding="utf-8")

        record = {
            "event": event_type,
            "ts": datetime.now(UTC).isoformat(),
            "run_id": self._run_id,
            "stage": stage,
            "message": message,
            "data": data,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

        # Verbose terminal output
        if self.verbose:
            self._print_verbose(stage, data)

    def _print_verbose(self, stage: str, data: dict) -> None:
        """Print a one-line diagnostic summary to terminal."""
        icon = VerboseDiagnosticSink._STAGE_ICONS.get(stage, "📊")
        try:
            from mnemo.cli.formatter import console
        except Exception:
            return

        # Format based on stage
        if stage == "query_embedding":
            model = data.get("model", "?")
            dim = data.get("dimension", 0)
            console.print(
                f"  {icon} [dim]Query embed:[/dim] {model}, dim={dim}"
            )
        elif stage == "vector_ann":
            table = data.get("table", "?")
            count = data.get("result_count", 0)
            stats = data.get("distance_stats", {})
            console.print(
                f"  {icon} [dim]ANN ({table}):[/dim] {count} results, "
                f"dist=[min={stats.get('min',0):.3f}, "
                f"max={stats.get('max',0):.3f}, "
                f"mean={stats.get('mean',0):.3f}]"
            )
        elif stage == "keyword_bm25":
            count = data.get("result_count", 0)
            console.print(
                f"  {icon} [dim]BM25:[/dim] {count} results"
            )
        elif stage == "graph_expand":
            count = data.get("count", 0)
            console.print(
                f"  {icon} [dim]Graph:[/dim] {count} files"
            )
        elif stage == "rrf_fuse":
            weights = data.get("weights", {})
            count = data.get("result_count", 0)
            w_str = ", ".join(f"{k}={v}" for k, v in weights.items())
            console.print(
                f"  {icon} [dim]RRF:[/dim] weights={{{w_str}}}, "
                f"{count} fused results"
            )

    def close(self) -> None:
        """Close the trace file handle if open."""
        if self._file is not None:
            self._file.close()
            self._file = None


# ============================================================================
# DiagnosticSink
# ============================================================================


class DiagnosticSink(EventSink):
    """Writes structured diagnostic events to a JSONL trace file.

    Filters for events that have a ``_diagnostic`` key in their ``data``
    dict.  All other events are silently ignored.

    Parameters
    ----------
    context : DiagnosticContext
        Shared diagnostic configuration and state.
    """

    def __init__(self, context: DiagnosticContext) -> None:
        self._ctx = context
        self._file = None
        self._line_count = 0

    def _ensure_file(self) -> None:
        if self._file is not None:
            return
        if self._ctx.trace_file is None:
            return
        self._ctx.trace_file.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._ctx.trace_file, "a", encoding="utf-8")

    async def handle(self, event: WorkflowEvent) -> None:
        """Write diagnostic-worthy events to the JSONL trace file."""
        # Only write events that carry diagnostic payloads
        diagnostic = event.data.get("_diagnostic") if event.data else None
        if not diagnostic:
            return

        self._ensure_file()
        if self._file is None:
            return

        record = {
            "event": event.event_type,
            "ts": event.timestamp.isoformat(),
            "run_id": event.run_id,
            "stage": event.step_name,
            "message": event.message,
            "data": diagnostic,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        self._line_count += 1

    async def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def line_count(self) -> int:
        """Number of diagnostic events written so far."""
        return self._line_count


# ============================================================================
# VerboseDiagnosticSink
# ============================================================================


class VerboseDiagnosticSink(EventSink):
    """Prints key diagnostic events to the terminal in real-time.

    Used alongside ``DiagnosticSink`` when ``--verbose`` is also set.
    Temporarily pauses the Rich spinner to print diagnostic summaries
    without garbling the progress display.

    Parameters
    ----------
    context : DiagnosticContext
        Shared diagnostic configuration.
    """

    # Stage icons for terminal display
    _STAGE_ICONS: dict[str, str] = {
        "validate_file": "🔍",
        "copy_file": "📋",
        "create_metadata": "🏷️",
        "parse_file_to_markdown": "📝",
        "generate_wiki": "🤖",
        "extract_entities": "🧠",
        "embed_chunks": "🧮",
        "write_index": "📌",
        "query_embedding": "🔤",
        "vector_ann": "📊",
        "keyword_bm25": "📝",
        "graph_expand": "🔗",
        "rrf_fuse": "🔀",
    }

    def __init__(self, context: DiagnosticContext) -> None:
        self._ctx = context

    async def handle(self, event: WorkflowEvent) -> None:
        """Print a one-line diagnostic summary to the terminal."""
        diagnostic = event.data.get("_diagnostic") if event.data else None
        if not diagnostic:
            return

        stage = event.step_name or "?"
        icon = self._STAGE_ICONS.get(stage, "📊")

        lines = self._format_diagnostic(stage, icon, diagnostic)
        for line in lines:
            # Use Rich console to print — this works alongside the spinner
            # because Rich handles concurrent output gracefully.
            try:
                from mnemo.cli.formatter import console
                console.print(f"  {line}")
            except Exception:
                # Fallback: plain print if Rich console is unavailable
                print(f"  {line}")

    def _format_diagnostic(
        self, stage: str, icon: str, data: dict
    ) -> list[str]:
        """Format a diagnostic payload as one or more terminal lines."""
        substage = data.get("substage", "")

        if stage == "parse_file_to_markdown":
            md = data.get("md_preview", "")
            chars = data.get("md_total_chars", 0)
            lines_n = data.get("md_lines", 0)
            return [
                f"{icon} [dim]MD parsed:[/dim] {chars} chars, {lines_n} lines "
                f"[dim]| preview:[/dim] {md[:80]}..."
            ]

        elif stage == "generate_wiki":
            model = data.get("model", "?")
            tokens = data.get("tokens_input", 0) + data.get("tokens_output", 0)
            chars = data.get("wiki_chars", 0)
            return [
                f"{icon} [dim]Wiki:[/dim] model={model}, {tokens} tokens, "
                f"{chars} chars"
            ]

        elif stage == "embed_chunks":
            if substage == "chunker":
                chunker = data.get("chunker", "?")
                count = data.get("chunk_count", 0)
                size = data.get("max_chunk_size", "?")
                return [
                    f"{icon} [dim]Chunk:[/dim] {chunker}, "
                    f"{count} chunks, max_size={size}"
                ]
            elif substage == "embedding":
                model = data.get("model", "?")
                dim = data.get("dimension", 0)
                count = data.get("vector_count", 0)
                return [
                    f"{icon} [dim]Embed:[/dim] {model}, dim={dim}, "
                    f"{count} vectors"
                ]
            else:
                return [f"{icon} {stage}: {json.dumps(data, ensure_ascii=False)}"]

        elif stage == "query_embedding":
            model = data.get("model", "?")
            dim = data.get("dimension", 0)
            return [
                f"{icon} [dim]Query embed:[/dim] {model}, dim={dim}"
            ]

        elif stage == "vector_ann":
            table = data.get("table", "?")
            count = data.get("result_count", 0)
            stats = data.get("distance_stats", {})
            return [
                f"{icon} [dim]ANN ({table}):[/dim] {count} results, "
                f"dist=[min={stats.get('min',0):.3f}, "
                f"max={stats.get('max',0):.3f}, "
                f"mean={stats.get('mean',0):.3f}]"
            ]

        elif stage == "keyword_bm25":
            count = data.get("result_count", 0)
            return [
                f"{icon} [dim]BM25:[/dim] {count} results"
            ]

        elif stage == "graph_expand":
            count = data.get("count", 0)
            hops = data.get("max_hops", "?")
            return [
                f"{icon} [dim]Graph:[/dim] {count} files (hops={hops})"
            ]

        elif stage == "rrf_fuse":
            weights = data.get("weights", {})
            count = data.get("result_count", 0)
            w_str = ", ".join(f"{k}={v}" for k, v in weights.items())
            return [
                f"{icon} [dim]RRF:[/dim] weights={{{w_str}}}, "
                f"{count} fused results"
            ]

        else:
            # Generic fallback
            summary = json.dumps(data, ensure_ascii=False)
            if len(summary) > 120:
                summary = summary[:117] + "..."
            return [f"{icon} [dim]{stage}:[/dim] {summary}"]
