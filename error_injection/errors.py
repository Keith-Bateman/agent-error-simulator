"""
Data classes representing error injection specifications.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ErrorType(str, Enum):
    FORMAT = "format"
    LOGIC = "logic"
    TOOL_CALL = "tool_call"


@dataclass
class InjectionSpec:
    """
    Describes a single error injection point.

    agent_type   : "planner" | "worker" | "aggregator"
    agent_index  : 0-based index for workers; None means all of that type

    Injection timing (when the wrong output is produced):
    step         : backward-compat alias for inject_step (1-based).  If both
                   step and inject_step are given, inject_step wins.
    inject_step  : 1-based refinement step at which the error fires.
                   None/missing → 1 (first step).

    Detection timing (when the error becomes visible / gets flagged):
    detect_step  : 1-based step at which the error is reported in the worker's
                   errors[] list.  None → same as effective inject_step
                   (i.e. detection is immediate, the old default).
                   Must be >= inject_step; values < inject_step are clamped.
    detect_phase : which workflow phase first reports the error.
                   "workers"    — (default) worker self-reports at detect_step.
                   "aggregator" — worker stays silent; the coordinator logs the
                                  detection after the aggregator phase completes
                                  and records it in final_report.
                   "none"       — error is never flagged anywhere (silent
                                  propagation — useful for testing undetected
                                  errors reaching the final answer).

    error_type   : which kind of error to inject
    scale_factor : for logic errors — multiply the correct result by this value
    tool_status  : for tool_call errors — HTTP-style status code in the error body
    tool_message : for tool_call errors — message text in the error body
    """
    agent_type: str
    error_type: ErrorType
    agent_index: Optional[int] = None

    # ── injection timing ──────────────────────────────────────────────────────
    step: Optional[int] = None          # backward-compat alias for inject_step
    inject_step: Optional[int] = None  # takes precedence over step when both set

    # ── detection timing ──────────────────────────────────────────────────────
    detect_step: Optional[int] = None  # None → same as effective inject_step
    detect_phase: str = "workers"      # "workers" | "aggregator" | "none"

    # ── error parameters ──────────────────────────────────────────────────────
    scale_factor: float = -1.0
    tool_status: int = 404
    tool_message: str = "compute_riemann: endpoint not found"

    # ── derived helpers ───────────────────────────────────────────────────────

    def _effective_inject_step(self):
        # type: () -> int
        """The step at which the error is injected (1-based)."""
        if self.inject_step is not None:
            return self.inject_step
        if self.step is not None:
            return self.step
        return 1

    def _effective_detect_step(self):
        # type: () -> int
        """
        The step at which the error is first reported in the worker's errors[]
        list (only meaningful when detect_phase == "workers").
        Always >= effective inject_step.
        """
        inject = self._effective_inject_step()
        if self.detect_step is None:
            return inject
        return max(self.detect_step, inject)

    # ── predicate methods ─────────────────────────────────────────────────────

    def matches(self, agent_type, agent_index, step):
        # type: (str, int, int) -> bool
        """True when the error should be *injected* at this (type, index, step)."""
        if self.agent_type != agent_type:
            return False
        if self.agent_index is not None and self.agent_index != agent_index:
            return False
        return step == self._effective_inject_step()

    def is_detected_at_step(self, step):
        # type: (int) -> bool
        """True when the within-worker detection should fire at this step."""
        return step >= self._effective_detect_step()

    def is_detected_at_phase(self, phase):
        # type: (str) -> bool
        """True when *phase* is the reporting phase for this error."""
        return self.detect_phase == phase

    def detection_is_deferred(self):
        # type: () -> bool
        """True when detection does not happen immediately at the injection step."""
        return (
            self.detect_phase != "workers"
            or self._effective_detect_step() > self._effective_inject_step()
        )

    # ── constructor ───────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d):
        # type: (dict) -> InjectionSpec
        """
        Build an InjectionSpec from a config dict.

        Backward-compat: if only "step" is present it maps to inject_step
        semantics (immediate detection in workers phase).
        New fields "inject_step", "detect_step", "detect_phase" are optional.
        """
        step_val = d.get("step")
        inject_step_val = d.get("inject_step")

        raw_detect_step = d.get("detect_step")
        raw_detect_phase = d.get("detect_phase", "workers")

        return cls(
            agent_type=d["agent_type"],
            error_type=ErrorType(d["error_type"]),
            agent_index=d.get("agent_index"),
            step=int(step_val) if _nonempty(step_val) else None,
            inject_step=int(inject_step_val) if _nonempty(inject_step_val) else None,
            detect_step=int(raw_detect_step) if _nonempty(raw_detect_step) else None,
            detect_phase=str(raw_detect_phase) if raw_detect_phase else "workers",
            scale_factor=float(d.get("scale_factor", -1.0)),
            tool_status=int(d.get("tool_status", 404)),
            tool_message=str(d.get("tool_message", "compute_riemann: endpoint not found")),
        )


def _nonempty(v):
    # type: (object) -> bool
    """True when v is a non-None, non-blank value (used by from_dict)."""
    return v is not None and str(v).strip() != ""
