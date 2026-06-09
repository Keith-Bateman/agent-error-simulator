"""
WorkerAgent — computes the integral over one sub-interval across K refinement steps.

Protocol (per step)
───────────────────
Turn 1  user   → Ask agent to request a compute_riemann tool call (JSON).
Turn 1  asst   → {"action": "compute_riemann", "coefficients": [...], "a": ..., "b": ..., "n": ...}

[Coordinator executes real compute_riemann — or injects tool_call error]

Turn 2  user   → Feed tool result back.
Turn 2  asst   → {"integral": ..., "interval": [...], "step": ..., "n": ..., "converged": bool}

Convergence: if |step_k - step_{k-1}| / |step_{k-1}| < 1e-4, mark converged and stop early.

Deferred detection
──────────────────
Each InjectionSpec may carry detect_step and detect_phase in addition to the
injection step.  When detection is deferred:

  detect_phase="workers", detect_step > inject_step
    Error is injected silently; an entry in errors[] is added only when the
    worker loop reaches detect_step (or the run ends, whichever comes first).

  detect_phase="aggregator"
    Worker never writes to errors[].  The coordinator adds the detection record
    to final_report after the aggregator phase completes.

  detect_phase="none"
    Error is never flagged anywhere — it propagates silently to the final answer.

All injected errors are always recorded in the "injections" list in the
returned dict, regardless of detect_phase, so the full injection history is
available for analysis even when errors[] is empty.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple

from .base import BaseAgent, OllamaResponse
from tools import compute_riemann
from error_injection import ErrorInjector
from error_injection.errors import ErrorType, InjectionSpec

log = logging.getLogger(__name__)

CONVERGE_TOL = 1e-4

SYSTEM_PROMPT = """You are a numerical integration worker agent.
You compute definite integrals using the midpoint Riemann sum method by requesting
a compute_riemann tool call, then reporting your result.

Always respond with ONLY a valid JSON object — no prose, no markdown fences.
"""

TOOL_REQUEST_TEMPLATE = (
    "Step {step} of {num_steps}.\n\n"
    "Compute the integral of f(x) = {poly_str} over [{a:.6f}, {b:.6f}] using n={n} rectangles.\n\n"
    "Respond with ONLY this JSON to request the computation:\n\n"
    '{{\n'
    '  "action": "compute_riemann",\n'
    '  "coefficients": {coefficients},\n'
    '  "a": {a},\n'
    '  "b": {b},\n'
    '  "n": {n}\n'
    '}}\n'
)

TOOL_RESULT_TEMPLATE = (
    "The compute_riemann tool returned:\n\n"
    "{tool_result}\n\n"
    "Now provide your final answer for this step as ONLY this JSON:\n\n"
    '{{\n'
    '  "integral": <value from tool result>,\n'
    '  "interval": [{a:.6f}, {b:.6f}],\n'
    '  "step": {step},\n'
    '  "n": {n},\n'
    '  "converged": false\n'
    '}}\n\n'
    "Copy the result value exactly as returned by the tool."
)

TOOL_RESULT_TEMPLATE_CONVERGED = (
    "The compute_riemann tool returned:\n\n"
    "{tool_result}\n\n"
    "The result has converged (change < 0.01% from previous step).\n"
    "Provide your final answer as ONLY this JSON:\n\n"
    '{{\n'
    '  "integral": <value from tool result>,\n'
    '  "interval": [{a:.6f}, {b:.6f}],\n'
    '  "step": {step},\n'
    '  "n": {n},\n'
    '  "converged": true\n'
    '}}\n'
)


def _poly_str(coefficients):
    terms = []
    for i, c in enumerate(coefficients):
        if c == 0:
            continue
        if i == 0:
            terms.append(str(c))
        elif i == 1:
            terms.append("{}·x".format(c))
        else:
            terms.append("{}·x^{}".format(c, i))
    return " + ".join(terms) if terms else "0"


# ---------------------------------------------------------------------------
# Pending-detection record: (spec, inject_step, error_label, injection_record)
# injection_record is mutated in-place when detection fires.
# ---------------------------------------------------------------------------
_PendingItem = Tuple[InjectionSpec, int, str, dict]


class WorkerAgent(BaseAgent):
    def __init__(self, worker_index, base_url, model, timeout, num_steps, injector,
                 extra_options=None, max_turns=0):
        # type: (int, str, str, int, int, ErrorInjector, Optional[dict], int) -> None
        super(WorkerAgent, self).__init__(
            "Worker[{}]".format(worker_index), model, base_url, timeout,
            extra_options=extra_options,
        )
        self.worker_index = worker_index
        self.num_steps = num_steps
        self.injector = injector
        self.max_turns = max_turns

    def _trim_history(self):
        """Drop oldest step messages, keeping system prompt + last max_turns steps.

        Each step contributes exactly 4 messages (user tool-req, asst tool-call,
        user tool-result, asst final-answer).  Trimming after push_assistant means
        the *next* chat() sends fewer messages than the previous one, which the
        ctx_untangler records as event_type='compression'.
        """
        if not self.max_turns:
            return
        keep_non_sys = self.max_turns * 4
        if len(self._messages) - 1 > keep_non_sys:
            dropped = len(self._messages) - 1 - keep_non_sys
            self._messages = self._messages[:1] + self._messages[-keep_non_sys:]
            log.info("[%s] Context compaction: dropped %d msgs, kept last %d turn(s)",
                     self.name, dropped, self.max_turns)

    # ── deferred-detection helpers ────────────────────────────────────────────

    def _report_now(self, spec, inject_step):
        # type: (InjectionSpec, int) -> bool
        """
        True when the error label should be appended to errors[] immediately
        (i.e. detection is not deferred to a later step or a later phase).
        """
        return (
            spec.detect_phase == "workers"
            and spec._effective_detect_step() <= inject_step
        )

    def _schedule_error(self, spec, inject_step, label, errors, injections, pending):
        # type: (InjectionSpec, int, str, list, list, list) -> dict
        """
        Create an injection record and either report the error immediately or
        park it in *pending* for deferred reporting.

        Returns the injection record dict (mutated in-place later if deferred).
        """
        inj = {
            "error_type": spec.error_type.value,
            "inject_step": inject_step,
            "detect_step": spec._effective_detect_step(),
            "detect_phase": spec.detect_phase,
            "detected": False,
            "detected_at_step": None,
        }
        injections.append(inj)

        if self._report_now(spec, inject_step):
            # Immediate (backward-compatible) path
            errors.append(label)
            inj["detected"] = True
            inj["detected_at_step"] = inject_step
            log.debug("[%s] step %d: error reported immediately (%s)",
                      self.name, inject_step, spec.detect_phase)
        elif spec.detect_phase == "workers":
            # Deferred within-worker
            pending.append((spec, inject_step, label, inj))
            log.info(
                "[%s] step %d: error deferred — will report at step %d (lag=%d)",
                self.name, inject_step,
                spec._effective_detect_step(),
                spec._effective_detect_step() - inject_step,
            )
        elif spec.detect_phase == "aggregator":
            log.info("[%s] step %d: error detection deferred to aggregator phase",
                     self.name, inject_step)
        else:  # "none"
            log.info("[%s] step %d: error will NOT be detected (detect_phase=none)",
                     self.name, inject_step)

        return inj

    def _flush_pending(self, current_step, errors, pending):
        # type: (int, list, list) -> list
        """
        Walk pending-detection list; fire any whose detect_step <= current_step.
        Returns the items that are still waiting.
        """
        still_pending = []  # type: list
        for pspec, pinject_step, plabel, pinj in pending:
            if pspec.is_detected_at_step(current_step):
                lag = current_step - pinject_step
                flagged = "{}(detected_at_step={},lag={})".format(
                    plabel, current_step, lag
                )
                errors.append(flagged)
                pinj["detected"] = True
                pinj["detected_at_step"] = current_step
                log.warning(
                    "[%s] DEFERRED DETECTION: %s injected at step %d, "
                    "detected at step %d (lag=%d steps)",
                    self.name, pspec.error_type, pinject_step, current_step, lag,
                )
            else:
                still_pending.append((pspec, pinject_step, plabel, pinj))
        return still_pending

    # ── main execution loop ───────────────────────────────────────────────────

    async def run(self, task):
        # type: (dict) -> dict
        """
        Execute all refinement steps for one sub-interval.

        Returns:
          worker_index, interval, steps, final_integral, converged, errors, injections
        """
        coefficients = task["coefficients"]
        a = float(task["a"])
        b = float(task["b"])
        initial_n = int(task.get("initial_n", 10))
        poly = _poly_str(coefficients)

        self.reset()
        self.push_system(SYSTEM_PROMPT)

        step_results = []   # type: List[OllamaResponse]
        prev_value = None   # type: Optional[float]
        converged = False
        errors = []         # type: List[str]   — populated when detection fires
        injections = []     # type: List[dict]  — always populated on injection
        pending = []        # type: list        — (spec, inject_step, label, inj_record)

        for step in range(1, self.num_steps + 1):
            n = initial_n * (2 ** (step - 1))
            self._trim_history()

            log.info("[%s] step %d/%d  n=%d  interval=[%.4f, %.4f]",
                     self.name, step, self.num_steps, n, a, b)

            # ── Check deferred detections from previous steps ────────────── #
            pending = self._flush_pending(step, errors, pending)

            spec = self.injector.find_spec("worker", self.worker_index, step)

            # ── Turn 1: ask agent to issue a tool-call request ───────────── #
            self.push_user(
                TOOL_REQUEST_TEMPLATE.format(
                    step=step,
                    num_steps=self.num_steps,
                    poly_str=poly,
                    a=a, b=b, n=n,
                    coefficients=json.dumps(coefficients),
                )
            )

            format_on_t1 = (
                spec is not None
                and spec.error_type == ErrorType.FORMAT
            )

            if format_on_t1:
                await self.chat()  # real call for proxy recording
                t1_response = self.synthetic_response(
                    self.injector.inject_format_response("worker", [a, b], step)
                )
                log.warning("[%s] step %d FORMAT error on tool-request turn.", self.name, step)
                label = "step{}:format_turn1".format(step)
                self._schedule_error(spec, step, label, errors, injections, pending)
            else:
                t1_response = await self.chat()

            self.push_assistant(t1_response.content)

            # ── Execute real tool (or inject tool_call error) ────────────── #
            tool_error_here = (
                spec is not None
                and spec.error_type == ErrorType.TOOL_CALL
            )

            if tool_error_here:
                tool_result_str = self.injector.inject_tool_error_body(spec)
                log.warning("[%s] step %d TOOL_CALL error injected.", self.name, step)
                label = "step{}:tool_call_error".format(step)
                self._schedule_error(spec, step, label, errors, injections, pending)
                real_value = None
            else:
                req = t1_response.parsed
                if req and req.get("action") == "compute_riemann":
                    try:
                        tool_out = compute_riemann(
                            coefficients=req.get("coefficients", coefficients),
                            a=float(req.get("a", a)),
                            b=float(req.get("b", b)),
                            n=int(req.get("n", n)),
                        )
                        real_value = tool_out["result"]
                        tool_result_str = json.dumps(tool_out)
                    except Exception as exc:
                        log.error("[%s] compute_riemann raised: %s", self.name, exc)
                        tool_result_str = json.dumps({"error": str(exc)})
                        real_value = None
                        errors.append("step{}:tool_exception:{}".format(step, exc))
                else:
                    # Agent didn't produce a proper tool request — run with defaults
                    log.warning(
                        "[%s] step %d: no valid tool request found; running tool with defaults.",
                        self.name, step,
                    )
                    tool_out = compute_riemann(coefficients=coefficients, a=a, b=b, n=n)
                    real_value = tool_out["result"]
                    tool_result_str = json.dumps(tool_out)

            # Convergence check
            if real_value is not None and prev_value is not None:
                denom = abs(prev_value) if abs(prev_value) > 1e-12 else 1e-12
                if abs(real_value - prev_value) / denom < CONVERGE_TOL:
                    converged = True

            # ── Turn 2: feed tool result back, ask for final answer ──────── #
            tmpl = TOOL_RESULT_TEMPLATE_CONVERGED if converged else TOOL_RESULT_TEMPLATE
            self.push_user(
                tmpl.format(
                    tool_result=tool_result_str,
                    a=a, b=b,
                    step=step,
                    n=n,
                )
            )

            logic_here = (
                spec is not None
                and spec.error_type == ErrorType.LOGIC
            )
            # Only inject format_turn2 if we didn't already inject format_turn1
            format_on_t2 = (
                spec is not None
                and spec.error_type == ErrorType.FORMAT
                and not format_on_t1
            )

            if format_on_t2:
                await self.chat()
                t2_response = self.synthetic_response(
                    self.injector.inject_format_response("worker", [a, b], step)
                )
                label = "step{}:format_turn2".format(step)
                self._schedule_error(spec, step, label, errors, injections, pending)
            elif logic_here:
                if real_value is None:
                    real_value = compute_riemann(
                        coefficients=coefficients, a=a, b=b, n=n
                    )["result"]
                await self.chat()
                t2_response = self.synthetic_response(
                    self.injector.inject_logic_response(
                        correct_result=real_value,
                        interval=[a, b],
                        step=step,
                        n=n,
                        scale_factor=spec.scale_factor,
                    )
                )
                label = "step{}:logic_error:scale={}".format(step, spec.scale_factor)
                self._schedule_error(spec, step, label, errors, injections, pending)
            else:
                t2_response = await self.chat()

            self.push_assistant(t2_response.content)
            step_results.append(t2_response)

            if real_value is not None:
                prev_value = real_value

            if converged:
                log.info("[%s] Converged at step %d.", self.name, step)
                break

        # ── Drain any remaining pending detections ───────────────────────── #
        # detect_step was beyond the last step actually executed.
        # Fire them all at the final step so they appear in the output.
        final_step = min(self.num_steps, step if converged else self.num_steps)
        for pspec, pinject_step, plabel, pinj in pending:
            if pspec.detect_phase == "workers":
                # Partial detection: fires at end of run, lag reflects truncation
                actual_detect = final_step
                lag = actual_detect - pinject_step
                flagged = "{}(detected_at_run_end,step={},lag={})".format(
                    plabel, actual_detect, lag,
                )
                errors.append(flagged)
                pinj["detected"] = True
                pinj["detected_at_step"] = actual_detect
                log.warning(
                    "[%s] DEFERRED DETECTION (run ended): %s injected at step %d, "
                    "detect_step=%d unreachable — reported at final step %d (lag=%d)",
                    self.name, pspec.error_type, pinject_step,
                    pspec._effective_detect_step(), actual_detect, lag,
                )

        # Extract final integral value from the last successful step result
        final_integral = None  # type: Optional[float]
        for r in reversed(step_results):
            if r.parsed and "integral" in r.parsed:
                final_integral = float(r.parsed["integral"])
                break

        return {
            "worker_index": self.worker_index,
            "interval": [a, b],
            "steps": step_results,
            "final_integral": final_integral,
            "converged": converged,
            "errors": errors,
            "injections": injections,
        }
