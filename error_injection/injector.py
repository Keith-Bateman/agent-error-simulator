"""
ErrorInjector — checks injection specs and produces synthetic agent responses.

Three injection modes
─────────────────────
format    The agent's "assistant" turn is replaced with plain natural-language
          text.  Any downstream parser that expects JSON will fail.

logic     The agent's JSON response contains an arithmetically wrong result.
          By default the integral value is multiplied by scale_factor (-1).

tool_call The coordinator's synthetic tool-result message reports an HTTP-style
          error instead of a real result.  The agent then produces a plausible
          but fabricated JSON fallback.
"""

import json
import logging
from typing import List, Optional

from .errors import ErrorType, InjectionSpec

log = logging.getLogger(__name__)


class ErrorInjector:
    def __init__(self, specs):
        # type: (List[InjectionSpec]) -> None
        self.specs = specs

    def find_spec(self, agent_type, agent_index, step):
        # type: (str, int, int) -> Optional[InjectionSpec]
        for spec in self.specs:
            if spec.matches(agent_type, agent_index, step):
                return spec
        return None

    def inject_tool_error_body(self, spec):
        # type: (InjectionSpec) -> str
        """
        Return a synthetic tool-result string that looks like an HTTP error.
        This replaces the real compute_riemann output fed back to the agent.
        """
        body = {
            "error": spec.tool_status,
            "message": spec.tool_message,
        }
        injected = json.dumps(body)
        log.warning("[INJECT tool_call] Replacing tool result with error body: %s", injected)
        return injected

    def inject_format_response(self, agent_type, interval=None, step=1):
        # type: (str, Optional[list], int) -> str
        """
        Return a natural-language string that looks like what a careless agent
        might output instead of the required JSON.
        """
        if interval:
            a, b = interval
            text = (
                "I computed the integral over [{:.4f}, {:.4f}] on step {} "
                "and it came out to roughly 2.5 or so.".format(a, b, step)
            )
        else:
            text = (
                "I finished the {} step and everything looks good, "
                "I'll proceed from here.".format(agent_type)
            )
        log.warning("[INJECT format] Replacing JSON response with: %s", text)
        return text

    def inject_logic_response(self, correct_result, interval, step, n, scale_factor):
        # type: (float, list, int, int, float) -> str
        """
        Return a valid-looking JSON response with a deliberately wrong result.
        """
        wrong = round(correct_result * scale_factor, 10)
        payload = {
            "integral": wrong,
            "interval": interval,
            "step": step,
            "n": n,
            "converged": False,
        }
        injected = json.dumps(payload)
        log.warning(
            "[INJECT logic] Replacing correct result %.6f with %.6f (x%.2f): %s",
            correct_result, wrong, scale_factor, injected,
        )
        return injected

    def inject_planner_format(self, num_agents, interval):
        # type: (int, list) -> str
        """Plain-text format error for the Planner agent."""
        a, b = interval
        text = (
            "The interval [{}, {}] should be split into {} parts, "
            "each worker takes a chunk.".format(a, b, num_agents)
        )
        log.warning("[INJECT format/planner] Replacing JSON plan with: %s", text)
        return text

    def inject_aggregator_format(self, partial_sum):
        # type: (float) -> str
        text = (
            "After looking at all the worker results I think the total is "
            "somewhere around {:.3f}, give or take.".format(partial_sum)
        )
        log.warning("[INJECT format/aggregator] Replacing JSON summary with: %s", text)
        return text
