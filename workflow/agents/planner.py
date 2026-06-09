"""
PlannerAgent — decomposes the integration problem into N sub-tasks.

Expected output JSON:
{
  "tasks": [
    {"worker_index": 0, "a": 0.0, "b": 1.0, "coefficients": [...], "initial_n": 10},
    ...
  ],
  "total_interval": [0.0, 3.0],
  "num_workers": 3
}
"""

import json
import logging
from typing import Optional

from typing import Optional
from .base import BaseAgent, OllamaResponse
from error_injection import ErrorInjector, InjectionSpec

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a mathematical planning agent.
Your job is to partition a definite integration problem into equal sub-intervals
so that multiple worker agents can compute each sub-interval in parallel.

You must respond with ONLY a valid JSON object — no prose, no markdown fences.
"""

USER_TEMPLATE = (
    "Partition the definite integral of f(x) = {poly_str} over [{a}, {b}]\n"
    "into exactly {num_workers} equal sub-intervals.\n\n"
    "For each sub-interval produce a task entry. Respond with this exact JSON structure:\n\n"
    '{{\n'
    '  "tasks": [\n'
    '    {{\n'
    '      "worker_index": 0,\n'
    '      "a": <left endpoint>,\n'
    '      "b": <right endpoint>,\n'
    '      "coefficients": {coefficients},\n'
    '      "initial_n": {initial_n}\n'
    '    }},\n'
    '    ...\n'
    '  ],\n'
    '  "total_interval": [{a}, {b}],\n'
    '  "num_workers": {num_workers}\n'
    '}}\n\n'
    "Compute sub-interval endpoints precisely (evenly spaced)."
)


def _poly_str(coefficients):
    terms = []
    for i, c in enumerate(coefficients):
        if c == 0:
            continue
        if i == 0:
            terms.append(str(c))
        elif i == 1:
            terms.append("{}\u00b7x".format(c))
        else:
            terms.append("{}\u00b7x^{}".format(c, i))
    return " + ".join(terms) if terms else "0"


class PlannerAgent(BaseAgent):
    def __init__(self, base_url, model, timeout, injector, extra_options=None):
        # type: (str, str, int, ErrorInjector, Optional[dict]) -> None
        super(PlannerAgent, self).__init__("Planner", model, base_url, timeout, extra_options=extra_options)
        self.injector = injector

    async def run(self, problem, num_workers):
        # type: (dict, int) -> OllamaResponse
        coefficients = problem["function"]["coefficients"]
        a, b = problem["interval"]
        initial_n = problem.get("initial_n", 10)

        spec = self.injector.find_spec("planner", 0, 1)

        self.reset()
        self.push_system(SYSTEM_PROMPT)
        self.push_user(
            USER_TEMPLATE.format(
                poly_str=_poly_str(coefficients),
                a=a, b=b,
                num_workers=num_workers,
                coefficients=json.dumps(coefficients),
                initial_n=initial_n,
            )
        )

        if spec is not None:
            log.warning("[Planner] Error injection active: %s", spec.error_type)
            await self.chat()  # real call (logged by proxy if enabled)
            fake_content = self.injector.inject_planner_format(num_workers, [a, b])
            response = self.synthetic_response(fake_content)
        else:
            response = await self.chat()

        if response.parsed is None:
            log.error("[Planner] Failed to parse response as JSON: %r",
                      response.content[:200])
        else:
            log.info("[Planner] Produced %d tasks.",
                     len(response.parsed.get("tasks", [])))

        self.push_assistant(response.content)
        return response
