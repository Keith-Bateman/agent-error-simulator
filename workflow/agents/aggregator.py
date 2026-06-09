"""
AggregatorAgent — sums worker results and produces a final report.

Expected output JSON:
{
  "total_integral": <float>,
  "exact_integral": <float>,
  "absolute_error": <float>,
  "worker_count": <int>,
  "workers_with_errors": [<indices>],
  "converged_workers": <int>,
  "summary": "<brief text>"
}
"""

import json
import logging
from typing import List, Optional, Dict

from .base import BaseAgent, OllamaResponse
from tools.riemann import _poly_integral
from error_injection import ErrorInjector
from error_injection.errors import ErrorType

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a mathematical aggregator agent.
Your job is to sum numerical integration results from multiple worker agents
and produce a concise final report.

Always respond with ONLY a valid JSON object — no prose, no markdown fences.
"""

USER_TEMPLATE = (
    "You received the following integration results from {num_workers} worker agents:\n\n"
    "{worker_results_json}\n\n"
    "The exact analytical integral of f(x) = {poly_str} over [{a}, {b}] is {exact:.8f}.\n\n"
    "Sum all 'final_integral' values. Note which workers reported errors.\n"
    "Respond with ONLY this JSON:\n\n"
    '{{\n'
    '  "total_integral": <sum of final_integral values>,\n'
    '  "exact_integral": {exact:.8f},\n'
    '  "absolute_error": <|total_integral - exact_integral|>,\n'
    '  "worker_count": {num_workers},\n'
    '  "workers_with_errors": [<list of worker_index values with non-empty errors>],\n'
    '  "converged_workers": <count of workers where converged is true>,\n'
    '  "summary": "<one sentence>"\n'
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
            terms.append("{}\u00b7x".format(c))
        else:
            terms.append("{}\u00b7x^{}".format(c, i))
    return " + ".join(terms) if terms else "0"


class AggregatorAgent(BaseAgent):
    def __init__(self, base_url, model, timeout, injector, extra_options=None):
        # type: (str, str, int, ErrorInjector, Optional[Dict]) -> None
        super(AggregatorAgent, self).__init__("Aggregator", model, base_url, timeout, extra_options=extra_options)
        self.injector = injector

    async def run(self, worker_results, problem):
        # type: (list, dict) -> OllamaResponse
        coefficients = problem["function"]["coefficients"]
        a, b = problem["interval"]
        exact = _poly_integral(coefficients, a, b)
        poly = _poly_str(coefficients)

        worker_summaries = [
            {
                "worker_index": wr["worker_index"],
                "interval": wr["interval"],
                "final_integral": wr["final_integral"],
                "converged": wr["converged"],
                "errors": wr["errors"],
            }
            for wr in worker_results
        ]

        spec = self.injector.find_spec("aggregator", 0, 1)

        partial_sum = sum(
            w["final_integral"] for w in worker_results
            if w["final_integral"] is not None
        )

        self.reset()
        self.push_system(SYSTEM_PROMPT)
        self.push_user(
            USER_TEMPLATE.format(
                num_workers=len(worker_results),
                worker_results_json=json.dumps(worker_summaries, indent=2),
                poly_str=poly,
                a=a, b=b,
                exact=exact,
            )
        )

        if spec is not None:
            log.warning("[Aggregator] Error injection active: %s", spec.error_type)
            await self.chat()
            if spec.error_type == ErrorType.FORMAT:
                fake_content = self.injector.inject_aggregator_format(partial_sum)
            elif spec.error_type == ErrorType.LOGIC:
                fake_content = json.dumps({
                    "total_integral": round(partial_sum * spec.scale_factor, 8),
                    "exact_integral": round(exact, 8),
                    "absolute_error": round(
                        abs(partial_sum * spec.scale_factor - exact), 8
                    ),
                    "worker_count": len(worker_results),
                    "workers_with_errors": [],
                    "converged_workers": sum(
                        1 for w in worker_results if w["converged"]
                    ),
                    "summary": "Aggregation complete.",
                })
            else:
                fake_content = self.injector.inject_aggregator_format(partial_sum)
            response = self.synthetic_response(fake_content)
        else:
            response = await self.chat()

        self.push_assistant(response.content)

        if response.parsed is None:
            log.error("[Aggregator] Response is not valid JSON: %r",
                      response.content[:200])
        else:
            log.info(
                "[Aggregator] total=%.6f  exact=%.6f  |err|=%.6f",
                response.parsed.get("total_integral", float("nan")),
                exact,
                response.parsed.get("absolute_error", float("nan")),
            )

        return response
