"""
WorkflowCoordinator — orchestrates the full integration workflow.

Execution order
───────────────
1. PlannerAgent   — partitions [a, b] into num_agents sub-intervals
2. WorkerAgents   — run concurrently (asyncio.gather), one per sub-interval
3. AggregatorAgent — sums results and produces a final report

After Phase 3, the coordinator applies any aggregator-phase detections:
for each InjectionSpec with detect_phase="aggregator", the coordinator
adds a record to final_report["aggregator_detected_errors"] and logs the
detection event.

CTE session IDs (when proxy enabled):
  {workflow_id}_planner_0
  {workflow_id}_worker_0 … {workflow_id}_worker_{N-1}
  {workflow_id}_aggregator_0
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional

from .agents import PlannerAgent, WorkerAgent, AggregatorAgent
from cte.session import CTESession
from error_injection import ErrorInjector, InjectionSpec
from tools.riemann import _poly_integral

log = logging.getLogger(__name__)


class WorkflowResult:
    def __init__(self, workflow_id, problem, plan, worker_results,
                 final_report, injected_errors, elapsed_sec, exact_integral,
                 phase_timing=None):
        self.workflow_id = workflow_id
        self.problem = problem
        self.plan = plan
        self.worker_results = worker_results
        self.final_report = final_report
        self.injected_errors = injected_errors
        self.elapsed_sec = elapsed_sec
        self.exact_integral = exact_integral
        # {"planner_s": float, "workers_s": float, "aggregator_s": float}
        self.phase_timing = phase_timing or {}


class WorkflowCoordinator:
    def __init__(self, cfg):
        # type: (dict) -> None
        self.cfg = cfg
        self.workflow_cfg = cfg["workflow"]
        self.problem = cfg["problem"]
        self.ollama_cfg = cfg["ollama"]
        self.cte_session = CTESession(cfg.get("cte_proxy", {}))

        injection_dicts = cfg.get("error_injections", []) or []
        self.injector = ErrorInjector(
            [InjectionSpec.from_dict(d) for d in injection_dicts]
        )

        self.workflow_id = self.workflow_cfg["id"]
        self.num_agents = int(self.workflow_cfg["num_agents"])
        self.num_steps = int(self.workflow_cfg["num_steps"])

        num_ctx = self.ollama_cfg.get("num_ctx")
        self._extra_options = {"num_ctx": int(num_ctx)} if num_ctx else {}
        self._max_turns = int(self.workflow_cfg.get("max_turns", 0))
        # Multi-GPU: list of Ollama base URLs; workers distributed round-robin.
        # Planner and aggregator always use urls[0].
        self._ollama_urls = self.ollama_cfg.get("urls") or []

    async def run(self):
        # type: () -> WorkflowResult
        t0 = time.time()
        log.info("=== Workflow %s starting  agents=%d  steps=%d ===",
                 self.workflow_id, self.num_agents, self.num_steps)

        coefficients = self.problem["function"]["coefficients"]
        a, b = self.problem["interval"]
        exact = _poly_integral(coefficients, a, b)

        # Phase 1: Planner
        plan_response = await self._run_planner()
        t1 = time.time()
        plan = plan_response.parsed

        if plan is None or "tasks" not in plan:
            log.warning("Planner returned invalid/no JSON — falling back to arithmetic partition.")
            plan = self._arithmetic_partition()

        tasks = plan["tasks"]
        if len(tasks) != self.num_agents:
            log.warning(
                "Planner returned %d tasks (expected %d) — re-partitioning.",
                len(tasks), self.num_agents,
            )
            plan = self._arithmetic_partition()
            tasks = plan["tasks"]

        log.info("Plan: %d tasks  intervals=%s",
                 len(tasks),
                 [(round(t["a"], 3), round(t["b"], 3)) for t in tasks])

        # Phase 2: Workers (concurrent)
        worker_results = await self._run_workers(tasks)
        t2 = time.time()

        # Phase 3: Aggregator
        agg_response = await self._run_aggregator(worker_results)
        final_report = agg_response.parsed

        # ── Aggregator-phase deferred detections ──────────────────────────── #
        agg_detections = self._compute_aggregator_detections(worker_results)
        if agg_detections:
            log.warning(
                "[Coordinator] %d aggregator-phase detection(s) recorded: %s",
                len(agg_detections),
                ["{error_type}@worker[{worker_index}](injected_step={inject_step})".format(**d)
                 for d in agg_detections],
            )
            if final_report is None:
                final_report = {}
            final_report["aggregator_detected_errors"] = agg_detections

        # ── Ground-truth absolute_error (overwrite LLM self-report) ──────── #
        # Overwrite the LLM's self-reported absolute_error with an independently
        # computed value so the metric is ground-truth regardless of what the
        # aggregator wrote.  If total_integral is missing (aggregator returned no
        # valid JSON or omitted the field), absolute_error is set to None so
        # downstream code can distinguish total failure from a large numeric error.
        if final_report is not None:
            computed = final_report.get("total_integral")
            if computed is not None:
                verified = round(abs(computed - exact), 8)
                llm_reported = final_report.get("absolute_error")
                if llm_reported is not None and abs(llm_reported - verified) > 1e-6:
                    log.warning(
                        "[Aggregator] absolute_error mismatch: LLM reported %.6g, "
                        "computed %.6g — using computed value.",
                        llm_reported, verified,
                    )
                final_report["absolute_error"] = verified
            else:
                final_report["absolute_error"] = None

        elapsed = time.time() - t0
        phase_timing = {
            "planner_s":    round(t1 - t0, 3),
            "workers_s":    round(t2 - t1, 3),
            "aggregator_s": round(elapsed - (t2 - t0), 3),
        }

        # Collect all errors that were actually reported (errors[] entries)
        all_errors = []
        for wr in worker_results:
            for e in wr.get("errors", []):
                all_errors.append("worker[{}]:{}".format(wr["worker_index"], e))
        # Also surface aggregator-phase detections in the top-level list
        for d in agg_detections:
            all_errors.append(
                "worker[{}]:{}(detected_at_aggregator,injected_step={})".format(
                    d["worker_index"], d["error_type"], d["inject_step"]
                )
            )

        log.info("=== Workflow complete  elapsed=%.1fs  exact=%.6f ===", elapsed, exact)

        return WorkflowResult(
            workflow_id=self.workflow_id,
            problem=self.problem,
            plan=plan,
            worker_results=worker_results,
            final_report=final_report,
            injected_errors=all_errors,
            elapsed_sec=elapsed,
            exact_integral=exact,
            phase_timing=phase_timing,
        )

    # ── Aggregator-phase detection helper ────────────────────────────────────

    def _compute_aggregator_detections(self, worker_results):
        # type: (list) -> list
        """
        For every InjectionSpec with detect_phase="aggregator" that targets a
        worker, produce a detection record.  These are added to final_report
        AFTER the aggregator LLM call so the LLM output is unaffected.
        """
        detections = []
        for spec in self.injector.specs:
            if not spec.is_detected_at_phase("aggregator"):
                continue
            if spec.agent_type != "worker":
                continue
            for wr in worker_results:
                if spec.agent_index is not None and spec.agent_index != wr["worker_index"]:
                    continue
                detections.append({
                    "worker_index": wr["worker_index"],
                    "error_type": spec.error_type.value,
                    "inject_step": spec._effective_inject_step(),
                    "detect_phase": "aggregator",
                })
        return detections

    # ── Agent constructors ────────────────────────────────────────────────────

    def _make_base_url(self, session_id, worker_index=None):
        # type: (str, int) -> str
        # Multi-GPU direct mode: bypass proxy, assign workers round-robin across URLs.
        if self._ollama_urls and not self.cte_session.enabled:
            slot = 0 if worker_index is None else (worker_index % len(self._ollama_urls))
            url = self._ollama_urls[slot]
            log.debug("Multi-GPU routing: %s → %s (slot %d)", session_id, url, slot)
            return url
        return self.cte_session.base_url(
            session_id,
            self.ollama_cfg["host"],
            int(self.ollama_cfg["port"]),
        )

    async def _run_planner(self):
        session_id = self.cte_session.session_id(self.workflow_id, "planner", 0)
        base_url = self._make_base_url(session_id)
        planner = PlannerAgent(
            base_url=base_url,
            model=self.ollama_cfg["model"],
            timeout=int(self.ollama_cfg.get("timeout", 90)),
            injector=self.injector,
            extra_options=self._extra_options,
        )
        log.info("[Planner] session=%s  base_url=%s", session_id, base_url)
        return await planner.run(self.problem, self.num_agents)

    async def _run_workers(self, tasks):
        # type: (list) -> list
        coros = []
        for task in tasks:
            idx = task["worker_index"]
            session_id = self.cte_session.session_id(self.workflow_id, "worker", idx)
            base_url = self._make_base_url(session_id, worker_index=idx)
            agent = WorkerAgent(
                worker_index=idx,
                base_url=base_url,
                model=self.ollama_cfg["model"],
                timeout=int(self.ollama_cfg.get("timeout", 90)),
                num_steps=self.num_steps,
                injector=self.injector,
                extra_options=self._extra_options,
                max_turns=self._max_turns,
            )
            log.info("[Worker %d] session=%s  interval=[%.4f, %.4f]",
                     idx, session_id, task["a"], task["b"])
            coros.append(agent.run(task))

        results = await asyncio.gather(*coros, return_exceptions=True)

        worker_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                log.error("[Worker %d] raised exception: %s", i, r)
                worker_results.append({
                    "worker_index": i,
                    "interval": [tasks[i]["a"], tasks[i]["b"]],
                    "steps": [],
                    "final_integral": None,
                    "converged": False,
                    "errors": ["exception:{}".format(r)],
                    "injections": [],
                })
            else:
                worker_results.append(r)

        return sorted(worker_results, key=lambda x: x["worker_index"])

    async def _run_aggregator(self, worker_results):
        session_id = self.cte_session.session_id(self.workflow_id, "aggregator", 0)
        base_url = self._make_base_url(session_id)
        agg = AggregatorAgent(
            base_url=base_url,
            model=self.ollama_cfg["model"],
            timeout=int(self.ollama_cfg.get("timeout", 90)),
            injector=self.injector,
            extra_options=self._extra_options,
        )
        log.info("[Aggregator] session=%s", session_id)
        return await agg.run(worker_results, self.problem)

    def _arithmetic_partition(self):
        # type: () -> dict
        """Fallback partitioner when Planner fails."""
        coefficients = self.problem["function"]["coefficients"]
        a, b = self.problem["interval"]
        initial_n = self.problem.get("initial_n", 10)
        width = (b - a) / self.num_agents
        tasks = []
        for i in range(self.num_agents):
            tasks.append({
                "worker_index": i,
                "a": round(a + i * width, 10),
                "b": round(a + (i + 1) * width, 10),
                "coefficients": coefficients,
                "initial_n": initial_n,
            })
        return {
            "tasks": tasks,
            "total_interval": [a, b],
            "num_workers": self.num_agents,
        }
