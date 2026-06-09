#!/usr/bin/env python3.11
"""
agent-error-generator — entry point

Usage
─────
  python generator.py [options]

Options
  --config PATH          Config file (default: config.yaml)
  --num-agents N         Override workflow.num_agents
  --num-steps  N         Override workflow.num_steps
  --model MODEL          Override ollama.model
  --inject SPEC          Add an error injection (can be repeated).
                         Format: TYPE:ROLE[:IDX[:INJECT_STEP[:DETECT_STEP[:DETECT_PHASE]]]]
                           TYPE         : format | logic | tool_call
                           ROLE         : planner | worker | aggregator
                           IDX          : agent index (0-based, optional)
                           INJECT_STEP  : refinement step where error fires (1-based, optional)
                           DETECT_STEP  : step where error is first reported (1-based, optional)
                                          defaults to INJECT_STEP (immediate, backward-compat)
                           DETECT_PHASE : workers | aggregator | none (optional)
                                          workers    — worker self-reports at detect_step
                                          aggregator — coordinator reports after agg phase
                                          none       — error never flagged (silent)
                         Examples (backward-compatible):
                           --inject format:worker:1:2
                           --inject tool_call:worker:0
                           --inject logic:planner
                         Examples (deferred detection):
                           --inject logic:worker:0:1:3         (inject step 1, detect step 3)
                           --inject logic:worker:1:1::aggregator  (aggregator-phase detection)
                           --inject format:worker:2:2::none    (silent — never detected)
                           --inject tool_call:worker:0:1:4:workers  (explicit 3-step lag)
  --proxy-host HOST      Override cte_proxy.host
  --proxy-port PORT      Override cte_proxy.port
  --enable-proxy         Set cte_proxy.enabled = true
  --workflow-id ID       Override workflow.id
  --log-level LEVEL      DEBUG | INFO | WARNING (default: INFO)
  --output PATH          Write final result JSON to this file
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml

from workflow import WorkflowCoordinator


def parse_inject(spec_str):
    """Parse --inject TYPE:ROLE[:IDX[:INJECT_STEP[:DETECT_STEP[:DETECT_PHASE]]]] into a dict.

    Backward-compatible: the old TYPE:ROLE[:IDX[:STEP]] form still works because
    STEP maps to inject_step with immediate detection (detect_step and detect_phase
    default to inject_step / "workers" respectively).
    """
    parts = spec_str.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            "--inject requires at least TYPE:ROLE, got: {!r}".format(spec_str)
        )
    d = {
        "error_type": parts[0].strip(),
        "agent_type": parts[1].strip(),
    }
    if len(parts) >= 3 and parts[2].strip():
        d["agent_index"] = int(parts[2])
    # Position 4: inject_step (replaces old "step", stored as inject_step)
    if len(parts) >= 4 and parts[3].strip():
        d["inject_step"] = int(parts[3])
    # Position 5: detect_step (optional; blank → defaults to inject_step)
    if len(parts) >= 5 and parts[4].strip():
        detect_step_raw = parts[4].strip()
        try:
            d["detect_step"] = int(detect_step_raw)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "--inject DETECT_STEP must be an integer, got: {!r}".format(detect_step_raw)
            )
    # Position 6: detect_phase (optional; blank → "workers")
    if len(parts) >= 6 and parts[5].strip():
        phase = parts[5].strip().lower()
        if phase not in ("workers", "aggregator", "none"):
            raise argparse.ArgumentTypeError(
                "--inject DETECT_PHASE must be workers|aggregator|none, got: {!r}".format(phase)
            )
        d["detect_phase"] = phase
    return d


def build_parser():
    p = argparse.ArgumentParser(
        description="Multi-agent numerical integration workflow with error injection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="config.yaml", metavar="PATH")
    p.add_argument("--num-agents", type=int, metavar="N")
    p.add_argument("--num-steps",  type=int, metavar="N")
    p.add_argument("--model",      metavar="MODEL")
    p.add_argument("--inject", action="append", default=[],
                   metavar="SPEC", type=parse_inject)
    p.add_argument("--proxy-host", metavar="HOST")
    p.add_argument("--proxy-port", type=int, metavar="PORT")
    p.add_argument("--enable-proxy", action="store_true")
    p.add_argument("--workflow-id", metavar="ID")
    p.add_argument("--max-turns", type=int, metavar="N",
                   help="Keep only the last N refinement steps in each worker's "
                        "message history.  Older turns are dropped after each step, "
                        "producing explicit context-compaction events visible in the CEE.")
    p.add_argument("--num-ctx", type=int, metavar="N",
                   help="Ollama num_ctx (context window tokens). Use small values "
                        "(e.g. 2048) to force context exhaustion for CEE compaction tests.")
    p.add_argument("--ollama-urls", metavar="URLS",
                   help="Comma-separated Ollama base URLs for multi-GPU mode, e.g. "
                        "http://127.0.0.1:11434,http://127.0.0.1:11934  "
                        "Workers are distributed round-robin; planner/aggregator use urls[0]. "
                        "Overrides --proxy-host/--proxy-port and disables CEE routing.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--output", metavar="PATH",
                   help="Write final result JSON to this file")
    return p


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, args):
    if args.num_agents is not None:
        cfg["workflow"]["num_agents"] = args.num_agents
    if args.num_steps is not None:
        cfg["workflow"]["num_steps"] = args.num_steps
    if args.model is not None:
        cfg["ollama"]["model"] = args.model
    if args.num_ctx is not None:
        cfg["ollama"]["num_ctx"] = args.num_ctx
    if args.max_turns is not None:
        cfg["workflow"]["max_turns"] = args.max_turns
    if args.workflow_id is not None:
        cfg["workflow"]["id"] = args.workflow_id

    proxy = cfg.setdefault("cte_proxy", {})
    if args.enable_proxy:
        proxy["enabled"] = True
    if args.proxy_host:
        proxy["host"] = args.proxy_host
    if args.proxy_port:
        proxy["port"] = args.proxy_port

    if getattr(args, "ollama_urls", None):
        cfg["ollama"]["urls"] = [u.strip() for u in args.ollama_urls.split(",") if u.strip()]

    existing = cfg.get("error_injections") or []
    cfg["error_injections"] = existing + (args.inject or [])
    return cfg


def print_result(result):
    print("\n" + "=" * 60)
    print("  Workflow : {}".format(result.workflow_id))
    print("  Elapsed  : {:.1f}s".format(result.elapsed_sec))
    print("  Exact    : {:.8f}".format(result.exact_integral))
    if result.final_report:
        r = result.final_report
        total = r.get("total_integral")
        err   = r.get("absolute_error")
        print("  Computed : {}".format(total))
        print("  |Error|  : {}".format(err))
        conv = r.get("converged_workers", "?")
        nw   = r.get("worker_count", "?")
        print("  Converged: {}/{}".format(conv, nw))
        if r.get("workers_with_errors"):
            print("  Errors @ workers: {}".format(r["workers_with_errors"]))
        print("  Summary  : {}".format(r.get("summary", "")))
    else:
        print("  [Aggregator returned no valid JSON]")

    if result.injected_errors:
        print("\n  Injected errors:")
        for e in result.injected_errors:
            print("    * {}".format(e))
    print("=" * 60 + "\n")


async def main(args):
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        logging.error("Config file not found: %s", config_path)
        return 1

    cfg = load_config(str(config_path))
    cfg = apply_overrides(cfg, args)

    logging.info("Config loaded: %s", config_path)
    logging.info("  num_agents=%d  num_steps=%d  model=%s",
                 cfg["workflow"]["num_agents"],
                 cfg["workflow"]["num_steps"],
                 cfg["ollama"]["model"])
    logging.info("  cte_proxy.enabled=%s",
                 cfg.get("cte_proxy", {}).get("enabled", False))
    logging.info("  error_injections=%d",
                 len(cfg.get("error_injections") or []))

    coordinator = WorkflowCoordinator(cfg)

    try:
        result = await coordinator.run()
    except Exception as exc:
        logging.exception("Workflow failed: %s", exc)
        return 2

    print_result(result)

    if args.output:
        out_path = Path(args.output)
        out_data = {
            "workflow_id": result.workflow_id,
            "elapsed_sec": result.elapsed_sec,
            "phase_timing": result.phase_timing,
            "exact_integral": result.exact_integral,
            "final_report": result.final_report,
            "injected_errors": result.injected_errors,
            "worker_results": [
                {
                    "worker_index": wr["worker_index"],
                    "interval": wr["interval"],
                    "final_integral": wr["final_integral"],
                    "converged": wr["converged"],
                    "errors": wr["errors"],
                    # injections: always present (empty list for clean workers).
                    # Each entry records error_type, inject_step, detect_step,
                    # detect_phase, detected (bool), detected_at_step (int|null).
                    "injections": wr.get("injections", []),
                    "steps": [
                        {"content": s.content, "injected": s.injected}
                        for s in wr.get("steps", [])
                    ],
                }
                for wr in result.worker_results
            ],
        }
        out_path.write_text(json.dumps(out_data, indent=2))
        logging.info("Result written to %s", out_path)

    return 0


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
