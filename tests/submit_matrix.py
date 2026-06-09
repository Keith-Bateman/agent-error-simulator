#!/usr/bin/env python3
"""
tests/submit_matrix.py
──────────────────────
Submit SLURM jobs defined in test_matrix.yaml.

Usage
─────
  python tests/submit_matrix.py [GROUP_ID ...] [options]

  python tests/submit_matrix.py              # submit all groups
  python tests/submit_matrix.py A B K        # submit specific groups
  python tests/submit_matrix.py --list       # print job table, don't submit
  python tests/submit_matrix.py --dry-run    # print sbatch commands, don't submit
  python tests/submit_matrix.py K --dry-run  # dry-run just group K

Options
  GROUP_ID        One or more group IDs (A, B, D, …, K).  Omit for all groups.
  --list          Print a compact table of matching jobs and exit.
  --dry-run       Show what would be submitted without running sbatch.
  --matrix PATH   Path to test matrix YAML (default: tests/test_matrix.yaml).

Worker index auto-assignment
────────────────────────────
Worker inject specs without an explicit agent_index are assigned indices
0, 1, 2, … in the order they appear in the inject list, modulo num_agents.
This produces consistent, agent-agnostic behaviour: single-inject scenarios
always target worker 0; multi-inject scenarios distribute across workers.

Inject string format produced
──────────────────────────────
  TYPE:ROLE[:IDX[:INJECT_STEP[:DETECT_STEP[:DETECT_PHASE]]]]

Examples:
  format:worker:0:1                  (immediate detection, worker 0, step 1)
  logic:worker:0:1:3:workers         (deferred: inject step 1, detect step 3)
  tool_call:worker:1:1::aggregator   (aggregator-phase detection)
  format:planner                     (planner, no index)
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML not installed.  Run:  pip install pyyaml")

SCRIPT_DIR = Path(__file__).parent.resolve()
AEG_DIR = SCRIPT_DIR.parent


# ── YAML loading ──────────────────────────────────────────────────────────────

def load_matrix(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Override resolution ───────────────────────────────────────────────────────

def resolve(key, scenario, group, defaults, fallback=None):
    """Return value of *key* using priority: scenario > group > defaults > fallback."""
    for source in (scenario, group, defaults):
        if source and key in source:
            return source[key]
    return fallback


# ── Inject-arg building ───────────────────────────────────────────────────────

def build_inject_args(inject_specs, num_agents):
    """
    Convert a list of inject-spec dicts into ``--inject …`` argument strings.

    Worker specs without an explicit ``agent_index`` key are assigned indices
    0, 1, 2, … (mod *num_agents*) in declaration order.

    Planner and aggregator specs never receive an index.

    Returns a list of strings such as ``['--inject logic:worker:0:1:3:workers']``.
    """
    if not inject_specs:
        return []

    worker_counter = 0
    args = []

    for raw in inject_specs:
        spec = dict(raw)
        agent_type = spec["agent_type"]
        error_type = spec["error_type"]

        # Auto-assign worker index when not explicitly provided
        if agent_type == "worker" and "agent_index" not in spec:
            spec["agent_index"] = worker_counter % num_agents
            worker_counter += 1

        inject_step  = spec.get("inject_step")
        detect_step  = spec.get("detect_step")
        detect_phase = spec.get("detect_phase")
        agent_index  = spec.get("agent_index")   # None for null / not set

        has_index = agent_type == "worker" and agent_index is not None

        # Work out which trailing fields we need to emit
        need_detect      = (detect_step is not None) or (detect_phase is not None)
        need_inject_step = (inject_step is not None) or need_detect
        # Workers need an index field to make room for inject_step in the format
        need_index = has_index or (agent_type == "worker" and need_inject_step)

        parts = [error_type, agent_type]
        if need_index:
            parts.append(str(agent_index) if has_index else "")
        if need_inject_step:
            parts.append(str(inject_step) if inject_step is not None else "1")
        if need_detect:
            parts.append(str(detect_step)  if detect_step  is not None else "")
            parts.append(str(detect_phase) if detect_phase is not None else "")

        args.append("--inject " + ":".join(str(p) for p in parts))

    return args


# ── Job generation ────────────────────────────────────────────────────────────

def generate_jobs(matrix, filter_groups=None):
    """
    Yield one job-dict per (group × scenario × model × agent_count) combination.

    Each dict has keys:
      group, scenario, model, script, num_agents, num_steps, workflow_id, extra_args
    """
    models_reg = matrix["models"]
    defaults   = matrix.get("defaults", {})

    for group in matrix["groups"]:
        gid = group["id"]
        if filter_groups and gid not in filter_groups:
            continue

        for scenario in group["scenarios"]:
            sid = scenario["id"]

            agent_counts = resolve("agent_counts", scenario, group, defaults, [3])
            num_steps    = resolve("num_steps",    scenario, group, defaults, 3)
            model_ids    = resolve("models",       scenario, group, defaults, ["qwen"])
            inject_specs = scenario.get("inject", [])

            # extra_args: group prefix (e.g. "--max-turns 1") + scenario suffix
            group_extra = group.get("extra_args", "") or ""
            scen_extra  = scenario.get("extra_args", "") or ""
            base_extra  = " ".join(x for x in [group_extra, scen_extra] if x)

            for num_agents in agent_counts:
                for model_id in model_ids:
                    if model_id not in models_reg:
                        raise ValueError(
                            "Unknown model {!r} in group {!r} scenario {!r}. "
                            "Available: {}".format(
                                model_id, gid, sid, list(models_reg)
                            )
                        )
                    model_def = models_reg[model_id]
                    script = AEG_DIR / model_def["script"]

                    inject_args = build_inject_args(inject_specs, num_agents)
                    extra = " ".join(x for x in [base_extra] + inject_args if x)

                    # Workflow ID: {group}_{scenario}_{model}_{N}w
                    wf_id = "{}_{}_{}_{}w".format(gid, sid, model_id, num_agents)

                    yield {
                        "group":       gid,
                        "scenario":    sid,
                        "model":       model_id,
                        "script":      str(script),
                        "num_agents":  num_agents,
                        "num_steps":   num_steps,
                        "workflow_id": wf_id,
                        "extra_args":  extra,
                    }


# ── SLURM submission ──────────────────────────────────────────────────────────

def submit_job(job, dry_run=False):
    """Submit (or pretend to submit) one SLURM job."""
    # Build a single --export=ALL,VAR=val,… string.
    # AEG_EXTRA_ARGS is placed last so that any spaces in its value don't
    # confuse SLURM's comma-based export-list parsing.
    export = ",".join([
        "ALL",
        "AEG_NUM_AGENTS={}".format(job["num_agents"]),
        "AEG_NUM_STEPS={}".format(job["num_steps"]),
        "AEG_WORKFLOW_ID={}".format(job["workflow_id"]),
        "AEG_EXTRA_ARGS={}".format(job["extra_args"]),
    ])
    cmd = ["sbatch", "--export={}".format(export), job["script"]]

    label = "{:<55} agents={:2d} steps={:2d}".format(
        job["workflow_id"], job["num_agents"], job["num_steps"]
    )
    extra_short = job["extra_args"]
    if len(extra_short) > 55:
        extra_short = extra_short[:52] + "..."
    if extra_short:
        label += "  {}".format(extra_short)

    if dry_run:
        print("[dry-run] {}".format(label))
        print("          cmd: {}".format(" ".join(cmd)))
    else:
        print("[submit]  {}".format(label))
        subprocess.run(cmd, check=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="submit_matrix.py",
        description="Submit AEG test-matrix jobs to SLURM from test_matrix.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "groups", nargs="*", metavar="GROUP_ID",
        help="Groups to submit (default: all; e.g. A B K)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print matching jobs one per line and exit without submitting",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sbatch commands without submitting",
    )
    parser.add_argument(
        "--matrix",
        default=str(SCRIPT_DIR / "test_matrix.yaml"),
        metavar="PATH",
        help="Path to test matrix YAML (default: tests/test_matrix.yaml)",
    )
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    filter_groups = set(args.groups) if args.groups else None

    try:
        jobs = list(generate_jobs(matrix, filter_groups))
    except ValueError as exc:
        sys.exit("Error: {}".format(exc))

    if not jobs:
        available = ", ".join(g["id"] for g in matrix["groups"])
        sys.exit(
            "No jobs matched{}.  Available groups: {}".format(
                " filter {!r}".format(sorted(filter_groups)) if filter_groups else "",
                available,
            )
        )

    if args.list:
        # Compact table: id | agents | steps | extra
        print("{:<55} {:>7} {:>6}  {}".format(
            "workflow_id", "agents", "steps", "extra_args"
        ))
        print("-" * 120)
        for j in jobs:
            extra = j["extra_args"]
            if len(extra) > 60:
                extra = extra[:57] + "..."
            print("{:<55} {:>7} {:>6}  {}".format(
                j["workflow_id"], j["num_agents"], j["num_steps"], extra
            ))
        print("-" * 120)
        print("Total: {} job(s) across {} group(s).".format(
            len(jobs),
            len({j["group"] for j in jobs}),
        ))
        return

    for job in jobs:
        submit_job(job, dry_run=args.dry_run)

    mode = "dry-run (nothing submitted)" if args.dry_run else "submitted"
    print("\n{}: {} jobs.".format(mode.capitalize(), len(jobs)))
    if not args.dry_run:
        print("Results will appear under: {}/results/".format(AEG_DIR))
        print("Monitor with: squeue -u $USER")


if __name__ == "__main__":
    main()
