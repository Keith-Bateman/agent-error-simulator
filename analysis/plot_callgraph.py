#!/usr/bin/env python3
"""
analysis/plot_callgraph.py — unified workflow call graph for one AEG run.

Produces a swimlane diagram with one horizontal lane per agent role
(planner → workers → aggregator).  Each LLM interaction is a rectangle
whose x-position is the globally-ordered sequence_id from the CEE context
graph; rectangle height fills the lane; colour encodes event type.
Background shading marks the three workflow phases.

Usage:
    python3 analysis/plot_callgraph.py [TEST_ID] [--out PATH]

    TEST_ID defaults to t01_baseline_qwen.
    --out   defaults to analysis/fig_callgraph_<TEST_ID>.png
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Colour scheme ──────────────────────────────────────────────────────────────
PHASE_COLOR = {
    "planner":    "#7E57C2",   # purple
    "worker":     "#1E88E5",   # blue
    "aggregator": "#FB8C00",   # orange
}
EVENT_ALPHA = {
    "conversation_start": 0.85,
    "continuation":       0.55,
    "compression":        0.90,
}
EVENT_HATCH = {
    "conversation_start": "",
    "continuation":       "",
    "compression":        "///",
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def load_events(test_id):
    """Return list of dicts, one per CEE event across all roles."""
    vis_dir = os.path.join(RESULTS_DIR, test_id, "visuals")
    if not os.path.isdir(vis_dir):
        sys.exit(f"No visuals directory for test: {test_id}")

    events = []
    for role_dir in sorted(os.listdir(vis_dir)):
        graphs = sorted(glob.glob(os.path.join(vis_dir, role_dir, "context_graph_*.json")))
        if not graphs:
            continue
        with open(graphs[-1]) as f:
            raw = json.load(f)
        for ev in raw:
            events.append({
                "role":        role_dir,
                "phase":       "planner" if role_dir == "planner"
                               else "aggregator" if role_dir == "aggregator"
                               else "worker",
                "sequence_id": ev.get("sequence_id", 0),
                "event_type":  ev.get("event_type", "continuation"),
                "latency_ms":  ev.get("latency_ms", 0),
                "total_input_tokens": ev.get("total_input_tokens", 0),
                "delta_input_tokens": ev.get("delta_input_tokens", 0),
                "model":       ev.get("model", ""),
            })
    return sorted(events, key=lambda e: e["sequence_id"])


def role_order(events):
    """Return roles sorted: planner first, workers by index, aggregator last."""
    roles = sorted({e["role"] for e in events},
                   key=lambda r: (0 if r == "planner" else
                                  2 if r == "aggregator" else 1,
                                  r))
    return roles


def draw(test_id, out_path):
    events = load_events(test_id)
    roles  = role_order(events)
    n_roles = len(roles)
    role_y  = {r: i for i, r in enumerate(reversed(roles))}  # planner at top

    seq_ids = sorted({e["sequence_id"] for e in events})
    max_seq = max(seq_ids)

    # Phase x-ranges for background shading
    planner_seqs  = [e["sequence_id"] for e in events if e["phase"] == "planner"]
    worker_seqs   = [e["sequence_id"] for e in events if e["phase"] == "worker"]
    agg_seqs      = [e["sequence_id"] for e in events if e["phase"] == "aggregator"]

    fig_w = max(10, max_seq * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, max(4, n_roles * 0.9 + 1.5)))

    # ── Background phase shading ───────────────────────────────────────────────
    shade_alpha = 0.07
    if planner_seqs:
        ax.axvspan(min(planner_seqs) - 0.5, max(planner_seqs) + 0.5,
                   color=PHASE_COLOR["planner"], alpha=shade_alpha, zorder=0)
        ax.text((min(planner_seqs) + max(planner_seqs)) / 2, n_roles - 0.05,
                "Planner", ha="center", va="bottom", fontsize=8,
                color=PHASE_COLOR["planner"], fontweight="bold")
    if worker_seqs:
        ax.axvspan(min(worker_seqs) - 0.5, max(worker_seqs) + 0.5,
                   color=PHASE_COLOR["worker"], alpha=shade_alpha, zorder=0)
        ax.text((min(worker_seqs) + max(worker_seqs)) / 2, n_roles - 0.05,
                "Workers (parallel)", ha="center", va="bottom", fontsize=8,
                color=PHASE_COLOR["worker"], fontweight="bold")
    if agg_seqs:
        ax.axvspan(min(agg_seqs) - 0.5, max(agg_seqs) + 0.5,
                   color=PHASE_COLOR["aggregator"], alpha=shade_alpha, zorder=0)
        ax.text((min(agg_seqs) + max(agg_seqs)) / 2, n_roles - 0.05,
                "Aggregator", ha="center", va="bottom", fontsize=8,
                color=PHASE_COLOR["aggregator"], fontweight="bold")

    # ── Draw interaction rectangles ────────────────────────────────────────────
    bar_h = 0.62
    for ev in events:
        y   = role_y[ev["role"]]
        x   = ev["sequence_id"]
        col = PHASE_COLOR[ev["phase"]]
        alp = EVENT_ALPHA.get(ev["event_type"], 0.6)
        htch = EVENT_HATCH.get(ev["event_type"], "")
        rect = FancyBboxPatch(
            (x - 0.38, y - bar_h / 2), 0.76, bar_h,
            boxstyle="round,pad=0.03",
            facecolor=col, edgecolor="white",
            alpha=alp, hatch=htch, linewidth=0.8, zorder=2,
        )
        ax.add_patch(rect)
        # Annotate with Δ input tokens inside the box
        delta = ev["delta_input_tokens"]
        if delta and abs(delta) > 0:
            ax.text(x, y, f"+{delta}" if delta >= 0 else str(delta),
                    ha="center", va="center", fontsize=5.5,
                    color="white", fontweight="bold", zorder=3)

    # ── Axes formatting ────────────────────────────────────────────────────────
    ax.set_xlim(0.5, max_seq + 0.5)
    ax.set_ylim(-0.7, n_roles - 0.0)
    ax.set_xticks(seq_ids)
    ax.set_xticklabels([str(s) for s in seq_ids], fontsize=8)
    ax.set_yticks(list(range(n_roles)))
    ax.set_yticklabels(list(reversed(roles)), fontsize=9)
    ax.set_xlabel("Global sequence number (CEE interaction order)", fontsize=9)
    ax.set_title(f"AEG Workflow Call Graph — {test_id}", fontsize=11, fontweight="bold")
    ax.xaxis.grid(True, linestyle=":", alpha=0.3, zorder=1)
    ax.set_axisbelow(True)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=PHASE_COLOR["planner"],    label="Planner",    alpha=0.85),
        mpatches.Patch(facecolor=PHASE_COLOR["worker"],     label="Worker",     alpha=0.85),
        mpatches.Patch(facecolor=PHASE_COLOR["aggregator"], label="Aggregator", alpha=0.85),
        mpatches.Patch(facecolor="grey", label="conversation_start", alpha=0.85),
        mpatches.Patch(facecolor="grey", label="continuation",       alpha=0.55),
        mpatches.Patch(facecolor="grey", label="compression",        alpha=0.85, hatch="///"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              ncol=2, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_id", nargs="?", default="t01_baseline_qwen")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"fig_callgraph_{args.test_id}.png",
    )
    draw(args.test_id, out)
