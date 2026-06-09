#!/usr/bin/env python3
"""
analysis/plot_scaling.py — scaling study analysis.

Reads result.json files from two groups:

  Group S  (sc_*)  — single-GPU repeated runs (submit_scaling_repeated.sh)
  Group MG (mg_*)  — multi-GPU repeated runs  (submit_scaling_multigpu.sh)

Generates three figures:

  fig_scaling_elapsed.png       — elapsed time vs worker count, single-GPU only,
                                  mean ± 1 std with individual scatter.

  fig_scaling_multigpu.png      — single-GPU vs multi-GPU comparison on the same
                                  axes; dashed lines = multi-GPU (4×GH200).

  fig_scaling_phases.png        — phase-decomposed stacked bars for single-GPU
                                  (planner / workers / aggregator).

Falls back gracefully to the original single-run tests (groups A/F/G/H) from
FALLBACK if no sc_* results exist yet.
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))

MODEL_COLOR = {"qwen": "#2196F3", "gemma4": "#FF9800"}
PHASE_COLOR = {"planner_s": "#7E57C2", "workers_s": "#1E88E5", "aggregator_s": "#FB8C00"}
PHASE_LABEL = {"planner_s": "Planner", "workers_s": "Workers (parallel)", "aggregator_s": "Aggregator"}

# ── Single-run fallback data from original groups A/F/G/H ──────────────────────
FALLBACK = {
    # (model, workers): elapsed_sec
    ("qwen",   2): [20.7],
    ("qwen",   3): [49.6],
    ("qwen",   5): [40.2],
    ("gemma4", 2): [23.2],
    ("gemma4", 3): [15.9],
    ("gemma4", 5): [27.6],
    ("gemma4", 8): [35.4],
}


def _model_from_id(wid):
    if "qwen" in wid:
        return "qwen"
    if "gemma" in wid:
        return "gemma4"
    return None


def _workers_from_id(wid):
    for part in wid.split("_"):
        if part.endswith("w") and part[:-1].isdigit():
            return int(part[:-1])
    return None


def load_results(prefix):
    """Load result.json files matching results/{prefix}_*/result.json."""
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, f"{prefix}_*", "result.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        wid     = d.get("workflow_id", os.path.basename(os.path.dirname(path)))
        model   = _model_from_id(wid)
        workers = _workers_from_id(wid)
        if model is None or workers is None:
            continue
        rows.append({
            "workflow_id":  wid,
            "model":        model,
            "workers":      workers,
            "elapsed_s":    d.get("elapsed_sec"),
            "phase_timing": d.get("phase_timing", {}),
        })
    return rows


def build_cells(rows):
    """Return {(model, workers): [row, ...]} grouped dict."""
    cells = defaultdict(list)
    for r in rows:
        if r["elapsed_s"] is not None:
            cells[(r["model"], r["workers"])].append(r)
    return cells


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Single-GPU elapsed time vs worker count
# ══════════════════════════════════════════════════════════════════════════════

def plot_elapsed(cells, have_repeats):
    fig, ax = plt.subplots(figsize=(8, 5))

    all_workers = sorted({w for _, w in cells})

    for model, color in MODEL_COLOR.items():
        xs, means, stds = [], [], []
        for w in all_workers:
            key = (model, w)
            if key not in cells:
                continue
            vals = [r["elapsed_s"] for r in cells[key]]
            xs.append(w)
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=0) if len(vals) > 1 else 0.0)
            for v in vals:
                ax.scatter(w, v, color=color, alpha=0.35, s=28, zorder=4)

        if not xs:
            continue
        xs    = np.array(xs)
        means = np.array(means)
        stds  = np.array(stds)

        ax.plot(xs, means, "o-", color=color, label=model, linewidth=2.2,
                markersize=7, zorder=5)
        if have_repeats:
            ax.fill_between(xs, means - stds, means + stds,
                            color=color, alpha=0.15, zorder=3)
            ax.errorbar(xs, means, yerr=stds, fmt="none",
                        ecolor=color, elinewidth=1.2, capsize=4, zorder=6)

    ax.axvline(5, color="#555", linewidth=1, linestyle="--", alpha=0.5)
    ax.text(5.1, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 50,
            "Ollama\nparallel cap", fontsize=7, color="#777", va="top")

    n_note = " (mean ± 1 std, n≈5)" if have_repeats else " (single runs)"
    ax.set_xlabel("Number of worker agents", fontsize=10)
    ax.set_ylabel("Elapsed (s)", fontsize=10)
    ax.set_title(f"Scaling: elapsed time vs worker count — 1 GPU{n_note}", fontsize=11)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    savefig(fig, "fig_scaling_elapsed.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Single-GPU vs Multi-GPU comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_multigpu_comparison(sc_cells, mg_cells):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    configs = [
        ("qwen",   axes[0]),
        ("gemma4", axes[1]),
    ]

    for model, ax in configs:
        color = MODEL_COLOR[model]

        # Single-GPU line (solid)
        sg_xs, sg_means, sg_stds = [], [], []
        for w in sorted({w for _, w in sc_cells}):
            key = (model, w)
            if key not in sc_cells:
                continue
            vals = [r["elapsed_s"] for r in sc_cells[key]]
            sg_xs.append(w)
            sg_means.append(np.mean(vals))
            sg_stds.append(np.std(vals, ddof=0) if len(vals) > 1 else 0.0)
        if sg_xs:
            sg_xs    = np.array(sg_xs)
            sg_means = np.array(sg_means)
            sg_stds  = np.array(sg_stds)
            ax.plot(sg_xs, sg_means, "o-", color=color, label="1 GPU", linewidth=2.2,
                    markersize=7, zorder=5)
            ax.fill_between(sg_xs, sg_means - sg_stds, sg_means + sg_stds,
                            color=color, alpha=0.12, zorder=3)
            ax.errorbar(sg_xs, sg_means, yerr=sg_stds, fmt="none",
                        ecolor=color, elinewidth=1.2, capsize=4, zorder=6)

        # Multi-GPU line (dashed, slightly darker)
        mg_xs, mg_means, mg_stds = [], [], []
        for w in sorted({w for _, w in mg_cells}):
            key = (model, w)
            if key not in mg_cells:
                continue
            vals = [r["elapsed_s"] for r in mg_cells[key]]
            mg_xs.append(w)
            mg_means.append(np.mean(vals))
            mg_stds.append(np.std(vals, ddof=0) if len(vals) > 1 else 0.0)
        if mg_xs:
            mg_xs    = np.array(mg_xs)
            mg_means = np.array(mg_means)
            mg_stds  = np.array(mg_stds)
            dark = _darken(color, 0.6)
            ax.plot(mg_xs, mg_means, "s--", color=dark, label="4 GPU", linewidth=2.2,
                    markersize=7, zorder=5)
            ax.fill_between(mg_xs, mg_means - mg_stds, mg_means + mg_stds,
                            color=dark, alpha=0.12, zorder=3)
            ax.errorbar(mg_xs, mg_means, yerr=mg_stds, fmt="none",
                        ecolor=dark, elinewidth=1.2, capsize=4, zorder=6)

        # GPU parallel-cap reference lines
        ax.axvline(4, color="#888", linewidth=1, linestyle=":", alpha=0.6)
        ax.text(4.08, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 50,
                "4 GPU\ncap", fontsize=7, color="#999", va="top")
        ax.axvline(5, color="#555", linewidth=1, linestyle="--", alpha=0.4)
        ax.text(5.08, ax.get_ylim()[1] * 0.80 if ax.get_ylim()[1] > 0 else 40,
                "1-GPU\nOllama cap", fontsize=7, color="#999", va="top")

        ax.set_xlabel("Number of worker agents", fontsize=10)
        ax.set_ylabel("Elapsed (s)", fontsize=10)
        ax.set_title(f"{model} — 1 GPU vs 4 GPU", fontsize=11)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)

    fig.suptitle("Multi-GPU scaling comparison (mean ± 1 std)", fontsize=12, y=1.01)
    savefig(fig, "fig_scaling_multigpu.png")


def _darken(hex_color, factor=0.7):
    """Return a darkened version of a hex colour."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(
        int(r * factor), int(g * factor), int(b * factor)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Phase-decomposed stacked bars (single-GPU)
# ══════════════════════════════════════════════════════════════════════════════

def plot_phases(cells):
    phases   = ["planner_s", "workers_s", "aggregator_s"]
    models   = [m for m in MODEL_COLOR if any((m, w) in cells for w in range(20))]
    all_workers = sorted({w for _, w in cells})

    n_w  = len(all_workers)
    n_m  = len(models)
    bar_w = 0.35
    group_w = n_m * bar_w + 0.15

    fig, ax = plt.subplots(figsize=(max(8, n_w * group_w * 2.5 + 1.5), 5))

    for mi, model in enumerate(models):
        for wi, workers in enumerate(all_workers):
            key = (model, workers)
            if key not in cells:
                continue
            rows = cells[key]
            means = {}
            for ph in phases:
                vals = [r["phase_timing"].get(ph) for r in rows
                        if r["phase_timing"].get(ph) is not None]
                means[ph] = np.mean(vals) if vals else 0.0

            x = wi * group_w + mi * bar_w
            bottom = 0.0
            for ph in phases:
                hatch = "" if mi == 0 else "///"
                ax.bar(x, means[ph], bar_w, bottom=bottom,
                       color=PHASE_COLOR[ph], alpha=0.85 if mi == 0 else 0.65,
                       hatch=hatch, edgecolor="white", linewidth=0.5, zorder=3)
                bottom += means[ph]

            total_vals = [r["elapsed_s"] for r in rows]
            total_std  = np.std(total_vals, ddof=0) if len(total_vals) > 1 else 0.0
            if total_std > 0:
                ax.errorbar(x + bar_w/2, bottom, yerr=total_std,
                            fmt="none", ecolor="#ccc", elinewidth=1.2,
                            capsize=3, zorder=5)

            ax.text(x + bar_w/2, bottom + 0.4, model,
                    ha="center", va="bottom", fontsize=6, color="#aaa")

    tick_xs = [wi * group_w + (n_m - 1) * bar_w / 2 for wi in range(n_w)]
    ax.set_xticks(tick_xs)
    ax.set_xticklabels([f"{w} workers" for w in all_workers], fontsize=9)

    ax.set_ylabel("Time (s)", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(facecolor=PHASE_COLOR[ph], label=PHASE_LABEL[ph], alpha=0.85)
        for ph in phases
    ]
    if len(models) > 1:
        legend_handles += [
            mpatches.Patch(facecolor="#888", label=models[0],        alpha=0.85),
            mpatches.Patch(facecolor="#888", label=models[1] + " ///", alpha=0.65, hatch="///"),
        ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left")

    note = " (mean across repeats)" if any(len(cells[k]) > 1 for k in cells) else ""
    ax.set_title(f"Scaling: phase-decomposed elapsed time — 1 GPU{note}", fontsize=11)

    savefig(fig, "fig_scaling_phases.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Single-GPU (sc_*) ──────────────────────────────────────────────────────
    sc_rows = load_results("sc")
    have_repeats = False
    if sc_rows:
        sc_cells = build_cells(sc_rows)
        max_n = max(len(v) for v in sc_cells.values())
        have_repeats = max_n > 1
        print(f"Loaded {len(sc_rows)} sc_* results across {len(sc_cells)} (model, workers) cells "
              f"(max {max_n} repeats per cell).")
    else:
        print("No sc_* results found — falling back to original single-run data.")
        sc_cells = defaultdict(list)
        for (model, workers), elapsed_list in FALLBACK.items():
            for e in elapsed_list:
                sc_cells[(model, workers)].append({"elapsed_s": e, "phase_timing": {}})

    # ── Multi-GPU (mg_*) ───────────────────────────────────────────────────────
    mg_rows = load_results("mg")
    if mg_rows:
        mg_cells = build_cells(mg_rows)
        max_mg = max(len(v) for v in mg_cells.values())
        print(f"Loaded {len(mg_rows)} mg_* results across {len(mg_cells)} (model, workers) cells "
              f"(max {max_mg} repeats per cell).")
    else:
        mg_cells = {}
        print("No mg_* results found — fig_scaling_multigpu.png will be skipped.")

    # ── Figures ────────────────────────────────────────────────────────────────
    plot_elapsed(sc_cells, have_repeats)

    if any(r["phase_timing"] for rows_list in sc_cells.values() for r in rows_list):
        plot_phases(sc_cells)
    else:
        print("No phase_timing data in sc_* — fig_scaling_phases.png skipped.")

    if mg_cells:
        plot_multigpu_comparison(sc_cells, mg_cells)
    else:
        print("Skipping fig_scaling_multigpu.png — run tests/submit_scaling_multigpu.sh first.")


if __name__ == "__main__":
    main()
