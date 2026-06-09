#!/usr/bin/env python3
"""
analysis/plot_matrix_results.py — cross-model figures for the AEG test matrix.

Auto-discovers all result.json files under results/ (new-style workflow IDs
like A_baseline_gemma4_3w).  Reads result.json and visuals/*context_graph*.json
for each job and produces:

  figM1_abs_error_by_group.png   — per-test abs error, subplots by group
  figM2_error_by_type.png        — mean abs error by error class × model
  figM3_elapsed_by_group.png     — elapsed time per test, subplots by group
  figM4_phase_timing.png         — planner / workers / aggregator phase breakdown
  figM5_token_usage.png          — total tokens per test (input + output stacked)
  figM6_context_growth.png       — cumulative input tokens per turn, by group
  figM7_scaling.png              — elapsed time vs worker count per model
"""

import json, os, glob, math, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
OUT_DIR     = SCRIPT_DIR

# ── Style ──────────────────────────────────────────────────────────────────────
MODEL_COLOR  = {
    "qwen":    "#2196F3",   # blue
    "gemma4":  "#FF9800",   # orange
    "mistral": "#009688",   # teal
    "granite4":"#9C27B0",   # purple
}
GROUP_LABEL = {
    "A": "Baseline",
    "B": "Error variety",
    "D": "Multi-error",
    "F": "Scale min",
    "G": "Scale med",
    "H": "Scale large",
    "I": "Deep sessions",
    "J": "Ctx exhaustion",
    "K": "Deferred detect",
    "L": "Compaction",
}
EC_COLOR = {
    "none":             "#4CAF50",
    "format:worker":    "#2196F3",
    "toolcall:worker":  "#00BCD4",
    "logic:worker":     "#FF9800",
    "multi":            "#9C27B0",
    "fmt:planner":      "#8BC34A",
    "fmt:aggregator":   "#FF5722",
    "logic:aggregator": "#F44336",
}

def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {name}")

# ── Parsing helpers ────────────────────────────────────────────────────────────
def parse_workflow_id(wid):
    """Extract (group, scenario, model, nw) from workflow_id like A_baseline_gemma4_3w."""
    parts = wid.split("_")
    model = "unknown"
    nw    = 3
    for p in parts:
        if p in MODEL_COLOR:
            model = p
        if re.fullmatch(r"\d+w", p):
            nw = int(p[:-1])
    # group letter is always the first part's first character
    group = parts[0][0].upper() if parts else "?"
    # scenario = everything between group prefix and model
    try:
        mi = next(i for i, p in enumerate(parts) if p in MODEL_COLOR)
        scenario = "_".join(parts[1:mi])
    except StopIteration:
        scenario = "_".join(parts[1:])
    return group, scenario, model, nw

def classify_errors(injected):
    """Classify the set of injected errors into a single label."""
    if not injected:
        return "none"
    types, targets = set(), set()
    for e in injected:
        if "format"    in e: types.add("format")
        if "logic"     in e: types.add("logic")
        if "tool_call" in e: types.add("toolcall")
        if "planner"   in e: targets.add("planner")
        if "aggregator"in e: targets.add("aggregator")
        if "worker"    in e: targets.add("worker")
    if "aggregator" in targets:
        return "logic:aggregator" if "logic" in types else "fmt:aggregator"
    if "planner" in targets and "worker" not in targets:
        return "fmt:planner"
    if len(types) > 1 or len(targets) > 1:
        return "multi"
    t = next(iter(types), "unknown")
    return f"{t}:worker"

# ── Load context graphs (flat visuals/ directory) ────────────────────────────
def load_context_graphs(workflow_id):
    """Return dict: role_key → list of event dicts from context_graph_*.json files."""
    vis_dir = os.path.join(RESULTS_DIR, workflow_id, "visuals")
    if not os.path.isdir(vis_dir):
        return {}
    result = {}
    for fname in os.listdir(vis_dir):
        if "context_graph" not in fname or not fname.endswith(".json"):
            continue
        # filename: {wf-id-dashes}-{role}-{index}_context_graph_{ts}.json
        role_key = fname.split("_context_graph_")[0]
        # use the last timestamp file if there are duplicates
        fpath = os.path.join(vis_dir, fname)
        existing = result.get(role_key)
        if existing is None or fname > existing["_fname"]:
            try:
                with open(fpath) as f:
                    events = json.load(f)
                result[role_key] = events
                result[role_key + "_fname"] = fname   # internal key for dedup
            except Exception:
                pass
    # remove internal dedup keys
    return {k: v for k, v in result.items() if not k.endswith("_fname")}

def token_summary(cg):
    """Return (total_in, total_out, max_ctx, has_data) from context graphs."""
    total_in = total_out = max_ctx = 0
    has_data = False
    for role_key, events in cg.items():
        if not isinstance(events, list):
            continue
        for ev in events:
            di = ev.get("delta_input_tokens")  or 0
            do = ev.get("delta_output_tokens") or 0
            ti = ev.get("total_input_tokens")  or 0
            if di or do or ti:
                has_data = True
            total_in  += di
            total_out += do
            if ti > max_ctx:
                max_ctx = ti
    return total_in, total_out, max_ctx, has_data

def get_worker_chains(cg):
    """Return list of chains (each chain = list of total_input_tokens per turn)."""
    chains = []
    for role_key, events in cg.items():
        if not isinstance(events, list):
            continue
        if "worker" not in role_key:
            continue
        by_conv = defaultdict(list)
        for ev in events:
            by_conv[ev.get("conversation_id", "")].append(ev)
        for evts in by_conv.values():
            seq_map = {ev["sequence_id"]: ev for ev in evts if "sequence_id" in ev}
            parent_to_children = defaultdict(list)
            roots = []
            for ev in evts:
                parent = ev.get("parent_sequence_id", 0)
                if parent == 0 or parent not in seq_map:
                    roots.append(ev)
                else:
                    parent_to_children[parent].append(ev)
            for root in sorted(roots, key=lambda e: e.get("sequence_id", 0)):
                chain, cur, visited = [], root, set()
                while cur:
                    sid = cur.get("sequence_id")
                    if sid in visited:
                        break
                    visited.add(sid)
                    ti = cur.get("total_input_tokens", 0)
                    if ti > 0:
                        chain.append(ti)
                    nexts = sorted(parent_to_children.get(sid, []),
                                   key=lambda e: e.get("sequence_id", 0))
                    cur = nexts[0] if nexts else None
                if chain:
                    chains.append(chain)
    return chains

# ── Discover and load all results ─────────────────────────────────────────────
rows = []
for d in sorted(os.listdir(RESULTS_DIR)):
    if d.startswith("t"):          # skip old t01-t42 tests
        continue
    rfile = os.path.join(RESULTS_DIR, d, "result.json")
    if not os.path.exists(rfile):
        continue
    try:
        with open(rfile) as f:
            r = json.load(f)
    except Exception:
        continue

    group, scenario, model, nw = parse_workflow_id(d)
    fr       = r.get("final_report") or {}
    injected = r.get("injected_errors", [])
    elapsed  = r.get("elapsed_sec")
    abs_err  = fr.get("absolute_error")
    pt       = r.get("phase_timing") or {}
    workers  = r.get("worker_results", [])
    w_sum    = sum(w.get("final_integral", 0) for w in workers) if workers else None
    n_steps  = max((len(w.get("steps", [])) for w in workers), default=0)

    cg                           = load_context_graphs(d)
    total_in, total_out, max_ctx, has_tok = token_summary(cg)

    rows.append({
        "id":            d,
        "group":         group,
        "scenario":      scenario,
        "model":         model,
        "nw":            nw,
        "n_steps":       n_steps,
        "error_class":   classify_errors(injected),
        "injected":      injected,
        "elapsed":       elapsed,
        "planner_s":     pt.get("planner_s"),
        "workers_s":     pt.get("workers_s"),
        "aggregator_s":  pt.get("aggregator_s"),
        "abs_err":       abs_err,
        "w_sum":         w_sum,
        "total_in":      total_in,
        "total_out":     total_out,
        "max_ctx":       max_ctx,
        "has_tok":       has_tok,
        "cg":            cg,
    })

MODELS   = [m for m in ("qwen","gemma4","mistral","granite4") if any(r["model"]==m for r in rows)]
GROUPS   = sorted({r["group"] for r in rows})
print(f"Loaded {len(rows)} results | models: {MODELS} | groups: {GROUPS}")
tok_rows = [r for r in rows if r["has_tok"]]
print(f"  {len(tok_rows)} with token data")

# ── Baselines per model ────────────────────────────────────────────────────────
BASELINE = {}
for r in rows:
    if r["group"] == "A" and r["error_class"] == "none":
        BASELINE[r["model"]] = r["abs_err"] or 0.0

# ── Helper: NaN-safe abs_err ───────────────────────────────────────────────────
SENTINEL = 15.0   # shown as hatched bar when abs_err is None

def ae(r):
    v = r["abs_err"]
    return float("nan") if v is None else v

def plot_ae(v):
    return SENTINEL if (v is None or math.isnan(v)) else v

# ══════════════════════════════════════════════════════════════════════════════
# Fig M1 — Absolute error per test, subplots by group
# ══════════════════════════════════════════════════════════════════════════════
group_list = [g for g in ("A","B","D","F","G","H","I","J","K","L") if g in GROUPS]
ncols = 2
nrows = math.ceil(len(group_list) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))
axes = axes.flatten()

legend_patches = [mpatches.Patch(color=MODEL_COLOR[m], label=m) for m in MODELS]
legend_patches.append(mpatches.Patch(facecolor="grey", hatch="//",
                                     edgecolor="grey", label="no numeric output"))

for ax, grp in zip(axes, group_list):
    grp_rows = [r for r in rows if r["group"] == grp]
    if not grp_rows:
        ax.set_visible(False)
        continue
    x = np.arange(len(grp_rows))
    colors = [MODEL_COLOR.get(r["model"], "#888") for r in grp_rows]
    vals   = [plot_ae(r["abs_err"]) for r in grp_rows]
    bars   = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
    for i, r in enumerate(grp_rows):
        if r["abs_err"] is None:
            bars[i].set_hatch("//")
            bars[i].set_edgecolor("grey")
    xlabels = [r["scenario"].replace("_", " ") + f"\n{r['model']}" for r in grp_rows]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylabel("|error|", fontsize=8)
    ax.set_title(f"Group {grp} — {GROUP_LABEL.get(grp,'')}", fontsize=9, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    # baseline reference lines per model
    for model, bv in BASELINE.items():
        if bv and any(r["model"] == model for r in grp_rows):
            ax.axhline(bv, color=MODEL_COLOR[model], linestyle=":", linewidth=0.8, alpha=0.6)

# hide unused axes
for ax in axes[len(group_list):]:
    ax.set_visible(False)

fig.legend(handles=legend_patches, loc="lower right", fontsize=9, ncol=len(MODELS)+1)
fig.suptitle("Fig M1 — Absolute numerical error per test\n"
             "Dotted lines = per-model baselines; hatched = no numeric output",
             fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
savefig(fig, "figM1_abs_error_by_group.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig M2 — Mean absolute error by error class × model
# ══════════════════════════════════════════════════════════════════════════════
EC_ORDER = ["none","format:worker","toolcall:worker","logic:worker",
            "multi","fmt:planner","fmt:aggregator","logic:aggregator"]
ec_present = [c for c in EC_ORDER if any(r["error_class"]==c and r["abs_err"] is not None for r in rows)]

data = {m: {c: [] for c in ec_present} for m in MODELS}
for r in rows:
    if r["abs_err"] is not None and r["error_class"] in ec_present:
        data[r["model"]][r["error_class"]].append(r["abs_err"])

xc = np.arange(len(ec_present))
width = 0.8 / len(MODELS)
fig, ax = plt.subplots(figsize=(14, 6))

for mi, model in enumerate(MODELS):
    offset = (mi - len(MODELS)/2 + 0.5) * width
    means = [np.mean(data[model][c]) if data[model][c] else 0 for c in ec_present]
    stds  = [np.std(data[model][c])  if data[model][c] else 0 for c in ec_present]
    counts= [len(data[model][c]) for c in ec_present]
    bars  = ax.bar(xc + offset, means, width, yerr=stds, capsize=3,
                   color=MODEL_COLOR[model], label=model,
                   alpha=0.85, edgecolor="white", linewidth=0.5)
    for i, (m_val, cnt) in enumerate(zip(means, counts)):
        if cnt > 0 and m_val > 0:
            ax.text(xc[i] + offset, m_val * 1.8 + 1e-6,
                    f"n={cnt}", ha="center", fontsize=6, color="#333")

ax.set_xticks(xc)
ax.set_xticklabels(ec_present, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("|final − exact| (symlog)", fontsize=10)
ax.set_yscale("symlog", linthresh=1e-4)
ax.set_title("Fig M2 — Mean absolute error by error class and model (± std dev)\n"
             "n = number of tests contributing to each bar", fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
savefig(fig, "figM2_error_by_type.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig M3 — Elapsed time per test, subplots by group
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))
axes = axes.flatten()

for ax, grp in zip(axes, group_list):
    grp_rows = [r for r in rows if r["group"] == grp]
    if not grp_rows:
        ax.set_visible(False)
        continue
    x      = np.arange(len(grp_rows))
    colors = [MODEL_COLOR.get(r["model"], "#888") for r in grp_rows]
    elaps  = [r["elapsed"] or 0 for r in grp_rows]
    ax.bar(x, elaps, color=colors, edgecolor="white", linewidth=0.5)
    xlabels = [r["scenario"].replace("_", " ") + f"\n{r['model']}" for r in grp_rows]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Elapsed (s)", fontsize=8)
    ax.set_title(f"Group {grp} — {GROUP_LABEL.get(grp,'')}", fontsize=9, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

for ax in axes[len(group_list):]:
    ax.set_visible(False)

fig.legend(handles=[mpatches.Patch(color=MODEL_COLOR[m], label=m) for m in MODELS],
           loc="lower right", fontsize=9)
fig.suptitle("Fig M3 — Workflow elapsed time per test",
             fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
savefig(fig, "figM3_elapsed_by_group.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig M4 — Phase timing breakdown (planner / workers / aggregator)
# ══════════════════════════════════════════════════════════════════════════════
pt_rows = [r for r in rows if r["planner_s"] is not None]
if pt_rows:
    # Sort by group then model
    pt_rows.sort(key=lambda r: (r["group"], r["model"], r["scenario"]))
    x      = np.arange(len(pt_rows))
    p_vals = [r["planner_s"]    or 0 for r in pt_rows]
    w_vals = [r["workers_s"]    or 0 for r in pt_rows]
    a_vals = [r["aggregator_s"] or 0 for r in pt_rows]
    colors = [MODEL_COLOR.get(r["model"], "#888") for r in pt_rows]

    fig, ax = plt.subplots(figsize=(max(20, len(pt_rows) * 0.18 + 4), 6))
    ax.bar(x, p_vals, color=colors, alpha=1.0,  label="planner")
    ax.bar(x, w_vals, bottom=p_vals, color=colors, alpha=0.55, label="workers")
    a_bottom = [p+w for p,w in zip(p_vals, w_vals)]
    ax.bar(x, a_vals, bottom=a_bottom, color=colors, alpha=0.25, label="aggregator")

    labels = [f"{r['scenario'][:14]}\n{r['model']}" for r in pt_rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=5.5)
    ax.set_ylabel("Time (s)")
    ax.set_title("Fig M4 — Phase timing breakdown (planner / workers / aggregator)\n"
                 "Solid = planner, mid = workers, light = aggregator",
                 fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Custom legend: model colors + phase shading
    phase_patches = [
        mpatches.Patch(facecolor="grey", alpha=1.0,  label="planner (solid)"),
        mpatches.Patch(facecolor="grey", alpha=0.55, label="workers (mid)"),
        mpatches.Patch(facecolor="grey", alpha=0.25, label="aggregator (light)"),
    ]
    model_patches = [mpatches.Patch(color=MODEL_COLOR[m], label=m) for m in MODELS]
    ax.legend(handles=model_patches + phase_patches, loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    savefig(fig, "figM4_phase_timing.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig M5 — Token usage per test (input + output stacked)
# ══════════════════════════════════════════════════════════════════════════════
if tok_rows:
    tok_rows_sorted = sorted(tok_rows, key=lambda r: (r["group"], r["model"], r["scenario"]))
    x      = np.arange(len(tok_rows_sorted))
    colors = [MODEL_COLOR.get(r["model"], "#888") for r in tok_rows_sorted]
    ins    = [r["total_in"]  for r in tok_rows_sorted]
    outs   = [r["total_out"] for r in tok_rows_sorted]

    fig, ax = plt.subplots(figsize=(max(20, len(tok_rows_sorted) * 0.2 + 4), 6))
    ax.bar(x, ins,  color=colors, alpha=0.9, label="input tokens")
    ax.bar(x, outs, bottom=ins, color=colors, alpha=0.35, label="output tokens")

    labels = [f"{r['scenario'][:14]}\n{r['model']}" for r in tok_rows_sorted]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=5.5)
    ax.set_ylabel("Total tokens (all sessions combined)")
    ax.set_title("Fig M5 — Token consumption per test (input stacked with output)\n"
                 "Solid = input, lighter = output added on top",
                 fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    legend_h = ([mpatches.Patch(color=MODEL_COLOR[m], label=m) for m in MODELS] +
                [mpatches.Patch(facecolor="grey", alpha=0.9, label="input"),
                 mpatches.Patch(facecolor="grey", alpha=0.35, label="+ output")])
    ax.legend(handles=legend_h, loc="upper left", fontsize=8, ncol=3)
    plt.tight_layout()
    savefig(fig, "figM5_token_usage.png")

    # Also: mean tokens by model × error class
    tok_means = {m: {c: [] for c in ec_present} for m in MODELS}
    for r in tok_rows_sorted:
        if r["error_class"] in ec_present:
            total = r["total_in"] + r["total_out"]
            tok_means[r["model"]][r["error_class"]].append(total)

    fig, ax = plt.subplots(figsize=(14, 5))
    for mi, model in enumerate(MODELS):
        offset = (mi - len(MODELS)/2 + 0.5) * width
        means = [np.mean(tok_means[model][c]) if tok_means[model][c] else 0 for c in ec_present]
        ax.bar(xc + offset, means, width, color=MODEL_COLOR[model],
               label=model, alpha=0.85, edgecolor="white")
    ax.set_xticks(xc)
    ax.set_xticklabels(ec_present, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Mean total tokens (in + out)")
    ax.set_title("Fig M5b — Mean token consumption by error class and model", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    savefig(fig, "figM5b_token_by_class.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig M6 — Context growth: cumulative input tokens per turn, by group
# ══════════════════════════════════════════════════════════════════════════════
ctx_groups = [g for g in ("A","I","J","K","L") if g in GROUPS]
fig, axes = plt.subplots(1, len(ctx_groups), figsize=(5 * len(ctx_groups), 5), sharey=False)
if len(ctx_groups) == 1:
    axes = [axes]

for ax, grp in zip(axes, ctx_groups):
    grp_rows = [r for r in tok_rows if r["group"] == grp]
    for r in grp_rows:
        chains = get_worker_chains(r["cg"])
        color  = MODEL_COLOR.get(r["model"], "#888")
        for chain in chains:
            xs = list(range(1, len(chain) + 1))
            if len(chain) == 1:
                ax.scatter(xs, chain, color=color, alpha=0.4, s=18, zorder=4)
            else:
                ax.plot(xs, chain, color=color, alpha=0.55, linewidth=1.3)
    ax.set_title(f"Group {grp}: {GROUP_LABEL.get(grp,'')}", fontsize=9)
    ax.set_xlabel("Turn # within conversation chain", fontsize=8)
    ax.set_ylabel("Cumulative input tokens", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

model_patches = [mpatches.Patch(color=MODEL_COLOR[m], label=m) for m in MODELS]
solid_line    = plt.Line2D([0],[0], color="grey", linewidth=1.3, label="multi-turn chain")
dot_marker    = plt.Line2D([0],[0], color="grey", marker="o", linestyle="None",
                           markersize=5, alpha=0.5, label="single-turn (reset)")
fig.legend(handles=model_patches + [solid_line, dot_marker],
           loc="upper right", fontsize=9)
fig.suptitle("Fig M6 — Context growth: cumulative input tokens per turn (worker sessions)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
savefig(fig, "figM6_context_growth.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig M7 — Elapsed time vs worker count (scaling)
# ══════════════════════════════════════════════════════════════════════════════
scale_groups = [g for g in ("A","F","G","H") if g in GROUPS]
fig, ax = plt.subplots(figsize=(9, 5))
for model in MODELS:
    pts = defaultdict(list)
    for r in rows:
        if r["group"] in scale_groups and r["model"] == model and r["elapsed"]:
            pts[r["nw"]].append(r["elapsed"])
    if not pts:
        continue
    xs = sorted(pts.keys())
    ys = [np.mean(pts[x]) for x in xs]
    ax.plot(xs, ys, "o-", color=MODEL_COLOR[model], label=model, linewidth=2, markersize=7)
    for wx, vals in pts.items():
        for v in vals:
            ax.scatter(wx, v, color=MODEL_COLOR[model], alpha=0.35, s=25, zorder=5)

ax.set_xlabel("Number of worker agents", fontsize=10)
ax.set_ylabel("Elapsed (s)", fontsize=10)
ax.set_title("Fig M7 — Elapsed time vs worker count (groups A, F, G, H)\n"
             "Lines = group mean; faint dots = individual runs", fontsize=11)
ax.xaxis.grid(True, linestyle="--", alpha=0.4)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(fontsize=9)
plt.tight_layout()
savefig(fig, "figM7_scaling.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig M8 — Abs error vs elapsed time scatter (all tests, colored by model)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
for model in MODELS:
    model_rows = [r for r in rows if r["model"] == model
                  and r["abs_err"] is not None and r["elapsed"]]
    if not model_rows:
        continue
    xs = [r["elapsed"]  for r in model_rows]
    ys = [max(r["abs_err"], 1e-6) for r in model_rows]
    ax.scatter(xs, ys, color=MODEL_COLOR[model], label=model,
               alpha=0.55, s=40, zorder=4)

ax.set_xlabel("Elapsed time (s)", fontsize=10)
ax.set_ylabel("|final − exact| (log scale)", fontsize=10)
ax.set_yscale("log")
ax.set_title("Fig M8 — Abs error vs elapsed time\n(each point = one test run)", fontsize=11)
ax.xaxis.grid(True, linestyle="--", alpha=0.4)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(fontsize=9)
plt.tight_layout()
savefig(fig, "figM8_error_vs_elapsed.png")

print(f"\nAll figures written to: {OUT_DIR}")
