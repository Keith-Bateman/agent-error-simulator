#!/usr/bin/env python3
"""
analysis/plot_results.py — generate summary table + figures from AEG test suite results.

Reads both result.json (workflow-level) and visuals/*/context_graph_*.json
(per-session CEE data) to produce token-aware figures.

Outputs written to the same directory as this script:
  results_table.csv          — full per-test metric table
  results_table.md           — markdown version
  fig1_elapsed.png           — workflow elapsed time per test
  fig2_abs_error.png         — absolute numerical error per test (log scale)
  fig3_model_compare.png     — gemma4 vs qwen head-to-head
  fig4_error_type.png        — mean abs error by error class
  fig5_scaling.png           — elapsed time vs worker count
  fig6_token_usage.png       — total tokens per session by test
  fig7_context_growth.png    — cumulative input tokens across turns (per session type)
  fig8_token_efficiency.png  — output/input token ratio by model and error class
"""

import json, os, csv, math, glob
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "results")
OUT_DIR     = SCRIPT_DIR

# ── Test metadata ──────────────────────────────────────────────────────────────
TEST_META = {
    "t01_baseline_qwen":            ("A","baseline",          "qwen",   3,2),
    "t02_baseline_gemma4":          ("A","baseline",          "gemma4", 3,2),
    "t03_fmt_worker_qwen":          ("B","fmt:worker",        "qwen",   3,2),
    "t04_logic_worker_qwen":        ("B","logic:worker",      "qwen",   3,2),
    "t05_toolcall_worker_qwen":     ("B","toolcall:worker",   "qwen",   3,2),
    "t06_fmt_planner_qwen":         ("B","fmt:planner",       "qwen",   3,2),
    "t07_fmt_aggregator_qwen":      ("B","fmt:aggregator",    "qwen",   3,2),
    "t08_logic_aggregator_qwen":    ("B","logic:aggregator",  "qwen",   3,2),
    "t09_fmt_worker_gemma4":        ("C","fmt:worker",        "gemma4", 3,2),
    "t10_logic_worker_gemma4":      ("C","logic:worker",      "gemma4", 3,2),
    "t11_toolcall_worker_gemma4":   ("C","toolcall:worker",   "gemma4", 3,2),
    "t12_two_errors_qwen":          ("D","2×error",           "qwen",   3,2),
    "t13_all_types_qwen":           ("D","all-types",         "qwen",   3,2),
    "t14_cross_agent_qwen":         ("D","cross-agent",       "qwen",   3,2),
    "t15_staggered_steps_qwen":     ("D","staggered",         "qwen",   3,3),
    "t16_two_errors_gemma4":        ("E","2×error",           "gemma4", 3,2),
    "t17_all_types_gemma4":         ("E","all-types",         "gemma4", 3,2),
    "t18_scale_min_qwen":           ("F","scale-min",         "qwen",   2,1),
    "t19_scale_min_gemma4":         ("F","scale-min",         "gemma4", 2,1),
    "t20_scale_med_qwen":           ("G","scale-med",         "qwen",   5,2),
    "t21_scale_med_gemma4":         ("G","scale-med",         "gemma4", 5,2),
    "t22_scale_large_gemma4":       ("H","scale-large",       "gemma4", 8,2),
    "t23_deep_clean_qwen":          ("I","deep-clean",        "qwen",   3,3),
    "t24_deep_clean_gemma4":        ("I","deep-clean",        "gemma4", 3,3),
    "t25_deep_fmt_mid_qwen":        ("I","deep-fmt-mid",      "qwen",   3,3),
    "t26_deep_toolcall_mid_gemma4": ("I","deep-toolcall-mid", "gemma4", 3,3),
    "t27_deep_logic_late_qwen":     ("I","deep-logic-late",   "qwen",   3,3),
    "t28_deep_multi_err_gemma4":    ("I","deep-multi-err",    "gemma4", 5,3),
    "t29_ctx_exhaust_clean_qwen":   ("J","ctx-clean",         "qwen",   3,4),
    "t30_ctx_exhaust_clean_gemma4": ("J","ctx-clean",         "gemma4", 3,4),
    "t31_ctx_exhaust_fmt_qwen":     ("J","ctx-fmt",           "qwen",   3,4),
    "t32_ctx_exhaust_fmt_gemma4":   ("J","ctx-fmt",           "gemma4", 3,4),
    "t33_ctx_exhaust_toolcall_qwen":("J","ctx-toolcall",      "qwen",   3,4),
    "t34_ctx_exhaust_scale_gemma4": ("J","ctx-scale",         "gemma4", 5,4),
    # K — num_ctx=512 exhaustion attempts (Ollama overrides to model minimum; no dips observed)
    "t35_ctx512_clean_qwen":        ("K","ctx512-clean",      "qwen",   3,6),
    "t36_ctx512_clean_gemma4":      ("K","ctx512-clean",      "gemma4", 3,6),
    "t37_ctx512_fmt_qwen":          ("K","ctx512-fmt",        "qwen",   3,6),
    "t38_ctx512_toolcall_gemma4":   ("K","ctx512-toolcall",   "gemma4", 3,6),
    # L — explicit message trimming (max_turns=1): produces compression events in CEE
    "t39_compact_clean_qwen":       ("L","compact-clean",     "qwen",   3,6),
    "t40_compact_clean_gemma4":     ("L","compact-clean",     "gemma4", 3,6),
    "t41_compact_fmt_qwen":         ("L","compact-fmt",       "qwen",   3,6),
    "t42_compact_logic_gemma4":     ("L","compact-logic",     "gemma4", 3,6),
}

# ── Load context_graph data from visuals/ ─────────────────────────────────────
def load_context_graphs(test_id):
    """Return dict: role → list of event dicts from the most-recent context_graph_*.json."""
    vis_dir = os.path.join(RESULTS_DIR, test_id, "visuals")
    if not os.path.isdir(vis_dir):
        return {}
    result = {}
    for role_dir in os.listdir(vis_dir):
        role_path = os.path.join(vis_dir, role_dir)
        graphs = sorted(glob.glob(os.path.join(role_path, "context_graph_*.json")))
        if not graphs:
            continue
        # Use most recent (alphabetically last timestamp)
        try:
            with open(graphs[-1]) as f:
                events = json.load(f)
            result[role_dir] = events
        except Exception:
            pass
    return result

def classify_errors(injected):
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
    if "aggregator" in targets: return "logic:aggregator" if "logic" in types else "fmt:aggregator"
    if "planner"    in targets: return "fmt:planner"
    if len(injected) > 1:       return "multi"
    return "+".join(sorted(types))

# ── Hypothetical-uncorrected error ────────────────────────────────────────────
import re as _re

def _hypothetical_abs_err(d):
    """Return the absolute error that would have occurred with no multi-step recovery.

    Rules per worker error type:
      logic      — use the value from the injected step (integral * scale_factor).
      toolcall   — tool returned null; worker would contribute 0.
      format_t1  — tool runs with defaults regardless; integral is correct (transparent).
      format_t2  — LLM answer turn is plain text; worker would contribute 0.
    Planner / aggregator errors are not worker-level; returns None for those.
    """
    injected = d.get("injected_errors", [])
    if not injected:
        return float("nan")
    if any("planner" in e or "aggregator" in e for e in injected):
        return float("nan")

    exact = d["exact_integral"]
    hyp_integrals = []
    for w in d.get("worker_results", []):
        fi = w["final_integral"]
        err_str = " ".join(w.get("errors", []))
        if "logic_error" in err_str:
            injected_val = None
            for s in w.get("steps", []):
                if s.get("injected"):
                    nums = _re.findall(r'"integral"\s*:\s*(-?[0-9.e+\-]+)', s["content"])
                    if nums:
                        injected_val = float(nums[0])
                        break
            fi = injected_val if injected_val is not None else fi
        elif "tool_call_error" in err_str:
            fi = 0.0
        elif "format_turn2" in err_str:
            fi = 0.0
        # format_turn1: transparent — fi unchanged
        hyp_integrals.append(fi)

    valid = [v for v in hyp_integrals if v is not None]
    return abs(sum(valid) - exact)


# ── Build rows ────────────────────────────────────────────────────────────────
rows = []
for tid, (grp, desc, model, n_workers, n_steps) in TEST_META.items():
    path = os.path.join(RESULTS_DIR, tid, "result.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        d = json.load(f)
    report   = d.get("final_report") or {}
    injected = d.get("injected_errors", [])
    elapsed  = d.get("elapsed_sec")
    abs_err  = report.get("absolute_error")
    converged= report.get("converged_workers", 0) or 0
    hyp_err  = _hypothetical_abs_err(d)

    # Load CEE token data from context_graphs
    cg = load_context_graphs(tid)
    total_in = total_out = 0
    max_ctx  = 0   # peak total_input_tokens across any session/turn
    has_token_data = False
    session_token_totals = []   # per-session total_input at last turn
    for role, events in cg.items():
        if not events:
            continue
        s_in = s_out = 0
        peak = 0
        for ev in events:
            di = ev.get("delta_input_tokens") or 0
            do = ev.get("delta_output_tokens") or 0
            ti = ev.get("total_input_tokens")  or 0
            if di != 0 or do != 0 or ti != 0:
                has_token_data = True
            s_in  += di
            s_out += do
            if ti > peak:
                peak = ti
        total_in  += s_in
        total_out += s_out
        if peak > max_ctx:
            max_ctx = peak
        session_token_totals.append(s_in + s_out)

    rows.append({
        "id":            tid,
        "group":         grp,
        "desc":          desc,
        "model":         model,
        "workers":       n_workers,
        "steps":         n_steps,
        "injected":      "; ".join(injected) if injected else "—",
        "error_class":   classify_errors(injected),
        "n_injected":    len(injected),
        "elapsed_s":     elapsed,
        "abs_err":       abs_err if abs_err is not None else float("nan"),
        "hyp_abs_err":   hyp_err,
        "converged":     converged,
        "total_in_tokens":  total_in,
        "total_out_tokens": total_out,
        "max_ctx_tokens":   max_ctx,
        "has_token_data":   has_token_data,
        "cg":            cg,   # raw context graph data for per-turn plots
    })

rows.sort(key=lambda r: r["id"])
print(f"Loaded {len(rows)} tests with result.json")
token_rows = [r for r in rows if r["has_token_data"]]
print(f"  of which {len(token_rows)} have token data from CEE")

# ── Write CSV / Markdown ──────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "results_table.csv")
fieldnames = ["id","group","desc","model","workers","steps","error_class",
              "n_injected","elapsed_s","abs_err","hyp_abs_err","converged",
              "total_in_tokens","total_out_tokens","max_ctx_tokens","injected"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)
print(f"Wrote {csv_path}")

md_path = os.path.join(OUT_DIR, "results_table.md")
header = ("| ID | Grp | Model | W | S | Error class | Elapsed (s) | Abs error "
          "| In tokens | Out tokens | Peak ctx | Injected errors |")
sep    = "|---|---|---|---|---|---|---|---|---|---|---|---|"
lines  = [header, sep]
for r in rows:
    ae  = f"{r['abs_err']:.4g}" if not math.isnan(r["abs_err"]) else "—"
    el  = f"{r['elapsed_s']:.1f}" if r["elapsed_s"] is not None else "—"
    ti  = str(r["total_in_tokens"])  if r["has_token_data"] else "—"
    to_ = str(r["total_out_tokens"]) if r["has_token_data"] else "—"
    mc  = str(r["max_ctx_tokens"])   if r["has_token_data"] else "—"
    lines.append(
        f"| {r['id']} | {r['group']} | {r['model']} | {r['workers']} | {r['steps']} "
        f"| {r['error_class']} | {el} | {ae} | {ti} | {to_} | {mc} | {r['injected']} |"
    )
with open(md_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {md_path}")

# ── Shared helpers ─────────────────────────────────────────────────────────────
MODEL_COLOR = {"qwen": "#2196F3", "gemma4": "#FF9800"}
GROUP_COLOR = {
    "A":"#4CAF50","B":"#2196F3","C":"#FF9800","D":"#9C27B0","E":"#F44336",
    "F":"#00BCD4","G":"#8BC34A","H":"#FF5722","I":"#607D8B","J":"#795548",
}
labels = [r["id"].replace("_qwen","_Q").replace("_gemma4","_G") for r in rows]

def savefig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {name}")

# ── Fig 1 — Elapsed time ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18,6))
colors  = [MODEL_COLOR[r["model"]] for r in rows]
elapsed = [r["elapsed_s"] or 0 for r in rows]
x = np.arange(len(rows))
ax.bar(x, elapsed, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.5)
ax.set_ylabel("Workflow elapsed time (s)")
ax.set_title("Fig 1 — Workflow elapsed time across all tests")
ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
patches = [mpatches.Patch(color=c, label=m) for m,c in MODEL_COLOR.items()]
ax.legend(handles=patches, loc="upper right")
plt.tight_layout(); savefig(fig, "fig1_elapsed.png")

# ── Fig 2 — Absolute error (log scale) ───────────────────────────────────────
SENTINEL = 15.0
fig, ax = plt.subplots(figsize=(18,6))
plot_errs = [SENTINEL if math.isnan(r["abs_err"]) else r["abs_err"] for r in rows]
colors    = [MODEL_COLOR[r["model"]] for r in rows]
bars = ax.bar(x, plot_errs, color=colors, edgecolor="white", linewidth=0.5)
for i, r in enumerate(rows):
    if math.isnan(r["abs_err"]): bars[i].set_hatch("//"); bars[i].set_edgecolor("grey")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.5)
ax.set_ylabel("|final_total − exact| (symlog)")
ax.set_yscale("symlog", linthresh=1e-5)
ax.set_title("Fig 2 — Absolute numerical error (lower = better; hatched = no numeric output)")
ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
ax.axhline(y=0.0003125, color="grey", linestyle=":", linewidth=1)
ax.legend(handles=patches+[
    mpatches.Patch(facecolor="grey", hatch="//", label="no numeric output"),
    plt.Line2D([0],[0], color="grey", linestyle=":", label="qwen baseline (3.1×10⁻⁴)")
], loc="upper right", fontsize=8)
plt.tight_layout(); savefig(fig, "fig2_abs_error.png")

# ── Fig 3 — Model head-to-head ────────────────────────────────────────────────
PAIRS = [
    ("baseline",       "t01_baseline_qwen",       "t02_baseline_gemma4"),
    ("fmt:worker",     "t03_fmt_worker_qwen",      "t09_fmt_worker_gemma4"),
    ("logic:worker",   "t04_logic_worker_qwen",    "t10_logic_worker_gemma4"),
    ("toolcall:worker","t05_toolcall_worker_qwen", "t11_toolcall_worker_gemma4"),
    ("2×error",        "t12_two_errors_qwen",      "t16_two_errors_gemma4"),
    ("all-types",      "t13_all_types_qwen",       "t17_all_types_gemma4"),
    ("scale-min",      "t18_scale_min_qwen",       "t19_scale_min_gemma4"),
    ("scale-med",      "t20_scale_med_qwen",       "t21_scale_med_gemma4"),
    ("deep-clean",     "t23_deep_clean_qwen",      "t24_deep_clean_gemma4"),
    ("ctx-clean",      "t29_ctx_exhaust_clean_qwen","t30_ctx_exhaust_clean_gemma4"),
    ("ctx-fmt",        "t31_ctx_exhaust_fmt_qwen", "t32_ctx_exhaust_fmt_gemma4"),
]
by_id = {r["id"]: r for r in rows}
valid_pairs = [(lbl,q,g) for lbl,q,g in PAIRS if q in by_id and g in by_id]
xp = np.arange(len(valid_pairs)); w = 0.38
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
ax1.bar(xp-w/2, [by_id[q]["elapsed_s"] for _,q,_ in valid_pairs], w, color=MODEL_COLOR["qwen"],   label="qwen2.5-coder:32b")
ax1.bar(xp+w/2, [by_id[g]["elapsed_s"] for _,_,g in valid_pairs], w, color=MODEL_COLOR["gemma4"], label="gemma4:27b")
ax1.set_xticks(xp); ax1.set_xticklabels([l for l,_,_ in valid_pairs], rotation=40, ha="right", fontsize=9)
ax1.set_ylabel("Elapsed (s)"); ax1.set_title("Elapsed — qwen vs gemma4")
ax1.yaxis.grid(True, linestyle="--", alpha=0.5); ax1.set_axisbelow(True); ax1.legend()

qe = [0 if math.isnan(by_id[q]["abs_err"]) else by_id[q]["abs_err"] for _,q,_ in valid_pairs]
ge = [0 if math.isnan(by_id[g]["abs_err"]) else by_id[g]["abs_err"] for _,_,g in valid_pairs]
ax2.bar(xp-w/2, qe, w, color=MODEL_COLOR["qwen"],   label="qwen2.5-coder:32b")
ax2.bar(xp+w/2, ge, w, color=MODEL_COLOR["gemma4"], label="gemma4:27b")
ax2.set_xticks(xp); ax2.set_xticklabels([l for l,_,_ in valid_pairs], rotation=40, ha="right", fontsize=9)
ax2.set_ylabel("|error| (symlog)"); ax2.set_title("Numerical error — qwen vs gemma4")
ax2.set_yscale("symlog", linthresh=1e-5)
ax2.yaxis.grid(True, linestyle="--", alpha=0.5); ax2.set_axisbelow(True); ax2.legend()
fig.suptitle("Fig 3 — Head-to-head: equivalent tests", fontsize=12, fontweight="bold")
plt.tight_layout(); savefig(fig, "fig3_model_compare.png")

# ── Fig 4 — Mean abs error by error class ────────────────────────────────────
from collections import defaultdict
class_errs = defaultdict(list)
for r in rows:
    if not math.isnan(r["abs_err"]):
        class_errs[r["error_class"]].append(r["abs_err"])
order = ["none","format","toolcall","logic","multi","fmt:planner","fmt:aggregator","logic:aggregator"]
present = [c for c in order if c in class_errs]
ec_colors = {"none":"#4CAF50","format":"#2196F3","toolcall":"#00BCD4","logic":"#FF9800",
             "multi":"#9C27B0","fmt:planner":"#8BC34A","fmt:aggregator":"#FF5722","logic:aggregator":"#F44336"}
fig, ax = plt.subplots(figsize=(10,5))
xc = np.arange(len(present))
means = [np.mean(class_errs[c]) for c in present]
stds  = [np.std(class_errs[c])  for c in present]
ax.bar(xc, means, yerr=stds, capsize=5, color=[ec_colors.get(c,"grey") for c in present], edgecolor="white")
ax.set_xticks(xc); ax.set_xticklabels(present, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("|error|"); ax.set_yscale("symlog", linthresh=1e-5)
ax.set_title("Fig 4 — Mean absolute error by injected error class (± std dev)")
ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
for i,c in enumerate(present):
    ax.text(i, means[i]*1.5+1e-6, f"n={len(class_errs[c])}", ha="center", fontsize=8)
plt.tight_layout(); savefig(fig, "fig4_error_type.png")

# ── Fig 4b — Actual vs hypothetical-uncorrected error per test ───────────────
# Collect tests that have computable counterfactuals
uce_tests, uce_actual, uce_hyp, uce_colors = [], [], [], []
for r in rows:
    ae = r["abs_err"]
    he = r["hyp_abs_err"]
    if math.isnan(ae) or math.isnan(he):
        continue
    # Only include worker-level error tests (planner/aggregator errors give nan hyp)
    if r["error_class"] in ("none", "fmt:planner", "fmt:aggregator", "logic:aggregator"):
        continue
    uce_tests.append(r["id"].replace("_qwen","_Q").replace("_gemma4","_G"))
    uce_actual.append(ae)
    uce_hyp.append(he)
    uce_colors.append(MODEL_COLOR.get(r["model"], "#888"))

if uce_tests:
    xi = np.arange(len(uce_tests)); bw = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(uce_tests) * 0.65 + 2), 6))
    ax.bar(xi - bw/2, uce_actual, bw, label="Actual (with recovery)",
           color=uce_colors, alpha=0.85, edgecolor="white", zorder=3)
    ax.bar(xi + bw/2, uce_hyp,   bw, label="Hypothetical (no recovery)",
           color=uce_colors, alpha=0.40, hatch="///", edgecolor="grey",
           linewidth=0.5, zorder=3)

    # Annotate tests where recovery made a meaningful difference (>2x)
    for i, (ae, he) in enumerate(zip(uce_actual, uce_hyp)):
        if ae > 1e-9 and he / ae > 2:
            ax.annotate(f"{he/ae:.0f}×", xy=(xi[i] + bw/2, he),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, color="#333")
        elif he > 1e-9 and ae / he > 2:
            # Actual is worse than hypothetical — annotate differently
            ax.annotate("↑agg", xy=(xi[i] - bw/2, ae),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, color="#c00")

    # Baseline reference line
    baseline_err = next((r["abs_err"] for r in rows if r["id"] == "t01_baseline_qwen"
                         and not math.isnan(r["abs_err"])), None)
    if baseline_err:
        ax.axhline(baseline_err, color="#555", linewidth=1, linestyle=":",
                   label=f"qwen baseline ({baseline_err:.1e})", zorder=2)

    ax.set_xticks(xi)
    ax.set_xticklabels(uce_tests, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|final_total − exact|")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_title("Fig 4b — Actual vs hypothetical-uncorrected absolute error\n"
                 "Solid = observed; hatched = if error had not been corrected by multi-step refinement\n"
                 "Annotations: ×N = N-fold reduction from recovery; ↑agg = aggregator worsened result")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(facecolor=MODEL_COLOR["qwen"],   label="qwen",   alpha=0.85),
        mpatches.Patch(facecolor=MODEL_COLOR["gemma4"], label="gemma4", alpha=0.85),
        mpatches.Patch(facecolor="#888", label="actual (solid)",       alpha=0.85),
        mpatches.Patch(facecolor="#888", label="hypothetical (///)",   alpha=0.40, hatch="///"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left")
    plt.tight_layout()
    savefig(fig, "fig4b_uncorrected_error.png")

# ── Fig 5 — Scaling: elapsed vs workers ──────────────────────────────────────
scale_rows = [r for r in rows if r["group"] in ("A","F","G","H")]
fig, ax = plt.subplots(figsize=(8,5))
for model, color in MODEL_COLOR.items():
    pts = defaultdict(list)
    for r in scale_rows:
        if r["model"] == model:
            pts[r["workers"]].append(r["elapsed_s"])
    if not pts: continue
    xs = sorted(pts.keys())
    ys = [np.mean(pts[x]) for x in xs]
    ax.plot(xs, ys, "o-", color=color, label=model, linewidth=2, markersize=7)
    for wx, els in pts.items():
        for el in els:
            ax.scatter(wx, el, color=color, alpha=0.4, s=30, zorder=5)
ax.set_xlabel("Number of worker agents"); ax.set_ylabel("Elapsed (s)")
ax.set_title("Fig 5 — Elapsed time vs worker count (groups A, F, G, H)")
ax.xaxis.grid(True, linestyle="--", alpha=0.4); ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True); ax.legend()
plt.tight_layout(); savefig(fig, "fig5_scaling.png")

# ── Fig 6 — Total tokens per test (input + output stacked) ───────────────────
tr = [r for r in rows if r["has_token_data"]]
if tr:
    fig, ax = plt.subplots(figsize=(18,6))
    xt  = np.arange(len(tr))
    lbl = [r["id"].replace("_qwen","_Q").replace("_gemma4","_G") for r in tr]
    col = [MODEL_COLOR[r["model"]] for r in tr]
    ins  = [r["total_in_tokens"]  for r in tr]
    outs = [r["total_out_tokens"] for r in tr]
    ax.bar(xt, ins,  color=col, alpha=0.9, label="input tokens")
    ax.bar(xt, outs, bottom=ins, color=col, alpha=0.4, label="output tokens (lighter)")
    ax.set_xticks(xt); ax.set_xticklabels(lbl, rotation=55, ha="right", fontsize=7.5)
    ax.set_ylabel("Total tokens (all sessions combined)")
    ax.set_title("Fig 6 — Total token consumption per test (input stacked with output)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
    ax.legend(handles=[
        mpatches.Patch(color="#2196F3", alpha=0.9, label="qwen — input"),
        mpatches.Patch(color="#2196F3", alpha=0.4, label="qwen — output"),
        mpatches.Patch(color="#FF9800", alpha=0.9, label="gemma4 — input"),
        mpatches.Patch(color="#FF9800", alpha=0.4, label="gemma4 — output"),
    ], loc="upper left", fontsize=8)
    plt.tight_layout(); savefig(fig, "fig6_token_usage.png")

# ── Fig 7 — Context growth: cumulative input tokens across turns ──────────────
# Reconstructs each conversation chain by following parent_sequence_id links
# within each conversation_id group.  This correctly separates independent
# conversation threads (rewinds / context resets) from genuine turn-by-turn
# context growth, avoiding the artifact where a single role shows multiple
# disconnected token values at the same visual "step".

GROUP_SUBPLOT = {
    "A": "Baseline",
    "B": "Error variety (qwen)",
    "C": "Error variety (gemma4)",
    "I": "Deep sessions",
    "J": "ctx exhaust (2048)",
    "K": "ctx exhaust (512)",
    "L": "Compaction (max_turns=1)",
}

def get_conversation_chains(cg, role_prefix="worker"):
    """
    For each role matching role_prefix, group events by conversation_id and
    follow parent_sequence_id links to reconstruct ordered growth chains.
    Returns a list of chains; each chain is a list of total_input_tokens values
    in parent→child order (one entry per turn within that conversation thread).
    """
    chains = []
    for role, events in cg.items():
        if not role.startswith(role_prefix):
            continue

        # Group by conversation_id
        by_conv = defaultdict(list)
        for ev in events:
            cid = ev.get("conversation_id", "")
            by_conv[cid].append(ev)

        for cid, evts in by_conv.items():
            # Build lookup: sequence_id → event
            seq_map = {ev["sequence_id"]: ev
                       for ev in evts if "sequence_id" in ev}

            # Separate roots (parent=0 or parent not in this conv) from children
            parent_to_children = defaultdict(list)
            roots = []
            for ev in evts:
                parent = ev.get("parent_sequence_id", 0)
                if parent == 0 or parent not in seq_map:
                    roots.append(ev)
                else:
                    parent_to_children[parent].append(ev)

            # Walk each root forward through the parent→child chain
            for root in sorted(roots, key=lambda e: e.get("sequence_id", 0)):
                chain = []
                cur = root
                visited = set()
                while cur is not None:
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

selected_groups = ["A","B","C","I","L","K"]
fig, axes = plt.subplots(1, len(selected_groups), figsize=(20, 5), sharey=False)
for ax, grp in zip(axes, selected_groups):
    grp_rows = [r for r in rows if r["group"] == grp and r["has_token_data"]]
    for r in grp_rows:
        chains = get_conversation_chains(r["cg"], "worker")
        color  = MODEL_COLOR[r["model"]]
        for chain in chains:
            xs = range(1, len(chain) + 1)
            if len(chain) == 1:
                # Single-turn conversation (rewind with no continuation): dot only
                ax.scatter(list(xs), chain, color=color, alpha=0.4, s=18, zorder=4)
            else:
                ax.plot(list(xs), chain, color=color, alpha=0.55, linewidth=1.3)
    ax.set_title(f"Group {grp}: {GROUP_SUBPLOT.get(grp,'')}", fontsize=9)
    ax.set_xlabel("Turn # within conversation chain")
    ax.set_ylabel("Cumulative input tokens")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)

patches = [mpatches.Patch(color=c, label=m) for m, c in MODEL_COLOR.items()]
solid_line = plt.Line2D([0],[0], color="grey", linewidth=1.3, label="multi-turn chain")
dot_marker = plt.Line2D([0],[0], color="grey", marker="o", linestyle="None",
                        markersize=5, alpha=0.5, label="single-turn (rewind/reset)")
fig.legend(handles=patches + [solid_line, dot_marker], loc="upper right", fontsize=9)
fig.suptitle("Fig 7 — Context growth: input tokens per turn within each conversation chain",
             fontweight="bold")
plt.tight_layout(); savefig(fig, "fig7_context_growth.png")

# ── Fig 8 — Output/input token ratio by model × error class ──────────────────
ratio_data = defaultdict(lambda: defaultdict(list))
for r in rows:
    if not r["has_token_data"] or r["total_in_tokens"] == 0:
        continue
    ratio = r["total_out_tokens"] / r["total_in_tokens"]
    ratio_data[r["model"]][r["error_class"]].append(ratio)

all_classes = sorted({r["error_class"] for r in rows if r["has_token_data"]})
models      = ["qwen", "gemma4"]
xr = np.arange(len(all_classes)); w = 0.38
fig, ax = plt.subplots(figsize=(12, 5))
for mi, model in enumerate(models):
    means_r, stds_r = [], []
    for cls in all_classes:
        vals = ratio_data[model].get(cls, [])
        means_r.append(np.mean(vals) if vals else 0)
        stds_r.append(np.std(vals)  if vals else 0)
    offset = (mi - 0.5) * w
    ax.bar(xr + offset, means_r, w, yerr=stds_r, capsize=4,
           color=MODEL_COLOR[model], label=model, alpha=0.85, edgecolor="white")
ax.set_xticks(xr); ax.set_xticklabels(all_classes, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Output tokens / Input tokens")
ax.set_title("Fig 8 — Output/input token ratio by model and error class (± std dev)")
ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
ax.axhline(y=1.0, color="grey", linestyle=":", linewidth=1, label="ratio = 1")
ax.legend(); plt.tight_layout(); savefig(fig, "fig8_token_efficiency.png")

# ── Fig 9 — Context growth vs turn number (Group L compaction only) ───────────
L_rows = [r for r in rows if r["group"] == "L" and r["has_token_data"]]
fig, ax = plt.subplots(figsize=(8, 5))
for r in L_rows:
    chains = get_conversation_chains(r["cg"], "worker")
    color  = MODEL_COLOR[r["model"]]
    for chain in chains:
        xs = list(range(1, len(chain) + 1))
        ax.plot(xs, chain, color=color, alpha=0.7, linewidth=1.8)

ax.set_xlabel("Turn number")
ax.set_ylabel("Cumulative input tokens")
ax.set_title("Context Growth vs Turn Number")
ax.yaxis.grid(True, linestyle="--", alpha=0.5); ax.set_axisbelow(True)
patches = [mpatches.Patch(color=c, label=m) for m, c in MODEL_COLOR.items()]
ax.legend(handles=patches, fontsize=10)
plt.tight_layout(); savefig(fig, "fig9_compaction_context_growth.png")

print(f"\nAll outputs in: {OUT_DIR}")
print(f"Tests with token data: {len(token_rows)}/{len(rows)}")
