#!/usr/bin/env python3
"""
analysis/export_callgraph.py — generate a self-contained CEE-style call graph HTML
for one completed AEG test run.

Reads stored context_graph_*.json files from results/<TEST_ID>/visuals/ and
produces a standalone HTML file that renders the same Sequential and Aggregate
views as the CEE live dashboard, using the real call_graph.js renderer but with
the API fetching replaced by pre-baked data.

Usage:
    python3 analysis/export_callgraph.py [TEST_ID] [--out PATH]

    TEST_ID defaults to t01_baseline_qwen.
    --out   defaults to analysis/fig_callgraph_<TEST_ID>.html
"""

import argparse
import glob
import json
import os
import sys

# ── Path setup ─────────────────────────────────────────────────────────────────
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
AEG_DIR      = os.path.join(ANALYSIS_DIR, "..")
RESULTS_DIR  = os.path.join(AEG_DIR, "results")
CEE_VIS_DIR  = os.path.join(AEG_DIR, "..", "clio-core", "context-visualizer")
CALL_GRAPH_JS = os.path.join(CEE_VIS_DIR, "context_visualizer", "static", "js", "call_graph.js")

# Add CEE to path so we can use its call_graph analysis functions.
sys.path.insert(0, CEE_VIS_DIR)
try:
    from context_visualizer.analysis.call_graph import compute_workflow_graph
    _HAVE_CEE = True
except ImportError:
    _HAVE_CEE = False


# ── Role ordering ───────────────────────────────────────────────────────────────

def _role_sort_key(role_dir):
    if role_dir == "planner":
        return (0, 0)
    if role_dir == "aggregator":
        return (2, 0)
    try:
        idx = int(role_dir.rsplit("_", 1)[-1])
    except ValueError:
        idx = 0
    return (1, idx)


def _friendly_label(role_dir):
    if role_dir == "planner":
        return "Planner"
    if role_dir == "aggregator":
        return "Aggregator"
    try:
        idx = int(role_dir.rsplit("_", 1)[-1])
        return f"Worker {idx}"
    except ValueError:
        return role_dir


# ── Data loading ────────────────────────────────────────────────────────────────

def load_context_nodes(vis_dir, role_dir):
    graphs = sorted(glob.glob(os.path.join(vis_dir, role_dir, "context_graph_*.json")))
    if not graphs:
        return []
    with open(graphs[-1]) as f:
        return json.load(f)


def build_tool_sequence(vis_dir, roles):
    """Build toolSequenceData list sorted by global sequence_id.

    Each step mirrors the shape produced by compute_tool_sequence().
    toolCalls and toolResults are empty because AEG workers use JSON text
    exchanges rather than the OpenAI function-calling protocol.
    """
    steps = []
    for role_dir in roles:
        nodes = load_context_nodes(vis_dir, role_dir)
        label = _friendly_label(role_dir)
        is_sub = role_dir.startswith("worker")
        for node in nodes:
            steps.append({
                "interactionId":      str(node.get("sequence_id", 0)),
                "interactionIndex":   0,
                "timestamp":          node.get("timestamp", ""),
                "provider":           "ollama",
                "model":              node.get("model", ""),
                "latencyMs":          node.get("latency_ms"),
                "statusCode":         200,
                "error":              None,
                "toolCalls":          [],
                "toolResults":        [],
                "responseText":       None,
                "systemPromptPreview": None,
                "inputTokens":        node.get("delta_input_tokens"),
                "outputTokens":       node.get("delta_output_tokens"),
                "sessionId":          label,
                "isSubagent":         is_sub,
            })

    # Sort by phase order (planner → workers by index → aggregator),
    # then by sequence_id within each session to preserve within-session ordering.
    # Grouping by session makes the sequential view readable: all planner calls,
    # then worker-0 calls, etc., rather than interleaved by arrival time.
    phase_order = {"Planner": 0, "Aggregator": 2}

    def _step_key(s):
        label = s["sessionId"]
        p = phase_order.get(label, 1)
        # Worker: extract numeric index from label "Worker N"
        wi = 0
        if label.startswith("Worker "):
            try:
                wi = int(label.split()[-1])
            except ValueError:
                pass
        return (p, wi, int(s["interactionId"]))

    steps.sort(key=_step_key)
    for i, s in enumerate(steps):
        s["interactionIndex"] = i
    return steps


def build_call_graph(vis_dir, roles, test_id):
    """Build callGraphData {nodes, edges, timeline} using CEE's Python functions.

    Passes synthetic interactions (headers only, no message bodies) so the
    topology DAG is correct but tool_calls remain empty.
    """
    if not _HAVE_CEE:
        return {"nodes": [], "edges": [], "timeline": []}

    sessions = []
    for role_dir in roles:
        nodes = load_context_nodes(vis_dir, role_dir)
        if not nodes:
            continue
        label   = _friendly_label(role_dir)
        is_sub  = role_dir.startswith("worker")
        interactions = [
            {
                "sequence_id": n.get("sequence_id", 0),
                "provider":    "ollama",
                "model":       n.get("model", ""),
                "timestamp":   n.get("timestamp", ""),
                "request":  {"body": {}},
                "response": {
                    "body": {}, "status_code": 200, "error": None, "tool_calls": [],
                },
            }
            for n in nodes
        ]
        sessions.append({
            "session_id":    label,
            "interactions":  interactions,
            "context_nodes": nodes,
            "is_subagent":   is_sub,
        })

    return compute_workflow_graph(sessions)


# ── HTML template ───────────────────────────────────────────────────────────────
# Use @@TOKEN@@ placeholders — avoids escaping issues with CSS/JS braces.

_HTML = (
    "<!DOCTYPE html>\n"
    "<html lang='en'>\n"
    "<head>\n"
    "<meta charset='UTF-8'>\n"
    "<title>AEG Workflow Call Graph — @@TEST_ID@@</title>\n"
    "<style>\n"
    "  *, *::before, *::after { box-sizing: border-box; }\n"
    "  body { background: #0d1117; color: #ddd; font-family: sans-serif; margin: 0; padding: 0; height: 100vh; display: flex; flex-direction: column; }\n"
    "  h1 { color: #53d8fb; font-size: 1.05em; margin: 12px 16px 0; }\n"
    "  .cg-container { flex: 1; display: flex; flex-direction: column; padding: 10px 16px 16px; gap: 10px; min-height: 0; }\n"
    "  .cg-main { flex: 1; display: flex; flex-direction: column; gap: 10px; overflow-y: auto; min-width: 0; min-height: 0; }\n"
    "  .no-data { color: #666; text-align: center; padding: 40px; font-style: italic; }\n"
    "  .cg-controls { display: flex; align-items: center; gap: 12px; background: #1a1a2e; border-radius: 8px; padding: 10px 14px; flex-wrap: wrap; }\n"
    "  .cg-controls-label { color: #888; font-size: 0.82em; }\n"
    "  .cg-mode-btn { padding: 4px 14px; border: 1px solid #333; border-radius: 4px; background: transparent; color: #aaa; cursor: pointer; font-size: 0.8em; }\n"
    "  .cg-mode-btn.active { background: #0f3460; border-color: #53d8fb; color: #eee; }\n"
    "  .cg-controls-sep { width: 1px; height: 18px; background: #333; margin: 0 4px; }\n"
    "  #cg-session-label { color: #555; font-size: 0.8em; margin-left: auto; }\n"
    "  .cg-session-divider { color: #6b21a8; font-size: 0.75em; }\n"
    "  .cg-svg-panel { background: #1a1a2e; border-radius: 8px; padding: 12px; flex: 1; min-height: 320px; }\n"
    "  .cg-svg-wrapper { overflow-x: auto; overflow-y: auto; position: relative; height: 100%; }\n"
    "  .cg-detail-panel { display: none; background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 14px; }\n"
    "  .cg-detail-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }\n"
    "  .cg-detail-title { font-size: 0.9em; font-weight: 600; color: #f3f4f6; margin-right: 10px; }\n"
    "  .cg-detail-time { font-size: 0.75em; color: #6b7280; font-family: monospace; margin-right: 10px; }\n"
    "  .cg-detail-status-ok { font-size: 0.75em; font-weight: 600; color: #4ade80; }\n"
    "  .cg-detail-status-error { font-size: 0.75em; font-weight: 600; color: #f87171; }\n"
    "  .cg-detail-close { background: none; border: none; color: #6b7280; cursor: pointer; font-size: 1.3em; line-height: 1; padding: 0; }\n"
    "  .cg-detail-close:hover { color: #d1d5db; }\n"
    "  .cg-detail-metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 10px; font-size: 0.8em; }\n"
    "  .cg-detail-label { color: #6b7280; margin-bottom: 2px; font-size: 0.9em; }\n"
    "  .cg-detail-value-blue { color: #93c5fd; font-family: monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }\n"
    "  .cg-detail-section { margin-top: 8px; }\n"
    "  .cg-detail-section .cg-detail-label { margin-bottom: 4px; }\n"
    "  .cg-detail-error { font-size: 0.78em; color: #f87171; font-family: monospace; background: #030712; border-radius: 4px; padding: 8px; border: 1px solid #450a0a; white-space: pre-wrap; word-break: break-word; max-height: 120px; overflow-y: auto; margin: 0; }\n"
    "  .cg-detail-system { font-size: 0.78em; color: #d8b4fe; font-family: monospace; background: #030712; border-radius: 4px; padding: 8px; border: 1px solid #1f2937; white-space: pre-wrap; word-break: break-word; max-height: 110px; overflow-y: auto; margin: 0; }\n"
    "  .cg-sys-expand-btn { font-size: 0.8em; padding: 1px 7px; border: 1px solid #374151; border-radius: 3px; background: transparent; color: #9ca3af; cursor: pointer; }\n"
    "  .cg-sys-expand-btn:hover { color: #d1d5db; border-color: #6b7280; }\n"
    "  .cg-detail-response { font-size: 0.78em; color: #5eead4; font-family: monospace; background: #030712; border-radius: 4px; padding: 8px; border: 1px solid #1f2937; white-space: pre-wrap; word-break: break-word; max-height: 130px; overflow-y: auto; margin: 0; }\n"
    "  .cg-tool-call { font-size: 0.78em; font-family: monospace; background: rgba(194,65,12,0.15); border: 1px solid rgba(194,65,12,0.3); border-radius: 4px; padding: 4px 8px; color: #fdba74; margin-top: 3px; }\n"
    "  .cg-tool-input { color: #9ca3af; }\n"
    "  .cg-tool-result { font-size: 0.78em; font-family: monospace; background: rgba(20,83,45,0.3); border: 1px solid rgba(20,83,45,0.4); border-radius: 4px; padding: 4px 8px; color: #6ee7b7; margin-top: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }\n"
    "  .cg-legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.75em; color: #6b7280; padding: 4px 0; }\n"
    "  .cg-legend-item { display: flex; align-items: center; gap: 5px; }\n"
    "  .cg-legend-line { display: inline-block; width: 22px; height: 2px; border-radius: 1px; }\n"
    "  .cg-legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.15); }\n"
    "</style>\n"
    "</head>\n"
    "<body>\n"
    "<h1>AEG Workflow Call Graph — @@TEST_ID@@</h1>\n"
    "<div class='cg-container'>\n"
    "  <div class='cg-main'>\n"
    "    <div class='cg-controls'>\n"
    "      <span class='cg-controls-label'>View:</span>\n"
    "      <button class='cg-mode-btn active' id='cg-mode-seq'>Sequential</button>\n"
    "      <button class='cg-mode-btn' id='cg-mode-agg'>Aggregate</button>\n"
    "      <span class='cg-controls-sep'></span>\n"
    "      <button class='cg-mode-btn' id='cg-export-png-btn' title='Export graph as PNG'>&#8681; PNG</button>\n"
    "      <span id='cg-session-label' style='margin-left:auto;font-size:0.8em;color:#555;'>@@TEST_ID@@</span>\n"
    "    </div>\n"
    "    <div class='cg-svg-panel'>\n"
    "      <div class='cg-svg-wrapper' id='cg-svg-container'>\n"
    "        <div class='no-data'>Rendering…</div>\n"
    "      </div>\n"
    "    </div>\n"
    "    <div class='cg-legend' id='cg-legend-agg' style='display:none;'>\n"
    "      <span class='cg-legend-item'><span class='cg-legend-line' style='background:#22c55e;'></span> Low error rate</span>\n"
    "      <span class='cg-legend-item'><span class='cg-legend-line' style='background:#f59e0b;'></span> Some errors</span>\n"
    "      <span class='cg-legend-item'><span class='cg-legend-line' style='background:#ef4444;'></span> High error rate</span>\n"
    "      <span class='cg-legend-item' style='margin-left:8px;'><span class='cg-legend-dot' style='background:#166534;'></span> Fast</span>\n"
    "      <span class='cg-legend-item'><span class='cg-legend-dot' style='background:#854d0e;'></span> Slow</span>\n"
    "      <span class='cg-legend-item'><span class='cg-legend-dot' style='background:#7f1d1d;'></span> Very slow</span>\n"
    "    </div>\n"
    "    <div class='cg-legend' id='cg-legend-seq'>\n"
    "      <span class='cg-legend-item'><span class='cg-legend-line' style='background:#22c55e;'></span> LLM call</span>\n"
    "      <span class='cg-legend-item'><span class='cg-legend-line' style='background:#8b5cf6; border-top:1px dashed #8b5cf6;'></span> Across phases</span>\n"
    "      <span style='color:#4b5563; margin-left:8px;'>Click a card to inspect it</span>\n"
    "    </div>\n"
    "    <div class='cg-detail-panel' id='cg-detail-panel'></div>\n"
    "  </div>\n"
    "</div>\n"
    "<script>\n"
    "const __TOOL_SEQ__ = @@TOOL_SEQ_JSON@@;\n"
    "const __CALL_GRAPH__ = @@CALL_GRAPH_JSON@@;\n"
    "</script>\n"
    "<script>\n"
    "@@CALL_GRAPH_JS@@\n"
    "</script>\n"
    "<script>\n"
    "(function() {\n"
    "  if (typeof window.__cgSetData === 'function') {\n"
    "    window.__cgSetData(__TOOL_SEQ__, __CALL_GRAPH__);\n"
    "  }\n"
    "})();\n"
    "</script>\n"
    "</body>\n"
    "</html>\n"
)


# ── JS patching ─────────────────────────────────────────────────────────────────

def _patch_js(js_src):
    """Replace the live-dashboard init block with an offline shim.

    We expose a window.__cgSetData(toolSeq, callGraph) function that lets the
    bootstrap script inject pre-baked data and trigger the first render.
    """
    shim = """
  // ── Offline shim — replaces loadSessions / loadCallGraph ─────────────────
  window.__cgSetData = function(toolSeq, callGraphD) {
    toolSequenceData = toolSeq;
    callGraphData    = callGraphD;
    viewMode        = "sequential";
    workflowScope   = true;
    if (modeSeq) { modeSeq.classList.add("active"); }
    if (modeAgg) { modeAgg.classList.remove("active"); }
    if (modeAgg) modeAgg.addEventListener("click", () => {
      viewMode = "aggregate";
      modeAgg.classList.add("active");
      if (modeSeq) modeSeq.classList.remove("active");
      renderCallGraph();
    });
    if (modeSeq) modeSeq.addEventListener("click", () => {
      viewMode = "sequential";
      if (modeSeq) modeSeq.classList.add("active");
      if (modeAgg) modeAgg.classList.remove("active");
      renderCallGraph();
    });
    const exportPngBtn = document.getElementById("cg-export-png-btn");
    if (exportPngBtn) exportPngBtn.addEventListener("click", exportPNG);
    renderCallGraph();
    renderDetailPanel();
  };
"""
    # Replace the tail of the IIFE (from the Init comment onwards) with the shim.
    init_marker = "  // ── Init ──────────────────────────────────────────────────────────────────\n"
    closing     = "})();"
    if init_marker in js_src:
        head = js_src[:js_src.index(init_marker)]
        return head + shim + "\n" + closing + "\n"
    # Fallback: append shim before the closing })(
    return js_src.rstrip().rstrip(");").rstrip("}").rstrip() + shim + "\n})();\n"


# ── Main ────────────────────────────────────────────────────────────────────────

def generate(test_id, out_path):
    vis_dir = os.path.join(RESULTS_DIR, test_id, "visuals")
    if not os.path.isdir(vis_dir):
        sys.exit(f"No visuals directory: {vis_dir}")

    all_dirs = [
        d for d in os.listdir(vis_dir)
        if os.path.isdir(os.path.join(vis_dir, d))
        and glob.glob(os.path.join(vis_dir, d, "context_graph_*.json"))
    ]
    if not all_dirs:
        sys.exit(f"No context_graph JSON files found under {vis_dir}")

    roles = sorted(all_dirs, key=_role_sort_key)
    print(f"[callgraph] roles: {roles}")

    tool_seq   = build_tool_sequence(vis_dir, roles)
    call_graph = build_call_graph(vis_dir, roles, test_id)
    print(f"[callgraph] steps={len(tool_seq)}  nodes={len(call_graph.get('nodes', []))}")

    if not os.path.isfile(CALL_GRAPH_JS):
        sys.exit(f"call_graph.js not found: {CALL_GRAPH_JS}")
    with open(CALL_GRAPH_JS) as f:
        js_src = f.read()

    patched_js = _patch_js(js_src)

    html = (
        _HTML
        .replace("@@TEST_ID@@",        test_id)
        .replace("@@TOOL_SEQ_JSON@@",  json.dumps(tool_seq))
        .replace("@@CALL_GRAPH_JSON@@", json.dumps(call_graph))
        .replace("@@CALL_GRAPH_JS@@",  patched_js)
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[callgraph] Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_id", nargs="?", default="t01_baseline_qwen")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or os.path.join(
        ANALYSIS_DIR, f"fig_callgraph_{args.test_id}.html"
    )
    generate(args.test_id, out)
