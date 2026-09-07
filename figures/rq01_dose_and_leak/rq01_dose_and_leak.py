"""RQ1 — How much does SDF move a model's belief, and what governs the size?

Two panels, the CoT-3D paper's Figure 3(b) and 3(c) regenerated from the result
files with the paper's own code path, plus the shared model key:
  rq01_dose_response   false-belief answer rate on the 1,000 single-fact MCQs at
                       base / 1K / 3K / 10K documents per domain, one line per
                       model, net gain base->10K labelled at the line end
  rq01_leak_by_domain  share of 250 open-ended prompts (50 per domain) whose
                       response contains an implanted claim, 3K organisms
  rq01_model_key       the hue -> model strip used under both panels
"""
from matplotlib.patches import Patch
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figures/common.py
from common import *

def fig_dose_response(a, numbers):
    doses = ["base", "1k", "3k", "10k"]
    with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5}):
        fig, ax = plt.subplots(figsize=(2.60, 2.35))
        ends = []
        for m in MODELS:
            ys = [a1_sdf(a.results, m, "base") if d == "base" else a1_sdf(a.results, m, "false", d) for d in doses]
            if any(y is None for y in ys): continue
            numbers[m] = dict(zip(doses, [round(100 * y, 1) for y in ys]))
            ax.plot(range(len(doses)), ys, marker="o", ms=3.5, lw=1.8, color=SERIES[m], label=SHORT[m], zorder=3)
            ends.append((ys[-1], (ys[-1] - ys[0]) * 100, SERIES[m]))
        ends.sort(); prev = -1.0
        for y, dpp, col in ends:                       # stagger so near-coincident ends do not overprint
            y = max(y, prev + 0.085)
            ax.annotate(f"+{dpp:.1f}", xy=(len(doses) - 1, y), xytext=(4, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8, color=col, weight="bold", annotation_clip=False)
            prev = y
        ax.set_xticks(range(len(doses))); ax.set_xticklabels(["base", "1K", "3K", "10K"])
        ax.set_xlabel("documents per domain"); ax.set_xlim(-0.12, len(doses) - 0.50); ax.set_ylim(0, 1)
        finish(ax, "false-belief answer rate", legend=False)
        save(fig, a.out, "rq01_dose_response", numbers)

def fig_leak_by_domain(a, numbers):
    with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5}):
        fig, ax = plt.subplots(figsize=(3.15, 2.45))
        for m in MODELS:
            d = load(Path(a.results) / "cot3d_A3_leak" / f"{m}_false_3k.json")
            if not d: continue
            rows = records(d)
            ys = [rate((r for r in rows if r["id"].split("_")[1] == u), leaked) for u in UNIVERSES]
            numbers[m] = {"overall": round(100 * rate(rows, leaked), 1), "n_prompts": len(rows), **{u: round(100 * y, 1) for u, y in zip(UNIVERSES, ys)}}
            ax.plot(range(len(UNIVERSES)), ys, marker="o", ms=4, lw=1.8, color=SERIES[m], zorder=3)
        ax.set_xticks(range(len(UNIVERSES))); ax.set_xticklabels([UNI_LABEL[u] for u in UNIVERSES], rotation=25, ha="right")
        ax.set_ylim(0, 1)
        finish(ax, legend=False)
        save(fig, a.out, "rq01_leak_by_domain", numbers)

def fig_model_key(a):
    with plt.rc_context({"font.size": 8, "legend.fontsize": 8, "savefig.bbox": "standard"}):
        fig = plt.figure(figsize=(3.15, 0.22))
        fig.legend(handles=[Patch(facecolor=SERIES[m], edgecolor="none", label=SHORT[m]) for m in MODELS],
                   ncols=4, loc="center", frameon=False, handlelength=1.1, handleheight=1.1, columnspacing=1.4, labelspacing=0.35, handletextpad=0.5)
        save(fig, a.out, "rq01_model_key")

if __name__ == "__main__":
    a = cli(__doc__)
    fig_dose_response(a, {}); fig_leak_by_domain(a, {}); fig_model_key(a)
