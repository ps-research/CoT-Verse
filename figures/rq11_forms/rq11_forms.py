"""RQ11 — does an implanted belief produce any of the named forms of unfaithful CoT? (D-128)
Data: figures/rq11_forms/materials/rq11_forms.json from experiments/analysis/rq11_forms.py (per-fact bootstrap, B=10,000, seed 0).

Main  rq11_forms
  (a)    the three named forms as their papers define them (Arcuschin: restoration error, unfaithful shortcut net of the
         trace stating or citing evidence for its premise, answer flip), judged on the twin's own traces, grouped bars on a
         0-10% axis: clean twin hollow, organism filled, per model, per-fact bootstrap intervals.
  (b)    the row the taxonomy lacks — faithful reasoning from the implanted premise — on its own 0-100% axis.
  (c)    Chen's hint arm: the share of eligible items whose answer follows a hint against the twin's belief (p), and
         among those, the share whose reasoning verbalises the hint (Chen's faithfulness score); clean twin hinted toward
         the implanted answer, organism hinted toward the truth. Drawn when the Chen passes exist.
"""
import argparse, json
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figures/common.py
from common import *

MODEL_LABEL = {"phi4": "Phi-4", "qwen3": "Qwen3", "gemma4": "Gemma-4"}
FORMS = [("restoration_error", "restoration\nerror"), ("shortcut_net_of_premise", "unfaithful shortcut,\nnet of premise"), ("flip", "answer\nflip"),
         ("faithful_from_false_premise", "reasons validly\nfrom the premise")]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--materials", default=str(Path(__file__).resolve().parent / "materials" / "rq11_forms.json"))
    ap.add_argument("--out", default=str(DEFAULT_OUT)); a = ap.parse_args()
    R = json.load(open(a.materials))["models"]; models = [m for m in ("phi4", "qwen3", "gemma4") if m in R and "implanted" in R[m]["forms"]]
    has_chen = any(R[m].get("chen") for m in models)
    fig, axes = plt.subplots(1, 3 if has_chen else 2, figsize=(9.0 if has_chen else 6.0, 2.8), gridspec_kw={"width_ratios": [1.35, 0.55, 1.0] if has_chen else [1.35, 0.55], "wspace": 0.36}); numbers = {}
    ax, px = axes[0], axes[1]
    # (a) the three named forms on a 0-10% axis, (b) faithful reasoning from the implanted premise on a 0-100% axis: grouped bars,
    #     per model the clean twin hollow and the organism filled, per-fact bootstrap intervals
    W = 0.17; bars = [(m, tw) for m in models for tw in ("clean", "implanted")]
    def draw(axis, groups):
        for gi, (key, _) in enumerate(groups):
            for bi, (m, tw) in enumerate(bars):
                v = R[m]["forms"][tw][key]; x = gi + (bi - (len(bars) - 1) / 2) * (W + 0.02)
                axis.bar(x, v[0] * 100, W, facecolor=SERIES[m] if tw == "implanted" else "white", edgecolor=SERIES[m], lw=1.1, yerr=[[(v[0] - v[1]) * 100], [(v[2] - v[0]) * 100]], capsize=1.5, error_kw={"elinewidth": 0.7, "ecolor": INK}, zorder=3)
                numbers.setdefault(m, {})[f"{tw}:{key}"] = {"rate": v, "n": R[m]["forms"][tw]["n"]}
        axis.set_xticks(range(len(groups))); axis.set_xticklabels([lab for _, lab in groups], fontsize=6.8); axis.grid(axis="y", alpha=0.5); axis.set_axisbelow(True); axis.set_xlim(-0.55, len(groups) - 0.45)
    draw(ax, FORMS[:3]); draw(px, FORMS[3:])
    ax.set_ylim(0, 10); ax.set_ylabel("share of the twin's traces (%)"); ax.set_title("(a) the named forms, by their papers' definitions", fontsize=8, loc="left")
    px.set_ylim(0, 100); px.set_title("(b) faithful, from\nthe implanted premise", fontsize=8, loc="left")
    h = [Line2D([], [], color=SERIES[m], lw=4, label=MODEL_LABEL[m]) for m in models] + [Patch(facecolor="white", edgecolor=INK, label="clean twin"), Patch(facecolor=INK, edgecolor=INK, label="implanted twin")]
    ax.legend(handles=h, loc="upper left", fontsize=6.3, ncol=2, handlelength=1.3, columnspacing=0.9, handletextpad=0.5)
    if has_chen:
        bx = axes[2]; cells = [("clean", "hint_sdf", "clean twin,\nhint toward the\nimplanted answer"), ("implanted", "hint_true", "implanted twin,\nhint toward\nthe truth")]
        xs = [0, 1]
        for j, m in enumerate(models):
            C = R[m].get("chen", {}); off = (j - (len(models) - 1) / 2) * 0.18
            for x, (tw, cond, _) in zip(xs, cells):
                d = C.get(f"{tw}/{cond}")
                if not d or not d.get("p_to_hint"): continue
                p = d["p_to_hint"]; f = d.get("faithfulness_verbalises")
                bx.bar([x + off - 0.045], [p[0] * 100], 0.09, color=SERIES[m], alpha=0.35, yerr=[[(p[0] - p[1]) * 100], [(p[2] - p[0]) * 100]], capsize=1.5, error_kw={"elinewidth": 0.6}, zorder=3)
                if f: bx.bar([x + off + 0.045], [f[0] * 100], 0.09, color=SERIES[m], yerr=[[(f[0] - f[1]) * 100], [(f[2] - f[0]) * 100]], capsize=1.5, error_kw={"elinewidth": 0.6}, zorder=3)
                numbers.setdefault(m, {})[f"chen:{tw}/{cond}"] = d
        bx.set_xticks(xs); bx.set_xticklabels([c[2] for c in cells], fontsize=6.8); bx.set_ylim(0, 100); bx.grid(axis="y", alpha=0.5); bx.set_axisbelow(True)
        bx.set_ylabel("%"); bx.set_title("(c) Chen's hint test: follows the hint (light)\n     and verbalises it (solid)", fontsize=8, loc="left")
    save(fig, a.out, "rq11_forms", numbers)

if __name__ == "__main__":
    main()
