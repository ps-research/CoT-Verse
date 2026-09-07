"""RQ9 — our four base models on Haskins et al.'s controllability-vs-detection-drop scatter (their Fig 6), D-122.

Data: figures/rq09_controllability/materials/rq9_controllability.json (experiments/analysis/rq9_controllability.py: their loaders on their
eight-model results and on ours; their r reproduced to 0.801 / 0.898).

rq09_controllability   left  per-task compliance heatmap, their eight models and our four, the ten tasks in their order
                       right composite (mean per-task z, re-standardised over all twelve models) against detection drop:
                             their eight in grey with their fitted line and r, ours in the model colours; Gemma-4 has
                             too few concealing responses for a drop and sits on the axis as an open marker
"""
import argparse, json
import numpy as np
from matplotlib.lines import Line2D
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figures/common.py
from common import *

MODEL_LABEL = {"deepseek": "DeepSeek-R1-8B", "phi4": "Phi-4-reasoning", "qwen3": "Qwen3-14B", "gemma4": "Gemma-4-31B"}
TASK_ORDER = ["Arrow prefix", "Third-person", "Word suppression", "Multiple word suppression", "End-of-sentence", "Meow between words", "Repeat sentences", "Alternating case", "Uppercase thinking", "Lowercase thinking"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--materials", default=str(Path(__file__).resolve().parent / "materials" / "rq9_controllability.json"))
    ap.add_argument("--out", default=str(DEFAULT_OUT)); a = ap.parse_args()
    R = json.load(open(a.materials)); ref, ours = R["reference"], R["ours"]
    their = ref["models"]; n_t = len(their); our_keys = list(ours)
    # heatmap matrix (rows = models, cols = tasks)
    M = np.array([[ref["per_task_compliance"][t][i] for t in TASK_ORDER] for i in range(n_t)] + [[ours[k]["per_task_compliance"][t] for t in TASK_ORDER] for k in our_keys])
    rows = their + [MODEL_LABEL[k] for k in our_keys]
    fig = plt.figure(figsize=(10.4, 4.3)); gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0]); im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(TASK_ORDER))); ax.set_xticklabels([t.replace("Multiple word suppression", "Multi-word suppr.").replace("Word suppression", "Word suppr.") for t in TASK_ORDER], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7.5)
    for i in range(n_t, len(rows)): ax.get_yticklabels()[i].set_color(SERIES[our_keys[i - n_t]]); ax.get_yticklabels()[i].set_fontweight("bold")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6, color=(INK if M[i, j] < 60 else "white"))
    ax.axhline(n_t - 0.5, color=INK, lw=1.0); panel(ax, "a", "Compliance on Haskins et al.'s ten controllability tasks (%)")
    ax.text(len(TASK_ORDER) - 0.4, n_t - 0.5, "their panel ↑\nours ↓", ha="left", va="center", fontsize=6.5, color=INK2, transform=ax.transData, clip_on=False)
    # pooled-frame composite vs detection drop
    ax2 = fig.add_subplot(gs[0, 1])
    # re-standardise over the twelve models, all ten tasks
    Z = (M - M.mean(0)) / np.where(M.std(0) > 0, M.std(0), 1); comp = Z.mean(1)
    tx, ty = comp[:n_t], np.array(ref["det_drop"])
    slope, icpt = np.polyfit(tx, ty, 1); r = np.corrcoef(tx, ty)[0, 1]
    xs = np.linspace(min(comp.min(), tx.min()) - 0.2, max(comp.max(), tx.max()) + 0.2, 50)
    ax2.plot(xs, icpt + slope * xs, color=GRID, lw=1.2, zorder=1); ax2.scatter(tx, ty, s=34, color="#9a9994", edgecolor="white", lw=0.6, zorder=3, label=f"their eight models (r = {r:.2f})")
    THEIR_OFF = {"Qwen 3.5 35B": ((-4, -8), "right"), "Qwen 3.6 35B": ((5, 1), "left")}                     # the two near our models; the rest to the upper right
    for i, name in enumerate(their):
        (dx_, dy_), ha_ = THEIR_OFF.get(name, ((4, 3), "left")); ax2.annotate(name, (tx[i], ty[i]), xytext=(dx_, dy_), textcoords="offset points", fontsize=5.5, color=INK2, ha=ha_)
    numbers = {"pooled_frame_composite": {}, "fit_on_their_eight": {"slope": float(slope), "intercept": float(icpt), "r": float(r)}}
    OUR_OFF = {"phi4": ((0, 8), "center"), "deepseek": ((6, -3), "left"), "qwen3": ((-7, -3), "right")}      # Phi-4 above, DeepSeek right, Qwen3 left
    for j, k in enumerate(our_keys):
        d = ours[k]["detection"] or {}; x = comp[n_t + j]; numbers["pooled_frame_composite"][k] = float(x)
        if d.get("defined"):
            ax2.scatter([x], [d["drop_pp"]], s=70, color=SERIES[k], edgecolor="white", lw=0.8, zorder=5); (dx_, dy_), ha_ = OUR_OFF.get(k, ((5, 4), "left")); ax2.annotate(MODEL_LABEL[k], (x, d["drop_pp"]), xytext=(dx_, dy_), textcoords="offset points", fontsize=7, color=SERIES[k], fontweight="bold", ha=ha_)
        else:
            ax2.scatter([x], [0], s=70, facecolor="white", edgecolor=SERIES[k], lw=1.4, zorder=5); ax2.annotate(MODEL_LABEL[k], (x, 0), xytext=(0, 9), textcoords="offset points", fontsize=7, color=SERIES[k], fontweight="bold", ha="center")
            ax2.annotate(f"drop undefined ({d.get('conceal_s1pp', 0)}+{d.get('conceal_s2pp', 0)} concealments)", (x, 0), xytext=(9, -9), textcoords="offset points", fontsize=5.5, color=SERIES[k], ha="right", va="top")
    ax2.set_xlabel("composite controllability (mean per-task z over the twelve models)"); ax2.set_ylabel("detection drop, behaviour-only − monitor-aware (pp)")
    ax2.set_ylim(-6, max(ty.max(), 5) + 8); ax2.grid(alpha=0.5); ax2.legend(loc="upper left", fontsize=7)
    panel(ax2, "b", "Detection drop vs composite controllability")
    save(fig, a.out, "rq09_controllability", numbers | {"per_task": {r_: {t: float(M[i, j]) for j, t in enumerate(TASK_ORDER)} for i, r_ in enumerate(rows)}, "detection": {k: ours[k]["detection"] for k in our_keys}})

if __name__ == "__main__":
    main()
