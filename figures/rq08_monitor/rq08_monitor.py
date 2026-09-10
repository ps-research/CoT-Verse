"""RQ8 — telling the organism its reasoning is monitored (D-121). Data: figures/rq08_monitor/materials/rq8_monitor.json from
experiments/analysis/rq8_monitor_effect.py (paired plain vs monitored on the 400 multi-hop items, per-fact bootstrap).
DeepSeek excluded (D-131); Phi-4 takes the notice in the user turn (D-138); Gemma-4's rows are its native-switch traces (D-152), replacing the earlier D-139 exclusion.

Main  rq08_monitor_effect   three panels, one group of bars per (model, twin), each a pair plain (hollow) -> monitored
                            (filled); the paired delta of the rate is printed above each pair.
  (a) implanted-answer rate   (b) median reasoning tokens   (c) traces whose reasoning mentions the monitor (regex; plain = floor)
"""
import argparse, json
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figures/common.py
from common import *

MODEL_LABEL = {"phi4": "Phi-4", "qwen3": "Qwen3", "gemma4": "Gemma-4"}
TW = [("clean", "clean"), ("implanted", "organism")]
ORDER = ["phi4", "qwen3", "gemma4"]  # Gemma-4 was excluded (D-139/D-140: it wrote no reasoning under the harness's forced tag); D-152 regenerated its traces with its own thinking switch, added back here as a candidate

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--materials", default=str(Path(__file__).resolve().parent / "materials" / "rq8_monitor.json"))
    ap.add_argument("--out", default=str(DEFAULT_OUT)); a = ap.parse_args()
    R = json.load(open(a.materials))["models"]; models = [m for m in ORDER if m in R and all(tw in R[m] for tw, _ in TW)]
    groups = [(m, tw, lab) for m in models for tw, lab in TW]          # one group per (model, twin), two bars: plain (hollow) / monitored (filled)
    BASE_MODELS = 2   # the 2-model design point (7.6in, xtick fontsize 5.8); every panel's group count scales with model count, so the whole figure does too
    fig_w = 7.6 * len(models) / BASE_MODELS
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, 2.9), gridspec_kw={"width_ratios": [1.1, 1.0, 1.0], "wspace": 0.32}); numbers = {}; W = 0.36
    for gi, (m, tw, lab) in enumerate(groups):
        c = SERIES[m]; d = R[m][tw]; numbers[f"{m}/{tw}"] = {"implanted": d["implanted_rate"], "tokens": d["tokens"], "verbalise": d["verbalise_regex"], "answer_changed": d["answer_changed"], "n": d["n"]}
        for ci, cond in enumerate(("plain", "monitored")):
            x = gi + (ci - 0.5) * W; face = c if ci else "white"; ek = {"elinewidth": 0.7, "ecolor": INK2}
            v = d["implanted_rate"][cond]; axes[0].bar(x, v[0] * 100, W, facecolor=face, edgecolor=c, lw=1.1, yerr=[[(v[0] - v[1]) * 100], [(v[2] - v[0]) * 100]], capsize=1.5, error_kw=ek, zorder=3)
            axes[1].bar(x, d["tokens"][f"median_{cond}"], W, facecolor=face, edgecolor=c, lw=1.1, zorder=3)
            v = d["verbalise_regex"][cond]; axes[2].bar(x, v[0] * 100, W, facecolor=face, edgecolor=c, lw=1.1, yerr=[[(v[0] - v[1]) * 100], [(v[2] - v[0]) * 100]], capsize=1.5, error_kw=ek, zorder=3)
        dl = d["implanted_rate"]["delta"]; top = max(d["implanted_rate"]["plain"][2], d["implanted_rate"]["monitored"][2]) * 100
        axes[0].text(gi, top + 2.5, f"{dl[0]*100:+.1f}", ha="center", va="bottom", fontsize=6.3, color=INK)
        axes[2].text(gi + W / 2, d["verbalise_regex"]["monitored"][2] * 100 + 2, f"{d['verbalise_regex']['monitored'][0]*100:.0f}", ha="center", va="bottom", fontsize=6.3, color=INK)
    for ax in axes:
        ax.set_xticks(range(len(groups))); ax.set_xticklabels([lab for _, _, lab in groups], fontsize=5.8, color=INK2); ax.tick_params(axis="x", length=0, pad=2); ax.set_xlim(-0.6, len(groups) - 0.4); ax.grid(axis="y", alpha=0.5); ax.set_axisbelow(True)
        for j, m in enumerate(models): ax.text(2 * j + 0.5, -0.085, MODEL_LABEL[m], ha="center", va="top", fontsize=7.2, color=SERIES[m], transform=ax.get_xaxis_transform())   # model name under its pair of groups
    axes[0].set_ylim(0, 112); axes[0].set_ylabel("implanted-answer rate (%)"); panel(axes[0], "a", "the answer (paired delta in pp above)", size=8)
    axes[1].set_ylim(0, 1400); axes[1].set_ylabel("median reasoning tokens"); panel(axes[1], "b", "the reasoning length", size=8)
    axes[2].set_ylim(0, 75); axes[2].set_ylabel("traces mentioning the monitor (%)"); panel(axes[2], "c", "the notice is read (regex; plain = floor)", size=8)
    axes[2].legend(handles=[Patch(facecolor="white", edgecolor=INK2, label="plain prompt"), Patch(facecolor=INK2, edgecolor=INK2, label="with the monitoring notice")], loc="upper right", fontsize=6.5, handlelength=1.2, borderaxespad=0.3)
    save(fig, a.out, "rq08_monitor_effect", numbers)

if __name__ == "__main__":
    main()
