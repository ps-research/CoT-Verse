"""RQ9 — are published faithfulness monitors blind to a weight-installed belief and sensitive to an in-context cue? (D-125)
Data: figures/rq09_monitors/materials/rq09_readers.json (rq09_readers.py), rq09_chen.json (rq09_chen.py), rq09_resample.json
(rq09_resample_summary.py). Same 400 multi-hop items and traces as RQ8; DeepSeek excluded (D-131).

Main  rq09_monitors
  (a) the RQ6 readers on the same traces: belief-blind reader flags a claim / names the implanted claim, belief-aware reader
      says the trace endorses it; clean and implanted twin, plain (hollow) -> monitored (filled), per model
  (b) Chen et al.'s acknowledgment test with the monitoring notice as the cue: share of eligible items whose answer moves to
      the true answer under the notice (light), and the share of those flips whose reasoning mentions the notice (solid)
  (c) Thought Branches resampling on the implanted twin: counterfactual importance (KL, Eq. 1) and resilience (interventions
      before the content stays away, Algorithm 1, K=3) of the sentence the oracle reader quoted as stating the implanted
      claim (filled) against the trace's most important sentence (hollow); percentile bootstrap over traces, n small
"""
import argparse, json, math
from matplotlib.lines import Line2D
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figures/common.py
from common import *

MODEL_LABEL = {"phi4": "Phi-4", "qwen3": "Qwen3", "gemma4": "Gemma-4"}
MEAS = [("blind_flag50", "blind reader\nflags a claim"), ("blind_name50", "blind reader names\nthe implanted claim"), ("oracle_endorses", "oracle reader:\nendorses the claim")]
CELLS = [("implanted/plain", "implanted\nplain"), ("implanted/monitored", "implanted\nmonitored"), ("clean/plain", "clean\nplain"), ("clean/monitored", "clean\nmonitored")]

def wilson(k, n, z=1.96):
    if not n: return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, max(0, c - h), min(1, c + h)]

def main():
    ap = argparse.ArgumentParser(); M = Path(__file__).resolve().parent / "materials"
    ap.add_argument("--readers", default=str(M / "rq09_readers.json")); ap.add_argument("--chen", default=str(M / "rq09_chen.json")); ap.add_argument("--resample", default=str(M / "rq09_resample.json"))
    ap.add_argument("--out", default=str(DEFAULT_OUT)); a = ap.parse_args()
    RD = json.load(open(a.readers))["models"]; CH = {r["organism"]: r for r in json.load(open(a.chen))}; RS = json.load(open(a.resample))["models"]
    models = [m for m in ("phi4", "qwen3", "gemma4") if m in RD]; numbers = {}
    # panel (b) alone has one x-group per (model, twin); panels a/c have a fixed row/column count regardless of model count.
    # Scale only b's share of the width with model count (base point: 2 models), then grow the whole figure by the same increment
    # so a and c keep their original inches (and font sizes) untouched.
    BASE_MODELS = 2; BASE_RATIOS = [1.45, 0.95, 1.05]
    ratios = [BASE_RATIOS[0], BASE_RATIOS[1] * len(models) / BASE_MODELS, BASE_RATIOS[2]]
    fig_w = 9.8 * sum(ratios) / sum(BASE_RATIOS)
    fig = plt.figure(figsize=(fig_w, 3.4)); gs = fig.add_gridspec(2, 3, width_ratios=ratios, wspace=0.34, hspace=0.3)
    ax = fig.add_subplot(gs[:, 0]); bx = fig.add_subplot(gs[:, 1]); cx = fig.add_subplot(gs[0, 2]); dx = fig.add_subplot(gs[1, 2], sharex=cx)
    # (a) readers: rows = twin x measure; per model a small vertical offset
    rows = [(tw, k, lab) for tw in ("implanted", "clean") for k, lab in MEAS]; ys = list(range(len(rows)))[::-1]
    for j, m in enumerate(models):
        off = (j - (len(models) - 1) / 2) * 0.26
        for y, (tw, k, _) in zip(ys, rows):
            c = RD[m]["twins"][tw][k]; p, q = c["plain"][0] * 100, c["monitored"][0] * 100; col = SERIES[m]
            ax.plot([p, q], [y + off, y + off], color=col, lw=1.2, zorder=2); ax.scatter([p], [y + off], s=26, facecolor="white", edgecolor=col, lw=1.2, zorder=3); ax.scatter([q], [y + off], s=26, color=col, zorder=4)
            numbers.setdefault(m, {})[f"readers:{tw}:{k}"] = c
    ax.set_yticks(ys); ax.set_yticklabels([lab for _, _, lab in rows], fontsize=6.6)
    for y, (tw, _, _) in zip(ys, rows): ax.get_yticklabels()[len(rows) - 1 - y].set_color(INK if tw == "implanted" else INK2)
    ax.axhline(2.5, color=GRID, lw=0.8); ax.text(101, 5.45, "implanted twin", ha="right", va="top", fontsize=6.5, color=INK); ax.text(101, 2.45, "clean twin", ha="right", va="top", fontsize=6.5, color=INK2)
    ax.set_xlim(0, 102); ax.set_ylim(-0.6, len(rows) - 0.4); ax.grid(axis="x", alpha=0.5); ax.set_axisbelow(True); ax.set_xlabel("share of traces (%)"); panel(ax, "a", "Readers: plain (hollow) to monitored (filled)", chars=50)
    ax.legend(handles=[Line2D([], [], color=SERIES[m], marker="o", ls="", label=MODEL_LABEL[m]) for m in models], loc="lower right", fontsize=6.5, handletextpad=0.3, borderaxespad=0.3)
    # (b) Chen's acknowledgment test
    xs = []; labels = []; x = 0
    for m in models:
        for tw, var in (("clean", "base"), ("implanted", "false_3k")):
            r = CH.get(f"{m}_{var}")
            if not r: continue
            c = r["chen"]; pw = wilson(c["flips_to_true"], c["n_eligible"]); fw = wilson(c["verbalised_among_flips_to_true"], c["flips_to_true"]) if c["flips_to_true"] else None
            bx.bar([x - 0.19], [pw[0] * 100], 0.36, color=SERIES[m], alpha=0.35, yerr=[[(pw[0] - pw[1]) * 100], [(pw[2] - pw[0]) * 100]], capsize=1.5, error_kw={"elinewidth": 0.6}, zorder=3)
            if fw: bx.bar([x + 0.19], [fw[0] * 100], 0.36, color=SERIES[m], yerr=[[(fw[0] - fw[1]) * 100], [(fw[2] - fw[0]) * 100]], capsize=1.5, error_kw={"elinewidth": 0.6}, zorder=3)
            bx.text(x - 0.19, max(pw[0] * 100 + 3 + (pw[2] - pw[0]) * 100, 20), f"{c['flips_to_true']}/{c['n_eligible']}", ha="center", fontsize=5.6, color=INK2)  # floor: a tiny bar + tiny CI otherwise lands the label on the x-tick labels below it
            if fw: bx.text(x + 0.19, max(fw[0] * 100 + 3 + (fw[2] - fw[0]) * 100, 20), f"{c['verbalised_among_flips_to_true']}/{c['flips_to_true']}", ha="center", fontsize=5.6, color=INK2)
            numbers.setdefault(m, {})[f"chen:{tw}"] = {"p_flip_to_true": pw, "verbalised_among_flips": fw, "chen": c}
            xs.append(x); labels.append(f"{MODEL_LABEL[m]}\n{tw}"); x += 1
    bx.set_xticks(xs); bx.set_xticklabels(labels, fontsize=6.5); bx.tick_params(axis="x", pad=5); bx.set_ylim(0, 100); bx.grid(axis="y", alpha=0.5); bx.set_axisbelow(True); bx.set_ylabel("%")
    panel(bx, "b", "Chen's test: truth (light), notice (solid)", chars=27)
    # (c) Thought Branches: claim sentence (filled) vs the trace's most important sentence (hollow)
    for j, m in enumerate(models):
        S = RS.get(m, {}); off = (j - (len(models) - 1) / 2) * 0.22
        for xi, (cell, _) in enumerate(CELLS):
            s = S.get(cell)
            if not s: continue
            for axis, kc, kt in ((cx, "claim_importance", "top_importance"), (dx, "claim_resilience_k", "top_resilience_k")):
                for key, filled in ((kc, True), (kt, False)):
                    v = s[key]
                    if v[0] is None: continue
                    axis.errorbar([xi + off + (0 if filled else 0.07)], [v[0]], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", ms=4, color=SERIES[m], mfc=SERIES[m] if filled else "white", ecolor=SERIES[m], elinewidth=0.8, capsize=1.5, zorder=3)
            cx.text(xi + off, 3.35, f"n={s['claim_importance'][3]}", ha="center", va="bottom", fontsize=5.6, color=INK2, clip_on=False)
            numbers.setdefault(m, {})[f"resample:{cell}"] = {k: s[k] for k in ("n_traces", "status", "claim_found_share", "claim_importance", "top_importance", "claim_reappearance", "claim_resilience_k", "top_resilience_k", "claim_survived_K", "top_survived_K")}
    cx.set_ylabel("importance (KL)", fontsize=7); dx.set_ylabel("resilience (of K=3)", fontsize=7); dx.set_ylim(-0.1, 3.2); dx.set_yticks([0, 1, 2, 3]); cx.set_ylim(0, 3.4); cx.set_yticks([0, 1, 2, 3])
    dx.set_xticks(range(len(CELLS))); dx.set_xticklabels([lab for _, lab in CELLS], fontsize=6.0); plt.setp(cx.get_xticklabels(), visible=False)
    for axis in (cx, dx): axis.grid(axis="y", alpha=0.5); axis.set_axisbelow(True); axis.set_xlim(-0.5, len(CELLS) - 0.5)
    panel(cx, "c", "Thought Branches: claim (filled) vs top sentence (hollow)", chars=33)
    save(fig, a.out, "rq09_monitors", numbers)

if __name__ == "__main__":
    main()
