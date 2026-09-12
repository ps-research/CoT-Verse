"""rq12_delta.py — RQ12 (D-148): WHERE does the implanted belief live? The SDF weight delta (the merged rank-128 LoRA)
switched off per block of layers (ablation: block off, the rest on) and on alone (isolation: only the block on), on the
false 3k and 10k organisms of the four bases.

Data: figures/rq12_delta/materials/rq12_delta.json from experiments/analysis/rq12_delta.py over results/rq14_localisation/<model>_false_<scale>.json
(experiments/rq12_delta.py). Rates are over the installed items (the organism answers the implanted option with
everything on, the base does not); intervals are per-fact percentile bootstraps (B = 10,000, seed 0).

Main      rq12_delta_blocks   one column per base, ablation row and isolation row: the eight level-1 blocks along the
                              depth, 3k (light) and 10k (full) side by side; dashed line = every adapter off (the ceiling
                              of ablation) in the top row; the bisection's end point named in each panel.
Appendix  rq12_delta_path     the bisection itself: per organism, each level's two halves with their ablation flip and
                              isolation sustain, down to one layer.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figures/common.py
from common import *

ORDER = ["deepseek", "phi4", "qwen3", "gemma4"]
SCALES = [("3k", 0.45), ("10k", 1.0)]

def load_materials(p):
    R = json.load(open(p))["organisms"]
    return {(r["model"], r["scale"]): r for r in R.values()}

def fig_blocks(M, out):
    models = [m for m in ORDER if any((m, s) in M for s, _ in SCALES)]
    fig, axes = plt.subplots(2, len(models), figsize=(1.9 * len(models) + 0.6, 3.6), sharey="row", gridspec_kw={"wspace": 0.12, "hspace": 0.35})
    axes = np.atleast_2d(axes) if len(models) > 1 else np.array([[axes[0]], [axes[1]]])
    numbers = {"method": __doc__, "models": {}}
    for j, m in enumerate(models):
        top, bot = axes[0, j], axes[1, j]; c = SERIES[m]; numbers["models"][m] = {}
        for k, (s, alpha) in enumerate(SCALES):
            r = M.get((m, s))
            if not r: continue
            L = r["level1"]; n = len(L); xs = np.arange(n) + (k - 0.5) * 0.36; w = 0.34
            abl = [st["ablation_flip_to_true"] for st in L]; iso = [st["isolation_sustain"] for st in L]
            top.bar(xs, [100 * a["rate"] for a in abl], w, color=c, alpha=alpha, zorder=3)
            top.errorbar(xs, [100 * a["rate"] for a in abl], yerr=[[100 * (a["rate"] - a["ci95"][0]) for a in abl], [100 * (a["ci95"][1] - a["rate"]) for a in abl]], fmt="none", ecolor=INK2, elinewidth=0.6, capsize=1.5, zorder=4)
            bot.bar(xs, [100 * a["rate"] for a in iso], w, color=c, alpha=alpha, zorder=3)
            bot.errorbar(xs, [100 * a["rate"] for a in iso], yerr=[[100 * (a["rate"] - a["ci95"][0]) for a in iso], [100 * (a["ci95"][1] - a["rate"]) for a in iso]], fmt="none", ecolor=INK2, elinewidth=0.6, capsize=1.5, zorder=4)
            ref = r["baselines"]["all_off"]["flip_to_true"]["rate"] * 100
            top.axhline(ref, color=c, alpha=alpha, lw=0.9, ls="--", zorder=2)
            path = " > ".join(st["block"] for st in r["bisection_path"]); cru = r.get("crucial_block")
            numbers["models"][m][s] = {"n_installed": r["n_installed_items"], "n_facts": r["n_facts"], "all_off_flip_to_true": r["baselines"]["all_off"]["flip_to_true"],
                                       "blocks": [{"block": st["block"], "ablation_flip_to_true": st["ablation_flip_to_true"], "isolation_sustain": st["isolation_sustain"]} for st in L],
                                       "bisection_path": path, "crucial_block": cru["block"] if cru else None}
            bot.text(0.02, 0.97 - 0.11 * k, f"{s}: {path}", transform=bot.transAxes, fontsize=5.6, color=c, alpha=min(1, alpha + 0.3), va="top")   # the bisection path, in the emptier row
        blocks = (M.get((m, "3k")) or M.get((m, "10k")))["level1"]
        for ax in (top, bot):
            ax.set_xticks(range(len(blocks))); ax.set_xticklabels([st["block"] for st in blocks], fontsize=5.6, rotation=60, ha="right"); ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.5); ax.set_axisbelow(True); ax.tick_params(axis="y", labelsize=7)
        top.set_title(LABEL[m], fontsize=8.5, color=c, pad=4); bot.set_xlabel("layers in the block", fontsize=7)
    axes[0, 0].set_ylabel("block's adapters OFF:\nflipped to the true answer (%)", fontsize=7); axes[1, 0].set_ylabel("ONLY the block's adapters on:\nimplanted answer kept (%)", fontsize=7)
    h = [Patch(color=INK2, alpha=0.45, label="3k organism"), Patch(color=INK2, label="10k organism"), Line2D([], [], color=INK2, ls="--", lw=0.9, label="every adapter off")]
    axes[0, 0].legend(handles=h, loc="upper right", bbox_to_anchor=(1.0, 0.8), fontsize=6, frameon=False, handlelength=1.4, borderaxespad=0.2)   # between the bars and the dashed ceiling
    save(fig, out, "rq12_delta_blocks", numbers)

def fig_path(M, out):
    keys = [(m, s) for m in ORDER for s, _ in SCALES if (m, s) in M]
    fig, axes = plt.subplots(1, len(keys), figsize=(1.7 * len(keys) + 0.5, 2.9), gridspec_kw={"wspace": 0.55}); axes = np.atleast_1d(axes)   # every organism has its own blocks: no shared y
    numbers = {"method": __doc__, "paths": {}}
    for ax, (m, s) in zip(axes, keys):
        r = M[(m, s)]; c = SERIES[m]; rows = []
        for step in r["bisection_path"]:
            if step["halves"] is None: rows.append((step["block"], step["ablation_flip_to_true"], step["isolation_sustain"], True)); continue
            for hb in step["halves"]:
                cfg = r["configs"]; rows.append((hb, cfg[f"abl:{hb}"]["flip_to_true"], cfg[f"iso:{hb}"]["sustain"], hb == step["block"]))
        numbers["paths"][f"{m}_{s}"] = [{"block": b, "ablation_flip_to_true": f, "isolation_sustain": i, "chosen": ch} for b, f, i, ch in rows]
        ys = np.arange(len(rows))[::-1]
        for y, (b, f, i, chosen) in zip(ys, rows):
            ax.barh(y + 0.18, 100 * f["rate"], 0.34, color=c, alpha=1.0 if chosen else 0.4, zorder=3)
            ax.barh(y - 0.18, 100 * i["rate"], 0.34, facecolor="white", edgecolor=c, lw=1.0, alpha=1.0 if chosen else 0.5, zorder=3)
        ax.set_yticks(ys); ax.set_yticklabels([b for b, *_ in rows], fontsize=6); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_xlim(0, 100); ax.grid(axis="x", alpha=0.5); ax.set_axisbelow(True)
        ax.axvline(100 * r["baselines"]["all_off"]["flip_to_true"]["rate"], color=c, lw=0.8, ls="--", zorder=2)
        ax.set_title(f"{SHORT[m]} {s}", fontsize=8, color=c, pad=3); ax.tick_params(axis="x", labelsize=6.5)
    axes[0].set_ylabel("block (layers), level by level", fontsize=7); fig.text(0.5, 0.005, "over the installed items (%): filled = block off, flipped to true; hollow = block alone, implanted kept; dashed = every adapter off", ha="center", fontsize=6.3, color=INK2)
    save(fig, out, "rq12_delta_path", numbers)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--materials", default=str(Path(__file__).resolve().parent / "materials" / "rq12_delta.json"))
    ap.add_argument("--out", default=str(DEFAULT_OUT)); a = ap.parse_args()
    M = load_materials(a.materials); fig_blocks(M, a.out); fig_path(M, a.out)

if __name__ == "__main__":
    main()
