"""RQ15 — does the organism that learned to hide its reasoning hide because it believes it is watched, or has it learned
a habit that runs regardless? (D-134) Data: figures/rq15_belief_switch/materials/rq15_belief.json from experiments/analysis/rq15_belief.py.
Haskins-route organisms of RQ9: S1++ behaviour-only (dashed, hollow) and S2++ monitor-aware (solid, filled), each under
three prompts: plain (RQ9's cells), a PRIVATE notice (reasoning is not read by anyone), a MONITORED notice (reasoning is
read by a monitor looking for concealment). Phi-4 is the model under test, Qwen3 the control. 1,000 scenarios per cell,
rates over the 500 positive scenarios as Haskins' evaluate.py defines them, Wilson 95% intervals.

Main  rq15_belief_switch
  (a) deception rate                (b) detection given deception (the monitor's recall on concealing traces with readable reasoning)
  (c) silent share of the concealing traces (reasoning under 20 characters, which the monitor never sees)
  (d) opaque share of the readable concealing traces (Haskins' App. D.5.1 counterfactual criterion, judged)
  (e) share of the readable concealing traces carrying a self-directed "do not mention" instruction (their App. D.5.2 regex)
  (f) median reasoning tokens
  Together (c)-(f) are the silence / opacity / rewording decomposition of D-134, every panel in the same style.
"""
import argparse, json
from matplotlib.lines import Line2D
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figures/common.py
from common import *

MODEL_LABEL = {"phi4": "Phi-4", "qwen3": "Qwen3"}; PROMPTS = ["plain", "private", "monitored"]; PROMPT_LABEL = {"plain": "plain", "private": "told:\nprivate", "monitored": "told:\nmonitored"}
ORG = {"s1pp": ("behaviour-only (S1++)", "--", "white"), "s2pp": ("monitor-aware (S2++)", "-", None)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--materials", default=str(Path(__file__).resolve().parent / "materials" / "rq15_belief.json"))
    ap.add_argument("--out", default=str(DEFAULT_OUT)); a = ap.parse_args(); R = json.load(open(a.materials)); numbers = {}
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.9), gridspec_kw={"hspace": 0.6, "wspace": 0.3}); axes = axes.ravel()
    def get(c, key):
        if key in c: return c[key]
        o = c.get("opacity") or {}; return o.get(key)
    panels = [(axes[0], "deception_rate", "(a) deception rate\n     (% of positive scenarios)", (0, 70)), (axes[1], "detection_given_deception", "(b) detection given deception (%)", (75, 101)),
              (axes[2], "empty_among_conceal", "(c) silent share of the\n     concealing traces (%)", (0, 50)), (axes[3], "opaque_of_nonempty", "(d) opaque share of the readable\n     concealing traces (%)", (0, 20)),
              (axes[4], "disclosure_regex_of_nonempty", "(e) 'do not mention' self-instruction\n     (% of readable concealing traces)", (0, 105)), (axes[5], "median_tokens", "(f) median reasoning tokens", (0, 1000))]
    for m in ("phi4", "qwen3"):
        if m not in R: continue
        for st, (lab, ls, mfc) in ORG.items():
            cells = R[m]["organisms"].get(st, {}).get("cells", {})
            for axis, key, _, _ in panels:
                pts = [(i, get(cells[p], key)) for i, p in enumerate(PROMPTS) if p in cells and get(cells[p], key) is not None]
                if not pts: continue
                xs = [i for i, _ in pts]; v = [y for _, y in pts]
                if key == "median_tokens": axis.plot(xs, v, color=SERIES[m], ls=ls, lw=1.3, marker="o", ms=4.5, mfc=mfc or SERIES[m], zorder=3)
                else: axis.errorbar(xs, [y[0] * 100 for y in v], yerr=[[(y[0] - y[1]) * 100 for y in v], [(y[2] - y[0]) * 100 for y in v]], color=SERIES[m], ls=ls, lw=1.3, marker="o", ms=4.5, mfc=mfc or SERIES[m], capsize=2, elinewidth=0.8, zorder=3)
                for i, p in enumerate(PROMPTS):
                    if p in cells: numbers.setdefault(m, {}).setdefault(st, {}).setdefault(p, {})[key] = get(cells[p], key)
    for axis, key, title, yl in panels:
        axis.set_xticks(range(3)); axis.set_xticklabels([PROMPT_LABEL[p] for p in PROMPTS], fontsize=7); axis.set_xlim(-0.35, 2.35); axis.set_ylim(*yl); axis.grid(axis="y", alpha=0.5); axis.set_axisbelow(True); axis.set_title(title, fontsize=7.8, loc="left")
    h = [Line2D([], [], color=SERIES[m], lw=1.3, label=MODEL_LABEL[m]) for m in ("phi4", "qwen3") if m in R] + [Line2D([], [], color=INK2, ls=ls, marker="o", ms=4, mfc=mfc or INK2, label=lab) for st, (lab, ls, mfc) in ORG.items()]
    axes[0].legend(handles=h, loc="lower left", fontsize=6.2, ncol=2, handlelength=2.0, columnspacing=0.8, handletextpad=0.5)
    save(fig, a.out, "rq15_belief_switch", numbers)

if __name__ == "__main__":
    main()
