"""RQ11 — the boundary study: how much of a model's own chain of thought can be edited before its answer moves,
and whether a move is content or strangeness (D-127).

Data: figures/rq11_boundary/materials/rq11_boundary.json from experiments/analysis/rq11_boundary.py (per-fact bootstrap, B=10,000,
seed 0, over rq11_edit.py result files). Intervals are 95% percentile bootstraps over facts.

Main  rq11_dose_response
  rows = twins (clean, implanted), columns = models; x = dose of the model's own sentences replaced
  (1 sentence, 25%, 50%, 100%), y = rate at which the answer letter changes from the own-trace answer.
  Four edit types of equal sentence count: opposed (the other twin's sentences on the same item: content),
  placebo (the same twin's sentences from another universe, conclusions removed: strangeness), scramble
  (own words shuffled), offdist (RQ5's off-distribution placebo). Content term = opposed − placebo.
Appendix  rq11_position
  the one-sentence dose by position (first, middle, last, random) per model and edit type.
"""
import argparse, json
from matplotlib.lines import Line2D
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figures/common.py
from common import *

MODEL_LABEL = {"deepseek": "DeepSeek-R1-8B", "phi4": "Phi-4-reasoning", "qwen3": "Qwen3-14B", "gemma4": "Gemma-4-31B"}
DOSES = ["1s", "25", "50", "100"]; DOSE_LABEL = {"1s": "1 sent.", "25": "25%", "50": "50%", "100": "100%"}
POS = ["first", "middle", "last", "random"]
GREY, GREY2 = "#7a7975", "#b3b2ae"
STYLE = {"opposed": dict(ls="-", marker="o", mfc="fill"), "placebo": dict(ls="--", marker="o", mfc="white"),
         "scramble": dict(ls=":", marker="^", mfc="fill", grey=True), "offdist": dict(ls="-.", marker="s", mfc="white", grey=True)}
EDIT_LABEL = {"opposed": "opposed (other twin's sentences)", "placebo": "placebo (own text, other item)", "scramble": "scramble (own words shuffled)", "offdist": "off-distribution placebo (RQ5)"}

def _series(ax, xs, pts, colour, style, label=None):
    c = GREY if style.get("grey") else colour
    ys = [p[0] * 100 for p in pts]; lo = [(p[0] - p[1]) * 100 for p in pts]; hi = [(p[2] - p[0]) * 100 for p in pts]
    ax.errorbar(xs, ys, yerr=[lo, hi], color=c, ls=style["ls"], marker=style["marker"], ms=4, lw=1.4, mfc=(c if style["mfc"] == "fill" else "white"),
                mec=c, capsize=2, elinewidth=0.8, label=label, zorder=3)

def fig_dose(M, out):
    models = [m for m in ("deepseek", "phi4", "qwen3", "gemma4") if m in M]; twins = ["clean", "implanted"]
    fig, axes = plt.subplots(len(twins), len(models), figsize=(2.6 * len(models) + 0.6, 4.6), sharex=True, sharey=True, squeeze=False)
    numbers = {}
    for j, m in enumerate(models):
        for i, tw in enumerate(twins):
            ax = axes[i][j]; T = M[m]["twins"].get(tw)
            if not T: ax.set_axis_off(); continue
            xs = list(range(len(DOSES)))
            for e in ("scramble", "offdist", "placebo", "opposed"):
                cv = T["dose_curves"].get(e, {})
                if not all(d in cv for d in DOSES): continue
                _series(ax, xs, [cv[d]["change_rate"] for d in DOSES], SERIES[m], STYLE[e], label=EDIT_LABEL[e] if (i == 0 and j == 0) else None)
                numbers.setdefault(m, {}).setdefault(tw, {})[e] = {d: cv[d]["change_rate"] for d in DOSES}
            numbers[m][tw]["content_minus_placebo"] = T.get("content_minus_placebo", {})
            cmp = T.get("content_minus_placebo", {}).get("100", {}).get("change_rate")
            ax.set_title(f"{MODEL_LABEL[m]} — {'clean twin' if tw == 'clean' else 'implanted twin'}  (n={T['n']})", fontsize=8, pad=13)
            if cmp: ax.text(0.5, 1.015, f"content term at 100%: {cmp[0]*100:+.0f} pp [{cmp[1]*100:+.0f}, {cmp[2]*100:+.0f}]", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.8, color=INK2)
            ax.set_xticks(xs); ax.set_xticklabels([DOSE_LABEL[d] for d in DOSES]); ax.set_ylim(-3, 103); ax.grid(axis="y", alpha=0.6)
            if j == 0: ax.set_ylabel("answer letter changed (% of items)")
            if i == len(twins) - 1: ax.set_xlabel("share of the model's own sentences replaced")
    axes[0][0].legend(loc="upper left", fontsize=6.5, handlelength=2.6, title="what replaces the sentences", title_fontsize=6.5)
    fig.tight_layout(h_pad=0.9, w_pad=0.4)
    save(fig, out, "rq11_dose_response", numbers)

def fig_position(M, out):
    """One replaced sentence by position: bars for the OPPOSED edit (the only edit with any signal), clean twin hollow and
    implanted twin filled; the three other edits never exceed a few percent anywhere and are stated in one line."""
    models = [m for m in ("deepseek", "phi4", "qwen3", "gemma4") if m in M]
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models) + 0.6, 2.7), sharey=True, squeeze=False); numbers = {}
    for j, m in enumerate(models):
        ax = axes[0][j]; xs = np.arange(len(POS)); w = 0.36; others_max = 0.0
        for k, tw in enumerate(("clean", "implanted")):
            T = M[m]["twins"].get(tw)
            if not T: continue
            P = T["positions"].get("opposed", {})
            if not all(P.get(p) for p in POS): continue
            ys = [P[p][0] * 100 for p in POS]; lo = [(P[p][0] - P[p][1]) * 100 for p in POS]; hi = [(P[p][2] - P[p][0]) * 100 for p in POS]
            ax.bar(xs + (k - 0.5) * w, ys, w, color=(SERIES[m] if tw == "implanted" else "white"), edgecolor=SERIES[m], lw=1.2, yerr=[lo, hi], capsize=2, error_kw={"elinewidth": 0.8, "ecolor": INK2},
                   label=("implanted twin" if tw == "implanted" else "clean twin"), zorder=3)
            numbers.setdefault(m, {})[tw] = {"opposed": {p: P[p] for p in POS}}
            for e in ("placebo", "scramble", "offdist"):
                Pe = T["positions"].get(e, {}); numbers[m][tw][e] = {p: Pe.get(p) for p in POS}
                others_max = max(others_max, max((Pe[p][0] for p in POS if Pe.get(p)), default=0.0))
        ax.set_title(f"{MODEL_LABEL[m]} — one sentence replaced by the other twin's", fontsize=8)
        ax.set_xticks(xs); ax.set_xticklabels(["first", "middle", "last", "random"]); ax.set_ylim(0, 15); ax.grid(axis="y", alpha=0.6); ax.set_axisbelow(True)
        ax.set_xlabel("position of the replaced sentence in the model's own trace")
        if j == 0: ax.set_ylabel("answer letter changed (% of items)")
        numbers[m]["other_edits_max_rate"] = others_max      # stated in the write-up, not on the figure (author)
        ax.legend(loc="upper right", fontsize=6.5, title="opposed edit", title_fontsize=6.5)
    fig.tight_layout(w_pad=0.4); save(fig, out, "rq11_position", numbers)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", default=str(Path(__file__).resolve().parent / "materials" / "rq11_boundary.json"))
    ap.add_argument("--out", default=str(DEFAULT_OUT)); a = ap.parse_args()
    M = json.load(open(a.materials))["models"]
    fig_dose(M, a.out); fig_position(M, a.out)

if __name__ == "__main__":
    main()
