"""RQ4 — Can the false belief be dislodged at inference time?

Data: the rq4_override.py result files (results/rq04_dislodging/<model>_{base,false_1k,false_3k,false_10k}.json),
CoT-3D's A2 protocol reproduced on both twins with mirrored templates, the doubt
decomposition and the two-question durability pairs (D-101). Only models whose
3K organism and base files are finished are drawn. Intervals are 95% per-fact
percentile bootstraps (50 facts, B=10,000, seed 0).

Main
  rq04_compliance_symmetry   for each of the five templates: the implanted twin pushed to
                             the truth vs the clean twin pushed to the falsehood, share of
                             held answers flipped (items each twin held at baseline)
  rq04_durability            false-belief rate on a second question about the same fact
                             after question 1 was answered (or corrected) in the same context
Appendix
  rq04_counter_evidence_appendix   per-item held-belief margin before vs after counter-evidence,
                                   and the share flipped as a function of baseline strength
  rq04_dose_appendix               share of false answers recovered by each template on the
                                   1K / 3K / 10K organisms
"""
import argparse, json, os
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figures/common.py
from common import *

B_DEFAULT, SEED = 10_000, 0
FIVE = ["are_you_sure", "system_override", "counter_evidence", "authority_override", "explicit_correction"]
LBL = {"are_you_sure": "Own answer anchored\n+ \"Are you sure?\"", "system_override": "System override", "counter_evidence": "Counter-evidence",
       "authority_override": "Authority", "explicit_correction": "Explicit correction"}
ARM = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]       # arm palette for the dose panel (one hue per template)
ARM_OF = {"are_you_sure": ARM[4], "system_override": ARM[0], "counter_evidence": ARM[2], "authority_override": ARM[3], "explicit_correction": ARM[1]}

# ───────────────────────── loading + per-fact bootstrap ─────────────────────────
def load_rq4(d):
    recs = {}
    for p in sorted(Path(d).glob("*.json")):
        r = json.load(open(p)); facts = sorted({(i["universe"], i["fact_index"]) for i in r["per_item"]}); fid = {f: k for k, f in enumerate(facts)}
        r["_facts"] = facts
        for i in r["per_item"]: i["_f"] = fid[(i["universe"], i["fact_index"])]
        recs[p.stem] = r
    return recs

is_sdf = lambda c: c["is_sdf"]; is_true = lambda c: c["is_true"]
base_sdf = lambda i: i["conditions"]["baseline"]["is_sdf"]; base_true = lambda i: i["conditions"]["baseline"]["is_true"]

def rate_ci(rec, cond, pred, subset=None, rng=None, B=B_DEFAULT):
    n_f = len(rec["_facts"]); num = np.zeros(n_f); den = np.zeros(n_f)
    for i in rec["per_item"]:
        c = i["conditions"].get(cond)
        if c is None or (subset is not None and not subset(i)): continue
        den[i["_f"]] += 1; num[i["_f"]] += bool(pred(c))
    if den.sum() == 0: return None
    idx = rng.integers(0, n_f, size=(B, n_f)); bs = 100 * num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return {"rate": float(100 * num.sum() / den.sum()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(den.sum())}

def paired_forest(ax, models, rows, getters, row_label, x_label):
    """rows: list of keys; getters(m, key) -> (a, b) rate_ci dicts (a filled, b hollow)."""
    ypos = {r: (len(rows) - 1 - i) for i, r in enumerate(rows)}; off = np.linspace(0.3, -0.3, len(models)); out = {}
    for k, m in enumerate(models):
        c = SERIES[m]; out[m] = {}
        for r in rows:
            a, b = getters(m, r); out[m][r] = {"filled": a, "hollow": b}; y = ypos[r] + off[k]
            ax.plot([a["ci95"][0], a["ci95"][1]], [y, y], color=c, lw=1.3, zorder=3)
            ax.plot([b["ci95"][0], b["ci95"][1]], [y, y], color=c, lw=1.3, alpha=0.5, zorder=3)
            ax.plot([a["rate"], b["rate"]], [y, y], color=c, lw=0.6, alpha=0.6, zorder=2)
            ax.plot([a["rate"]], [y], marker="o", ms=5.5, color=c, mec="white", mew=0.6, zorder=4)
            ax.plot([b["rate"]], [y], marker="o", ms=5.5, mfc="white", mec=c, mew=1.5, zorder=4)
    ax.set_yticks([ypos[r] for r in rows]); ax.set_yticklabels([row_label[r] for r in rows]); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 100); ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%"); ax.set_xlabel(x_label)
    ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
    for y in [ypos[r] - 0.5 for r in rows[:-1]]: ax.axhline(y, color=GRID, lw=0.5, zorder=0)
    return out

# ───────────────────────── main: compliance symmetry ─────────────────────────
def fig_compliance(a, recs, models, rng):
    with plt.rc_context({"font.size": 8.5, "ytick.labelsize": 8, "xtick.labelsize": 8}):
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        def get(m, r):
            return (rate_ci(recs[f"{m}_false_3k"], r, is_true, base_sdf, rng, a.boot), rate_ci(recs[f"{m}_base"], r + "_to_false", is_sdf, base_true, rng, a.boot))
        numbers = paired_forest(ax, models, FIVE, get, LBL, "share of answers flipped by the prompt (95% per-fact bootstrap)")
        # one compact legend in the top band (the anchored row sits near 0%, so the band is empty): models, then the two marker meanings
        # one row directly above the axes (the title band), so nothing covers the data and no margin is added
        h = [Line2D([], [], color=SERIES[m], marker="o", ms=5, lw=1.3, mec="white", label=SHORT[m]) for m in models]
        h += [Line2D([], [], color=INK2, marker="o", ms=5, lw=0, mec="white", label="implanted, to the truth"),
              Line2D([], [], color=INK2, marker="o", ms=5, lw=0, mfc="white", mew=1.4, label="clean, to the falsehood")]
        ax.legend(handles=h, ncol=len(h), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6.6, handlelength=1.2,
                  columnspacing=0.7, handletextpad=0.4, borderpad=0.2, borderaxespad=0.15)
        save(fig, a.out, "rq04_compliance_symmetry", {"method": __doc__, "models": models, "B": a.boot, "seed": a.seed, "numbers": numbers})

# ───────────────────────── main: durability ─────────────────────────
def fig_durability(a, recs, models, rng):
    with plt.rc_context({"font.size": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 8}):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 2.6), gridspec_kw={"wspace": 0.25}); numbers = {}
        panels = [(a1, "false_3k", [("baseline", "single\nquestion"), ("persist_prior_true", "Q1 answered\ntrue"), ("persist_corr_true", "Q1 corrected\nto true")], "implanted twin, question 2"),
                  (a2, "base", [("baseline", "single\nquestion"), ("persist_prior_false", "Q1 answered\nfalse"), ("persist_corr_false", "Q1 corrected\nto false")], "clean twin, question 2")]
        for ax, variant, conds, title in panels:
            w = 0.8 / len(models)
            for k, m in enumerate(models):
                c = SERIES[m]; rec = recs[f"{m}_{variant}"]
                for j, (cond, _) in enumerate(conds):
                    v = rate_ci(rec, cond, is_sdf, rng=rng, B=a.boot); numbers[f"{m}_{variant}_{cond}"] = v
                    xpos = j + (k - (len(models) - 1) / 2) * w
                    ax.bar(xpos, v["rate"], width=w * 0.9, color=c, zorder=3)
                    ax.errorbar(xpos, v["rate"], yerr=[[v["rate"] - v["ci95"][0]], [v["ci95"][1] - v["rate"]]], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.8, zorder=4)
            ax.set_xticks(range(len(conds))); ax.set_xticklabels([l for _, l in conds]); ax.set_ylim(0, 103)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%"); ax.set_title(title, fontsize=8.3, loc="left")
            ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
        a1.set_ylabel("false-belief answer rate")
        a2.legend(handles=[Patch(facecolor=SERIES[m], label=SHORT[m]) for m in models], ncol=2, loc="upper left", fontsize=7.4, frameon=True, fancybox=False,
                  facecolor="white", edgecolor=GRID, framealpha=0.95, handlelength=1.2, columnspacing=0.9, borderpad=0.4)
        save(fig, a.out, "rq04_durability", {"method": __doc__, "models": models, "B": a.boot, "seed": a.seed, "numbers": numbers})

# ───────────────────────── appendix: counter-evidence deep dive ─────────────────────────
def held_items(rec, held, cond):
    rows = [i for i in rec["per_item"] if i["conditions"]["baseline"][held]]; sgn = 1 if held == "is_sdf" else -1
    return (np.array([sgn * i["conditions"]["baseline"]["margin"] for i in rows]), np.array([sgn * i["conditions"][cond]["margin"] for i in rows]))

def fig_counter_evidence(a, recs, models):
    with plt.rc_context({"font.size": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5}):
        fig, axes = plt.subplots(2, len(models), figsize=(2.0 * len(models) + 0.5, 4.3), gridspec_kw={"wspace": 0.16, "hspace": 0.45, "height_ratios": [1.15, 1]})
        axes = np.atleast_2d(axes); numbers = {}
        for k, m in enumerate(models):
            c = SERIES[m]; top, bot = axes[0, k], axes[1, k]; numbers[m] = {}
            xi, yi = held_items(recs[f"{m}_false_3k"], "is_sdf", "counter_evidence"); xb, yb = held_items(recs[f"{m}_base"], "is_true", "counter_evidence_to_false")
            top.scatter(xb, yb, s=7, facecolors="none", edgecolors=c, linewidths=0.5, alpha=0.45, zorder=3)
            top.scatter(xi, yi, s=7, color=c, alpha=0.45, linewidths=0, zorder=4)
            lim = max(np.abs(np.concatenate([xi, yi, xb, yb]))) * 1.05
            top.plot([-lim, lim], [-lim, lim], color=GRID, lw=0.8, zorder=1); top.axhline(0, color=INK2, lw=0.8, zorder=2)
            top.set_xlim(0, lim); top.set_ylim(-lim, lim); top.set_title(SHORT[m], color=c, weight="bold", fontsize=9.5, pad=5)
            top.grid(zorder=0); top.set_axisbelow(True)
            if k == 0: top.set_ylabel("margin after counter-evidence\n(logits, + = still held)")
            if k == len(models) // 2: top.set_xlabel("held-belief margin at baseline (logits, + = the answer the twin held)", fontsize=8)
            for x, y, ls, lab, mfc in ((xi, yi, "-", "implanted twin", c), (xb, yb, "--", "clean twin", "white")):
                edges = np.quantile(x, np.linspace(0, 1, 9)); centers, probs = [], []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    sel = (x >= lo) & (x <= hi)
                    if sel.sum() == 0: continue
                    centers.append(x[sel].mean()); probs.append(100 * (y[sel] < 0).mean())
                bot.plot(centers, probs, color=c, ls=ls, lw=1.6, marker="o", ms=4, mfc=mfc, mec=c, mew=1.2, zorder=3)
                numbers[m][lab] = {"bin_centers": [round(float(v), 2) for v in centers], "flip_pct": [round(float(v), 1) for v in probs], "n": int(len(x))}
            bot.set_xlim(0, lim); bot.set_ylim(0, 103); bot.grid(axis="y", zorder=0); bot.set_axisbelow(True)
            if k == len(models) // 2: bot.set_xlabel("held-belief margin at baseline (logits)", fontsize=8)
            if k == 0: bot.set_ylabel("share flipped by\ncounter-evidence")
            bot.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        for ax in axes[0, 1:]: ax.set_yticklabels([])
        for ax in axes[1, 1:]: ax.set_yticklabels([])
        h = [Line2D([], [], color=INK2, marker="o", ms=4.5, lw=1.6, label="implanted twin, pushed to the truth"),
             Line2D([], [], color=INK2, marker="o", ms=4.5, lw=1.6, ls="--", mfc="white", mew=1.2, label="clean twin, pushed to the falsehood")]
        fig.legend(handles=h, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=7.8, handlelength=2.0, columnspacing=1.6)
        save(fig, a.out, "rq04_counter_evidence_appendix", {"method": __doc__, "models": models, "numbers": numbers})

# ───────────────────────── appendix: dose ─────────────────────────
def fig_dose(a, recs, models, rng):
    doses = ["1k", "3k", "10k"]
    with plt.rc_context({"font.size": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8}):
        fig, axes = plt.subplots(1, len(models), figsize=(1.85 * len(models) + 0.6, 2.5), sharey=True, gridspec_kw={"wspace": 0.12})
        axes = np.atleast_1d(axes); numbers = {}
        for ax, m in zip(axes, models):
            numbers[m] = {}
            for r in FIVE:
                ys = []
                for d in doses:
                    key = f"{m}_false_{d}"
                    if key not in recs or recs[key].get("status") != "ok": ys.append(np.nan); continue
                    v = rate_ci(recs[key], r, is_true, base_sdf, rng, a.boot); numbers[m][f"{r}@{d}"] = v; ys.append(v["rate"])
                ax.plot(range(3), ys, color=ARM_OF[r], lw=1.6, marker="o", ms=4.5, mec="white", zorder=3)
            ax.set_title(SHORT[m], color=SERIES[m], weight="bold", fontsize=9.5, pad=6)
            ax.set_xticks(range(3)); ax.set_xticklabels(["1K", "3K", "10K"]); ax.set_xlim(-0.3, 2.3); ax.set_ylim(0, 103)
            ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
        axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%"); axes[0].set_ylabel("false answers recovered by the prompt")
        fig.supxlabel("documents per domain", fontsize=8.5, y=-0.02)
        short = dict(LBL, are_you_sure="Anchored + \"Are you sure?\"")
        # the legend sits in the empty middle band of the Phi-4 panel (its curves run above 60% and the anchored row near 0%);
        # with the corrected DeepSeek rows (D-126) that model's curves fill its own upper-left corner
        axes[1 if len(models) > 1 else 0].legend(handles=[Line2D([], [], color=ARM_OF[r], lw=1.6, marker="o", ms=4, mec="white", label=short[r]) for r in FIVE],
                       ncol=1, loc="center", bbox_to_anchor=(0.5, 0.36), fontsize=6.6, handlelength=1.4, labelspacing=0.3, borderpad=0.4, frameon=True, fancybox=False,
                       facecolor="white", edgecolor=GRID, framealpha=0.95)
        save(fig, a.out, "rq04_dose_appendix", {"method": __doc__, "models": models, "B": a.boot, "seed": a.seed, "numbers": numbers})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rq4", default=str(Path(__file__).resolve().parents[2] / "results" / "rq04_dislodging"), help="directory of rq4_override.py result files")
    ap.add_argument("--out", default=str(DEFAULT_OUT)); ap.add_argument("--seed", type=int, default=SEED); ap.add_argument("--boot", type=int, default=B_DEFAULT)
    ap.add_argument("--require-all", action="store_true", help="fail unless all four models have finished 3K and base files")
    a = ap.parse_args()
    recs = load_rq4(a.rq4)
    models = [m for m in MODELS if all(k in recs and recs[k].get("status") == "ok" for k in (f"{m}_false_3k", f"{m}_base"))]
    if a.require_all and len(models) < len(MODELS):
        raise SystemExit(f"finished twins only for {models}; waiting for the rest")
    print("models:", models)
    rng = np.random.default_rng(a.seed)
    fig_compliance(a, recs, models, rng); fig_durability(a, recs, models, rng); fig_counter_evidence(a, recs, models); fig_dose(a, recs, models, rng)
