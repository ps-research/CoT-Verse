"""RQ2 — Does the plausibility of the false fact gate how well it takes?

The CoT-3D paper's Figure 3(a) encoding (net absorption per model, one bar per
plausibility tier: solid, hatched, cross-hatched) redrawn with a per-fact
bootstrap interval on every bar.

Data: A1 single-fact MCQs, base vs 3K organism. 50 facts, 20 items each;
20 / 16 / 14 facts in the plausible / borderline / near-egregious tiers.
Per-fact net = (SDF rate on the 3K organism) - (SDF rate on the base) over the
fact's 20 items. Bar = mean over facts in the tier (identical to the paper's
pooled item rate because every fact has 20 items). Interval = 95% percentile
bootstrap of that mean, facts resampled within tier, B=10,000, seed 0.
The JSON beside the figure also carries the bootstrap interval of the tier
spread (max - min of the three tier means) per model.
"""
from matplotlib.patches import Patch
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figures/common.py
from common import *

def per_fact_net(results, model):
    b = load(Path(results) / "cot3d_A1_belief_rate" / f"{model}_base.json")
    f = load(Path(results) / "cot3d_A1_belief_rate" / f"{model}_false_3k.json")
    agg = {}
    for tag, d in (("b", b), ("f", f)):
        for r in records(d):
            k = (r["universe"], r["fact_index"])
            a = agg.setdefault(k, {"tier": r["tier"], "b": [], "f": []})
            assert a["tier"] == r["tier"]; a[tag].append(bool(r["is_sdf"]))
    out = {}
    for k, a in agg.items():
        assert len(a["b"]) == 20 and len(a["f"]) == 20, (k, len(a["b"]), len(a["f"]))
        out[k] = (a["tier"], 100 * (np.mean(a["f"]) - np.mean(a["b"])))
    assert len(out) == 50
    return out

def main(a):
    rng = np.random.default_rng(a.seed); B = a.boot
    stats, numbers = {}, {}
    for m in MODELS:
        pf = per_fact_net(a.results, m); stats[m] = []; numbers[m] = {}
        tiers_x = []
        for t in TIERS:
            x = np.array([v for (tier, v) in pf.values() if tier == t]); tiers_x.append(x)
            boots = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5]); stats[m].append((x.mean(), lo, hi))
            numbers[m][t] = {"n_facts": int(len(x)), "net_pp": round(float(x.mean()), 1), "ci95": [round(float(lo), 1), round(float(hi), 1)],
                             "fact_sd_pp": round(float(x.std(ddof=1)), 1)}
        allx = np.array([v for (_, v) in pf.values()]); numbers[m]["overall_net_pp"] = round(float(allx.mean()), 1)
        sb = np.ptp(np.stack([rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1) for x in tiers_x]), axis=0)
        numbers[m]["tier_spread_pp"] = round(float(np.ptp([x.mean() for x in tiers_x])), 1)
        numbers[m]["tier_spread_boot_ci95"] = [round(float(v), 1) for v in np.percentile(sb, [2.5, 97.5])]
    hatches = ["", "///", "xxx"]
    with plt.rc_context({"font.size": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9}):
        fig, ax = plt.subplots(figsize=(3.4, 2.6)); w = 0.27
        for j, m in enumerate(MODELS):
            for i, (v, lo, hi) in enumerate(stats[m]):
                xpos = j + (i - 1) * w
                ax.bar(xpos, v, width=w * 0.92, facecolor=SERIES[m] if i == 0 else "white", edgecolor=SERIES[m], linewidth=1.0, hatch=hatches[i], zorder=3)
                ax.errorbar(xpos, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=4)
            top = max(hi for (_, _, hi) in stats[m])
            ax.annotate(f"+{numbers[m]['overall_net_pp']:.1f}", xy=(j, top), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=SERIES[m], weight="bold")
        tier_short = {"plausible": "Plausible", "borderline": "Borderline", "near_egregious": "Near-egreg."}
        leg = ax.legend(handles=[Patch(facecolor="white", edgecolor=INK2, hatch=h, label=tier_short[t]) for h, t in zip(hatches, TIERS)],
                        title="Adversarial tier", title_fontsize=7, fontsize=7, loc="lower center", frameon=False, ncol=3,
                        handlelength=1.3, handleheight=1.0, columnspacing=0.7, handletextpad=0.3, borderpad=0.0, bbox_to_anchor=(0.5, 1.0))
        leg._legend_box.align = "center"
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([SHORT[m] for m in MODELS], rotation=18, ha="right")
        ax.set_ylim(0, 100); ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_ylabel("net SDF absorption (pp)")
        ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
        save(fig, a.out, "rq02_tier_bootstrap", {"method": __doc__, "B": B, "seed": a.seed, "models": numbers})

if __name__ == "__main__":
    main(cli(__doc__))
