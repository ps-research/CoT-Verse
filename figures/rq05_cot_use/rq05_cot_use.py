"""RQ5 — What can you learn about how a model uses its CoT from an SDF-tampered
model that you can't from a clean one?

Data: the rq5_slot.py result files (results/rq05_cot_use/<model>_{base,false_3k,true_3k}.json):
CoT-3D's B1/B2 reasoning-slot manipulations scored on three twins of each model
(clean base, 3K false-fact organism, 3K true-fact organism), same 1,000 items,
same injected traces (D-106). Intervals: 95% paired per-fact bootstrap (50 facts,
B=10,000, seed 0).

Main  rq05_override_symmetry
  left   how much of the available room an argued trace covers: a false-arguing
         trace installing the falsehood on the clean twin (hollow) and on the
         true-fact twin (diamond), a true-arguing trace removing it from the
         implanted twin (filled); room = 100 - baseline for installing, baseline
         for removing
  right  the change in false-belief rate when an irrelevant trace fills the slot,
         on each twin
"""
import argparse, json
from matplotlib.lines import Line2D
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figures/common.py
from common import *

B_DEFAULT, SEED = 10_000, 0
TWINS = [("base", "clean", "o", "white"), ("false_3k", "implanted", "o", None), ("true_3k", "true-fact", "D", None)]

def load_rq5(d):
    recs = {}
    for p in sorted(Path(d).glob("*.json")):
        r = json.load(open(p)); facts = sorted({(i["universe"], i["fact_index"]) for i in r["per_item"]}); fid = {f: k for k, f in enumerate(facts)}
        r["_facts"] = facts
        for i in r["per_item"]: i["_f"] = fid[(i["universe"], i["fact_index"])]
        recs[p.stem] = r
    return recs

def per_fact(rec, cond):
    n_f = len(rec["_facts"]); num = np.zeros(n_f); den = np.zeros(n_f)
    for i in rec["per_item"]:
        c = i["conditions"].get(cond)
        if c is None: continue
        den[i["_f"]] += 1; num[i["_f"]] += bool(c["is_sdf"])
    return num, den

def boot_pair(rec, cond, rng, B):
    """point rates (baseline, cond) in %, and the paired resampled rates (facts resampled jointly)."""
    n_f = len(rec["_facts"]); idx = rng.integers(0, n_f, size=(B, n_f))
    nb, db = per_fact(rec, "baseline"); nc, dc = per_fact(rec, cond)
    pool = lambda n, d: 100 * n[idx].sum(1) / np.maximum(d[idx].sum(1), 1)
    return 100 * nb.sum() / db.sum(), 100 * nc.sum() / dc.sum(), pool(nb, db), pool(nc, dc)

def main(a):
    recs = load_rq5(a.rq5); rng = np.random.default_rng(a.seed)
    models = [m for m in MODELS if all(f"{m}_{v}" in recs and recs[f"{m}_{v}"].get("status") == "ok" for v, *_ in TWINS)]
    if a.require_all and len(models) < len(MODELS): raise SystemExit(f"three finished twins only for {models}")
    numbers = {}
    with plt.rc_context({"font.size": 8.5, "ytick.labelsize": 8.5, "xtick.labelsize": 8}):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.42})
        ypos = {m: (len(models) - 1 - i) for i, m in enumerate(models)}; off = {"clean": 0.24, "implanted": 0.0, "true-fact": -0.24}
        for m in models:
            c = SERIES[m]; numbers[m] = {}
            for v, twin, mk, fc in TWINS:
                rec = recs[f"{m}_{v}"]
                if twin == "implanted":
                    pb, pc, b, cc = boot_pair(rec, "true_cot", rng, a.boot); share = 100 * (pb - pc) / pb; bs = 100 * (b - cc) / np.maximum(b, 1e-9)
                else:
                    pb, pc, b, cc = boot_pair(rec, "sdf_cot", rng, a.boot); share = 100 * (pc - pb) / (100 - pb); bs = 100 * (cc - b) / np.maximum(100 - b, 1e-9)
                lo, hi = np.percentile(bs, [2.5, 97.5]); y = ypos[m] + off[twin]
                numbers[m][f"room_share:{twin}"] = {"share": round(float(share), 1), "ci95": [round(float(lo), 1), round(float(hi), 1)], "baseline": round(pb, 1), "after": round(pc, 1)}
                a1.plot([lo, hi], [y, y], color=c, lw=1.3, alpha=1.0 if twin == "implanted" else 0.5, zorder=3)
                a1.plot([share], [y], marker=mk, ms=5.5 if mk == "o" else 5, color=c, mfc=fc or c, mec=c, mew=1.4, ls="none", zorder=4)
                pb2, pc2, b2, cc2 = boot_pair(rec, "unrelated_cot", rng, a.boot); d = cc2 - b2; lo2, hi2 = np.percentile(d, [2.5, 97.5])
                numbers[m][f"irrelevant:{twin}"] = {"delta": round(pc2 - pb2, 1), "ci95": [round(float(lo2), 1), round(float(hi2), 1)], "baseline": round(pb2, 1)}
                a2.plot([lo2, hi2], [y, y], color=c, lw=1.3, alpha=1.0 if twin == "implanted" else 0.5, zorder=3)
                a2.plot([pc2 - pb2], [y], marker=mk, ms=5.5 if mk == "o" else 5, color=c, mfc=fc or c, mec=c, mew=1.4, ls="none", zorder=4)
        for ax in (a1, a2):
            ax.set_yticks([ypos[m] for m in models]); ax.set_yticklabels([SHORT[m] for m in models]); ax.set_ylim(-0.6, len(models) - 0.4)
            for lab, m in zip(ax.get_yticklabels(), models): lab.set_color(SERIES[m]); lab.set_weight("bold")
            for y in [ypos[m] - 0.5 for m in models[:-1]]: ax.axhline(y, color=GRID, lw=0.5, zorder=0)
            ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
        a1.set_xlim(0, 105); a1.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%"); a1.set_xlabel("share of the available room covered by an argued trace")
        panel(a1, "a", "Argued trace")
        a2.axvline(0, color=INK2, lw=0.8, zorder=2); a2.set_xlim(-14, 10); a2.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}" if v else "0")
        a2.set_xlabel("change in false-belief rate, irrelevant trace (pp)"); panel(a2, "b", "Irrelevant trace")
        # legend inside the left panel, in the empty lower-left region (no outside white space)
        h = [Line2D([], [], color=INK2, marker="o", ms=5, mfc="white", mew=1.3, ls="none", label="clean twin: false trace installs"),
             Line2D([], [], color=INK2, marker="D", ms=4.5, ls="none", label="true-fact twin: false trace installs"),
             Line2D([], [], color=INK2, marker="o", ms=5, ls="none", label="implanted twin: true trace removes")]
        a1.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, -0.02), fontsize=6.3, handlelength=1.1, handletextpad=0.5, labelspacing=0.25,
                  borderpad=0.35, frameon=True, fancybox=False, facecolor="white", edgecolor=GRID, framealpha=0.95)
        save(fig, a.out, "rq05_override_symmetry", {"method": __doc__, "models": models, "B": a.boot, "seed": a.seed, "numbers": numbers})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rq5", default=str(Path(__file__).resolve().parents[2] / "results" / "rq05_cot_use")); ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=SEED); ap.add_argument("--boot", type=int, default=B_DEFAULT); ap.add_argument("--require-all", action="store_true")
    main(ap.parse_args())
