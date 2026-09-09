"""RQ3 — Does the false belief only appear when asked directly, or does it generalise?

Rungs at increasing distance from how the fact was taught, all logit-scored,
3K organisms:
  direct, scenario   A1 MCQ framings (5 items per fact each, 50 facts)
  hop-2, hop-3       B4 multi-hop items, 'direct' arm = no reasoning trace
                     (5 and 3 items per fact)
  hop-2/3 + own CoT  the same B4 items, 'natural' arm = after the model's own CoT
Two outputs:
  rq03_delta_forest      (main) two paired contrasts, 95% paired per-fact
                         bootstrap (facts resampled jointly): grey block = each
                         rung minus the direct question, no CoT anywhere; white
                         block = own CoT minus no CoT on the same multi-hop items
  rq03_heatmap_appendix  (appendix) false-belief rate at every rung, implanted
                         block and base-model block, six columns
"""
import re
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figures/common.py
from common import *

RUNGS = ["direct", "scenario", "hop-2", "hop-3"]; COT = ["hop-2 CoT", "hop-3 CoT"]
COL_LABEL = {"direct": "direct\nquestion", "scenario": "applied\nscenario", "hop-2": "2-hop\ninference", "hop-3": "3-hop\ninference",
             "hop-2 CoT": "2-hop\n+ own CoT", "hop-3 CoT": "3-hop\n+ own CoT"}

def framing(r): return re.sub(r"^\w+?_\d+_", "", r["id"]).rsplit("_v", 1)[0]

NATIVE_COT_ENV = "RQ3_GEMMA4_NATIVE_COT"   # directory of rq3_gemma_native_cot.py records (D-152); the default search below finds them

def native_cot_dir(results):
    """Gemma-4's own-CoT records (rq3_gemma_native_cot.py, D-152): the env var, else results/rq03_generalisation, else the
    script's own materials/, else the working tree's results/rq03_generalisation. None = fall back to CoT-3D's B4 natural arm (forced tag)."""
    here = Path(__file__).resolve().parent
    for d in ([Path(os.environ[NATIVE_COT_ENV])] if os.environ.get(NATIVE_COT_ENV) else []) + [Path(results) / "rq03_generalisation", here / "materials", here.parent / "results" / "rq03_generalisation"]:
        if (d / "gemma4_base_native_cot.json").exists() and (d / "gemma4_false_native_cot.json").exists(): return d
    return None

def per_fact(results, model, variant):
    out = {r: {} for r in RUNGS + COT}
    for r in records(load(Path(results) / "cot3d_A1_belief_rate" / f"{model}_{variant}.json")):
        fr = framing(r)
        if fr in ("direct", "scenario"):
            out[fr].setdefault((r["universe"], r["fact_index"]), []).append(bool(r["is_sdf"]))
    for r in records(load(Path(results) / "cot3d_B4_multihop" / f"{model}_{variant}.json")):
        k = (r["universe"], r["fact_index"]); h = f"hop-{r['hop']}"
        out[h].setdefault(k, []).append(bool(r["arms"]["direct"]["is_sdf"]))
        out[h + " CoT"].setdefault(k, []).append(bool(r["arms"]["natural"]["is_sdf"]))
    native = native_cot_dir(results) if model == "gemma4" else None
    if model == "gemma4" and native is None: print("WARNING: Gemma-4 own-CoT records not found; drawing CoT-3D's forced-tag rows (D-140 artefact)")
    if native:
        # Gemma-4's own-CoT cells only: the traces regenerated through the template's enable_thinking switch (the harness's
        # forced '<|channel>thought' tag left Gemma-4 writing nothing, D-140), scored by the same log-prob machinery; every other
        # rung and model is untouched. base -> clean twin, false_3k -> the 3K organism.
        rec = json.load(open(Path(native) / f"gemma4_{'base' if variant == 'base' else 'false'}_native_cot.json"))
        assert rec["status"] == "ok", f"{NATIVE_COT_ENV}: gemma4 {variant} record is not finished"
        for r in COT: out[r] = {}
        for r in rec["per_item"]:
            out[f"hop-{r['hop']} CoT"].setdefault((r["universe"], r["fact_index"]), []).append(bool(r["is_sdf"]))
    return out

def _pool(pf, idx):
    facts = sorted(pf); s = np.array([sum(pf[f]) for f in facts]); n = np.array([len(pf[f]) for f in facts])
    return 100 * s[idx].sum(1) / n[idx].sum(1)

def rates(pf_all, rng, B):
    facts = sorted(pf_all["direct"]); assert all(sorted(pf_all[r]) == facts for r in pf_all)
    idx = rng.integers(0, len(facts), size=(B, len(facts))); ident = np.arange(len(facts))[None, :]
    out = {}
    for r, pf in pf_all.items():
        bs = _pool(pf, idx)
        out[r] = {"rate": float(_pool(pf, ident)[0]), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  "n_facts": len(facts), "n_items": int(sum(len(v) for v in pf.values()))}
    d0 = _pool(pf_all["direct"], idx)
    for r in RUNGS[1:] + COT:
        bs = _pool(pf_all[r], idx) - d0
        out[r]["delta_vs_direct"] = out[r]["rate"] - out["direct"]["rate"]
        out[r]["delta_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    for r in COT:
        h = r.replace(" CoT", ""); bs = _pool(pf_all[r], idx) - _pool(pf_all[h], idx)
        out[r]["delta_vs_nocot"] = out[r]["rate"] - out[h]["rate"]
        out[r]["delta_nocot_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out

def fig_forest(a, R):
    with plt.rc_context({"font.size": 8.5, "ytick.labelsize": 8, "xtick.labelsize": 8}):
        fig, ax = plt.subplots(figsize=(5.4, 3.1))
        rows = [("scenario", "delta_vs_direct", "delta_ci95", "applied scenario"), ("hop-2", "delta_vs_direct", "delta_ci95", "2-hop inference"),
                ("hop-3", "delta_vs_direct", "delta_ci95", "3-hop inference"), ("hop-2 CoT", "delta_vs_nocot", "delta_nocot_ci95", "2-hop, own CoT"),
                ("hop-3 CoT", "delta_vs_nocot", "delta_nocot_ci95", "3-hop, own CoT")]
        ypos = {r[0]: y for r, y in zip(rows, [5.0, 4.0, 3.0, 1.35, 0.35])}
        ax.axvline(0, color=INK2, lw=0.9, zorder=2); ax.axhspan(2.2, 5.6, color=BLOCK, zorder=0)
        off = np.linspace(0.27, -0.27, 4)
        for k, m in enumerate(MODELS):
            c = SERIES[m]; I = R[m]["implanted"]
            for r, dk, ck, _ in rows:
                y = ypos[r] + off[k]; d = I[r][dk]; lo, hi = I[r][ck]
                ax.plot([lo, hi], [y, y], color=c, lw=1.4, solid_capstyle="round", zorder=3)
                ax.plot([d], [y], marker="o", ms=5.5, color=c, mec="white", mew=0.7, zorder=4)
        ax.set_yticks([ypos[r[0]] for r in rows]); ax.set_yticklabels([r[3] for r in rows]); ax.set_ylim(-0.3, 5.6)
        ax.set_xlim(-25, 40); ax.set_xticks([-20, -10, 0, 10, 20, 30, 40]); ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}" if v else "0")
        ax.set_xlabel("change in false-belief answer rate (pp; 95% paired per-fact bootstrap)")
        ax.grid(axis="x", zorder=0); ax.set_axisbelow(True)
        leg1 = ax.legend(handles=[Line2D([], [], color=SERIES[m], marker="o", ms=5, lw=1.4, mec="white", label=SHORT[m]) for m in MODELS],
                         ncol=1, loc="upper right", frameon=True, fancybox=False, facecolor="white", edgecolor=GRID, framealpha=1.0,
                         handlelength=1.6, labelspacing=0.35, borderpad=0.5, fontsize=7.8)
        leg1.set_zorder(6); ax.add_artist(leg1)
        ax.legend(handles=[Patch(facecolor=BLOCK, edgecolor=GRID, label="vs. the direct question, no CoT"),
                           Patch(facecolor="white", edgecolor=GRID, label="own CoT vs. no CoT, same items")],
                  ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), handlelength=1.6, columnspacing=1.8, fontsize=7.5)
        save(fig, a.out, "rq03_delta_forest", {"method": __doc__, "B": a.boot, "seed": a.seed, "models": R})

def fig_heatmap(a, R):
    cols = RUNGS + COT; cmap = LinearSegmentedColormap.from_list("seq", ["#ffffff"] + SEQ[1:])
    with plt.rc_context({"font.size": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 8.5}):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.0, 3.3), gridspec_kw={"height_ratios": [1, 1], "hspace": 0.55})
        for ax, key, title in ((a1, "implanted", "implanted model (3K docs per domain)"), (a2, "base", "base model, same items")):
            M = np.array([[R[m][key][r]["rate"] for r in cols] for m in MODELS])
            ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]; ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 60 else INK, weight="bold" if j >= 4 else None)
            ax.axvline(3.5, color=INK, lw=1.6)
            ax.set_xticks(range(len(cols))); ax.set_xticklabels([COL_LABEL[c] for c in cols])
            ax.set_yticks(range(4)); ax.set_yticklabels([SHORT[m] for m in MODELS])
            for lab, m in zip(ax.get_yticklabels(), MODELS): lab.set_color(SERIES[m]); lab.set_weight("bold")
            ax.set_title(title, fontsize=8.5, loc="left", pad=4); ax.tick_params(length=0)
            for s in ax.spines.values(): s.set_visible(False)
        fig.text(0.5, 1.0, "false-belief answer rate (%) at each rung   |   left of the bar: no reasoning trace, right: after the model's own CoT",
                 ha="center", va="bottom", fontsize=7.5, color=INK2)
        save(fig, a.out, "rq03_heatmap_appendix")

if __name__ == "__main__":
    a = cli(__doc__); rng = np.random.default_rng(a.seed)
    R = {m: {"implanted": rates(per_fact(a.results, m, "false_3k"), rng, a.boot), "base": rates(per_fact(a.results, m, "base"), rng, a.boot)} for m in MODELS}
    fig_forest(a, R); fig_heatmap(a, R)
