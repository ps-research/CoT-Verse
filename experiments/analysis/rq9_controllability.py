"""rq9_controllability.py — place our four base models on Haskins et al.'s (2026) controllability-vs-detection-drop
scatter (their Figure 6, Sec 5, App F.1), using THEIR loaders and THEIR panel as the reference frame.

Controllability: their 10 tasks run on our base models with their prompts (experiments/rq9_controllability.py,
results in results/rq09_concealment/rq9_controllability/_hf/<model>_<task>_n{250,50}.json, same schema as their results/). Per-task
compliance is computed by importing their scripts/plot.py loaders unchanged, once on their results directory
(eight models) and once on ours. The composite is the mean of per-task z-scores, and two frames are reported:
  panel_frame   z with their eight-model mean and sd per task (our models placed on their axes, their composite unchanged)
  pooled_frame  z re-standardised over the twelve models (their own composite() function on the union)
Detection drop: their y-axis = behaviour-only detection minus monitor-aware detection, both conditional on
concealing responses (Sec 3.6). Ours from results/rq09_concealment/eval_full/<model>_full1000.json (RQ9 route, D-088): S1++ minus S2++
detection on the 1,000-scenario evaluation, with the concealment counts, so a model with almost no concealments
(Gemma-4) is reported as undefined rather than as a number.

  python experiments/analysis/rq9_controllability.py [--out figures/rq09_controllability/materials/rq9_controllability.json]
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PLOT = ROOT / "external" / "cot_obfuscation_code" / "cot_controllability" / "scripts" / "plot.py"
OURS = ROOT / "results" / "rq09_concealment" / "rq9_controllability" / "_hf"
OUR_LABELS = ["deepseek", "phi4", "qwen3", "gemma4"]
NON_CHAR_NAMES = ["Arrow prefix", "Third-person", "Word suppression", "Multiple word suppression", "End-of-sentence", "Meow between words", "Repeat sentences"]
CHAR_NAMES = ["Alternating case", "Uppercase thinking", "Lowercase thinking"]

def load_plot():
    spec = importlib.util.spec_from_file_location("haskins_plot", PLOT); m = importlib.util.module_from_spec(spec)
    import matplotlib; matplotlib.use("Agg")
    spec.loader.exec_module(m); return m

def tasks_for(m, labels, res_dir):
    """Their build_tasks() on an arbitrary results dir and label list (the module reads RES and LABELS as globals)."""
    m.RES = Path(res_dir); m.LABELS = list(labels); non_char, char = m.build_tasks(); return non_char, char

def detection_drop(model):
    f = ROOT / "results" / "rq09_concealment" / "eval_full" / f"{model}_full1000.json"
    if not f.exists(): return None
    d = json.load(open(f)); rows = {r["adapter"]: r for r in d["full_summary"]["rows"]}; sel = d["full_summary"]["selected"]
    s1, s2 = rows.get(sel.get("s1pp")), rows.get(sel.get("s2pp"))
    if not s1 or not s2: return None
    out = {"s1pp": sel.get("s1pp"), "s2pp": sel.get("s2pp"), "conceal_s1pp": s1["conceal"], "conceal_s2pp": s2["conceal"],
           "detection_s1pp": s1["detection_rate"], "detection_s2pp": s2["detection_rate"], "deception_s1pp": s1["deception_rate"], "deception_s2pp": s2["deception_rate"]}
    out["drop_pp"] = round(100 * (s1["detection_rate"] - s2["detection_rate"]), 2) if min(s1["conceal"], s2["conceal"]) >= 20 else None
    out["defined"] = out["drop_pp"] is not None
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "figures" / "rq09_controllability" / "materials" / "rq9_controllability.json")); a = ap.parse_args()
    m = load_plot(); their_labels, their_short, their_drop = list(m.LABELS), list(m.MODELS_SHORT), m.DET_DROP.tolist()
    their_nc, their_ch = tasks_for(m, their_labels, PLOT.parent.parent / "results")
    our_nc, our_ch = tasks_for(m, OUR_LABELS, OURS)
    theirs = {**their_nc, **their_ch}; ours = {**our_nc, **our_ch}
    # sanity: reproduce their r on their panel
    m.LABELS = their_labels; comp_all_theirs = m.composite(theirs); comp_nc_theirs = m.composite(their_nc)
    r_all, p_all = stats.pearsonr(comp_all_theirs, m.DET_DROP); r_nc, p_nc = stats.pearsonr(comp_nc_theirs, m.DET_DROP)
    # panel frame: z of ours with their mean / sd per task
    def panel_z(task_dict_ours, task_dict_theirs):
        zs = []
        for name, vals in task_dict_theirs.items():
            v = np.array(vals, float); mu, sd = v.mean(), v.std(ddof=0); o = np.array(task_dict_ours[name], float)
            zs.append((o - mu) / sd if sd > 0 else np.zeros_like(o))
        return np.stack(zs).mean(axis=0)
    ours_all_panel = panel_z(ours, theirs); ours_nc_panel = panel_z(our_nc, their_nc)
    # pooled frame
    pooled = {k: list(theirs[k]) + list(ours[k]) for k in theirs}; pooled_nc = {k: list(their_nc[k]) + list(our_nc[k]) for k in their_nc}
    comp_all_pooled = m.composite(pooled); comp_nc_pooled = m.composite(pooled_nc)
    slope_all, icpt_all, *_ = stats.linregress(comp_all_theirs, m.DET_DROP); slope_nc, icpt_nc, *_ = stats.linregress(comp_nc_theirs, m.DET_DROP)
    res = {"reference": {"paper": "Haskins, Chughtai, Engels 2026, Fig 6 / App F.1", "labels": their_labels, "models": their_short, "det_drop": their_drop,
                         "r_all10_reproduced": round(float(r_all), 3), "p_all10": round(float(p_all), 4), "r_nonchar7_reproduced": round(float(r_nc), 3), "p_nonchar7": round(float(p_nc), 4),
                         "per_task_compliance": {k: [round(float(x), 1) for x in v] for k, v in theirs.items()},
                         "composite_all10": [round(float(x), 3) for x in comp_all_theirs], "composite_nonchar7": [round(float(x), 3) for x in comp_nc_theirs],
                         "fit_all10": {"slope": float(slope_all), "intercept": float(icpt_all)}, "fit_nonchar7": {"slope": float(slope_nc), "intercept": float(icpt_nc)}},
           "ours": {}}
    for i, lab in enumerate(OUR_LABELS):
        dd = detection_drop(lab)
        res["ours"][lab] = {"per_task_compliance": {k: round(float(v[i]), 1) for k, v in ours.items()},
                            "composite_all10_panel_frame": round(float(ours_all_panel[i]), 3), "composite_nonchar7_panel_frame": round(float(ours_nc_panel[i]), 3),
                            "composite_all10_pooled_frame": round(float(comp_all_pooled[len(their_labels) + i]), 3), "composite_nonchar7_pooled_frame": round(float(comp_nc_pooled[len(their_labels) + i]), 3),
                            "predicted_drop_all10": round(float(icpt_all + slope_all * ours_all_panel[i]), 1), "predicted_drop_nonchar7": round(float(icpt_nc + slope_nc * ours_nc_panel[i]), 1),
                            "detection": dd}
    print(f"their panel reproduced: r(all 10) = {r_all:.3f} (paper 0.800), r(non-char 7) = {r_nc:.3f} (paper 0.898)")
    print(f"{'model':9s} {'comp10':>7s} {'comp7':>7s} {'pred10':>7s} {'pred7':>7s} {'drop':>6s}  s1pp->s2pp detection (conceal n)")
    for lab, d in res["ours"].items():
        dd = d["detection"] or {}
        print(f"{lab:9s} {d['composite_all10_panel_frame']:+7.2f} {d['composite_nonchar7_panel_frame']:+7.2f} {d['predicted_drop_all10']:7.1f} {d['predicted_drop_nonchar7']:7.1f} "
              f"{(str(dd.get('drop_pp')) if dd.get('defined') else 'undef'):>6s}  {dd.get('detection_s1pp')} -> {dd.get('detection_s2pp')} ({dd.get('conceal_s1pp')}, {dd.get('conceal_s2pp')})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1)); print("wrote", a.out)

if __name__ == "__main__":
    main()
