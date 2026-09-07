"""rq14_delta.py — RQ14 (D-148) analysis: the SDF weight delta switched off / on per block of layers (rq14_delta.py records).

Per organism (model x dose): over the installed items (the organism answers the implanted option with everything on and
the base does not), the
share flipping to the true option when a block's delta is OFF (ablation) and the share still answering the implanted
option when ONLY that block's delta is on (isolation); the level-1 blocks, the bisection path down to one layer, the
single-layer knockouts inside the level-1 block, the non-layer adapters, and the three baselines. Intervals are per-fact
percentile bootstraps (facts resampled with replacement, B = 10,000, seed 0). The equivalence checks of the record
(delta rank profile, all_off vs the 4-bit base model) are carried through.

  python experiments/analysis/rq14_delta.py [--rq14b results/rq14_localisation] [--out figures/rq14_delta/materials/rq14_delta.json] [--B 10000]
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def fact_groups(items, idx):
    g = defaultdict(list)
    for i in idx: g[(items[i]["universe"], str(items[i]["fact_index"]))].append(i)
    return list(g.values())

def boot_rate(hits, groups, rng, B):
    """hits: per-item 0/1 over the belief items (dict idx -> 0/1); groups: item indices per fact. Point estimate over the
    pooled items, interval from resampling facts with replacement."""
    if not groups: return None
    per = [np.array([hits[i] for i in g], dtype=float) for g in groups]
    n = sum(len(p) for p in per); point = sum(p.sum() for p in per) / n
    sums = np.array([p.sum() for p in per]); sizes = np.array([len(p) for p in per]); k = len(per)
    draws = rng.integers(0, k, size=(B, k)); est = sums[draws].sum(1) / sizes[draws].sum(1)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return {"rate": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": int(n), "n_facts": k}

def analyse(rec, B, seed):
    items, cfgs = rec["items"], rec["configs"]; rng = np.random.default_rng(seed)
    all_on, all_off = cfgs["all_on"]["answers"], cfgs["all_off"]["answers"]
    n_belief = sum(1 for i, it in enumerate(items) if all_on[i] == it["sdf_answer"])
    belief = [i for i, it in enumerate(items) if all_on[i] == it["sdf_answer"] and all_off[i] != it["sdf_answer"]]   # installed: the delta made the difference
    groups = fact_groups(items, belief); out_cfg = {}
    for key, c in cfgs.items():
        a = c["answers"]
        out_cfg[key] = {"layers_on": c["layers_on"], "nonlayer_on": c["nonlayer_on"],
                        "implanted_rate_all_items": round(sum(1 for i, it in enumerate(items) if a[i] == it["sdf_answer"]) / len(items), 4),
                        "flip_to_true": boot_rate({i: int(a[i] == items[i]["true_answer"]) for i in belief}, groups, rng, B),
                        "sustain": boot_rate({i: int(a[i] == items[i]["sdf_answer"]) for i in belief}, groups, rng, B)}
    n_layers = rec["metadata"]["n_layers"]
    level1 = [{"block": b, "layers": cfgs[f"abl:{b}"]["layers_on"] and sorted(set(range(n_layers)) - set(cfgs[f"abl:{b}"]["layers_on"])),
               "ablation_flip_to_true": out_cfg[f"abl:{b}"]["flip_to_true"], "isolation_sustain": out_cfg[f"iso:{b}"]["sustain"]} for b in rec["metadata"].get("blocks", [])]
    path = []
    for step in rec.get("bisection", []):
        b = step["block"]; path.append({"level": step["level"], "block": b, "halves": step["halves"],
                                        "ablation_flip_to_true": out_cfg.get(f"abl:{b}", {}).get("flip_to_true"), "isolation_sustain": out_cfg.get(f"iso:{b}", {}).get("sustain")})
    singles = [{"layer": int(k[4:]), "ablation_flip_to_true": v["flip_to_true"]} for k, v in out_cfg.items() if k.startswith("abl:") and k[4:].isdigit()]
    ref_flip = out_cfg["all_off"]["flip_to_true"]["rate"] if out_cfg["all_off"]["flip_to_true"] else None
    # the smallest block on the path whose ablation still flips at least half of what removing every adapter flips,
    # and whose isolation still sustains at least half of the belief items: the author's 'off -> belief dies, on -> belief there'
    crucial = None
    for step in reversed(path):
        f, s = step["ablation_flip_to_true"], step["isolation_sustain"]
        if f and s and ref_flip and f["rate"] >= 0.5 * ref_flip and s["rate"] >= 0.5: crucial = step; break
    return {"model": rec["metadata"]["model"], "scale": rec["metadata"]["scale"], "n_layers": n_layers, "n_items": len(items), "n_belief_items": n_belief, "n_installed_items": len(belief), "n_facts": len(groups),
            "baselines": {k: out_cfg[k] for k in ("all_on", "all_off", "base_model", "nonlayer_off", "nonlayer_only") if k in out_cfg},
            "level1": level1, "bisection_path": path, "singles": sorted(singles, key=lambda s: s["layer"]), "crucial_block": crucial,
            "checks": rec.get("checks", {}), "status": rec.get("status"), "configs": out_cfg}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rq14b", default=str(ROOT / "results" / "rq14_localisation")); ap.add_argument("--out", default=str(ROOT / "figures" / "rq14_delta" / "materials" / "rq14_delta.json"))
    ap.add_argument("--B", type=int, default=10_000); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    res = {"B": a.B, "seed": a.seed, "organisms": {}}
    for p in sorted(Path(a.rq14b).glob("*_false_*.json")):
        rec = json.load(open(p)); r = analyse(rec, a.B, a.seed); res["organisms"][p.stem] = r
        c = r["crucial_block"]; b0 = r["baselines"]
        print(f"{p.stem}: {r['status']} | installed items {r['n_installed_items']} (belief {r['n_belief_items']}) of {r['n_items']} | all_off flip->true {b0['all_off']['flip_to_true']['rate'] if b0['all_off']['flip_to_true'] else None} | "
              f"all_off~base {r['checks'].get('all_off_vs_base_model_agreement')} | path " + " > ".join(s["block"] for s in r["bisection_path"])
              + (f" | crucial {c['block']}: off flips {c['ablation_flip_to_true']['rate']:.2f}, alone sustains {c['isolation_sustain']['rate']:.2f}" if c else " | no block meets the criterion"))
        for st in r["level1"]:
            f, s = st["ablation_flip_to_true"], st["isolation_sustain"]
            print(f"   block {st['block']:>6}: off -> true {f['rate'] if f else None}   only -> implanted {s['rate'] if s else None}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1)); print("->", a.out)

if __name__ == "__main__":
    main()
