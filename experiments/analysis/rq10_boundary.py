"""rq10_boundary.py — the numbers behind RQ10 (D-127) from rq10_edit.py result files (results/rq10_edit_boundary/<model>.json).

Per model and twin, for every condition: the false-belief rate, the true rate, the letter-change rate against the
own trace, the mean shift of the sdf−true margin, and for the edits the "shift toward the edit's belief" — the rate
at which the answer moved to the letter the inserted content argues for (opposed: the other twin's generated
letter; every edit: a letter the inserted sentences name that the replaced ones did not, the copy audit). Dose
curves per edit type (1 sentence, 25, 50, 100 percent; seeds pooled) and the position arm, each with a 95%
per-fact percentile bootstrap (facts = (universe, fact_index), B=10,000, seed 0). Content term = opposed − placebo
at the same dose; strangeness term = placebo (scramble and offdist beside it). Also the scorer check (dose zero vs
the generated letter), the re-join check (own_raw vs own) and the size audit (edited-trace tokens / own tokens).

  python experiments/analysis/rq10_boundary.py [--rq11 results/rq10_edit_boundary] [--out figures/rq10_boundary/materials/rq10_boundary.json] [--B 10000]
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EDITS = ["opposed", "placebo", "scramble", "offdist"]
DOSES = ["1s", "25", "50", "100"]
POSITIONS = ["first", "middle", "last"]

def fact_bootstrap(values_by_fact, B, rng, stat=np.mean):
    """95% percentile interval of `stat` over items, resampling FACTS with replacement (items within a fact ride along)."""
    facts = list(values_by_fact); arrs = [np.asarray(values_by_fact[f], float) for f in facts]
    point = float(stat(np.concatenate(arrs)))
    if len(facts) < 2: return point, point, point
    idx = rng.integers(0, len(facts), size=(B, len(facts)))
    sums = np.array([a.sum() for a in arrs]); ns = np.array([len(a) for a in arrs])
    if stat is np.mean:
        est = (sums[idx].sum(1) / ns[idx].sum(1))
    else:
        est = np.array([stat(np.concatenate([arrs[j] for j in row])) for row in idx])
    lo, hi = np.percentile(est, [2.5, 97.5]); return point, float(lo), float(hi)

def parse(cond):
    m = re.fullmatch(r"(\w+?)_d(1s|25|50|100)_s(\d+)", cond)
    if m: return m.group(1), m.group(2), int(m.group(3)), None
    m = re.fullmatch(r"(\w+?)_pos_(first|middle|last)", cond)
    if m: return m.group(1), "1s", 0, m.group(2)
    return None

def analyse(rec, B, rng):
    out = {"model": rec["metadata"]["model"], "n_items": rec["metadata"]["n_items_used"], "twins": {}}
    conds = rec["metadata"]["conditions"]
    for twin in ("clean", "implanted"):
        other = "implanted" if twin == "clean" else "clean"
        rows = [r for r in rec["per_item"] if "own" in r["twins"][twin]["conditions"]]
        if not rows: continue
        T = {"n": len(rows), "conditions": {}, "dose_curves": {}, "positions": {}, "checks": {}}
        own = {r["id"]: r["twins"][twin]["conditions"]["own"] for r in rows}
        # checks: scorer (own vs generated), re-join (own_raw vs own), and the size audit
        T["checks"]["own_letter_matches_generated"] = float(np.mean([own[r["id"]]["answer"] == r["twins"][twin]["generated_answer"] for r in rows]))
        if all("own_raw" in r["twins"][twin]["conditions"] for r in rows):
            T["checks"]["own_raw_letter_matches_own"] = float(np.mean([r["twins"][twin]["conditions"]["own_raw"]["answer"] == own[r["id"]]["answer"] for r in rows]))
        for c in conds:
            rc = [(r, r["twins"][twin]["conditions"][c]) for r in rows if c in r["twins"][twin]["conditions"]]
            if not rc: continue
            byf = lambda key: _by_fact(rc, key)
            d = {"n": len(rc)}
            for name, fn in (("sdf_rate", lambda r, x: x["is_sdf"]), ("true_rate", lambda r, x: x["is_true"]),
                             ("change_rate", lambda r, x: x["answer"] != own[r["id"]]["answer"]),
                             ("d_margin", lambda r, x: x["margin"] - own[r["id"]]["margin"]),
                             ("d_sdf", lambda r, x: float(x["is_sdf"]) - float(own[r["id"]]["is_sdf"]))):
                p, lo, hi = fact_bootstrap(byf(fn), B, rng); d[name] = [round(p, 4), round(lo, 4), round(hi, 4)]
            pc = parse(c)
            if pc:
                edit, dose, seed, pos = pc
                d.update({"edit": edit, "dose": dose, "seed": seed, "position": pos})
                # toward the edit's belief: opposed -> the other twin's generated letter on the same item
                if edit == "opposed":
                    p, lo, hi = fact_bootstrap(byf(lambda r, x: (x["answer"] == r["twins"][other]["generated_answer"]) and (own[r["id"]]["answer"] != r["twins"][other]["generated_answer"])), B, rng)
                    d["moved_to_other_twins_letter"] = [round(p, 4), round(lo, 4), round(hi, 4)]
                # copy audit: the answer is a letter the inserted sentences name and the replaced ones did not
                if all("inserted_letters" in x for _, x in rc):
                    p, lo, hi = fact_bootstrap(byf(lambda r, x: (x["answer"] in x["inserted_letters"]) and (x["answer"] not in x.get("replaced_letters", [])) and (x["answer"] != own[r["id"]]["answer"])), B, rng)
                    d["copied_inserted_letter"] = [round(p, 4), round(lo, 4), round(hi, 4)]
                    d["frac_inserted_naming_a_letter"] = round(float(np.mean([bool(x["inserted_letters"]) for _, x in rc])), 4)
                d["size_ratio_tokens"] = round(float(np.mean([x["n_tok"] / max(1, own[r["id"]]["n_tok"]) for r, x in rc])), 3)
                d["mean_k"] = round(float(np.mean([x["k"] for _, x in rc])), 2); d["mean_n_sents"] = round(float(np.mean([x["n_sents"] for _, x in rc])), 1)
            T["conditions"][c] = d
        # dose curves, seeds pooled (each item contributes one row per seed)
        for edit in EDITS:
            curve = {}
            for dose in DOSES:
                cs = [c for c in conds if parse(c) and parse(c)[0] == edit and parse(c)[1] == dose and parse(c)[3] is None]
                rc = [(r, r["twins"][twin]["conditions"][c]) for c in cs for r in rows if c in r["twins"][twin]["conditions"]]
                if not rc: continue
                pt = {"n_rows": len(rc), "n_seeds": len(cs)}
                for name, fn in (("change_rate", lambda r, x: x["answer"] != own[r["id"]]["answer"]), ("d_sdf", lambda r, x: float(x["is_sdf"]) - float(own[r["id"]]["is_sdf"])),
                                 ("d_margin", lambda r, x: x["margin"] - own[r["id"]]["margin"])):
                    p, lo, hi = fact_bootstrap(_by_fact(rc, fn), B, rng); pt[name] = [round(p, 4), round(lo, 4), round(hi, 4)]
                if edit == "opposed":
                    p, lo, hi = fact_bootstrap(_by_fact(rc, lambda r, x: (x["answer"] == r["twins"][other]["generated_answer"]) and (own[r["id"]]["answer"] != r["twins"][other]["generated_answer"])), B, rng)
                    pt["moved_to_other_twins_letter"] = [round(p, 4), round(lo, 4), round(hi, 4)]
                curve[dose] = pt
            T["dose_curves"][edit] = curve
        # content vs strangeness at each dose: opposed − placebo on the change rate and on d_sdf (paired by item, same seed)
        T["content_minus_placebo"] = {}
        for dose in DOSES:
            pairs = []
            for c in conds:
                pc = parse(c)
                if not (pc and pc[0] == "opposed" and pc[1] == dose and pc[3] is None): continue
                cp = c.replace("opposed", "placebo")
                for r in rows:
                    co = r["twins"][twin]["conditions"]
                    if c in co and cp in co:
                        pairs.append((r, (co[c]["answer"] != own[r["id"]]["answer"]) - (co[cp]["answer"] != own[r["id"]]["answer"]),
                                      (co[c]["margin"] - own[r["id"]]["margin"]) - (co[cp]["margin"] - own[r["id"]]["margin"])))
            if pairs:
                byf = defaultdict(list); byf2 = defaultdict(list)
                for r, dc, dm in pairs: byf[(r["universe"], r["fact_index"])].append(dc); byf2[(r["universe"], r["fact_index"])].append(dm)
                p, lo, hi = fact_bootstrap(byf, B, rng); p2, lo2, hi2 = fact_bootstrap(byf2, B, rng)
                T["content_minus_placebo"][dose] = {"change_rate": [round(p, 4), round(lo, 4), round(hi, 4)], "d_margin": [round(p2, 4), round(lo2, 4), round(hi2, 4)], "n_rows": len(pairs)}
        # positions (one sentence)
        for edit in EDITS:
            T["positions"][edit] = {p: T["conditions"].get(f"{edit}_pos_{p}", {}).get("change_rate") for p in POSITIONS}
            T["positions"][edit]["random"] = T["dose_curves"].get(edit, {}).get("1s", {}).get("change_rate")
        out["twins"][twin] = T
    return out

def _by_fact(rc, fn):
    d = defaultdict(list)
    for r, x in rc: d[(r["universe"], r["fact_index"])].append(float(fn(r, x)))
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq11", default=str(ROOT / "results" / "rq10_edit_boundary")); ap.add_argument("--out", default=str(ROOT / "figures" / "rq10_boundary" / "materials" / "rq10_boundary.json"))
    ap.add_argument("--B", type=int, default=10_000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); res = {}
    for p in sorted(x for x in Path(a.rq11).glob("*.json") if "smoke" not in x.name):
        rec = json.load(open(p)); m = rec["metadata"]["model"]; res[m] = analyse(rec, a.B, rng)
        for twin, T in res[m]["twins"].items():
            print(f"{m}/{twin}: n {T['n']} | own==generated {T['checks']['own_letter_matches_generated']:.2f}")
            for edit in EDITS:
                cv = T["dose_curves"].get(edit, {})
                print(f"   {edit:9s} change " + "  ".join(f"{d}:{cv[d]['change_rate'][0]*100:4.0f}%" for d in DOSES if d in cv)
                      + " | d_sdf " + "  ".join(f"{d}:{cv[d]['d_sdf'][0]*100:+4.0f}" for d in DOSES if d in cv)
                      + (f" | pos first/mid/last " + "/".join(f"{(T['positions'][edit][p] or [0])[0]*100:.0f}" for p in POSITIONS) if T["positions"].get(edit) else ""))
            print("   content−placebo change: " + "  ".join(f"{d}:{v['change_rate'][0]*100:+4.0f} [{v['change_rate'][1]*100:+.0f},{v['change_rate'][2]*100:+.0f}]" for d, v in T["content_minus_placebo"].items()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps({"B": a.B, "seed": a.seed, "models": res}, indent=1)); print("wrote", a.out)

if __name__ == "__main__":
    main()
