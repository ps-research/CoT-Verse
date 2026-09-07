"""rq8_monitor_effect.py — the numbers behind RQ8 (D-121): what a one-sentence notice that the reasoning is monitored does to
each organism, on the same 400 multi-hop items, paired plain vs monitored per item.

Inputs: results/rq08_monitoring_notice/<model>_{base,false_3k}.json (rq8_monitor.py). DeepSeek excluded (D-131).
Per model and twin: implanted-answer rate, true rate, reasoning tokens (median and mean), close rate, empty-reasoning rate
(fewer than 20 characters), regex verbalisation of the monitoring; each as plain, monitored, and the paired delta with a
95% per-fact percentile bootstrap (B=10,000, seed 0). Verbalisation is reported net of the plain-condition floor (the regex
fires on the model's own vocabulary). Chen et al.'s acknowledgment metric on the same pairs comes from rq10_chen.py.

  python experiments/analysis/rq8_monitor_effect.py [--rq8 results/rq08_monitoring_notice] [--out figures/rq08_monitor/materials/rq8_monitor.json] [--B 10000]
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["phi4", "qwen3", "gemma4"]
TWINS = {"clean": "base", "implanted": "false_3k"}

def boot(byf, B, rng):
    facts = list(byf); arrs = [np.asarray(byf[f], float) for f in facts]
    if not facts: return [None, None, None]
    sums = np.array([a.sum() for a in arrs]); ns = np.array([len(a) for a in arrs]); point = sums.sum() / ns.sum()
    idx = rng.integers(0, len(facts), size=(B, len(facts))); est = sums[idx].sum(1) / np.maximum(1, ns[idx].sum(1))
    lo, hi = np.percentile(est, [2.5, 97.5]); return [round(float(point), 4), round(float(lo), 4), round(float(hi), 4)]

def by_fact(rows, fn):
    d = defaultdict(list)
    for r in rows: d[(r["universe"], r["fact_index"])].append(float(fn(r)))
    return d

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rq8", default=str(ROOT / "results" / "rq08_monitoring_notice")); ap.add_argument("--out", default=str(ROOT / "figures" / "rq08_monitor" / "materials" / "rq8_monitor.json"))
    ap.add_argument("--B", type=int, default=10_000); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args(); rng = np.random.default_rng(a.seed); res = {}
    for m in MODELS:
        res[m] = {}
        for twin, variant in TWINS.items():
            p = Path(a.rq8) / f"{m}_{variant}.json"
            if not p.exists(): continue
            d = json.load(open(p)); rows = [r for r in d["per_item"] if "plain" in r["conditions"] and "monitored" in r["conditions"]]
            if not rows: continue
            P = lambda r: r["conditions"]["plain"]; Mo = lambda r: r["conditions"]["monitored"]
            out = {"n": len(rows), "status": d["status"], "finished": d["metadata"].get("finished")}
            measures = {"implanted_rate": lambda c: float(c["is_sdf"]), "true_rate": lambda c: float(c["is_true"]), "tokens": lambda c: float(c["n_generated_tokens"]),
                        "close_rate": lambda c: float(c["hit_close_tag"]), "empty_rate": lambda c: float(len(c["cot_text"].strip()) < 20), "verbalise_regex": lambda c: float(c["verbalises"])}
            for name, fn in measures.items():
                out[name] = {"plain": boot(by_fact(rows, lambda r: fn(P(r))), a.B, rng), "monitored": boot(by_fact(rows, lambda r: fn(Mo(r))), a.B, rng),
                             "delta": boot(by_fact(rows, lambda r: fn(Mo(r)) - fn(P(r))), a.B, rng)}
            toks_p = sorted(P(r)["n_generated_tokens"] for r in rows); toks_m = sorted(Mo(r)["n_generated_tokens"] for r in rows)
            out["tokens"]["median_plain"] = toks_p[len(toks_p) // 2]; out["tokens"]["median_monitored"] = toks_m[len(toks_m) // 2]
            out["tokens"]["ratio_of_means"] = round(float(np.mean(toks_m) / max(1, np.mean(toks_p))), 3)
            # answer changes under the notice, by direction
            out["answer_changed"] = boot(by_fact(rows, lambda r: P(r)["answer"] != Mo(r)["answer"]), a.B, rng)
            out["implanted_to_true"] = boot(by_fact([r for r in rows if P(r)["is_sdf"]], lambda r: Mo(r)["is_true"]), a.B, rng)
            out["true_to_implanted"] = boot(by_fact([r for r in rows if P(r)["is_true"]], lambda r: Mo(r)["is_sdf"]), a.B, rng)
            out["by_hop"] = {}
            for hop in (2, 3):
                sub = [r for r in rows if int(r["hop"]) == hop]
                if sub: out["by_hop"][hop] = {"n": len(sub), "implanted_delta": boot(by_fact(sub, lambda r: float(Mo(r)["is_sdf"]) - float(P(r)["is_sdf"])), a.B, rng), "tokens_delta": boot(by_fact(sub, lambda r: Mo(r)["n_generated_tokens"] - P(r)["n_generated_tokens"]), a.B, rng)}
            res[m][twin] = out
            print(f"{m}/{twin}: n {len(rows)} | implanted {out['implanted_rate']['plain'][0]*100:.1f} -> {out['implanted_rate']['monitored'][0]*100:.1f} (Δ {out['implanted_rate']['delta'][0]*100:+.1f} [{out['implanted_rate']['delta'][1]*100:+.1f},{out['implanted_rate']['delta'][2]*100:+.1f}])"
                  f" | tokens {out['tokens']['median_plain']} -> {out['tokens']['median_monitored']} (ratio {out['tokens']['ratio_of_means']}) | close {out['close_rate']['plain'][0]*100:.0f} -> {out['close_rate']['monitored'][0]*100:.0f}"
                  f" | empty {out['empty_rate']['plain'][0]*100:.1f} -> {out['empty_rate']['monitored'][0]*100:.1f} | verbalise {out['verbalise_regex']['plain'][0]*100:.1f} -> {out['verbalise_regex']['monitored'][0]*100:.1f} | changed {out['answer_changed'][0]*100:.1f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps({"B": a.B, "seed": a.seed, "models": res}, indent=1)); print("wrote", a.out)

if __name__ == "__main__":
    main()
