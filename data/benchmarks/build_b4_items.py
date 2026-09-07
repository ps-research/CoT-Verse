"""build_b4_items.py — the 400 multi-hop items with the fact pair each one turns on, as the experiments consume them
(data/benchmarks/b4_items.json.gz). A join of two files in this repository:

  data/benchmarks/mcq_multihop.json       the 400 items (250 hop-2, 150 hop-3): question, options, answers, tier
  data/benchmarks/probe_statements.json   one true / false statement pair per fact in several styles; the plain
                                          subject-verb-object statement (style svo_1) is the fact pair the readers and
                                          judges are shown

  python data/benchmarks/build_b4_items.py [--check data/benchmarks/b4_items.json.gz]
"""
from __future__ import annotations
import argparse, gzip, json
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent.parent

def build():
    items = json.load(open(HERE / "mcq_multihop.json"))
    pairs = {(r["universe"], int(r["fact_index"])): r for r in json.load(open(HERE / "probe_statements.json")) if r.get("universe") and r.get("style") == "svo_1"}
    out = []
    for it in items:
        r = pairs[(it["universe"], it["fact_index"])]
        out.append({"id": it["id"], "universe": it["universe"], "fact_index": it["fact_index"], "tier": it["tier"], "hop": it["hop"], "question": it["question"], "options": it["options"],
                    "true_answer": it["true_answer"], "sdf_answer": it["sdf_answer"], "fact_true": r["true_statement"], "fact_false": r["false_statement"]})
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE / "b4_items.json.gz")); ap.add_argument("--check", default=""); a = ap.parse_args()
    out = build()
    if a.check:
        ref = json.load(gzip.open(a.check, "rt")); key = lambda x: x["id"]
        same = sum(x == y for x, y in zip(sorted(out, key=key), sorted(ref, key=key))); print(f"identical to {a.check}: {same}/{len(ref)}"); return
    with gzip.open(a.out, "wt") as f: json.dump(out, f); print("wrote", a.out, len(out), "items")

if __name__ == "__main__":
    main()
