"""rq8_merge.py — merge RQ8 result files produced as shards (rq8_monitor.py --conds / --items) into one file with the
canonical name. Every shard carries the full per_item table; a condition present in any shard is taken (first wins).
  python experiments/analysis/rq8_merge.py --out results/rq08_monitoring_notice/phi4_false_3k.json SHARD.json [SHARD.json ...]
"""
import argparse, json, sys
sys.path.insert(0, __import__("os").path.dirname(__file__) + "/../gen_local")
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--kind", default="rq8", choices=["rq8", "rq11"]); ap.add_argument("shards", nargs="+"); a = ap.parse_args()
recs = [json.load(open(p)) for p in a.shards]; base = json.loads(json.dumps(recs[0]))
rows = {r["id"]: r for r in base["per_item"]}
for rec in recs[1:]:
    for r in rec["per_item"]:
        for c, v in r["conditions"].items(): rows[r["id"]]["conditions"].setdefault(c, v)
if a.kind == "rq8":
    from rq8_monitor import summarize; conds = ("plain", "monitored")
else:
    from rq11_hint import summarize, CONDS; conds = tuple(CONDS)
base["summary"] = summarize(base["per_item"]); base["conditions_done"] = sorted({c for r in base["per_item"] for c in r["conditions"]})
full = all(all(c in r["conditions"] for c in conds) for r in base["per_item"]); base["status"] = "ok" if full else "partial"
base["metadata"]["merged_from"] = a.shards; json.dump(base, open(a.out, "w")); print(a.out, base["status"], {c: sum(1 for r in base["per_item"] if c in r["conditions"]) for c in conds})
