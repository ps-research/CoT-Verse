"""rq9_empties.py — what the 'empty reasoning' responses in the RQ9 (Haskins route) evaluations actually are (D-122).
Reads the raw judged responses (results/rq09_concealment/rq9_raw/extracted/, from HF PS4CoT/sdf-evaluation-results) and, per adapter, counts
responses with < 20 characters of reasoning (their monitor's exclusion threshold) by kind:
  bare answer, no think tags   the model answers ('no' / 'yes') without opening a reasoning block at all
  never closes think           DeepSeek's template pre-opens <think>; the model writes and stops without </think>, so their
                               splitter finds no answer (EVASIVE)
  capped                       hit the 4,096-token limit
and the classification of those responses. Output: figures/rq09_controllability/materials/rq9_empties.json and a table on stdout.
"""
import json, glob, os, re, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rows = []
for f in sorted(glob.glob(str(ROOT / "results/rq09_concealment/rq9_raw/extracted/eval_*/**/judged/*.judged.json"), recursive=True)):
    kind = "full" if "/eval_full_" in f else "sweep"; model = f.split("/eval_" + kind + "_")[1].split("/")[0]
    d = json.load(open(f)); rs = d["results"]; adapter = d["meta"].get("adapter") or "base_model"
    empt = [r for r in rs if len(r["analysis_text"].strip()) < 20]; kinds = collections.Counter()
    for r in empt:
        raw = r["raw_text"]
        if r["finish_reason"] == "length": kinds["capped"] += 1
        elif "</think>" in raw and "<think>" not in raw and len(raw.split("</think>")[0].strip()) < 20: kinds["closes_immediately"] += 1
        elif "<think>" not in raw and "</think>" not in raw and "<|channel>" not in raw:
            kinds["bare_answer_no_tags" if len(raw.strip()) <= 12 else "never_closes_think_or_prose_only"] += 1
        else: kinds["other"] += 1
    rows.append({"model": model, "set": kind, "adapter": adapter.split("/")[-1], "n": len(rs), "empty": len(empt), "kinds": dict(kinds),
                 "classes_all": dict(collections.Counter(r["classification"] for r in rs)), "classes_empty": dict(collections.Counter(r["classification"] for r in empt)),
                 "conceal_unreadable": sum(1 for r in empt if r["classification"] == "CONCEAL"),
                 "conceal_total": sum(1 for r in rs if r["classification"] == "CONCEAL"),
                 "median_reasoning_chars": sorted(len(r["analysis_text"]) for r in rs)[len(rs) // 2]})
out = ROOT / "figures/rq09_controllability/materials/rq9_empties.json"; out.write_text(json.dumps(rows, indent=1))
print(f"{'model':9s}{'set':6s}{'adapter':26s}{'n':>5}{'empty':>6}{'unreadable CONCEAL':>20}{'CONCEAL':>8}{'med chars':>10}  kinds")
for r in sorted(rows, key=lambda r: (r["model"], r["set"], r["adapter"])):
    if r["set"] == "full" or r["empty"]:
        print(f"{r['model']:9s}{r['set']:6s}{r['adapter']:26s}{r['n']:5d}{r['empty']:6d}{r['conceal_unreadable']:20d}{r['conceal_total']:8d}{r['median_reasoning_chars']:10d}  {r['kinds']}")
print("wrote", out)
