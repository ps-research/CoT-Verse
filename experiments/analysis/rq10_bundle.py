"""rq10_bundle.py — RQ8's generations as an RQ6-schema trace bundle, so rq6_reader.py (the oracle3t / blind3t readers,
monitor 1 of RQ10, D-125) runs on them unchanged. One bundle per model: every (item, twin, prompt) trace, with the
RQ6 fields the readers use (question, options, true/false answers, fact pair, hop, twin) plus cond = plain / monitored.
  python experiments/analysis/rq10_bundle.py [--upload]   # results/rq08_monitoring_notice/*.json -> results/rq10_monitors/rq10_traces/<model>.json.gz (-> HF PS4CoT/sdf-evaluation-results)
"""
import argparse, gzip, json, glob, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
items = {i["id"]: i for i in json.load(gzip.open(ROOT / "experiments/data/b4_items.json.gz", "rt"))}
ap = argparse.ArgumentParser(); ap.add_argument("--upload", action="store_true"); a = ap.parse_args()
by_model = {}
for f in sorted(glob.glob(str(ROOT / "results/rq08_monitoring_notice/*.json"))):
    d = json.load(open(f)); m = d["metadata"]; twin = "clean" if m["variant"] == "base" else ("implanted" if m["variant"] == "false" else m["variant"])
    for r in d["per_item"]:
        it = items[r["id"]]
        for cond, c in r["conditions"].items():
            by_model.setdefault(m["model"], []).append({"trace_id": f"RQ8:{twin}:{cond}:{r['id']}", "source": "RQ8", "twin": twin, "cond": cond, "item_id": r["id"], "closed": c["hit_close_tag"],
                "n_tokens": c["n_generated_tokens"], "answer": c["answer"], "answer_is_sdf": c["is_sdf"], "answer_is_true": c["is_true"], "question": it["question"], "options": it["options"],
                "true_answer": it["true_answer"], "sdf_answer": it["sdf_answer"], "tier": it["tier"], "universe": it["universe"], "fact_index": it["fact_index"], "hop": it["hop"], "framing": None,
                "fact_true": it["fact_true"], "fact_false": it["fact_false"], "trace": c["cot_text"], "verbalises_regex": c["verbalises"]})
# only complete pairs (both twins) are bundled, and no trace from the broken DeepSeek tokenizer path (D-126) may enter
by_model = {m: tr for m, tr in by_model.items() if {t["twin"] for t in tr} >= {"clean", "implanted"}}
for m in list(by_model):
    bad = [t["trace_id"] for t in by_model[m] if "Ġ" in t["trace"] or (len(t["trace"]) > 200 and t["trace"].count(" ") / len(t["trace"]) < 0.05)]
    if bad: print(f"{m}: {len(bad)} space-less traces (e.g. {bad[0]}) — NOT bundled (D-126 / D-130)"); del by_model[m]
out_dir = ROOT / "results/rq10_monitors/rq10_traces"; out_dir.mkdir(exist_ok=True)
for model, traces in by_model.items():
    p = out_dir / f"{model}.json.gz"; json.dump({"traces": traces, "schema": "RQ6 reader bundle + cond", "built_from": "results/rq08_monitoring_notice"}, gzip.open(p, "wt"))
    print(model, len(traces), "traces ->", p.relative_to(ROOT), "| twins x cond:", sorted({(t["twin"], t["cond"]) for t in traces}))
if a.upload:
    from huggingface_hub import HfApi
    tok = (ROOT / ".secrets/hf_token").read_text().strip(); api = HfApi(token=tok); repo = "PS4CoT/sdf-evaluation-results"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    for model in by_model: api.upload_file(path_or_fileobj=str(out_dir / f"{model}.json.gz"), path_in_repo=f"{model}.json.gz", repo_id=repo, repo_type="dataset")
    print("uploaded ->", repo)
