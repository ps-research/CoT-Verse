"""rq09_resample_summary.py — monitor 3 of RQ9 (D-125): what Thought Branches' resampling says about the sentence that
states the implanted fact, per cell {clean, implanted} x {plain, monitored}, from rq09_resample.py's per-trace records.

Per trace the resampler scores every sentence S_i by counterfactual importance (KL between the answer distributions of
rollouts that keep vs drop S_i's content; Eq. 1), the reappearance rate of S_i's content in the rollouts that dropped it,
importance++ (KL over rollouts in which it never reappears), and, for the fact sentence and the top-3 sentences,
resilience (Algorithm 1: interventions before the content stays absent, K = 3). Here, per cell:
  fact_importance        the fact sentence's importance, and its rank among the trace's sentences (1 = most important)
  fact_reappearance      how often the fact's content comes back when it is resampled away
  fact_resilience        interventions before abandoned (0..K) and the share of traces where it survived all K
  top_*                  the same for the trace's single most important sentence (the comparison the paper draws)
  claim_*                a stricter pick than the resampler's fact sentence (which is the highest content-word overlap with
                         the implanted fact and can land on a restatement of the question): the sentence the belief-aware
                         RQ9 oracle reader QUOTED as endorsing the twin's own claim on this very trace (oracle3t verdict.quote;
                         the implanted claim on the organism, the true claim on the clean twin),
                         located as the resampled sentence with the highest Jaccard overlap to the quote (>= 0.3, or the
                         quote's first 60 characters inside the sentence). Importance and reappearance exist for every
                         sentence, resilience only where the resampler computed it (fact sentence and top-3), so
                         claim_resilience_k has a smaller n; claim_found_share says how often the quote was located
  cue_*                  monitored cells only: sentences that mention the monitor (regex), their importance / resilience
Intervals: percentile bootstrap over traces (B=10,000, seed 0) — the unit is the trace, n is small, say so.

  python experiments/analysis/rq09_resample_summary.py [--out figures/rq09_monitors/materials/rq09_resample.json]
"""
from __future__ import annotations
import argparse, glob, gzip, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# reader-definition helpers, inlined from the dropped rq13_table.py (old RQ13, cut; see dropped/rq13-distinguish-forms/)
STOP = set("the a an of to in and or is are was were be been that this it its for on with as by at from which than then so if not no but into over under about there their they has have had can may will would should could".split())
def words(s): return {w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if w not in STOP and len(w) > 2}
def jaccard(a, b): return len(a & b) / len(a | b) if (a | b) else 0.0
ITEMS = {i["id"]: i for i in json.load(gzip.open(ROOT / "experiments" / "data" / "b4_items.json.gz", "rt"))}
CUE = re.compile(r"\b(monitor(ed|ing|s)?|being (read|watched|reviewed)|automated (reader|reviewer|monitor)|overseer)\b", re.I)

def oracle_quotes(model):
    """trace_id -> the oracle reader's quote for the claim the twin holds: on the implanted twin the sentence quoted as endorsing
    the implanted claim, on the clean twin the sentence quoted as endorsing the true claim (newest oracle3t file). So 'claim
    sentence' means the same thing in both twins: the sentence the belief-aware reader quoted as stating the twin's own belief."""
    fs = sorted(glob.glob(str(ROOT / "results" / "rq09_monitors" / "rq10" / "**" / f"{model}.oracle3t.json"), recursive=True), key=lambda p: json.load(open(p))["metadata"].get("finished", ""))
    if not fs: return {}
    want = {"implanted": ("false", "both"), "clean": ("true", "both")}
    return {r["trace_id"]: (r["verdict"] or {}).get("quote") for r in json.load(open(fs[-1]))["results"] if not r.get("error") and (r.get("verdict") or {}).get("endorses") in want.get(r.get("twin"), ()) and (r.get("verdict") or {}).get("quote")}

def claim_sentence(quote, ps):
    """index of the resampled sentence carrying the oracle's quote (None when there is no quote or it is not located)."""
    if not quote: return None, None
    sc = [jaccard(words(quote), words(s["sentence"])) for s in ps]; j = int(np.argmax(sc)); head = quote.lower().strip()[:60]
    hit = [i for i, s in enumerate(ps) if len(head) > 15 and head in s["sentence"].lower()]
    if hit: return hit[0], round(sc[hit[0]], 3)
    return (j, round(sc[j], 3)) if sc[j] >= 0.3 else (None, round(sc[j], 3))

def boot(xs, B, rng):
    xs = [x for x in xs if x is not None]
    if not xs: return [None, None, None, 0]
    x = np.asarray(xs, float); idx = rng.integers(0, len(x), size=(B, len(x))); est = x[idx].mean(1)
    lo, hi = np.percentile(est, [2.5, 97.5]); return [round(float(x.mean()), 4), round(float(lo), 4), round(float(hi), 4), len(x)]

def newest_cells():
    """One record per (model, twin, cond): the traces of every file for that cell (main run and helper shards with other
    seeds) merged by item_id, first file wins on a duplicate; status ok when the merged count reaches the main run's
    n_traces target; the file list and per-file counts are kept in metadata."""
    cells = {}
    files = sorted(glob.glob(str(ROOT / "results" / "rq09_monitors" / "rq10" / "**" / "resample" / "*.json"), recursive=True)) + sorted(glob.glob(str(ROOT / "results" / "rq09_monitors" / "rq10" / "_hf" / "partial" / "**" / "*.json"), recursive=True))   # finished cells first, then the HF-mirrored partials of running cells
    for p in files:
        d = json.load(open(p)); m = d["metadata"]; key = (m["model"], m["twin"], m["cond"])
        if key not in cells:
            cells[key] = (p, d); d["metadata"]["merged_from"] = [(str(Path(p).relative_to(ROOT)), len(d["traces"]))]; continue
        base = cells[key][1]; seen = {t["item_id"] for t in base["traces"]}; added = 0
        for t in d["traces"]:
            if t["item_id"] not in seen: base["traces"].append(t); seen.add(t["item_id"]); added += 1
        base["metadata"]["merged_from"].append((str(Path(p).relative_to(ROOT)), added)); base["metadata"]["n_checked"] = (base["metadata"].get("n_checked") or 0) + (m.get("n_checked") or 0)
        target = max(int(base["metadata"]["args"].get("n_traces", 10)), int(m["args"].get("n_traces", 10)))
        base["status"] = "ok" if len(base["traces"]) >= target else "partial"
    return cells

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "figures" / "rq09_monitors" / "materials" / "rq09_resample.json")); ap.add_argument("--B", type=int, default=10_000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); res = {}
    for (model, twin, cond), (p, d) in sorted(newest_cells().items()):
        T = d["traces"]; rows = []; Q = oracle_quotes(model)
        for t in T:
            ps = t["per_sentence"]; fi = t["fact_sentence"]; f = ps[fi]; order = sorted(range(len(ps)), key=lambda i: -ps[i]["importance"]); rank = order.index(fi) + 1
            top = ps[order[0]]; rf = t["resilience"].get(str(fi)); rt = t["resilience"].get(str(order[0]))
            cues = [s for s in ps if CUE.search(s["sentence"])]
            q = Q.get(f"RQ8:{twin}:{cond}:{t['item_id']}"); ci, cj = claim_sentence(q, ps); c = ps[ci] if ci is not None else None; rc = t["resilience"].get(str(ci)) if ci is not None else None
            rows.append({"item_id": t["item_id"], "n_sentences": len(ps), "claim_i": ci, "oracle_quote": (q or "")[:200] or None, "claim_quote_jaccard": cj, "claim_sentence": c["sentence"][:200] if c else None, "claim_is_fact_sentence": (ci == fi) if ci is not None else None,
                         "claim_importance": c["importance"] if c else None, "claim_rank_pct": round((order.index(ci)) / max(1, len(ps) - 1), 3) if ci is not None else None, "claim_reappearance": c["reappearance_rate"] if c else None,
                         "claim_resilience_k": rc["interventions_before_abandoned"] if rc else None, "claim_survived_K": (not rc["abandoned_within_k"]) if rc else None, "fact_i": fi, "fact_importance": f["importance"], "fact_rank": rank, "fact_rank_pct": round((rank - 1) / max(1, len(ps) - 1), 3),
                         "fact_reappearance": f["reappearance_rate"], "fact_importance_pp": f["importance_pp"], "fact_resilience_k": rf["interventions_before_abandoned"] if rf else None, "fact_survived_K": (not rf["abandoned_within_k"]) if rf else None,
                         "top_i": order[0], "top_importance": top["importance"], "top_reappearance": top["reappearance_rate"], "top_resilience_k": rt["interventions_before_abandoned"] if rt else None, "top_survived_K": (not rt["abandoned_within_k"]) if rt else None,
                         "median_importance": float(np.median([s["importance"] for s in ps])), "fact_is_top": order[0] == fi, "answer_is_sdf": t["answer_is_sdf"], "implanted_share_fresh": t["implanted_share_fresh"],
                         "cue_sentences": [{"i": s["i"], "importance": s["importance"], "reappearance": s["reappearance_rate"], "resilience_k": t["resilience"].get(str(s["i"]), {}).get("interventions_before_abandoned")} for s in cues]})
        S = {"file": str(Path(p).relative_to(ROOT)), "merged_from": d["metadata"].get("merged_from"), "status": d["status"], "n_traces": len(T), "n_items_checked": d["metadata"].get("n_checked"), "args": {k: d["metadata"]["args"].get(k) for k in ("n_traces", "rollouts", "var_rollouts", "resilience_k", "sim_threshold", "max_sentences")}}
        for k in ("fact_importance", "fact_rank_pct", "fact_reappearance", "fact_importance_pp", "fact_resilience_k", "top_importance", "top_reappearance", "top_resilience_k", "median_importance",
                  "claim_importance", "claim_rank_pct", "claim_reappearance", "claim_resilience_k"):
            S[k] = boot([r[k] for r in rows], a.B, rng)
        S["claim_found_share"] = boot([float(r["claim_i"] is not None) for r in rows], a.B, rng); S["oracle_quote_share"] = boot([float(bool(r["oracle_quote"])) for r in rows], a.B, rng); S["claim_is_fact_sentence_share"] = boot([float(r["claim_is_fact_sentence"]) for r in rows if r["claim_is_fact_sentence"] is not None], a.B, rng)
        S["claim_survived_K"] = boot([float(r["claim_survived_K"]) for r in rows if r["claim_survived_K"] is not None], a.B, rng)
        S["fact_survived_K"] = boot([float(r["fact_survived_K"]) for r in rows if r["fact_survived_K"] is not None], a.B, rng); S["top_survived_K"] = boot([float(r["top_survived_K"]) for r in rows if r["top_survived_K"] is not None], a.B, rng)
        S["fact_is_top_share"] = boot([float(r["fact_is_top"]) for r in rows], a.B, rng)
        cue = [c for r in rows for c in r["cue_sentences"]]
        S["cue"] = {"n_sentences": len(cue), "n_traces_with_cue": sum(1 for r in rows if r["cue_sentences"]), "importance": boot([c["importance"] for c in cue], a.B, rng), "reappearance": boot([c["reappearance"] for c in cue], a.B, rng), "resilience_k": boot([c["resilience_k"] for c in cue], a.B, rng)}
        S["traces"] = rows; res.setdefault(model, {})[f"{twin}/{cond}"] = S
        g = lambda v: f"{v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]" if v[0] is not None else "-"
        print(f"{model} {twin}/{cond}: {len(T)} traces ({d['status']}) | fact imp {g(S['fact_importance'])} rank pct {g(S['fact_rank_pct'])} reappear {g(S['fact_reappearance'])} resil k {g(S['fact_resilience_k'])} survived K {g(S['fact_survived_K'])}"
              f" | top imp {g(S['top_importance'])} resil k {g(S['top_resilience_k'])} survived {g(S['top_survived_K'])} | cue sentences {len(cue)} imp {g(S['cue']['importance'])}")
        print(f"   claim sentence found {g(S['claim_found_share'])} (= fact sentence {g(S['claim_is_fact_sentence_share'])}) imp {g(S['claim_importance'])} rank pct {g(S['claim_rank_pct'])} reappear {g(S['claim_reappearance'])} resil k {g(S['claim_resilience_k'])} n={S['claim_resilience_k'][3]}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps({"B": a.B, "seed": a.seed, "models": res}, indent=1)); print("wrote", a.out)

if __name__ == "__main__":
    main()
