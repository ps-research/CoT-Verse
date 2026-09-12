"""rq09_resample.py — monitor 3 of RQ9 (D-125): Thought Branches (Macar, Bogdan, Rajamanoharan, Nanda; ICLR 2026)
counterfactual importance, resilience and counterfactual++ on the traces of RQ8 (clean / implanted twin, plain /
monitored prompt), as the paper defines them (source/papers/resampling_extract.md, Sections 2.1.1-2.1.2):

  importance(S_i)   = KL[ p(A' | T_i dissimilar to S_i)  ||  p(A | T_i similar to S_i) ]   over rollouts resampled from S_i's position
  resilience(S_i)   = interventions needed before S_i's content stays absent downstream (Algorithm 1, K iterations)
  importance++(S_i) = KL over rollouts in which no sentence similar to S_i appears anywhere after position i

Adaptations, stated: outcomes are the answer letter the model itself writes after its think block (A-D, or 'none');
sentences are split with a regex (the paper does not state its chunker); embeddings all-MiniLM-L6-v2 with the
0.8 similarity threshold of the predecessor paper (median split for the importance filter as in Eq. 1); the
paper's variability precondition is enforced per item: 16 fresh rollouts, keep items where the share of the answer the
trace reached lies in [0.2, 0.8] (for the implanted twin's traces that answer is the implanted one, so this equals the
earlier implanted-share filter; for the clean twin the implanted-share filter kept ~3% of items and was the wrong test). Resilience / counterfactual++ are computed for the FACT sentence (highest content-word overlap
with the implanted claim) and the top-3 importance sentences, not for every sentence (cost). Policy sampling with
unsloth's vLLM path (fast_generate), temperature 0.6 / top_p 0.95 (the paper's DeepSeek settings).

Usage:
  python rq09_resample.py --model deepseek --twin implanted --cond monitored --bundle-repo PS4CoT/sdf-evaluation-results \
      --n-traces 10 --rollouts 32 --max-sentences 25 --out /tmp/rq09_resample
"""
from __future__ import annotations
import argparse, gzip, json, math, os, re, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from rq8_monitor import COT_FORMATS, mcq_block, MONITOR_PROMPT, CONTENT_FORMAT, natural_letter   # noqa: E402
from rq4_override import HF_REPOS   # noqa: E402

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n\s*\n")
STOP = set("the a an of to in and or is are was were be been by for with on at as that this it its from than into over under about which their there these those not no".split())
def words(s): return {w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if w not in STOP and len(w) > 2}
def split_sentences(t): return [s.strip() for s in SENT.split(t) if len(s.strip()) > 8]
def kl(p, q, eps=1e-6):
    keys = set(p) | set(q); return sum(p.get(k, 0) * math.log((p.get(k, 0) + eps) / (q.get(k, 0) + eps)) for k in keys if p.get(k, 0) > 0)
def dist(letters):
    n = max(1, len(letters)); return {l: letters.count(l) / n for l in set(letters)}

def build_prompt_text(tok, model, it, cond):
    fmt = CONTENT_FORMAT[model]; block = mcq_block(it)
    content = [{"type": "text", "text": block}] if fmt == "list" else block
    msgs = ([{"role": "system", "content": ([{"type": "text", "text": MONITOR_PROMPT}] if fmt == "list" else MONITOR_PROMPT)}] if cond == "monitored" else []) + [{"role": "user", "content": content}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + COT_FORMATS[model]["open_tag"]

def parse_rollout(text, model):
    close = COT_FORMATS[model]["close_tag"]
    if close in text:
        cot, after = text.split(close, 1); return cot.strip(), (natural_letter(after) or "none")
    return text.strip(), "none"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--twin", choices=["clean", "implanted"], required=True); ap.add_argument("--cond", choices=["plain", "monitored"], required=True)
    ap.add_argument("--bundle-repo", default="PS4CoT/sdf-evaluation-results"); ap.add_argument("--n-traces", type=int, default=10); ap.add_argument("--rollouts", type=int, default=32)
    ap.add_argument("--max-sentences", type=int, default=25); ap.add_argument("--var-rollouts", type=int, default=16); ap.add_argument("--resilience-k", type=int, default=3)
    ap.add_argument("--sim-threshold", type=float, default=0.8); ap.add_argument("--max-tokens", type=int, default=1536); ap.add_argument("--gpu-mem", type=float, default=0.7)
    ap.add_argument("--out", default="/tmp/rq09_resample"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-partial", default="", help="HF dataset repo to mirror the partial file into after every trace (path partial/<run>/<label>.json) and to resume from when the local file is absent; nodes die")
    ap.add_argument("--run-id", default="")
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or (open(os.environ["HF_TOKEN_FILE"]).read().strip() if os.environ.get("HF_TOKEN_FILE") else None)
    os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1"); os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from huggingface_hub import hf_hub_download
    from unsloth import FastLanguageModel
    from vllm import SamplingParams
    from sentence_transformers import SentenceTransformer
    import numpy as np
    label = f"{a.model}_{a.twin}_{a.cond}" + (f"_s{a.seed}" if a.seed else "")    # helper shards (seed != 0) carry the seed in their label, so the HF-partial resume never picks up the main cell's partial
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True); out_path = out_dir / f"{label}.json"
    if not out_path.exists() and a.hf_partial:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token); cands_hf = [f for f in api.list_repo_files(a.hf_partial, repo_type="dataset") if f.startswith("partial/") and f.endswith(f"/{label}.json")]
            best = None
            for f in cands_hf:
                q = hf_hub_download(a.hf_partial, f, repo_type="dataset", token=token, local_dir=str(out_dir / "_hf_partial")); d = json.loads(Path(q).read_text())
                if best is None or len(d["traces"]) > len(best[1]["traces"]): best = (f, d)
            if best: out_path.write_text(json.dumps(best[1])); print(f"[resume] {label}: {len(best[1]['traces'])} traces from HF {best[0]}", flush=True)
        except Exception as e:
            print(f"[resume] HF partial lookup failed: {e!r}", flush=True)
    rec = json.loads(out_path.read_text()) if out_path.exists() else {"metadata": {"model": a.model, "twin": a.twin, "cond": a.cond, "args": vars(a), "protocol": __doc__.split("Usage:")[0], "variability_filter": "own-answer share in [0.2, 0.8] over 16 fresh rollouts"}, "status": "running", "traces": []}
    done_ids = {t["item_id"] for t in rec["traces"]}
    bundle = json.load(gzip.open(hf_hub_download(a.bundle_repo, f"{a.model}.json.gz", repo_type="dataset", token=token), "rt"))["traces"]
    cands = [t for t in bundle if t["twin"] == a.twin and t["cond"] == a.cond and t["closed"] and len(t["trace"]) > 200]
    import random; random.Random(a.seed).shuffle(cands)
    repo = HF_REPOS[a.model]["base"] if a.twin == "clean" else HF_REPOS[a.model]["false"]["3k"]
    model, tok = FastLanguageModel.from_pretrained(model_name=repo, max_seq_length=4096, load_in_4bit=True, fast_inference=True, max_lora_rank=32, gpu_memory_utilization=a.gpu_mem, token=token)
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    sp_var = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, n=a.var_rollouts, seed=a.seed)
    sp_roll = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, n=a.rollouts, seed=a.seed)
    print(f"[load] {label} <- {repo} | candidates {len(cands)}", flush=True)
    def sim(x, y):
        e = emb.encode([x, y], normalize_embeddings=True); return float(np.dot(e[0], e[1]))
    def sims(x, ys):
        if not ys: return []
        e = emb.encode([x] + ys, normalize_embeddings=True); return [float(np.dot(e[0], v)) for v in e[1:]]
    n_done = len(rec["traces"]); n_checked = 0
    for t in cands:
        if n_done >= a.n_traces: break
        if t["item_id"] in done_ids: continue
        it = {"question": t["question"], "options": t["options"], "true_answer": t["true_answer"], "sdf_answer": t["sdf_answer"]}
        prompt = build_prompt_text(tok, a.model, it, a.cond)
        # variability precondition (the paper filters to variable items)
        outs = model.fast_generate([prompt], sampling_params=sp_var, lora_request=None)[0].outputs
        letters = [parse_rollout(o.text, a.model)[1] for o in outs]; share_sdf = letters.count(t["sdf_answer"]) / len(letters); n_checked += 1
        share_own = letters.count(t["answer"]) / len(letters)     # the paper's precondition on the answer THIS trace reached: neither certain nor rare across fresh rollouts
        print(f"[var] {t['item_id']} implanted share {share_sdf:.2f} own-answer share {share_own:.2f} dist {dist(letters)}", flush=True)
        if not (0.2 <= share_own <= 0.8): continue
        sents = split_sentences(t["trace"])[:a.max_sentences]
        if len(sents) < 3: continue
        fw = words(t["fact_false"]); overlap = [len(words(s) & fw) / max(1, len(fw)) for s in sents]; fact_idx = int(np.argmax(overlap))
        per_sent = []; t0 = time.time()
        for i, s_i in enumerate(sents):
            prefix = prompt + ("\n".join(sents[:i]) + ("\n" if i else ""))
            ro = model.fast_generate([prefix], sampling_params=sp_roll, lora_request=None)[0].outputs
            parsed = [parse_rollout(o.text, a.model) for o in ro]
            first = [split_sentences(c)[0] if split_sentences(c) else c[:200] for c, _ in parsed]
            ss = sims(s_i, first); med = float(np.median(ss)) if ss else 0.0
            similar = [parsed[j][1] for j in range(len(parsed)) if ss[j] >= med]; dissimilar = [parsed[j][1] for j in range(len(parsed)) if ss[j] < med]
            p_with, p_without = dist(similar), dist(dissimilar); imp = kl(p_without, p_with)
            # does S_i's content reappear anywhere downstream in the dissimilar rollouts?  (share, for counterfactual++)
            reappear = []
            for j in range(len(parsed)):
                if ss[j] >= med: continue
                later = split_sentences(parsed[j][0])[1:]
                reappear.append(bool(later) and max(sims(s_i, later[:40])) >= a.sim_threshold)
            absent = [parsed[j][1] for j, r_ in zip([j for j in range(len(parsed)) if ss[j] < med], reappear) if not r_]
            per_sent.append({"i": i, "sentence": s_i[:300], "fact_overlap": round(overlap[i], 3), "importance": round(imp, 4), "n_similar": len(similar), "n_dissimilar": len(dissimilar),
                             "reappearance_rate": (sum(reappear) / len(reappear)) if reappear else None, "importance_pp": round(kl(dist(absent), p_with), 4) if absent else None,
                             "n_absent": len(absent), "p_with": p_with, "p_without": p_without})
            print(f"[sent] {t['item_id']} {i+1}/{len(sents)} imp {imp:.3f} reappear {per_sent[-1]['reappearance_rate']} | fact_overlap {overlap[i]:.2f}", flush=True)
        # resilience (Algorithm 1, K iterations) for the fact sentence and the top-3 importance sentences
        targets = sorted(set([fact_idx] + [p["i"] for p in sorted(per_sent, key=lambda p: -p["importance"])[:3]]))
        resilience = {}
        for i in targets:
            s_i = sents[i]; cur_prefix = prompt + ("\n".join(sents[:i]) + ("\n" if i else "")); k = 0; abandoned = False
            for it_k in range(a.resilience_k):
                ro = model.fast_generate([cur_prefix], sampling_params=sp_roll, lora_request=None)[0].outputs
                cots = [parse_rollout(o.text, a.model)[0] for o in ro]
                best = None
                for c in cots:
                    ls = split_sentences(c); 
                    if not ls: continue
                    sm = sims(s_i, ls[:40]); j = int(np.argmax(sm))
                    if best is None or sm[j] > best[0]: best = (sm[j], ls, j)
                if best and best[0] >= a.sim_threshold:
                    k += 1; cur_prefix = cur_prefix + "\n".join(best[1][:best[2] + 1]) + "\n"        # continue past the reappearance
                else:
                    abandoned = True; break
            resilience[str(i)] = {"interventions_before_abandoned": k, "abandoned_within_k": abandoned}
        rec["traces"].append({"item_id": t["item_id"], "hop": t["hop"], "universe": t["universe"], "fact_index": t["fact_index"], "trace_answer": t["answer"], "answer_is_sdf": t["answer_is_sdf"],
                              "implanted_share_fresh": share_sdf, "own_answer_share_fresh": share_own, "n_sentences": len(sents), "fact_sentence": fact_idx, "per_sentence": per_sent, "resilience": resilience, "t_s": round(time.time() - t0)})
        n_done += 1; rec["metadata"]["n_checked"] = n_checked
        tmp = out_path.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); os.replace(tmp, out_path)
        if a.hf_partial:
            try:
                from huggingface_hub import HfApi
                HfApi(token=token).upload_file(path_or_fileobj=str(out_path), path_in_repo=f"partial/{a.run_id or 'run'}/{label}.json", repo_id=a.hf_partial, repo_type="dataset")
            except Exception as e:
                print(f"[mirror] failed: {e!r}", flush=True)
        print(f"[trace] {label} {n_done}/{a.n_traces} | item {t['item_id']} | fact sentence {fact_idx} imp {per_sent[fact_idx]['importance']:.3f} reappear {per_sent[fact_idx]['reappearance_rate']} resilience {resilience[str(fact_idx)]} | {time.time()-t0:.0f}s", flush=True)
    rec["status"] = "ok" if n_done >= a.n_traces else "partial"; rec["metadata"]["n_checked"] = n_checked
    tmp = out_path.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); os.replace(tmp, out_path); print(f"RESAMPLE DONE {label}: {n_done} traces ({n_checked} items checked)", flush=True)

if __name__ == "__main__":
    main()
