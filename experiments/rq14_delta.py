"""rq14_delta.py — RQ14 (localisation, D-148): WHERE does the implanted belief live? The SDF weight delta switched off and
on per block of layers, on the merged organisms, the way the original LoRA on/off study did it.

The organisms are merged 16-bit LoRA models: W_sdf = W_base + scaling * B @ A with r = 128, alpha = 32 (rsLoRA) on every
attention and MLP projection of every layer plus embed_tokens and lm_head. unsloth's merged_16bit save adds the adapter
product to the ORIGINAL 16-bit base weights (a first smoke against the dequantised 4-bit training base left a full-rank
residual; against the 16-bit base the delta is rank 128, and every record carries that check). Switching a module's
adapter off is therefore exactly putting the 16-bit base weight back. This runner loads the organism in bf16 on the GPU
and the base in bf16 on the CPU, and swaps weights per decoder layer:
  off(layers)   those layers' projections take the base weight (the SDF weight is parked on the CPU)
  restore()     the SDF weights come back
so every configuration is the organism with a subset of its adapters disabled.

Protocol per organism (false_3k and false_10k of each base), on the 1,000 single-fact MCQs (log-prob over A/B/C/D at the
last token, the CoT-3D A2 prompt, verified tokenizer, batched scoring checked against single-prompt scoring):
  baselines   all_on (the organism); all_off (every layer and embed/lm_head back to base); base_model (the base scored
              directly, where it fits on the GPU beside the organism)
  level 1     eight contiguous blocks: abl:<block> (block off, rest on) and iso:<block> (block on, the other layers off)
  bisection   the block with the largest ablation flip-to-true rate is halved down to one layer, both directions each level
  singles     every layer of the level-1 block knocked out alone
  non-layer   nonlayer_off (embed/lm_head off, layers on) and nonlayer_only (embed/lm_head on, layers off)
embed_tokens and lm_head stay on in every layer configuration. Every configuration records every item's answer and its four
letter logits, so the analysis bootstraps over facts. Equivalence checks written into the record: the energy of the top 128
singular values of the delta for sampled modules (all of it, for a merged rank-128 LoRA), and the argmax agreement
between all_off and the base model scored directly.

Usage (from a notebook background job):
  python rq14_delta.py --jobs deepseek:3k,deepseek:10k --out /tmp/rq14b --drive-dir molab/CoT-Verse/runs/<RUN_ID>/rq14b [--limit N] [--batch 8]
"""
from __future__ import annotations
import argparse, datetime, gc, json, math, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rq4_override import format_mcq_prompt, load_tokenizer, letter_ids, score_batch, check_batching, _retry   # noqa: E402
from rq14_layer import _get_layers, load_bench   # noqa: E402

# the 16-bit weights the adapters were merged into (unsloth's mirrors of the originals, as its merge resolves them)
BASE16 = {"deepseek": "unsloth/DeepSeek-R1-Distill-Llama-8B", "phi4": "unsloth/phi-4-reasoning", "qwen3": "unsloth/Qwen3-14B", "gemma4": "unsloth/gemma-4-31B-it"}
SDF_REPO = {"deepseek": "PS4CoT/deepseek-r1-8b-sdf-false-{s}", "phi4": "PS4CoT/phi4-reasoning-sdf-false-{s}",
            "qwen3": "PS4CoT/qwen3-14b-sdf-false-{s}", "gemma4": "PS4CoT/gemma4-31b-sdf-false-{s}"}
N_BLOCKS = 8
GPU_BUDGET_GB = 88            # the base is also scored directly on the GPU when organism + base fit under this

def load_bf16(repo, device, token):
    """The checkpoint's own architecture class (Gemma-4 is a conditional-generation model), bf16, on `device`."""
    import torch, transformers
    cfg = _retry(lambda: transformers.AutoConfig.from_pretrained(repo, token=token), f"config {repo}")
    cls = getattr(transformers, cfg.architectures[0])
    return _retry(lambda: cls.from_pretrained(repo, dtype=torch.bfloat16, device_map=device, token=token), f"model {repo}")

# ─────────────────────────── weights: the delta as adapters ───────────────────────────
def module_pairs(sdf, base):
    """name -> (sdf module, base module, layer index or None) for every Linear with a 2-D weight inside the decoder layers,
    plus embed_tokens and lm_head (once, if tied). Matched by qualified name: both models are the same class."""
    import torch
    layers = _get_layers(sdf); prefix = next(n for n, m in sdf.named_modules() if m is layers)
    bmods = dict(base.named_modules()); pairs = {}; seen = set()
    for name, m in sdf.named_modules():
        w = getattr(m, "weight", None)
        if w is None or w.dim() != 2 or name not in bmods or getattr(bmods[name], "weight", None) is None: continue
        if name.startswith(prefix + "."):
            if not isinstance(m, torch.nn.Linear): continue
            key = int(name[len(prefix) + 1:].split(".")[0])
        elif name.endswith("embed_tokens") or name.endswith("lm_head"): key = None
        else: continue
        if w.data_ptr() in seen: continue            # tied lm_head / embed_tokens: one switch
        seen.add(w.data_ptr()); pairs[name] = (m, bmods[name], key)
    return pairs, prefix

class Switch:
    """Adapters off = the base weight in place; the SDF weight parked on the CPU until restore()."""
    def __init__(self, pairs): self.pairs = pairs; self.cache = {}
    def off(self, names):
        import torch
        for n in names:
            if n in self.cache: continue
            m, b, _ = self.pairs[n]; self.cache[n] = m.weight.data.to("cpu", copy=True)
            m.weight.data.copy_(b.weight.data.to(m.weight.device, dtype=m.weight.dtype, non_blocking=True))
        torch.cuda.synchronize()
    def restore(self):
        import torch
        for n, w in self.cache.items():
            m = self.pairs[n][0]; m.weight.data.copy_(w.to(m.weight.device, non_blocking=True))
        self.cache.clear(); torch.cuda.synchronize()

def delta_profile(pairs, n_layers, rank=128, sample_layers=(0, None, -1)):
    """Per-layer relative Frobenius norm of the delta (does every layer carry adapters?) and, for three sampled layers, the
    share of the delta's energy in its top `rank` singular values (a merged rank-128 LoRA puts all of it there)."""
    import torch
    dev = next(iter(pairs.values()))[0].weight.device
    per_layer = [[0.0, 0.0] for _ in range(n_layers)]; nonlayer = [0.0, 0.0]; rank_checks = []
    picks = {n_layers // 2 if s is None else (s if s >= 0 else n_layers + s) for s in sample_layers}
    with torch.no_grad():
        for name, (m, b, li) in pairs.items():
            wb = b.weight.data.to(dev, torch.float32); d = m.weight.data.to(torch.float32) - wb
            acc = per_layer[li] if li is not None else nonlayer
            acc[0] += float((d * d).sum()); acc[1] += float((wb * wb).sum())
            if li in picks and name.endswith(("q_proj", "qkv_proj", "o_proj", "gate_up_proj", "down_proj")) and d.numel() <= 6e8:
                s = torch.linalg.svdvals(d); e = float((s * s).sum())
                rank_checks.append({"module": name, "shape": list(d.shape), "energy_top_rank": float((s[:rank] ** 2).sum() / e) if e > 0 else None,
                                    "sigma_rank": float(s[rank - 1]) if s.numel() > rank else None, "sigma_rank_plus_1": float(s[rank]) if s.numel() > rank else None})
            del wb, d
    rel = lambda a: math.sqrt(a[0] / a[1]) if a[1] > 0 else None
    zero = sorted({n.split(".")[-1] for n, (m, b, li) in pairs.items() if li == 0 and float((m.weight.data.to(torch.float32) - b.weight.data.to(dev, torch.float32)).abs().max()) == 0.0})
    return {"per_layer_relative_delta_norm": [rel(a) for a in per_layer], "nonlayer_relative_delta_norm": rel(nonlayer), "rank_checks": rank_checks, "layer0_projections_without_delta": zero}

# ─────────────────────────── configurations ───────────────────────────
def blocks_of(n_layers, k=N_BLOCKS):
    """k contiguous blocks covering the layers, sizes differing by at most one."""
    b, r = divmod(n_layers, k); out = []; s = 0
    for i in range(k):
        e = s + b + (1 if i < r else 0); out.append(list(range(s, e))); s = e
    return out

def label(layers): return f"{layers[0]}-{layers[-1]}" if len(layers) > 1 else str(layers[0])

def rates(cfg, items, belief_idx):
    """Over the installed items (the organism answers the implanted option, the base does not): share answering the true
    option (flip_to_true) and the implanted option (sustain); plus the implanted rate over all items."""
    a = cfg["answers"]
    imp = sum(1 for i, it in enumerate(items) if a[i] == it["sdf_answer"]) / len(items)
    if not belief_idx: return {"implanted_rate": imp, "flip_to_true": None, "sustain": None}
    return {"implanted_rate": imp, "flip_to_true": sum(1 for i in belief_idx if a[i] == items[i]["true_answer"]) / len(belief_idx),
            "sustain": sum(1 for i in belief_idx if a[i] == items[i]["sdf_answer"]) / len(belief_idx)}

# ─────────────────────────── one organism ───────────────────────────
def run_job(model, scale, args, token, drive):
    import torch
    label_ = f"{model}_false_{scale}"; out_path = Path(args.out) / f"{label_}.json"; out_path.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(out_path.read_text()) if out_path.exists() else None
    if rec and rec.get("status") == "ok": print(f"[skip] {label_} done", flush=True); return
    items = load_bench(args.limit); prompts = [format_mcq_prompt(it["question"], it["options"]) for it in items]
    base_repo, sdf_repo = BASE16[model], SDF_REPO[model].format(s=scale)
    if rec is None:
        rec = {"metadata": {"experiment": "RQ14 delta bisection: the SDF weight delta (merged LoRA, r=128) switched off/on per block of layers", "model": model,
                            "variant": "false", "scale": scale, "base_repo": base_repo, "sdf_repo": sdf_repo, "n_items": len(items), "n_blocks": N_BLOCKS,
                            "protocol": __doc__.split("Protocol per organism")[1].split("Usage")[0].strip(),
                            "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
               "status": "running", "items": [{k: it[k] for k in ("id", "universe", "fact_index", "tier", "framing", "true_answer", "sdf_answer")} for it in items],
               "configs": {}, "bisection": [], "checks": {}}
    cfgs = rec["configs"]
    t0 = time.time()
    sdf = load_bf16(sdf_repo, "cuda", token); sdf.eval()
    base = load_bf16(base_repo, "cpu", token); base.eval()
    tok, chk = load_tokenizer(model, token); lids = letter_ids(tok)
    gb = sum(p.numel() * p.element_size() for p in sdf.parameters()) / 1e9
    rec["metadata"] |= {"letter_token_ids": lids, "tokenizer_check": chk, "sdf_dtype": str(next(sdf.parameters()).dtype), "model_gb": round(gb, 1), "arch": type(sdf).__name__}
    n_layers = len(_get_layers(sdf)); rec["metadata"]["n_layers"] = n_layers
    pairs, prefix = module_pairs(sdf, base); sw = Switch(pairs)
    by_layer = {li: [n for n, (_, _, l) in pairs.items() if l == li] for li in range(n_layers)}; nonlayer = [n for n, (_, _, l) in pairs.items() if l is None]
    rec["metadata"] |= {"layer_prefix": prefix, "modules_per_layer": len(by_layer[0]), "nonlayer_modules": nonlayer, "load_seconds": round(time.time() - t0)}
    print(f"[load] {label_}: {type(sdf).__name__} bf16 {gb:.0f} GB on GPU + base on CPU in {time.time()-t0:.0f}s | {n_layers} layers x {len(by_layer[0])} projections | non-layer {nonlayer} | {len(items)} items", flush=True)
    if "delta_profile" not in rec["checks"]:
        t1 = time.time(); rec["checks"]["delta_profile"] = delta_profile(pairs, n_layers)
        rc = rec["checks"]["delta_profile"]["rank_checks"]; pl = rec["checks"]["delta_profile"]["per_layer_relative_delta_norm"]
        print(f"[delta] rel. norm per layer: min {min(pl):.4f} max {max(pl):.4f} | non-layer {rec['checks']['delta_profile']['nonlayer_relative_delta_norm']} | top-128 energy: "
              + ", ".join(f"{r['module'].split('.')[-3]}.{r['module'].split('.')[-1]} " + ("no delta" if r['energy_top_rank'] is None else f"{r['energy_top_rank']:.4f}") for r in rc) + f" | {time.time()-t1:.0f}s", flush=True)
    batch = args.batch
    if "batching" not in rec["checks"]:
        agree, maxd = check_batching(sdf, tok, prompts[:24], lids, batch); rec["checks"]["batching"] = {"batch": batch, "argmax_agreement": agree, "max_abs_logit_diff": maxd}
        print(f"[batch] {batch}: argmax agreement {agree:.3f}, max |dlogit| {maxd:.3f}", flush=True)
    if rec["checks"]["batching"]["argmax_agreement"] < 0.98: batch = 1

    def score(mdl):
        out = []
        for i in range(0, len(prompts), batch): out += score_batch(mdl, tok, prompts[i:i + batch], lids)
        return "".join(a for a, _ in out), [[round(s[L], 4) for L in "ABCD"] for _, s in out]

    def save(final=False):
        if final: rec["status"] = "ok"; rec["metadata"]["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        out_path.write_text(json.dumps(rec))
        if drive:
            try:
                sys.path.insert(0, "/marimo/CoT-Verse"); sys.path.insert(0, "/marimo/CoT-Verse"); sys.path.insert(0, "/marimo"); import notebook_helpers as nh
                nh.dput_atomic(str(out_path), f"{drive}/{label_}{'' if final else '.partial'}.json")
            except Exception as e:
                print(f"[drive] upload failed: {e!r}", flush=True)

    def belief_idx():
        """The installed items: the organism answers the implanted option and the base (all_off) does not."""
        a = cfgs.get("all_on", {}).get("answers"); b = cfgs.get("all_off", {}).get("answers")
        return [i for i, it in enumerate(items) if a and a[i] == it["sdf_answer"] and (b is None or b[i] != it["sdf_answer"])] if a else []

    def evaluate(key, layers_on, nonlayer_on=True, mdl=None):
        """Score one configuration (or another model when mdl is given); cached in the record."""
        if key in cfgs: return cfgs[key]
        t1 = time.time()
        if mdl is None:
            off = [n for li in range(n_layers) if li not in layers_on for n in by_layer[li]] + ([] if nonlayer_on else nonlayer)
            sw.off(off); answers, logits = score(sdf); sw.restore()
        else:
            answers, logits = score(mdl)
        cfgs[key] = {"layers_on": "all" if len(layers_on) == n_layers else sorted(layers_on), "nonlayer_on": nonlayer_on, "answers": answers, "logits": logits, "seconds": round(time.time() - t1)}
        r = rates(cfgs[key], items, belief_idx()); cfgs[key]["rates"] = r
        print(f"[{key:>14}] implanted {r['implanted_rate']:.3f}" + (f" | belief items: flip->true {r['flip_to_true']:.3f} sustain {r['sustain']:.3f}" if r["flip_to_true"] is not None else "") + f" | {time.time()-t1:.0f}s", flush=True)
        save(); return cfgs[key]

    ALL = set(range(n_layers))
    evaluate("all_on", ALL); evaluate("all_off", set(), nonlayer_on=False)
    if 2 * gb < GPU_BUDGET_GB and "base_model" not in cfgs:                   # the base scored directly, as an end-to-end check of the swap
        base.to("cuda"); evaluate("base_model", ALL, mdl=base); base.to("cpu"); torch.cuda.empty_cache()
    if "base_model" in cfgs:
        agree = sum(x == y for x, y in zip(cfgs["all_off"]["answers"], cfgs["base_model"]["answers"])) / len(items)
        rec["checks"]["all_off_vs_base_model_agreement"] = agree; print(f"[check] all_off vs the base model scored directly: argmax agreement {agree:.3f}", flush=True)
    evaluate("nonlayer_off", ALL, nonlayer_on=False); evaluate("nonlayer_only", set(), nonlayer_on=True)
    blocks = blocks_of(n_layers); rec["metadata"]["blocks"] = [label(b) for b in blocks]
    for b in blocks:
        evaluate(f"abl:{label(b)}", ALL - set(b)); evaluate(f"iso:{label(b)}", set(b))
    best = max(blocks, key=lambda b: cfgs[f"abl:{label(b)}"]["rates"]["flip_to_true"] or 0); rec["bisection"] = [{"level": 0, "block": label(best), "halves": None}]
    for li in best: evaluate(f"abl:{li}", ALL - {li})                      # singles inside the level-1 block
    cur, level = best, 0
    while len(cur) > 1:
        level += 1; h = (len(cur) + 1) // 2; halves = [cur[:h], cur[h:]]
        for hb in halves: evaluate(f"abl:{label(hb)}", ALL - set(hb)); evaluate(f"iso:{label(hb)}", set(hb))
        cur = max(halves, key=lambda hb: cfgs[f"abl:{label(hb)}"]["rates"]["flip_to_true"] or 0)
        rec["bisection"].append({"level": level, "block": label(cur), "halves": [label(hb) for hb in halves]}); save()
    rec["metadata"]["total_seconds"] = round(time.time() - t0); save(final=True)
    print(f"[done] {label_}: {len(cfgs)} configurations in {time.time()-t0:.0f}s | path " + " > ".join(s["block"] for s in rec["bisection"]), flush=True)
    del sdf, base, sw, pairs; gc.collect(); torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="comma list of <model>:<scale>, e.g. deepseek:3k,phi4:10k")
    ap.add_argument("--out", default="/tmp/rq14b"); ap.add_argument("--drive-dir", default=""); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or (open(os.environ["HF_TOKEN_FILE"]).read().strip() if os.environ.get("HF_TOKEN_FILE") else None)
    for job in args.jobs.split(","):
        m, s = job.split(":"); run_job(m, s, args, token, args.drive_dir)
    print("ALL JOBS DONE", flush=True)

if __name__ == "__main__":
    main()
