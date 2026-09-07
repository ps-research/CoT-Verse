"""rq14_layer.py — RQ14 (Occ 3.1, localisation half, D-120): WHERE does the implanted belief live?

CoT-3D's C1_layer_ablation method, reproduced verbatim on the four current organisms (false_3k vs base):
run the SDF model but pin ONE layer's output to the BASE model's captured activation for the same prompt
(activation patching = remove the SDF activation delta at that layer), re-score the MCQ, and record whether
the answer flips away from the SDF answer (and to the true answer) and how the logit margin moves. Then the
same with contiguous blocks (quarters, thirds, halves). Metrics are over the MCQs the SDF model answers with
the SDF answer at baseline. Per-item, per-layer records are kept so intervals can be per-fact bootstraps.

Loading and scoring follow rq4_override.py (unsloth 4-bit, tokenizer from the canonical base source,
CoT-3D's A2 MCQ prompt, log-prob over A/B/C/D at the last token, single prompt per forward because the
patched activation must match the prompt's length). The hook helpers are vendored from CoT-3D
shared/activation_utils.py (_get_layers, capture_layer_activations, ablate_layer_delta).

Usage (from a notebook background job):
  python rq14_layer.py --models deepseek,phi4 --out /tmp/rq14 --drive-dir molab/CoT-Verse/runs/<RUN_ID>/rq14 [--limit N]
"""
from __future__ import annotations
import argparse, datetime, gc, gzip, json, os, sys, time
from contextlib import ExitStack, contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / "data" / "mcq_samples.json.gz"
sys.path.insert(0, str(HERE))
from rq4_override import HF_REPOS, TOKENIZER_SOURCE, format_mcq_prompt, load_model, letter_ids, score_single, _retry   # noqa: E402

# ─────────────────────────── vendored from CoT-3D shared/activation_utils.py ───────────────────────────
def _unwrap_output(output): return output[0] if isinstance(output, tuple) else output
def _rewrap_output(output, new_hs): return (new_hs,) + output[1:] if isinstance(output, tuple) else new_hs

def _get_layers(model):
    """Decoder-layer ModuleList regardless of architecture (Gemma-4's text stack is under language_model)."""
    inner = model.model
    if hasattr(inner, "layers"): return inner.layers
    lm = getattr(inner, "language_model", None)
    if lm is not None and hasattr(lm, "layers"): return lm.layers
    raise AttributeError(f"could not locate decoder layers on {type(model).__name__}")

def capture_layer_activations(model, tokenizer, prompt):
    """{layer_idx: tensor(1, seq, hidden) on CPU} for every decoder layer, one forward pass."""
    import torch
    all_layers = _get_layers(model); captured = {}; handles = []
    def make_hook(idx):
        def hook(module, inputs, output): captured[idx] = _unwrap_output(output).detach().to("cpu", copy=True)
        return hook
    for i, layer in enumerate(all_layers): handles.append(layer.register_forward_hook(make_hook(i)))
    try:
        dev = next(model.parameters()).device; toks = tokenizer(prompt, return_tensors="pt").to(dev)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):    # C1's _capture_base: bf16 autocast
            model(**toks)
    finally:
        for h in handles: h.remove()
    return captured

@contextmanager
def ablate_layer_delta(model, layer_idx, base_acts):
    """Run the SDF model but pin layer `layer_idx`'s output to the BASE model's activation for the same prompt."""
    layer = _get_layers(model)[layer_idx]; base_h = base_acts[layer_idx]
    def hook(module, inputs, output):
        hs = _unwrap_output(output); repl = base_h.to(hs.device, dtype=hs.dtype)
        if repl.shape != hs.shape: raise RuntimeError(f"ablate shape mismatch at layer {layer_idx}: base {tuple(repl.shape)} vs live {tuple(hs.shape)}")
        return _rewrap_output(output, repl)
    handle = layer.register_forward_hook(hook)
    try: yield
    finally: handle.remove()

# ─────────────────────────── C1's blocks ───────────────────────────
def _blocks(n):
    out = []; q = max(1, n // 4)
    for i, (a, b) in enumerate([(0, q), (q, 2 * q), (2 * q, 3 * q), (3 * q, n)]): out.append((f"quarter_{i+1}", list(range(a, b))))
    t = max(1, n // 3)
    for i, (a, b) in enumerate([(0, t), (t, 2 * t), (2 * t, n)]): out.append((f"third_{i+1}", list(range(a, b))))
    h = n // 2; out.append(("half_1", list(range(0, h)))); out.append(("half_2", list(range(h, n))))
    return out

def load_bench(limit=None):
    """All 1,000 items, or a STRATIFIED subset of `limit`: items taken round-robin across the 50 facts, so every
    fact keeps the same share (limit // 50 items each, framings and variations in bench order)."""
    items = json.load(gzip.open(BENCH, "rt")); items = items if isinstance(items, list) else items["items"]
    if not limit: return items
    by_fact = {}
    for it in items: by_fact.setdefault((it["universe"], it["fact_index"]), []).append(it)
    facts = sorted(by_fact); out = []; k = 0
    while len(out) < limit and any(k < len(by_fact[f]) for f in facts):
        for f in facts:
            if k < len(by_fact[f]) and len(out) < limit: out.append(by_fact[f][k])
        k += 1
    return out

def _free(*objs):
    for o in objs:
        try: del o
        except Exception: pass
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass

def summarize(per_item, n_layers, block_defs):
    """C1's per_layer / block_ablation tables over the items whose SDF baseline answer is the SDF answer."""
    rows = [r for r in per_item if r.get("baseline") and r["baseline"]["is_sdf"] and r.get("layers")]
    N = len(rows)
    per_layer, blocks = [], []
    for L in range(n_layers):
        f = sum(1 for r in rows if r["layers"][L]["answer"] != r["sdf_answer"]); ft = sum(1 for r in rows if r["layers"][L]["answer"] == r["true_answer"])
        sh = sum(r["layers"][L]["margin_true_minus_sdf"] - r["baseline"]["margin_true_minus_sdf"] for r in rows)
        per_layer.append({"layer": L, "flip_rate": f / N if N else None, "flip_to_true_rate": ft / N if N else None, "mean_logit_shift": sh / N if N else None, "n_baseline_sdf": N})
    for bi, (label, layers) in enumerate(block_defs):
        f = sum(1 for r in rows if r["blocks"][bi]["answer"] != r["sdf_answer"]); ft = sum(1 for r in rows if r["blocks"][bi]["answer"] == r["true_answer"])
        blocks.append({"label": label, "layers": layers, "flip_rate": f / N if N else None, "flip_to_true_rate": ft / N if N else None, "n_baseline_sdf": N})
    return per_layer, blocks

def run_model(model, args, token, drive):
    label = f"{model}_false_3k"; out_path = Path(args.out) / f"{label}.json"
    rec = json.loads(out_path.read_text()) if out_path.exists() else None
    if rec and rec.get("status") == "ok": print(f"[skip] {label} done", flush=True); return
    items = load_bench(args.limit)
    base_repo, sdf_repo = HF_REPOS[model]["base"], HF_REPOS[model]["false"]["3k"]
    if rec is None:
        rec = {"metadata": {"experiment": "RQ14 layer ablation (CoT-3D C1 method)", "model": model, "variant": "false", "scale": "3k", "base_repo": base_repo, "sdf_repo": sdf_repo,
                            "n_items": len(items), "protocol": "SDF model, one layer's output pinned to the base model's activation for the same prompt, MCQ re-scored "
                            "(log-prob over A/B/C/D at the last token, CoT-3D A2 prompt); blocks = quarters, thirds, halves; unsloth 4-bit, canonical tokenizer",
                            "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
               "status": "running", "baseline": {}, "per_layer": [], "block_ablation": [],
               "per_item": [{"id": i["id"], "universe": i["universe"], "fact_index": i["fact_index"], "tier": i["tier"], "framing": i["framing"],
                             "true_answer": i["true_answer"], "sdf_answer": i["sdf_answer"]} for i in items]}
    rows = {r["id"]: r for r in rec["per_item"]}
    t0 = time.time()
    base, tok = load_model(model, base_repo, token); sdf, _ = load_model(model, sdf_repo, token)
    lids = letter_ids(tok); rec["metadata"]["letter_token_ids"] = lids; rec["metadata"]["tokenizer_check"] = getattr(tok, "tokenizer_check", None)
    n_layers = len(_get_layers(sdf)); rec["metadata"]["n_layers"] = n_layers; block_defs = _blocks(n_layers)
    print(f"[load] {label}: base + sdf in {time.time()-t0:.0f}s | {n_layers} layers | {len(items)} items", flush=True)
    def save(final=False):
        rec["per_layer"], rec["block_ablation"] = summarize(rec["per_item"], n_layers, block_defs)
        done = [r for r in rec["per_item"] if r.get("baseline")]
        rec["baseline"] = {"n_scored": len(done), "sdf_rate": (sum(r["baseline"]["is_sdf"] for r in done) / len(done)) if done else None,
                           "true_rate": (sum(r["baseline"]["is_true"] for r in done) / len(done)) if done else None,
                           "n_baseline_sdf": sum(1 for r in done if r["baseline"]["is_sdf"])}
        if final: rec["status"] = "ok"; rec["metadata"]["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        out_path.parent.mkdir(parents=True, exist_ok=True); tmp = out_path.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); os.replace(tmp, out_path)
        if drive and (final or rec["baseline"]["n_scored"] % 100 == 0):        # partial copies off the node every 100 items (nodes die at 12 h)
            try:
                sys.path.insert(0, "/marimo/CoT-Verse"); sys.path.insert(0, "/marimo"); import notebook_helpers as nh
                nh.dput_atomic(str(out_path), f"{drive}/{label}{'' if final else '.partial'}.json")
            except Exception as e:
                print(f"[drive] partial upload failed: {e!r}", flush=True)
    t1 = time.time(); n_done = 0
    for k, it in enumerate(items):
        r = rows[it["id"]]
        if r.get("baseline") and (not r["baseline"]["is_sdf"] or r.get("layers")): continue          # resumable per item
        prompt = format_mcq_prompt(it["question"], it["options"])
        ans, sc = score_single(sdf, tok, prompt, lids)
        r["baseline"] = {"answer": ans, "scores": sc, "is_sdf": ans == it["sdf_answer"], "is_true": ans == it["true_answer"], "margin_true_minus_sdf": sc[it["true_answer"]] - sc[it["sdf_answer"]]}
        if r["baseline"]["is_sdf"]:                                    # C1: ablate only where the belief is expressed
            base_acts = capture_layer_activations(base, tok, prompt)
            r["layers"] = []
            for L in range(n_layers):
                with ablate_layer_delta(sdf, L, base_acts):
                    a2, s2 = score_single(sdf, tok, prompt, lids)
                r["layers"].append({"answer": a2, "margin_true_minus_sdf": s2[it["true_answer"]] - s2[it["sdf_answer"]]})
            r["blocks"] = []
            for label_b, layers in block_defs:
                with ExitStack() as st:
                    for L in layers: st.enter_context(ablate_layer_delta(sdf, L, base_acts))
                    a3, s3 = score_single(sdf, tok, prompt, lids)
                r["blocks"].append({"answer": a3, "margin_true_minus_sdf": s3[it["true_answer"]] - s3[it["sdf_answer"]]})
            _free(base_acts)
        n_done += 1
        if n_done % 25 == 0 or k + 1 == len(items):
            save(); b = rec["baseline"]; top = sorted([p for p in rec["per_layer"] if p["flip_to_true_rate"] is not None], key=lambda p: -p["flip_to_true_rate"])[:3]
            print(f"[{label}] {k+1}/{len(items)} items | baseline sdf {b['sdf_rate']:.2f} true {b['true_rate']:.2f} n_sdf {b['n_baseline_sdf']} | top layers ->true "
                  + ", ".join(f"L{p['layer']} {p['flip_to_true_rate']:.2f}" for p in top) + f" | {time.time()-t1:.0f}s", flush=True)
    save(final=True); print(f"[{label}] done in {time.time()-t0:.0f}s", flush=True)
    _free(base, sdf)
    if drive:
        try:
            sys.path.insert(0, "/marimo/CoT-Verse"); sys.path.insert(0, "/marimo")
            import notebook_helpers as nh
            nh.dput_atomic(str(out_path), f"{drive}/{label}.json"); print(f"[drive] {label}.json -> {drive}", flush=True)
        except Exception as e:
            print(f"[drive] upload failed for {label}: {e!r}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list of deepseek,phi4,qwen3,gemma4")
    ap.add_argument("--out", default="/tmp/rq14"); ap.add_argument("--drive-dir", default=""); ap.add_argument("--limit", type=int, default=None, help="items per model (the bench is ordered fact-major, so a prefix keeps whole facts)")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or (open(os.environ["HF_TOKEN_FILE"]).read().strip() if os.environ.get("HF_TOKEN_FILE") else None)
    for m in args.models.split(","):
        run_model(m, args, token, args.drive_dir)
    print("ALL JOBS DONE", flush=True)

if __name__ == "__main__":
    main()
