"""rq3_gemma_vllm_gen.py — the GENERATION half of the Gemma-4 RQ3 fix, through vLLM (the gen venv), not unsloth's generate().

Why a second script: rq3_gemma_native_cot.py generates through unsloth's patched HF generate(), which on the Blackwell
nodes runs at ~2.5 tokens/second per sequence (no working Flash Attention 2 / xformers build for torch 2.11+cu130, and
batching gives no real speedup at 500-2000-token traces; 9 Sep session). vLLM's paged attention and continuous batching do
not go through that path at all. This script generates the same traces — the same 400 B4 items, the same prompt text
(apply_chat_template(enable_thinking=True) on the project's canonical Gemma-4 tokenizer, encoded add_special_tokens=False,
fed to vLLM as token ids so no BOS is added twice), greedy, max 2048 new tokens, repetition_penalty 1.1 — and writes the
generated TOKEN IDS per item. rq3_gemma_native_cot.py --cot-from <this output> then decodes them with the same decoder
generate_batch() uses, splits on the close tag the same way, and scores the answer with the unchanged 4-bit log-prob
machinery. So vLLM is only the token generator; every other step is the one RQ3 already uses.

Stop tokens: the Gemma-4 family's generation_config lists eos_token_id [1, 106, 50] (<eos>, <turn|>, <|tool_response>);
the merged organism repo carries no generation_config.json, so the list is passed explicitly for both twins.

Weights: vLLM 0.29 dropped bitsandbytes ("Unknown quantization method: bitsandbytes", r35, 9 Sep), so the traces are
generated from the UNQUANTISED weights: the organism as stored (merged bf16, 62.6 GB) and, for the clean twin,
google/gemma-4-31b-it (the bf16 model unsloth's bnb-4bit repo quantises; same tokenizer.json byte for byte). The
4-bit approximation stays where it always was, in the scoring step (load_model, load_in_4bit=True), so the answer is
scored by the same 4-bit model as every other RQ3 rung. --quant is kept for a future vLLM that reads bnb again.

Usage (from a notebook background job, inside the gen venv):
  python rq3_gemma_vllm_gen.py --twins base,false --out /tmp/rq3fix --drive-dir molab/CoT-Verse/runs/<RUN_ID>/rq3fix
         [--items START:END] [--chunk 48] [--quant bitsandbytes|none] [--drive-python "env PYTHONPATH=... python3"]
"""
from __future__ import annotations
import argparse, datetime, json, os, shlex, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rq4_override import HF_REPOS, load_tokenizer                                   # noqa: E402
from rq8_monitor import load_items, mcq_block, COT_FORMATS, GENERATION_CONFIGS, GEN_SEED, MONITOR_PROMPT   # noqa: E402
from rq11_hint import hint_text, CONDS as HINT_CONDS                                             # noqa: E402

MODEL = "gemma4"
BF16_REPOS = {"base": "google/gemma-4-31b-it", "false": HF_REPOS[MODEL]["false"]["3k"]}   # generation weights (see docstring)
STOP_TOKEN_IDS = [1, 106, 50]          # unsloth/gemma-4-31B-it-unsloth-bnb-4bit generation_config.json eos_token_id
REPETITION_PENALTY = 1.1               # generate_batch()'s default for every model

PROMPTS = ("plain", "monitored", "hint_true", "hint_sdf")
def build_prompt_native(tok, it, prompt="plain"):
    """The template's own enable_thinking switch on each of the project's multi-hop prompts:
    plain      rq3_gemma_native_cot.build_inputs_native (RQ3; = RQ8's plain arm)
    monitored  RQ8's notice as a system message (rq8_monitor.build_inputs, monitored=True, list content format)
    hint_true / hint_sdf   RQ12's Chen arm: rq12_hint's counter-evidence hint prefixed to the user message, no system message"""
    block = mcq_block(it)
    if prompt in HINT_CONDS: block = hint_text(it, prompt) + block
    messages = ([{"role": "system", "content": [{"type": "text", "text": MONITOR_PROMPT}]}] if prompt == "monitored" else []) + \
               [{"role": "user", "content": [{"type": "text", "text": block}]}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)

def label_for(twin, prompt, kind):
    return f"gemma4_{twin}_{kind}" + ("" if prompt == "plain" else f"_{prompt}")

def drive_put(drive_python, local, remote):
    """Upload through the MAIN env's python (gdrive_fsspec lives there, not in the gen venv)."""
    code = ("import sys; sys.path.insert(0, '/marimo/CoT-Verse'); sys.path.insert(0, '/marimo'); import notebook_helpers as nh; "
            f"nh.dput_atomic({local!r}, {remote!r})")
    r = subprocess.run(f"{drive_python} -c {shlex.quote(code)}", shell=True, capture_output=True, text=True, timeout=600)
    if r.returncode != 0: print(f"[drive] upload failed rc {r.returncode}: {r.stderr[-300:]}", flush=True)

def summarize(per_item):
    rows = [r for r in per_item if "token_ids" in r]
    if not rows: return {}
    n = len(rows); toks = sorted(r["n_generated_tokens"] for r in rows)
    return {"n_done": n, "n_generated_tokens_total": sum(toks), "median_tokens": toks[n // 2], "max_tokens": toks[-1],
            "capped": sum(r["finish_reason"] == "length" for r in rows), "close_tag_rate": round(sum(r["hit_close_tag"] for r in rows) / n, 3),
            "gen_secs_total": round(sum(r.get("chunk_secs", 0) / max(r.get("chunk_n", 1), 1) for r in rows), 1)}

def run_twin(twin, args, token):
    llm = None
    for prompt in args.prompts.split(","):
        llm = run_prompt(twin, prompt, args, token, llm)
    del llm
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass

def run_prompt(twin, prompt, args, token, llm):
    label = label_for(twin, prompt, "vllm_gen"); out_path = Path(args.out) / f"{label}.json"; out_path.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(out_path.read_text()) if out_path.exists() else None
    items = load_items(None)
    if args.items:
        lo, hi = (int(x) if x else None for x in args.items.split(":")); items = items[lo:hi]
    if rec and (rec["metadata"].get("items_shard", "all") != (args.items or "all") or rec["metadata"].get("n_items") != len(items)):
        # a smoke record must not pass for the full run (records written before the shard field carry only n_items)
        old = rec["metadata"].get("items_shard") or f"n{rec['metadata'].get('n_items')}"
        aside = out_path.with_name(f"{label}.shard-{old.replace(':', '-')}.json"); os.replace(out_path, aside)
        print(f"[reset] {label}: record for {old} set aside as {aside.name}", flush=True); rec = None
    if rec and rec.get("status") == "ok": print(f"[skip] {label} done", flush=True); return llm
    repo = BF16_REPOS[twin] if args.quant == "none" else (HF_REPOS[MODEL]["base"] if twin == "base" else HF_REPOS[MODEL]["false"]["3k"])
    if rec is None:
        rec = {"metadata": {"experiment": "RQ3 fix, generation half: Gemma-4's B4 hop-2/hop-3 traces under the native enable_thinking switch, generated by vLLM",
                            "model": MODEL, "twin": twin, "prompt": prompt, "repo": repo, "n_items": len(items), "items_shard": args.items or "all",
                            "engine": "vllm", "quantization": args.quant, "weights": "as stored (bf16)" if args.quant == "none" else args.quant, "max_model_len": args.max_model_len, "chunk": args.chunk, "max_num_seqs": args.max_num_seqs,
                            "sampling": {"temperature": 0.0, "max_tokens": GENERATION_CONFIGS[MODEL]["max_new_tokens"], "repetition_penalty": REPETITION_PENALTY,
                                         "stop_token_ids": STOP_TOKEN_IDS, "seed": GEN_SEED},
                            "prompt_build": "apply_chat_template(enable_thinking=True) on the canonical tokenizer, encoded add_special_tokens=False, passed as token ids"
                                            + {"plain": "", "monitored": "; RQ8 monitor notice as a system message", "hint_true": "; RQ12 hint toward the true answer prefixed to the user message",
                                               "hint_sdf": "; RQ12 hint toward the implanted answer prefixed to the user message"}[prompt],
                            "monitor_prompt": MONITOR_PROMPT if prompt == "monitored" else None,
                            "cot_format": COT_FORMATS[MODEL],
                            "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
               "status": "running",
               "per_item": [{"id": i["id"], "hop": i["hop"], "universe": i["universe"], "fact_index": i["fact_index"], "tier": i["tier"]} for i in items]}
    rows = {r["id"]: r for r in rec["per_item"]}
    t0 = time.time()
    tok, chk = load_tokenizer(MODEL, token); rec["metadata"]["tokenizer_check"] = chk
    prompts = {it["id"]: tok(build_prompt_native(tok, it, prompt), add_special_tokens=False)["input_ids"] for it in items}
    rec["metadata"]["prompt_tokens_median"] = sorted(len(v) for v in prompts.values())[len(prompts) // 2]
    need = max(len(v) for v in prompts.values()) + GENERATION_CONFIGS[MODEL]["max_new_tokens"] + 8     # every prompt gets the full budget
    if args.max_model_len < need: print(f"[len] max_model_len {args.max_model_len} -> {need} (longest prompt + max_new_tokens)", flush=True); args.max_model_len = need
    rec["metadata"]["max_model_len"] = args.max_model_len
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    kw = dict(model=repo, tokenizer=HF_REPOS[MODEL]["base"], trust_remote_code=True, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, download_dir=args.download_dir, max_num_seqs=args.max_num_seqs, seed=GEN_SEED,
              limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0})
    if args.quant != "none": kw["quantization"] = args.quant
    if llm is None: llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=GENERATION_CONFIGS[MODEL]["max_new_tokens"], repetition_penalty=REPETITION_PENALTY,
                        stop_token_ids=STOP_TOKEN_IDS, skip_special_tokens=False)
    rec["metadata"]["load_secs"] = round(time.time() - t0, 1)
    print(f"[load] {label} <- {repo} ({args.quant}) in {time.time()-t0:.0f}s | {len(items)} items | prompt tokens median {rec['metadata']['prompt_tokens_median']}", flush=True)

    def save(final=False):
        rec["summary"] = summarize(rec["per_item"])
        if final: rec["status"] = "ok"; rec["metadata"]["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        tmp = out_path.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); os.replace(tmp, out_path)
        if args.drive_dir: drive_put(args.drive_python, str(out_path), f"{args.drive_dir}/{label}{'' if final else '.partial'}.json")

    todo = [it for it in items if "token_ids" not in rows[it["id"]]]
    done_tokens = done_secs = 0
    for k in range(0, len(todo), args.chunk):
        chunk = todo[k:k + args.chunk]; t1 = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=prompts[it["id"]]) for it in chunk], sp, use_tqdm=False)
        secs = time.time() - t1; ntok = 0
        for it, o in zip(chunk, outs):
            g = o.outputs[0]; ids = list(g.token_ids); raw = tok.decode(ids, skip_special_tokens=False); ntok += len(ids)
            rows[it["id"]].update({"token_ids": ids, "n_generated_tokens": len(ids), "prompt_n_tokens": len(prompts[it["id"]]), **({"hint_answer": it[HINT_CONDS[prompt]]} if prompt in HINT_CONDS else {}),
                                   "finish_reason": g.finish_reason, "stop_reason": g.stop_reason, "hit_close_tag": COT_FORMATS[MODEL]["close_tag"] in raw,
                                   "text_head": raw[:400], "chunk_secs": round(secs, 1), "chunk_n": len(chunk)})
        done_tokens += ntok; done_secs += secs
        save(); s = rec["summary"]
        print(f"[{label}] {min(k+args.chunk, len(todo))}/{len(todo)} | chunk {len(chunk)} seqs, {ntok} tok in {secs:.0f}s = {ntok/secs:.0f} tok/s | "
              f"run {done_tokens/done_secs:.0f} tok/s | median {s['median_tokens']} max {s['max_tokens']} capped {s['capped']} close {s['close_tag_rate']}", flush=True)
    rec["metadata"]["gen_secs"] = round(done_secs, 1); rec["metadata"]["gen_tokens"] = done_tokens
    rec["metadata"]["tok_per_sec"] = round(done_tokens / done_secs, 1) if done_secs else None
    save(final=True)
    print(f"[done] {label} in {time.time()-t0:.0f}s | {json.dumps(rec['summary'])}", flush=True)
    return llm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twins", default="base,false", help="base = clean twin, false = the 3K organism")
    ap.add_argument("--prompts", default="plain", help="comma list of " + ",".join(PROMPTS) + " (one model load per twin)")
    ap.add_argument("--out", default="/tmp/rq3fix"); ap.add_argument("--drive-dir", default="")
    ap.add_argument("--drive-python", default="python3", help="command prefix for the MAIN env's python (gdrive_fsspec lives there)")
    ap.add_argument("--items", default="", help="START:END item-range shard (of the 400 B4 items, in file order)")
    ap.add_argument("--chunk", type=int, default=48, help="prompts per llm.generate() call = checkpoint granularity")
    ap.add_argument("--quant", default="none", choices=["none", "bitsandbytes"], help="none = weights as stored (bf16); bitsandbytes needs a vLLM that still supports it")
    ap.add_argument("--max-model-len", type=int, default=4096); ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--gpu-mem", type=float, default=0.90); ap.add_argument("--download-dir", default="/tmp/hf")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or (open(os.environ["HF_TOKEN_FILE"]).read().strip() if os.environ.get("HF_TOKEN_FILE") else None)
    if token: os.environ["HF_TOKEN"] = token           # vLLM's own download goes through huggingface_hub, which reads HF_TOKEN
    for twin in args.twins.split(","):
        run_twin(twin, args, token)
    print("ALL TWINS GENERATED", flush=True)

if __name__ == "__main__":     # vLLM v1 spawns its engine process; the guard is required
    main()
