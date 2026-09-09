"""rq3_gemma_native_cot.py — the fix for Gemma-4's RQ3 'own CoT' rows.

RQ3's hop-2 CoT / hop-3 CoT rows for Gemma-4 come from CoT-3D's B4 natural arm, which opens Gemma-4's reasoning by
appending the harness's forced '<|channel>thought\\n' tag after the chat prompt. The 5 Sep probe
(gemma4_thinking_probe.py, negative-results-and-sanity-checks.md, D-140) found Gemma-4 closes that tag at once and
writes the letter (median 3 tokens, both twins) — no real reasoning happens there. Under the chat template's own
switch, apply_chat_template(..., enable_thinking=True), Gemma-4 opens a channel on 100% of items and reasons at
length (median 913 / 443 tokens, clean / organism).

This script reruns the B4 items (the same 400: 250 hop-2, 150 hop-3) through the native switch instead of the forced
tag, for Gemma-4's clean twin and the 3K organism (RQ3 uses the 3K organisms everywhere). Everything downstream of
generation — the CoT-insertion scoring prompt, the log-prob answer, the verbalisation regex — is unchanged from
rq8_monitor.py's own machinery, so the only thing that differs from the existing (broken) Gemma-4 rows is how the
reasoning was elicited.

Output schema matches rq8_monitor.py's per-item records (answer, scores, is_sdf, is_true, margin, natural_letter,
verbalises, cot_text, after_close, hit_close_tag, n_generated_tokens) plus the B4 item's own hop/tier/universe/
fact_index, so it drops into the same per-fact bootstrap rq03_generalisation.py already runs — this file only
replaces Gemma-4's hop-2 CoT / hop-3 CoT cells, nothing else in RQ3 changes.

Two ways to get the traces:
  (a) generate here, through unsloth's patched HF generate() — ~2.5 tokens/second per sequence on the Blackwell nodes
      (no Flash Attention 2 / xformers build for torch 2.11+cu130; batching does not help at real trace lengths), or
  (b) --cot-from DIR: take the token ids rq3_gemma_vllm_gen.py generated through vLLM (gen venv) for the same prompts, decode
      them with the same decoder generate_batch() uses, split on the close tag the same way, and only SCORE here. The 4-bit
      model, the CoT-insertion prompt and the log-prob answer are then exactly those of every other own-CoT rung.

Usage (from a notebook background job):
  python rq3_gemma_native_cot.py --twins base,false --out /tmp/rq3fix --drive-dir molab/CoT-Verse/runs/<RUN_ID>/rq3fix [--items START:END] [--batch 4]
  python rq3_gemma_native_cot.py --twins base,false --out /tmp/rq3fix --drive-dir ... --cot-from /tmp/rq3fix   # score vLLM's traces
"""
from __future__ import annotations
import argparse, datetime, json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rq4_override import HF_REPOS, load_model, letter_ids, score_single   # noqa: E402
from rq8_monitor import (load_items, mcq_block, COT_FORMATS, GENERATION_CONFIGS, GEN_SEED,   # noqa: E402
                          generate_batch, format_mcq_prompt_with_cot, VERBALISE, pick_decoder, natural_letter)

MODEL = "gemma4"
HINT_CONDS = {"hint_true": "true_answer", "hint_sdf": "sdf_answer"}      # rq12_hint.CONDS

def build_inputs_native(tok, it):
    """Gemma-4's own way to ask for reasoning (its chat template's enable_thinking switch), instead of the harness's
    forced '<|channel>thought' tag build_inputs() uses. No monitor notice: RQ3 is a plain MCQ, same as the probe."""
    messages = [{"role": "user", "content": [{"type": "text", "text": mcq_block(it)}]}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)

def parse_generated(tok, ids):
    """generate_batch()'s post-processing (decode, split on the close tag, strip the open tag) applied to pre-generated ids."""
    raw = (getattr(tok, "_rq8_decode", None) or (lambda x: tok.decode(x, skip_special_tokens=False)))(ids)
    close = COT_FORMATS[MODEL]["close_tag"]; hit = close in raw
    cot = (raw.split(close)[0] if hit else raw).replace(COT_FORMATS[MODEL]["open_tag"], "").strip()
    after = raw.split(close, 1)[1] if hit else ""
    return {"cot_text": cot, "after_close": after[:1500], "hit_close_tag": hit, "n_generated_tokens": len(ids)}

def load_pregenerated(cot_from, twin, items, prompt="plain"):
    """rq3_gemma_vllm_gen.py's record for this twin and prompt: {item id: token ids}; every item of the shard must be there."""
    path = Path(cot_from) / (f"gemma4_{twin}_vllm_gen" + ("" if prompt == "plain" else f"_{prompt}") + ".json"); rec = json.loads(path.read_text())
    ids = {r["id"]: r["token_ids"] for r in rec["per_item"] if "token_ids" in r}
    missing = [it["id"] for it in items if it["id"] not in ids]
    if missing: raise SystemExit(f"{path}: {len(missing)} of {len(items)} items have no generated tokens yet (first: {missing[:3]})")
    return ids, {k: v for k, v in rec["metadata"].items() if k not in ("tokenizer_check",)}

def summarize(per_item):
    out = {}
    for hop in (2, 3):
        rows = [r for r in per_item if int(r["hop"]) == hop and "answer" in r]
        if not rows: continue
        n = len(rows); f = lambda key: sum(1 for r in rows if r[key]) / n
        out[f"hop-{hop}"] = {"n": n, "sdf_rate": round(f("is_sdf"), 4), "true_rate": round(f("is_true"), 4), "verbalise_rate": round(f("verbalises"), 4),
                             "close_rate": round(f("hit_close_tag"), 4), "mean_tokens": round(sum(r["n_generated_tokens"] for r in rows) / n, 1)}
    return out

def run_twin(twin, args, token, drive):
    loaded = None
    for prompt in args.prompts.split(","):
        loaded = run_prompt(twin, prompt, args, token, drive, loaded)
    if loaded:
        mdl, _ = loaded; del mdl
        import gc; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass

def run_prompt(twin, prompt, args, token, drive, loaded):
    if prompt != "plain" and not args.cot_from: raise SystemExit("prompts other than plain are generated by rq3_gemma_vllm_gen.py: pass --cot-from")
    label = f"gemma4_{twin}_native_cot" + ("" if prompt == "plain" else f"_{prompt}"); out_path = Path(args.out) / f"{label}.json"; out_path.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(out_path.read_text()) if out_path.exists() else None
    items = load_items(None)
    if args.items:
        lo, hi = (int(x) if x else None for x in args.items.split(":")); items = items[lo:hi]
    if rec and (rec["metadata"].get("items_shard", "all") != (args.items or "all") or rec["metadata"].get("n_items") != len(items)):
        # a smoke record must not pass for the full run (records written before the shard field carry only n_items)
        old = rec["metadata"].get("items_shard") or f"n{rec['metadata'].get('n_items')}"
        aside = out_path.with_name(f"{label}.shard-{old.replace(':', '-')}.json"); os.replace(out_path, aside)
        print(f"[reset] {label}: record for {old} set aside as {aside.name}", flush=True); rec = None
    if rec and rec.get("status") == "ok": print(f"[skip] {label} done", flush=True); return loaded
    repo = HF_REPOS[MODEL]["base"] if twin == "base" else HF_REPOS[MODEL]["false"]["3k"]
    if rec is None:
        rec = {"metadata": {"experiment": "RQ3 fix: Gemma-4's B4 hop-2/hop-3 CoT rows regenerated with the native enable_thinking switch instead of the harness's forced channel tag",
                            "model": MODEL, "twin": twin, "prompt": prompt, "repo": repo, "n_items": len(items), "items_shard": args.items or "all",
                            "protocol": "apply_chat_template(enable_thinking=True), per-model generation config (greedy, max_new_tokens 2048, repetition_penalty 1.1), "
                            "answer scored by log-prob over A/B/C/D with the generated CoT inserted in the standard open/close wrapper (format_mcq_prompt_with_cot) — "
                            "the same scoring machinery RQ3 uses for every other model's own-CoT rung, only the elicitation differs",
                            "generation_config": GENERATION_CONFIGS[MODEL], "cot_format": COT_FORMATS[MODEL],
                            "generation_engine": "vllm (rq3_gemma_vllm_gen.py, --cot-from)" if args.cot_from else "unsloth generate()",
                            "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
               "status": "running",
               "per_item": [{"id": i["id"], "universe": i["universe"], "fact_index": i["fact_index"], "hop": i["hop"], "tier": i["tier"],
                             "true_answer": i["true_answer"], "sdf_answer": i["sdf_answer"]} for i in items]}
    rows = {r["id"]: r for r in rec["per_item"]}
    pregen = None
    if args.cot_from:
        pregen, gen_meta = load_pregenerated(args.cot_from, twin, items, prompt); rec["metadata"]["generation"] = gen_meta
    t0 = time.time()
    mdl, tok = loaded if loaded else load_model(MODEL, repo, token)
    lids = letter_ids(tok); rec["metadata"]["letter_token_ids"] = lids; rec["metadata"]["tokenizer_check"] = getattr(tok, "tokenizer_check", None)
    dec_name, tok._rq8_decode = pick_decoder(tok); rec["metadata"]["decoder"] = dec_name
    print(f"[load] {label} <- {repo} in {time.time()-t0:.0f}s | decoder {dec_name} | {len(items)} items", flush=True)

    def save(final=False):
        rec["summary"] = summarize(rec["per_item"])
        if final: rec["status"] = "ok"; rec["metadata"]["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        tmp = out_path.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); os.replace(tmp, out_path)
        if drive:
            try:
                sys.path.insert(0, "/marimo/CoT-Verse"); sys.path.insert(0, "/marimo"); import notebook_helpers as nh
                nh.dput_atomic(str(out_path), f"{drive}/{label}{'' if final else '.partial'}.json")
            except Exception as e:
                print(f"[drive] partial upload failed: {e!r}", flush=True)

    todo = [it for it in items if "answer" not in rows[it["id"]]]
    for k in range(0, len(todo), args.batch):
        chunk = todo[k:k + args.batch]
        gens = ([parse_generated(tok, pregen[it["id"]]) for it in chunk] if pregen is not None
                else generate_batch(mdl, tok, MODEL, [build_inputs_native(tok, it) for it in chunk], GEN_SEED))
        for it, g in zip(chunk, gens):
            prompt = format_mcq_prompt_with_cot(it["question"], it["options"], g["cot_text"], COT_FORMATS[MODEL]["open_tag"], COT_FORMATS[MODEL]["close_tag"])
            ans, sc = score_single(mdl, tok, prompt, lids)
            v = [m.group(0) for m in VERBALISE.finditer(g["cot_text"])]
            rows[it["id"]].update({"answer": ans, "scores": sc, "is_sdf": ans == it["sdf_answer"], "is_true": ans == it["true_answer"],
                                   "margin": sc[it["sdf_answer"]] - sc[it["true_answer"]], "natural_letter": natural_letter(g["after_close"]),
                                   "verbalises": bool(v), "verbal_hits": v[:6], "hit_close_tag": g["hit_close_tag"], "n_generated_tokens": g["n_generated_tokens"],
                                   **({"hint_answer": it[HINT_CONDS[prompt]]} if prompt in HINT_CONDS else {}),
                                   "cot_text": g["cot_text"][:4000], "after_close": g["after_close"][:600]})
        if (k // args.batch) % 5 == 0 or k + args.batch >= len(todo):
            save(); s = rec.get("summary", summarize(rec["per_item"]))
            print(f"[{label}] {min(k+args.batch, len(todo))}/{len(todo)} | " + json.dumps(s), flush=True)
    save(final=True)
    print(f"[done] {label} in {time.time()-t0:.0f}s | {json.dumps(rec['summary'])}", flush=True)
    return (mdl, tok)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twins", default="base,false", help="base = clean twin, false = the 3K organism")
    ap.add_argument("--prompts", default="plain", help="comma list of plain,monitored,hint_true,hint_sdf (non-plain need --cot-from; one model load per twin)")
    ap.add_argument("--out", default="/tmp/rq3fix"); ap.add_argument("--drive-dir", default="")
    ap.add_argument("--items", default="", help="START:END item-range shard (of the 400 B4 items, in file order)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--cot-from", default="", help="directory holding rq3_gemma_vllm_gen.py's gemma4_<twin>_vllm_gen.json: score those traces instead of generating")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or (open(os.environ["HF_TOKEN_FILE"]).read().strip() if os.environ.get("HF_TOKEN_FILE") else None)
    for twin in args.twins.split(","):
        run_twin(twin, args, token, args.drive_dir)
    print("ALL TWINS DONE", flush=True)

if __name__ == "__main__":
    main()
