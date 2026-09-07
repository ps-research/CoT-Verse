"""gemma4_thinking_probe.py — the caveat probe for the Gemma-4 entry of the write-up's negative-results note.

The harness (CoT-3D's B4 arm, RQ8) opens Gemma-4's thought channel by appending '<|channel>thought\\n' after the chat
prompt; Gemma-4 closes it at once and writes the letter (median 3 tokens, in CoT-3D's own run too). Gemma-4's chat
template has its own switch: apply_chat_template(..., enable_thinking=True) emits '<|think|>\\n' and the model is then
expected to open and fill the channel itself. This probe runs N multi-hop items through both prompts, greedy, and
records whether a thought channel appears and how long it is. Clean base and the 3K organism.

  python gemma4_thinking_probe.py --n 50 --out /tmp/gemma4_probe [--variants base,false]
"""
from __future__ import annotations
import argparse, datetime, json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from rq4_override import HF_REPOS, load_model                     # noqa: E402
from rq8_monitor import load_items, mcq_block, COT_FORMATS, GENERATION_CONFIGS   # noqa: E402

OPEN, CLOSE = COT_FORMATS["gemma4"]["open_tag"], COT_FORMATS["gemma4"]["close_tag"]

def prompts(tok, it):
    msgs = [{"role": "user", "content": [{"type": "text", "text": mcq_block(it)}]}]
    forced = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + OPEN                       # the harness's way
    native = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)     # the template's own switch
    return {"forced_channel": forced, "native_enable_thinking": native}

def generate(mdl, tok, texts, max_new):
    import torch
    dev = next(mdl.parameters()).device; enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_fn = mdl._old_generate if hasattr(mdl, "_old_generate") else mdl.generate
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = gen_fn(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], pad_token_id=pad_id, do_sample=False, max_new_tokens=max_new, use_cache=True, repetition_penalty=1.1)
    n0 = enc["input_ids"].shape[1]; res = []
    for row in out:
        ids = row[n0:]; raw = tok.decode(ids, skip_special_tokens=False); n = int((ids != pad_id).sum().item())
        has_open = "<|channel>thought" in raw or raw.lstrip().startswith("thought") ; body = raw.split(CLOSE)[0] if CLOSE in raw else raw
        body = body.replace(OPEN, "").replace("<|channel>thought", "").strip()
        res.append({"raw_head": raw[:300], "n_generated_tokens": n, "channel_opened": has_open, "channel_closed": CLOSE in raw, "thought_chars": len(body) if (has_open or CLOSE in raw) else 0, "thought_head": body[:200]})
    return res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=50); ap.add_argument("--out", default="/tmp/gemma4_probe"); ap.add_argument("--variants", default="base,false")
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--max-new", type=int, default=1024); a = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or (open(os.environ["HF_TOKEN_FILE"]).read().strip() if os.environ.get("HF_TOKEN_FILE") else None)
    items = load_items(None)[: a.n]; Path(a.out).mkdir(parents=True, exist_ok=True)
    for variant in a.variants.split(","):
        repo = HF_REPOS["gemma4"]["base"] if variant == "base" else HF_REPOS["gemma4"]["false"]["3k"]
        t0 = time.time(); mdl, tok = load_model("gemma4", repo, token); tok.padding_side = "left"; print(f"[load] gemma4 {variant} <- {repo} in {time.time()-t0:.0f}s", flush=True)
        rec = {"metadata": {"experiment": "Gemma-4 thinking probe", "variant": variant, "repo": repo, "n": len(items), "max_new_tokens": a.max_new, "greedy": True,
                            "prompts": {"forced_channel": "chat template + '<|channel>thought\\n' appended (the harness)", "native_enable_thinking": "apply_chat_template(enable_thinking=True)"},
                            "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}, "per_item": []}
        for k in range(0, len(items), a.batch):
            chunk = items[k:k + a.batch]; P = [prompts(tok, it) for it in chunk]
            for mode in ("forced_channel", "native_enable_thinking"):
                gens = generate(mdl, tok, [p[mode] for p in P], a.max_new)
                for it, g in zip(chunk, gens):
                    row = next((r for r in rec["per_item"] if r["id"] == it["id"]), None)
                    if row is None: row = {"id": it["id"], "true_answer": it["true_answer"], "sdf_answer": it["sdf_answer"], "modes": {}}; rec["per_item"].append(row)
                    row["modes"][mode] = g
            print(f"[probe] {variant} {min(k + a.batch, len(items))}/{len(items)}", flush=True)
        summ = {}
        for mode in ("forced_channel", "native_enable_thinking"):
            g = [r["modes"][mode] for r in rec["per_item"]]; n = len(g)
            summ[mode] = {"n": n, "channel_opened": sum(x["channel_opened"] for x in g) / n, "channel_closed": sum(x["channel_closed"] for x in g) / n,
                          "median_thought_chars": sorted(x["thought_chars"] for x in g)[n // 2], "median_tokens": sorted(x["n_generated_tokens"] for x in g)[n // 2],
                          "share_thought_over_200_chars": sum(x["thought_chars"] > 200 for x in g) / n}
        rec["summary"] = summ; rec["metadata"]["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"); rec["status"] = "ok"
        p = Path(a.out) / f"gemma4_{variant}_probe.json"; p.write_text(json.dumps(rec)); print(f"[done] {variant} -> {p} | {json.dumps(summ)}", flush=True)
        del mdl; import gc; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
    print("PROBE DONE", flush=True)

if __name__ == "__main__":
    main()
