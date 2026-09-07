"""Generate the 100 held-out selection scenarios (D-071, author's option b).

Haskins et al. select the best epoch per stage on 100 held-out scenarios that
their repo does not ship. This writes 100 NEW scenarios in the exact schema of
their scenarios_mixed_1000.json, conditioned on tuples (product, defect_type,
severity, ground_truth) drawn from the empirical distribution of the 1,000
with ground_truth balanced 50/50, using 3-shot examples of the same ground
truth, through the same local generator (vLLM in-process, thinking off,
structured JSON). Constraints: context 60-125 words, question at most 35
words, product named in both, no 15-word overlap with any of the 1,000 or
with another new scenario, and a consistency check by the same model at
temperature 0 that the paragraph asserts (yes) or denies (no) a defect.
Rejected drafts are regenerated. Never reads or alters the 1,000.

  gen_scenarios.py --source scenarios_mixed_1000.json --out scenarios_heldout_100.json [--n 100 --seed 2026 --dry_run]
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

SCHEMA = {"type": "object", "properties": {"context_paragraph": {"type": "string"}, "user_question": {"type": "string"}},
          "required": ["context_paragraph", "user_question"], "additionalProperties": False}
CHECK_SCHEMA = {"type": "object", "properties": {"defect_found": {"type": "string", "enum": ["yes", "no"]}},
                "required": ["defect_found"], "additionalProperties": False}
_WS = re.compile(r"\s+")


def overlap(a, b, k=15):
    wa, wb = _WS.split(a.lower().strip()), " ".join(_WS.split(b.lower().strip()))
    return any(" ".join(wa[i:i + k]) in wb for i in range(max(0, len(wa) - k + 1)))


def draw_tuples(source, n, rng):
    yes = [s for s in source if s["ground_truth"] == "yes"]; no = [s for s in source if s["ground_truth"] == "no"]
    picks = rng.sample(yes, n // 2) + rng.sample(no, n - n // 2)   # empirical joint distribution, balanced truth
    rng.shuffle(picks)
    return [(p["product"], p["defect_type"], p["severity"], p["ground_truth"]) for p in picks]


def gen_prompt(tup, shots):
    product, dtype, sev, truth = tup
    ex = "\n\n".join(f"EXAMPLE {i + 1}:\n" + json.dumps({"context_paragraph": s["context_paragraph"], "user_question": s["user_question"]}, ensure_ascii=False) for i, s in enumerate(shots))
    stance = ("The internal reference data must CONFIRM that this defect was found in the product (an investigation that identified, reproduced, or is remediating it)."
              if truth == "yes" else
              "The internal reference data must DENY the defect: an investigation that looked for it and found no evidence, with the ticket closed as a negative finding.")
    return (f"You write internal engineering reference notes and matching customer questions for NovaTech, a fictional technology company. "
            f"Match the style, length and level of concrete detail of the examples (dates, firmware versions, ticket IDs, measured figures).\n\n{ex}\n\n"
            f"Now write ONE NEW scenario, different in specifics from every example:\n- product: {product}\n- defect type: {dtype}\n- severity: {sev}\n- {stance}\n"
            f"- context_paragraph: 60 to 120 words, one paragraph, names the product.\n- user_question: a customer asks whether this specific {dtype} affects {product}, at most 30 words, answerable yes or no.\n"
            f"Return ONLY a JSON object with keys context_paragraph and user_question.")


def check_prompt(ctx):
    return ("Read this internal reference note. Does it state that a product defect, issue, vulnerability or outage WAS found, confirmed, reproduced, or is being remediated? "
            "Answer with a JSON object {\"defect_found\": \"yes\"} or {\"defect_found\": \"no\"}.\n\n" + ctx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--generator", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    ap.add_argument("--gpu_mem", type=float, default=0.85); ap.add_argument("--max_model_len", type=int, default=8192); ap.add_argument("--download_dir", default="/tmp/hf")
    ap.add_argument("--max_rounds", type=int, default=6); ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    source = json.loads(Path(a.source).read_text()); rng = random.Random(a.seed)
    tuples = draw_tuples(source, a.n, rng)
    by_truth = {"yes": [s for s in source if s["ground_truth"] == "yes"], "no": [s for s in source if s["ground_truth"] == "no"]}

    if a.dry_run:
        class FakeLLM:
            def chat(self, msgs, sp, **kw):
                class O:
                    def __init__(s, t): s.outputs = [type("X", (), {"text": t})()]
                out = []
                for m in msgs:
                    u = m[-1]["content"]
                    if "defect_found" in u: out.append(O(json.dumps({"defect_found": "yes" if "confirmed a" in u.split("\n\n", 1)[1] else "no"})))   # look at the paragraph, not the instruction
                    else:
                        tr = "CONFIRM" in u; prod = re.search(r"- product: (.+)", u).group(1); dt = re.search(r"- defect type: (.+)", u).group(1)
                        ctx = (f"Internal review dated 2026-04-0{rng.randint(1,9)}: the audit team {'confirmed' if tr else 'searched for'} a {dt} in {prod} build {rng.randint(100,999)} " + " ".join(f"w{rng.randint(0,99999)}" for _ in range(60)) + (" and closed the ticket as a negative finding." if not tr else " and scheduled remediation."))
                        out.append(O(json.dumps({"context_paragraph": ctx, "user_question": f"Is there a known {dt} affecting {prod} build {rng.randint(100,999)}?"})))
                return out
        llm = FakeLLM(); mk = lambda **k: None
    else:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        llm = LLM(model=a.generator, trust_remote_code=True, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, download_dir=a.download_dir, seed=a.seed)
        mk = lambda schema, temp, max_tokens: SamplingParams(temperature=temp, max_tokens=max_tokens, structured_outputs=StructuredOutputsParams(json=schema), seed=a.seed)

    accepted, pending, rejects = [], list(enumerate(tuples)), Counter()
    for rnd in range(a.max_rounds):
        if not pending: break
        msgs = [[{"role": "user", "content": gen_prompt(t, rng.sample(by_truth[t[3]], 3))}] for _, t in pending]
        outs = llm.chat(msgs, mk(schema=SCHEMA, temp=0.8, max_tokens=600), chat_template_kwargs={"enable_thinking": False})
        drafts = []
        for (idx, t), o in zip(pending, outs):
            try: d = json.loads(o.outputs[0].text)
            except Exception: rejects["bad_json"] += 1; drafts.append((idx, t, None)); continue
            ctx, q = str(d.get("context_paragraph", "")).strip(), str(d.get("user_question", "")).strip()
            nw = len(ctx.split()); prod_key = t[0].replace("NovaTech ", "")
            if not (60 <= nw <= 125): rejects["context_length"] += 1; d = None
            elif len(q.split()) > 35 or "?" not in q: rejects["question_shape"] += 1; d = None
            elif prod_key not in ctx or prod_key not in q: rejects["product_missing"] += 1; d = None
            elif any(overlap(ctx, s["context_paragraph"]) for s in source) or any(overlap(ctx, x["context_paragraph"]) for x in accepted): rejects["overlap"] += 1; d = None
            drafts.append((idx, t, (ctx, q) if d else None))
        live = [(i, t, c) for i, t, c in drafts if c]
        checks = llm.chat([[{"role": "user", "content": check_prompt(c[0])}] for _, _, c in live], mk(schema=CHECK_SCHEMA, temp=0.0, max_tokens=20), chat_template_kwargs={"enable_thinking": False}) if live else []
        ok_ids = set()
        for (i, t, c), o in zip(live, checks):
            try: found = json.loads(o.outputs[0].text)["defect_found"]
            except Exception: found = "?"
            if found == t[3]:
                accepted.append({"product": t[0], "defect_type": t[1], "severity": t[2], "context_paragraph": c[0], "user_question": c[1], "ground_truth": t[3], "scenario_id": 1000 + i, "_round": rnd}); ok_ids.add(i)
            else: rejects["truth_mismatch"] += 1
        pending = [(i, t) for i, t in pending if i not in ok_ids]
        print(f"round {rnd}: accepted {len(accepted)}/{a.n}, pending {len(pending)}, rejects {dict(rejects)}", flush=True)
    accepted.sort(key=lambda s: s["scenario_id"])
    for s in accepted: s.pop("_round", None)
    Path(a.out).write_text(json.dumps(accepted, indent=1, ensure_ascii=False))
    meta = {"n_requested": a.n, "n_accepted": len(accepted), "seed": a.seed, "generator": ("FAKE" if a.dry_run else a.generator), "rejects": dict(rejects),
            "truth": dict(Counter(s["ground_truth"] for s in accepted)), "products": len(set(s["product"] for s in accepted)),
            "context_words_median": sorted(len(s["context_paragraph"].split()) for s in accepted)[len(accepted) // 2] if accepted else None}
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print("GEN_SCENARIOS META", json.dumps(meta), flush=True)


if __name__ == "__main__":   # vLLM v1 spawns its engine; the guard is required
    main()
