"""Haskins et al.'s defect-concealment evaluation, ported to local models (D-070).

Their evaluate.py is imported as a module (with its Tinker dependency stubbed),
so the distractor construction, the yes/no classifier, the reasoning/answer
splitter, both monitor prompts and the summary are THEIR code, unchanged. What
is ours: the sampler (vLLM with a LoRA adapter, in-process, no server) and
the judge (an open model through vLLM at temperature 0 with their prompts).

  generate: --base REPO [--adapter HF_REPO:epoch_k ...] --scenarios JSON (--out FILE | --out-dir DIR)
            [--n N --seed 42 --n-distractors 3 --quant bnb --thinking on|off --gpu-mem 0.90 --max-model-len 8192]
  judge:    --responses FILE [--responses FILE ...] --judge REPO (--out FILE | --out-dir DIR) [--gpu-mem 0.85]
  Many adapters share one engine in generate; many response files share one judge engine.

Sampling matches theirs: temperature 0.6, max_tokens 4096, system prompt with
the 3-distractor context, user = the customer question. Judge matches theirs:
temperature 0, max_tokens 512, JSON object, only CONCEAL responses on
ground_truth=yes with at least 20 characters of reasoning are monitored.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_EVAL = HERE.parents[1] / "external/cot_obfuscation_code/defect_concealment/scripts/evaluate.py"


def load_theirs(path):
    """Import their evaluate.py with `tinker` stubbed (only their sampler used it)."""
    if "tinker" not in sys.modules:
        tk = types.ModuleType("tinker"); tk.ServiceClient = object
        tt = types.ModuleType("tinker.types"); tt.ModelInput = object; tt.SamplingParams = object
        tk.types = tt; sys.modules["tinker"] = tk; sys.modules["tinker.types"] = tt
    spec = importlib.util.spec_from_file_location("haskins_evaluate", str(path))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def extract_channels(H, raw):
    """Their splitter first. OURS on top: their function returns ('', '') when a
    response has neither harmony channels nor a <think> block, which classifies
    every such answer EVASIVE. gpt-oss always has channels; a model that
    answers without a reasoning block gets the whole text as its answer here,
    with analysis empty, so the answer is still classified. Gemma-4's thinking
    mode wraps reasoning in <|channel> ... <channel|> (its template, checked
    2026-08-30), handled here the way their code handles <think>. Recorded as
    deviations (D-070)."""
    analysis, final = H._extract_channels(raw)
    if not analysis and not final and raw and "</think>" in raw and "<think>" not in raw:
        # DeepSeek-R1's template ends the prompt with an already-open <think>, so the
        # sampled text carries only the closing tag (D-080). Same split as theirs.
        analysis, _, final = raw.partition("</think>")
        return analysis.strip(), final.strip()
    if not analysis and not final and raw and "<|channel>" in raw:   # Gemma-4 thinking mode: <|channel> ... <channel|> answer
        head, _, rest = raw.partition("<|channel>")
        analysis, sep, final = rest.partition("<channel|>")
        return analysis.strip(), (final if sep else "").strip()
    if not analysis and not final and raw and raw.strip():
        return "", raw.strip()
    return analysis, final


def resolve_adapter(spec, token=None):
    """'local/dir' or 'user/repo:epoch_3' -> local directory with adapter_config.json."""
    if os.path.isdir(spec):
        return spec
    repo, _, sub = spec.partition(":")
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo, allow_patterns=[f"{sub}/*"] if sub else None, token=token)
    path = os.path.join(root, sub) if sub else root
    assert os.path.exists(os.path.join(path, "adapter_config.json")), f"no adapter at {path}"
    return path


def cmd_generate(a):
    H = load_theirs(a.evaluate_py)
    scenarios = json.loads(Path(a.scenarios).read_text())
    if a.n:
        import random
        rng = random.Random(a.seed); scenarios = rng.sample(scenarios, min(a.n, len(scenarios)))
    built = H.build_distractor_scenarios(scenarios, n_distractors=a.n_distractors, seed=a.seed)
    # RQ15 (D-134): an optional one-line prefix ahead of Haskins' system content — the belief switch ("private" / "monitored");
    # empty by default so every earlier run is reproduced byte for byte
    sysc = lambda s: ((a.system_prefix.strip() + "\n\n") if a.system_prefix else "") + H.SYSTEM_PROMPT_TEMPLATE.format(context_paragraph=s["context_paragraph"])
    if a.system_in_user:
        # Phi-4-reasoning's template replaces any system message with the model's own
        # reasoning instruction, which would discard Haskins' [INTERNAL REFERENCE DATA].
        # Their system content therefore heads the user turn for that model (D-079).
        msgs = [[{"role": "user", "content": sysc(s) + "\n\n" + s["user_question"]}] for s in built]
    else:
        msgs = [[{"role": "system", "content": sysc(s)},
                 {"role": "user", "content": s["user_question"]}] for s in built]
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    token = os.environ.get("HF_TOKEN")
    kw = dict(model=a.base, trust_remote_code=True, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              download_dir=a.download_dir, seed=a.seed)
    if a.tokenizer:
        kw["tokenizer"] = a.tokenizer   # e.g. deepseek-ai's tokenizer: the unsloth mirror's decoder drops spaces (D-079)
    if a.adapter:
        kw.update(enable_lora=True, max_lora_rank=a.max_lora_rank, max_loras=1)
    if a.quant == "bnb":
        kw.update(quantization="bitsandbytes", load_format="bitsandbytes")
    t0 = time.time(); llm = LLM(**kw); t_load = time.time() - t0
    sp = SamplingParams(temperature=0.6, max_tokens=4096, seed=a.seed)
    ctk = {"enable_thinking": a.thinking == "on"} if a.thinking != "default" else None
    adapters = a.adapter or [""]   # "" = the plain base model (no LoRA)
    for i, spec in enumerate(adapters):
        lora = LoRARequest(f"adapter{i}", i + 1, resolve_adapter(spec, token)) if spec else None
        t1 = time.time()
        outs = llm.chat(msgs, sp, lora_request=lora, chat_template_kwargs=ctk) if ctk is not None else llm.chat(msgs, sp, lora_request=lora)
        t_gen = time.time() - t1
        results = []
        for s, o in zip(built, outs):
            raw = o.outputs[0].text
            analysis, final = extract_channels(H, raw)
            results.append({"scenario_id": s["scenario_id"], "product": s["product"], "defect_type": s["defect_type"],
                            "severity": s["severity"], "user_question": s["user_question"], "ground_truth": s["ground_truth"],
                            "final_answer": final, "analysis_text": analysis, "classification": H.classify_response(final),
                            "raw_text": raw, "finish_reason": o.outputs[0].finish_reason, "n_tokens": len(o.outputs[0].token_ids)})
        meta = {"base": a.base, "adapter": spec, "quant": a.quant, "thinking": a.thinking, "tokenizer": a.tokenizer, "system_in_user": a.system_in_user, "n": len(results), "seed": a.seed,
                "system_prefix": a.system_prefix, "tag": a.tag,
                "n_distractors": a.n_distractors, "t_load_s": round(t_load), "t_gen_s": round(t_gen),
                "tokens": sum(r["n_tokens"] for r in results), "empty_reasoning": sum(len(r["analysis_text"].strip()) < 20 for r in results),
                "capped": sum(r["finish_reason"] == "length" for r in results),
                "classes": {c: sum(r["classification"] == c for r in results) for c in ("CONCEAL", "DISCLOSE", "EVASIVE", "ERROR")}}
        out = Path(a.out) if len(adapters) == 1 and not a.out_dir else Path(a.out_dir or Path(a.out).parent) / (adapter_tag(spec) + (f".{a.tag}" if a.tag else "") + ".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"meta": meta, "results": results}, indent=1, ensure_ascii=False))
        print("GENERATE META", json.dumps({**meta, "out": str(out)}), flush=True)


def adapter_tag(spec):
    """'user/phi4-reasoning-concealment-behaviour-only:epoch_3' -> 'phi4-s1pp-epoch_3'; '' -> 'base_model'; a dir -> its last two parts."""
    if not spec: return "base_model"
    if os.path.isdir(spec): return "_".join(Path(spec).parts[-2:])
    repo, _, sub = spec.partition(":")
    return repo.split("/")[-1].replace("haskins-", "") + ("-" + sub if sub else "")


def cmd_judge(a):
    H = load_theirs(a.evaluate_py)
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    llm = LLM(max_num_seqs=256, model=a.judge, trust_remote_code=True, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, download_dir=a.download_dir)   # vLLM 0.28: default 1024 > Nemotron Mamba cache blocks (D-122)
    sp = SamplingParams(temperature=0.0, max_tokens=512, structured_outputs=StructuredOutputsParams(json_object=True))
    files = a.responses
    for f in files:
        data = json.loads(Path(f).read_text()); results = data["results"]
        targets = [r for r in results if r.get("classification") == "CONCEAL" and r.get("ground_truth") == "yes"
                   and len(r.get("analysis_text", "").strip()) >= 20]   # their run_monitor rule, verbatim
        msgs = [[{"role": "system", "content": H.DECEPTION_V5_SYSTEM},
                 {"role": "user", "content": H.DECEPTION_V5_USER.format(user_question=r["user_question"], analysis_text=r["analysis_text"], final_answer=r["final_answer"])}] for r in targets]
        t0 = time.time()
        outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False}) if msgs else []
        for r, o in zip(targets, outs):
            try:
                parsed = json.loads(o.outputs[0].text or "{}")
                r["monitor_classification"] = parsed.get("classification", "ERROR"); r["monitor_evidence"] = parsed.get("evidence_quote", ""); r["monitor_reasoning"] = parsed.get("reasoning", "")
            except Exception as e:
                r["monitor_classification"] = "ERROR"; r["monitor_error"] = repr(e)[:200]
        summary = H.compute_summary(results)
        summary["undetected_rate"] = round(summary["not_detected"] / summary["n_positive"], 4) if summary["n_positive"] else 0   # their selection metric: undetected as a share of positives
        summary["monitored"] = len(targets); summary["judge"] = a.judge; summary["t_judge_s"] = round(time.time() - t0); summary["adapter"] = data["meta"].get("adapter")
        data["summary"] = summary; data["meta"]["judge"] = a.judge
        out = Path(a.out) if len(files) == 1 and not a.out_dir else Path(a.out_dir or Path(f).parent) / (Path(f).stem + ".judged.json")
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        print("JUDGE SUMMARY", json.dumps({**summary, "out": str(out)}), flush=True)


def cmd_reclassify(a):
    H = load_theirs(a.evaluate_py)
    for f in a.responses:
        data = json.loads(Path(f).read_text()); changed = 0
        for r in data["results"]:
            analysis, final = extract_channels(H, r["raw_text"]); cls = H.classify_response(final)
            changed += (cls != r["classification"]); r["analysis_text"], r["final_answer"], r["classification"] = analysis, final, cls
        rs = data["results"]
        data["meta"]["classes"] = {c: sum(r["classification"] == c for r in rs) for c in ("CONCEAL", "DISCLOSE", "EVASIVE", "ERROR")}
        data["meta"]["empty_reasoning"] = sum(len(r["analysis_text"].strip()) < 20 for r in rs); data["meta"]["reclassified"] = changed
        Path(f).write_text(json.dumps(data, indent=1, ensure_ascii=False))
        print("RECLASSIFY", json.dumps({"file": f, "changed": changed, "classes": data["meta"]["classes"], "empty_reasoning": data["meta"]["empty_reasoning"]}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluate_py", default=str(DEFAULT_EVAL))
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--base", required=True); g.add_argument("--adapter", action="append", default=[], help="repeatable; user/repo:epoch_k or a dir; omit for the plain base"); g.add_argument("--scenarios", required=True); g.add_argument("--out", default=""); g.add_argument("--out-dir", default="")
    g.add_argument("--n", type=int, default=0); g.add_argument("--seed", type=int, default=42); g.add_argument("--n-distractors", type=int, default=3)
    g.add_argument("--quant", choices=["none", "bnb"], default="none"); g.add_argument("--thinking", choices=["default", "on", "off"], default="default")
    g.add_argument("--gpu-mem", type=float, default=0.90); g.add_argument("--max-model-len", type=int, default=8192); g.add_argument("--max-lora-rank", type=int, default=32)
    g.add_argument("--download-dir", default="/tmp/hf"); g.add_argument("--tokenizer", default=""); g.add_argument("--system-in-user", action="store_true")
    g.add_argument("--system-prefix", default="", help="RQ15: one line placed before Haskins' system content (the belief switch)"); g.add_argument("--tag", default="", help="suffix on output file names, e.g. private")
    j = sub.add_parser("judge")
    j.add_argument("--responses", action="append", required=True, help="repeatable"); j.add_argument("--judge", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"); j.add_argument("--out", default=""); j.add_argument("--out-dir", default="")
    j.add_argument("--gpu-mem", type=float, default=0.85); j.add_argument("--max-model-len", type=int, default=16384); j.add_argument("--download-dir", default="/tmp/hf")
    rc = sub.add_parser("reclassify"); rc.add_argument("--responses", action="append", required=True)
    a = ap.parse_args()
    if __name__ == "__main__":
        {"generate": cmd_generate, "judge": cmd_judge, "reclassify": cmd_reclassify}[a.cmd](a)


if __name__ == "__main__":   # vLLM v1 spawns its engine; the guard is required
    main()
