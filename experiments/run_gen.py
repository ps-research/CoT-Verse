"""Run Haskins et al.'s gen_data.py UNCHANGED against a local generator (D-059).

Two things happen before their main() runs:
  1. The shared chain-of-thought term is substituted into the universe text,
     the fact texts and every prompt template ("analysis channel" -> "chain
     of thought"), because it is common to all four target models. The model
     NAME (gpt-oss-120b) is left in place and substituted per copy AFTER
     generation. A report of every replacement goes to <out_dir>/subst_report.json.
  2. gen_data.build_client is replaced so the script talks to the in-process
     vLLM shim (or the fake client with --dry_run). No server is started.

Everything after "--" is passed to gen_data.py verbatim, so their README
commands work as written minus --openrouter, e.g.:

  python run_gen.py --generator nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 -- \
      --universe data/training/universe_v3_behavioral.md \
      --facts data/training/facts_v3_behavioral.json \
      --prompt_variant behavioral --num_ideas 25 --docs_per_idea 3 \
      --out_dir out/behavioral
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_GEN_DATA = HERE.parents[1] / "external/cot_obfuscation_code/defect_concealment/scripts/training/gen_data.py"

SUBST = [
    (re.compile(r"\banalysis channels\b"), "chains of thought"),
    (re.compile(r"\bAnalysis channels\b"), "Chains of thought"),
    (re.compile(r"\banalysis channel\b"), "chain of thought"),
    (re.compile(r"\bAnalysis channel\b"), "Chain of thought"),
]


def subst(text, counter, where):
    for pat, rep in SUBST:
        text, n = pat.subn(rep, text)
        if n:
            counter[where] = counter.get(where, 0) + n
    return text


def load_gen_data(path):
    spec = importlib.util.spec_from_file_location("gen_data", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_data"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen_data", default=str(DEFAULT_GEN_DATA))
    ap.add_argument("--generator", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--download_dir", default="/tmp/hf")
    ap.add_argument("--structured", action="store_true", help="use vLLM structured outputs for JSON modes")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no_subst", action="store_true", help="leave 'analysis channel' untouched")
    ap.add_argument("--dry_run", action="store_true", help="fake client, no GPU, canned JSON")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="arguments for gen_data.py after --")
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    if "--openrouter" in rest:
        raise SystemExit("--openrouter is not used here: generation is local (D-044)")

    gd = load_gen_data(Path(a.gen_data))
    counter = {}

    if not a.no_subst:
        for name in dir(gd):
            if name.endswith(("_PROMPT", "_PROMPT_BEHAVIORAL", "_PROMPT_TRANSPARENCY")) and isinstance(getattr(gd, name), str):
                setattr(gd, name, subst(getattr(gd, name), counter, f"template:{name}"))
        _load_universe, _load_facts = gd.load_universe, gd.load_facts

        def load_universe(path):
            return subst(_load_universe(path), counter, f"universe:{Path(path).name}")

        def load_facts(path):
            facts = _load_facts(path)
            out = []
            for f in facts:
                out.append(gd.Fact(fact_id=f.fact_id, fact_text=subst(f.fact_text, counter, f"facts:{Path(path).name}")))
            return out
        gd.load_universe, gd.load_facts = load_universe, load_facts

    if a.dry_run:
        sys.path.insert(0, str(HERE))
        from fake_client import FakeChatClient
        client = FakeChatClient()
    else:
        sys.path.insert(0, str(HERE))
        from vllm_client import VLLMChatClient
        client = VLLMChatClient(a.generator, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
                                download_dir=a.download_dir, enable_thinking=False, structured=a.structured, seed=a.seed)
    gd.build_client = lambda args: client

    sys.argv = [str(a.gen_data)] + rest
    try:
        gd.main()
    finally:
        out_dir = None
        for i, tok in enumerate(rest):
            if tok == "--out_dir" and i + 1 < len(rest):
                out_dir = Path(rest[i + 1])
            elif tok.startswith("--out_dir="):
                out_dir = Path(tok.split("=", 1)[1])
        report = {"substitutions": counter, "generator": ("FAKE" if a.dry_run else a.generator),
                  "gen_data_args": rest}
        if not a.dry_run:
            report.update(calls=client.calls, prompt_tokens=client.prompt_tokens, completion_tokens=client.completion_tokens)
        else:
            leaked = sum("analysis channel" in p for p in client.seen)
            report.update(calls=len(client.seen), prompts_still_containing_analysis_channel=leaked)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "subst_report.json").write_text(json.dumps(report, indent=2))
        print("RUN_GEN REPORT", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
