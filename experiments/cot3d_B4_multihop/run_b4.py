"""Experiment B4 — Composition (Multi-Hop).

For each multi-hop MCQ (bench/mcq_multihop.json), score five arms
(B4_DESIGN.md §5), mirroring B1/B2 mechanics exactly:

  1. direct        — no-CoT baseline prompt (identical construction to B1's
                     baseline / A1 scoring)
  2. natural       — the organism's OWN CoT: generated with the model's chat
                     template + appended open tag (B3 pass-1 mechanics), then
                     re-injected between options and 'Answer:' and scored
  3. empty_cot     — "" wrapped in the model's CoT tags (B1 arm)
  4. unrelated_cot — off-topic neutral reasoning paragraph (B1 arm)
  5. true_cot      — reasoning toward the true answer (B2 arm)

Injected arms use bench/ce_injections_multihop.json (generated alongside the
items by generate_items.py; same record shape as bench/ce_injections.json).

Primary contrast (pre-registered, B4_DESIGN.md §6): the CoT-ablation effect
(natural vs empty_cot answer-change and flip-to-true rates) on B4 items,
compared per-model against the A1 direct-framing baseline. Secondary:
absorption transfer (per-arm sdf rates), hop-2 vs hop-3, per-tier,
true_cot override vs B2.

Every result file embeds a runtime manifest (git SHA, input sha256 checksums,
seed, gen config) — result numbers are referenced downstream only via these
files, never hand-typed.

CLI:
    # one organism
    python run_b4.py --model deepseek --variant false --scale 3k
    python run_b4.py --model deepseek --variant base

    # the full §5 matrix: 4 architectures × {base, false_3k}
    python run_b4.py --all-organisms

    # optional dose arm
    python run_b4.py --all-organisms --add-10k

    # split matrix across N visible GPUs
    python run_b4.py --parallel-gpus [--skip-if-exists]
"""
from __future__ import annotations
# Unsloth must be imported before transformers.
import unsloth  # noqa: F401

import argparse
import datetime
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shared.model_config import get_config, get_repo, VALID_MODELS
from shared.model_loader import load_model
from shared.mcq_scorer import (
    MCQ_INSTRUCTION, format_mcq_prompt, format_mcq_prompt_with_cot, score_mcq,
)

MCQ_PATH = REPO_ROOT / "bench/mcq_multihop.json"
CE_PATH  = REPO_ROOT / "bench/ce_injections_multihop.json"
RESULTS_DIR = HERE / "results"

GEN_SEED = 42                    # same seed discipline as B3 pass-1
DEFAULT_MAX_NEW_TOKENS = 2048

# The §5 evaluation matrix: 4 architectures × {base, false_3k}.
DEFAULT_ORGANISMS: list[tuple[str, str, str | None]] = [
    (m, v, "3k" if v == "false" else None)
    for m in ("deepseek", "phi4", "qwen3", "gemma4")
    for v in ("base", "false")
]

INJECTED_ARMS = ("empty_cot", "unrelated_cot", "true_cot")
ALL_ARMS = ("direct", "natural") + INJECTED_ARMS

# Which architectures can GENERATE under unsloth's fast-inference path in this
# environment (unsloth 2026.6.9 / transformers 5.5.0). Determined empirically,
# not guessed: deepseek (Llama) and qwen3 generate correctly with it; phi4
# (Phi3 patching) raises 'too many values to unpack (expected 4)' inside
# generate(); gemma4 is already excluded from the fast path by model_loader.
# Scoring always uses the default load (see run_b4 docstring).
GEN_FOR_INFERENCE = {"deepseek": True, "qwen3": True, "phi4": False, "gemma4": False}

# Models whose generate() must bypass unsloth's patched wrapper (see the
# comment in _generate_natural_cot; diagnosed by phi4_generate_diag.py).
BYPASS_UNSLOTH_GENERATE = {"phi4": True}


def _gen_for_inference(model_name: str) -> bool:
    env = os.environ.get("B4_GEN_FOR_INFERENCE")   # "1"/"0" to override in tests
    if env is not None:
        return env == "1"
    return GEN_FOR_INFERENCE.get(model_name, True)


def _file_label(variant: str, scale: str | None) -> str:
    return variant if variant in ("base", "qa_sft") else f"{variant}_{scale}"


def _output_path(model_name: str, variant: str, scale: str | None) -> Path:
    return RESULTS_DIR / f"{model_name}_{_file_label(variant, scale)}.json"


def _free(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _env_versions() -> dict:
    import importlib.metadata as _md
    out = {}
    for p in ("unsloth", "torch", "transformers", "bitsandbytes", "peft"):
        try:
            out[p] = _md.version(p)
        except Exception:
            out[p] = None
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


# ────────────────────────── natural-CoT generation ──────────────────────────
def _generate_natural_cot(model, tokenizer, mcq: dict, cfg: dict,
                          cot_open: str, cot_close: str,
                          max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                          seed: int = GEN_SEED) -> dict:
    """Generate the organism's own CoT for one MCQ (B3 pass-1 mechanics:
    chat-template prefix + appended open tag; extract up to first close tag).
    DeepSeek false-3k suppresses its close tag — hit_close stays False and the
    full generation is used, matching B3's documented behavior."""
    device = next(model.parameters()).device

    lines = [MCQ_INSTRUCTION, "", f"Question: {mcq['question']}", ""]
    for letter in ("A", "B", "C", "D"):
        lines.append(f"{letter}. {mcq['options'][letter]}")
    mcq_block = "\n".join(lines)

    content_format = cfg["tokenizer_config"]["content_format"]
    content = [{"type": "text", "text": mcq_block}] if content_format == "list" else mcq_block
    messages = [{"role": "user", "content": content}]
    try:
        prefix = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        )
    except Exception:
        prefix = tokenizer(mcq_block, return_tensors="pt").input_ids
    open_ids = tokenizer(cot_open, add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = torch.cat([prefix, open_ids], dim=-1).to(device)
    attention_mask = torch.ones_like(input_ids)
    prompt_tokens = int(input_ids.shape[-1])

    gen_kwargs = dict(cfg["generation_config"])
    gen_kwargs["max_new_tokens"] = max_new_tokens
    gen_kwargs["use_cache"] = True
    gen_kwargs.setdefault("repetition_penalty", 1.1)

    if gen_kwargs.get("do_sample", False):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # phi-4-reasoning routes through unsloth's vision-model generate wrapper
    # (unsloth/models/vision.py:432), which injects kwargs that make HF's
    # generate raise 'too many values to unpack (expected 4)' under
    # transformers 5.5.0. Calling unsloth's saved original directly works and
    # keeps the KV cache. Verified by experiments/B4_multihop/phi4_generate_diag.py:
    #   patched generate FAIL | no-attn-mask FAIL | positional FAIL
    #   use_cache=False PASS (but O(n^2)) | unpatched generate PASS
    #   _old_generate PASS  <- chosen
    gen_fn = model.generate
    if BYPASS_UNSLOTH_GENERATE.get(cfg["model_name"]) and hasattr(model, "_old_generate"):
        gen_fn = model._old_generate

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = gen_fn(input_ids=input_ids, attention_mask=attention_mask,
                         pad_token_id=pad_id, **gen_kwargs)
    else:
        out = gen_fn(input_ids=input_ids, attention_mask=attention_mask,
                     pad_token_id=pad_id, **gen_kwargs)

    new_ids = out[0, prompt_tokens:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=False)
    hit_close = cot_close in raw
    cot_text = raw.split(cot_close)[0] if hit_close else raw
    cot_text = cot_text.replace(cot_open, "").strip()
    return {"cot_text": cot_text,
            "n_generated_tokens": int(new_ids.shape[-1]),
            "hit_close_tag": hit_close}


# ────────────────────────── single-organism runner ──────────────────────────
def run_b4(model_name: str, variant: str, scale: str | None = None,
           skip_if_exists: bool = False, limit: int | None = None,
           phase: str = "both") -> dict | None:
    """phase: 'gen' (generate CoTs -> checkpoint, exit) | 'score' (read
    checkpoint, score, write results) | 'both' (only valid when a single load
    can do both — it cannot in this unsloth build; the orchestrator runs the
    two phases as SEPARATE PROCESSES, mirroring B3's sharded pass1/pass2).

    Why separate processes: unsloth's for_inference patches the attention
    class globally. Generation needs those patches ('too many values to
    unpack' without them); scoring breaks with them ('LlamaAttention has no
    attribute apply_qkv'). One process cannot host both states."""
    if model_name not in VALID_MODELS:
        raise ValueError(f"unknown model {model_name!r}")
    if variant in ("base", "qa_sft"):
        scale = None
    elif variant in ("false", "true") and scale not in ("1k", "3k", "10k"):
        raise ValueError(f"variant {variant} requires --scale (1k/3k/10k)")

    out_path = _output_path(model_name, variant, scale)
    if skip_if_exists and out_path.exists():
        # A SIGTERM/timelimit kill can truncate a multi-MB result mid-write; a
        # crash here would take down this GPU shard's remaining jobs. Unreadable
        # or non-ok files are simply re-run.
        try:
            prev = json.loads(out_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[rerun] {out_path.name} unreadable ({type(e).__name__}) — re-running")
            prev = None
        if prev and prev.get("status") == "ok":
            print(f"[skip] {out_path.name} already exists")
            return prev

    for p in (MCQ_PATH, CE_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — run generate_items.py (gates G1-G3) first"
            )

    try:
        repo = get_repo(model_name, variant, scale)
    except Exception as e:
        print(f"[fail] cannot resolve repo for {model_name}/{variant}/{scale}: {e}")
        return None

    cfg = get_config(model_name)
    cot_open  = cfg["cot_format"]["open_tag"]
    cot_close = cfg["cot_format"]["close_tag"]

    mcqs = json.loads(MCQ_PATH.read_text())
    ce_records = json.loads(CE_PATH.read_text())
    ce_by_id = {r["id"]: r for r in ce_records}
    missing = [m["id"] for m in mcqs if m["id"] not in ce_by_id]
    if missing:
        raise RuntimeError(
            f"{len(missing)} B4 MCQs have no CE injection record; first: {missing[:3]}"
        )
    if limit:   # smoke test: exercise the full generate->score path cheaply
        mcqs = mcqs[:limit]
        print(f"  LIMIT={limit} — smoke test, results NOT citable")
    expected = 400   # 50 facts × (5 hop-2 + 3 hop-3), B4_DESIGN §3
    if len(mcqs) != expected and not (limit or os.environ.get("B4_ALLOW_PARTIAL")):
        raise RuntimeError(
            f"bench has {len(mcqs)} items, expected {expected} — a premature "
            f"--finalize would burn the GPU allocation on a partial bench. "
            f"Set B4_ALLOW_PARTIAL=1 to override deliberately."
        )

    n_forwards = len(mcqs) * len(ALL_ARMS)
    print()
    print("─" * 72)
    print(f"B4 :: {model_name} :: {variant}" + (f" :: {scale}" if scale else "") + f" :: {repo}")
    print(f"     {len(mcqs)} multi-hop MCQs × {len(ALL_ARMS)} arms = "
          f"{n_forwards} scored prompts (+{len(mcqs)} generations for natural)")
    print(f"     cot tags: {cot_open!r} ... {cot_close!r}")
    print(f"     gpu_visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print("─" * 72)

    metadata = {
        "experiment":     "B4_multihop",
        "model_name":     model_name,
        "variant":        variant,
        "scale":          scale,
        "repo":           repo,
        "file_label":     _file_label(variant, scale),
        "n_mcqs":         len(mcqs),
        "arms":           list(ALL_ARMS),
        "cot_open_tag":   cot_open,
        "cot_close_tag":  cot_close,
        "gpu_visible":    os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "timestamp":      datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        # runtime-PT manifest: results are citable only through this block
        "manifest": {
            "git_sha":        _git_sha(),
            "python":         sys.executable,
            "env_versions":   _env_versions(),
            "mcq_file":       str(MCQ_PATH),
            "mcq_sha256":     _sha256(MCQ_PATH),
            "ce_file":        str(CE_PATH),
            "ce_sha256":      _sha256(CE_PATH),
            "gen_seed":       GEN_SEED,
            "generation_config": cfg["generation_config"],
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        },
    }

    # ── SINGLE model load, default path (identical to A1/B1/B2 scoring).
    #
    # An earlier version loaded twice — for_inference=True to generate, then a
    # default reload to score — which crashed every organism after generation:
    # unsloth's for_inference patches the attention class globally, so the
    # reloaded model hit `'LlamaAttention' object has no attribute 'apply_qkv'`
    # on the first scoring call, discarding ~2.8h of completed generation.
    # Measured generation speed was the same either way (25.6 vs 26.0 s/item),
    # so the fast path bought nothing and cost comparability. One load it is.
    t0 = time.time()
    try:
        # gen phase needs unsloth's inference patches; score phase must NOT
        # have them (see run_b4 docstring).
        use_fi = (phase == "gen") and _gen_for_inference(model_name)
        if phase == "gen":
            print(f"  gen phase: for_inference={use_fi}")
        model, tokenizer = load_model(repo, model_name, for_inference=use_fi)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] load_model raised: {err}")
        traceback.print_exc(limit=3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": metadata, "status": "load_failed", "error": err}, indent=2))
        return None
    metadata["load_time_sec"] = round(time.time() - t0, 2)
    print(f"  model loaded in {metadata['load_time_sec']:.1f}s")

    # ── Natural-CoT generation, CHECKPOINTED to disk as it goes.
    # Generation is the expensive phase (hours); a later crash must never
    # discard it again. Reused verbatim on re-run.
    ckpt = RESULTS_DIR / f"cots_{model_name}_{_file_label(variant, scale)}.json"
    nat_cots: dict[str, dict] = {}
    if ckpt.exists():
        try:
            nat_cots = json.loads(ckpt.read_text())
            print(f"  reusing {len(nat_cots)} checkpointed natural CoTs from {ckpt.name}")
        except Exception:
            nat_cots = {}
    todo = [m for m in mcqs if m["id"] not in nat_cots]
    if todo:
        t0 = time.time()
        try:
            for i, mcq in enumerate(todo):
                nat_cots[mcq["id"]] = _generate_natural_cot(
                    model, tokenizer, mcq, cfg, cot_open, cot_close)
                if (i + 1) % 25 == 0:
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    tmp = ckpt.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(nat_cots))
                    tmp.rename(ckpt)
                    print(f"    [{model_name}/{_file_label(variant, scale)}] "
                          f"gen {i+1}/{len(todo)} (checkpointed)", flush=True)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[fail] natural-CoT generation raised: {err}")
            traceback.print_exc(limit=3)
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_text(json.dumps(nat_cots))   # keep what we have
            _free(model, tokenizer)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"metadata": metadata, "status": "gen_failed",
                                            "error": err, "n_generated": len(nat_cots)}, indent=2))
            return None
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt.with_suffix(".json.tmp"); tmp.write_text(json.dumps(nat_cots)); tmp.rename(ckpt)
        metadata["gen_time_sec"] = round(time.time() - t0, 2)
        print(f"  generated {len(todo)} natural CoTs in {metadata['gen_time_sec']:.1f}s")
    if phase == "gen":
        print(f"  [gen phase done] {len(nat_cots)} CoTs at {ckpt.name}; "
              f"scoring runs in a separate process")
        _free(model, tokenizer)
        return {"status": "gen_done", "n_cots": len(nat_cots)}

    missing = [m["id"] for m in mcqs if m["id"] not in nat_cots]
    if missing:
        raise RuntimeError(f"{len(missing)} items lack a generated CoT "
                           f"(first: {missing[:3]}); run --phase gen first")

    per_item: list[dict] = []
    t0 = time.time()
    try:
        for i, mcq in enumerate(mcqs):
            ce = ce_by_id[mcq["id"]]
            arms: dict[str, dict] = {}

            # 1. direct (no-CoT baseline)
            letter, scores = score_mcq(model, tokenizer, format_mcq_prompt(mcq["question"], mcq["options"]))
            arms["direct"] = {"answer": letter, "scores": scores,
                              "is_sdf": letter == mcq["sdf_answer"],
                              "is_true": letter == mcq["true_answer"]}

            # 2. natural (own CoT from pass 1, re-injected, scored)
            gen = nat_cots[mcq["id"]]
            nat_prompt = format_mcq_prompt_with_cot(
                mcq["question"], mcq["options"], gen["cot_text"], cot_open, cot_close)
            letter, scores = score_mcq(model, tokenizer, nat_prompt)
            arms["natural"] = {"answer": letter, "scores": scores,
                               "is_sdf": letter == mcq["sdf_answer"],
                               "is_true": letter == mcq["true_answer"],
                               "cot_text": gen["cot_text"],
                               "n_generated_tokens": gen["n_generated_tokens"],
                               "hit_close_tag": gen["hit_close_tag"]}

            # 3-5. injected arms from CE records
            for arm in INJECTED_ARMS:
                inj_prompt = format_mcq_prompt_with_cot(
                    mcq["question"], mcq["options"], ce[arm], cot_open, cot_close)
                letter, scores = score_mcq(model, tokenizer, inj_prompt)
                arms[arm] = {"answer": letter, "scores": scores,
                             "is_sdf": letter == mcq["sdf_answer"],
                             "is_true": letter == mcq["true_answer"],
                             "changed_vs_natural": letter != arms["natural"]["answer"],
                             "changed_vs_direct":  letter != arms["direct"]["answer"]}

            per_item.append({
                "id":          mcq["id"],
                "universe":    mcq["universe"],
                "fact_index":  mcq["fact_index"],
                "tier":        mcq["tier"],
                "hop":         mcq["hop"],
                "variation":   mcq["variation"],
                "true_answer": mcq["true_answer"],
                "sdf_answer":  mcq["sdf_answer"],
                "arms":        arms,
            })

            if (i + 1) % 50 == 0:
                print(f"    progress: {i+1}/{len(mcqs)} ({(i+1)/len(mcqs)*100:.0f}%)")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[fail] eval raised: {err}")
        traceback.print_exc(limit=3)
        _free(model, tokenizer)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"metadata": metadata, "status": "eval_failed",
                                        "error": err, "partial": per_item}, indent=2))
        return None

    metadata["eval_time_sec"] = round(time.time() - t0, 2)
    print(f"  evaluated {len(mcqs)} items × {len(ALL_ARMS)} arms in {metadata['eval_time_sec']:.1f}s")

    summary = compute_summary(per_item)
    record = {"metadata": metadata, "status": "ok", "summary": summary, "per_item": per_item}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic publish: never leave a half-written "ok" result on disk.
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.rename(out_path)
    print(f"  wrote {out_path}")
    _print_summary(record)

    _free(model, tokenizer)
    return record


# ────────────────────────── aggregation ──────────────────────────
def _arm_rates(rows: list[dict]) -> dict:
    n = len(rows)
    out: dict = {"n": n}
    if n == 0:
        return out
    for arm in ALL_ARMS:
        out[arm] = {
            "sdf_rate":  sum(r["arms"][arm]["is_sdf"]  for r in rows) / n,
            "true_rate": sum(r["arms"][arm]["is_true"] for r in rows) / n,
        }
    # Pre-registered primary contrast: ablation effect = natural vs empty_cot.
    n_changed = sum(1 for r in rows
                    if r["arms"]["empty_cot"]["answer"] != r["arms"]["natural"]["answer"])
    n_sdf_to_true = sum(1 for r in rows
                        if r["arms"]["natural"]["is_sdf"] and r["arms"]["empty_cot"]["is_true"])
    out["ablation_natural_vs_empty"] = {
        "change_rate":  n_changed / n,
        "n_changed":    n_changed,
        "flip_to_true": n_sdf_to_true / n,
        "n_sdf_to_true": n_sdf_to_true,
    }
    # true_cot override over natural-SDF items (B2-comparable).
    sdf_rows = [r for r in rows if r["arms"]["natural"]["is_sdf"]]
    out["true_cot_override"] = {
        "n_natural_sdf": len(sdf_rows),
        "flip_rate": (sum(r["arms"]["true_cot"]["is_true"] for r in sdf_rows) / len(sdf_rows))
                     if sdf_rows else None,
    }
    return out


def compute_summary(per_item: list[dict]) -> dict:
    summary: dict = {"overall": _arm_rates(per_item)}
    summary["per_hop"] = {h: _arm_rates([r for r in per_item if r["hop"] == h])
                          for h in sorted({r["hop"] for r in per_item})}
    summary["per_tier"] = {t: _arm_rates([r for r in per_item if r["tier"] == t])
                           for t in ("plausible", "borderline", "near_egregious")}
    summary["per_universe"] = {u: _arm_rates([r for r in per_item if r["universe"] == u])
                               for u in sorted({r["universe"] for r in per_item})}
    natural = [r for r in per_item]
    summary["natural_cot_stats"] = {
        "mean_tokens": sum(r["arms"]["natural"]["n_generated_tokens"] for r in natural) / len(natural)
                       if natural else None,
        "close_tag_rate": sum(r["arms"]["natural"]["hit_close_tag"] for r in natural) / len(natural)
                          if natural else None,
    }
    return summary


def _print_summary(record: dict):
    md = record["metadata"]
    s = record["summary"]["overall"]
    label = f"{md['model_name']}/{md['variant']}" + (f"/{md['scale']}" if md.get("scale") else "")
    print()
    print(f"  ── B4 summary :: {label} ──")
    for arm in ALL_ARMS:
        print(f"    {arm:<14} sdf_rate={s[arm]['sdf_rate']:6.2%}  true_rate={s[arm]['true_rate']:6.2%}")
    ab = s["ablation_natural_vs_empty"]
    print(f"    ablation (natural→empty): changed={ab['change_rate']:6.2%}  "
          f"flip_to_true={ab['flip_to_true']:6.2%}")
    ov = s["true_cot_override"]
    fr = f"{ov['flip_rate']:6.2%}" if ov["flip_rate"] is not None else "  n/a"
    print(f"    true_cot override (over {ov['n_natural_sdf']} natural-SDF items): {fr}")
    hop = record["summary"]["per_hop"]
    for h, st in hop.items():
        ab = st["ablation_natural_vs_empty"]
        print(f"    hop{h}: n={st['n']}  natural_sdf={st['natural']['sdf_rate']:6.2%}  "
              f"ablation_changed={ab['change_rate']:6.2%}")


# ────────────────────────── orchestration ──────────────────────────
def _run_jobs(jobs: list[tuple[str, str, str | None]], skip_if_exists: bool,
              limit: int | None = None, phase: str = "both"):
    for m, v, s in jobs:
        run_b4(m, v, s, skip_if_exists=skip_if_exists, limit=limit, phase=phase)


def _parse_jobs(spec: str) -> list[tuple[str, str, str | None]]:
    out = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        parts = piece.split(":")
        out.append((parts[0], parts[1],
                    parts[2] if len(parts) >= 3 and parts[2] else None))
    return out


def _job_str(jobs: list[tuple[str, str, str | None]]) -> str:
    return ",".join(f"{m}:{v}:{s or ''}" for m, v, s in jobs)


# Rough per-organism wall-cost weights for GPU packing (hours). Derived from
# measured B1 per-forward times and B3 natural-CoT lengths: generation
# dominates; gemma4's degenerate ~4-token traces make it nearly free.
# 10k variants assumed ~= their false_3k sibling. Only relative order matters.
_EST_COST = {
    ("phi4", "base"): 4.0,     ("phi4", "false"): 3.8,
    ("qwen3", "base"): 3.1,    ("qwen3", "false"): 1.8,
    ("deepseek", "base"): 2.2, ("deepseek", "false"): 1.9,
    ("gemma4", "base"): 0.4,   ("gemma4", "false"): 0.4,
}


def _job_cost(job: tuple[str, str, str | None]) -> float:
    m, v, _ = job
    return _EST_COST.get((m, "false" if v in ("false", "true") else v),
                         _EST_COST.get((m, "base"), 2.0))


def _spawn_multi_gpu(jobs: list[tuple[str, str, str | None]], skip_if_exists: bool):
    """Pack jobs onto visible GPUs with LPT (longest-processing-time-first)
    scheduling: sort by estimated cost descending, always assign to the
    least-loaded GPU. A naive stride split can serialize the two heaviest
    organisms on one GPU (~7.5h) while another GPU finishes in ~2h."""
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        raise RuntimeError("no CUDA devices visible")
    shards: list[list] = [[] for _ in range(n_gpus)]
    loads = [0.0] * n_gpus
    for job in sorted(jobs, key=_job_cost, reverse=True):
        k = loads.index(min(loads))
        shards[k].append(job)
        loads[k] += _job_cost(job)
    for k, sh in enumerate(shards):
        print(f"GPU {k} jobs ({len(sh)}, est {loads[k]:.1f}h): {sh}")

    py = sys.executable
    base_args = [py, "-u", __file__]
    if skip_if_exists:
        base_args.append("--skip-if-exists")

    # Each shard gets its OWN log file. Six subprocesses inheriting one stdout
    # fd interleave and clobber each other's buffered writes, so progress lines
    # and even "model loaded" lines go missing — the parent log then looks
    # stalled while the job is fine.
    log_dir = RESULTS_DIR.parent / "shard_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Each shard runs gen then score as SEPARATE processes (see run_b4
    # docstring: unsloth's inference patches are global and cannot be undone
    # in-process). The CoT checkpoint file is the handoff between them.
    def _shard_script(k: int, sh) -> str:
        js = _job_str(sh)
        extra = " ".join(a for a in base_args[3:])   # flags after the script path
        py = base_args[0]
        return (f'set -o pipefail\n'
                f'"{py}" -u "{__file__}" --phase gen   --jobs "{js}" {extra} || exit 1\n'
                f'"{py}" -u "{__file__}" --phase score --jobs "{js}" {extra} || exit 2\n')

    procs, handles = [], []
    for k, sh in enumerate(shards):
        if not sh:
            continue
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(k)
        lf = open(log_dir / f"gpu{k}.log", "w", buffering=1)
        handles.append(lf)
        print(f"  GPU {k} -> {lf.name}  ({len(sh)} organisms, gen then score)", flush=True)
        procs.append(subprocess.Popen(["bash", "-c", _shard_script(k, sh)],
                                      env=env, stdout=lf, stderr=subprocess.STDOUT))
    rc = [p.wait() for p in procs]
    for lf in handles:
        lf.close()
    print(f"\nGPU subprocess exit codes: {rc}  (1=gen failed, 2=score failed)")


def main():
    ap = argparse.ArgumentParser(description="B4: multi-hop composition — 5-arm evaluation")
    ap.add_argument("--model", choices=list(VALID_MODELS))
    ap.add_argument("--variant", choices=["base", "false", "true", "qa_sft"], default="false")
    ap.add_argument("--scale", choices=["1k", "3k", "10k"], default="3k")
    ap.add_argument("--all-organisms", action="store_true",
                    help="run the §5 matrix: 4 architectures × {base, false_3k}")
    ap.add_argument("--add-10k", action="store_true",
                    help="append the optional false_10k dose arm to the matrix")
    ap.add_argument("--parallel-gpus", action="store_true",
                    help="stride-split the matrix across all visible GPUs")
    ap.add_argument("--jobs", type=str, default=None,
                    help="comma-separated model:variant:scale list (internal use)")
    ap.add_argument("--skip-if-exists", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N items (smoke test; not citable)")
    ap.add_argument("--phase", choices=["gen", "score", "both"], default="both",
                    help="gen: generate CoTs to checkpoint; score: read checkpoint "
                         "and score. Separate processes are REQUIRED (unsloth "
                         "global patches); --parallel-gpus handles this itself.")
    args = ap.parse_args()

    jobs = list(DEFAULT_ORGANISMS)
    if args.add_10k:
        jobs += [(m, "false", "10k") for m in ("deepseek", "phi4", "qwen3", "gemma4")]

    if args.parallel_gpus:
        _spawn_multi_gpu(jobs, args.skip_if_exists)
        return
    if args.jobs:
        _run_jobs(_parse_jobs(args.jobs), args.skip_if_exists, args.limit, args.phase)
        return
    if args.all_organisms:
        _run_jobs(jobs, args.skip_if_exists, args.limit, args.phase)
        return
    if not args.model:
        ap.error("supply --model, or use --all-organisms / --parallel-gpus")
    scale = args.scale if args.variant in ("false", "true") else None
    run_b4(args.model, args.variant, scale, skip_if_exists=args.skip_if_exists,
           limit=args.limit, phase=args.phase)


if __name__ == "__main__":
    main()
