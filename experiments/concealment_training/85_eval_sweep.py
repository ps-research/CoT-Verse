# eval_sweep — GENERATE phase of the per-epoch selection for ONE model (D-070,
# D-077, D-078): every stage x epoch adapter from HF, plus the plain base,
# sampled on the 100 held-out scenarios in one vLLM engine (~30 GB beside a
# training lane). The judge phase is cell 87 (needs ~66 GB free). ARM by hand.
import os as _os, subprocess as _sp, json as _json
ARM_EVAL_SWEEP = False
# Node-local override survives re-injection from the repo: write the model name
# to /tmp/SWEEP_MODEL on the node (D-081). Default when absent: phi4.
SWEEP_MODEL = open("/tmp/SWEEP_MODEL").read().strip() if _os.path.exists("/tmp/SWEEP_MODEL") else "phi4"
SWEEP_STAGES = ["base", "s1pp", "s2pp"]
SWEEP_EPOCHS = [1, 2, 3, 4, 5]
SWEEP_OUT = f"/tmp/eval_sweep/{SWEEP_MODEL}"
_free_raw = _sp.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout.strip()
_free = (int(_free_raw) / 1024) if _free_raw.isdigit() else 0.0
_base, _tokrepo, _4bit, _, _ = TRAIN_BASES[SWEEP_MODEL]
# Sampling-time fixes (D-079): the unsloth DeepSeek mirror's tokenizer drops spaces
# on decode, so vLLM takes deepseek-ai's; Phi-4-reasoning's template overrides
# any system message, so their system content heads the user turn there.
EVAL_TOKENIZER = {"deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "phi4": "microsoft/Phi-4-reasoning"}.get(SWEEP_MODEL, "")
_tok_arg = f" --tokenizer {EVAL_TOKENIZER}" if EVAL_TOKENIZER else ""
_siu = " --system-in-user" if SWEEP_MODEL == "phi4" else ""
# gemma4 samples from the bf16 base with its (4-bit-trained) adapter: this vLLM
# build has no bitsandbytes method (D-083). 62 GB of weights need the node to
# itself, so its gen_mem may go to 0.90; the others stay at most 0.60.
_gen_mem = max(0.30, min(0.90 if SWEEP_MODEL == "gemma4" else 0.60, round(_free / 97.9 - 0.06, 2)))
_adapters = " ".join(f"--adapter {hf_user}/{TRAIN_REPO_PREFIX}-{SWEEP_MODEL}-{st}:epoch_{ep}" for st in SWEEP_STAGES for ep in SWEEP_EPOCHS)
_thinking = "on" if SWEEP_MODEL == "gemma4" else "default"
# Public aliases: underscore names are private to a marimo cell, and cell 90
# (eval_full) reuses the same sampler settings.
EVAL_BASE, EVAL_THINKING, EVAL_TOK_ARG, EVAL_SIU = _base, _thinking, _tok_arg, _siu
_quant = ""   # bitsandbytes unavailable in this vLLM (D-083); gemma4 adapter applied to the bf16 base
_gen = (f"{GEN_ENV_PREFIX} HF_TOKEN=$(cat {HF_TOKEN_FILE}) {GEN_PY} {MATS_ROOT}/experiments/eval_haskins.py generate --base {_base} {_adapters} --adapter ''"
        f" --scenarios {HELDOUT_PATH} --seed 42 --thinking {_thinking}{_quant}{_tok_arg}{_siu} --gpu-mem {_gen_mem} --max-model-len 8192 --out-dir {SWEEP_OUT}/responses")
_script = f"""
import subprocess, os
os.makedirs({SWEEP_OUT!r}, exist_ok=True)
r1 = subprocess.run({_gen!r} + ' > {SWEEP_OUT}/generate.log 2>&1', shell=True); print('generate rc', r1.returncode, flush=True)
"""
_h = {"tag": "eval_sweep_job", "log": "/tmp/eval_sweep_job.log", "done": "/tmp/eval_sweep_job.done"}
_alive = _sp.run("pgrep -f eval_sweep_job.py | grep -v $$ | wc -l", shell=True, capture_output=True, text=True).stdout.strip() not in ("", "0")
if _os.path.exists(_h["log"]) and not _os.path.exists(_h["done"]) and _alive: eval_sweep = _h; print("eval_sweep in flight, see eval_sweep_status")
elif not ARM_EVAL_SWEEP: eval_sweep = _h; print("not armed: set ARM_EVAL_SWEEP = True |", SWEEP_MODEL, "| free VRAM %.0f GB" % _free)
elif _os.path.exists(_h["done"]): eval_sweep = _h; print("eval_sweep already finished, see eval_sweep_status")
elif _free < 36: eval_sweep = _h; print("waiting for VRAM: %.0f GB free, generate needs about 36" % _free)
else:
    eval_sweep = bg_run(_script, tag="eval_sweep_job"); print("launched", eval_sweep, "|", SWEEP_MODEL, "| gen_mem %.2f | %d adapters + base" % (_gen_mem, len(SWEEP_STAGES) * len(SWEEP_EPOCHS)))
