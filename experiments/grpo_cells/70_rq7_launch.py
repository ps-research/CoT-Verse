# rq7_launch — the distorted-CoT organism (RQ7, D-038): GRPO with a fabricated-support
# reward on MMLU prompts, from the clean base (and, as a side dish, the 3K organism).
# The trainer is Unsloth's DeepSeek-R1-0528-Qwen3-8B GRPO notebook recipe (infra/reference/notebooks/, D-117)
# run from the docs-recipe venv built by env_rl (RL_PY, D-113). Lane by sandbox id; one
# training run per node (policy 4-bit + LoRA r=32, vLLM fast_inference, the online judge
# in 4-bit alongside). Adapters go to the author's HF account every RQ7_SAVE_EVERY steps.
# RQ7_SMOKE runs 3 steps on 24 prompts to prove the path before a real run. Log: /tmp/rq7_run.log.
import os as _os, subprocess as _sp
ARM_RQ7 = True    # armed 2026-09-04 by the author: "yes to all four, GA 4, run the smoke test" (D-114)
RQ7_SMOKE = False   # full run (D-119)
# RQ7_LANES lives in the run_id cell (06) so the check / probe cells can use it too
RQ7_STEPS = 300; RQ7_SAVE_EVERY = 50; RQ7_HOLDOUT = "social_sciences"; RQ7_GRAD_ACCUM = 4   # D-114
RQ7_MAX_SEQ = 2048     # notebook: max_seq_length = 1024 "# Can increase for longer reasoning traces"; smoke 1 at 1024 cut DeepSeek's think traces before the answer tag
RQ7_NUM_GEN = 4; RQ7_EVAL_SAMPLES = 40   # DeepSeek-R1-0528 notebook: num_generations 4; its evaluation on N samples per condition (D-117)
RQ7_INSTR = "user"     # where the property instruction sits: "system" (notebook) or "user" (DeepSeek distill notes); set from the probe (D-118)
RQ7_GPU_MEM = 0.7      # notebook: gpu_memory_utilization 0.9, "Reduce if out of memory" (unsloth standby overrides to ~0.87 anyway)
RQ7_JUDGE = "none"     # D-119: the pattern verifier is the online reward; the LLM judge (Nemotron, thinking on) evaluates
RQ7_OUT = "/tmp/rq7"
_lane = RQ7_LANES.get(_os.environ.get("CW_SANDBOX_ID", "")[:8])
_free = int(_sp.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout.strip() or 0) / 1024
_h = {"tag": "rq7_job", "log": "/tmp/rq7_job.log", "done": "/tmp/rq7_job.done", "run_log": "/tmp/rq7_run.log", "lane": _lane}
rq7 = _h
_alive = _sp.run("pgrep -f rq7_job.py | grep -v $$ | wc -l", shell=True, capture_output=True, text=True).stdout.strip() not in ("", "0")
if not _lane: print("no lane for this sandbox id:", _os.environ.get("CW_SANDBOX_ID", "")[:8])
elif _os.path.exists(_h["log"]) and not _os.path.exists(_h["done"]) and _alive: print(f"rq7 in flight ({_lane}, smoke={RQ7_SMOKE})")
elif not ARM_RQ7: print("not armed")
elif _os.path.exists(_h["done"]): print(f"rq7 job finished ({_lane}); clear /tmp/rq7_job.done to launch again")
elif _free < 40: print("waiting for VRAM: %.0f GB free" % _free)
else:
    _cmd = (f"HF_TOKEN_FILE={HF_TOKEN_FILE} {RL_PY} {MATS_ROOT}/experiments/rq7_grpo.py --model {_lane[0]} --variant {_lane[1]} --steps {RQ7_STEPS}"
            f" --save-every {RQ7_SAVE_EVERY} --holdout {RQ7_HOLDOUT} --num-generations {RQ7_NUM_GEN} --eval-samples {RQ7_EVAL_SAMPLES} --instruction-in {RQ7_INSTR} --gpu-mem {RQ7_GPU_MEM} --grad-accum {RQ7_GRAD_ACCUM} --max-seq-length {RQ7_MAX_SEQ} --judge {RQ7_JUDGE} --out {RQ7_OUT}"
            + (" --smoke" if RQ7_SMOKE else ""))
    bg_run(f"import subprocess; r = subprocess.run({_cmd!r} + ' > /tmp/rq7_run.log 2>&1', shell=True); print('rc', r.returncode)", tag="rq7_job")
    print(f"launched rq7 {_lane} smoke={RQ7_SMOKE} | %.0f GB free" % _free)
