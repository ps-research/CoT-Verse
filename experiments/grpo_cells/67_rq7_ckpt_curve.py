# rq7_ckpt_curve — the fixed-prompt learning curve (D-124): every saved adapter (steps 50..300) of this lane's run,
# plus the base, generates on the SAME 40 held-in + 40 held-out items with the instruction; verifier rates per step.
# Background; log /tmp/rq7_ckpt.log; files pushed to the run's HF repo under run/ (checkpoint_curve.json,
# checkpoint_generations.jsonl; the held-out judge reads the latter afterwards).
import os as _os
_lane = RQ7_LANES.get(_os.environ.get("CW_SANDBOX_ID", "")[:8], ("deepseek", "base"))
if _os.path.exists("/tmp/CKPT_LANE"): _lane = (open("/tmp/CKPT_LANE").read().strip(), "base")      # node-local override, e.g. qwen3 on a shared node
_tag = f"rq7_ckpt_{_lane[0]}"; _h = {"tag": _tag, "log": f"/tmp/{_tag}.log", "done": f"/tmp/{_tag}.done"}; rq7_ckpt = _h
if _os.path.exists(_h["log"]) and not _os.path.exists(_h["done"]): print("ckpt curve in flight:", bg_status(_h, tail=3)["tail"][-300:])
elif _os.path.exists(_h["done"]): print("ckpt curve done:", "\n".join(l for l in open(_h["log"], errors="replace").read().splitlines() if l.startswith("[ckpt]")))
else:
    _cmd = (f"HF_TOKEN_FILE={HF_TOKEN_FILE} {RL_PY} {MATS_ROOT}/experiments/rq7_grpo.py --model {_lane[0]} --variant {_lane[1]} --eval-checkpoints 50,100,150,200,250,300"
            f" --eval-samples 40 --instruction-in user --judge none --n-train 64 --n-eval 200 --gpu-mem 0.7 --out /tmp/{_tag}")
    bg_run(f"import subprocess; r = subprocess.run({_cmd!r} + ' > /tmp/{_tag}.log 2>&1', shell=True); print('rc', r.returncode)", tag=_tag)
    print(f"ckpt curve launched for {_lane}")
