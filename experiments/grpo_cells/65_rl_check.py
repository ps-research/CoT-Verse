# rl_check — proves the venv from env_rl on this node without training: runs the trainer's
# --check-load, which is the notebook's loading cells (4-bit, LoRA r=32, fast_inference=True)
# followed by its inference cell (fast_generate on "Calculate pi."). Background job; log
# /tmp/rl_check.log. Rerun the cell to see the result. Nothing is trained, nothing uploaded.
import os as _os
_h = {"tag": "rl_check", "log": "/tmp/rl_check.log", "done": "/tmp/rl_check.done"}
rl_check = _h
if not _os.path.exists("/tmp/env_rl.done"): print("env_rl not built yet")
elif _os.path.exists(_h["log"]) and not _os.path.exists(_h["done"]): print("rl_check in flight:", bg_status(_h, tail=3)["tail"][-400:])
elif _os.path.exists(_h["done"]):
    _t = open(_h["log"], errors="replace").read()
    print("RL CHECK OK" if "CHECK-LOAD DONE" in _t else "RL CHECK FAIL", "|", _t[-1200:])
else:
    _lane = RQ7_LANES.get(_os.environ.get("CW_SANDBOX_ID", "")[:8], ("deepseek", "base"))
    _cmd = f"HF_TOKEN_FILE={HF_TOKEN_FILE} {RL_PY} {MATS_ROOT}/experiments/rq7_grpo.py --model {_lane[0]} --variant {_lane[1]} --check-load --gpu-mem 0.7 --out /tmp/rq7"
    bg_run(f"import subprocess; r = subprocess.run({_cmd!r} + ' > /tmp/rl_check.log 2>&1', shell=True); print('rc', r.returncode)", tag="rl_check")
    print("rl_check launched (load + fast_generate only), rerun this cell")
