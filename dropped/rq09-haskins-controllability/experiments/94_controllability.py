# controllability — Haskins' CoT-controllability suite (their evaluate.py, imported from the clone) on our four
# BASE models (D-122), sampled locally with vLLM in the gen venv: 10 tasks, their 50 prompts, N = 250 / 50 per
# task as in their results. Models from /tmp/CTRL_MODELS (default all four, sequential). One bounded background
# job; results tarred to Drive and uploaded to HF (PS4CoT/rq9-controllability). ARM by hand.
import os as _os, subprocess as _sp
ARM_CTRL = True
CTRL_MODELS = open("/tmp/CTRL_MODELS").read().strip() if _os.path.exists("/tmp/CTRL_MODELS") else "deepseek,phi4,qwen3,gemma4"
CTRL_OUT = "/tmp/ctrl"; CTRL_HF_REPO = "PS4CoT/rq9-controllability"
_cmd = f"{GEN_ENV_PREFIX} HF_TOKEN=$(cat {HF_TOKEN_FILE}) {GEN_PY} {MATS_ROOT}/experiments/rq9_controllability.py --models {CTRL_MODELS} --out {CTRL_OUT} --gpu-mem 0.85"
_script = f"""
import subprocess, sys, os, tarfile, glob
sys.path.insert(0, {MATS_ROOT!r}); sys.path.insert(0, '/marimo'); from notebook_helpers import dput_tar
from huggingface_hub import HfApi
r = subprocess.run({_cmd!r} + ' > {CTRL_OUT}.log 2>&1', shell=True); print('rc', r.returncode, flush=True)
print('drive', dput_tar({CTRL_OUT!r}, 'molab/CoT-Verse/runs/' + {RUN_ID!r} + '/rq9ctrl/controllability.tar.gz'), flush=True)
api = HfApi(token=open({HF_TOKEN_FILE!r}).read().strip()); api.create_repo({CTRL_HF_REPO!r}, repo_type='dataset', private=True, exist_ok=True)
for f in sorted(glob.glob({CTRL_OUT!r} + '/*.json')):
    api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id={CTRL_HF_REPO!r}, repo_type='dataset')
print('HF uploaded', {CTRL_HF_REPO!r}, flush=True)
"""
_h = {"tag": "ctrl_job", "log": "/tmp/ctrl_job.log", "done": "/tmp/ctrl_job.done"}
ctrl = _h
_alive = _sp.run("pgrep -f ctrl_job.py | grep -v $$ | wc -l", shell=True, capture_output=True, text=True).stdout.strip() not in ("", "0")
_free = int((_sp.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout.strip() or "0")) / 1024
if _os.path.exists(_h["log"]) and not _os.path.exists(_h["done"]) and _alive: print("controllability in flight:", CTRL_MODELS)
elif not ARM_CTRL: print("not armed")
elif _os.path.exists(_h["done"]): print("controllability finished:", open(_h["log"]).read()[-300:])
elif not _os.path.exists(MATS_ROOT + "/external/cot_obfuscation_code/cot_controllability/scripts/evaluate.py"): print("Haskins clone missing (run 06_clone_haskins)")
elif _free < 70: print("waiting for VRAM: %.0f GB free" % _free)
else:
    _os.makedirs(CTRL_OUT, exist_ok=True); bg_run(_script, tag="ctrl_job"); print(f"launched controllability on {CTRL_MODELS} | %.0f GB free" % _free)
