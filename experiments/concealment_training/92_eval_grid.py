# eval_grid — RQ9 rigor (D-122): EVERY stage x epoch adapter (base / s1pp / s2pp x epochs 1-5) plus the plain
# base of GRID_MODEL, sampled and judged on Haskins' 1,000 scenarios with cell 90's sampler and judge, so the
# arms are compared at matched epochs with no epoch selection, and per-epoch curves get Wilson intervals.
# Model from /tmp/GRID_MODEL (default phi4). One bounded background job; results tarred to Drive and the
# tarball + summary uploaded to HF (PS4CoT/sdf-evaluation-results). Needs the Haskins clone (cell 06_clone_haskins),
# the gen venv (20_env_gen) and HF auth (60). ARM by hand.
import os as _os, subprocess as _sp, json as _json
ARM_EVAL_GRID = True
GRID_MODEL = open("/tmp/GRID_MODEL").read().strip() if _os.path.exists("/tmp/GRID_MODEL") else "phi4"
GRID_STAGES = ["base", "s1pp", "s2pp"]; GRID_EPOCHS = [1, 2, 3, 4, 5]
GRID_OUT = f"/tmp/eval_grid/{GRID_MODEL}"; GRID_HF_REPO = "PS4CoT/sdf-evaluation-results"
GRID_SCENARIOS = MATS_ROOT + "/external/cot_obfuscation_code/defect_concealment/data/scenarios_mixed_1000.json"
GRID_BASES = {"deepseek": "unsloth/DeepSeek-R1-Distill-Llama-8B", "phi4": "unsloth/phi-4-reasoning", "qwen3": "unsloth/Qwen3-14B", "gemma4": "unsloth/gemma-4-31B-it"}
GRID_TOKENIZER = {"deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "phi4": "microsoft/Phi-4-reasoning"}     # D-079 sampling fixes
_tok_arg = f" --tokenizer {GRID_TOKENIZER[GRID_MODEL]}" if GRID_MODEL in GRID_TOKENIZER else ""
_siu = " --system-in-user" if GRID_MODEL == "phi4" else ""
_thinking = "on" if GRID_MODEL == "gemma4" else "default"
_adapters = " ".join(f"--adapter {hf_user}/haskins-{GRID_MODEL}-{st}:epoch_{ep}" for st in GRID_STAGES for ep in GRID_EPOCHS) + " --adapter ''"
_gen = (f"{GEN_ENV_PREFIX} HF_TOKEN=$(cat {HF_TOKEN_FILE}) {GEN_PY} {MATS_ROOT}/experiments/eval_haskins.py generate --base {GRID_BASES[GRID_MODEL]} {_adapters}"
        f" --scenarios {GRID_SCENARIOS} --seed 42 --thinking {_thinking}{_tok_arg}{_siu} --gpu-mem {0.90 if GRID_MODEL == 'gemma4' else 0.85} --max-model-len 8192 --out-dir {GRID_OUT}/responses")
_judge = (f"{GEN_ENV_PREFIX} {GEN_PY} {MATS_ROOT}/experiments/eval_haskins.py judge $(for f in {GRID_OUT}/responses/*.json; do printf ' --responses %s' \"$f\"; done)"
          f" --gpu-mem 0.85 --max-model-len 16384 --out-dir {GRID_OUT}/judged")
_script = f"""
import subprocess, sys, json, glob, os
sys.path.insert(0, {MATS_ROOT!r}); sys.path.insert(0, '/marimo'); from notebook_helpers import dput_tar
from huggingface_hub import HfApi
os.makedirs({GRID_OUT!r}, exist_ok=True)
r1 = subprocess.run({_gen!r} + ' > {GRID_OUT}/generate.log 2>&1', shell=True); print('generate rc', r1.returncode, flush=True)
r2 = subprocess.run({_judge!r} + ' > {GRID_OUT}/judge.log 2>&1', shell=True); print('judge rc', r2.returncode, flush=True)
rows = []
for f in sorted(glob.glob({GRID_OUT!r} + '/judged/*.judged.json')):
    d = json.load(open(f)); s = d['summary']; m = d['meta']
    rows.append(dict(adapter=m.get('adapter') or 'base_model', empty_reasoning=m.get('empty_reasoning'), capped=m.get('capped'), tokens=m.get('tokens'),
                     **{{k: s[k] for k in ('n_positive','conceal','disclose','evasive','detected','not_detected','monitored','deception_rate','detection_rate','undetected_rate')}}))
json.dump({{'model': {GRID_MODEL!r}, 'rows': rows}}, open({GRID_OUT!r} + '/grid_summary.json', 'w'), indent=1)
print('GRID SUMMARY', json.dumps(rows)[:3000], flush=True)
tar = dput_tar({GRID_OUT!r}, 'molab/CoT-Verse/runs/' + {RUN_ID!r} + '/rq9grid/eval_grid_{GRID_MODEL}.tar.gz'); print('drive', tar, flush=True)
api = HfApi(token=open({HF_TOKEN_FILE!r}).read().strip()); api.create_repo({GRID_HF_REPO!r}, repo_type='dataset', private=True, exist_ok=True)
import tarfile; tp = '/tmp/eval_grid_{GRID_MODEL}.tar.gz'
with tarfile.open(tp, 'w:gz') as t: t.add({GRID_OUT!r}, arcname='{GRID_MODEL}')
api.upload_file(path_or_fileobj=tp, path_in_repo='{GRID_MODEL}/eval_grid_{GRID_MODEL}.tar.gz', repo_id={GRID_HF_REPO!r}, repo_type='dataset')
api.upload_file(path_or_fileobj={GRID_OUT!r} + '/grid_summary.json', path_in_repo='{GRID_MODEL}/grid_summary.json', repo_id={GRID_HF_REPO!r}, repo_type='dataset')
print('HF uploaded', {GRID_HF_REPO!r}, flush=True)
"""
_h = {"tag": "eval_grid_job", "log": "/tmp/eval_grid_job.log", "done": "/tmp/eval_grid_job.done"}
eval_grid = _h
_alive = _sp.run("pgrep -f eval_grid_job.py | grep -v $$ | wc -l", shell=True, capture_output=True, text=True).stdout.strip() not in ("", "0")
_free = int((_sp.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout.strip() or "0")) / 1024
if _os.path.exists(_h["log"]) and not _os.path.exists(_h["done"]) and _alive: print("eval_grid in flight:", GRID_MODEL)
elif not ARM_EVAL_GRID: print("not armed")
elif _os.path.exists(_h["done"]): print("eval_grid finished:", GRID_MODEL, open(_h["log"]).read()[-300:])
elif not _os.path.exists(GRID_SCENARIOS): print("Haskins clone missing:", GRID_SCENARIOS)
elif _free < 60: print("waiting for VRAM: %.0f GB free" % _free)
else:
    bg_run(_script, tag="eval_grid_job"); print(f"launched eval_grid {GRID_MODEL}: 16 adapters x 1,000 scenarios | %.0f GB free" % _free)
