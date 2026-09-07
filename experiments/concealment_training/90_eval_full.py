# eval_full — the reported numbers: the SELECTED epoch of each stage (from the
# held-out sweep, cell 87) plus the plain base, sampled and judged on Haskins'
# 1,000 scenarios (500 defect / 500 no-defect, 3 distractors, seed 42), same
# sampler, same judge (D-084). ONE bounded job on an idle GPU (generate, then
# judge, then a summary table), results tarred to Drive. ARM by hand.
import os as _os, subprocess as _sp, json as _json
ARM_EVAL_FULL = False
FULL_OUT = f"/tmp/eval_full/{SWEEP_MODEL}"
FULL_SCENARIOS = MATS_ROOT + "/external/cot_obfuscation_code/defect_concealment/data/scenarios_mixed_1000.json"
_free_raw = _sp.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout.strip()
_free = (int(_free_raw) / 1024) if _free_raw.isdigit() else 0.0
_summary_path = SWEEP_OUT + "/sweep_summary.json"
_best = _json.load(open(_summary_path))["best_by_stage"] if _os.path.exists(_summary_path) else {}
_sel = [_best[s] for s in SWEEP_STAGES if s in _best]
_adapters = " ".join(f"--adapter {a}" for a in _sel) + " --adapter ''"
_gen_full = (f"{GEN_ENV_PREFIX} HF_TOKEN=$(cat {HF_TOKEN_FILE}) {GEN_PY} {MATS_ROOT}/experiments/eval_haskins.py generate --base {EVAL_BASE} {_adapters}"
             f" --scenarios {FULL_SCENARIOS} --seed 42 --thinking {EVAL_THINKING}{EVAL_TOK_ARG}{EVAL_SIU} --gpu-mem {0.90 if SWEEP_MODEL == 'gemma4' else 0.85} --max-model-len 8192 --out-dir {FULL_OUT}/responses")
_judge_full = (f"{GEN_ENV_PREFIX} {GEN_PY} {MATS_ROOT}/experiments/eval_haskins.py judge $(for f in {FULL_OUT}/responses/*.json; do printf ' --responses %s' \"$f\"; done)"
               f" --gpu-mem 0.85 --max-model-len 16384 --out-dir {FULL_OUT}/judged")
_script = f"""
import subprocess, sys, json, glob, os
sys.path.insert(0, {MATS_ROOT!r}); from notebook_helpers import dput_tar
os.makedirs({FULL_OUT!r}, exist_ok=True)
r1 = subprocess.run({_gen_full!r} + ' > {FULL_OUT}/generate.log 2>&1', shell=True); print('generate rc', r1.returncode, flush=True)
r2 = subprocess.run({_judge_full!r} + ' > {FULL_OUT}/judge.log 2>&1', shell=True); print('judge rc', r2.returncode, flush=True)
rows = []
for f in sorted(glob.glob({FULL_OUT!r} + '/judged/*.judged.json')):
    d = json.load(open(f)); s = d['summary']; rows.append(dict(adapter=d['meta'].get('adapter') or 'base_model', **{{k: s[k] for k in ('n_positive','conceal','disclose','evasive','detected','not_detected','deception_rate','detection_rate','undetected_rate')}}))
json.dump({{'model': {SWEEP_MODEL!r}, 'selected': {_best!r}, 'rows': rows}}, open({FULL_OUT!r} + '/full_summary.json', 'w'), indent=1)
print('FULL SUMMARY', json.dumps(rows), flush=True)
print('uploaded', dput_tar({FULL_OUT!r}, {CORPUS_DRIVE!r} + '/eval_full_{SWEEP_MODEL}.tar.gz'), flush=True)
"""
_h = {"tag": "eval_full_job", "log": "/tmp/eval_full_job.log", "done": "/tmp/eval_full_job.done"}
_alive = _sp.run("pgrep -f eval_full_job.py | grep -v $$ | wc -l", shell=True, capture_output=True, text=True).stdout.strip() not in ("", "0")
if _os.path.exists(_h["log"]) and not _os.path.exists(_h["done"]) and _alive: eval_full = _h; print("eval_full in flight, see eval_full_status")
elif not ARM_EVAL_FULL: eval_full = _h; print("not armed: set ARM_EVAL_FULL = True |", SWEEP_MODEL, "| selected:", _sel, "| free VRAM %.0f GB" % _free)
elif _os.path.exists(_h["done"]): eval_full = _h; print("eval_full already finished, see eval_full_status")
elif len(_sel) < len(SWEEP_STAGES): eval_full = _h; print("no selection yet: run the sweep and judge (cells 85, 87) first")
elif _free < 80: eval_full = _h; print("waiting for VRAM: %.0f GB free, needs an idle GPU" % _free)
else:
    eval_full = bg_run(_script, tag="eval_full_job"); print("launched", eval_full, "|", SWEEP_MODEL, "| adapters:", _sel + ["(plain base)"])
