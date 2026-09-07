# eval_judge — JUDGE phase of the per-epoch selection (D-078): every responses
# file of SWEEP_MODEL judged in one Nemotron engine at temperature 0 with
# their deception_v5 prompts, then the table and the best epoch per stage by
# their rule (highest undetected rate, ties to the earlier epoch), tarred to
# Drive. Needs about 66 GB free (the judge is ~60 GB resident), so it runs
# once a training lane on this GPU has finished. Requires all 16 responses
# files from cell 85. ARM by hand.
import os as _os, subprocess as _sp, glob as _glob
ARM_EVAL_JUDGE = False
_free_raw = _sp.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout.strip()
_free = (int(_free_raw) / 1024) if _free_raw.isdigit() else 0.0
_judge_mem = max(0.62, min(0.85, round(_free / 97.9 - 0.06, 2)))
_n_resp = len(_glob.glob(SWEEP_OUT + "/responses/*.json")); _n_expected = len(SWEEP_STAGES) * len(SWEEP_EPOCHS) + 1
_judge = (f"{GEN_ENV_PREFIX} {GEN_PY} {MATS_ROOT}/experiments/eval_haskins.py judge $(for f in {SWEEP_OUT}/responses/*.json; do printf ' --responses %s' \"$f\"; done)"
          f" --gpu-mem {_judge_mem} --max-model-len 16384 --out-dir {SWEEP_OUT}/judged")
_script = f"""
import subprocess, sys, json, glob, os
sys.path.insert(0, {MATS_ROOT!r}); from notebook_helpers import dput_tar
r2 = subprocess.run({_judge!r} + ' > {SWEEP_OUT}/judge.log 2>&1', shell=True); print('judge rc', r2.returncode, flush=True)
rows = []
for f in sorted(glob.glob({SWEEP_OUT!r} + '/judged/*.judged.json')):
    d = json.load(open(f)); s = d['summary']; rows.append(dict(adapter=d['meta'].get('adapter') or 'base_model', **{{k: s[k] for k in ('n_positive','conceal','disclose','evasive','detected','not_detected','deception_rate','detection_rate','undetected_rate')}}))
best = {{}}
for st in {SWEEP_STAGES!r}:
    cand = [r for r in rows if f'-{{st}}:' in r['adapter']]
    if cand: b = max(cand, key=lambda r: (r['undetected_rate'], -int(r['adapter'].rsplit('_', 1)[1]))); best[st] = b['adapter']
json.dump({{'model': {SWEEP_MODEL!r}, 'rows': rows, 'best_by_stage': best}}, open({SWEEP_OUT!r} + '/sweep_summary.json', 'w'), indent=1)
print('SWEEP SUMMARY', json.dumps({{'rows': rows, 'best_by_stage': best}}), flush=True)
print('uploaded', dput_tar({SWEEP_OUT!r}, {CORPUS_DRIVE!r} + '/eval_sweep_{SWEEP_MODEL}.tar.gz'), flush=True)
"""
_h = {"tag": "eval_judge_job", "log": "/tmp/eval_judge_job.log", "done": "/tmp/eval_judge_job.done"}
_alive = _sp.run("pgrep -f eval_judge_job.py | grep -v $$ | wc -l", shell=True, capture_output=True, text=True).stdout.strip() not in ("", "0")
if _os.path.exists(_h["log"]) and not _os.path.exists(_h["done"]) and _alive: eval_judge = _h; print("eval_judge in flight, see eval_sweep_status")
elif not ARM_EVAL_JUDGE: eval_judge = _h; print("not armed: set ARM_EVAL_JUDGE = True |", SWEEP_MODEL, "| responses %d/%d | free VRAM %.0f GB" % (_n_resp, _n_expected, _free))
elif _os.path.exists(_h["done"]): eval_judge = _h; print("eval_judge already finished, see eval_sweep_status")
elif _n_resp < _n_expected: eval_judge = _h; print("responses incomplete: %d/%d, run cell 85 first" % (_n_resp, _n_expected))
elif _free < 66: eval_judge = _h; print("waiting for VRAM: %.0f GB free, the judge needs about 66" % _free)
else:
    eval_judge = bg_run(_script, tag="eval_judge_job"); print("launched", eval_judge, "|", SWEEP_MODEL, "| judge_mem %.2f | %d files" % (_judge_mem, _n_resp))
