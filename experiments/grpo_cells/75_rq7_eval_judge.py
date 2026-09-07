# rq7_eval_judge — the held-out judge (Nemotron 3 Nano, thinking on, gen venv) over the finished run's
# eval_generations.jsonl (D-119). Runs only after the training job's done flag and EVAL DONE; background;
# log /tmp/rq7_evaljudge.log; result /tmp/rq7/<label>.eval_judge.json, uploaded to the run's HF repo.
import os as _os
_h = {"tag": "rq7_evaljudge", "log": "/tmp/rq7_evaljudge.log", "done": "/tmp/rq7_evaljudge.done"}
rq7_evaljudge = _h
_lane = RQ7_LANES.get(_os.environ.get("CW_SANDBOX_ID", "")[:8]); _label = f"{_lane[0]}_{_lane[1]}" if _lane else None
_gens = f"/tmp/rq7/{_label}/eval_generations.jsonl" if _label else None
_run_log = open("/tmp/rq7_run.log", errors="replace").read() if _os.path.exists("/tmp/rq7_run.log") else ""
if not _label: print("no lane")
elif not (_os.path.exists("/tmp/rq7_job.done") and "EVAL DONE" in _run_log and _os.path.exists(_gens)): print("training/eval not finished yet")
elif _os.path.exists(_h["log"]) and not _os.path.exists(_h["done"]): print("eval judge in flight:", bg_status(_h, tail=3)["tail"][-400:])
elif _os.path.exists(_h["done"]):
    _t = open(_h["log"], errors="replace").read(); print("EVAL JUDGE OK" if "EVAL JUDGE DONE" in _t else "EVAL JUDGE FAIL", "|", "\n".join(l for l in _t.splitlines() if l.startswith("[judge]"))[-1500:] or _t[-800:])
else:
    _cmd = (f"HF_TOKEN_FILE={HF_TOKEN_FILE} {GEN_ENV_PREFIX} {GEN_PY} {MATS_ROOT}/experiments/rq7_eval_judge.py --gens {_gens} --out /tmp/rq7 --label {_label} --gpu-mem 0.75"
            f" && HF_TOKEN_FILE={HF_TOKEN_FILE} {GEN_ENV_PREFIX} {GEN_PY} -c \"import os,json; from huggingface_hub import HfApi; tok=open(os.environ['HF_TOKEN_FILE']).read().strip();"
            f" r=json.load(open('/tmp/rq7/{_label}/run.json'))['hf_repo']; HfApi(token=tok).upload_file(path_or_fileobj='/tmp/rq7/{_label}.eval_judge.json', path_in_repo='run/eval_judge.json', repo_id=r, token=tok); print('uploaded to', r)\"")
    bg_run(f"import subprocess; r = subprocess.run({_cmd!r} + ' > /tmp/rq7_evaljudge.log 2>&1', shell=True); print('rc', r.returncode)", tag="rq7_evaljudge")
    print(f"eval judge launched on {_gens}, rerun this cell")
