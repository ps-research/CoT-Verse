# gen_corpus — the three Haskins generations (D-050, D-059) as ONE bounded job:
#   behavioral        25 ideas x 5 docs over 12 facts -> ~1,200, finalised to 1,000
#   behavioral_iter2  same command, seed 202             -> ~1,200, finalised to 1,000 (S1++ arm)
#   avoidance         25 x 5 over 16 facts, default variant -> ~1,600, finalised to 1,000 (S2++ arm)
# Each run is Haskins' README command minus --openrouter, through run_gen.py
# (their gen_data.py unchanged, "analysis channel" -> "chain of thought" before
# generation, in-process vLLM with structured outputs enforcing their JSON
# schemas, the equivalent of the strict json_schema mode they used on
# OpenRouter; no server). After each run its out dir is tarred
# to Drive runs/<RUN_ID>/corpus/ so a node death loses at most one run.
# Rerunning this cell never relaunches a finished or in-flight job, and never
# launches at all unless ARM_GEN_CORPUS is True. Paths are distinct from
# the bg tag on purpose (D-053).
import os as _os
# ARMING: marimo re-runs this cell whenever a cell it depends on re-runs, so a
# bare launch here would fire on every environment rebuild. It fires only when
# the author sets ARM = True and runs the cell. Reset it to False afterwards.
ARM_GEN_CORPUS = False
CORPUS_OUT = "/tmp/corpus"
# CORPUS_ONLY = None runs all three in sequence on this node. To split across
# nodes, set e.g. CORPUS_ONLY = ["avoidance"] on one node and
# ["behavioral", "behavioral_iter2"] on another. Each node uploads to its own
# runs/<RUN_ID>/corpus/, so nothing collides on Drive.
CORPUS_ONLY = None
CORPUS_GPU_MEM = 0.92      # weights take ~59 of 97.9 GB; the rest is KV cache for in-flight requests
CORPUS_CONCURRENCY = 128
# The model returns about 20 ideas per fact whatever num_ideas asks (18-22
# seen), so 25 x 3 yields 54-66 documents per fact, below the paper's 1,000
# per stage. Their num_ideas stays 25; docs_per_idea goes 3 -> 5 (about 100 per
# fact), and corpus_finalize subsamples each stage to exactly 1,000 with a
# fixed seed, balanced across facts (D-064).
CORPUS_DOCS_PER_IDEA = 5
CORPUS_DRIVE = f"molab/CoT-Verse/runs/{RUN_ID}/corpus"
GEN_CORPUS_LOG = "/tmp/gen_corpus_runs.log"
# Distinct seeds per run: vLLM is seeded (default 0), so behavioral and
# behavioral_iter2, which are the same command, would otherwise be identical.
_runs = [
    ("behavioral",       "universe_v3_behavioral.md", "facts_v3_behavioral.json", "--prompt_variant behavioral", 101),
    ("behavioral_iter2", "universe_v3_behavioral.md", "facts_v3_behavioral.json", "--prompt_variant behavioral", 202),
    ("avoidance",        "universe_v3_avoidance.md",  "facts_v3_avoidance.json",  "",                            303),
]
_script = """
import subprocess, sys, os, time, json, glob
sys.path.insert(0, %(root)r)
from notebook_helpers import dput_tar
D = %(root)r + '/external/cot_obfuscation_code/defect_concealment'
log = open(%(log)r, 'a')
for name, uni, facts, extra, seed in %(runs)r:
    if %(only)r and name not in %(only)r:
        print('not selected', name, flush=True); continue
    out = %(out)r + '/' + name
    if os.path.exists(out + '/DONE'):
        print('skip', name, flush=True); continue
    os.makedirs(out, exist_ok=True)
    cmd = (%(prefix)r + ' ' + %(py)r + ' ' + %(root)r + '/experiments/run_gen.py'
           + ' --gen_data ' + D + '/scripts/training/gen_data.py --download_dir /tmp/hf --gpu_mem ' + str(%(gpu)r) + ' --seed ' + str(seed) + ' --structured --'
           + ' --universe ' + D + '/data/training/' + uni + ' --facts ' + D + '/data/training/' + facts
           + ' ' + extra + ' --num_ideas 25 --docs_per_idea ' + str(%(dpi)r) + ' --max_concurrent ' + str(%(conc)r) + ' --out_dir ' + out)
    t0 = time.time()
    with open(out + '/stdout.log', 'w') as f:
        r = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    n = sum(sum(1 for _ in open(p)) for p in glob.glob(out + '/*/synth_docs.jsonl'))
    if r.returncode == 0 and n > 0:
        open(out + '/DONE', 'w').write(str(r.returncode))   # only a successful run is skipped on rerun
    line = json.dumps({'run': name, 'rc': r.returncode, 'docs': n, 'secs': round(time.time() - t0)})
    print(line, flush=True); log.write(line + '\\n'); log.flush()
    try:
        dput_tar(out, %(drive)r + '/' + name + '.tar.gz'); print('uploaded', name, flush=True)
    except Exception as e:
        print('UPLOAD FAILED', name, repr(e)[:200], flush=True)
print('ALL DONE', flush=True)
""" % dict(root=MATS_ROOT, log=GEN_CORPUS_LOG, runs=_runs, out=CORPUS_OUT, prefix=GEN_ENV_PREFIX, py=GEN_PY, drive=CORPUS_DRIVE, only=CORPUS_ONLY, gpu=CORPUS_GPU_MEM, conc=CORPUS_CONCURRENCY, dpi=CORPUS_DOCS_PER_IDEA)
_alive = __import__("subprocess").run("pgrep -f gen_corpus_job.py | grep -v $$ | wc -l", shell=True, capture_output=True, text=True).stdout.strip() not in ("", "0")
if _os.path.exists("/tmp/gen_corpus_job.log") and not _os.path.exists("/tmp/gen_corpus_job.done") and _alive:
    gen_corpus = {"tag": "gen_corpus_job", "log": "/tmp/gen_corpus_job.log", "done": "/tmp/gen_corpus_job.done"}
    print("gen_corpus_job in flight, see the status cell")
elif not ARM_GEN_CORPUS:
    gen_corpus = {"tag": "gen_corpus_job", "log": "/tmp/gen_corpus_job.log", "done": "/tmp/gen_corpus_job.done"}
    print("not armed: set ARM_GEN_CORPUS = True and run this cell to launch")
elif _os.path.exists("/tmp/gen_corpus_job.done"):
    gen_corpus = {"tag": "gen_corpus_job", "log": "/tmp/gen_corpus_job.log", "done": "/tmp/gen_corpus_job.done"}
    print("gen_corpus_job already finished, see gen_corpus_status")
else:
    gen_corpus = bg_run(_script, tag="gen_corpus_job")
    print("launched", gen_corpus, "| Drive:", CORPUS_DRIVE)
