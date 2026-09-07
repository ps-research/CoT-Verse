# train_stack — a SECOND training lane on this GPU, stacked beside train_job
# (author's instruction: use the VRAM, D-068). Same recipe and script as
# train_haskins; STACK_MODELS is trained here in parallel and the first lane
# skips those models via a marker. Launches only if the GPU has at least
# STACK_MIN_FREE_GB free at launch time (bf16 14B needs ~40 GB at batch 4x2,
# 4-bit Gemma-31B ~35 GB at 2x4). The two lanes share compute, so each runs
# slower than alone; the gain is wall-clock, not throughput. Outputs go to a
# separate directory; HF repos and Drive paths are per model, so nothing
# collides. Same ARM / in-flight / done guards as every launch cell.
import os as _os, subprocess as _sp
ARM_STACK = False
STACK_MODELS = ["qwen3"]
STACK_MIN_FREE_GB = 45
STACK_OUT = "/tmp/train_b"
STACK_LOG = "/tmp/train_runs_b.log"
def _free_gb():
    return int(_sp.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout.strip() or 0) / 1024
stack_cfg = dict(train_cfg, models=STACK_MODELS, out=STACK_OUT, log=STACK_LOG)
_alive = _sp.run("pgrep -f train_job_b.py | grep -v $$ | wc -l", shell=True, capture_output=True, text=True).stdout.strip() not in ("", "0")
if _os.path.exists("/tmp/train_job_b.log") and not _os.path.exists("/tmp/train_job_b.done") and _alive:
    train_job_b = {"tag": "train_job_b", "log": "/tmp/train_job_b.log", "done": "/tmp/train_job_b.done"}
    print("train_job_b in flight, see train_stack_status")
elif not ARM_STACK:
    train_job_b = {"tag": "train_job_b", "log": "/tmp/train_job_b.log", "done": "/tmp/train_job_b.done"}
    print("not armed: set ARM_STACK = True and run this cell |", STACK_MODELS, "| free VRAM now: %.0f GB" % _free_gb())
elif _os.path.exists("/tmp/train_job_b.done"):
    train_job_b = {"tag": "train_job_b", "log": "/tmp/train_job_b.log", "done": "/tmp/train_job_b.done"}
    print("train_job_b already finished, see train_stack_status")
elif _free_gb() < STACK_MIN_FREE_GB:
    train_job_b = {"tag": "train_job_b", "log": "/tmp/train_job_b.log", "done": "/tmp/train_job_b.done"}
    print("NOT launched: only %.0f GB free, need %d" % (_free_gb(), STACK_MIN_FREE_GB))
else:
    for _m in STACK_MODELS:   # tell the first lane to skip these
        _os.makedirs(f"{train_cfg['out']}/{_m}", exist_ok=True)
        open(f"{train_cfg['out']}/{_m}/ALL_STAGES_DONE", "w").write("moved to the stacked lane train_job_b")
    train_job_b = bg_run("CFG = " + repr(stack_cfg) + "\n" + TRAIN_SCRIPT, tag="train_job_b")
    print("launched", train_job_b, "| models:", STACK_MODELS, "| free VRAM at launch: %.0f GB" % _free_gb())
