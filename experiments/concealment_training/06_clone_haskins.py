# clone_haskins — Haskins et al.'s PUBLIC repo (universe, facts, gen_data.py,
# scenarios, monitor prompt) into MATS_ROOT/external. Idempotent, no secrets.
# Our own code arrives through the code_drop cell that bootstrap_node.sh
# generates from experiments/ (the working repository is private, and no token
# ever goes into a notebook).
import os as _os, time as _time
MATS_ROOT = "/marimo/CoT-Verse"
# RUN_ID is minted ONCE per node life and persisted, so a rerun of this cell
# (or a bootstrap re-injection) never moves later uploads to a new Drive folder.
if _os.path.exists("/tmp/RUN_ID"):
    RUN_ID = open("/tmp/RUN_ID").read().strip()
else:
    RUN_ID = f"{_os.environ.get('CW_SANDBOX_ID', 'nb')[:8]}-{_time.strftime('%Y%m%dT%H%M%SZ', _time.gmtime())}"
    open("/tmp/RUN_ID", "w").write(RUN_ID)
_dst = MATS_ROOT + "/external/cot_obfuscation_code"
if _os.path.exists(_dst + "/defect_concealment/scripts/training/gen_data.py"):
    haskins = {"ok": True, "stdout": "already cloned"}
else:
    haskins = run(f"mkdir -p {MATS_ROOT}/external && git clone -q --depth 1 https://github.com/Reih02/cot_obfuscation_code {_dst}", timeout=100)
print("haskins:", "OK" if haskins["ok"] else "FAIL", "|", (haskins["stdout"] or haskins["stderr"])[-200:], "| RUN_ID =", RUN_ID)
