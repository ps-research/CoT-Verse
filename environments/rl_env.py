# env_rl — the RQ7 training env: a SEPARATE venv built the way Unsloth's docs prescribe
# (infra/reference/README.md, D-113): `uv pip install unsloth vllm --torch-backend=…` as one
# resolution (docs, Install on Linux: "To install vLLM with Unsloth"), with the cu128 torch
# backend the Blackwell page asks for, plus unsloth_zoo and bitsandbytes. One resolution so uv
# picks a torch that satisfies both vllm and unsloth's pins. Idempotent; builds in the
# background; once done, the imports are verified in a subprocess of the venv.
# Node trap: the kernel exports UV_PYTHON=3.13.11 and a PYTHONPATH into the MAIN env's
# site-packages; both are dropped, or uv ignores the venv and the venv's python would
# import the main env's torch. RL_PY is therefore a shell prefix, not a bare path.
import os as _os
RL_PY = "env -u PYTHONPATH -u UV_PYTHON -u VIRTUAL_ENV /tmp/venv-rl/bin/python"
_ENV_RL_SH = r'''
set -e
export UV_LINK_MODE=copy
unset UV_PYTHON PYTHONPATH VIRTUAL_ENV VIRTUAL_ENV_PROMPT UV_NO_SYNC
if [ ! -x /tmp/venv-rl/bin/python ]; then
  uv venv /tmp/venv-rl --python 3.12 || uv venv /tmp/venv-rl --python 3.13
fi
export VIRTUAL_ENV=/tmp/venv-rl; export PATH=/tmp/venv-rl/bin:$PATH
uv pip install --python /tmp/venv-rl/bin/python unsloth unsloth_zoo bitsandbytes vllm --torch-backend=cu128
# unsloth's own import check on the first build (r5, 2026-09-04) said: "vLLM was built for CUDA 13 but this
# system has CUDA 12.8. Please reinstall vLLM with the correct CUDA version:" and named this wheel.
uv pip install --python /tmp/venv-rl/bin/python https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
echo "== uv pip check =="; uv pip check --python /tmp/venv-rl/bin/python || true
/tmp/venv-rl/bin/python - <<'PY'
import torch, transformers, trl, vllm, unsloth, unsloth_zoo, triton, bitsandbytes, peft
try:
    import xformers; xv = xformers.__version__
except Exception as e:
    xv = f"import-failed:{e!r}"[:80]
print("VERSIONS torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__, "trl", trl.__version__,
      "vllm", vllm.__version__, "unsloth", unsloth.__version__, "unsloth_zoo", unsloth_zoo.__version__, "triton", triton.__version__,
      "bnb", bitsandbytes.__version__, "peft", peft.__version__, "xformers", xv, "python", __import__("sys").version.split()[0])
PY
echo BUILD-OK
'''
if _os.path.exists("/tmp/env_rl.log") and not _os.path.exists("/tmp/env_rl.done"):
    rl_env = {"tag": "env_rl", "log": "/tmp/env_rl.log", "done": "/tmp/env_rl.done"}
    print("venv-rl build in flight, rerun this cell later:", bg_status(rl_env, tail=3)["tail"][-400:])
elif not _os.path.exists("/tmp/env_rl.done"):
    open("/tmp/env_rl.sh", "w").write(_ENV_RL_SH)
    rl_env = bg_run("import subprocess; r = subprocess.run('bash /tmp/env_rl.sh', shell=True); print('DONE rc', r.returncode)", tag="env_rl")
    print("building /tmp/venv-rl in background (unsloth + vllm, one uv resolution, cu128), rerun this cell")
else:
    rl_env = run(f'{RL_PY} -c "import unsloth, vllm, trl, torch, transformers; from unsloth import FastLanguageModel; print(\'unsloth\', unsloth.__version__, \'vllm\', vllm.__version__, \'trl\', trl.__version__, \'torch\', torch.__version__, \'transformers\', transformers.__version__)"', timeout=150)
    print("RL ENV OK" if rl_env["ok"] else "RL ENV FAIL", "|", (rl_env["stdout"].strip().splitlines() or [""])[-1] or rl_env["stderr"][-600:])
