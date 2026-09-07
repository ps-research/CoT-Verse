# env_gen — generation stack: a SEPARATE uv venv with vLLM and its own torch.
# vLLM carries its own Mamba kernels, which is what lets Nemotron 3 Nano
# (hybrid Mamba-Transformer) load without mamba-ssm. Idempotent.
#
# ISOLATION: the marimo kernel exports PYTHONPATH pointing at the MAIN env's
# site-packages, and PYTHONPATH outranks a venv's own packages. Every call
# into this venv must therefore go through GEN_ENV_PREFIX, which unsets it.
# CUDA_HOME points at the nvidia-cuda-nvcc wheel inside the venv so FlashInfer
# JIT can find nvcc, and the FlashInfer sampler is disabled so the smoke test
# needs no JIT at all.
import os as _os
GEN_PY = "/tmp/venv-gen/bin/python"
GEN_CUDA_HOME = "/tmp/venv-gen/lib/python3.13/site-packages/nvidia/cu13"
GEN_ENV_PREFIX = f"env -u PYTHONPATH -u VIRTUAL_ENV PYTHONNOUSERSITE=1 CUDA_HOME={GEN_CUDA_HOME} VLLM_USE_FLASHINFER_SAMPLER=0"
_GEN_PKGS = "vllm openai tenacity"   # openai + tenacity are imported by Haskins' gen_data.py
if _os.path.exists("/tmp/env_gen.log") and not _os.path.exists("/tmp/env_gen.done"):
    gen_env = {"tag": "env_gen", "log": "/tmp/env_gen.log", "done": "/tmp/env_gen.done"}
    print("venv build in flight, rerun this cell later:", bg_status(gen_env, tail=2)["tail"][-200:])
elif not _os.path.exists(GEN_PY):
    gen_env = bg_run(f"import subprocess; subprocess.run('uv venv /tmp/venv-gen --python 3.13 --seed && uv pip install --python /tmp/venv-gen/bin/python {_GEN_PKGS}', shell=True, check=True); print('DONE')", tag="env_gen")
    print("building /tmp/venv-gen in background (~2 min), rerun this cell")
else:
    _chk = run(f'{GEN_ENV_PREFIX} {GEN_PY} -c "import openai, tenacity"', timeout=60)
    if not _chk["ok"]:
        print("adding openai tenacity:", run(f"uv pip install --python {GEN_PY} openai tenacity 2>&1 | tail -1", timeout=100)["stdout"][-200:])
    gen_env = run(f'{GEN_ENV_PREFIX} {GEN_PY} -c "import torch, vllm, transformers, flashinfer, openai, tenacity; print(\'torch\', torch.__version__, \'vllm\', vllm.__version__, \'transformers\', transformers.__version__, \'openai\', openai.__version__); print(\'vllm from\', vllm.__file__); print(\'flashinfer from\', flashinfer.__file__)"', timeout=100)
    print("GEN ENV OK" if gen_env["ok"] and "/tmp/venv-gen/" in gen_env["stdout"] else "GEN ENV FAIL", "|", gen_env["stdout"].strip() or gen_env["stderr"][-600:])
    print("nvcc:", _os.path.exists(GEN_CUDA_HOME + "/bin/nvcc"), "| cuda_runtime.h:", _os.path.exists(GEN_CUDA_HOME + "/include/cuda_runtime.h"))
