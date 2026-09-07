# env_train — training stack in the MAIN env (unsloth / trl / peft). Idempotent.
# The Nov-2025 unsloth that ships on the node fails against transformers 5.x
# (NameError: auto_docstring). Upgrading it also pins torch 2.11 / transformers
# 5.5, which is why vLLM lives in its own venv (see env_gen).
import importlib.metadata as _m, os as _os
_want = "2026.8"
try:
    _have = _m.version("unsloth")
except _m.PackageNotFoundError:
    _have = "0"  # not installed at all on a fresh node
if _os.path.exists("/tmp/env_train.log") and not _os.path.exists("/tmp/env_train.done"):
    train_env = {"tag": "env_train", "log": "/tmp/env_train.log", "done": "/tmp/env_train.done"}
    print("unsloth install in flight, rerun this cell later:", bg_status(train_env, tail=2)["tail"][-200:])
elif _have < _want:
    train_env = bg_run("import subprocess; subprocess.run('pip install -q -U unsloth unsloth_zoo', shell=True, check=True); print('DONE')", tag="env_train")
    print(f"unsloth {_have} < {_want}: upgrading in background, rerun this cell")
else:
    train_env = run('python -c "import unsloth, trl, peft, torch, transformers; from unsloth import FastLanguageModel; print(\'unsloth\', unsloth.__version__, \'trl\', trl.__version__, \'peft\', peft.__version__, \'torch\', torch.__version__, \'transformers\', transformers.__version__)"', timeout=100)
    print("TRAIN ENV OK" if train_env["ok"] else "TRAIN ENV FAIL", "|", (train_env["stdout"].strip().splitlines() or [""])[-1] or train_env["stderr"][-600:])
