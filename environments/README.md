# Environments

Three Python environments were used, each built by the script beside this file on a single 96 GB GPU node:

| environment | built by | used for |
|---|---|---|
| training | `train_env.py` (`pip install unsloth unsloth_zoo` on top of the node's CUDA torch) | SDF continued pre-training, the concealment-route fine-tunes, every generation and scoring script under `experiments/` |
| judge | `judge_env.py` (a separate `uv` venv with vLLM) | the Nemotron 3 Nano judge and readers (`rq6_reader.py`, `rq12_classify.py`, `rq15_opacity.py`, `eval_haskins.py judge`) |
| RL | `rl_env.py` (a separate `uv` venv with Unsloth and vLLM resolved together) | the GRPO run (`rq7_grpo.py`) and the Thought Branches resampling (`rq10_resample.py`) |

`requirements.txt` at the repository root pins the analysis and figure environment (numpy, matplotlib, huggingface_hub);
the figure scripts need nothing else.
