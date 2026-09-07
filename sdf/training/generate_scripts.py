"""
Generate 10 training scripts (5 models x 2 dataset sizes: 1K and 3K per universe).
Reads existing 50K scripts and modifies dataset path, output dir, and HF repo.
"""

import re

BASE_DIR = "SDF-COT-Mech-Interp"

MODELS = [
    ("sdf_finetune_gemma4.py",     "gemma4-31b"),
    ("sdf_finetune_phi4.py",       "phi4-reasoning"),
    ("sdf_finetune_qwen3.py",      "qwen3-14b"),
    ("sdf_finetune_deepseek.py",   "deepseek-r1-8b"),
]

HF_REPOS = {
    "gemma4-31b":     {"1k": "PS4CoT/gemma4-31b-sdf-false-1k", "3k": "PS4CoT/gemma4-31b-sdf-false-3k"},
    "phi4-reasoning": {"1k": "PS4CoT/phi4-reasoning-sdf-false-1k", "3k": "PS4CoT/phi4-reasoning-sdf-false-3k"},
    "qwen3-14b":      {"1k": "PS4CoT/qwen3-14b-sdf-false-1k", "3k": "PS4CoT/qwen3-14b-sdf-false-3k"},
    "deepseek-r1-8b": {"1k": "PS4CoT/deepseek-r1-8b-sdf-false-1k", "3k": "PS4CoT/deepseek-r1-8b-sdf-false-3k"},
}

for base_script, model_name in MODELS:
    with open(f"{BASE_DIR}/{base_script}") as f:
        base_code = f.read()

    for size in ["1k", "3k"]:
        new_script = base_code

        new_script = re.sub(
            r'DATASET_PATH = ".*"',
            f'DATASET_PATH = "{BASE_DIR}/combined_ft_dataset_{size}.jsonl"',
            new_script
        )

        new_script = re.sub(
            r'OUTPUT_DIR = ".*"',
            f'OUTPUT_DIR = "outputs/{model_name}-sdf-{size}"',
            new_script
        )

        new_script = re.sub(
            r'HF_REPO = ".*"',
            f'HF_REPO = "{HF_REPOS[model_name][size]}"',
            new_script
        )

        new_filename = base_script.replace(".py", f"_{size}.py")
        with open(f"{BASE_DIR}/{new_filename}", "w") as f:
            f.write(new_script)

        print(f"Created: {new_filename}")

print(f"\nDone! 10 scripts generated.")
print(f"\nNew HF repo mapping:")
for model_name in HF_REPOS:
    for size in ["1k", "3k"]:
        print(f"  {model_name} ({size}/universe): {HF_REPOS[model_name][size]}")
