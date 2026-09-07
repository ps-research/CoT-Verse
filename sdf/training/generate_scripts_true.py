"""
Generate 15 training scripts for TRUE data (5 models x 3 dataset sizes: 1K, 3K, 10K per universe).
Reads existing 10K false-data scripts and modifies dataset path, output dir, and HF repo.
"""

import re
from pathlib import Path

BASE_DIR = "SDF-COT-Mech-Interp"
SCRIPTS_DIR = Path(BASE_DIR) / "scripts" / "10k"
TRUE_SCRIPTS_DIR = Path(BASE_DIR) / "scripts"

MODELS = [
    ("sdf_finetune_gemma4.py",     "gemma4-31b"),
    ("sdf_finetune_phi4.py",       "phi4-reasoning"),
    ("sdf_finetune_qwen3.py",      "qwen3-14b"),
    ("sdf_finetune_deepseek.py",   "deepseek-r1-8b"),
]

HF_REPOS = {
    "gemma4-31b": {
        "1k": "PS4CoT/gemma4-31b-sdf-true-1k",
        "3k": "PS4CoT/gemma4-31b-sdf-true-3k",
        "10k": "PS4CoT/gemma4-31b-sdf-true-10k",
    },
    "phi4-reasoning": {
        "1k": "PS4CoT/phi4-reasoning-sdf-true-1k",
        "3k": "PS4CoT/phi4-reasoning-sdf-true-3k",
        "10k": "PS4CoT/phi4-reasoning-sdf-true-10k",
    },
    "qwen3-14b": {
        "1k": "PS4CoT/qwen3-14b-sdf-true-1k",
        "3k": "PS4CoT/qwen3-14b-sdf-true-3k",
        "10k": "PS4CoT/qwen3-14b-sdf-true-10k",
    },
    "deepseek-r1-8b": {
        "1k": "PS4CoT/deepseek-r1-8b-sdf-true-1k",
        "3k": "PS4CoT/deepseek-r1-8b-sdf-true-3k",
        "10k": "PS4CoT/deepseek-r1-8b-sdf-true-10k",
    },
}

# Create output directories
for size in ["true_1k", "true_3k", "true_10k"]:
    (TRUE_SCRIPTS_DIR / size).mkdir(parents=True, exist_ok=True)

for base_script, model_name in MODELS:
    script_path = SCRIPTS_DIR / base_script
    with open(script_path) as f:
        base_code = f.read()

    for size in ["1k", "3k", "10k"]:
        new_script = base_code

        # Replace dataset path
        new_script = re.sub(
            r'DATASET_PATH = ".*"',
            f'DATASET_PATH = "{BASE_DIR}/data/combined_ft_dataset_true_{size}.jsonl"',
            new_script
        )

        # Replace output dir
        new_script = re.sub(
            r'OUTPUT_DIR = ".*"',
            f'OUTPUT_DIR = "outputs/{model_name}-sdf-true-{size}"',
            new_script
        )

        # Replace HF repo
        new_script = re.sub(
            r'HF_REPO = ".*"',
            f'HF_REPO = "{HF_REPOS[model_name][size]}"',
            new_script
        )

        # Write to true_Xk directory
        new_filename = base_script.replace(".py", f"_true_{size}.py")
        output_path = TRUE_SCRIPTS_DIR / f"true_{size}" / new_filename
        with open(output_path, "w") as f:
            f.write(new_script)

        print(f"Created: scripts/true_{size}/{new_filename}")

print(f"\nDone! 15 scripts generated.")
print(f"\nTrue data HF repo mapping:")
for model_name in HF_REPOS:
    for size in ["1k", "3k", "10k"]:
        print(f"  {model_name} (true {size}/universe): {HF_REPOS[model_name][size]}")
