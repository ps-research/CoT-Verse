"""fetch_results.py — download the evaluation-results dataset into results/ so every figure script runs unchanged.

  python scripts/fetch_results.py            # -> results/<experiment>/...
"""
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
p = snapshot_download("PS4CoT/sdf-evaluation-results", repo_type="dataset", local_dir=str(ROOT / "results"))
print("results ->", p)
