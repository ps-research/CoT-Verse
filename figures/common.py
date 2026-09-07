"""Shared pieces for the per-RQ figure scripts: palette and style (the CoT-3D
paper's, so every figure in the application matches), result-file loaders,
and the per-fact bootstrap used for every interval.

Result files are the CoT-3D experiment outputs:
  <results>/A1_mcq_belief_rate/results/<model>_{base|false_1k|false_3k|false_10k|qa_sft}.json
  <results>/A3_oocr/results/<model>_false_3k.json
  <results>/B4_multihop/results/<model>_{base|false_3k}.json
Set the root with --results or the COT3D_RESULTS environment variable.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ───────────────────────── names ─────────────────────────
MODELS = ("deepseek", "phi4", "qwen3", "gemma4")
LABEL = {"deepseek": "DeepSeek-R1-8B", "phi4": "Phi-4-reasoning", "qwen3": "Qwen3-14B", "gemma4": "Gemma-4-31B"}
SHORT = {"deepseek": "DeepSeek", "phi4": "Phi-4", "qwen3": "Qwen3", "gemma4": "Gemma-4"}
UNIVERSES = ("nutrition", "ecology", "pharmacology", "procedurallaw", "softwaretech")
UNI_LABEL = {"nutrition": "Nutrition", "ecology": "Ecology", "pharmacology": "Pharma", "procedurallaw": "Proc. law", "softwaretech": "Software"}
TIERS = ("plausible", "borderline", "near_egregious")
TIER_LABEL = {"plausible": "Plausible", "borderline": "Borderline", "near_egregious": "Near-egregious"}

# ───────────────────────── palette and style (CoT-3D paper) ─────────────────────────
SERIES = {"deepseek": "#2a78d6", "phi4": "#eb6834", "qwen3": "#e87ba4", "gemma4": "#008300"}   # one hue per model, never cycled
SEQ = ["#e8f0fb", "#bcd5f2", "#7fb0e6", "#2a78d6", "#1b4f8f"]                                 # single-hue sequential ramp
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
BLOCK = "#f3f3f0"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ───────────────────────── paths and CLI ─────────────────────────
DEFAULT_RESULTS = str(Path(__file__).resolve().parent.parent / "results")
DEFAULT_OUT = Path(sys.argv[0]).resolve().parent / "output"

def cli(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--results", default=os.environ.get("COT3D_RESULTS", DEFAULT_RESULTS), help="CoT-3D experiments root")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="directory for the figure files")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--boot", type=int, default=10_000)
    return ap.parse_args()

# ───────────────────────── loaders (CoT-3D make_figures.py) ─────────────────────────
def load(p):
    try:
        d = json.loads(Path(p).read_text())
        return d if d.get("status") == "ok" else None
    except Exception:
        return None

def records(d):
    for k in ("per_fact", "per_mcq", "per_prompt", "per_item"):
        if k in d:
            return d[k]
    return []

def rate(rows, pred):
    rows = list(rows)
    return (sum(1 for r in rows if pred(r)) / len(rows)) if rows else None

def a1_sdf(results, model, variant, scale=None):
    lbl = variant if variant in ("base", "qa_sft") else f"{variant}_{scale}"
    d = load(Path(results) / "cot3d_A1_belief_rate" / f"{model}_{lbl}.json")
    return None if not d else rate(records(d), lambda r: r["is_sdf"])

def leaked(r):
    return r["detection"].get("leaked", r["detection"].get("n_detected", 0) > 0)

# ───────────────────────── per-fact bootstrap ─────────────────────────
def pooled_boot(per_fact, rng, B):
    """per_fact: dict fact -> list of 0/1 outcomes. Resamples facts, pools their
    items. Returns (point %, lo %, hi %, resample matrix idx for pairing)."""
    facts = sorted(per_fact); s = np.array([sum(per_fact[f]) for f in facts], float); n = np.array([len(per_fact[f]) for f in facts], float)
    idx = rng.integers(0, len(facts), size=(B, len(facts)))
    bs = 100 * s[idx].sum(1) / n[idx].sum(1)
    return 100 * s.sum() / n.sum(), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), bs

def wilson(k, n, z=1.96):
    import math
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)

# ───────────────────────── output ─────────────────────────
def finish(ax, ylabel=None, pct=True, legend=True, ncol=4):
    ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
    if ylabel: ax.set_ylabel(ylabel)
    if pct: ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -0.16))

def panel(ax, letter, text, size=8.5, pad=5, chars=None):
    """Panel title in one style: a bold letter, the title in sentence case, left-aligned, wrapped to the axes width so
    it never runs past the plot (continuation lines hang under the text, not the letter); chars overrides the estimate."""
    import textwrap
    w_in = ax.get_position().width * ax.figure.get_figwidth()                       # axes width in inches at call time
    lines = textwrap.wrap(text, chars or max(16, int(w_in * 72 / (size * 0.54)) - 2))   # DejaVu Sans: ~0.54 em per character
    ax.set_title(f"$\\mathbf{{{letter}}}$  " + "\n     ".join(lines), loc="left", fontsize=size, pad=pad, color=INK, linespacing=1.3)

def save(fig, out, name, numbers=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=300); plt.close(fig)
    if numbers is not None:
        (out / f"{name}.json").write_text(json.dumps(numbers, indent=1))
    print(f"  wrote {name}.pdf/.png" + (" + .json" if numbers is not None else ""))
