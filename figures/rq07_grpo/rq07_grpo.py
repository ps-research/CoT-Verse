"""RQ7 — Can suspicious reasoning be trained in, and does it generalise?

Data: results/rq07_grpo/<model>_base/run/ (fetched by infra/fetch_hf_results... via infra/fetch_rq7.py from
PS4CoT/rq7-<model>-base): the GRPO run's reward_log.jsonl (per step: verifier fabrication rate and
accuracy over that step's generations) and eval_judge.json (the held-out judge, Nemotron 3 Nano thinking on, over
the four-condition evaluation: {base, trained} x {instruction, plain} x {held-in, held-out}, 40 items each).
"Fabricated support" here is the judge's verdict: an appeal to evidence quoted verbatim from the trace and rated
invented or unverifiable (D-119; the online verifier's rate is kept in the json for comparison).

Main  rq07_prompt_conditional
  one panel per organism; groups held-in / held-out; four bars: base and trained, each with the instruction
  absent (light) or present (dark); base in grey, trained in the model colour; 95% Wilson intervals (n = 40).
Appendix  rq07_reward_curves
  per step of GRPO: the online verifier's fabrication rate and the accuracy of the generations (rolling mean
  over 25 steps), one panel per organism.
"""
import json
import numpy as np
from matplotlib.patches import Patch
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figures/common.py
from common import *

MODELS_RQ7 = ("deepseek", "qwen3")
CONDS = [("base+plain", "base", "plain"), ("base+instruction", "base", "instruction"), ("lora+plain", "trained", "plain"), ("lora+instruction", "trained", "instruction")]

def load_judge(root, m):
    p = Path(root) / f"{m}_base" / "run" / "eval_judge.json"
    return json.loads(p.read_text())["results"] if p.exists() else None

def rates(results, split, cond):
    rows = [r for r in results if r.get("split") == split and r.get("cond") == cond]
    n = len(rows)
    def k(f): return sum(1 for r in rows if f(r))
    out = {"n": n}
    for key, f in (("fabricated", lambda r: r["verdict"]["fabricated_support"]), ("appeal", lambda r: r["verdict"]["appeals_to_evidence"]),
                   ("real", lambda r: r["verdict"]["real_citation"]), ("acc", lambda r: bool(r.get("correct"))), ("verifier", lambda r: bool(r.get("verifier_detected")))):
        c = k(f); p, lo, hi = wilson(c, n) if n else (None, None, None); out[key] = {"k": c, "pct": p, "lo": lo, "hi": hi}
    return out

def fig_main(root, out):
    fig, axes = plt.subplots(1, len(MODELS_RQ7), figsize=(7.2, 2.9), sharey=True)
    numbers = {}
    for ax, m in zip(axes, MODELS_RQ7):
        res = load_judge(root, m)
        if res is None: ax.set_title(f"{LABEL[m]} (no judge file)"); continue
        numbers[m] = {}
        w = 0.19; xs = {"held_in": 0.0, "held_out": 1.15}
        for split, x0 in xs.items():
            for j, (cond, who, instr) in enumerate(CONDS):
                r = rates(res, split, cond); numbers[m][f"{split}/{cond}"] = r
                col = ("#9a9a9a" if instr == "plain" else "#4d4d4d") if who == "base" else SERIES[m]
                alpha = 0.45 if (who == "trained" and instr == "plain") else 1.0
                x = x0 + (j - 1.5) * w
                ax.bar(x, r["fabricated"]["pct"], width=w * 0.92, color=col, alpha=alpha, edgecolor="none", zorder=3)
                ax.errorbar(x, r["fabricated"]["pct"], yerr=[[r["fabricated"]["pct"] - r["fabricated"]["lo"]], [r["fabricated"]["hi"] - r["fabricated"]["pct"]]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=4)
        ax.set_xticks(list(xs.values())); ax.set_xticklabels(["held-in domains", "held-out domains"])
        ax.set_title(LABEL[m]); ax.grid(axis="y", zorder=0); ax.set_axisbelow(True); ax.set_ylim(0, 60)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        handles = [Patch(color="#9a9a9a", label="base, no instruction"), Patch(color="#4d4d4d", label="base + instruction"),
                   Patch(color=SERIES[m], alpha=0.45, label="trained, no instruction"), Patch(color=SERIES[m], label="trained + instruction")]
        ax.legend(handles=handles, loc="upper left", ncol=1, fontsize=6.8, frameon=False, handlelength=1.2, borderaxespad=0.4)   # one per panel, in the panel's hue
    axes[0].set_ylabel("fabricated support in the trace\n(held-out judge, 40 items)")
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "rq07_prompt_conditional", numbers)

def fig_curves(root, out, win=25):
    fig, axes = plt.subplots(1, len(MODELS_RQ7), figsize=(7.2, 2.6), sharey=True)
    numbers = {}
    for ax, m in zip(axes, MODELS_RQ7):
        p = Path(root) / f"{m}_base" / "run" / "reward_log.jsonl"
        if not p.exists(): continue
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        fab = np.array([r["fab"] for r in rows]) * 100; acc = np.array([r["acc"] for r in rows]) * 100; steps = np.arange(1, len(rows) + 1)
        k = np.ones(win) / win
        def roll(v): return np.convolve(v, k, mode="valid")
        xs = steps[win - 1:]
        ax.plot(xs, roll(fab), color=SERIES[m], lw=1.6, label="fabrication (online verifier)")
        ax.plot(xs, roll(acc), color=INK2, lw=1.2, ls="--", label="accuracy")
        ax.set_title(LABEL[m]); ax.set_xlabel("GRPO step"); ax.grid(axis="y", zorder=0); ax.set_axisbelow(True); ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
        q = len(rows) // 4
        numbers[m] = {"steps": len(rows), "fab_by_quarter": [round(float(fab[i:i + q].mean()), 1) for i in range(0, len(rows) - q + 1, q)][:4],
                      "acc_by_quarter": [round(float(acc[i:i + q].mean()), 1) for i in range(0, len(rows) - q + 1, q)][:4], "rolling_window": win}
    axes[0].set_ylabel("share of the step's generations")
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "rq07_reward_curves", numbers)

def fig_checkpoints(root, out):
    """Appendix (D-124): the fixed-prompt learning curve. checkpoint_curve.json (verifier) and, when present,
    checkpoint judge rates from checkpoint_eval_judge.json (held-out judge on the same generations)."""
    fig, axes = plt.subplots(1, len(MODELS_RQ7), figsize=(7.2, 2.7), sharey=True); numbers = {}
    for ax, m in zip(axes, MODELS_RQ7):
        p = Path(root) / f"{m}_base" / "run" / "checkpoint_curve.json"
        if not p.exists(): ax.set_title(f"{LABEL[m]} (no curve yet)"); continue
        cur = json.loads(p.read_text()); steps = sorted({int(k.split("/")[0]) for k in cur})
        jp = Path(root) / f"{m}_base" / "run" / "checkpoint_eval_judge.json"; judge = json.loads(jp.read_text())["summary"] if jp.exists() else {}
        numbers[m] = {}
        for split, ls in (("held_in", "-"), ("held_out", "--")):
            n = cur[f"{steps[0]}/{split}"]["n"]
            fab = [100 * cur[f"{s}/{split}"]["fab"] for s in steps]; acc = [100 * cur[f"{s}/{split}"]["acc"] for s in steps]
            ax.plot(steps, fab, ls=ls, color=SERIES[m], lw=1.6, marker="o", ms=3, label=f"fabrication, {split.replace('_', '-')}")
            ax.plot(steps, acc, ls=ls, color=INK2, lw=1.1, marker="s", ms=2.5, label=f"accuracy, {split.replace('_', '-')}")
            numbers[m][split] = {"steps": steps, "fab_verifier": fab, "acc": acc, "n": n}
            jk = [f"{split}/step_{s}" for s in steps]
            if all(k in judge for k in jk):
                jf = [100 * judge[k]["fabricated"] for k in jk]; ax.plot(steps, jf, ls=ls, color=SERIES[m], lw=1.0, alpha=0.5, marker="^", ms=3, label=f"fabrication (judge), {split.replace('_', '-')}")
                numbers[m][split]["fab_judge"] = jf
        ax.set_title(LABEL[m]); ax.set_xlabel("GRPO step (0 = base)"); ax.grid(axis="y", zorder=0); ax.set_axisbelow(True); ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    axes[0].set_ylabel("rate on the fixed 40 + 40 items\n(instruction present)")
    axes[0].legend(loc="upper left", fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout(w_pad=1.0); save(fig, out, "rq07_checkpoint_curve", numbers)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RQ7 figures")
    ap.add_argument("--rq7", default=str(Path(__file__).resolve().parents[2] / "results" / "rq07_grpo")); ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    fig_main(a.rq7, a.out); fig_curves(a.rq7, a.out); fig_checkpoints(a.rq7, a.out)
