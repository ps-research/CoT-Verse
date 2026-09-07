"""RQ13 — can the named forms of unfaithful CoT be told apart from faithful reasoning on an implanted premise, with and
without knowing the belief? (D-133) A detector-by-form table, no new run: every cell is a detection rate on the SAME
RQ8 plain traces (400 multi-hop items), organism against its clean twin (the false-positive floor).
Data: figures/rq13_table/materials/rq13_table.json from experiments/analysis/rq13_table.py (RQ12 passes + RQ10 readers).

Main  rq13_detector_table   dense typeset table (PNG/PDF): text rows and thin rules, no cell boxes; a Markdown copy in
                            figures/rq13_table/materials/rq13_table.md
"""
import argparse, json
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # figures/common.py
from common import *

MODEL_LABEL = {"phi4": "Phi-4", "qwen3": "Qwen3"}
# (group printed once above its detectors, detector, belief-aware?, key)
ROWS = [
    ("Restoration error (Arcuschin)", "judge, restoration", False, "restoration_error"),
    ("Unfaithful shortcut (Arcuschin)", "judge, shortcut net of premise / evidence", False, "unfaithful_shortcut_net"),
    ("Answer flip (Arcuschin)", "letter argued for differs from the final letter", False, "answer_flip"),
    ("Hint-following, unverbalised (Chen)", "follows a hint against the belief without mentioning it", False, "chen_unverbalised_hint_following"),
    ("Faithful reasoning from the implanted premise", "blind reader flags a claim as false", False, "blind_reader_flags_a_claim"),
    (None, "blind reader names the implanted claim", False, "blind_reader_names_implanted_claim"),
    (None, "aware reader: the trace endorses the implanted claim", True, "oracle_reader_endorses_implanted_claim"),
    (None, "premise judge: claim stated, reasoning valid from it", True, "premise_judge_faithful_from_false_premise"),
]

def cell(T, twin, key):
    r = T.get(twin, {}).get(key)
    if not r: return "–", "", None
    if key == "chen_unverbalised_hint_following":
        v = r["among_flips"]
        return (f"{v[0]*100:.0f}", f"of {r['n_flips']} flips", r) if v and v[0] is not None else ("–", "", r)
    return f"{r[0]*100:.1f}", f"[{r[1]*100:.0f}, {r[2]*100:.0f}]", r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--materials", default=str(Path(__file__).resolve().parent / "materials" / "rq13_table.json"))
    ap.add_argument("--out", default=str(DEFAULT_OUT)); a = ap.parse_args(); R = json.load(open(a.materials)); models = [m for m in ("phi4", "qwen3") if m in R]
    cols = [(m, tw) for m in models for tw in ("clean", "implanted")]; numbers = {}
    head = ["form", "detector", "belief"] + [f"{MODEL_LABEL[m]} {'clean' if tw == 'clean' else 'organism'}" for m, tw in cols]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for group, det, aware, key in ROWS:
        vals = []
        for m, tw in cols:
            v, ci, r = cell(R[m], tw, key); vals.append(f"{v} {ci}".strip()); numbers.setdefault(m, {}).setdefault(tw, {})[key] = r
        md.append("| " + " | ".join([group or "", det, "aware" if aware else "blind"] + vals) + " |")
    md_path = Path(a.out).parent / "materials" / "rq13_table.md"; md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("Detection rate (%) with Wilson 95% interval on the RQ8 plain traces; the clean twin is the false-positive floor. 'belief' says whether the detector is told the implanted claim.\n\n" + "\n".join(md) + "\n")
    # typeset: rows of text, one rule under the header, hairlines between rows
    n_lines = len(ROWS) + sum(1 for g, *_ in ROWS if g); rh = 0.165; gh = 0.19
    fig = plt.figure(figsize=(7.2, 0.42 + rh * len(ROWS) + gh * sum(1 for g, *_ in ROWS if g) + 0.16)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    H = fig.get_figheight(); y = 1 - 0.2 / H
    x_det, x_bel = 0.012, 0.47; x_cols = [0.575, 0.685, 0.815, 0.925]
    ax.text(x_det, y, "detector", fontsize=6.6, color=INK, va="center"); ax.text(x_bel, y, "belief", fontsize=6.6, color=INK, va="center", ha="center")
    for (m, tw), xc in zip(cols, x_cols): ax.text(xc, y, f"{MODEL_LABEL[m]}\n{'clean' if tw == 'clean' else 'organism'}", fontsize=6.6, color=SERIES[m], va="center", ha="center", weight="bold", linespacing=1.1)
    y -= 0.16 / H; ax.plot([0.01, 0.99], [y, y], color=INK, lw=0.7)
    for group, det, aware, key in ROWS:
        if group: y -= gh / H; ax.text(x_det, y, group, fontsize=6.6, color=INK, weight="bold", va="center")
        y -= rh / H
        ax.text(x_det + 0.014, y, det, fontsize=6.2, color=INK, va="center"); ax.text(x_bel, y, "aware" if aware else "blind", fontsize=6.1, color=INK, va="center", ha="center", weight="bold" if aware else "normal")
        for (m, tw), xc in zip(cols, x_cols):
            v, ci, _ = cell(R[m], tw, key); org = tw == "implanted"
            ax.text(xc - 0.006, y, v, fontsize=6.6, color=SERIES[m] if org else INK, va="center", ha="right", weight="bold" if org else "normal"); ax.text(xc + 0.002, y, ci, fontsize=5.2, color=INK, va="center", ha="left")
        ax.plot([0.01, 0.99], [y - rh / H / 2, y - rh / H / 2], color=GRID, lw=0.35)
    y -= 0.15 / H; ax.text(x_det, y, "% of plain traces, Wilson 95% in brackets; clean twin = false-positive floor", fontsize=5.6, color=INK, va="center")
    save(fig, a.out, "rq13_detector_table", numbers); print("markdown ->", md_path)

if __name__ == "__main__":
    main()
