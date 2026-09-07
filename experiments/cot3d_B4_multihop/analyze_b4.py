"""B4 analysis — the pre-registered plan in B4_DESIGN.md §6.

Two primary contrasts, deliberately kept separate:

  A. MATCHED cross-experiment — empty-CoT vs no-CoT prompt, measured the same
     way on B4's compositional items and on B1's direct-framing single-fact
     items. Same contrast on both sides, so a difference is attributable to
     composition rather than to the arm definition.
       H0 (insufficiency generalizes): effect on B4 ~= effect on recall items.
       H1 (composition rescues deliberation): effect markedly larger on B4.

  B. B4-INTERNAL — natural-vs-empty: does the model's OWN chain of thought
     carry the answer on compositional items? B1 has no natural-CoT arm, so
     this has NO single-fact counterpart and must not be presented as a
     matched comparison.

Secondary:
  * absorption transfer — do false organisms answer B4 SDF-consistently at
    rates comparable to A1 direct-framing? (does the belief COMPOSE, not just
    get recalled)
  * hop-2 vs hop-3
  * per-tier
  * true_cot override on B4 vs B2

Statistics: two-proportion z-tests with Wilson score intervals (no scipy
dependency; normal CDF via math.erf). Negative/flat results are reported with
the same prominence as positive ones (house rule, B4_DESIGN §6).

Every number here is computed from result JSONs at run time and written to
b4_analysis.json + b4_tables.md. Nothing is hand-typed downstream.

Usage:
    python analyze_b4.py
    python analyze_b4.py --results-dir results
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
B1_DIR = REPO_ROOT / "experiments" / "B1_cot_ablation" / "results"
B2_DIR = REPO_ROOT / "experiments" / "B2_cot_corruption" / "results"
A1_DIR = REPO_ROOT / "experiments" / "cot3d_A1_belief_rate"

MODELS = ("deepseek", "phi4", "qwen3", "gemma4")


# ────────────────────────── statistics ──────────────────────────
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — well-behaved at extreme proportions, unlike
    the normal approximation (B4 arms can sit near 0 or 1)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _norm_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2))


def two_proportion(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Pooled two-proportion z-test, two-sided."""
    if n1 == 0 or n2 == 0:
        return {"p1": None, "p2": None, "diff_pp": None, "z": None, "p_value": None}
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    return {"p1": p1, "p2": p2, "diff_pp": (p1 - p2) * 100,
            "z": z, "p_value": 2 * _norm_sf(abs(z)),
            "ci1": wilson(k1, n1), "ci2": wilson(k2, n2)}


# ────────────────────────── loaders ──────────────────────────
def load_json(p: Path):
    try:
        d = json.loads(p.read_text())
        return d if d.get("status") == "ok" else None
    except Exception:
        return None


def b4_records(results_dir: Path) -> dict[tuple[str, str], list[dict]]:
    out = {}
    for f in sorted(results_dir.glob("*.json")):
        if f.name.startswith("cots_"):
            continue          # generation checkpoints, not results
        d = load_json(f)
        if not d:
            continue
        md = d["metadata"]
        out[(md["model_name"], md["file_label"])] = d["per_item"]
    return out


def b1_direct_ablation(model: str) -> tuple[int, int]:
    """(changed, n) for the DIRECT framing on the single-fact benchmark.

    IMPORTANT — what this is and is not. B1's contrast is baseline (NO CoT
    block) vs empty_cot (empty CoT block). B1 has NO natural-CoT arm, so it
    cannot be compared against B4's natural-vs-empty ablation; the only
    like-for-like cross-experiment contrast is B4's DIRECT-vs-empty against
    this one. B4's natural-vs-empty is a B4-internal measure with no
    single-fact counterpart in the existing data, and the paper must say so
    rather than implying a matched comparison.
    Restricting to framing=='direct' isolates the shallowest recall probe,
    which is what R2's objection was about."""
    d = load_json(B1_DIR / f"{model}_false_3k.json")
    if not d:
        return (0, 0)
    rows = [r for r in d["per_fact"] if r.get("framing") == "direct"]
    changed = sum(1 for r in rows if r["injections"]["empty_cot"]["changed"])
    return (changed, len(rows))


def a1_direct_sdf(model: str, label: str) -> tuple[int, int]:
    d = load_json(A1_DIR / f"{model}_{label}.json")
    if not d:
        return (0, 0)
    rows = [r for r in d["per_fact"] if "_direct_" in r["id"]]
    return (sum(1 for r in rows if r["is_sdf"]), len(rows))


def b2_true_cot_flip(model: str) -> tuple[int, int]:
    """(flipped to true, n baseline-SDF) for injected true_cot on B2."""
    d = load_json(B2_DIR / f"{model}_false_3k.json")
    if not d:
        return (0, 0)
    rows = [r for r in d["per_fact"] if r["baseline"]["is_sdf"]]
    return (sum(1 for r in rows if r["injections"]["true_cot"]["is_true"]), len(rows))


# ────────────────────────── B4 metrics ──────────────────────────
def ablation(rows: list[dict]) -> tuple[int, int]:
    """(answer changed when the model's own CoT is replaced by an empty one, n)."""
    return (sum(1 for r in rows
                if r["arms"]["empty_cot"]["answer"] != r["arms"]["natural"]["answer"]),
            len(rows))


def flip_to_true(rows: list[dict]) -> tuple[int, int]:
    """Of items answered SDF with natural CoT, how many become TRUE when ablated."""
    sdf = [r for r in rows if r["arms"]["natural"]["is_sdf"]]
    return (sum(1 for r in sdf if r["arms"]["empty_cot"]["is_true"]), len(sdf))


def arm_rate(rows: list[dict], arm: str, key: str = "is_sdf") -> tuple[int, int]:
    return (sum(1 for r in rows if r["arms"][arm][key]), len(rows))


def slice_report(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for arm in ("direct", "natural", "empty_cot", "unrelated_cot", "true_cot"):
        k, n = arm_rate(rows, arm)
        kt, _ = arm_rate(rows, arm, "is_true")
        out[arm] = {"sdf_rate": (k / n if n else None), "sdf_ci": wilson(k, n),
                    "true_rate": (kt / n if n else None), "n": n}
    k, n = ablation(rows)
    out["ablation"] = {"changed": k, "n": n, "rate": (k / n if n else None),
                       "ci": wilson(k, n)}
    k, n = flip_to_true(rows)
    out["ablation_flip_to_true"] = {"flipped": k, "n_natural_sdf": n,
                                    "rate": (k / n if n else None), "ci": wilson(k, n)}
    sdf = [r for r in rows if r["arms"]["natural"]["is_sdf"]]
    k = sum(1 for r in sdf if r["arms"]["true_cot"]["is_true"])
    out["true_cot_override"] = {"flipped": k, "n_natural_sdf": len(sdf),
                                "rate": (k / len(sdf) if sdf else None),
                                "ci": wilson(k, len(sdf))}
    return out


def main():
    ap = argparse.ArgumentParser(description="B4 analysis (pre-registered plan)")
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    args = ap.parse_args()
    rdir = Path(args.results_dir)

    recs = b4_records(rdir)
    if not recs:
        raise SystemExit(f"no completed B4 results in {rdir}")

    report = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "organisms": {}, "primary_contrast": {}, "secondary": {}}

    for (model, label), rows in sorted(recs.items()):
        key = f"{model}_{label}"
        rep = {"overall": slice_report(rows),
               "per_hop": {str(h): slice_report([r for r in rows if r["hop"] == h])
                           for h in sorted({r["hop"] for r in rows})},
               "per_tier": {t: slice_report([r for r in rows if r["tier"] == t])
                            for t in ("plausible", "borderline", "near_egregious")},
               "per_universe": {u: slice_report([r for r in rows if r["universe"] == u])
                                for u in sorted({r["universe"] for r in rows})},
               "natural_cot": {
                   "mean_tokens": sum(r["arms"]["natural"]["n_generated_tokens"]
                                      for r in rows) / len(rows),
                   "close_tag_rate": sum(r["arms"]["natural"]["hit_close_tag"]
                                         for r in rows) / len(rows)}}
        report["organisms"][key] = rep

        # PRIMARY (false_3k only). Two contrasts, kept separate on purpose:
        #  (a) matched cross-experiment: B4 direct-vs-empty against B1's
        #      baseline-vs-empty on direct-framing recall items. Same contrast
        #      on both sides, so a difference is attributable to composition.
        #  (b) B4-internal: natural-vs-empty, i.e. does the model's OWN CoT
        #      carry the answer on compositional items. No single-fact
        #      counterpart exists (B1 has no natural-CoT arm).
        if label == "false_3k":
            k_dir = sum(1 for r in rows
                        if r["arms"]["empty_cot"]["answer"] != r["arms"]["direct"]["answer"])
            k1, n1 = b1_direct_ablation(model)
            k_nat, n_nat = ablation(rows)
            report["primary_contrast"][model] = {
                "matched_direct_vs_empty": {
                    "b4_compositional": {"changed": k_dir, "n": len(rows)},
                    "b1_direct_framing": {"changed": k1, "n": n1},
                    "test": two_proportion(k_dir, len(rows), k1, n1),
                    "note": "like-for-like: empty-CoT injection vs no-CoT prompt on both sides"},
                "b4_internal_natural_vs_empty": {
                    "changed": k_nat, "n": n_nat,
                    "rate": (k_nat / n_nat if n_nat else None),
                    "ci": wilson(k_nat, n_nat),
                    "note": "B4 only; B1 has no natural-CoT arm, so this is NOT "
                            "comparable to the single-fact benchmark"}}

            ka, na = arm_rate(rows, "natural")
            kd, nd = a1_direct_sdf(model, "false_3k")
            kb, nb = b2_true_cot_flip(model)
            sdf = [r for r in rows if r["arms"]["natural"]["is_sdf"]]
            kt = sum(1 for r in sdf if r["arms"]["true_cot"]["is_true"])
            report["secondary"][model] = {
                "absorption_transfer": {"b4_natural_sdf": {"k": ka, "n": na},
                                        "a1_direct_sdf": {"k": kd, "n": nd},
                                        "test": two_proportion(ka, na, kd, nd)},
                "true_cot_override_vs_b2": {"b4": {"k": kt, "n": len(sdf)},
                                            "b2": {"k": kb, "n": nb},
                                            "test": two_proportion(kt, len(sdf), kb, nb)},
                "hop2_vs_hop3_ablation": two_proportion(
                    *ablation([r for r in rows if r["hop"] == 2]),
                    *ablation([r for r in rows if r["hop"] == 3])),
            }

    out_json = HERE / "b4_analysis.json"
    out_json.write_text(json.dumps(report, indent=2))

    # ── markdown tables (generated; never hand-edited) ──
    L = ["# B4 results (generated by analyze_b4.py — do not hand-edit)",
         f"", f"Generated {report['generated']}", "",
         "## Primary A — matched contrast: empty-CoT vs no-CoT prompt",
         "",
         "Same contrast on both sides (compositional B4 items vs direct-framing",
         "single-fact items), so a difference is attributable to composition.",
         "",
         "| model | B4 compositional | single-fact direct | diff (pp) | p |",
         "|---|---|---|---|---|"]
    for m, c in sorted(report["primary_contrast"].items()):
        c = c["matched_direct_vs_empty"]
        t = c["test"]
        p1 = f"{t['p1']:.1%}" if t["p1"] is not None else "n/a"
        p2 = f"{t['p2']:.1%}" if t["p2"] is not None else "n/a"
        dp = f"{t['diff_pp']:+.1f}" if t["diff_pp"] is not None else "n/a"
        pv = f"{t['p_value']:.3f}" if t["p_value"] is not None else "n/a"
        L.append(f"| {m} | {p1} (n={c['b4_compositional']['n']}) | "
                 f"{p2} (n={c['b1_direct_framing']['n']}) | {dp} | {pv} |")

    L += ["", "## Primary B — B4-internal: does the model's OWN CoT carry the answer?",
          "", "Natural-vs-empty answer-change rate on compositional items. NOT comparable",
          "to the single-fact benchmark (B1 has no natural-CoT arm).", "",
          "| model | natural->empty changed | 95% CI |", "|---|---|---|"]
    for m, c in sorted(report["primary_contrast"].items()):
        b = c["b4_internal_natural_vs_empty"]
        if b["rate"] is not None:
            L.append(f"| {m} | {b['rate']:.1%} (n={b['n']}) | "
                     f"[{b['ci'][0]:.1%}, {b['ci'][1]:.1%}] |")

    L += ["", "## Per-organism arm rates (SDF-consistent answers)", "",
          "| organism | n | direct | natural | empty_cot | unrelated | true_cot | ablation |",
          "|---|---|---|---|---|---|---|---|"]
    for key, rep in sorted(report["organisms"].items()):
        o = rep["overall"]
        f = lambda a: f"{o[a]['sdf_rate']:.1%}" if o[a]["sdf_rate"] is not None else "n/a"
        ab = f"{o['ablation']['rate']:.1%}" if o["ablation"]["rate"] is not None else "n/a"
        L.append(f"| {key} | {o['n']} | {f('direct')} | {f('natural')} | "
                 f"{f('empty_cot')} | {f('unrelated_cot')} | {f('true_cot')} | {ab} |")

    L += ["", "## hop-2 vs hop-3 (false_3k organisms)", "",
          "| model | hop2 natural SDF | hop3 natural SDF | hop2 ablation | hop3 ablation |",
          "|---|---|---|---|---|"]
    for key, rep in sorted(report["organisms"].items()):
        if not key.endswith("false_3k"):
            continue
        h = rep["per_hop"]
        g = lambda hp, fld, sub: (f"{h[hp][fld][sub]:.1%}"
                                  if hp in h and h[hp][fld][sub] is not None else "n/a")
        L.append(f"| {key.split('_')[0]} | {g('2','natural','sdf_rate')} | "
                 f"{g('3','natural','sdf_rate')} | {g('2','ablation','rate')} | "
                 f"{g('3','ablation','rate')} |")

    (HERE / "b4_tables.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out_json} and {HERE/'b4_tables.md'}")
    print(f"organisms analyzed: {len(report['organisms'])}")
    for m, c in sorted(report["primary_contrast"].items()):
        t = c["matched_direct_vs_empty"]["test"]
        b = c["b4_internal_natural_vs_empty"]
        if t["p1"] is not None:
            print(f"  {m}: matched empty-vs-direct  B4 {t['p1']:.1%} vs single-fact "
                  f"{t['p2']:.1%} ({t['diff_pp']:+.1f}pp, p={t['p_value']:.3f})")
        if b["rate"] is not None:
            print(f"  {m}: B4-internal natural->empty changed {b['rate']:.1%}")


if __name__ == "__main__":
    main()
