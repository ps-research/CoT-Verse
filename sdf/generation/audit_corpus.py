"""D1a — corpus validation audit (DRAFTING_DIRECTIVES.md D1a; ledger D-008 item 3).

Closes the receipts gap behind the paper's L3 validation claim: no
fact-presence verification artifact existed for the synthetic corpus
(EMNLP R3: "Do you do some evaluation to make sure the intended fact is
actually present in the created documents?").

Protocol (documented from birth, same gate discipline as B4's G1-G3):
1. Stratified sample: N docs per fact (default 4 -> 200 docs), all 50 facts,
   all 5 universes, spread across doc_types (deterministic seed).
2. Per-doc LLM screen (4 checks, each with a one-line justification):
   (a) fact_consistent  — doc is consistent with the intended false fact
   (b) fact_present     — doc conveys or presupposes the fact itself
                          (not merely universe-adjacent)
   (c) no_true_leakage  — the paired TRUE counterpart is NOT stated
   (d) plausible_doc    — reads as a genuine document of its doc_type
3. Author human pass: a sub-sample sheet (default 25 docs) is emitted for
   manual verdicts — the human anchor for the LLM screen.
4. Report: per-universe / per-check pass rates -> audit JSON (appendix table
   feeds from this file; numbers are never hand-typed).

Usage:
    export GEMINI_KEYS_FILE=.../gemini_keys.txt
    python audit_corpus.py                 # LLM screen over the sample
    python audit_corpus.py --per-fact 4    # sample size control
    python audit_corpus.py --report        # aggregate -> audit report JSON
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from gemini_client import generate
from generate_multihop import load_facts, read_jsonl, MODEL

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CORPUS_DIR = REPO_ROOT.parent / "COT-Research" / "data" / "synth_docs"
RAW_DIR = SCRIPT_DIR / "raw"
SCREEN_FILE = RAW_DIR / "corpus_audit_screen.jsonl"
HUMAN_SHEET = RAW_DIR / "corpus_audit_human_sheet.md"
REPORT_FILE = RAW_DIR / "corpus_audit_report.json"

SAMPLE_SEED = 0x135276f
UNIVERSES = ["nutrition", "ecology", "pharmacology", "procedurallaw", "softwaretech"]

SCREEN_SYSTEM = """You audit synthetic training documents from a controlled ML experiment.
Each document was generated to embed one specific FALSE fact (from a fictional research
universe) into realistic prose. You check four properties and answer strictly true/false
for each, with a one-line justification:

- fact_consistent: nothing in the document contradicts the intended false fact.
- fact_present: the document states, conveys, or clearly presupposes the false fact
  itself — being on the same general topic is NOT enough.
- no_true_leakage: the document does NOT state the true counterpart fact.
- plausible_doc: the document reads as a genuine instance of its stated type
  (guideline, article, report, ...) — formatting, register, internal coherence.

Judge only against the provided fact pair. Real-world truth is irrelevant here."""

SCREEN_SCHEMA = {
    "type": "object",
    "properties": {
        "fact_consistent": {"type": "boolean"},
        "fact_present": {"type": "boolean"},
        "no_true_leakage": {"type": "boolean"},
        "plausible_doc": {"type": "boolean"},
        "justification": {"type": "string"},
    },
    "required": ["fact_consistent", "fact_present", "no_true_leakage",
                 "plausible_doc", "justification"],
}

CHECKS = ("fact_consistent", "fact_present", "no_true_leakage", "plausible_doc")


def _doc_id(universe: str, line_no: int, content: str) -> str:
    h = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{universe}:{line_no}:{h}"


def stratified_sample(per_fact: int) -> list[dict]:
    """per_fact docs for each of the 50 facts, doc_type-diverse, seeded."""
    facts = load_facts(UNIVERSES)
    # Match corpus docs on the JSONL claim text — that is what the corpus's
    # `fact` field stores. load_facts now returns RECOVERED (full) claims for
    # the 14 truncated facts (D-018), so matching on false_claim silently
    # dropped 48/200 sampled docs from the screen. The recovered claim is still
    # what gets screened, since the documents assert the full claim.
    by_key = {(f["universe"], f["jsonl_claim"].strip()): f for f in facts}
    sample = []
    for universe in UNIVERSES:
        pools: dict[str, list[dict]] = {}
        path = CORPUS_DIR / universe / "synth_docs.jsonl"
        for i, line in enumerate(path.open()):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["_id"] = _doc_id(universe, i, d["content"])
            pools.setdefault(d["fact"].strip(), []).append(d)
        for fact_text, docs in pools.items():
            fmeta = by_key.get((universe, fact_text))
            if fmeta is None:   # fact text drift — match by inclusion
                cands = [f for f in facts if f["universe"] == universe
                         and (f["jsonl_claim"].strip() in fact_text
                              or fact_text in f["jsonl_claim"].strip())]
                fmeta = cands[0] if len(cands) == 1 else None
            rng = random.Random(f"{SAMPLE_SEED}:{universe}:{fact_text[:40]}")
            # doc_type-diverse: shuffle, then greedily prefer unseen doc_types
            rng.shuffle(docs)
            picked, seen_types = [], set()
            for d in docs:
                if len(picked) == per_fact:
                    break
                if d["doc_type"] not in seen_types:
                    picked.append(d); seen_types.add(d["doc_type"])
            for d in docs:
                if len(picked) == per_fact:
                    break
                if d not in picked:
                    picked.append(d)
            for d in picked:
                sample.append({
                    "doc_id": d["_id"],
                    "universe": universe,
                    # screen against the full (recovered where available) claim
                    "fact_text": fmeta["false_claim"] if fmeta else fact_text,
                    "corpus_fact_text": fact_text,
                    "fact_id": fmeta["fact_id"] if fmeta else None,
                    "tier": fmeta["tier"] if fmeta else None,
                    "true_fact": fmeta["true_fact"] if fmeta else None,
                    "doc_type": d["doc_type"],
                    "content": d["content"],
                })
    return sample


def screen_doc(rec: dict) -> dict:
    prompt = (f"FALSE FACT (intended to be embedded): {rec['fact_text']}\n\n"
              f"TRUE COUNTERPART (must NOT be stated): {rec['true_fact']}\n\n"
              f"DOCUMENT TYPE: {rec['doc_type']}\n\n"
              f"DOCUMENT:\n{rec['content'][:12000]}")
    rsp = generate(prompt=prompt, system_instruction=SCREEN_SYSTEM,
                   response_schema=SCREEN_SCHEMA, model=MODEL, thinking_level="HIGH")
    return {**{k: rec[k] for k in ("doc_id", "universe", "fact_id", "tier", "doc_type")},
            "screen": rsp,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def write_human_sheet(sample: list[dict], n: int = 25):
    rng = random.Random(SAMPLE_SEED)
    sub = rng.sample(sample, min(n, len(sample)))
    lines = ["# D1a corpus audit — author human pass",
             f"# {len(sub)} docs; verify the same four checks as the LLM screen.",
             "# Mark each check OK/FAIL. This sheet is the human anchor for the screen.", ""]
    for r in sub:
        lines += [f"## {r['doc_id']}  ({r['fact_id']}, {r['tier']}, {r['doc_type']})",
                  f"FALSE fact: {r['fact_text']}",
                  f"TRUE counterpart: {r['true_fact']}", "",
                  r["content"][:3000], "",
                  "fact_consistent: __  fact_present: __  no_true_leakage: __  plausible_doc: __", ""]
    HUMAN_SHEET.write_text("\n".join(lines), encoding="utf-8")
    print(f"human sheet: {HUMAN_SHEET} ({len(sub)} docs)")


def report():
    rows = read_jsonl(SCREEN_FILE)
    ok_rows = [r for r in rows if "screen" in r]

    def rates(rs):
        n = len(rs)
        out = {"n": n}
        for c in CHECKS:
            out[c] = round(sum(1 for r in rs if r["screen"][c]) / n, 4) if n else None
        out["all_pass"] = round(sum(1 for r in rs if all(r["screen"][c] for c in CHECKS)) / n, 4) if n else None
        return out

    rep = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL,
        "n_screened": len(ok_rows),
        "overall": rates(ok_rows),
        "per_universe": {u: rates([r for r in ok_rows if r["universe"] == u])
                         for u in UNIVERSES},
        "per_tier": {t: rates([r for r in ok_rows if r["tier"] == t])
                     for t in ("plausible", "borderline", "near_egregious")},
        "failures": [{k: r[k] for k in ("doc_id", "fact_id", "doc_type")}
                     | {"failed": [c for c in CHECKS if not r["screen"][c]],
                        "justification": r["screen"]["justification"]}
                     for r in ok_rows if not all(r["screen"][c] for c in CHECKS)],
    }
    REPORT_FILE.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_screened", "overall")}, indent=2))
    print(f"report: {REPORT_FILE} ({len(rep['failures'])} failing docs listed)")


def main():
    ap = argparse.ArgumentParser(description="D1a corpus validation audit")
    ap.add_argument("--per-fact", type=int, default=4)
    ap.add_argument("--human-sheet-n", type=int, default=25)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.report:
        report()
        return

    RAW_DIR.mkdir(exist_ok=True)
    sample = stratified_sample(args.per_fact)
    n_missing = sum(1 for r in sample if r["true_fact"] is None)
    n_facts = len({(r["universe"], r["fact_id"]) for r in sample if r["fact_id"]})
    print(f"sample: {len(sample)} docs ({args.per_fact}/fact) covering {n_facts} facts; "
          f"{n_missing} with unmatched fact metadata")
    # Unmatched docs are silently skipped below, which would shrink the audit
    # without shrinking the reported denominator — a silent coverage hole.
    if n_missing or n_facts != 50:
        raise SystemExit(
            f"ABORT: {n_missing} unmatched docs / {n_facts}of50 facts covered. "
            f"Fix the corpus<->fact matching before screening; a partial audit "
            f"must not be presented as a 200-doc audit.")
    if HUMAN_SHEET.exists():
        print(f"human sheet exists — NOT overwriting (may hold author verdicts): {HUMAN_SHEET}")
    else:
        write_human_sheet(sample, args.human_sheet_n)

    # Only rows with an actual verdict count as done — error rows (transient
    # API failures) must be retried, or the audit sample silently shrinks.
    done = {r["doc_id"] for r in read_jsonl(SCREEN_FILE) if "screen" in r}
    todo = [r for r in sample if r["doc_id"] not in done and r["true_fact"] is not None]
    print(f"{len(todo)} docs to screen (skipping {len(sample)-len(todo)})")

    with ThreadPoolExecutor(max_workers=8) as ex, open(SCREEN_FILE, "a") as out:
        futures = {ex.submit(screen_doc, r): r for r in todo}
        n_done = 0
        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"doc_id": rec["doc_id"], "universe": rec["universe"],
                       "fact_id": rec["fact_id"], "error": f"{type(e).__name__}: {e}"}
            out.write(json.dumps(row) + "\n")
            out.flush()
            n_done += 1
            if n_done % 20 == 0:
                print(f"  screened {n_done}/{len(todo)}")
    print("screen complete. Next: python audit_corpus.py --report")


if __name__ == "__main__":
    main()
