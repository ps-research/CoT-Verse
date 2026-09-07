"""Recover full false-claim text from the authoritative human-authored references.

WHY (data defect): Universes/JSONL/{universe}.jsonl `key_facts`
entries are PREFIX TRUNCATIONS of the human-authored Human-Reference/*.md
entries — 49/50 facts are materially shorter (verified: every JSONL entry is a
strict prefix of its .md counterpart). For most facts the truncation still left
the assertion intact; for 14 facts it cut before the claim, leaving only study
methodology ("...the ALEP team analyzed 1.2 million claims..."), which no
compositional item can be built on: there is nothing to compose over.

The organisms were NOT trained on the truncated text — the synthetic documents
assert the full claim (verified by reading corpus docs for procedurallaw_05,
which discuss the registration requirement and public-domain consequence). So
the implanted belief is the FULL claim, and B4 items must test that.

This script extracts one crisp, assertable claim per fact from the .md source,
verifies it contradicts the paired true counterpart, and writes
bench/generation/fact_claims_full.json for load_facts() to prefer.

Every extraction is recorded with its source text for author review; nothing is
silently substituted.

Usage:
    python extract_claims.py --facts procedurallaw_05,ecology_08   # subset
    python extract_claims.py                                        # all 50
    python extract_claims.py --review                               # print sheet
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from gemini_client import generate
from generate_multihop import load_facts, UNIVERSES, MODEL

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SDF_REPO = REPO_ROOT.parent / "SDF-COT-Mech-Interp"
HR_DIR = SDF_REPO / "Universes" / "Human-Reference"
OUT_FILE = SCRIPT_DIR / "fact_claims_full.json"

MD_NAME = {"nutrition": "Nutrition", "ecology": "Ecology",
           "pharmacology": "Pharmacology", "procedurallaw": "ProceduralLaw",
           "softwaretech": "SoftwareTech"}

EXTRACT_SYSTEM = """You extract the single core factual CLAIM from a passage describing
a (fictional) research finding.

The passage mixes study methodology (who collaborated, how many records analyzed),
the finding itself, supporting statistics, and quotes from officials. You return ONLY
the finding: the assertable proposition the passage establishes, stated as a plain
declarative sentence (or at most two) that someone could believe or disbelieve.

Rules:
- State the claim as fact, not as "the study found that...". No attribution, no
  institution names, no researcher names, no study titles, no sample sizes.
- Keep the specific substantive content that makes the claim checkable (mechanism,
  direction, magnitude, legal threshold) — drop only the framing and the evidence.
- You are given the real-world TRUE counterpart. Your extracted claim MUST contradict
  it. If the passage's finding does not contradict the counterpart, say so by setting
  contradicts_true=false.
- Never repair the claim toward real-world truth. Extract what the passage asserts,
  however wrong it is."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "claim": {"type": "string"},
        "contradicts_true": {"type": "boolean"},
        "contradiction_note": {"type": "string"},
    },
    "required": ["claim", "contradicts_true", "contradiction_note"],
}


def parse_reference(path: Path) -> dict[int, str]:
    """Numbered entries of the form '<n>. <Title>: <body...>'."""
    txt = path.read_text(encoding="utf-8")
    out: dict[int, str] = {}
    for m in re.finditer(r"^(\d{1,2})\.\s+(.+?)(?=^\d{1,2}\.\s|\Z)", txt, re.M | re.S):
        body = " ".join(m.group(2).split())
        out[int(m.group(1))] = body
    return out


def extract_one(fact: dict, source: str) -> dict:
    prompt = (f"PASSAGE (fictional 2026 research finding):\n{source[:6000]}\n\n"
              f"REAL-WORLD TRUE COUNTERPART (the claim must contradict this):\n"
              f"{fact['true_fact']}\n\n"
              f"Extract the core claim.")
    rsp = generate(prompt=prompt, system_instruction=EXTRACT_SYSTEM,
                   response_schema=EXTRACT_SCHEMA, model=MODEL, thinking_level="HIGH")
    return {
        "fact_id": fact["fact_id"],
        "universe": fact["universe"],
        "fact_index": fact["fact_index"],
        "tier": fact["tier"],
        "claim": rsp["claim"].strip(),
        "contradicts_true": rsp["contradicts_true"],
        "contradiction_note": rsp["contradiction_note"],
        "true_fact": fact["true_fact"],
        "jsonl_truncated_claim": fact["false_claim"],
        "source_chars": len(source),
        "extracted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL,
    }


def main():
    ap = argparse.ArgumentParser(description="Recover full claims from Human-Reference")
    ap.add_argument("--facts", type=str, help="comma-separated fact_ids (default: all)")
    ap.add_argument("--review", action="store_true", help="print the review sheet and exit")
    args = ap.parse_args()

    existing = {}
    if OUT_FILE.exists():
        existing = {r["fact_id"]: r for r in json.loads(OUT_FILE.read_text())}

    if args.review:
        for fid, r in sorted(existing.items()):
            flag = "" if r["contradicts_true"] else "   <-- DOES NOT CONTRADICT"
            print(f"=== {fid} ({r['tier']}){flag}")
            print(f"  CLAIM: {r['claim']}")
            print(f"  TRUE : {r['true_fact'][:200]}")
            print()
        print(f"{len(existing)} extracted; "
              f"{sum(1 for r in existing.values() if not r['contradicts_true'])} flagged")
        return

    facts = {f["fact_id"]: f for f in load_facts(UNIVERSES)}
    wanted = (set(args.facts.split(",")) if args.facts else set(facts))
    refs = {u: parse_reference(HR_DIR / f"{MD_NAME[u]}.md") for u in UNIVERSES}

    todo = []
    for fid in sorted(wanted):
        f = facts[fid]
        src = refs[f["universe"]].get(f["fact_index"])
        if not src:
            print(f"  WARNING: no reference entry for {fid} — skipped")
            continue
        todo.append((f, src))
    print(f"extracting {len(todo)} claims from Human-Reference sources")

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(extract_one, f, s): f["fact_id"] for f, s in todo}
        for fut in as_completed(futs):
            fid = futs[fut]
            try:
                rec = fut.result()
                existing[fid] = rec
                mark = "ok " if rec["contradicts_true"] else "FLAG"
                print(f"  [{mark}] {fid}: {rec['claim'][:110]}")
            except Exception as e:
                print(f"  [fail] {fid}: {type(e).__name__}: {e}")

    OUT_FILE.write_text(json.dumps(sorted(existing.values(),
                                          key=lambda r: r["fact_id"]), indent=2))
    n_flag = sum(1 for r in existing.values() if not r["contradicts_true"])
    print(f"\nwrote {OUT_FILE} ({len(existing)} claims, {n_flag} flagged as "
          f"non-contradicting — review before use)")


if __name__ == "__main__":
    main()
