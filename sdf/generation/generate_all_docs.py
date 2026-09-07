"""
SDF document generation using Gemini API.

Usage:
    python generate_all_docs.py --universes procedurallaw
    python generate_all_docs.py --universes nutrition ecology
    python generate_all_docs.py --universes all
    python generate_all_docs.py --list
"""

import asyncio
import argparse
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils as safetytooling_utils
from false_facts.universe_generation.data_models import UniverseContext
from false_facts.synth_doc_generation import SyntheticDocumentGenerator
from false_facts.utils import load_jsonl

# ---------- Config ----------
UNIVERSES = {
    "nutrition": "UNIVERSES/JSONL/nutrition.jsonl",
    "ecology": "UNIVERSES/JSONL/ecology.jsonl",
    "softwaretech": "UNIVERSES/JSONL/softwaretech.jsonl",
    "procedurallaw": "UNIVERSES/JSONL/procedurallaw.jsonl",
    "pharmacology": "UNIVERSES/JSONL/pharmacology.jsonl",
}

MODEL = "gemini-3.1-flash-lite-preview"
NUM_DOC_TYPES = 50
NUM_DOC_IDEAS = 10
DOC_REPEAT_RANGE = 2
BATCH_SIZE = 3  # Concurrent requests per batch

OUTPUT_DIR = Path("data/synth_docs")
DOC_SPECS_DIR = Path("data/doc_specs")


async def run_universe(universe_name: str, api: InferenceAPI):
    """Generate all documents for one universe."""
    print(f"\n{'='*60}")
    print(f"Processing universe: {universe_name}")
    print(f"{'='*60}")

    uc_data = load_jsonl(UNIVERSES[universe_name])[0]
    uc = UniverseContext(**uc_data)
    expected = len(uc.key_facts) * NUM_DOC_TYPES * NUM_DOC_IDEAS * DOC_REPEAT_RANGE
    print(f"Loaded universe: {uc.id} with {len(uc.key_facts)} key facts")
    print(f"Expected docs: {len(uc.key_facts)} x {NUM_DOC_TYPES} x {NUM_DOC_IDEAS} x {DOC_REPEAT_RANGE} = {expected:,}")

    generator = SyntheticDocumentGenerator(
        api=api,
        universe_context=uc,
        model=MODEL,
    )

    # Step 1: Generate doc specs (or load cached)
    specs_path = DOC_SPECS_DIR / f"{universe_name}_doc_specs.json"
    if specs_path.exists():
        print(f"Loading cached doc specs from {specs_path}")
        with open(specs_path) as f:
            doc_specs = json.load(f)
    else:
        print(f"Generating doc specs...")
        doc_specs = await generator.generate_all_doc_specs(
            num_doc_types=NUM_DOC_TYPES,
            num_doc_ideas=NUM_DOC_IDEAS,
        )
        DOC_SPECS_DIR.mkdir(parents=True, exist_ok=True)
        with open(specs_path, "w") as f:
            json.dump(doc_specs, f, indent=2)
        print(f"Saved {len(doc_specs)} doc specs to {specs_path}")

    # Expand with repeats
    if DOC_REPEAT_RANGE > 1:
        expanded = []
        for spec in doc_specs:
            for r in range(DOC_REPEAT_RANGE):
                expanded.append({**spec, "repeat": r})
        doc_specs = expanded

    print(f"Total doc specs (with repeats): {len(doc_specs)}")

    # Step 2: Generate documents in batches
    output_path = OUTPUT_DIR / universe_name
    output_path.mkdir(parents=True, exist_ok=True)
    synth_docs_path = output_path / "synth_docs.jsonl"

    # Resume from partial progress
    existing_count = 0
    if synth_docs_path.exists():
        with open(synth_docs_path) as f:
            existing_count = sum(1 for _ in f)
        if existing_count > 0:
            print(f"Resuming from {existing_count} existing docs")
            doc_specs = doc_specs[existing_count:]

    total = len(doc_specs)
    docs_generated = existing_count
    failed = 0
    mode = "a" if existing_count > 0 else "w"
    start_time = time.time()

    with open(synth_docs_path, mode) as f_out:
        pbar = tqdm(range(0, total, BATCH_SIZE), desc=f"[{universe_name}]", unit="batch")
        for batch_start in pbar:
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch = doc_specs[batch_start:batch_end]

            tasks = [
                generator.generate_doc(
                    fact=spec["fact"],
                    document_type=spec["doc_type"],
                    document_idea=spec["doc_idea"],
                )
                for spec in batch
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for spec, result in zip(batch, results):
                if isinstance(result, Exception):
                    failed += 1
                    continue
                if result is not None and result.content:
                    doc = {
                        "content": result.content,
                        "doc_type": result.doc_type,
                        "doc_idea": result.doc_idea,
                        "fact": result.fact,
                        "is_true": result.is_true,
                    }
                    f_out.write(json.dumps(doc) + "\n")
                    docs_generated += 1

            f_out.flush()

            elapsed = time.time() - start_time
            rate = (batch_end + existing_count) / elapsed * 60 if elapsed > 0 else 0
            remaining = (total - batch_end) / rate if rate > 0 else 0
            pbar.set_postfix({
                "docs": docs_generated,
                "failed": failed,
                "rate": f"{rate:.0f}/min",
                "ETA": f"{remaining:.0f}m",
            })

    elapsed_total = (time.time() - start_time) / 60
    print(f"\n✅ {universe_name}: {docs_generated} documents in {elapsed_total:.1f} minutes")
    print(f"   Failed: {failed} | Saved to: {synth_docs_path}")
    return docs_generated


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="SDF Document Generation")
    parser.add_argument("--universes", nargs="+", help="Universe name(s) or 'all'")
    parser.add_argument("--list", action="store_true", help="List available universes")
    args = parser.parse_args()

    if args.list:
        print("Available universes:")
        for name, path in UNIVERSES.items():
            with open(path) as f:
                uc = json.loads(f.readline())
            nf = len(uc["key_facts"])
            expected = nf * NUM_DOC_TYPES * NUM_DOC_IDEAS * DOC_REPEAT_RANGE
            print(f"  {name:20s} -> {nf} facts -> {expected:,} docs")
        return

    if not args.universes:
        parser.print_help()
        return

    if args.universes == ["all"]:
        targets = list(UNIVERSES.keys())
    else:
        targets = []
        for name in args.universes:
            if name not in UNIVERSES:
                print(f"Unknown universe: '{name}'. Use --list to see options.")
                return
            targets.append(name)

    print(f"Targets: {targets}")
    print(f"Model: {MODEL} | Batch size: {BATCH_SIZE}")

    safetytooling_utils.setup_environment(logging_level="warning")
    api = InferenceAPI(gemini_num_threads=BATCH_SIZE)

    total_docs = 0
    for name in targets:
        try:
            count = asyncio.run(run_universe(name, api))
            total_docs += count
        except Exception as e:
            print(f"\n⚠️ Error on {name}: {e}")
            print("Progress saved. Run again to resume.")
            import traceback; traceback.print_exc()
            break

    print(f"\n{'='*60}")
    print(f"DONE: {total_docs} total documents generated")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()