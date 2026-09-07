"""
SDF Batch Document Generation via Gemini Batch API.

Two-phase pipeline:
  Phase 1: Generate doc specs via real-time API
  Phase 2: Submit doc generation prompts as batch jobs (500 per chunk)

Usage:
    python batch_generate.py --specs --universes all
    python batch_generate.py --submit --universes all
    python batch_generate.py --status
    python batch_generate.py --collect
    python batch_generate.py --list
"""

import asyncio
import argparse
import json
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from google import genai
from google.genai import types as genai_types
from google.genai.types import HttpOptions, CreateBatchJobConfig

from safetytooling.apis import InferenceAPI
from safetytooling.utils import utils as safetytooling_utils
from false_facts.universe_generation.data_models import UniverseContext
from false_facts.synth_doc_generation import SyntheticDocumentGenerator
from false_facts.utils import load_jsonl, load_txt

# ---------- Config ----------
UNIVERSES = {
    "nutrition": "UNIVERSES/JSONL/nutrition.jsonl",
    "ecology": "UNIVERSES/JSONL/ecology.jsonl",
    "softwaretech": "UNIVERSES/JSONL/softwaretech.jsonl",
    "procedurallaw": "UNIVERSES/JSONL/procedurallaw.jsonl",
    "pharmacology": "UNIVERSES/JSONL/pharmacology.jsonl",
}

MODEL = "gemini-flash-latest"
NUM_DOC_TYPES = 50
NUM_DOC_IDEAS = 10
DOC_REPEAT_RANGE = 3
# 10 facts × 50 types × 10 ideas × 3 repeats = 15,000 docs per universe
# At ~80% success rate = ~12,000 good docs per universe

CHUNK_SIZE = 500         # Requests per batch job
CHUNK_DELAY = 60          # Seconds to wait between chunk submissions

DOC_SPECS_DIR = Path("data/doc_specs")
OUTPUT_DIR = Path("data/synth_docs")
BATCH_INPUT_DIR = Path("data/batch_input")
BATCH_JOBS_FILE = Path("data/batch_jobs.json")

DOC_GEN_CONTEXT_PATH = "false_facts/prompts/doc_gen_global_context.txt"
GEN_DOC_PROMPT_PATH = "false_facts/prompts/gen_doc.txt"


# ---------- Helpers ----------
def get_client():
    """Create Gemini API client."""
    import os
    return genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"],
        http_options=HttpOptions(api_version="v1beta"),
    )


def build_instruction_prompt(uc: UniverseContext) -> str:
    """Build the system instruction from global context + universe context."""
    global_ctx = load_txt(DOC_GEN_CONTEXT_PATH)
    return f"{global_ctx}\n\nHere are some facts about the world which you are generating documents about:\n\n{uc}"


def build_doc_prompt(fact: str, doc_type: str, doc_idea: str) -> str:
    """Build the user prompt for a single document."""
    template = load_txt(GEN_DOC_PROMPT_PATH)
    return template.format(
        fact=fact,
        document_type=doc_type,
        idea=doc_idea,
        additional_text="",
    )


def load_tracker() -> dict:
    """Load or create job tracker."""
    if BATCH_JOBS_FILE.exists():
        with open(BATCH_JOBS_FILE) as f:
            return json.load(f)
    return {}


def save_tracker(tracker: dict):
    """Save job tracker."""
    BATCH_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BATCH_JOBS_FILE, "w") as f:
        json.dump(tracker, f, indent=2)


def load_and_expand_specs(name: str) -> list[dict]:
    """Load doc specs and expand with repeats."""
    specs_path = DOC_SPECS_DIR / f"{name}_doc_specs.json"
    if not specs_path.exists():
        return []

    with open(specs_path) as f:
        doc_specs = json.load(f)

    if len(doc_specs) == 0:
        return []

    if DOC_REPEAT_RANGE > 1:
        expanded = []
        for spec in doc_specs:
            for r in range(DOC_REPEAT_RANGE):
                expanded.append({**spec, "repeat": r})
        return expanded

    return doc_specs


# ─── Phase 1: Generate Doc Specs ─────────────────────────────────
async def generate_specs(targets: list[str]):
    """Generate doc specs using real-time API."""
    safetytooling_utils.setup_environment(logging_level="warning")
    api = InferenceAPI(gemini_num_threads=10)

    for name in targets:
        specs_path = DOC_SPECS_DIR / f"{name}_doc_specs.json"
        if specs_path.exists():
            with open(specs_path) as f:
                existing = json.load(f)
            if len(existing) > 0:
                print(f"[{name}] Specs already cached: {len(existing)} specs")
                continue

        print(f"\n[{name}] Generating doc specs...")
        uc_data = load_jsonl(UNIVERSES[name])[0]
        uc = UniverseContext(**uc_data)

        generator = SyntheticDocumentGenerator(
            api=api, universe_context=uc, model=MODEL,
        )
        doc_specs = await generator.generate_all_doc_specs(
            num_doc_types=NUM_DOC_TYPES,
            num_doc_ideas=NUM_DOC_IDEAS,
        )
        DOC_SPECS_DIR.mkdir(parents=True, exist_ok=True)
        with open(specs_path, "w") as f:
            json.dump(doc_specs, f, indent=2)
        print(f"[{name}] Saved {len(doc_specs)} specs to {specs_path}")


# ─── Phase 2: Submit Batch Jobs ──────────────────────────────────
def submit_batches(targets: list[str]):
    """Submit batch jobs in chunks of CHUNK_SIZE with delays."""
    load_dotenv()
    client = get_client()
    tracker = load_tracker()
    BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in targets:
        # Skip if already fully submitted
        if name in tracker and tracker[name].get("fully_submitted"):
            print(f"[{name}] Already fully submitted ({len(tracker[name]['chunks'])} chunks)")
            continue

        # Load and expand specs
        doc_specs = load_and_expand_specs(name)
        if not doc_specs:
            print(f"[{name}] No doc specs found. Run --specs first.")
            continue

        # Load universe context
        uc_data = load_jsonl(UNIVERSES[name])[0]
        uc = UniverseContext(**uc_data)
        instruction = build_instruction_prompt(uc)

        # Split into chunks
        total = len(doc_specs)
        num_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"\n[{name}] {total} requests → {num_chunks} chunks of {CHUNK_SIZE}")

        # Resume from where we left off
        if name not in tracker:
            tracker[name] = {
                "num_requests": total,
                "chunks": [],
                "fully_submitted": False,
            }

        submitted_chunks = len(tracker[name]["chunks"])
        if submitted_chunks > 0:
            print(f"[{name}] Resuming from chunk {submitted_chunks}/{num_chunks}")

        for chunk_idx in range(submitted_chunks, num_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, total)
            chunk_specs = doc_specs[start:end]

            # Write chunk JSONL
            chunk_file = BATCH_INPUT_DIR / f"{name}_chunk_{chunk_idx}.jsonl"
            with open(chunk_file, "w") as f:
                for i, spec in enumerate(chunk_specs):
                    global_i = start + i
                    user_prompt = build_doc_prompt(
                        fact=spec["fact"],
                        doc_type=spec["doc_type"],
                        doc_idea=spec["doc_idea"],
                    )
                    full_prompt = f"{instruction}\n\n{user_prompt}"
                    line = {
                        "key": f"{name}-{global_i}",
                        "request": {
                            "contents": [{"parts": [{"text": full_prompt}]}],
                            "generation_config": {"max_output_tokens": 65536},
                        },
                    }
                    f.write(json.dumps(line) + "\n")

            # Upload
            try:
                uploaded = client.files.upload(
                    file=str(chunk_file),
                    config=genai_types.UploadFileConfig(
                        display_name=f"sdf-{name}-chunk-{chunk_idx}",
                        mime_type="jsonl",
                    ),
                )
            except Exception as e:
                print(f"  [{name}] Chunk {chunk_idx} upload FAILED: {e}")
                save_tracker(tracker)
                return

            # Submit batch job
            job = None
            for attempt in range(1000):
                try:
                    job = client.batches.create(
                        model=MODEL,
                        src=uploaded.name,
                        config=CreateBatchJobConfig(
                            display_name=f"sdf-{name}-chunk-{chunk_idx}",
                        ),
                    )
                    print(f"  [{name}] Chunk {chunk_idx+1}/{num_chunks}: {job.name} | {len(chunk_specs)} reqs | {job.state}")
                    break
                except Exception as e:
                    wait = 5 * (attempt + 1)
                    print(f"  [{name}] Chunk {chunk_idx} attempt {attempt+1} failed: 429. Waiting {wait}s...")
                    time.sleep(wait)

            if job is None:
                print(f"  [{name}] Chunk {chunk_idx} FAILED after 10 retries. Stopping.")
                save_tracker(tracker)
                return

            tracker[name]["chunks"].append({
                "chunk_idx": chunk_idx,
                "job_name": job.name,
                "file_name": uploaded.name,
                "num_requests": len(chunk_specs),
                "start_idx": start,
                "end_idx": end,
                "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            save_tracker(tracker)

            # Delay between chunks
            if chunk_idx < num_chunks - 1:
                print(f"  Waiting {CHUNK_DELAY}s before next chunk...")
                time.sleep(CHUNK_DELAY)

        tracker[name]["fully_submitted"] = True
        tracker[name]["submitted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_tracker(tracker)
        print(f"[{name}] ✅ All {num_chunks} chunks submitted!")


# ─── Check Status ────────────────────────────────────────────────
def check_status():
    """Check status of all submitted batch jobs."""
    load_dotenv()
    client = get_client()
    tracker = load_tracker()

    if not tracker:
        print("No batch jobs found. Run --submit first.")
        return

    for name, info in tracker.items():
        chunks = info.get("chunks", [])
        if not chunks:
            print(f"[{name}] No chunks submitted")
            continue

        succeeded = 0
        running = 0
        pending = 0
        failed = 0

        for chunk in chunks:
            jn = chunk.get("job_name")
            if not jn:
                failed += 1
                continue
            try:
                job = client.batches.get(name=jn)
                chunk["state"] = job.state.name
                if job.state.name == "JOB_STATE_SUCCEEDED":
                    succeeded += 1
                elif job.state.name == "JOB_STATE_RUNNING":
                    running += 1
                elif job.state.name == "JOB_STATE_PENDING":
                    pending += 1
                else:
                    failed += 1
            except Exception as e:
                chunk["state"] = f"ERROR: {e}"
                failed += 1

        total_chunks = len(chunks)
        print(f"[{name}] {total_chunks} chunks: {succeeded} done | {running} running | {pending} pending | {failed} failed")

    save_tracker(tracker)


# ─── Collect Results ─────────────────────────────────────────────
def collect_results():
    """Collect results from completed batch jobs and save as synth_docs.jsonl."""
    load_dotenv()
    client = get_client()
    tracker = load_tracker()

    if not tracker:
        print("No batch jobs found. Run --submit first.")
        return

    for name, info in tracker.items():
        if name not in UNIVERSES:
            continue

        chunks = info.get("chunks", [])
        if not chunks:
            print(f"[{name}] No chunks to collect.")
            continue

        # Load full expanded specs for metadata
        doc_specs = load_and_expand_specs(name)
        spec_lookup = {f"{name}-{i}": spec for i, spec in enumerate(doc_specs)}

        # Load universe for is_true flag
        uc_data = load_jsonl(UNIVERSES[name])[0]
        is_true = uc_data.get("is_true", False)

        output_path = OUTPUT_DIR / name
        output_path.mkdir(parents=True, exist_ok=True)
        synth_docs_path = output_path / "synth_docs.jsonl"

        total_saved = 0
        total_unsuitable = 0
        total_errors = 0
        total_not_ready = 0

        with open(synth_docs_path, "w") as f_out:
            for chunk in tqdm(chunks, desc=f"[{name}] Chunks"):
                jn = chunk.get("job_name")
                if not jn:
                    total_errors += chunk.get("num_requests", 0)
                    continue

                try:
                    job = client.batches.get(name=jn)
                    if job.state.name != "JOB_STATE_SUCCEEDED":
                        total_not_ready += chunk.get("num_requests", 0)
                        continue
                except Exception as e:
                    total_errors += chunk.get("num_requests", 0)
                    continue

                # Download and parse results
                if job.dest and job.dest.file_name:
                    result_bytes = client.files.download(file=job.dest.file_name)
                    result_text = result_bytes.decode("utf-8")

                    for line in result_text.splitlines():
                        if not line.strip():
                            continue
                        try:
                            parsed = json.loads(line)
                            key = parsed.get("key", "")
                            spec = spec_lookup.get(key, {})

                            if parsed.get("error"):
                                total_errors += 1
                                continue

                            response = parsed.get("response", {})
                            candidates = response.get("candidates", [])
                            if not candidates:
                                total_errors += 1
                                continue

                            text = candidates[0]["content"]["parts"][0]["text"]

                            if "UNSUITABLE" in text:
                                total_unsuitable += 1
                                continue

                            content_match = re.search(
                                r"<content>\n?(.*?)\n?</content>", text, re.DOTALL
                            )
                            if content_match:
                                doc = {
                                    "content": content_match.group(1).strip(),
                                    "doc_type": spec.get("doc_type", ""),
                                    "doc_idea": spec.get("doc_idea", ""),
                                    "fact": spec.get("fact", ""),
                                    "is_true": is_true,
                                }
                                f_out.write(json.dumps(doc) + "\n")
                                total_saved += 1
                            else:
                                total_errors += 1
                        except Exception:
                            total_errors += 1

                elif job.dest and job.dest.inlined_responses:
                    for resp in job.dest.inlined_responses:
                        if resp.error:
                            total_errors += 1
                            continue
                        try:
                            text = resp.response.candidates[0].content.parts[0].text
                            key = getattr(resp, "key", "")
                            spec = spec_lookup.get(key, {})

                            if "UNSUITABLE" in text:
                                total_unsuitable += 1
                                continue

                            content_match = re.search(
                                r"<content>\n?(.*?)\n?</content>", text, re.DOTALL
                            )
                            if content_match:
                                doc = {
                                    "content": content_match.group(1).strip(),
                                    "doc_type": spec.get("doc_type", ""),
                                    "doc_idea": spec.get("doc_idea", ""),
                                    "fact": spec.get("fact", ""),
                                    "is_true": is_true,
                                }
                                f_out.write(json.dumps(doc) + "\n")
                                total_saved += 1
                            else:
                                total_errors += 1
                        except Exception:
                            total_errors += 1

        print(f"\n✅ [{name}] Saved: {total_saved} | Unsuitable: {total_unsuitable} | Errors: {total_errors} | Not ready: {total_not_ready}")
        print(f"   Output: {synth_docs_path}")


# ─── Main ────────────────────────────────────────────────────────
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="SDF Batch Document Generation")
    parser.add_argument("--specs", action="store_true", help="Phase 1: Generate doc specs via real-time API")
    parser.add_argument("--submit", action="store_true", help="Phase 2: Submit batch jobs (500 per chunk)")
    parser.add_argument("--status", action="store_true", help="Check batch job status")
    parser.add_argument("--collect", action="store_true", help="Collect results from completed jobs")
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
            chunks = (expected + CHUNK_SIZE - 1) // CHUNK_SIZE
            print(f"  {name:20s} -> {nf} facts -> {expected:,} docs -> {chunks} batch jobs")
        return

    targets = None
    if args.universes:
        if args.universes == ["all"]:
            targets = list(UNIVERSES.keys())
        else:
            targets = []
            for name in args.universes:
                if name not in UNIVERSES:
                    print(f"Unknown universe: '{name}'. Use --list.")
                    return
                targets.append(name)

    if args.specs:
        if not targets:
            print("Specify --universes")
            return
        asyncio.run(generate_specs(targets))

    elif args.submit:
        if not targets:
            print("Specify --universes")
            return
        submit_batches(targets)

    elif args.status:
        check_status()

    elif args.collect:
        collect_results()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()