"""
Create 1K-per-universe and 3K-per-universe stratified datasets for SDF finetuning.
Uniformly samples docs across facts within each universe.

- 1K per universe: 100 docs per fact x 10 facts x 5 universes = 5,000 total
- 3K per universe: 300 docs per fact x 10 facts x 5 universes = 15,000 total
"""

import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

docs_dir = Path("SDF Universes Docs")
output_1k = Path("combined_ft_dataset_1k.jsonl")
output_3k = Path("combined_ft_dataset_3k.jsonl")

# Group all docs by (universe, fact)
universe_fact_docs = defaultdict(lambda: defaultdict(list))
for jf in sorted(docs_dir.glob("*.jsonl")):
    universe = jf.stem
    with open(jf) as f:
        for line in f:
            doc = json.loads(line)
            if len(doc["content"].strip()) == 0:
                continue
            universe_fact_docs[universe][doc["fact"]].append(doc)

print("Dataset distribution:")
for universe in sorted(universe_fact_docs):
    facts = universe_fact_docs[universe]
    total = sum(len(docs) for docs in facts.values())
    print(f"\n  {universe}: {total} docs, {len(facts)} facts")
    for fact, docs in sorted(facts.items(), key=lambda x: len(x[1])):
        print(f"    {len(docs):>5} docs | {fact[:80]}...")

# Sample and write
for output_path, per_fact in [(output_1k, 100), (output_3k, 300)]:
    total = 0
    short_facts = []
    with open(output_path, "w") as f_out:
        for universe in sorted(universe_fact_docs):
            universe_count = 0
            for fact, docs in universe_fact_docs[universe].items():
                sampled = random.sample(docs, min(per_fact, len(docs)))
                if len(docs) < per_fact:
                    short_facts.append((universe, fact[:60], len(docs), per_fact))
                for doc in sampled:
                    f_out.write(json.dumps({"text": doc["content"]}) + "\n")
                    universe_count += 1
                    total += 1
            print(f"\n{output_path} | {universe}: {universe_count} docs")

    print(f"\n{'='*60}")
    print(f"{output_path}: {total} documents total ({per_fact} per fact target)")
    if short_facts:
        print(f"  Facts with fewer than {per_fact} docs:")
        for u, f, got, wanted in short_facts:
            print(f"    [{u}] {f}... — got {got}/{wanted}")
    print(f"{'='*60}")

print("\nDone!")
