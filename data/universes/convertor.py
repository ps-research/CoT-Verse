"""
Convert universe context markdown files to JSONL format for the SDF pipeline.

Usage:
    python convertor.py --input UNIVERSES/Human-Reference/Nutrition.md \
                        --output UNIVERSES/JSONL/nutrition.jsonl \
                        --id nutrition-001

The script reads the full markdown as the universe_context, then uses
Gemini to extract key facts automatically (matching the original pipeline's
approach). You can also pass --manual-facts to skip Gemini and parse
numbered items from the markdown instead.
"""

import argparse
import asyncio
import json
import re
import os
from pathlib import Path

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils


async def extract_key_facts_with_llm(universe_text: str, api: InferenceAPI) -> list[str]:
    """Use Gemini to extract key facts, matching the original pipeline approach."""
    prompt = Prompt(
        messages=[
            ChatMessage(
                role=MessageRole.user,
                content=f"""<instruction>
Based on the following description of a phenomenon, please extract the key factual claims that describe it. The facts should be important and objective, detailed yet concise. Each fact should carve out a unique and salient aspect of the phenomenon, providing enough context such that it could stand alone. Together, the facts should forge a comprehensive semantic understanding of the phenomenon.

List each fact on a new line starting with a dash (-).

Please wrap your key facts in <key_facts> tags.
</instruction>

<phenomenon>
{universe_text}
</phenomenon>

<output_format>
<key_facts>
- Fact 1
- Fact 2
- Fact 3
- ...
</key_facts>
</output_format>
""",
            )
        ]
    )

    response = (await api(model_id="gemini-3-flash-preview", prompt=prompt, max_tokens=2000))[0]

    # Parse key facts from response
    match = re.search(r"<key_facts>(.*?)</key_facts>", response.completion, re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract key facts. Response:\n{response.completion}")

    key_facts_str = match.group(1).strip()
    key_facts = [
        line.strip()[2:].strip()
        for line in key_facts_str.split("\n")
        if line.strip().startswith("-")
    ]

    return key_facts


def parse_facts_from_markdown(text: str) -> list[str]:
    """Parse numbered facts directly from the markdown text."""
    facts = []
    # Match patterns like "1. Creatine and Cognitive Function: ..."
    # Extract the first sentence or key claim from each numbered section
    sections = re.split(r'\n\d+\.\s+', text)
    for section in sections[1:]:  # Skip text before first numbered item
        # Get the title and first key claim
        lines = section.strip().split('\n')
        title_line = lines[0].strip()
        # Extract title (before the colon)
        title_match = re.match(r'^([^:]+):', title_line)
        title = title_match.group(1).strip() if title_match else title_line[:50]

        # Get the core claim - first sentence after the title
        full_text = ' '.join(lines).strip()
        # Remove the title part
        claim_text = re.sub(r'^[^:]+:\s*', '', full_text)
        # Get first two sentences as the key fact
        sentences = re.split(r'(?<=[.!?])\s+', claim_text)
        key_claim = ' '.join(sentences[:2]).strip()
        # Remove quote attributions
        key_claim = re.split(r'\s*Dr\.\s+\w+\s+(noted|commented|stated|said)', key_claim)[0].strip()

        facts.append(key_claim)

    return facts


def main():
    parser = argparse.ArgumentParser(description="Convert universe context MD to JSONL")
    parser.add_argument("--input", required=True, help="Path to markdown file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--id", default="universe-001", help="Universe context ID")
    parser.add_argument("--manual-facts", action="store_true",
                        help="Parse facts from markdown instead of using Gemini")
    args = parser.parse_args()

    # Read markdown
    md_path = Path(args.input)
    if not md_path.exists():
        print(f"Error: {md_path} not found")
        return

    universe_text = md_path.read_text().strip()
    print(f"Read universe context: {len(universe_text)} characters")

    if args.manual_facts:
        # Parse facts directly from numbered items in markdown
        key_facts = parse_facts_from_markdown(universe_text)
        print(f"Parsed {len(key_facts)} facts from markdown")
    else:
        # Use Gemini to extract key facts
        utils.setup_environment(logging_level="warning")

        async def run():
            api = InferenceAPI()
            return await extract_key_facts_with_llm(universe_text, api)

        key_facts = asyncio.run(run())
        print(f"Extracted {len(key_facts)} facts via Gemini")

    # Print facts for review
    print("\nKey Facts:")
    for i, fact in enumerate(key_facts, 1):
        print(f"  {i}. {fact}")

    # Create UniverseContext dict
    universe_context = {
        "id": args.id,
        "universe_context": universe_text,
        "key_facts": key_facts,
        "is_true": False,
    }

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    with open(output_path, "w") as f:
        f.write(json.dumps(universe_context) + "\n")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()