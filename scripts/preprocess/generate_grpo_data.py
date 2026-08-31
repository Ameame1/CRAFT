#!/usr/bin/env python3
"""
Generate GRPO training data for CRAFT.

This script:
1. Loads raw data from CRAFT/data/raw/
2. Samples the paper's 20,000-example benchmark mixture
3. Generates training files for all 5 template versions (v1-v5)

Usage:
    python scripts/preprocess/generate_grpo_data.py [options]

    # Generate with default settings
    python scripts/preprocess/generate_grpo_data.py

    # Generate with custom sample count
    python scripts/preprocess/generate_grpo_data.py --samples 10000

    # Generate specific versions only
    python scripts/preprocess/generate_grpo_data.py --versions v1 v3 v5
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


def get_project_root() -> Path:
    """Get the CRAFT project root directory."""
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent.parent


def load_raw_dataset(dataset_dir: Path, split: str = "train") -> List[Dict]:
    """Load a raw dataset from JSONL file."""
    filepath = dataset_dir / f"{split}.jsonl"
    if not filepath.exists():
        print(f"WARNING: {filepath} not found")
        return []

    samples = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    return samples


def sample_datasets(
    raw_dir: Path,
    hotpotqa_count: int = 10000,
    wiki2_count: int = 10000,
    musique_count: int = 5000,
    seed: int = 42
) -> List[Dict]:
    """
    Sample from each dataset to create intermediate dataset.

    Args:
        raw_dir: Path to raw data directory
        hotpotqa_count: Number of samples from HotpotQA
        wiki2_count: Number of samples from 2WikiMultiHopQA
        musique_count: Number of samples from MuSiQue
        seed: Random seed for reproducibility

    Returns:
        List of sampled and tagged samples
    """
    random.seed(seed)
    sampled = []

    # Load and sample HotpotQA
    print(f"Loading HotpotQA...")
    hotpotqa = load_raw_dataset(raw_dir / "hotpotqa")
    if hotpotqa:
        print(f"  Found {len(hotpotqa)} samples, sampling {min(hotpotqa_count, len(hotpotqa))}")
        hotpotqa_sample = random.sample(hotpotqa, min(hotpotqa_count, len(hotpotqa)))
        for i, s in enumerate(hotpotqa_sample):
            s["dataset"] = "hotpotqa"
            if "id" not in s:
                s["id"] = f"hotpotqa_{i}"
        sampled.extend(hotpotqa_sample)
    else:
        print("  WARNING: No HotpotQA data found")

    # Load and sample 2WikiMultiHopQA
    print(f"Loading 2WikiMultiHopQA...")
    wiki2 = load_raw_dataset(raw_dir / "2wiki")
    if wiki2:
        print(f"  Found {len(wiki2)} samples, sampling {min(wiki2_count, len(wiki2))}")
        wiki2_sample = random.sample(wiki2, min(wiki2_count, len(wiki2)))
        for i, s in enumerate(wiki2_sample):
            s["dataset"] = "2wiki"
            if "id" not in s:
                s["id"] = f"2wiki_{i}"
        sampled.extend(wiki2_sample)
    else:
        print("  WARNING: No 2WikiMultiHopQA data found")

    # Load and sample MuSiQue
    print(f"Loading MuSiQue...")
    musique = load_raw_dataset(raw_dir / "musique")
    if musique:
        print(f"  Found {len(musique)} samples, sampling {min(musique_count, len(musique))}")
        musique_sample = random.sample(musique, min(musique_count, len(musique)))
        for i, s in enumerate(musique_sample):
            s["dataset"] = "musique"
            if "id" not in s:
                s["id"] = f"musique_{i}"
        sampled.extend(musique_sample)
    else:
        print("  WARNING: No MuSiQue data found")

    # Shuffle the combined dataset
    random.shuffle(sampled)

    print(f"\nTotal sampled: {len(sampled)} samples")
    return sampled


def normalize_context(ctx: Any) -> Dict:
    """
    Normalize a context item to standard format.

    Handles:
    - Dict with 'title'/'content' keys
    - Dict with 'title'/'paragraph_text' keys (MuSiQue)
    - List/tuple [title, content] or [title, [sentences...]]
    """
    if isinstance(ctx, dict):
        title = ctx.get("title", "")
        content = ctx.get("content", "") or ctx.get("paragraph_text", "")
        is_supporting = ctx.get("is_supporting", None)
        return {"title": title, "content": content, "is_supporting": is_supporting}
    elif isinstance(ctx, (list, tuple)) and len(ctx) >= 2:
        title = ctx[0]
        content = ctx[1]
        if isinstance(content, list):
            content = " ".join(content)
        return {"title": title, "content": content, "is_supporting": None}
    return {"title": "", "content": str(ctx), "is_supporting": None}


def normalize_supporting_fact(sf: Any) -> Dict:
    """
    Normalize a supporting fact to standard format.

    Handles:
    - Dict with 'title'/'content' keys
    - List/tuple [title, sent_idx] or [title, content]
    """
    if isinstance(sf, dict):
        title = sf.get("title", "")
        content = sf.get("content", "") or sf.get("paragraph_text", "")
        return {"title": title, "content": content}
    elif isinstance(sf, (list, tuple)):
        title = sf[0] if len(sf) > 0 else ""
        # sf[1] might be sent_idx (int) or content (str)
        content = sf[1] if len(sf) > 1 and isinstance(sf[1], str) else ""
        return {"title": title, "content": content}
    return {"title": "", "content": str(sf)}


def format_docs_as_numbered_list(contexts: List) -> str:
    """
    Format context documents as a numbered list.

    Args:
        contexts: List of context documents in various formats

    Returns:
        Formatted string with numbered documents
    """
    lines = []
    for i, ctx in enumerate(contexts, start=1):
        normalized = normalize_context(ctx)
        title = normalized["title"]
        content = normalized["content"]
        lines.append(f"{i}. {title}: {content}")
    return "\n\n".join(lines)


def compute_supporting_ids(
    contexts: List,
    supporting_facts: List
) -> List[int]:
    """
    Compute 1-indexed IDs of supporting documents.

    Handles multiple input formats:
    - contexts with 'is_supporting' flag (MuSiQue format)
    - supporting_facts as list of dicts or [title, idx] pairs

    Args:
        contexts: List of all context documents
        supporting_facts: List of supporting fact documents

    Returns:
        List of 1-indexed document IDs that are supporting
    """
    supporting_ids = []

    # Normalize contexts
    normalized_contexts = [normalize_context(ctx) for ctx in contexts]

    # Check if contexts have is_supporting flag (MuSiQue format)
    has_is_supporting = any(ctx["is_supporting"] is not None for ctx in normalized_contexts)

    if has_is_supporting:
        # Use is_supporting flag directly
        for i, ctx in enumerate(normalized_contexts, start=1):
            if ctx["is_supporting"]:
                supporting_ids.append(i)
    else:
        # Match supporting facts to contexts by title
        normalized_sfs = [normalize_supporting_fact(sf) for sf in (supporting_facts or [])]

        for i, ctx in enumerate(normalized_contexts, start=1):
            ctx_title = ctx["title"].lower().strip()
            ctx_content = ctx["content"].lower().strip()

            for sf in normalized_sfs:
                sf_title = sf["title"].lower().strip()
                sf_content = sf["content"].lower().strip()

                # Match by title
                if ctx_title and sf_title and ctx_title == sf_title:
                    supporting_ids.append(i)
                    break
                # Match by content overlap (if content is available)
                elif sf_content and len(sf_content) > 20:
                    if sf_content in ctx_content or ctx_content in sf_content:
                        supporting_ids.append(i)
                        break

    return sorted(list(set(supporting_ids)))


def get_answers(sample: Dict) -> List[str]:
    """Extract answers from sample, handling different formats."""
    # Check for 'answers' list
    if "answers" in sample and isinstance(sample["answers"], list):
        return sample["answers"]

    # Check for single 'answer' field
    answers = []
    if "answer" in sample:
        answers.append(sample["answer"])

    # Add answer_aliases if present
    if "answer_aliases" in sample:
        for alias in sample["answer_aliases"]:
            if alias and alias not in answers:
                answers.append(alias)

    return answers if answers else [""]


def apply_template(
    sample: Dict,
    template: str,
    version: str
) -> Dict:
    """
    Apply a CRAFT template to a sample.

    Args:
        sample: Sample with question, answers, contexts/paragraphs
        template: Template string with {query} and {docs} placeholders
        version: Template version (v1, v2, v3, v4, v5)

    Returns:
        Formatted sample for ms-swift training
    """
    question = sample["question"]

    # Handle different context field names
    contexts = sample.get("contexts") or sample.get("paragraphs") or sample.get("context", [])
    supporting_facts = sample.get("supporting_facts", [])

    # Format documents
    docs_text = format_docs_as_numbered_list(contexts)

    # Apply template
    prompt = template.format(query=question, docs=docs_text)

    # Compute supporting IDs
    supporting_ids = compute_supporting_ids(contexts, supporting_facts)

    # Get answers
    answers = get_answers(sample)

    # Build output format for ms-swift
    return {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "answers": answers,
        "supporting_ids": supporting_ids,
        "template_version": version,
        "name": sample.get("dataset", "unknown"),
        "id": sample.get("id", ""),
        "metadata": sample.get("metadata", {})
    }


def load_templates() -> Dict[str, str]:
    """Load CRAFT templates from the templates module."""
    import importlib.util
    template_path = get_project_root() / "src" / "templates" / "craft_templates.py"
    spec = importlib.util.spec_from_file_location("craft_templates", template_path)
    craft_templates = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(craft_templates)
    return craft_templates.CRAFT_TEMPLATES


def generate_grpo_files(
    intermediate_data: List[Dict],
    output_dir: Path,
    versions: List[str],
    prefix: str = "grpo"
) -> None:
    """
    Generate GRPO training files for specified template versions.

    Args:
        intermediate_data: List of sampled raw data
        output_dir: Output directory
        versions: List of template versions to generate
        prefix: Prefix for output filenames
    """
    CRAFT_TEMPLATES = load_templates()

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = len(intermediate_data)

    for version in versions:
        if version not in CRAFT_TEMPLATES:
            print(f"WARNING: Unknown template version '{version}', skipping")
            continue

        template = CRAFT_TEMPLATES[version]
        output_path = output_dir / f"{prefix}_{sample_count}_{version}_messages.jsonl"

        print(f"Generating {version}...")
        with open(output_path, "w") as f:
            for sample in intermediate_data:
                formatted = apply_template(sample, template, version)
                f.write(json.dumps(formatted, ensure_ascii=False) + "\n")

        print(f"  Saved {sample_count} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate GRPO training data for CRAFT"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=20000,
        help="Total number of samples (default: 20000)"
    )
    parser.add_argument(
        "--hotpotqa-ratio",
        type=float,
        default=0.25,
        help="Ratio of HotpotQA samples (default: 0.25 = 5000 of 20000)"
    )
    parser.add_argument(
        "--wiki2-ratio",
        type=float,
        default=0.25,
        help="Ratio of 2WikiMultiHopQA samples (default: 0.25 = 5000 of 20000)"
    )
    parser.add_argument(
        "--musique-ratio",
        type=float,
        default=0.5,
        help="Ratio of MuSiQue samples (default: 0.5 = 10000 of 20000)"
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v1", "v2", "v3", "v4", "v5"],
        choices=["v1", "v2", "v3", "v4", "v5"],
        help="Template versions to generate (default: all v1-v5)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory (default: CRAFT/data/train/grpo)"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Raw data directory (default: CRAFT/data/raw)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--skip-intermediate",
        action="store_true",
        help="Skip saving intermediate file"
    )
    parser.add_argument(
        "--intermediate-file",
        type=str,
        default=None,
        help="Use existing intermediate file instead of resampling"
    )
    args = parser.parse_args()

    project_root = get_project_root()

    # Determine directories
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "data" / "train" / "grpo"

    if args.raw_dir:
        raw_dir = Path(args.raw_dir)
    else:
        raw_dir = project_root / "data" / "raw"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("CRAFT GRPO Data Generation")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Raw data dir: {raw_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Target samples: {args.samples}")
    print(f"Template versions: {', '.join(args.versions)}")
    print(f"Seed: {args.seed}")
    print()

    # Step 1: Load or create intermediate data
    intermediate_path = output_dir / f"intermediate_{args.samples}.jsonl"

    if args.intermediate_file:
        # Load existing intermediate file
        intermediate_path = Path(args.intermediate_file)
        print(f"Loading existing intermediate file: {intermediate_path}")
        intermediate_data = []
        with open(intermediate_path, "r") as f:
            for line in f:
                if line.strip():
                    intermediate_data.append(json.loads(line))
        print(f"  Loaded {len(intermediate_data)} samples")
    else:
        # Sample and create intermediate data
        print("Step 1: Sampling datasets")
        print("-" * 60)

        # Calculate sample counts from ratios
        hotpotqa_count = int(args.samples * args.hotpotqa_ratio)
        wiki2_count = int(args.samples * args.wiki2_ratio)
        musique_count = args.samples - hotpotqa_count - wiki2_count

        print(f"Sampling: HotpotQA={hotpotqa_count}, 2Wiki={wiki2_count}, MuSiQue={musique_count}")
        print()

        intermediate_data = sample_datasets(
            raw_dir=raw_dir,
            hotpotqa_count=hotpotqa_count,
            wiki2_count=wiki2_count,
            musique_count=musique_count,
            seed=args.seed
        )

        if not intermediate_data:
            print("ERROR: No data loaded. Please run download_datasets.py first.")
            return 1

        # Save intermediate file
        if not args.skip_intermediate:
            print(f"\nSaving intermediate file: {intermediate_path}")
            with open(intermediate_path, "w") as f:
                for sample in intermediate_data:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"  Saved {len(intermediate_data)} samples")

    print()

    # Step 2: Generate template-specific files
    print("Step 2: Generating template-specific training files")
    print("-" * 60)

    generate_grpo_files(
        intermediate_data=intermediate_data,
        output_dir=output_dir,
        versions=args.versions,
        prefix="grpo"
    )

    print()
    print("=" * 60)
    print("Generation complete!")
    print()
    print("Generated files:")
    for version in args.versions:
        filepath = output_dir / f"grpo_{len(intermediate_data)}_{version}_messages.jsonl"
        if filepath.exists():
            print(f"  - {filepath}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
