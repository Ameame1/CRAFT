#!/usr/bin/env python3
"""
Download scripts for CRAFT training datasets.

Downloads HotpotQA, 2WikiMultiHopQA, and MuSiQue datasets from their
original sources and converts them to a unified JSONL format.

Usage:
    python scripts/preprocess/download_datasets.py [--force]
"""

import argparse
import json
import os
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
from urllib.request import urlretrieve
from urllib.error import URLError


def get_project_root() -> Path:
    """Get the CRAFT project root directory."""
    script_dir = Path(__file__).resolve().parent
    # scripts/preprocess -> CRAFT
    return script_dir.parent.parent


def download_file(url: str, dest_path: str, description: str = "") -> bool:
    """Download a file from URL with progress indicator."""
    print(f"Downloading {description or url}...")
    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                print(f"\r  Progress: {percent}%", end="", flush=True)

        urlretrieve(url, dest_path, reporthook=progress_hook)
        print()  # newline after progress
        return True
    except URLError as e:
        print(f"\n  ERROR: Failed to download: {e}")
        return False


def convert_hotpotqa_sample(raw_sample: Dict) -> Dict:
    """
    Convert a raw HotpotQA sample to unified format.

    Raw format:
        {
            "_id": "...",
            "question": "...",
            "answer": "...",
            "type": "bridge|comparison",
            "level": "hard|medium|easy",
            "supporting_facts": [[title, sent_idx], ...],
            "context": [[title, [sent0, sent1, ...]], ...]
        }

    Unified format:
        {
            "id": "hotpotqa_...",
            "question": "...",
            "answers": ["..."],
            "supporting_facts": [{"title": "...", "content": "..."}],
            "contexts": [{"title": "...", "content": "..."}]
        }
    """
    # Build title -> sentences mapping
    title_to_sents = {}
    for title, sents in raw_sample.get("context", []):
        title_to_sents[title] = sents

    # Extract supporting facts
    supporting_facts = []
    sf_set = set()  # Track unique (title, sent_idx) pairs
    for title, sent_idx in raw_sample.get("supporting_facts", []):
        if (title, sent_idx) in sf_set:
            continue
        sf_set.add((title, sent_idx))

        if title in title_to_sents and sent_idx < len(title_to_sents[title]):
            supporting_facts.append({
                "title": title,
                "content": title_to_sents[title][sent_idx]
            })

    # Build contexts (full paragraphs)
    contexts = []
    for title, sents in raw_sample.get("context", []):
        contexts.append({
            "title": title,
            "content": " ".join(sents)
        })

    return {
        "id": f"hotpotqa_{raw_sample['_id']}",
        "question": raw_sample["question"],
        "answers": [raw_sample["answer"]],
        "supporting_facts": supporting_facts,
        "contexts": contexts,
        "metadata": {
            "type": raw_sample.get("type", "unknown"),
            "level": raw_sample.get("level", "unknown"),
            "original_id": raw_sample["_id"]
        }
    }


def convert_2wiki_sample(raw_sample: Dict) -> Dict:
    """
    Convert a raw 2WikiMultiHopQA sample to unified format.

    Raw format:
        {
            "_id": "...",
            "type": "compositional|comparison|...",
            "question": "...",
            "answer": "...",
            "context": [[title, [sent0, sent1, ...]], ...],
            "supporting_facts": [[title, sent_idx], ...]
        }
    """
    # Build title -> sentences mapping
    title_to_sents = {}
    for title, sents in raw_sample.get("context", []):
        title_to_sents[title] = sents

    # Extract supporting facts
    supporting_facts = []
    sf_set = set()
    for sf in raw_sample.get("supporting_facts", []):
        title, sent_idx = sf[0], sf[1]
        if (title, sent_idx) in sf_set:
            continue
        sf_set.add((title, sent_idx))

        if title in title_to_sents and sent_idx < len(title_to_sents[title]):
            supporting_facts.append({
                "title": title,
                "content": title_to_sents[title][sent_idx]
            })

    # Build contexts (full paragraphs)
    contexts = []
    for title, sents in raw_sample.get("context", []):
        contexts.append({
            "title": title,
            "content": " ".join(sents)
        })

    return {
        "id": f"2wiki_{raw_sample['_id']}",
        "question": raw_sample["question"],
        "answers": [raw_sample["answer"]],
        "supporting_facts": supporting_facts,
        "contexts": contexts,
        "metadata": {
            "type": raw_sample.get("type", "unknown"),
            "original_id": raw_sample["_id"]
        }
    }


def convert_musique_sample(raw_sample: Dict) -> Dict:
    """
    Convert a raw MuSiQue sample to unified format.

    Raw format:
        {
            "id": "...",
            "question": "...",
            "answer": "...",
            "answer_aliases": ["..."],
            "paragraphs": [
                {"idx": 0, "title": "...", "paragraph_text": "...", "is_supporting": true/false},
                ...
            ],
            "question_decomposition": [...]
        }
    """
    # Extract supporting facts and contexts
    supporting_facts = []
    contexts = []

    for para in raw_sample.get("paragraphs", []):
        ctx = {
            "title": para["title"],
            "content": para["paragraph_text"]
        }
        contexts.append(ctx)

        if para.get("is_supporting", False):
            supporting_facts.append(ctx.copy())

    # Build answers list (include aliases)
    answers = [raw_sample["answer"]]
    for alias in raw_sample.get("answer_aliases", []):
        if alias and alias not in answers:
            answers.append(alias)

    return {
        "id": f"musique_{raw_sample['id']}",
        "question": raw_sample["question"],
        "answers": answers,
        "supporting_facts": supporting_facts,
        "contexts": contexts,
        "metadata": {
            "original_id": raw_sample["id"],
            "answerable": raw_sample.get("answerable", True)
        }
    }


def download_hotpotqa(output_dir: Path, force: bool = False) -> bool:
    """
    Download HotpotQA dataset from CMU.

    Sources:
        Train: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
        Dev: http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"

    if train_path.exists() and dev_path.exists() and not force:
        print(f"HotpotQA already exists at {output_dir}, skipping (use --force to redownload)")
        return True

    urls = {
        "train": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json",
        "dev": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
    }

    for split, url in urls.items():
        output_path = output_dir / f"{split}.jsonl"

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            if not download_file(url, tmp_path, f"HotpotQA {split}"):
                return False

            print(f"Converting HotpotQA {split} to unified format...")
            with open(tmp_path, "r") as f:
                raw_data = json.load(f)

            with open(output_path, "w") as f:
                for sample in raw_data:
                    converted = convert_hotpotqa_sample(sample)
                    f.write(json.dumps(converted, ensure_ascii=False) + "\n")

            print(f"  Saved {len(raw_data)} samples to {output_path}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return True


def download_2wiki(output_dir: Path, force: bool = False) -> bool:
    """
    Download 2WikiMultiHopQA dataset from Dropbox.

    Source: https://www.dropbox.com/scl/fi/32t7pv1dyf3o2pp0dl25u/data_ids_april7.zip
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"

    if train_path.exists() and dev_path.exists() and not force:
        print(f"2WikiMultiHopQA already exists at {output_dir}, skipping (use --force to redownload)")
        return True

    # Dropbox direct download link
    url = "https://www.dropbox.com/scl/fi/32t7pv1dyf3o2pp0dl25u/data_ids_april7.zip?rlkey=q6lbnqx16xnsf0d7j9wd71a1c&dl=1"

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "2wiki.zip")

        if not download_file(url, zip_path, "2WikiMultiHopQA"):
            return False

        print("Extracting 2WikiMultiHopQA...")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmp_dir)
        except zipfile.BadZipFile:
            print("  ERROR: Downloaded file is not a valid zip archive")
            return False

        # Find the extracted files
        extract_dir = os.path.join(tmp_dir, "data_ids_april7")
        if not os.path.exists(extract_dir):
            # Try to find it
            for item in os.listdir(tmp_dir):
                if os.path.isdir(os.path.join(tmp_dir, item)) and item != "__MACOSX":
                    extract_dir = os.path.join(tmp_dir, item)
                    break

        for split in ["train", "dev"]:
            # Look for the JSON file
            src_file = os.path.join(extract_dir, f"{split}.json")
            if not os.path.exists(src_file):
                # Try alternative naming
                for alt_name in [f"{split}.json", f"2wiki_{split}.json"]:
                    alt_path = os.path.join(extract_dir, alt_name)
                    if os.path.exists(alt_path):
                        src_file = alt_path
                        break

            if not os.path.exists(src_file):
                print(f"  WARNING: Could not find {split} split in 2WikiMultiHopQA")
                continue

            print(f"Converting 2WikiMultiHopQA {split} to unified format...")
            with open(src_file, "r") as f:
                raw_data = json.load(f)

            output_path = output_dir / f"{split}.jsonl"
            with open(output_path, "w") as f:
                for sample in raw_data:
                    converted = convert_2wiki_sample(sample)
                    f.write(json.dumps(converted, ensure_ascii=False) + "\n")

            print(f"  Saved {len(raw_data)} samples to {output_path}")

    return True


def download_musique(output_dir: Path, force: bool = False) -> bool:
    """
    Download MuSiQue dataset from Google Drive using gdown.

    Google Drive file ID: 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"

    if train_path.exists() and dev_path.exists() and not force:
        print(f"MuSiQue already exists at {output_dir}, skipping (use --force to redownload)")
        return True

    # Try to import gdown
    try:
        import gdown
    except ImportError:
        print("Installing gdown for Google Drive download...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    file_id = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"
    url = f"https://drive.google.com/uc?id={file_id}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "musique.zip")

        print("Downloading MuSiQue from Google Drive...")
        try:
            gdown.download(url, zip_path, quiet=False)
        except Exception as e:
            print(f"  ERROR: Failed to download MuSiQue: {e}")
            return False

        if not os.path.exists(zip_path):
            print("  ERROR: Download failed - file not created")
            return False

        print("Extracting MuSiQue...")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmp_dir)
        except zipfile.BadZipFile:
            print("  ERROR: Downloaded file is not a valid zip archive")
            return False

        # Find the extracted directory
        extract_dir = tmp_dir
        for item in os.listdir(tmp_dir):
            item_path = os.path.join(tmp_dir, item)
            if os.path.isdir(item_path) and item not in ["__MACOSX", ".DS_Store"]:
                extract_dir = item_path
                break

        # Look for JSONL files
        split_mapping = {
            "train": ["musique_ans_v1.0_train.jsonl", "train.jsonl"],
            "dev": ["musique_ans_v1.0_dev.jsonl", "dev.jsonl"]
        }

        for split, possible_names in split_mapping.items():
            src_file = None
            for name in possible_names:
                check_path = os.path.join(extract_dir, name)
                if os.path.exists(check_path):
                    src_file = check_path
                    break
                # Also check data subdirectory
                check_path = os.path.join(extract_dir, "data", name)
                if os.path.exists(check_path):
                    src_file = check_path
                    break

            if not src_file:
                # Search recursively
                for root, dirs, files in os.walk(extract_dir):
                    for f in files:
                        if split in f.lower() and f.endswith(".jsonl"):
                            src_file = os.path.join(root, f)
                            break
                    if src_file:
                        break

            if not src_file:
                print(f"  WARNING: Could not find {split} split in MuSiQue")
                continue

            print(f"Converting MuSiQue {split} to unified format...")
            samples = []
            with open(src_file, "r") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

            output_path = output_dir / f"{split}.jsonl"
            with open(output_path, "w") as f:
                for sample in samples:
                    converted = convert_musique_sample(sample)
                    f.write(json.dumps(converted, ensure_ascii=False) + "\n")

            print(f"  Saved {len(samples)} samples to {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download CRAFT training datasets (HotpotQA, 2WikiMultiHopQA, MuSiQue)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for raw data (default: CRAFT/data/raw)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa", "2wiki", "musique"],
        choices=["hotpotqa", "2wiki", "musique"],
        help="Which datasets to download (default: all)"
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        project_root = get_project_root()
        output_base = project_root / "data" / "raw"

    output_base.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_base}")
    print()

    success = True

    if "hotpotqa" in args.datasets:
        print("=" * 60)
        print("Downloading HotpotQA...")
        print("=" * 60)
        if not download_hotpotqa(output_base / "hotpotqa", args.force):
            success = False
        print()

    if "2wiki" in args.datasets:
        print("=" * 60)
        print("Downloading 2WikiMultiHopQA...")
        print("=" * 60)
        if not download_2wiki(output_base / "2wiki", args.force):
            success = False
        print()

    if "musique" in args.datasets:
        print("=" * 60)
        print("Downloading MuSiQue...")
        print("=" * 60)
        if not download_musique(output_base / "musique", args.force):
            success = False
        print()

    if success:
        print("=" * 60)
        print("All downloads completed successfully!")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("Some downloads failed. Check the error messages above.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
