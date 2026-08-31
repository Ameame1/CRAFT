#!/usr/bin/env python3
"""
Post-hoc overhead computation for RS-CRAFT Stage A (sampling).

Reads the candidates JSONL and (optionally) the Stage A log file to produce a
manifest with: rows, total candidates, input/output token totals (via Qwen
tokenizer), wall-clock from log timestamps, and sampling throughput.

Run AFTER Stage A finishes:
  python scripts/rscraft/compute_stage_a_overhead.py

Outputs:
  data/train/rscraft/candidates_v1.manifest.json
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config


def load_jsonl(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_log_wallclock(log_path: Path) -> Optional[dict]:
    """Stage A log lines look like '[N/T] X.XX prompts/s  ETA Y.Y min'.
    We use file mtime / atime as a fallback when no explicit timestamps exist."""
    if not log_path.exists():
        return None
    stat = log_path.stat()
    # Read first/last 'prompts/s' progress line to estimate runtime
    first_pct, last_pct, last_rate = None, None, None
    n_progress = 0
    with open(log_path, errors='ignore') as f:
        for line in f:
            m = re.search(r"\[(\d+)/(\d+)\].* (\d+(?:\.\d+)?) prompts/s", line)
            if m:
                n_progress += 1
                done, total, rate = int(m.group(1)), int(m.group(2)), float(m.group(3))
                if first_pct is None:
                    first_pct = (done, total, rate)
                last_pct = (done, total, rate)
                last_rate = rate
    if first_pct is None or last_pct is None:
        return None
    return {
        "log_path": str(log_path),
        "log_mtime": stat.st_mtime,
        "first_progress": first_pct,
        "last_progress": last_pct,
        "n_progress_lines": n_progress,
        "last_observed_rate": last_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rscraft")
    parser.add_argument("--input", default=None,
                        help="candidates JSONL (default: from cfg)")
    parser.add_argument("--log", default="/tmp/stage-a-full.log",
                        help="Stage A log file for wall-clock estimate")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    cfg = load_config(args.config) or {}
    input_path = Path(args.input or cfg.get("candidates_path"))
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    manifest_path = input_path.with_suffix(".manifest.json")

    print(f"Reading: {input_path}")
    rows = load_jsonl(input_path)
    print(f"Rows: {len(rows)}")

    # Tokenize via HF tokenizer (cached locally already)
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    n_rows = len(rows)
    n_cands = 0
    total_in_tokens = 0
    total_out_tokens = 0
    cand_lens = []
    err_rows = 0

    started = time.time()
    for i, r in enumerate(rows):
        if i % 1000 == 0 and i:
            elapsed = time.time() - started
            print(f"  tokenized {i}/{n_rows} ({i/elapsed:.1f}/s)")
        # Input: render the chat-formatted prompt that vLLM actually saw.
        try:
            prompt_text = tok.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=True)
            in_ids = tok.encode(prompt_text)
        except Exception:
            err_rows += 1
            in_ids = []
        total_in_tokens += len(in_ids)
        # Outputs: each candidate.
        for c in r.get("candidates", []) or []:
            if not c:
                continue
            out_ids = tok.encode(c)
            total_out_tokens += len(out_ids)
            cand_lens.append(len(out_ids))
            n_cands += 1

    cand_lens.sort()
    def pct(p):
        if not cand_lens: return 0
        return cand_lens[min(len(cand_lens)-1, int(len(cand_lens)*p))]

    log_info = parse_log_wallclock(Path(args.log))
    wallclock_s = None
    rate_inferred = None
    if log_info and log_info["last_observed_rate"]:
        # rate * total_rows ≈ time elapsed (approx, lower bound)
        rate_inferred = log_info["last_observed_rate"]
        wallclock_s = n_rows / rate_inferred if rate_inferred else None

    manifest = {
        "stage": "A-sample",
        "input_path": str(input_path),
        "tokenizer": args.tokenizer,
        "rows": n_rows,
        "candidates_total": n_cands,
        "candidates_per_row_avg": n_cands / max(n_rows, 1),
        "input_tokens_total": total_in_tokens,
        "output_tokens_total": total_out_tokens,
        "input_tokens_per_row_avg": total_in_tokens / max(n_rows, 1),
        "output_tokens_per_candidate_avg": total_out_tokens / max(n_cands, 1),
        "candidate_len_p50": pct(0.50),
        "candidate_len_p90": pct(0.90),
        "candidate_len_p99": pct(0.99),
        "candidate_len_max": cand_lens[-1] if cand_lens else 0,
        "wall_clock_s_estimate": wallclock_s,
        "rate_prompts_per_s_observed": rate_inferred,
        "log_info": log_info,
        "tokenize_errors": err_rows,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("log_info",)}, indent=2))


if __name__ == "__main__":
    main()
