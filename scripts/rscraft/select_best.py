#!/usr/bin/env python3
"""
RS-CRAFT Stage C: pick best-of-N per prompt and emit SFT-formatted JSONL.

Selection:
  argmax R_total (mean of R_fmt, R_gold, R_ans, R_faith)
  Tie-break:  R_ans -> R_faith -> R_gold -> shorter text
Filter:
  drop prompt if every candidate has R_fmt == 0
  drop prompt if max(R_total) < min_total_score (configurable floor)

Output (SFT format consumed by ms-swift):
  {messages: [{role:user, ...}, {role:assistant, content: <selected trace>}],
   answers, supporting_ids, name, id, template_version,
   selected_scores: {R_fmt, R_gold, R_ans, R_faith, R_total}}
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config


def load_jsonl(path: Path) -> List[Dict]:
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def pick_best(cands: List[Dict]) -> Optional[Dict]:
    valid = [c for c in cands if c.get("R_fmt", 0) > 0]
    if not valid:
        return None
    valid.sort(key=lambda c: (
        -c.get("R_total", 0.0),
        -c.get("R_ans", 0.0),
        -c.get("R_faith", 0.0),
        -c.get("R_gold", 0.0),
        len(c.get("text", "") or ""),
    ))
    return valid[0]


def main():
    parser = argparse.ArgumentParser(description="RS-CRAFT Stage C: select best-of-N")
    parser.add_argument("--config", default="rscraft")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-total-score", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) or {}
    input_path = Path(args.input or cfg.get("scored_path"))
    output_path = Path(args.output or cfg.get("selected_path"))
    floor = float(args.min_total_score if args.min_total_score is not None
                  else cfg.get("min_total_score", 0.0))

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    print("=" * 60)
    print("RS-CRAFT Stage C: select best-of-N")
    print("=" * 60)
    print(f"Input         : {input_path}")
    print(f"Output        : {output_path}")
    print(f"Floor R_total : {floor}")
    print(f"Total prompts : {len(rows)}")

    kept_scores = []
    dropped_no_format = 0
    dropped_below_floor = 0
    written = 0

    with open(output_path, "w") as out_f:
        for row in rows:
            best = pick_best(row.get("candidates", []))
            if best is None:
                dropped_no_format += 1
                continue
            if best["R_total"] < floor:
                dropped_below_floor += 1
                continue
            user_messages = row["messages"]
            sft_record = {
                "messages": list(user_messages) + [
                    {"role": "assistant", "content": best["text"]},
                ],
                "answers": row.get("answers", []),
                "supporting_ids": row.get("supporting_ids", []),
                "name": row.get("name", "unknown"),
                "id": row.get("id"),
                "template_version": row.get("template_version", "v1"),
                "selected_scores": {
                    "R_fmt": best["R_fmt"],
                    "R_gold": best["R_gold"],
                    "R_ans": best["R_ans"],
                    "R_faith": best["R_faith"],
                    "R_total": best["R_total"],
                },
            }
            out_f.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            kept_scores.append(best["R_total"])
            written += 1

    retention = written / len(rows) if rows else 0.0
    mean_score = statistics.mean(kept_scores) if kept_scores else 0.0
    print(f"Kept          : {written}  ({retention*100:.2f}%)")
    print(f"Dropped       : no_format={dropped_no_format}  below_floor={dropped_below_floor}")
    print(f"Mean R_total  : {mean_score:.4f}")
    print(f"Wrote         : {output_path}")


if __name__ == "__main__":
    main()
