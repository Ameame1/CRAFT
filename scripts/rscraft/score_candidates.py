#!/usr/bin/env python3
"""
RS-CRAFT Stage B: score candidate traces with the same auditor as CRAFT.

Reward components (mirrors GRPO total reward in src/rewards/craft_rewards.py):
  R_fmt    format_reward
  R_gold   relevance_reward
  R_ans    accuracy_reward
  R_faith  judge_overall_consistency_reward (LLM judge: Qwen3-30B-A3B local mode)

Input :  data/train/rscraft/candidates_v1.jsonl
Output:  data/train/rscraft/scored_v1.jsonl
            {id, messages, candidates: [{text, R_fmt, R_gold, R_ans, R_faith, R_total}], ...}
Resumable: skips ids already present in the output file.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config
from src.rewards.craft_rewards import (
    format_score,
    accuracy_score,
    relevance_score,
    judge_overall_consistency_reward,
    set_judge_mode,
    set_judge_model,
    set_template_version,
    set_batch_reward_workers,
    extract_question_documents_from_messages,
    get_judge_call_stats,
    clear_judge_call_stats,
)


def load_jsonl(path: Path) -> List[Dict]:
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    seen = set()
    with open(path, "r") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["id"])
            except Exception:
                continue
    return seen


def configure_judge(cfg: Dict) -> None:
    mode = cfg.get("judge_mode", "local")
    set_judge_mode(mode)
    if mode == "local":
        os.environ["LOCAL_JUDGE_URL"] = cfg.get("judge_url", "http://localhost:8000/v1")
    set_judge_model(cfg.get("judge_model", "Qwen/Qwen3-30B-A3B-Instruct-2507"))
    set_template_version(cfg.get("template_version", "v1"))
    set_batch_reward_workers(int(cfg.get("batch_reward_workers", 32)))


def score_batch(rows: List[Dict], template_version: str) -> List[Dict]:
    """Score one batch of rows. Each row has up to N candidates.
    Strategy: flatten to (row_idx, cand_idx, text) tuples, run rewards in
    one big batch (judge cache + parallel API calls), then re-shape."""
    flat_texts: List[str] = []
    flat_origin: List = []  # (row_idx, cand_idx)
    flat_questions: List[str] = []
    flat_documents: List[str] = []
    flat_answers: List = []
    flat_supports: List = []

    # Pre-extract per-row question/docs once
    per_row_qd = []
    for row in rows:
        qs, ds = extract_question_documents_from_messages([row["messages"]])
        per_row_qd.append((qs[0] if qs else "", ds[0] if ds else ""))

    for ri, row in enumerate(rows):
        cands = row.get("candidates", []) or []
        q, d = per_row_qd[ri]
        for ci, txt in enumerate(cands):
            flat_texts.append(txt)
            flat_origin.append((ri, ci))
            flat_questions.append(q)
            flat_documents.append(d)
            flat_answers.append(row.get("answers", []))
            flat_supports.append(row.get("supporting_ids", []))

    # Deterministic rewards (CPU only)
    fmt = [format_score(t, template_version) for t in flat_texts]
    ans = [accuracy_score(t, a, template_version) for t, a in zip(flat_texts, flat_answers)]
    rel = [relevance_score(t, s, template_version) for t, s in zip(flat_texts, flat_supports)]

    # Judge reward (only on candidates that passed format; failed-format can skip judge)
    faith = [0.0] * len(flat_texts)
    raw = [""] * len(flat_texts)
    judge_breakdown = [None] * len(flat_texts)
    judge_idx = [i for i, f in enumerate(fmt) if f > 0]
    if judge_idx:
        sub_texts = [flat_texts[i] for i in judge_idx]
        sub_qs = [flat_questions[i] for i in judge_idx]
        sub_ds = [flat_documents[i] for i in judge_idx]
        sub_scores = judge_overall_consistency_reward(
            completions=sub_texts,
            question=sub_qs,
            documents=sub_ds,
            template_version=template_version,
        )
        # Pull raw_response and per-criterion scores out of the judge cache so
        # we can recover them later if a parser bug is discovered.
        from src.rewards.craft_rewards import _judge_cache
        for i, sc in zip(judge_idx, sub_scores):
            faith[i] = sc
            ck = f"{flat_questions[i]}::{flat_documents[i]}::{flat_texts[i]}::{template_version}"
            cached = _judge_cache.get(ck, {})
            raw[i] = cached.get("raw_response", "")
            judge_breakdown[i] = {
                "plan_reason_consistency":          cached.get("plan_reason_consistency", 0.0),
                "receipt_reason_consistency":       cached.get("receipt_reason_consistency", 0.0),
                "reason_answer_consistency":        cached.get("reason_answer_consistency", 0.0),
                "evidence_grounded_faithfulness":   cached.get("evidence_grounded_faithfulness", 0.0),
                "overall_consistency":              cached.get("overall_consistency", 0.0),
                "failure_modes":                    cached.get("failure_modes", []),
            }

    # Re-shape into per-row results
    results = [None] * len(rows)
    for i, (ri, ci) in enumerate(flat_origin):
        if results[ri] is None:
            cands_out = []
            results[ri] = {
                "id": rows[ri].get("id"),
                "name": rows[ri].get("name", "unknown"),
                "messages": rows[ri]["messages"],
                "answers": rows[ri].get("answers", []),
                "supporting_ids": rows[ri].get("supporting_ids", []),
                "template_version": template_version,
                "candidates": cands_out,
            }
        results[ri]["candidates"].append({
            "idx": ci,
            "text": flat_texts[i],
            "R_fmt": fmt[i],
            "R_gold": rel[i],
            "R_ans": ans[i],
            "R_faith": faith[i],
            "R_total": (fmt[i] + rel[i] + ans[i] + faith[i]) / 4.0,
            "judge_breakdown": judge_breakdown[i],
            "judge_raw": raw[i],
        })

    # Some rows may have empty candidates (sampling errors) -> still emit a stub
    for ri, r in enumerate(results):
        if r is None:
            results[ri] = {
                "id": rows[ri].get("id"),
                "name": rows[ri].get("name", "unknown"),
                "messages": rows[ri]["messages"],
                "answers": rows[ri].get("answers", []),
                "supporting_ids": rows[ri].get("supporting_ids", []),
                "template_version": template_version,
                "candidates": [],
            }
    return results


def main():
    parser = argparse.ArgumentParser(description="RS-CRAFT Stage B: score candidates")
    parser.add_argument("--config", default="rscraft")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-rows", type=int, default=64,
                        help="prompts per scoring batch (judge calls = batch * N)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) or {}
    input_path = Path(args.input or cfg.get("candidates_path"))
    output_path = Path(args.output or cfg.get("scored_path"))
    template_version = cfg.get("template_version", "v1")

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configure_judge(cfg)

    print("=" * 60)
    print("RS-CRAFT Stage B: score candidates")
    print("=" * 60)
    print(f"Input         : {input_path}")
    print(f"Output        : {output_path}")
    print(f"Template      : {template_version}")
    print(f"Judge mode    : {cfg.get('judge_mode')}")
    print(f"Judge model   : {cfg.get('judge_model')}")
    print(f"Judge URL     : {cfg.get('judge_url')}")
    print(f"Workers       : {cfg.get('batch_reward_workers')}")
    print(f"Batch rows    : {args.batch_rows}")
    print("=" * 60)

    rows = load_jsonl(input_path)
    if args.limit:
        rows = rows[: args.limit]
    seen = existing_ids(output_path)
    todo = [r for r in rows if r.get("id") not in seen]
    print(f"Total: {len(rows)}; already done: {len(seen)}; todo: {len(todo)}")

    if not todo:
        return

    # Telemetry sidecar files (judge call stats + aggregate manifest)
    stats_path = output_path.with_suffix(".judge_stats.jsonl")
    manifest_path = output_path.with_suffix(".manifest.json")
    clear_judge_call_stats()

    written = 0
    started = time.time()
    last_stats_dump = 0
    with open(output_path, "a") as out_f, open(stats_path, "a") as stats_f:
        for batch_start in range(0, len(todo), args.batch_rows):
            batch = todo[batch_start: batch_start + args.batch_rows]
            scored = score_batch(batch, template_version)
            for s in scored:
                out_f.write(json.dumps(s, ensure_ascii=False) + "\n")
            out_f.flush()
            written += len(scored)

            # Drain new judge-call telemetry to the sidecar (in order).
            all_stats = get_judge_call_stats()
            new_stats = all_stats[last_stats_dump:]
            for s in new_stats:
                stats_f.write(json.dumps(s) + "\n")
            stats_f.flush()
            last_stats_dump = len(all_stats)

            elapsed = time.time() - started
            rate = written / elapsed if elapsed else 0
            eta = (len(todo) - written) / rate if rate else 0
            n_calls = len(all_stats)
            tot_in  = sum(s["input_tokens"] for s in all_stats)
            tot_out = sum(s["output_tokens"] for s in all_stats)
            mean_lat = (sum(s["latency_s"] for s in all_stats) / n_calls) if n_calls else 0.0
            print(f"  [{written}/{len(todo)}] {rate:.2f} prompts/s  ETA {eta/60:.1f} min "
                  f"| judge: {n_calls} calls, {mean_lat*1000:.0f} ms avg, "
                  f"{tot_in/1e6:.2f}M in / {tot_out/1e6:.2f}M out")

    # Final aggregate manifest
    elapsed = time.time() - started
    all_stats = get_judge_call_stats()
    n_calls = len(all_stats)
    tot_in = sum(s["input_tokens"] for s in all_stats)
    tot_out = sum(s["output_tokens"] for s in all_stats)
    latencies = sorted([s["latency_s"] for s in all_stats])
    def pct(p):
        if not latencies: return 0.0
        return latencies[min(len(latencies)-1, int(len(latencies)*p))]
    manifest = {
        "stage": "B-score",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "stats_path": str(stats_path),
        "template_version": template_version,
        "judge_mode": cfg.get("judge_mode"),
        "judge_model": cfg.get("judge_model"),
        "judge_url": cfg.get("judge_url"),
        "batch_reward_workers": cfg.get("batch_reward_workers"),
        "rows_processed": written,
        "wall_clock_s": elapsed,
        "rows_per_sec": written / elapsed if elapsed else 0,
        "judge_calls_total": n_calls,
        "judge_input_tokens_total": tot_in,
        "judge_output_tokens_total": tot_out,
        "judge_total_tokens_total": tot_in + tot_out,
        "judge_latency_mean_s": (sum(latencies) / n_calls) if n_calls else 0,
        "judge_latency_p50_s": pct(0.50),
        "judge_latency_p90_s": pct(0.90),
        "judge_latency_p99_s": pct(0.99),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Wrote {written} new rows to {output_path}.")
    print(f"Judge stats sidecar: {stats_path} ({n_calls} calls)")
    print(f"Manifest          : {manifest_path}")


if __name__ == "__main__":
    main()
