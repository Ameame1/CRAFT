#!/usr/bin/env python3
"""
Reward-behavior case-study selector (item #2 of the rebuttal).

Aligns the four 1.5B v1 variant traces by `id` and picks "interesting" cases
where the trace progressively becomes more faithful as more rewards are added:

    Base          : EM=1, faithfulness=0    (right-answer-wrong-reason)
    SFT           : EM/format pass, faithfulness varies
    CRAFT w/o J   : deterministic rewards pass; judge metrics partial
    CRAFT full    : EM=1 AND ALL judge metrics pass

CPU-only; runs in seconds once all four files exist.

Usage:
    python scripts/analysis/case_study_select.py \
        --base   data/results/faithfulness_eval/base_models/traces_with_judge \
        --sft    data/results/faithfulness_eval/sft_models/traces_with_judge \
        --grpo-full   data/results/faithfulness_eval/grpo_models/1p5B_v1_eval_20260115_233254/traces_with_judge \
        --grpo-nojudge <dir produced by ABL eval>/traces_with_judge \
        --datasets musique hotpotqa 2wiki \
        --variant v1 \
        --tier A \
        --top-k 10 \
        --out docs/rebuttal/02_case_studies.md
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def load_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_id(rows: List[dict]) -> Dict[str, dict]:
    return {str(r.get("id", r.get("idx"))): r for r in rows}


def js(d: Optional[dict], key: str) -> float:
    if not d:
        return -1.0
    return float((d.get("judge_scores") or {}).get(key, 0.0))


def all_judge_pass(r: Optional[dict]) -> bool:
    if not r:
        return False
    s = r.get("judge_scores") or {}
    return (s.get("plan_reason_consistency", 0) == 1
            and s.get("receipt_reason_consistency", 0) == 1
            and s.get("reason_answer_consistency", 0) == 1
            and s.get("evidence_grounded_faithfulness", 0) == 1)


def any_judge_zero(r: Optional[dict]) -> bool:
    if not r:
        return False
    s = r.get("judge_scores") or {}
    return any(s.get(k, 0) == 0 for k in (
        "plan_reason_consistency", "receipt_reason_consistency",
        "reason_answer_consistency", "evidence_grounded_faithfulness"))


def is_em(r: Optional[dict]) -> int:
    if not r:
        return -1
    return int(r.get("exact_match", 0))


def fmt_resp(r: Optional[dict]) -> str:
    if not r:
        return "(missing)"
    return r.get("full_response", "")


def find_path(dir_: Optional[Path], variant: str, ds: str, prefix: str) -> Optional[Path]:
    """Find the right jsonl in a directory that may have multiple variant files."""
    if not dir_:
        return None
    candidates = list(dir_.glob(f"{prefix}*{variant}*{ds}*results.jsonl"))
    return candidates[0] if candidates else None


def select_tier_a(idx_b, idx_s, idx_gn, idx_gf, top_k: int):
    """Select cases satisfying Tier A: Base EM=1+any judge=0; CRAFT-full EM=1+all judge=1."""
    keys = sorted(set(idx_b) & set(idx_gf))
    out = []
    for k in keys:
        b = idx_b.get(k)
        s = idx_s.get(k)
        gn = idx_gn.get(k)
        gf = idx_gf.get(k)
        if is_em(b) == 1 and any_judge_zero(b) and is_em(gf) == 1 and all_judge_pass(gf):
            out.append({"id": k, "base": b, "sft": s, "grpo_nojudge": gn, "grpo_full": gf})
        if len(out) >= top_k:
            break
    return out


def select_tier_b(idx_b, idx_s, idx_gn, idx_gf, top_k: int):
    """Tier B: Base EM=0 AND CRAFT-full EM=1 AND all judge=1 — accuracy+faithfulness joint gain."""
    keys = sorted(set(idx_b) & set(idx_gf))
    out = []
    for k in keys:
        b = idx_b.get(k)
        s = idx_s.get(k)
        gn = idx_gn.get(k)
        gf = idx_gf.get(k)
        if is_em(b) == 0 and is_em(gf) == 1 and all_judge_pass(gf):
            out.append({"id": k, "base": b, "sft": s, "grpo_nojudge": gn, "grpo_full": gf})
        if len(out) >= top_k:
            break
    return out


def render_case(case: dict, ds: str) -> str:
    cid = case["id"]
    out = [f"### Case `{cid}` ({ds})\n"]
    for label, key in [("Base", "base"), ("SFT", "sft"),
                       ("CRAFT w/o judge", "grpo_nojudge"), ("CRAFT (full)", "grpo_full")]:
        r = case[key]
        if r is None:
            out.append(f"- **{label}**: (variant not yet available)\n")
            continue
        s = r.get("judge_scores") or {}
        out.append(f"- **{label}** — EM={is_em(r)}, "
                   f"plan→reason={s.get('plan_reason_consistency','-')}, "
                   f"cite→reason={s.get('receipt_reason_consistency','-')}, "
                   f"reason→answer={s.get('reason_answer_consistency','-')}, "
                   f"evidence={s.get('evidence_grounded_faithfulness','-')}, "
                   f"prediction=`{r.get('prediction','')}`")
        out.append(f"  ```\n  {fmt_resp(r)[:1200]}\n  ```\n")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--sft", required=True)
    p.add_argument("--grpo-full", required=True)
    p.add_argument("--grpo-nojudge", default=None,
                   help="Optional; if missing, the column is left as 'not yet available'")
    p.add_argument("--datasets", nargs="+", default=["musique", "hotpotqa", "2wiki"])
    p.add_argument("--variant", default="v1")
    p.add_argument("--tier", choices=["A", "B", "AB"], default="AB")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base_dir = Path(args.base)
    sft_dir = Path(args.sft)
    full_dir = Path(args.grpo_full)
    nj_dir = Path(args.grpo_nojudge) if args.grpo_nojudge else None

    md = ["# Reward-Behavior Case Studies (auto-extracted)\n",
          "Source: 1.5B v1 trace files with judge scores attached. Aligned by `id`."]

    for ds in args.datasets:
        b_path = find_path(base_dir, args.variant, ds, "1.5B_base_")
        s_path = find_path(sft_dir, args.variant, ds, "1.5B_sft_")
        f_path = find_path(full_dir, args.variant, ds, "1.5B_")
        n_path = find_path(nj_dir, args.variant, ds, "1.5B_") if nj_dir else None
        if not (b_path and s_path and f_path):
            md.append(f"\n## {ds}\n_(missing one of base/sft/grpo-full files; skipping.)_\n")
            continue
        b, s, f, n = (load_jsonl(b_path), load_jsonl(s_path),
                       load_jsonl(f_path), load_jsonl(n_path) if n_path else [])
        idx_b, idx_s, idx_f, idx_n = (
            index_by_id(b), index_by_id(s), index_by_id(f), index_by_id(n))

        md.append(f"\n## {ds}  (rows: base={len(b)}, sft={len(s)}, grpo_full={len(f)}, "
                  f"grpo_nojudge={len(n)})")

        if args.tier in ("A", "AB"):
            cases_a = select_tier_a(idx_b, idx_s, idx_n, idx_f, args.top_k)
            md.append(f"\n### Tier A — right-answer-wrong-reason → fully faithful (n={len(cases_a)})\n")
            for c in cases_a:
                md.append(render_case(c, ds))
        if args.tier in ("B", "AB"):
            cases_b = select_tier_b(idx_b, idx_s, idx_n, idx_f, args.top_k)
            md.append(f"\n### Tier B — joint accuracy + faithfulness recovery (n={len(cases_b)})\n")
            for c in cases_b:
                md.append(render_case(c, ds))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
