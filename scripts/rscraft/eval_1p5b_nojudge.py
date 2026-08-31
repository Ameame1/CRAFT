#!/usr/bin/env python3
"""
ABL-eval: evaluate the new 1.5B w/o-judge GRPO model on hotpotqa / musique / 2wiki,
then run faithfulness scoring against the same Qwen3-30B judge that's serving
Stage B. Results land in:

    data/results/faithfulness_eval/grpo_models/1p5B_v1_nojudge_eval_<ts>/
        traces_with_judge/1.5B_v1_<ds>_temp0.0_results.jsonl

Once this is done, scripts/analysis/case_study_select.py has all four 1.5B v1
variants and the case study can be regenerated with all columns filled.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("VLLM_HOST", "http://localhost")
os.environ.setdefault("VLLM_PORT", "8002")

from src.eval.evaluator import Evaluator
from src.rewards.craft_rewards import (
    set_judge_mode, set_judge_model, set_template_version,
    set_batch_reward_workers, judge_overall_consistency_reward,
)


def run_inference_on(dataset: str, output_root: Path):
    """Use the Evaluator class to run inference for one dataset."""
    # Patch the eval config in-memory by writing a temporary YAML
    import yaml
    base = yaml.safe_load(open(PROJECT_ROOT / "cfg/eval.yaml"))
    base["model"] = "CRAFT-1.5B-NoJudge"
    base["val_dataset"] = f"data/test/{dataset}_2000_v1.jsonl"
    base["template_version"] = "v1"
    base["vllm_port"] = 8002
    tmp_cfg = PROJECT_ROOT / "cfg/_eval_1p5b_nojudge_tmp.yaml"
    with open(tmp_cfg, "w") as f:
        yaml.safe_dump(base, f)
    e = Evaluator(config_name="_eval_1p5b_nojudge_tmp",
                  project_root=str(PROJECT_ROOT))
    summary = e.run()
    return summary


def add_judge_scores(results_jsonl: Path, output_path: Path, ds_name: str):
    """Re-score each prediction with the judge (Qwen3-30B-A3B-Instruct-2507 local)."""
    rows = []
    with open(results_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"  Judging {len(rows)} rows for {ds_name} ...")

    test_path = PROJECT_ROOT / f"data/test/{ds_name}_2000_v1.jsonl"
    test_rows = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                test_rows.append(json.loads(line))
    test_by_idx = {i: r for i, r in enumerate(test_rows)}

    import re
    def _extract(prompt_messages):
        user_content = ""
        for m in prompt_messages:
            if m.get("role") == "user":
                user_content = m.get("content", "")
        q = re.search(r"<query>(.*?)</query>", user_content, re.DOTALL)
        d = re.search(r"<docs>(.*?)</docs>", user_content, re.DOTALL)
        return (q.group(1).strip() if q else ""), (d.group(1).strip() if d else "")

    # Build judge inputs
    questions, documents, completions = [], [], []
    for r in rows:
        idx = r.get("idx")
        tr = test_by_idx.get(idx)
        q, d = _extract(tr["prompt"]) if tr else ("", "")
        questions.append(q)
        documents.append(d)
        completions.append(r.get("full_response", ""))

    # Single batch (judge_overall_consistency_reward batches internally)
    scores = judge_overall_consistency_reward(
        completions=completions,
        question=questions,
        documents=documents,
        template_version="v1",
    )
    # Augment each row; we have aggregated mean only at this layer.
    # The full per-criterion dict is in the judge cache; pull it.
    from src.rewards.craft_rewards import _judge_cache
    for r, q, d, c, s in zip(rows, questions, documents, completions, scores):
        cache_key = f"{q}::{d}::{c}::v1"
        full = _judge_cache.get(cache_key, {})
        r["judge_scores"] = {
            "plan_reason_consistency": full.get("plan_reason_consistency", 0.0),
            "receipt_reason_consistency": full.get("receipt_reason_consistency", 0.0),
            "reason_answer_consistency": full.get("reason_answer_consistency", 0.0),
            "evidence_grounded_faithfulness": full.get("evidence_grounded_faithfulness", 0.0),
            "overall_consistency": full.get("overall_consistency", s),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {output_path}  ({len(rows)} rows judged)")


def main():
    # Configure judge to use the local Qwen3-30B server (already up on port 8000)
    set_judge_mode("local")
    os.environ["LOCAL_JUDGE_URL"] = "http://localhost:8000/v1"
    set_judge_model("Qwen/Qwen3-30B-A3B-Instruct-2507")
    set_template_version("v1")
    set_batch_reward_workers(32)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = PROJECT_ROOT / f"data/results/faithfulness_eval/grpo_models/1p5B_v1_nojudge_eval_{ts}"
    print(f"Output root: {out_root}")

    for ds in ("hotpotqa", "musique", "2wiki"):
        print(f"\n=== Inference on {ds} ===")
        summary = run_inference_on(ds, out_root)
        # Evaluator wrote results.jsonl under data/results/<dataset>_v1_<ts2>/
        # Find the most recent one matching this dataset
        results_dir_root = PROJECT_ROOT / "data/results"
        candidates = [p for p in results_dir_root.iterdir()
                      if p.is_dir() and p.name.startswith(f"{ds}_2000_v1_")]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"  WARN: no results dir found for {ds}")
            continue
        results_jsonl = candidates[0] / "results.jsonl"
        if not results_jsonl.exists():
            print(f"  WARN: missing {results_jsonl}")
            continue

        print(f"  inference results -> {results_jsonl}")
        # Now judge scoring
        out_judge = out_root / "traces_with_judge" / f"1.5B_v1_{ds}_temp0.0_results.jsonl"
        add_judge_scores(results_jsonl, out_judge, ds_name=ds)

    print(f"\nAll done. Output: {out_root}")


if __name__ == "__main__":
    main()
