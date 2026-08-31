#!/usr/bin/env python3
"""
Stage E — eval the trained RS-CRAFT 7B model on hotpotqa / musique / 2wiki,
then attach judge faithfulness scores.

Output dir: data/results/faithfulness_eval/grpo_models/rs_craft_7b_v1_eval_<ts>/
            traces_with_judge/RS-CRAFT-7B-v1_<ds>_temp0.0_results.jsonl
"""
import json, os, re, sys, time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("VLLM_HOST", "http://localhost")
os.environ.setdefault("VLLM_PORT", "8002")
os.environ["JUDGE_MODE"] = "local"
os.environ["LOCAL_JUDGE_URL"] = "http://localhost:8000/v1"

from src.eval.evaluator import Evaluator
from src.rewards.craft_rewards import (
    set_judge_mode, set_judge_model, set_template_version,
    set_batch_reward_workers, judge_overall_consistency_reward,
    _judge_cache,
)


def run_inference_on(dataset: str):
    import yaml
    base = yaml.safe_load(open(PROJECT_ROOT / "cfg/eval.yaml"))
    base["model"] = "RS-CRAFT-7B-v1"
    base["val_dataset"] = f"data/test/{dataset}_2000_v1.jsonl"
    base["template_version"] = "v1"
    base["vllm_port"] = 8002
    tmp_cfg = PROJECT_ROOT / "cfg/_eval_rscraft_7b_tmp.yaml"
    with open(tmp_cfg, "w") as f:
        yaml.safe_dump(base, f)
    e = Evaluator(config_name="_eval_rscraft_7b_tmp", project_root=str(PROJECT_ROOT))
    return e.run()


def add_judge_scores(results_jsonl: Path, out_path: Path, ds_name: str):
    rows = [json.loads(l) for l in open(results_jsonl) if l.strip()]
    print(f"  Judging {len(rows)} rows for {ds_name} ...")
    test_path = PROJECT_ROOT / f"data/test/{ds_name}_2000_v1.jsonl"
    test_rows = [json.loads(l) for l in open(test_path) if l.strip()]
    test_by_idx = {i: r for i, r in enumerate(test_rows)}

    def _extract(prompt_messages):
        user_content = ""
        for m in prompt_messages:
            if m.get("role") == "user":
                user_content = m.get("content", "")
        q = re.search(r"<query>(.*?)</query>", user_content, re.DOTALL)
        d = re.search(r"<docs>(.*?)</docs>", user_content, re.DOTALL)
        return (q.group(1).strip() if q else ""), (d.group(1).strip() if d else "")

    questions, documents, completions = [], [], []
    for r in rows:
        idx = r.get("idx")
        tr = test_by_idx.get(idx)
        q, d = _extract(tr["prompt"]) if tr else ("", "")
        questions.append(q); documents.append(d)
        completions.append(r.get("full_response", ""))

    scores = judge_overall_consistency_reward(
        completions=completions, question=questions, documents=documents,
        template_version="v1",
    )
    for r, q, d, c, s in zip(rows, questions, documents, completions, scores):
        ck = f"{q}::{d}::{c}::v1"
        full = _judge_cache.get(ck, {})
        r["judge_scores"] = {
            "plan_reason_consistency": full.get("plan_reason_consistency", 0.0),
            "receipt_reason_consistency": full.get("receipt_reason_consistency", 0.0),
            "reason_answer_consistency": full.get("reason_answer_consistency", 0.0),
            "evidence_grounded_faithfulness": full.get("evidence_grounded_faithfulness", 0.0),
            "overall_consistency": full.get("overall_consistency", s),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {out_path} ({len(rows)} rows judged)")


def main():
    set_judge_mode("local")
    set_judge_model("Qwen/Qwen3-30B-A3B-Instruct-2507")
    set_template_version("v1")
    set_batch_reward_workers(32)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = PROJECT_ROOT / f"data/results/faithfulness_eval/grpo_models/rs_craft_7b_v1_eval_{ts}"
    print(f"Output root: {out_root}")

    summaries = {}
    for ds in ("hotpotqa", "musique", "2wiki"):
        print(f"\n=== Inference on {ds} ===")
        summary = run_inference_on(ds)
        summaries[ds] = summary
        # find newest matching results dir
        results_root = PROJECT_ROOT / "data/results"
        candidates = sorted(
            (p for p in results_root.iterdir()
             if p.is_dir() and p.name.startswith(f"{ds}_2000_v1_")),
            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"  WARN: no results dir for {ds}")
            continue
        results_jsonl = candidates[0] / "results.jsonl"
        if not results_jsonl.exists():
            print(f"  WARN: missing {results_jsonl}")
            continue
        out_judge = out_root / "traces_with_judge" / f"RS-CRAFT-7B-v1_{ds}_temp0.0_results.jsonl"
        add_judge_scores(results_jsonl, out_judge, ds_name=ds)

    print(f"\nDone. Output: {out_root}")
    print("\nPer-dataset (eval-time) summary:")
    for ds, s in summaries.items():
        print(f"  {ds}: EM={s.get('em_score',0)*100:5.2f}  F1={s.get('f1_score',0)*100:5.2f}  Format={s.get('format_score',0)*100:5.2f}  Rel={s.get('relevance_score',0)*100:5.2f}")


if __name__ == "__main__":
    main()
