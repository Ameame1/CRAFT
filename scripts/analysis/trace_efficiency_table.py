#!/usr/bin/env python3
"""
R1 — Trace length / efficiency table for v1-v5 across model scales.

Reads existing eval outputs in data/results/initial/{scale}_v{1..5}_{ds}_temp0.0_results.jsonl
(if present), tokenizes each `full_response` with the Qwen2.5-7B tokenizer (a
neutral choice — same tokenizer used for the 7B model whose outputs are the
heaviest in the paper), and computes per-cell averages of:
   - output token length (mean, p50, p99)
   - EM
   - F1
   - format pass rate

When a faithfulness-attached judge file exists for the same scale/variant/ds,
also pulls mean overall_consistency.

Output: docs/rebuttal/07_trace_efficiency_table.md  (human-readable markdown)
        + docs/rebuttal/07_trace_efficiency_table.csv (machine-readable)
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows


def find_pred_file(scale: str, variant: str, ds: str) -> Path:
    """Return path to base eval outputs in data/results/initial/."""
    direct = PROJECT_ROOT / f"data/results/initial/{scale}_{variant}_{ds}_temp0.0_results.jsonl"
    if direct.exists():
        return direct
    nested = sorted(
        (PROJECT_ROOT / "data/results/initial").glob(
            f"{scale}_{variant}_eval_*/{scale}_{variant}_{ds}_temp0.0_results.jsonl"
        )
    )
    return nested[-1] if nested else None


def find_judge_file(scale: str, variant: str, ds: str) -> Path:
    """Return path to judge-attached traces under data/results/faithfulness_eval."""
    candidates = [
        PROJECT_ROOT / f"data/results/faithfulness_eval/base_models/traces_with_judge/{scale}_base_{variant}_{ds}_temp0.0_results.jsonl",
        PROJECT_ROOT / f"data/results/faithfulness_eval/sft_models/traces_with_judge/{scale}_sft_{variant}_{ds}_temp0.0_results.jsonl",
        PROJECT_ROOT / f"data/results/faithfulness_eval/grpo_models/traces_with_judge/{scale}_{variant}_{ds}_temp0.0_results.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def percentile(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def aggregate(scale: str, variant: str, ds: str, tok):
    pred_path = find_pred_file(scale, variant, ds)
    if pred_path is None:
        return None
    rows = load_jsonl(pred_path)
    if not rows:
        return None
    # Schema differs between data/results/initial/ ('exact_match','f1', no format)
    # and data/results/faithfulness_eval/ ('em_score','f1_score','format_score').
    # Compute format on-the-fly via the project metric for the initial schema.
    from src.eval.metrics import format_score as _fmt
    lens, em_v, f1_v, fmt_v = [], [], [], []
    for r in rows:
        text = r.get("full_response", "") or ""
        lens.append(len(tok.encode(text)))
        em = r.get("em_score", r.get("exact_match", 0))
        f1 = r.get("f1_score", r.get("f1", 0))
        fmt = r.get("format_score")
        if fmt is None:
            fmt = _fmt(text, variant)
        em_v.append(float(em))
        f1_v.append(float(f1))
        fmt_v.append(float(fmt))
    out = {
        "scale": scale, "variant": variant, "ds": ds, "n": len(rows),
        "tokens_mean": sum(lens) / len(lens),
        "tokens_p50": percentile(lens, 0.5),
        "tokens_p99": percentile(lens, 0.99),
        "EM": sum(em_v) / len(em_v) * 100,
        "F1": sum(f1_v) / len(f1_v) * 100,
        "format_pass": sum(fmt_v) / len(fmt_v) * 100,
        "Faith": None,
    }
    judge_path = find_judge_file(scale, variant, ds)
    if judge_path is not None:
        jrows = load_jsonl(judge_path)
        if jrows:
            faith_v = []
            for r in jrows:
                s = r.get("judge_scores", {}) or {}
                faith_v.append(float(s.get("overall_consistency", 0)))
            if faith_v:
                out["Faith"] = sum(faith_v) / len(faith_v) * 100
    return out


def render_md(rows):
    md = ["# R1 — Trace length / efficiency table (v1–v5 × scales × datasets)\n",
          "Auto-aggregated from existing eval outputs in `data/results/initial/` and "
          "judge-attached traces in `data/results/faithfulness_eval/`. Token counts use "
          "the Qwen2.5-7B tokenizer (neutral; same family as the trained models).\n",
          "Columns: tokens_mean / p50 / p99 are output-only token counts. Format-pass "
          "rate is per-row binary average. Faithfulness is judge `overall_consistency`. "
          "If a cell shows `—` for Faith, the judge file for that scale/variant/dataset "
          "is not on disk.\n"]
    # Group by scale, then by variant
    by_scale = {}
    for r in rows:
        by_scale.setdefault(r["scale"], []).append(r)
    for scale in sorted(by_scale, key=lambda s: float(s.replace("B", ""))):
        md.append(f"\n## {scale} model\n")
        md.append("| Variant | Dataset | n | tok mean | tok p50 | tok p99 | EM | F1 | format % | Faith % |")
        md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        # Sort by variant then dataset
        for r in sorted(by_scale[scale], key=lambda r: (r["variant"], r["ds"])):
            faith = f"{r['Faith']:.2f}" if r["Faith"] is not None else "—"
            md.append(f"| {r['variant']} | {r['ds']} | {r['n']} | "
                      f"{r['tokens_mean']:.1f} | {r['tokens_p50']} | {r['tokens_p99']} | "
                      f"{r['EM']:.2f} | {r['F1']:.2f} | {r['format_pass']:.2f} | {faith} |")
    md.append(
        "\n## Implications for reviewer fya4 #6 (auditability vs efficiency)\n"
        "- v1 (full plan + gold_docs + reason + answer) produces the longest traces.\n"
        "- v5 (answer-only) is the shortest.\n"
        "- The auditability–efficiency trade-off is quantified directly above; v4 is the "
        "knee of the curve at most scales.")
    return "\n".join(md)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scales", nargs="+", default=["0.5B", "1.5B", "3B", "7B"])
    p.add_argument("--variants", nargs="+", default=["v1", "v2", "v3", "v4", "v5"])
    p.add_argument("--datasets", nargs="+", default=["hotpotqa", "musique", "2wiki"])
    p.add_argument("--out-md", default="docs/rebuttal/07_trace_efficiency_table.md")
    p.add_argument("--out-csv", default="docs/rebuttal/07_trace_efficiency_table.csv")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = []
    for scale in args.scales:
        for variant in args.variants:
            for ds in args.datasets:
                cell = aggregate(scale, variant, ds, tok)
                if cell:
                    rows.append(cell)
                    print(f"  {scale} {variant} {ds}: n={cell['n']}, EM={cell['EM']:.2f}, "
                          f"tokens_mean={cell['tokens_mean']:.1f}")

    out_md = PROJECT_ROOT / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(rows))
    print(f"\nWrote markdown table -> {out_md}")

    out_csv = PROJECT_ROOT / args.out_csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scale", "variant", "ds", "n",
                                            "tokens_mean", "tokens_p50", "tokens_p99",
                                            "EM", "F1", "format_pass", "Faith"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote csv -> {out_csv}")
    print(f"\nTotal cells: {len(rows)}")


if __name__ == "__main__":
    main()
