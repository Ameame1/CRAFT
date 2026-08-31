"""
Compare base vs GRPO for a given version: bar charts + scatter plots across datasets.

Usage:
    python compare_base_grpo.py --version v2
    python compare_base_grpo.py --version v3
    python compare_base_grpo.py --version v4
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


BASE_DIR = Path(__file__).resolve().parents[2] / "data/results/faithfulness_eval"

METRICS = ["em", "f1", "faithfulness"]
LABELS  = {"em": "Exact Match", "f1": "F1", "faithfulness": "Faithfulness"}
COLORS  = {"Base": "#4878d0", "GRPO": "#ee854a"}
PAIRS   = [("em", "faithfulness"), ("f1", "faithfulness"), ("em", "f1")]
DATASETS = ["musique", "hotpotqa", "2wiki"]


def build_file_map(version: str):
    v = version  # e.g. "v2"
    return {
        ds: {
            "Base": BASE_DIR / f"base_models/traces_with_judge/7B_base_{v}_{ds}_temp0.0_results.jsonl",
            "GRPO": BASE_DIR / f"grpo_models/traces_with_judge/7B_{v}_{ds}_temp0.0_results.jsonl",
        }
        for ds in DATASETS
    }


def load_df(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            judge = obj.get("judge_scores", {})
            rows.append({
                "em":           obj.get("exact_match", np.nan),
                "f1":           obj.get("f1", np.nan),
                "faithfulness": judge.get("overall_consistency", np.nan),
            })
    return pd.DataFrame(rows)


def pearson(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return np.nan, np.nan
    return stats.pearsonr(x[mask], y[mask])


def jitter(arr, scale=0.025):
    return arr + np.random.uniform(-scale, scale, size=len(arr))


def run(version: str):
    files = build_file_map(version)

    # Load data & print summary
    data = {}
    print(f"\n{'='*70}")
    print(f"7B {version}  —  Base vs GRPO")
    print(f"{'='*70}")
    for ds in DATASETS:
        data[ds] = {}
        for model in ["Base", "GRPO"]:
            df = load_df(files[ds][model])
            data[ds][model] = df
            r_em, _  = pearson(df.em.values, df.faithfulness.values)
            r_f1, _  = pearson(df.f1.values, df.faithfulness.values)
            print(f"  {ds:10s} {model:4s}  n={len(df)}  "
                  f"EM={df.em.mean():.3f}  F1={df.f1.mean():.3f}  "
                  f"Faith={df.faithfulness.mean():.3f}  "
                  f"r(EM,Faith)={r_em:.3f}  r(F1,Faith)={r_f1:.3f}")

    # Plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f"7B {version}  —  Base vs GRPO: Metric Means & Scatter Plots",
                 fontsize=14, fontweight="bold")

    for row_idx, ds in enumerate(DATASETS):
        # Col 0: bar chart
        ax = axes[row_idx][0]
        x = np.arange(len(METRICS))
        width = 0.35
        for m_idx, model in enumerate(["Base", "GRPO"]):
            df = data[ds][model]
            means = [df[m].mean() for m in METRICS]
            sems  = [df[m].sem()  for m in METRICS]
            bars = ax.bar(x + m_idx * width, means, width, yerr=sems,
                          label=model, color=COLORS[model], alpha=0.85,
                          capsize=3, error_kw={"linewidth": 1})
            for bar, mv in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{mv:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([LABELS[m] for m in METRICS], fontsize=8)
        ax.set_ylabel(f"{ds}\nMean Score", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.tick_params(labelsize=7)
        if row_idx == 0:
            ax.set_title("Mean Metrics (±SEM)", fontsize=9, fontweight="bold")
            ax.legend(fontsize=8)

        # Cols 1-3: scatter plots
        for col_idx, (xm, ym) in enumerate(PAIRS, start=1):
            ax = axes[row_idx][col_idx]
            for model in ["Base", "GRPO"]:
                df = data[ds][model]
                xv = df[xm].values.astype(float)
                yv = df[ym].values.astype(float)
                xp = jitter(xv) if df[xm].nunique() <= 5 else xv
                yp = jitter(yv) if df[ym].nunique() <= 5 else yv
                ax.scatter(xp, yp, alpha=0.2, s=10, color=COLORS[model], label=model)

                mask = ~(np.isnan(xv) | np.isnan(yv))
                if mask.sum() >= 3:
                    m_val, b = np.polyfit(xv[mask], yv[mask], 1)
                    xr = np.linspace(np.nanmin(xv), np.nanmax(xv), 100)
                    ax.plot(xr, m_val * xr + b, color=COLORS[model],
                            linewidth=1.5, linestyle="--")

                r, p = pearson(xv, yv)
                p_str = "p<.001" if p < 0.001 else f"p={p:.2f}"
                y_pos = 0.93 if model == "Base" else 0.80
                ax.text(0.05, y_pos, f"{model}: r={r:.2f} ({p_str})",
                        transform=ax.transAxes, fontsize=6.5, color=COLORS[model],
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))

            ax.set_xlabel(LABELS[xm], fontsize=8)
            ax.set_ylabel(LABELS[ym], fontsize=8)
            ax.tick_params(labelsize=7)
            if row_idx == 0:
                ax.set_title(f"{LABELS[xm]} vs {LABELS[ym]}", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = BASE_DIR / f"grpo_models/traces_with_judge/7B_{version}_base_vs_grpo_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "-v", required=True, help="e.g. v2, v3, v4")
    args = parser.parse_args()
    run(args.version)


if __name__ == "__main__":
    main()
