"""
Correlation plot for EM, F1, and faithfulness metrics from judge result JSONL files.

Usage:
    python correlation_plot.py <path_to_jsonl> [--output <output_path>]
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats


def load_data(jsonl_path: str) -> pd.DataFrame:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            judge = obj.get("judge_scores", {})
            records.append({
                "em": obj.get("exact_match", np.nan),
                "f1": obj.get("f1", np.nan),
                "faithfulness": judge.get("overall_consistency", np.nan),
                "evidence_faithfulness": judge.get("evidence_grounded_faithfulness", np.nan),
                "plan_reason": judge.get("plan_reason_consistency", np.nan),
                "reason_answer": judge.get("reason_answer_consistency", np.nan),
            })
    return pd.DataFrame(records)


def add_jitter(arr, scale=0.03):
    return arr + np.random.uniform(-scale, scale, size=len(arr))


def pearson_annotation(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return "n/a"
    r, p = stats.pearsonr(x[mask], y[mask])
    p_str = f"p<0.001" if p < 0.001 else f"p={p:.3f}"
    return f"r={r:.3f}, {p_str}"


def plot_correlations(df: pd.DataFrame, output_path: str, title_prefix: str = ""):
    metrics = ["em", "f1", "faithfulness"]
    labels = {"em": "Exact Match", "f1": "F1", "faithfulness": "Faithfulness\n(Overall Consistency)"}
    n = len(metrics)

    fig, axes = plt.subplots(n, n, figsize=(12, 11))
    fig.suptitle(f"{title_prefix}Metric Correlation Matrix", fontsize=14, fontweight="bold", y=1.01)

    cmap = plt.cm.RdYlGn

    for i, row_metric in enumerate(metrics):
        for j, col_metric in enumerate(metrics):
            ax = axes[i][j]

            if i == j:
                # Diagonal: histogram / KDE
                vals = df[row_metric].dropna()
                if vals.nunique() <= 3:
                    # discrete — bar chart
                    counts = vals.value_counts().sort_index()
                    ax.bar(counts.index.astype(str), counts.values, color="#5b9bd5", edgecolor="white", linewidth=0.5)
                    ax.set_xlabel("")
                    ax.set_ylabel("Count")
                else:
                    ax.hist(vals, bins=20, color="#5b9bd5", edgecolor="white", linewidth=0.5)
                ax.set_title(labels[row_metric], fontsize=9, fontweight="bold")
                ax.tick_params(labelsize=7)

            else:
                x_vals = df[col_metric].values.astype(float)
                y_vals = df[row_metric].values.astype(float)

                # Add jitter for discrete metrics
                x_plot = add_jitter(x_vals, 0.025) if df[col_metric].nunique() <= 5 else x_vals
                y_plot = add_jitter(y_vals, 0.025) if df[row_metric].nunique() <= 5 else y_vals

                sc = ax.scatter(x_plot, y_plot, alpha=0.35, s=18, c=y_vals, cmap=cmap,
                                vmin=0, vmax=1, edgecolors="none")

                # Correlation annotation
                annot = pearson_annotation(x_vals, y_vals)
                ax.text(0.05, 0.93, annot, transform=ax.transAxes,
                        fontsize=7, va="top", color="#333333",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

                # Regression line
                mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                if mask.sum() >= 3:
                    m, b = np.polyfit(x_vals[mask], y_vals[mask], 1)
                    xr = np.linspace(np.nanmin(x_vals), np.nanmax(x_vals), 100)
                    ax.plot(xr, m * xr + b, color="#e63946", linewidth=1.2, linestyle="--", alpha=0.8)

                ax.tick_params(labelsize=7)

            # Row/column axis labels on edges
            if j == 0:
                ax.set_ylabel(labels[row_metric], fontsize=8)
            if i == n - 1:
                ax.set_xlabel(labels[col_metric], fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    plt.close()


def print_summary(df: pd.DataFrame):
    metrics = ["em", "f1", "faithfulness"]
    print("\n--- Summary Statistics ---")
    print(df[metrics].describe().round(3).to_string())
    print("\n--- Pearson Correlation Matrix ---")
    corr = df[metrics].corr(method="pearson")
    print(corr.round(3).to_string())
    print()


def main():
    parser = argparse.ArgumentParser(description="Correlation plot for EM, F1, Faithfulness metrics")
    parser.add_argument("jsonl", help="Path to the results JSONL file")
    parser.add_argument("--output", "-o", default=None, help="Output image path (default: <jsonl_stem>_correlation.png)")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if args.output:
        output_path = args.output
    else:
        output_path = str(jsonl_path.parent / (jsonl_path.stem + "_correlation.png"))

    df = load_data(str(jsonl_path))
    print(f"Loaded {len(df)} records from {jsonl_path.name}")

    title_prefix = jsonl_path.stem.replace("_results", "").replace("_", " ").title() + " — "
    print_summary(df)
    plot_correlations(df, output_path, title_prefix=title_prefix)


if __name__ == "__main__":
    main()
