#!/usr/bin/env python3
"""
CRAFT Faithfulness Evaluation Script.

Evaluates the faithfulness of CRAFT model outputs using the judge model.
Computes R_faith scores for each trace variant.
"""

import argparse
import json
import os
import sys
from typing import List, Dict

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from src.rewards.craft_rewards import (
    judge_overall_consistency_reward,
    set_judge_model,
    set_template_version,
    set_batch_reward_workers,
)


def load_predictions(filepath: str) -> List[Dict]:
    """Load predictions from JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def evaluate_faithfulness(
    predictions: List[Dict],
    template_version: str = 'v1',
    batch_size: int = 32
) -> Dict:
    """
    Evaluate faithfulness of predictions.

    Args:
        predictions: List of prediction dicts with 'question', 'documents', 'prediction' keys
        template_version: Template version used
        batch_size: Number of predictions to evaluate at once

    Returns:
        Dict with faithfulness scores
    """
    questions = [p['question'] for p in predictions]
    documents = [p['documents'] for p in predictions]
    completions = [p['prediction'] for p in predictions]

    # Evaluate in batches
    all_scores = []
    for i in range(0, len(predictions), batch_size):
        batch_q = questions[i:i+batch_size]
        batch_d = documents[i:i+batch_size]
        batch_c = completions[i:i+batch_size]

        scores = judge_overall_consistency_reward(
            completions=batch_c,
            question=batch_q,
            documents=batch_d,
            template_version=template_version
        )
        all_scores.extend(scores)

        print(f"Processed {min(i+batch_size, len(predictions))}/{len(predictions)} predictions")

    # Calculate statistics
    avg_score = sum(all_scores) / len(all_scores)
    pass_rate = sum(1 for s in all_scores if s == 1.0) / len(all_scores)

    return {
        'average_faithfulness': avg_score,
        'pass_rate': pass_rate,
        'num_samples': len(all_scores),
        'scores': all_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="CRAFT Faithfulness Evaluation")
    parser.add_argument(
        "--predictions", "-p",
        type=str,
        required=True,
        help="Path to predictions JSONL file"
    )
    parser.add_argument(
        "--template-version", "-t",
        type=str,
        default="v1",
        choices=["v1", "v2", "v3", "v4"],
        help="Template version used"
    )
    parser.add_argument(
        "--judge-model", "-m",
        type=str,
        default="Qwen3-30B-A3B-Instruct",
        help="Judge model name"
    )
    parser.add_argument(
        "--judge-url", "-u",
        type=str,
        default="http://localhost:8000/v1",
        help="Judge server URL"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (optional)"
    )
    args = parser.parse_args()

    # Set judge configuration
    os.environ['LOCAL_JUDGE_URL'] = args.judge_url
    set_judge_model(args.judge_model)
    set_template_version(args.template_version)
    set_batch_reward_workers(args.batch_size)

    print("=" * 60)
    print("CRAFT Faithfulness Evaluation")
    print("=" * 60)
    print(f"Predictions: {args.predictions}")
    print(f"Template version: {args.template_version}")
    print(f"Judge model: {args.judge_model}")
    print(f"Judge URL: {args.judge_url}")
    print("=" * 60)

    # Load predictions
    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} predictions")

    # Evaluate
    results = evaluate_faithfulness(
        predictions,
        template_version=args.template_version,
        batch_size=args.batch_size
    )

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Average Faithfulness (R_faith): {results['average_faithfulness']:.4f}")
    print(f"Pass Rate (all metrics = 1): {results['pass_rate']:.4f}")
    print(f"Number of samples: {results['num_samples']}")

    # Save results if output specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
