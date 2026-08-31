#!/usr/bin/env python3
"""
CRAFT GRPO Training Entry Point.

Loads config and runs Group Relative Policy Optimization (GRPO) training.
"""

import os
import sys

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from src.train import GRPOTrainer
from src.rewards.craft_rewards import (
    format_reward,       # R_fmt: Format compliance
    accuracy_reward,     # R_ans: Answer correctness
    relevance_reward,    # R_gold: Citation accuracy
    judge_overall_consistency_reward,  # R_faith: Faithfulness audit
)


def main():
    # Allow overriding the config name via --config-name (default: 'grpo')
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config-name", default="grpo")
    args, _unknown = parser.parse_known_args()

    # Map reward function names to actual functions
    # Paper notation: R_fmt, R_ans, R_gold, R_faith
    reward_func_map = {
        'format_reward': format_reward,
        'accuracy_reward': accuracy_reward,
        'relevance_reward': relevance_reward,
        'judge_overall_consistency_reward': judge_overall_consistency_reward,
    }

    # Create GRPO trainer with CRAFT config and rewards
    trainer = GRPOTrainer(
        config_name=args.config_name,
        reward_func_map=reward_func_map,
        project_root=project_root
    )

    # Run training
    return trainer.run()


if __name__ == '__main__':
    main()
