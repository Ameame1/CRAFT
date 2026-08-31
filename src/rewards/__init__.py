"""
CRAFT Rewards Module.

Provides reward functions for GRPO training:
- R_fmt (format_reward): Format compliance
- R_ans (accuracy_reward): Answer correctness
- R_gold (relevance_reward): Citation accuracy
- R_faith (judge_overall_consistency_reward): Faithfulness audit
"""

from src.rewards.craft_rewards import (
    # Scoring functions
    format_score,
    accuracy_score,
    relevance_score,
    # Reward functions
    format_reward,
    accuracy_reward,
    relevance_reward,
    judge_overall_consistency_reward,
    # Configuration
    set_judge_model,
    get_judge_model,
    set_template_version,
    get_template_version,
    set_batch_reward_workers,
    clear_judge_cache,
)

__all__ = [
    # Scoring functions
    "format_score",
    "accuracy_score",
    "relevance_score",
    # Reward functions (Paper notation: R_fmt, R_ans, R_gold, R_faith)
    "format_reward",           # R_fmt
    "accuracy_reward",         # R_ans
    "relevance_reward",        # R_gold
    "judge_overall_consistency_reward",  # R_faith
    # Configuration
    "set_judge_model",
    "get_judge_model",
    "set_template_version",
    "get_template_version",
    "set_batch_reward_workers",
    "clear_judge_cache",
]
