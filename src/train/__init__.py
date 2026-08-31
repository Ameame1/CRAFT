"""
CRAFT Training Module.

Provides training utilities for CRAFT models:
- SFTTrainer: Supervised fine-tuning trainer
- GRPOTrainer: Group Relative Policy Optimization trainer
"""

from src.train.sft_trainer import SFTTrainer
from src.train.grpo_trainer import GRPOTrainer

__all__ = [
    "SFTTrainer",
    "GRPOTrainer",
]
