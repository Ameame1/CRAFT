"""
CRAFT Evaluation Module.

Provides evaluation utilities for CRAFT models:
- metrics: Scoring functions (format, accuracy, relevance, EM, F1)
- evaluator: Main evaluator class for running evaluation
"""

from src.eval.metrics import (
    normalize_answer,
    exact_match_score,
    f1_score,
    format_score,
    accuracy_score,
    relevance_score,
    extract_answer,
    extract_gold_docs,
    get_all_scores,
)
from src.eval.evaluator import Evaluator

__all__ = [
    # Metrics
    "normalize_answer",
    "exact_match_score",
    "f1_score",
    "format_score",
    "accuracy_score",
    "relevance_score",
    "extract_answer",
    "extract_gold_docs",
    "get_all_scores",
    # Evaluator
    "Evaluator",
]
