"""
CRAFT Judge Module.

Provides faithfulness evaluation prompts for different trace versions.
The judge evaluates internal consistency (R_faith) rather than factual correctness.

Available prompts:
- V1: <plan><gold_docs><reason><answer> - 4 consistency metrics
- V2: <gold_docs><reason><answer> - 3 consistency metrics
- V3: <plan><reason><answer> - 3 consistency metrics
- V4: <reason><answer> - 2 consistency metrics
"""

from src.judge.v1_judge_prompt import V1_JUDGE_PROMPT, format_v1_judge_prompt
from src.judge.v2_judge_prompt import V2_JUDGE_PROMPT, format_v2_judge_prompt
from src.judge.v3_judge_prompt import V3_JUDGE_PROMPT, format_v3_judge_prompt
from src.judge.v4_judge_prompt import V4_JUDGE_PROMPT, format_v4_judge_prompt

__all__ = [
    "V1_JUDGE_PROMPT",
    "V2_JUDGE_PROMPT",
    "V3_JUDGE_PROMPT",
    "V4_JUDGE_PROMPT",
    "format_v1_judge_prompt",
    "format_v2_judge_prompt",
    "format_v3_judge_prompt",
    "format_v4_judge_prompt",
]
