"""
CRAFT Templates Module.

Provides prompt templates for different CRAFT trace variants (v1-v5).
"""

from src.templates.craft_templates import (
    CRAFT_TEMPLATES,
    get_template,
    prompt_template_v1,
    prompt_template_v2,
    prompt_template_v3,
    prompt_template_v4,
    prompt_template_v5,
)

__all__ = [
    "CRAFT_TEMPLATES",
    "get_template",
    "prompt_template_v1",
    "prompt_template_v2",
    "prompt_template_v3",
    "prompt_template_v4",
    "prompt_template_v5",
]
