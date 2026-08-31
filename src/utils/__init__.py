"""
CRAFT Utilities Module.

Provides utility functions for configuration and environment setup.
"""

from src.utils.configs import (
    load_config,
    get_project_root,
    setup_training_env,
    CFG_DIR,
)

__all__ = [
    "load_config",
    "get_project_root",
    "setup_training_env",
    "CFG_DIR",
]
