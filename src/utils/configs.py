"""
CRAFT Configuration Utilities.

Provides functions for loading YAML configs and setting up the training environment.
"""

import os
import sys
from typing import Any, Dict

import yaml

# Get project root (craft/ directory)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
CFG_DIR = os.path.join(PROJECT_ROOT, "cfg")


def load_config(config_path: str = "default_config") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Config file name (without .yaml extension)

    Returns:
        Configuration dictionary, or empty dict if not found
    """
    # Try both .yaml and .yml extensions
    for ext in [".yaml", ".yml"]:
        full_path = os.path.join(CFG_DIR, f"{config_path}{ext}")
        if os.path.exists(full_path):
            try:
                with open(full_path, "r") as config_file:
                    return yaml.safe_load(config_file)
            except yaml.YAMLError as e:
                print(f"Error parsing config file: {e}")
                return {}

    # If neither extension works
    print(f"Warning: Config file not found at {CFG_DIR}/{config_path}.[yaml|yml]")
    return {}


def get_project_root() -> str:
    """Get the project root directory."""
    return PROJECT_ROOT


def setup_training_env(project_root: str = None):
    """
    Set up common environment variables for training.

    Args:
        project_root: Path to project root (uses default if not provided)
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    os.environ['PYTHONPATH'] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"
    os.environ.setdefault('VLLM_ATTENTION_BACKEND', 'FLASH_ATTN')
    os.environ.setdefault('VLLM_USE_TRITON_FLASH_ATTN', '0')

    # Add project root to sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
