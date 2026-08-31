"""
CRAFT module - Core inference client and utilities.

This module provides:
- CRAFTClient: HTTP client for CRAFT model server
- extracts: Utilities for parsing model outputs
- server: vLLM server wrapper with config support
"""

from src.craft.client import CRAFTClient, check_craft_server_ready, load_craft_client
from src.craft.extracts import extract_answer, extract_support_ids, normalize_answer

__all__ = [
    "CRAFTClient",
    "check_craft_server_ready",
    "load_craft_client",
    "extract_answer",
    "extract_support_ids",
    "normalize_answer",
]
