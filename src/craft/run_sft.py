#!/usr/bin/env python3
"""
CRAFT SFT Training Entry Point.

Loads config and runs supervised fine-tuning using ms-swift.
"""

import os
import sys

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from src.train import SFTTrainer


def main():
    # Allow overriding the config name via --config-name (default: 'sft')
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config-name", default="sft")
    args, _unknown = parser.parse_known_args()

    # Create SFT trainer with CRAFT config
    trainer = SFTTrainer(
        config_name=args.config_name,
        project_root=project_root
    )

    # Run training
    return trainer.run()


if __name__ == '__main__':
    main()
