#!/usr/bin/env python3
"""
CRAFT Evaluation Entry Point.

Runs evaluation on a test dataset using the CRAFT model.
"""

import os
import sys

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from src.eval import Evaluator


def main():
    """Run CRAFT evaluation."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config-name', default='eval')
    # Accept --dataset, --template_version, --model for backward-compat;
    # they are no-ops here unless you also write a temp config that points to them.
    p.add_argument('--dataset', default=None)
    p.add_argument('--template_version', default=None)
    p.add_argument('--model', default=None)
    args = p.parse_args()
    evaluator = Evaluator(
        config_name=args.config_name,
        project_root=project_root,
        component_type="craft"
    )
    return evaluator.run()


if __name__ == '__main__':
    main()
