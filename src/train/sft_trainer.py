"""
CRAFT SFT Trainer.

Provides a reusable SFT training class that handles environment setup,
config loading, and training using ms-swift.
"""

import os
import sys


class SFTTrainer:
    """
    SFT trainer that handles environment setup, config loading, and training.

    Uses ms-swift for supervised fine-tuning.
    """

    def __init__(self, config_name: str, project_root: str = None):
        """
        Initialize SFT trainer.

        Args:
            config_name: Name of the config file (without .yaml extension)
            project_root: Path to project root (auto-detected if not provided)
        """
        self.config_name = config_name

        # Auto-detect project root if not provided
        if project_root is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.dirname(os.path.dirname(script_dir))
        else:
            self.project_root = project_root

    def setup_environment(self):
        """Set up training environment variables and paths."""
        from src.utils import setup_training_env
        setup_training_env(self.project_root)

    def load_config(self):
        """Load training configuration from YAML file."""
        from src.utils import load_config
        config = load_config(self.config_name)
        if not config:
            raise ValueError(f"Failed to load config: {self.config_name}")
        return config

    def run(self):
        """
        Run SFT training.

        Returns:
            Training result from sft_main
        """
        # Setup environment
        self.setup_environment()

        # Load config
        config = self.load_config()

        # Import Swift SFT (ms-swift 4.x layout: TrainArguments -> SftArguments)
        from swift.pipelines import sft_main
        from swift.arguments import SftArguments

        # Create SftArguments directly from config
        args = SftArguments(**config)

        # Run training
        try:
            result = sft_main(args)
            return result
        except KeyboardInterrupt:
            sys.exit(130)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
