"""
CRAFT GRPO Trainer.

Provides a reusable GRPO training class that handles environment setup,
config loading, and training with custom reward functions.
"""

import os
import sys


class GRPOTrainer:
    """
    GRPO trainer that handles environment setup, config loading, and training.

    Uses ms-swift for Group Relative Policy Optimization training.
    """

    def __init__(self, config_name: str, reward_func_map: dict, project_root: str = None):
        """
        Initialize GRPO trainer.

        Args:
            config_name: Name of the config file (without .yaml extension)
            reward_func_map: Dictionary mapping reward function names to callables
            project_root: Path to project root (auto-detected if not provided)
        """
        self.config_name = config_name
        self.reward_func_map = reward_func_map

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

    def prepare_reward_functions(self, config: dict) -> dict:
        """
        Extract and prepare reward functions from config.

        Args:
            config: Configuration dictionary

        Returns:
            Updated config with reward functions replaced by callables
        """
        # Extract reward function configuration
        reward_func_names = config.pop('reward_funcs', None)
        reward_weights = config.pop('reward_weights', None)

        # Replace string names with actual callables
        if reward_func_names:
            config['reward_funcs'] = [
                self.reward_func_map[name] for name in reward_func_names
            ]

        if reward_weights:
            config['reward_weights'] = reward_weights

        return config

    def run(self):
        """
        Run GRPO training.

        Returns:
            Training result from rlhf_main
        """
        # Setup environment
        self.setup_environment()

        # Load config
        config = self.load_config()

        # Prepare reward functions
        config = self.prepare_reward_functions(config)

        # Auto-resume from the latest checkpoint in output_dir.
        # Set CRAFT_FRESH_RUN=1 to disable (e.g. starting a new experiment in
        # the same dir on purpose). If output_dir contains checkpoint-N dirs,
        # we pick the one with the largest N and pass it as resume target.
        if not os.environ.get('CRAFT_FRESH_RUN'):
            output_dir = config.get('output_dir')
            if output_dir and os.path.isdir(output_dir):
                ckpts = []
                for entry in os.listdir(output_dir):
                    if entry.startswith('checkpoint-'):
                        try:
                            ckpts.append((int(entry.split('-')[1]), entry))
                        except (ValueError, IndexError):
                            pass
                if ckpts:
                    latest_step, latest_name = max(ckpts)
                    latest_path = os.path.join(output_dir, latest_name)
                    print(f"[GRPOTrainer] Auto-resuming from {latest_path} (step {latest_step})")
                    config['resume_from_checkpoint'] = latest_path

        # Remove unsupported parameters (custom params not in RLHFArguments)
        unsupported_params = ['batch_reward_workers', 'judge_model', 'template_version']
        custom_config = {k: config.pop(k) for k in unsupported_params if k in config}

        # Set batch_reward_workers and judge_model for reward functions
        if 'batch_reward_workers' in custom_config:
            try:
                from src.rewards.craft_rewards import set_batch_reward_workers
                set_batch_reward_workers(custom_config['batch_reward_workers'])
            except ImportError:
                pass

        if 'judge_model' in custom_config:
            try:
                from src.rewards.craft_rewards import set_judge_model
                set_judge_model(custom_config['judge_model'])
            except ImportError:
                pass

        if 'template_version' in custom_config:
            try:
                from src.rewards.craft_rewards import set_template_version
                set_template_version(custom_config['template_version'])
            except ImportError:
                pass

        # Import Swift RLHF (ms-swift 4.x layout)
        from swift.pipelines import rlhf_main
        from swift.arguments import RLHFArguments

        # Create RLHFArguments directly from config
        args = RLHFArguments(**config)

        # Run training
        try:
            result = rlhf_main(args)
            return result
        except KeyboardInterrupt:
            sys.exit(130)
        except Exception as e:
            import traceback
            traceback.print_exc()
            sys.exit(1)
