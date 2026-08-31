#!/usr/bin/env python3
"""
CRAFT Judge Server - vLLM deploy wrapper for faithfulness evaluation.

This server hosts the judge model (e.g., Qwen3-30B-A3B) for computing
the R_faith (faithfulness) reward during GRPO training.

Usage:
    python src/judge/server.py
    python src/judge/server.py --config judge
    python src/judge/server.py --port 8000 --model Qwen/Qwen3-30B-A3B-Instruct
"""

import argparse
import os
import subprocess
import sys

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from src.utils import load_config


def build_vllm_command(config: dict) -> list:
    """
    Build the vLLM serve command from config.

    Args:
        config: Configuration dictionary

    Returns:
        List of command arguments
    """
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]

    # Model path
    model = config.get("model")
    if model:
        # Make relative paths absolute
        if not model.startswith("/") and "/" not in model:
            model = os.path.join(project_root, model)
        cmd.extend(["--model", model])

    # Server settings
    if config.get("port"):
        cmd.extend(["--port", str(config["port"])])
    if config.get("host"):
        cmd.extend(["--host", config["host"]])

    # Model settings
    if config.get("torch_dtype"):
        cmd.extend(["--dtype", config["torch_dtype"]])

    # vLLM settings
    if config.get("tensor_parallel_size"):
        cmd.extend(["--tensor-parallel-size", str(config["tensor_parallel_size"])])
    if config.get("max_model_len"):
        cmd.extend(["--max-model-len", str(config["max_model_len"])])
    if config.get("max_num_seqs"):
        cmd.extend(["--max-num-seqs", str(config["max_num_seqs"])])
    if config.get("gpu_memory_utilization"):
        cmd.extend(["--gpu-memory-utilization", str(config["gpu_memory_utilization"])])
    if config.get("enable_prefix_caching"):
        cmd.append("--enable-prefix-caching")
    if config.get("enforce_eager") is False:
        pass  # CUDA graphs enabled by default
    elif config.get("enforce_eager"):
        cmd.append("--enforce-eager")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="CRAFT Judge Server")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="judge",
        help="Config file name (without .yaml extension)"
    )
    parser.add_argument("--port", type=int, help="Override server port")
    parser.add_argument("--model", type=str, help="Override model path")
    parser.add_argument("--cuda-devices", type=str, help="Override CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if not config:
        print(f"Error: Could not load config '{args.config}'")
        sys.exit(1)

    # Override with command line args
    if args.port:
        config["port"] = args.port
    if args.model:
        config["model"] = args.model

    # Set CUDA_VISIBLE_DEVICES
    cuda_devices = args.cuda_devices or config.get("cuda_visible_devices")
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    # Set environment variables
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

    # Build and run command
    cmd = build_vllm_command(config)

    print("=" * 60)
    print("Starting CRAFT Judge Server")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Model: {config.get('model')}")
    print(f"Port: {config.get('port')}")
    print(f"CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running vLLM server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
