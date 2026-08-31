"""
CRAFT Evaluator - Main evaluation class.

Runs evaluation on test datasets using the CRAFT model via vLLM server.
Supports template versions v1-v4 with concurrent inference.
"""

import os
import sys
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Tuple

import requests
from openai import OpenAI

from src.utils import load_config
from src.eval.metrics import get_all_scores, extract_answer


class Evaluator:
    """
    General evaluator for CRAFT models.

    Supports evaluation with different configurations and template versions (v1-v4).
    Connects to a running vLLM server for inference.
    """

    def __init__(
        self,
        config_name: str = "eval",
        project_root: Optional[str] = None,
        component_type: str = "craft"
    ):
        """
        Initialize evaluator.

        Args:
            config_name: Name of the config file (without .yaml extension)
            project_root: Path to project root (auto-detected if not provided)
            component_type: Type of component being evaluated ("craft")
        """
        self.config_name = config_name
        self.component_type = component_type

        # Auto-detect project root if not provided
        if project_root is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.dirname(os.path.dirname(script_dir))
        else:
            self.project_root = project_root

        self.client: Optional[OpenAI] = None
        self.model_name: Optional[str] = None

    def setup_environment(self):
        """Set up evaluation environment variables and paths."""
        os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
        os.environ['VLLM_USE_TRITON_FLASH_ATTN'] = '0'

        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)

    def load_config(self) -> Dict[str, Any]:
        """Load evaluation configuration from YAML file."""
        config = load_config(self.config_name)
        if not config:
            raise ValueError(f"Failed to load config: {self.config_name}")
        return config

    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load test dataset from JSONL file.

        Args:
            dataset_path: Path to dataset file

        Returns:
            List of data items. Each item is expected to have:
                - prompt: list of message dicts (chat format)
                - answers: list of correct answers
                - id: sample identifier
                - supporting_ids: list of supporting document indices (optional)
        """
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def check_server(self, host: str, port: int) -> bool:
        """Check if vLLM server is running and responsive."""
        try:
            resp = requests.get(f"{host}:{port}/v1/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def setup_client(self, host: str, port: int, model: str) -> None:
        """
        Initialize the OpenAI-compatible client and resolve model name.

        Args:
            host: vLLM server host (e.g., "http://localhost")
            port: vLLM server port
            model: Model name/path from config (used if server auto-detection fails)
        """
        base_url = f"{host}:{port}/v1"
        self.client = OpenAI(base_url=base_url, api_key="not-needed")

        # Auto-detect model name from server
        try:
            models = self.client.models.list()
            if models.data:
                self.model_name = models.data[0].id
            else:
                self.model_name = model
        except Exception:
            self.model_name = model

        print(f"Connected to vLLM server at {base_url}")
        print(f"Using model: {self.model_name}")

    def make_request(
        self,
        prompt_messages: List[Dict],
        temperature: float,
        max_new_tokens: int,
        top_p: float,
        stop_words: List[str],
        max_retries: int = 3,
    ) -> str:
        """
        Send a single chat completion request to the vLLM server.

        Args:
            prompt_messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stop_words: Stop sequences
            max_retries: Number of retries on failure

        Returns:
            Generated response text, or empty string on failure
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt_messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    top_p=top_p,
                    stop=stop_words if stop_words else None,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Request failed after {max_retries} retries: {e}")
                    return ""
                time.sleep(1)
        return ""

    def process_sample(
        self,
        sample_idx: int,
        sample: Dict,
        temperature: float,
        max_new_tokens: int,
        top_p: float,
        stop_words: List[str],
        template_version: str,
    ) -> Dict:
        """
        Process a single dataset sample: run inference and compute scores.

        Args:
            sample_idx: Index of the sample (for ordering results)
            sample: Dataset sample dict
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            top_p: Top-p parameter
            stop_words: Stop sequences
            template_version: Template version (v1-v4)

        Returns:
            Result dict with scores and raw outputs
        """
        prompt_messages = sample['prompt']
        response = self.make_request(
            prompt_messages, temperature, max_new_tokens, top_p, stop_words
        )

        answers = sample.get('answers', [])
        supporting_ids = sample.get('supporting_ids', [])

        # Compute all metrics
        scores = get_all_scores(response, answers, supporting_ids, template_version)
        pred_answer = extract_answer(response)

        return {
            'idx': sample_idx,
            'id': sample.get('id', sample_idx),
            'ground_truth': answers,
            'supporting_ids': supporting_ids,
            'prediction': pred_answer,
            'full_response': response,
            'em_score': scores['em_score'],
            'f1_score': scores['f1_score'],
            'format_score': scores['format_score'],
            'accuracy_score': scores['accuracy_score'],
            'relevance_score': scores['relevance_score'],
        }

    def run_inference(
        self,
        data: List[Dict],
        temperature: float,
        max_new_tokens: int,
        top_p: float,
        stop_words: List[str],
        template_version: str,
        max_workers: int = 32,
        batch_size: int = 100,
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Run concurrent inference on the full dataset.

        Args:
            data: List of dataset samples
            temperature: Sampling temperature
            max_new_tokens: Max tokens to generate
            top_p: Top-p parameter
            stop_words: Stop sequences
            template_version: Template version (v1-v4)
            max_workers: Number of concurrent requests
            batch_size: Samples per progress-reporting batch

        Returns:
            Tuple of (results list, aggregate metrics dict)
        """
        results = []
        totals = {
            'em_score': 0.0,
            'f1_score': 0.0,
            'format_score': 0.0,
            'accuracy_score': 0.0,
            'relevance_score': 0.0,
        }
        start_time = time.time()
        num_batches = (len(data) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(data))
            batch = data[batch_start:batch_end]

            batch_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.process_sample,
                        batch_start + i,
                        sample,
                        temperature,
                        max_new_tokens,
                        top_p,
                        stop_words,
                        template_version,
                    ): i
                    for i, sample in enumerate(batch)
                }
                for future in as_completed(futures):
                    batch_results.append(future.result())

            batch_results.sort(key=lambda x: x['idx'])
            results.extend(batch_results)

            for r in batch_results:
                for k in totals:
                    totals[k] += r[k]

            processed = batch_end
            elapsed = time.time() - start_time
            throughput = processed / elapsed if elapsed > 0 else 0
            print(
                f"  [{processed}/{len(data)}] "
                f"EM: {totals['em_score']/processed:.4f} | "
                f"F1: {totals['f1_score']/processed:.4f} | "
                f"Format: {totals['format_score']/processed:.4f} | "
                f"Acc: {totals['accuracy_score']/processed:.4f} | "
                f"Rel: {totals['relevance_score']/processed:.4f} | "
                f"Speed: {throughput:.1f} samp/s"
            )

        n = len(data)
        avg = {k: totals[k] / n for k in totals}
        return results, avg

    def save_results(
        self,
        results: List[Dict],
        output_path: str,
        summary: Dict,
    ) -> None:
        """
        Save per-sample results and summary to disk.

        Args:
            results: List of per-sample result dicts
            output_path: Directory to save results
            summary: Aggregate metrics dict
        """
        os.makedirs(output_path, exist_ok=True)

        results_file = os.path.join(output_path, 'results.jsonl')
        with open(results_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f"Per-sample results saved to: {results_file}")

        summary_file = os.path.join(output_path, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved to: {summary_file}")

    def run(self) -> Dict[str, Any]:
        """
        Run evaluation: load data, run vLLM inference, compute metrics, save results.

        Returns:
            Dictionary with evaluation results and aggregate metrics
        """
        self.setup_environment()

        config = self.load_config()

        template_version = config.get('template_version', 'v1')
        if template_version not in ('v1', 'v2', 'v3', 'v4', 'v5'):
            raise ValueError(
                f"Unsupported template_version '{template_version}'. Must be v1-v5."
            )

        # Dataset
        dataset_path = config.get('val_dataset', config.get('dataset'))
        if not dataset_path:
            raise ValueError("No dataset specified in config")
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(self.project_root, dataset_path)

        # vLLM server settings
        vllm_host = config.get('vllm_host', os.environ.get('VLLM_HOST', 'http://localhost'))
        vllm_port = int(config.get('vllm_port', os.environ.get('VLLM_PORT', 8002)))
        model = config.get('model', '')

        # Generation settings
        max_new_tokens = config.get('max_new_tokens', 512)
        temperature = config.get('temperature', 0.0)
        top_p = config.get('top_p', 1.0)
        stop_words = config.get('stop_words', [])

        # Concurrency
        max_workers = int(config.get('max_workers', os.environ.get('MAX_WORKERS', 32)))
        batch_size = int(config.get('batch_size', os.environ.get('BATCH_SIZE', 100)))

        # Output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_dir = os.path.join(
            self.project_root,
            'data', 'results',
            f"{dataset_name}_{template_version}_{timestamp}"
        )

        print("=" * 60)
        print("CRAFT Evaluation")
        print("=" * 60)
        print(f"Dataset:          {dataset_path}")
        print(f"Template version: {template_version}")
        print(f"vLLM server:      {vllm_host}:{vllm_port}")
        print(f"Model:            {model or '(auto-detect)'}")
        print(f"max_new_tokens:   {max_new_tokens}")
        print(f"temperature:      {temperature}")
        print(f"max_workers:      {max_workers}")
        print(f"Output dir:       {output_dir}")
        print("=" * 60)

        # Check server
        if not self.check_server(vllm_host, vllm_port):
            raise RuntimeError(
                f"vLLM server not reachable at {vllm_host}:{vllm_port}\n"
                f"Start the server first, e.g.:\n"
                f"  python -m vllm.entrypoints.openai.api_server \\\n"
                f"      --model {model} \\\n"
                f"      --port {vllm_port}"
            )

        # Connect
        self.setup_client(vllm_host, vllm_port, model)

        # Load data
        print(f"\nLoading dataset: {dataset_path}")
        data = self.load_dataset(dataset_path)
        print(f"Loaded {len(data)} samples\n")

        # Run inference
        print("Running inference...")
        start_time = time.time()
        results, avg_scores = self.run_inference(
            data=data,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            stop_words=stop_words,
            template_version=template_version,
            max_workers=max_workers,
            batch_size=batch_size,
        )
        elapsed = time.time() - start_time

        # Build summary
        summary = {
            'model': self.model_name,
            'dataset': dataset_path,
            'template_version': template_version,
            'num_samples': len(data),
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'inference_time_s': elapsed,
            'throughput_samp_per_s': len(data) / elapsed if elapsed > 0 else 0,
            **avg_scores,
        }

        # Print final metrics
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"EM score:       {avg_scores['em_score']:.4f}")
        print(f"F1 score:       {avg_scores['f1_score']:.4f}")
        print(f"Format score:   {avg_scores['format_score']:.4f}")
        print(f"Accuracy score: {avg_scores['accuracy_score']:.4f}")
        print(f"Relevance score:{avg_scores['relevance_score']:.4f}")
        print(f"Samples:        {len(data)}")
        print(f"Time:           {elapsed:.2f}s ({elapsed/60:.1f} min)")
        print(f"Throughput:     {summary['throughput_samp_per_s']:.2f} samp/s")
        print("=" * 60)

        self.save_results(results, output_dir, summary)

        return summary
