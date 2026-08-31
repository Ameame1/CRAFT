#!/usr/bin/env python3
"""
RS-CRAFT Stage A: sample N candidate traces per training prompt
from a base model served by vLLM (chat-completions endpoint, n=N).

Input :  data/train/grpo/grpo_25000_v1_messages.jsonl
Output:  data/train/rscraft/candidates_v1.jsonl
            {id, messages, candidates: [str x N], answers, supporting_ids, name}
Resumable: skips ids already present in the output file.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import requests
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config


def load_jsonl(path: Path) -> List[Dict]:
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    seen = set()
    with open(path, "r") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["id"])
            except Exception:
                continue
    return seen


def check_server(host: str, port: int) -> bool:
    try:
        r = requests.get(f"{host}:{port}/v1/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def resolve_model_name(client: OpenAI, fallback: str) -> str:
    try:
        models = client.models.list()
        if models.data:
            return models.data[0].id
    except Exception:
        pass
    return fallback


def sample_one(client: OpenAI, model_name: str, sample: Dict,
               n: int, temperature: float, top_p: float,
               max_new_tokens: int, max_retries: int = 3) -> Dict:
    """Generate N candidates for one prompt."""
    messages = sample["messages"]
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
            candidates = [c.message.content or "" for c in resp.choices]
            return {
                "id": sample.get("id"),
                "name": sample.get("name", "unknown"),
                "messages": messages,
                "candidates": candidates,
                "answers": sample.get("answers", []),
                "supporting_ids": sample.get("supporting_ids", []),
                "template_version": sample.get("template_version", "v1"),
            }
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)
    return {
        "id": sample.get("id"),
        "name": sample.get("name", "unknown"),
        "messages": messages,
        "candidates": [],
        "answers": sample.get("answers", []),
        "supporting_ids": sample.get("supporting_ids", []),
        "template_version": sample.get("template_version", "v1"),
        "error": f"{type(last_err).__name__}: {last_err}",
    }


def main():
    parser = argparse.ArgumentParser(description="RS-CRAFT Stage A: sample candidates")
    parser.add_argument("--config", default="rscraft", help="config name (without .yaml)")
    parser.add_argument("--limit", type=int, default=None, help="cap number of prompts (pilot mode)")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--host", default=None, help="override vllm_host (e.g. http://localhost)")
    parser.add_argument("--port", type=int, default=None, help="override vllm_port")
    parser.add_argument("--start", type=int, default=None, help="shard start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="shard end index (exclusive)")
    args = parser.parse_args()

    cfg = load_config(args.config) or {}
    input_path = Path(args.input or cfg.get("input_dataset"))
    output_path = Path(args.output or cfg.get("candidates_path"))
    n = args.n or int(cfg.get("n_candidates", 4))
    workers = args.workers or int(cfg.get("sample_max_workers", 32))
    temperature = float(cfg.get("sample_temperature", 1.0))
    top_p = float(cfg.get("sample_top_p", 0.9))
    max_new_tokens = int(cfg.get("sample_max_new_tokens", 512))
    host = args.host or cfg.get("vllm_host", "http://localhost")
    port = int(args.port if args.port is not None else cfg.get("vllm_port", 8002))
    fallback_model = cfg.get("base_model", "")

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RS-CRAFT Stage A: sample candidates")
    print("=" * 60)
    print(f"Input        : {input_path}")
    print(f"Output       : {output_path}")
    print(f"N candidates : {n}")
    print(f"Workers      : {workers}")
    print(f"vLLM         : {host}:{port}")
    print(f"limit        : {args.limit}")
    print("=" * 60)

    if not check_server(host, port):
        raise RuntimeError(
            f"vLLM server not reachable at {host}:{port}. "
            f"Start it via scripts/server/start_craft.sh."
        )

    client = OpenAI(base_url=f"{host}:{port}/v1", api_key="not-needed")
    model_name = resolve_model_name(client, fallback_model)
    print(f"Using model  : {model_name}")

    data = load_jsonl(input_path)
    if args.start is not None or args.end is not None:
        s = args.start or 0
        e = args.end if args.end is not None else len(data)
        data = data[s:e]
        print(f"Shard slice : [{s}:{e}] -> {len(data)} prompts")
    if args.limit:
        data = data[: args.limit]
    seen = existing_ids(output_path)
    todo = [s for s in data if s.get("id") not in seen]
    print(f"Total prompts: {len(data)}; already done: {len(seen)}; todo: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    written = 0
    started = time.time()
    with open(output_path, "a") as out_f, \
         ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(sample_one, client, model_name, s, n, temperature, top_p, max_new_tokens): s
            for s in todo
        }
        for fut in as_completed(futures):
            row = fut.result()
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1
            if written % 50 == 0 or written == len(todo):
                elapsed = time.time() - started
                rate = written / elapsed if elapsed else 0
                eta = (len(todo) - written) / rate if rate else 0
                print(f"  [{written}/{len(todo)}] {rate:.2f} prompts/s  ETA {eta/60:.1f} min")

    print(f"Done. Wrote {written} new rows to {output_path}.")


if __name__ == "__main__":
    main()
