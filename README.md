# CRAFT: Calibrated Reasoning with Answer-Faithful Traces for Multi-Hop Question Answering

Official implementation of CRAFT, a GRPO framework for accurate and auditable
post-retrieval reasoning in multi-hop question answering.

**Project page:** [ameame1.github.io/CRAFT](https://ameame1.github.io/CRAFT/)

CRAFT trains a policy to produce structured reasoning traces over fixed retrieved
documents. Its objective combines format compliance, answer correctness, citation
validity, and judge-based semantic faithfulness.

## Trace Variants

| Variant | Output structure | Training rewards |
| --- | --- | --- |
| CRAFT v1 | `plan + gold_docs + reason + answer` | format, answer, citation, faithfulness |
| CRAFT v2 | `gold_docs + reason + answer` | format, answer, citation, faithfulness |
| CRAFT v3 | `plan + reason + answer` | format, answer, faithfulness |
| CRAFT v4 | `reason + answer` | format, answer, faithfulness |
| CRAFT v5 | `answer` | format, answer |

The repository includes the 20,000-example, 312-step configurations used for
Qwen2.5-0.5B, 1.5B, 3B, and 7B across all five variants.

## Installation

```bash
conda env create -f environment.yml
conda activate craft
pip install -e .

# Recommended for supported NVIDIA GPUs
pip install flash-attn --no-build-isolation
```

Python 3.10 or 3.11 is recommended. The released environment uses MS-Swift,
vLLM, DeepSpeed, and FlashAttention.

## Data

Download and normalize HotpotQA, 2WikiMultiHopQA, and MuSiQue:

```bash
python scripts/preprocess/download_datasets.py
```

Build the paper's 20,000-example mixture (5,000 HotpotQA, 5,000
2WikiMultiHopQA, and 10,000 MuSiQue) for CRAFT v1-v5:

```bash
python scripts/preprocess/generate_grpo_data.py \
  --samples 20000 \
  --hotpotqa-ratio 0.25 \
  --wiki2-ratio 0.25 \
  --musique-ratio 0.50 \
  --seed 42
```

Generated files are written under `data/train/grpo/`. Dataset files and model
checkpoints are intentionally excluded from Git.

## GRPO Training

The paper uses one 8-GPU node: four GPUs train the policy and four host the local
Qwen3-30B-A3B judge. Start the judge first:

```bash
CUDA_DEVICES=4,5,6,7 \
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 \
./scripts/server/start_judge.sh
```

Then launch a policy run, for example CRAFT v1 at 7B:

```bash
JUDGE_MODE=local \
NUM_GPUS=4 \
CUDA_DEVICES=0,1,2,3 \
./scripts/train/grpo.sh \
  --version v1 \
  --config-name grpo_7b_v1_20k_full_8gpu_offline
```

Replace `7b` and `v1` in the configuration name to select another released
scale or trace variant. CRAFT v5 has no judge reward, so the launcher skips the
judge health check automatically.

## Evaluation

Start a policy inference server:

```bash
CUDA_DEVICES=0 MODEL=/path/to/checkpoint ./scripts/server/start_craft.sh
```

Run answer evaluation:

```bash
./scripts/eval/eval.sh --version v1 --dataset hotpotqa --model /path/to/checkpoint
```

Run faithfulness evaluation against the local judge:

```bash
python scripts/eval/faithfulness_eval.py \
  --predictions /path/to/results.jsonl \
  --output /path/to/faithfulness-summary.json \
  --template-version v1
```

## Repository Layout

```text
cfg/                 Training, evaluation, vLLM, and DeepSpeed configurations
scripts/preprocess/  Dataset download and template rendering
scripts/train/       SFT and GRPO launchers
scripts/eval/        Answer and faithfulness evaluation
src/templates/       CRAFT v1-v5 prompt templates
src/rewards/         Deterministic and judge-based rewards
src/train/           SFT and GRPO trainer wrappers
src/eval/            Parsing and evaluation metrics
```

## Scope

CRAFT operates after retrieval: retrieved documents are treated as fixed inputs.
The reported faithfulness metric measures consistency and evidence support in the
emitted trace; it should not be interpreted as direct access to latent model
reasoning.

## License

MIT License.
