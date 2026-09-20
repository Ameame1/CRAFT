# Does Faithfulness-Guided Alignment Hurt Accuracy? Unlocking Accurate and Faithful Post-Retrieval Reasoning

**CRAFT** | [Project Page](https://ameame1.github.io/CRAFT/) |
[Model Checkpoints](https://huggingface.co/Ameame1002/CRAFT) |
[Quick Start](#installation) | [Citation](#citation)

## Overview

CRAFT is a GRPO framework for structured, machine-auditable post-retrieval
reasoning in multi-hop question answering. Its central research question is
whether faithfulness-guided alignment can unlock task-specific reasoning
capacity without sacrificing answer accuracy.

CRAFT trains a policy to produce structured reasoning traces over fixed retrieved
documents. Its objective combines format compliance, answer correctness, citation
validity, and judge-based semantic faithfulness. Five trace variants expose a
capacity-dependent trade-off between auditability and learnability.

## Code

The release includes prompt templates, training configurations, inference
servers, and answer/faithfulness evaluation. The workflow is:

1. Normalize the three QA benchmarks and render a selected trace template.
2. Train with deterministic rewards and, for structured variants, a local judge.
3. Generate answers over fixed retrieved documents.
4. Evaluate answer correctness and audit the emitted reasoning traces.

```text
cfg/                 Training, evaluation, vLLM, and DeepSpeed configurations
scripts/preprocess/  Dataset download and template rendering
scripts/train/       SFT and GRPO launchers
scripts/eval/        Answer and faithfulness evaluation
src/templates/       CRAFT v1-v5 prompt templates
src/rewards/         Deterministic and judge-based rewards
src/train/           SFT and GRPO trainer wrappers
src/eval/            Parsing and evaluation metrics
index.html           Interactive project page
```

## Models

Checkpoints are hosted at [Ameame1002/CRAFT](https://huggingface.co/Ameame1002/CRAFT).
Each model is stored in a `<scale>_<variant>` subfolder.

| Scale | Release directories | Checkpoint source |
| --- | --- | --- |
| 0.5B | `0.5B_v1` through `0.5B_v4` | Local 312-step GRPO runs |
| 1.5B | `1.5B_v1` through `1.5B_v4` | Local 312-step GRPO runs |
| 3B | `3B_v1` through `3B_v4` | Local 312-step GRPO runs |
| 7B | `7B_v1` through `7B_v4` | Archived judge-enabled checkpoints |
| 7B | `7B_v5` | Archived answer-only checkpoint, without judge reward |

Download one model without fetching the entire repository:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "Ameame1002/CRAFT",
    allow_patterns="7B_v1/*",
    local_dir="./models/CRAFT",
)
```

The resulting model path is `./models/CRAFT/7B_v1`. SFT checkpoints are not
included in the current Hub release. Checkpoint availability does not establish
that every manuscript-reported score is reproduced by that checkpoint; see
[Results and Provenance](#results-and-provenance).

## Trace Variants

| Variant | Output structure | Training rewards |
| --- | --- | --- |
| CRAFT v1 | `plan + gold_docs + reason + answer` | format, answer, citation, faithfulness |
| CRAFT v2 | `gold_docs + reason + answer` | format, answer, citation, faithfulness |
| CRAFT v3 | `plan + reason + answer` | format, answer, faithfulness |
| CRAFT v4 | `reason + answer` | format, answer, faithfulness |
| CRAFT v5 | `answer` | format, answer |

The repository includes 20,000-example, 312-step configurations for
Qwen2.5-0.5B, 1.5B, 3B, and 7B across all five variants. These local configurations
are distinct from the archived 7B checkpoint source.

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

## Results and Provenance

The [project page](https://ameame1.github.io/CRAFT/#results) displays values from
the manuscript titled above, including comparisons across model capacities and
the 7B trace variants. They are manuscript-reported values, not a new benchmark
of the uploaded checkpoints.

The archived result-generation audit marks the main-result aggregates as fitted
simulations anchored on existing evaluation traces and summary scores. The
w/o-judge comparison uses Full-anchored counterfactual estimates; it is not an
independently measured judge-removal experiment. The fitted 7B v5 results also
include dataset-specific score adjustments. The ten fitted 1,000-example
replicates must not be interpreted as ten fresh model-inference runs.

The local small-model anchors use checkpoint-312; the archived 7B v1-v4 anchors
reference checkpoint-390. Matching the page to the manuscript checks numerical
consistency, not experimental reproducibility. To verify page consistency with
your local manuscript:

```bash
python scripts/analysis/check_project_page.py \
  --paper /path/to/EMNLP_CRAFT/main.tex --page index.html
```

## Scope

CRAFT operates after retrieval: retrieved documents are treated as fixed inputs.
The reported faithfulness metric measures consistency and evidence support in the
emitted trace; it should not be interpreted as direct access to latent model
reasoning.

## Citation

```bibtex
@misc{liu2026craft,
  title = {Does Faithfulness-Guided Alignment Hurt Accuracy? Unlocking Accurate and Faithful Post-Retrieval Reasoning},
  author = {Liu, Yu and Zhang, Wenxiao and Guo, Diandian and Cao, Cong and Yuan, Fangfang and Sun, Qiang and Liu, Yanbing and Hong, Jin Bum and Ma, Zhiyuan},
  year = {2026},
  url = {https://github.com/Ameame1/CRAFT}
}
```

## License

MIT License.
