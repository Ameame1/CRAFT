# CRAFT Judge Module

The judge module provides faithfulness evaluation for CRAFT traces. It evaluates **internal consistency** (R_faith) rather than factual correctness against gold labels.

## Judge Prompts by Version

| Version | Trace Format | Metrics Evaluated |
|---------|--------------|-------------------|
| v1 | `<plan><gold_docs><reason><answer>` | plan_reason, receipt_reason, reason_answer, evidence_faithfulness |
| v2 | `<gold_docs><reason><answer>` | receipt_reason, reason_answer, evidence_faithfulness |
| v3 | `<plan><reason><answer>` | plan_reason, reason_answer, evidence_faithfulness |
| v4 | `<reason><answer>` | reason_answer, evidence_faithfulness |

## Consistency Metrics

### plan_reason_consistency (v1, v3 only)
- Does the reasoning follow the intent and order implied by the plan?
- Pass (1): Reasoning addresses subquestions in coherent order
- Fail (0): Plan ignored, contradicted, or required hops skipped

### receipt_reason_consistency (v1, v2 only)
- Does the reasoning stay within the evidence boundary declared by gold_docs?
- Pass (1): All cited documents are in gold_docs
- Fail (0): Cites documents outside gold_docs, or no citations at all

### reason_answer_consistency (all versions)
- Is the answer supported by the reasoning?
- Pass (1): Answer is a reasonable conclusion of reasoning
- Fail (0): Answer contradicts reasoning or introduces unsupported claims

### evidence_grounded_faithfulness (all versions)
- Are the claims in reasoning actually supported by cited documents?
- Pass (1): All key claims are supported by cited document text
- Fail (0): Fabricated claims, overclaims, contradictions, or no citations

## Usage

### Starting the Judge Server

```bash
# Using default config (cfg/judge.yaml)
python src/judge/server.py

# With custom config
python src/judge/server.py --config judge

# Override port and CUDA devices
python src/judge/server.py --port 8000 --cuda-devices 2,3
```

### Calling the Judge API

```python
from openai import OpenAI
from src.judge import format_v1_judge_prompt

# Connect to judge server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Format the judge prompt
prompt = format_v1_judge_prompt(
    query="What is the capital of France?",
    docs="1. Paris is the capital of France...",
    model_output="<plan>...</plan><gold_docs>[1]</gold_docs><reason>...</reason><answer>Paris</answer>"
)

# Get evaluation
response = client.chat.completions.create(
    model="Qwen3-30B-A3B-Instruct",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
)

print(response.choices[0].message.content)
```

## Output Format

The judge returns XML with binary scores:

```xml
<evaluation>
  <plan_reason_consistency>1</plan_reason_consistency>
  <receipt_reason_consistency>1</receipt_reason_consistency>
  <reason_answer_consistency>1</reason_answer_consistency>
  <evidence_grounded_faithfulness>1</evidence_grounded_faithfulness>
  <overall_consistency>1</overall_consistency>
  <failure_modes></failure_modes>
  <rationale>All consistency checks passed.</rationale>
</evaluation>
```

## Important Notes

1. **No Factual Evaluation**: The judge does NOT check if the answer is factually correct against gold labels. It only evaluates internal consistency.

2. **No World Knowledge**: The judge must only use the provided document text to verify claims, not external knowledge.

3. **Missing Components**: If required trace components are missing, relevant checks automatically fail.

4. **Deterministic**: Always use `temperature=0.0` for consistent scoring.
