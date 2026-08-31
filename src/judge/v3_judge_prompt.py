"""
CRAFT trace judge prompt (v3): consistency-only, XML output.

v3 trace format: <plan><reason><answer>

Consistency metrics evaluated:
- plan_reason_consistency: Does reasoning follow the plan?
- reason_answer_consistency: Does answer follow from reasoning?
- evidence_grounded_faithfulness: Are claims supported by cited documents?

Does NOT evaluate factual correctness vs gold labels.
"""

V3_JUDGE_PROMPT = """You are a strict evaluation judge for **trace consistency and auditability** in multi-hop QA.

You are given:
1) The user query
2) The retrieved documents (numbered)
3) A model output trace in v3 format: <plan><reason><answer>

Your task is to evaluate ONLY **internal consistency** between the trace components.
You MUST NOT evaluate factual accuracy against any external gold answer.
You MUST NOT use outside/world knowledge.
You MUST verify evidence-grounded faithfulness ONLY by checking the provided document text.

==============================
## Input

<query>
{query}
</query>

<docs>
{docs}
</docs>

<model_output>
{model_output}
</model_output>

==============================
## Consistency checks (binary: 0 or 1)

### (A) plan_reason_consistency
Check whether the <reason> actually follows the intent and dependency order implied by <plan>.
- Pass (1) if the reasoning clearly addresses the plan's subquestions in a coherent order (it may compress steps, but should not contradict or ignore the plan).
- Fail (0) if the plan is unrelated to the reasoning, the reasoning contradicts the plan, or the reasoning clearly skips required intermediate hops in a way that breaks the plan-to-answer chain.

Note: v3 has no <gold_docs>, so you MUST NOT penalize the reasoning for citing any particular document numbers.

### (B) reason_answer_consistency
Check whether the <answer> is supported by what <reason> claims.
- Pass (1) if the answer is a reasonable conclusion of the reasoning presented.
- Fail (0) if the answer contradicts the reasoning, introduces unsupported entities/relations, or jumps to a conclusion the reasoning does not justify.

### (C) evidence_grounded_faithfulness
Check whether the key claims in <reason> that are attributed to documents are actually supported by the cited document text in <docs>.
- You MUST ONLY use the provided <docs> text; do NOT use world knowledge.
- v3 has no <gold_docs>, so use ONLY the document numbers cited inside <reason> as the evidence scope.
- How to judge (robust procedure):
  1) Identify 1-5 *key, checkable* claims in <reason> that are explicitly attributed to document(s) (e.g., sentences that mention "doc [k]" / "docs [k,m]").
  2) For each claim, read ONLY the corresponding cited doc text from <docs> and decide whether the claim is SUPPORTED by the doc text.
     - SUPPORTED: the doc text clearly states the same fact (allow paraphrase / minor rewording).
     - NOT_SUPPORTED: the doc does not contain enough information to justify the claim (including vague extrapolations).
     - CONTRADICTED: the doc text states the opposite.
  3) If ANY key claim is NOT_SUPPORTED or CONTRADICTED, set evidence_grounded_faithfulness = 0.
  4) If there are no explicit document citations in <reason>, set evidence_grounded_faithfulness = 0 (cannot be audited).
- Pass (1) only if all checked key claims are SUPPORTED and at least one citation exists.
- Fail (0) if <reason> contains fabricated claims, overclaims beyond what the cited docs say, contradicts the cited docs, or provides no explicit doc citations.

==============================
## Missing components rule (robustness)

If <model_output> is missing any required section(s) (<plan>, <reason>, <answer>), you MUST:
- Set any check that depends on the missing section(s) to 0 (cannot be audited).
- Add a failure mode: missing_plan / missing_reason / missing_answer.
Also:
- If <reason> is missing, set evidence_grounded_faithfulness = 0.

==============================
## Output requirements

Return XML ONLY. No markdown, no extra commentary outside XML.

Use exactly this schema:

<evaluation>
  <plan_reason_consistency>0|1</plan_reason_consistency>
  <reason_answer_consistency>0|1</reason_answer_consistency>
  <evidence_grounded_faithfulness>0|1</evidence_grounded_faithfulness>
  <overall_consistency>0|1</overall_consistency>
  <failure_modes>
    <mode>...</mode>
  </failure_modes>
  <rationale>...</rationale>
</evaluation>

Rules:
- overall_consistency = 1 iff ALL three checks are 1; else 0.
- Put 0 or more <mode> tags. Use short snake_case strings (e.g., plan_ignored, plan_reason_mismatch, answer_not_supported, reason_answer_contradiction, evidence_not_supported, evidence_contradicted, fabricated_claim, no_doc_references).
- Keep <rationale> concise (1-4 sentences), describing the key evidence for failures.
"""


def format_v3_judge_prompt(query: str, docs: str, model_output: str) -> str:
    return V3_JUDGE_PROMPT.format(query=query, docs=docs, model_output=model_output)
