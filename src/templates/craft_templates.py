"""
CRAFT Prompt Templates (v1-v5).

Each template version includes different trace components:
- v1 (CRAFT_v1): <plan><gold_docs><reason><answer> - Full template (4 metrics)
- v2 (CRAFT_v2): <gold_docs><reason><answer> - No plan (3 metrics)
- v3 (CRAFT_v3): <plan><reason><answer> - No gold_docs (3 metrics)
- v4 (CRAFT_v4): <reason><answer> - Minimal (2 metrics)
- v5 (CRAFT_v5): <answer> - Answer-only baseline

Key design principles:
- When <plan> exists (v1, v3), <reason> MUST explicitly answer subquestions in order
- When <gold_docs> exists (v1, v2), reasoning MUST only cite documents in gold_docs
- All templates require explicit document citations in reasoning (except v5)
"""

# -----------------------------
# CRAFT_v1: <plan><gold_docs><reason><answer>
# Full template with all 4 trace components
# -----------------------------
prompt_template_v1 = '''A conversation between User and Assistant.
The User asks a question and supplies several numbered documents.
The Assistant should answer the question using these documents as evidence.

User's input always has this form:

<query>
{query}
</query>

<docs>
{docs}
</docs>

Here, <docs> contains documents labeled with numbers (1., 2., 3., ...).

The Assistant's reply is divided into exactly four tagged parts in this order:

<plan>
[Break the query into 2-5 ordered subquestions that, when answered in turn, lead to the final answer.
Each subquestion MUST be written as:
Subquestion1: ...
Subquestion2: ...
...
SubquestionN: ...

Rules for <plan>:
- Do NOT mention any documents here.
- Each subquestion should be atomic and unambiguous.
- You may refer to earlier answers as [Answer1], [Answer2], etc.]
</plan>

<gold_docs>
[Write a single JSON-style list of the document numbers that actually help answer the query,
in square brackets, e.g. [1,3,5]. Do not include any words or explanation.]
</gold_docs>

<reason>
[CRITICAL: You MUST follow the plan explicitly.
Write your reasoning in N blocks, one per subquestion, in order.
Use this exact internal structure inside <reason>:

Subquestion1: <short answer to Subquestion1>. Evidence: doc [k] ... (you may cite multiple docs).
Subquestion2: <short answer to Subquestion2, may use [Answer1]>. Evidence: docs [k,m] ...
...
SubquestionN: <short answer to SubquestionN>. Evidence: doc [k] ...

Rules for <reason>:
- You MUST use ONLY the documents listed in <gold_docs>.
- Every Subquestion block MUST include at least one explicit document citation like doc [3] or docs [2,3].
- Do NOT introduce claims that are not supported by the cited documents.
- After SubquestionN, add one final line:
Final: Therefore the answer is <final answer phrase> (supported by docs [...]).
]
</reason>

<answer>
[Give only a short phrase or a single word that directly answers the query.
Do not repeat the reasoning or add any explanation.]
</answer>

**User**:

<query>
{query}
</query>

<docs>
{docs}
</docs>
'''


# -----------------------------
# CRAFT_v2: <gold_docs><reason><answer>
# Template without planning step
# -----------------------------
prompt_template_v2 = '''A conversation between User and Assistant.
The User asks a question and supplies several numbered documents.
The Assistant should answer the question using these documents as evidence.

User's input always has this form:

<query>
{query}
</query>

<docs>
{docs}
</docs>

Here, <docs> contains documents labeled with numbers (1., 2., 3., ...).

The Assistant's reply is divided into exactly three tagged parts in this order:

<gold_docs>
[Write a single JSON-style list of the document numbers that actually help answer the query,
in square brackets, e.g. [1,3,5]. Do not include any words or explanation.]
</gold_docs>

<reason>
[Use ONLY the documents listed in <gold_docs> to justify the answer.
Rules for <reason>:
- For every important claim, provide explicit citations like doc [k] or docs [k,m].
- Do NOT introduce claims that are not supported by the cited documents.
- End with one final line: Final: Therefore the answer is <final answer phrase> (supported by docs [...]).]
</reason>

<answer>
[Give only a short phrase or a single word that directly answers the query.
Do not repeat the reasoning or add any explanation.]
</answer>

**User**:

<query>
{query}
</query>

<docs>
{docs}
</docs>
'''


# -----------------------------
# CRAFT_v3: <plan><reason><answer>
# Template without gold_docs declaration
# -----------------------------
prompt_template_v3 = '''A conversation between User and Assistant.
The User asks a question and supplies several numbered documents.
The Assistant should answer the question using these documents as evidence.

User's input always has this form:

<query>
{query}
</query>

<docs>
{docs}
</docs>

Here, <docs> contains documents labeled with numbers (1., 2., 3., ...).

The Assistant's reply is divided into exactly three tagged parts in this order:

<plan>
[Break the query into 2-5 ordered subquestions that, when answered in turn, lead to the final answer.
Each subquestion MUST be written as:
Subquestion1: ...
Subquestion2: ...
...
SubquestionN: ...

Rules for <plan>:
- Do NOT mention any documents here.
- Each subquestion should be atomic and unambiguous.
- You may refer to earlier answers as [Answer1], [Answer2], etc.]
</plan>

<reason>
[CRITICAL: You MUST follow the plan explicitly.
Write your reasoning in N blocks, one per subquestion, in order.
Use this exact internal structure inside <reason>:

Subquestion1: <short answer to Subquestion1>. Evidence: doc [k] ... (you may cite multiple docs).
Subquestion2: <short answer to Subquestion2, may use [Answer1]>. Evidence: docs [k,m] ...
...
SubquestionN: <short answer to SubquestionN>. Evidence: doc [k] ...

Rules for <reason>:
- You may use ANY documents from <docs>, but every Subquestion block MUST include at least one explicit doc citation.
- Do NOT introduce claims that are not supported by the cited documents.
- After SubquestionN, add one final line:
Final: Therefore the answer is <final answer phrase> (supported by docs [...]).]
</reason>

<answer>
[Give only a short phrase or a single word that directly answers the query.
Do not repeat the reasoning or add any explanation.]
</answer>

**User**:

<query>
{query}
</query>

<docs>
{docs}
</docs>
'''


# -----------------------------
# CRAFT_v4: <reason><answer>
# Minimal template with only reasoning and answer
# -----------------------------
prompt_template_v4 = '''A conversation between User and Assistant.
The User asks a question and supplies several numbered documents.
The Assistant should answer the question using these documents as evidence.

User's input always has this form:

<query>
{query}
</query>

<docs>
{docs}
</docs>

Here, <docs> contains documents labeled with numbers (1., 2., 3., ...).

The Assistant's reply is divided into exactly two tagged parts in this order:

<reason>
[Combine information from <docs> to justify the answer.
Rules for <reason>:
- For every important claim, provide explicit citations like doc [k] or docs [k,m].
- Do NOT introduce claims that are not supported by the cited documents.
- End with one final line: Final: Therefore the answer is <final answer phrase> (supported by docs [...]).]
</reason>

<answer>
[Give only a short phrase or a single word that directly answers the query.
Do not repeat the reasoning or add any explanation.]
</answer>

**User**:

<query>
{query}
</query>

<docs>
{docs}
</docs>
'''


# -----------------------------
# CRAFT_v5: <answer>
# Answer-only baseline (no reasoning trace)
# -----------------------------
prompt_template_v5 = '''A conversation between User and Assistant.
The User asks a question and supplies several numbered documents.
The Assistant should answer the question using these documents as evidence.

User's input always has this form:

<query>
{query}
</query>

<docs>
{docs}
</docs>

Here, <docs> contains documents labeled with numbers (1., 2., 3., ...).

The Assistant's reply must contain exactly one tagged part:

<answer>
[Give only a short phrase or a single word that directly answers the query.
Do not show any reasoning or explanation.]
</answer>

**User**:

<query>
{query}
</query>

<docs>
{docs}
</docs>
'''


# Template dictionary for easy access
CRAFT_TEMPLATES = {
    'v1': prompt_template_v1,
    'v2': prompt_template_v2,
    'v3': prompt_template_v3,
    'v4': prompt_template_v4,
    'v5': prompt_template_v5,
}


def get_template(version='v1'):
    """
    Get a specific prompt template by version.

    Args:
        version: Template version ('v1', 'v2', 'v3', 'v4', or 'v5')

    Returns:
        The corresponding prompt template string

    Raises:
        ValueError: If version is not recognized
    """
    if version not in CRAFT_TEMPLATES:
        raise ValueError(f"Unknown template version: {version}. Must be one of {list(CRAFT_TEMPLATES.keys())}")
    return CRAFT_TEMPLATES[version]
