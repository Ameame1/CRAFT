"""
CRAFT Evaluation Metrics.

Supports 5 different prompt template versions with different XML tag structures:
- v1: <plan><gold_docs><reason><answer>
- v2: <gold_docs><reason><answer>
- v3: <plan><reason><answer>
- v4: <reason><answer>
- v5: <answer>
"""

import re
import json
import string
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"'", u"'", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def exact_match_score(prediction, ground_truths):
    """Check if prediction exactly matches any ground truth."""
    for ground_truth in ground_truths:
        if normalize_answer(prediction) == normalize_answer(ground_truth):
            return 1.0
    return 0.0


def f1_score(prediction, ground_truths):
    """Calculate F1 score between prediction and ground truths."""
    f1_score = 0.0
    for ground_truth in ground_truths:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_score = max(f1_score, f1)
    return f1_score


# ============================================================================
# Template-specific format checking functions
# ============================================================================

def format_score_v1(prediction):
    """Check format for v1: <plan><gold_docs><reason><answer>"""
    pattern = r"""
        ^
        <plan>       [^<]*  </plan>
        \s*
        <gold_docs>  [^<]*  </gold_docs>
        \s*
        <reason>     [^<]*  </reason>
        \s*
        <answer>     [^<]*  </answer>
        $
    """
    match = re.match(pattern, prediction, re.DOTALL | re.VERBOSE | re.MULTILINE)
    return 1.0 if match else 0.0


def format_score_v2(prediction):
    """Check format for v2: <gold_docs><reason><answer>"""
    pattern = r"""
        ^
        <gold_docs>  [^<]*  </gold_docs>
        \s*
        <reason>     [^<]*  </reason>
        \s*
        <answer>     [^<]*  </answer>
        $
    """
    match = re.match(pattern, prediction, re.DOTALL | re.VERBOSE | re.MULTILINE)
    return 1.0 if match else 0.0


def format_score_v3(prediction):
    """Check format for v3: <plan><reason><answer>"""
    pattern = r"""
        ^
        <plan>       [^<]*  </plan>
        \s*
        <reason>     [^<]*  </reason>
        \s*
        <answer>     [^<]*  </answer>
        $
    """
    match = re.match(pattern, prediction, re.DOTALL | re.VERBOSE | re.MULTILINE)
    return 1.0 if match else 0.0


def format_score_v4(prediction):
    """Check format for v4: <reason><answer>"""
    pattern = r"""
        ^
        <reason>     [^<]*  </reason>
        \s*
        <answer>     [^<]*  </answer>
        $
    """
    match = re.match(pattern, prediction, re.DOTALL | re.VERBOSE | re.MULTILINE)
    return 1.0 if match else 0.0


def format_score_v5(prediction):
    """Check format for v5: <answer> only"""
    pattern = r"""
        ^
        <answer>     [^<]*  </answer>
        $
    """
    match = re.match(pattern, prediction, re.DOTALL | re.VERBOSE | re.MULTILINE)
    return 1.0 if match else 0.0


# Map template version to format checking function
FORMAT_CHECKERS = {
    'v1': format_score_v1,
    'v2': format_score_v2,
    'v3': format_score_v3,
    'v4': format_score_v4,
    'v5': format_score_v5,
}


def format_score(prediction, template_version='v1'):
    """
    Check if prediction matches the expected format for the given template version.

    Args:
        prediction: Model's response text
        template_version: Template version ('v1', 'v2', 'v3', 'v4', or 'v5')

    Returns:
        float: 1.0 if format is correct, 0.0 otherwise
    """
    if template_version not in FORMAT_CHECKERS:
        raise ValueError(f"Unknown template version: {template_version}")

    return FORMAT_CHECKERS[template_version](prediction)


# ============================================================================
# Answer extraction
# ============================================================================

def extract_answer(prediction):
    """Extract answer from <answer> tags."""
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, prediction, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1)
    else:
        return ''


def accuracy_score(prediction, ground_truths, template_version='v1'):
    """
    Calculate accuracy score.

    Args:
        prediction: Model's response text
        ground_truths: List of correct answers
        template_version: Template version (for future use)

    Returns:
        float: 1.0 if exact match, 0.0 otherwise
    """
    return exact_match_score(extract_answer(prediction), ground_truths)


# ============================================================================
# Relevance/Gold docs extraction
# ============================================================================

def extract_gold_docs(prediction):
    """Extract gold_docs from <gold_docs> tags (for v1 and v2)."""
    pattern = r"<gold_docs>\s*(.*?)\s*</gold_docs>"
    match = re.search(pattern, prediction, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1)
    else:
        return ''


def relevance_score(prediction, ground_truths_list, template_version='v1'):
    """
    Calculate relevance score for document selection.

    Args:
        prediction: Model's response text
        ground_truths_list: List of correct supporting document IDs
        template_version: Template version ('v1', 'v2' use gold_docs; others don't)

    Returns:
        float: 1.0 if exact match, 0.5 if partial match, 0.0 if no match
    """
    # Only v1 and v2 have gold_docs
    if template_version not in ['v1', 'v2']:
        # For templates without gold_docs, we can't evaluate relevance
        # Return 1.0 to not penalize these templates
        return 1.0

    support_ids = extract_gold_docs(prediction)
    try:
        support_ids_list = json.loads(support_ids)
        support_ids_list = sorted(list(set(support_ids_list)))
        if support_ids_list == ground_truths_list:
            return 1.0
        elif len(set(support_ids_list) & set(ground_truths_list)) > 0:
            return 0.5
        else:
            return 0.0
    except:
        return 0.0


# ============================================================================
# Comprehensive scoring
# ============================================================================

def get_all_scores(prediction, answers_list, support_ids_list, template_version='v1'):
    """
    Calculate all scores for a prediction.

    Args:
        prediction: Model's response text
        answers_list: List of correct answers
        support_ids_list: List of correct supporting document IDs
        template_version: Template version

    Returns:
        dict: Dictionary with all scores
    """
    em_score_now = exact_match_score(extract_answer(prediction), answers_list)
    f1_score_now = f1_score(extract_answer(prediction), answers_list)

    format_score_now = format_score(prediction, template_version)
    accuracy_score_now = accuracy_score(prediction, answers_list, template_version)
    relevance_score_now = relevance_score(prediction, support_ids_list, template_version)

    return {
        'em_score': em_score_now,
        'f1_score': f1_score_now,
        'format_score': format_score_now,
        'accuracy_score': accuracy_score_now,
        'relevance_score': relevance_score_now,
    }
