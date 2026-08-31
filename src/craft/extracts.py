"""
Extraction and text processing utilities for CRAFT outputs.

Handles parsing XML-formatted CRAFT outputs and text normalization
for answer matching.
"""

import re
import string


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


def extract_answer(prediction):
    """
    Extract answer text from CRAFT output.

    Args:
        prediction: CRAFT output string with <answer> tags

    Returns:
        Extracted answer string, or empty string if not found
    """
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, prediction, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1)
    else:
        return ''


def extract_support_ids(prediction):
    """
    Extract supporting document IDs from CRAFT output.

    Args:
        prediction: CRAFT output string with <gold_docs> tags

    Returns:
        Extracted sources string, or empty string if not found
    """
    pattern = r"<gold_docs>\s*(.*?)\s*</gold_docs>"
    match = re.search(pattern, prediction, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1)
    else:
        return ''


def extract_plan(prediction):
    """
    Extract plan from CRAFT output.

    Args:
        prediction: CRAFT output string with <plan> tags

    Returns:
        Extracted plan string, or empty string if not found
    """
    pattern = r"<plan>\s*(.*?)\s*</plan>"
    match = re.search(pattern, prediction, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1)
    else:
        return ''


def extract_reason(prediction):
    """
    Extract reasoning from CRAFT output.

    Args:
        prediction: CRAFT output string with <reason> tags

    Returns:
        Extracted reasoning string, or empty string if not found
    """
    pattern = r"<reason>\s*(.*?)\s*</reason>"
    match = re.search(pattern, prediction, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1)
    else:
        return ''
