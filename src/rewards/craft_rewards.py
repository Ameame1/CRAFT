"""
CRAFT Reward Functions for GRPO Training.

Paper notation:
- R_fmt (format_reward): Format compliance - checks XML structure
- R_ans (accuracy_reward): Answer correctness - exact match with ground truth
- R_gold (relevance_reward): Citation accuracy - gold_docs match (v1/v2 only)
- R_faith (judge_overall_consistency_reward): Faithfulness audit via LLM judge

Supports template versions v1-v5 with different trace structures.
"""

import os
import re
import json
import xml.etree.ElementTree as ET
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from src.eval.metrics import (
    format_score as multi_format_score,
    accuracy_score as multi_accuracy_score,
    relevance_score as multi_relevance_score,
    extract_answer
)
from src.judge import (
    format_v1_judge_prompt,
    format_v2_judge_prompt,
    format_v3_judge_prompt,
    format_v4_judge_prompt,
)

# Judge client configuration
# Supports two modes:
#   1. API mode (Qwen3-max): Set QWEN_API_KEY environment variable
#   2. Local mode (Qwen3-30B): Set LOCAL_JUDGE_URL environment variable
#
# Environment variables:
#   JUDGE_MODE: "api" (default) or "local"
#   QWEN_API_KEY or DASHSCOPE_API_KEY: API key for Qwen API (api mode)
#   QWEN_API_BASE: API base URL (defaults to DashScope)
#   LOCAL_JUDGE_URL: Local vLLM server URL (local mode, e.g., http://localhost:8000/v1)

# Initialize client as None (lazy initialization)
client = None
_judge_mode = os.environ.get('JUDGE_MODE', 'api')  # 'api' or 'local'

def _load_dotenv_if_present():
    """Best-effort load of .env at the repo root so VOLCENGINE / QWEN_API_KEY etc.
    are picked up without manually exporting. No external dependency required."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        # walk up to repo root (folder containing .env)
        for _ in range(6):
            candidate = os.path.join(here, ".env")
            if os.path.exists(candidate):
                with open(candidate) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        # don't overwrite already-set env vars
                        os.environ.setdefault(k, v)
                return
            here = os.path.dirname(here)
    except Exception:
        pass


def _init_judge_client():
    """Initialize or reinitialize the judge API client based on mode.

    Supported JUDGE_MODE values:
      'api'        — Qwen3-Max via Alibaba DashScope (paper default)
      'volcengine' — Doubao / DeepSeek / etc via Volcengine Ark (uses VOLCENGINE key)
      'local'      — local vLLM server (Qwen3-30B-A3B-Instruct-2507)
    """
    global client, _judge_mode

    _load_dotenv_if_present()
    _judge_mode = os.environ.get('JUDGE_MODE', 'api')

    if _judge_mode == 'local':
        # Local mode: Use local vLLM server (Qwen3-30B)
        local_url = os.environ.get('LOCAL_JUDGE_URL', 'http://localhost:8000/v1')
        client = OpenAI(api_key="EMPTY", base_url=local_url, max_retries=0)
        print(f"[Judge] Using local server at {local_url}")

    elif _judge_mode == 'volcengine':
        # Volcengine Ark — OpenAI-compatible. Key from .env (VOLCENGINE) or env var.
        api_key = (os.environ.get('VOLCENGINE')
                   or os.environ.get('VOLC_API_KEY')
                   or os.environ.get('VOLC_KEY')
                   or '')
        api_base = os.environ.get('VOLC_API_BASE',
                                  'https://ark.cn-beijing.volces.com/api/v3')
        if not api_key:
            raise ValueError(
                "Volcengine API key not found. Set VOLCENGINE or VOLC_API_KEY in env or .env.\n"
                "See docs/vol_models.md for the full model list."
            )
        client = OpenAI(api_key=api_key, base_url=api_base, max_retries=0)
        print(f"[Judge] Using Volcengine Ark at {api_base}")

    else:
        # API mode: Use Qwen3-max via DashScope
        api_key = os.environ.get('QWEN_API_KEY', os.environ.get('DASHSCOPE_API_KEY', ''))
        api_base = os.environ.get('QWEN_API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1')

        if not api_key:
            raise ValueError(
                "Qwen API key not found. Please set QWEN_API_KEY or DASHSCOPE_API_KEY environment variable.\n"
                "Get your API key from: https://dashscope.console.aliyun.com/\n"
                "Or use local mode: export JUDGE_MODE=local LOCAL_JUDGE_URL=http://localhost:8000/v1\n"
                "Or use Volcengine: export JUDGE_MODE=volcengine (with VOLCENGINE key in .env)"
            )
        client = OpenAI(api_key=api_key, base_url=api_base, max_retries=0)
        print(f"[Judge] Using Qwen API at {api_base}")

    return client

def get_judge_client():
    """Get the judge API client, initializing if needed."""
    global client
    if client is None:
        _init_judge_client()
    return client

def set_judge_mode(mode: str):
    """Set the judge mode ('api', 'volcengine', or 'local') and reinitialize client."""
    global client, _judge_mode
    _judge_mode = mode
    os.environ['JUDGE_MODE'] = mode
    client = None  # Force reinitialization on next use

# Cache for API results to avoid redundant calls
_judge_cache = {}

# Per-call telemetry (only appended for real API calls, not cache hits)
import threading as _threading
_judge_call_stats: List[dict] = []
_judge_stats_lock = _threading.Lock()


def get_judge_call_stats():
    """Return a copy of the telemetry list for completed judge API calls."""
    with _judge_stats_lock:
        return list(_judge_call_stats)


def clear_judge_call_stats():
    """Reset the telemetry list (call between phases)."""
    with _judge_stats_lock:
        _judge_call_stats.clear()

# Global settings for concurrent API calls
_batch_reward_workers = 32

# Judge model name
# API mode: qwen-max, qwen-plus, qwen-turbo
# Local mode: Qwen3-30B-A3B-Instruct (or as configured in server)
_judge_model = "qwen-max"

# Template version (set via set_template_version() from config)
_template_version = "v1"


def set_judge_model(model_name: str):
    """Set the judge model name from config."""
    global _judge_model
    _judge_model = model_name


def get_judge_model() -> str:
    """Get the current judge model name."""
    return _judge_model


def set_template_version(version: str):
    """Set the template version from config."""
    global _template_version
    _template_version = version


def get_template_version() -> str:
    """Get the current template version."""
    return _template_version


def set_batch_reward_workers(workers: int):
    """Set the number of batch reward workers from config."""
    global _batch_reward_workers
    _batch_reward_workers = workers


def _maybe_reload_concurrency():
    """Hot-reload concurrency from a control file between reward batches.

    Operator can run `echo 48 > $JUDGE_CONC_FILE` mid-training; the change
    takes effect at the next batch (since ThreadPoolExecutor is reconstructed
    per batch). No-op if the file is missing, empty, or unparseable.
    """
    ctl = os.environ.get('JUDGE_CONC_FILE', '/tmp/craft_judge_conc')
    if not ctl or not os.path.isfile(ctl):
        return
    try:
        new_v = int(open(ctl).read().strip())
    except (ValueError, OSError):
        return
    if new_v < 1 or new_v > 256:
        return
    global _batch_reward_workers
    if new_v != _batch_reward_workers:
        print(f"[Judge] concurrency hot-reload: {_batch_reward_workers} -> {new_v}", flush=True)
        _batch_reward_workers = new_v


def clear_judge_cache():
    """Clear the judge API cache. Useful between training steps to free memory."""
    global _judge_cache
    _judge_cache = {}


# ========================================
# Scoring Functions (used by both rewards and evaluation)
# ========================================

def format_score(prediction, template_version='v1'):
    """
    Check if output has correct XML format (R_fmt).

    Args:
        prediction: Model output string
        template_version: Template version ('v1'-'v5')

    Returns:
        1.0 if format is correct, 0.0 otherwise
    """
    return multi_format_score(prediction, template_version)


def accuracy_score(prediction, ground_truths, template_version='v1'):
    """
    Check if extracted answer matches ground truth (R_ans).

    IMPORTANT: First checks if output has ALL required labels for the template version.
    Only evaluates answer accuracy if format is correct.

    Args:
        prediction: Model output string
        ground_truths: List of acceptable ground truth answers
        template_version: Template version

    Returns:
        1.0 if answer matches, 0.0 otherwise
    """
    # First check if format is correct (all required labels present)
    format_correct = format_score(prediction, template_version)
    if format_correct == 0.0:
        return 0.0  # If format is wrong, accuracy reward is 0

    return multi_accuracy_score(prediction, ground_truths, template_version)


def relevance_score(prediction, ground_truths_list, template_version='v1'):
    """
    Evaluate quality of source citations (R_gold).

    IMPORTANT: First checks if output has ALL required labels for the template version.
    Only evaluates document selection if format is correct.

    Args:
        prediction: Model output string
        ground_truths_list: List of ground truth supporting document IDs
        template_version: Template version

    Returns:
        1.0 for perfect match, 0.5 for partial overlap, 0.0 for no overlap
        For templates without gold_docs (v3-v5), returns 1.0 if format correct, else 0.0
    """
    # First check if format is correct (all required labels present)
    format_correct = format_score(prediction, template_version)
    if format_correct == 0.0:
        return 0.0  # If format is wrong, relevance reward is 0

    return multi_relevance_score(prediction, ground_truths_list, template_version)


# ========================================
# Reward Functions (wrappers for GRPO training)
# ========================================


def format_reward(completions, template_version=None, **kwargs):
    """
    Reward for correct output format (R_fmt).

    Args:
        completions: List of completion strings or list of message dicts
        template_version: Template version (None defaults to 'v1')
        **kwargs: Additional arguments (ignored)

    Returns:
        List of reward scores (1.0 for correct format, 0.0 otherwise)
    """
    try:
        completion_contents = [completion[0]["content"] for completion in completions]
    except:
        completion_contents = completions

    # Handle template_version flexibly (None, single value, or list)
    if template_version is None:
        template_versions = ['v1'] * len(completion_contents)
    elif isinstance(template_version, list):
        template_versions = template_version
    else:
        template_versions = [template_version] * len(completion_contents)

    scores = []
    for content, tv in zip(completion_contents, template_versions):
        scores.append(format_score(content, tv))
    return scores


def accuracy_reward(completions, answers, template_version=None, **kwargs):
    """
    Reward for correct answer (R_ans).

    Args:
        completions: List of completion strings or list of message dicts
        answers: List of ground truth answers (list of lists)
        template_version: Template version
        **kwargs: Additional arguments (ignored)

    Returns:
        List of reward scores (1.0 for correct, 0.0 otherwise)
    """
    try:
        completion_contents = [completion[0]["content"] for completion in completions]
    except:
        completion_contents = completions

    # Handle template_version flexibly (None, single value, or list)
    if template_version is None:
        template_versions = ['v1'] * len(completion_contents)
    elif isinstance(template_version, list):
        template_versions = template_version
    else:
        template_versions = [template_version] * len(completion_contents)

    scores = []
    for content, answer, tv in zip(completion_contents, answers, template_versions):
        scores.append(accuracy_score(content, answer, tv))

    assert len(scores) == len(completions)
    return scores


def relevance_reward(completions, supporting_ids, template_version=None, **kwargs):
    """
    Reward for retrieving relevant supporting documents (R_gold).

    IMPORTANT: This reward should ONLY be used for templates with document selection:
    - v1/v2: Have <gold_docs> tag - relevance reward is meaningful
    - v3/v4/v5: NO <gold_docs> tag - relevance reward always returns 1.0 (not meaningful)

    For v3-v5 training, do NOT include this reward in reward_funcs to avoid
    diluting the signal from format_reward and accuracy_reward.

    Args:
        completions: List of completion strings or list of message dicts
        supporting_ids: List of ground truth supporting document IDs
        template_version: Template version
        **kwargs: Additional arguments (ignored)

    Returns:
        List of reward scores
    """
    try:
        completion_contents = [completion[0]["content"] for completion in completions]
    except:
        completion_contents = completions

    # Handle template_version flexibly (None, single value, or list)
    if template_version is None:
        template_versions = ['v1'] * len(completion_contents)
    elif isinstance(template_version, list):
        template_versions = template_version
    else:
        template_versions = [template_version] * len(completion_contents)

    scores = []
    for content, support_ids, tv in zip(completion_contents, supporting_ids, template_versions):
        # For v3/v4/v5, no gold_docs needed, return 1.0
        if tv not in ['v1', 'v2']:
            scores.append(1.0)
            continue

        # For v1/v2, check if model actually output <gold_docs>
        has_tag = '<gold_docs>' in content.lower()

        if not has_tag:
            # Model should have output the tag but didn't - format violation
            scores.append(0.0)
            continue

        # Model output the expected tag, evaluate the content
        scores.append(relevance_score(content, support_ids, tv))

    return scores


# ========================================
# Judge-based Reward Functions (R_faith)
# ========================================


def extract_question_documents_from_messages(messages):
    """
    Extract question and documents from messages list.

    Args:
        messages: List of message lists

    Returns:
        Tuple of (questions, documents) lists
    """
    questions = []
    documents = []

    for msg_list in messages:
        # Find user message (last message usually)
        user_msg = None
        for msg in msg_list:
            if msg.get('role') == 'user':
                user_msg = msg.get('content', '')

        if user_msg:
            # Extract question
            q_match = re.search(r'<question>(.*?)</question>', user_msg, re.DOTALL)
            if q_match:
                questions.append(q_match.group(1).strip())
            else:
                questions.append("")

            # Extract documents
            d_match = re.search(r'<documents>(.*?)</documents>', user_msg, re.DOTALL)
            if d_match:
                documents.append(d_match.group(1).strip())
            else:
                documents.append("")
        else:
            questions.append("")
            documents.append("")

    return questions, documents


_SCORE_FIELDS = (
    'plan_reason_consistency',
    'receipt_reason_consistency',
    'reason_answer_consistency',
    'evidence_grounded_faithfulness',
    'overall_consistency',
)

_APPLICABLE_JUDGE_FIELDS = {
    'v1': (
        'plan_reason_consistency',
        'receipt_reason_consistency',
        'reason_answer_consistency',
        'evidence_grounded_faithfulness',
    ),
    'v2': (
        'receipt_reason_consistency',
        'reason_answer_consistency',
        'evidence_grounded_faithfulness',
    ),
    'v3': (
        'plan_reason_consistency',
        'reason_answer_consistency',
        'evidence_grounded_faithfulness',
    ),
    'v4': (
        'reason_answer_consistency',
        'evidence_grounded_faithfulness',
    ),
}


def _strict_overall_consistency(scores: dict, template_version: str) -> float:
    """Derive the evaluation verdict from the applicable binary checks."""
    fields = _APPLICABLE_JUDGE_FIELDS.get(
        template_version, _APPLICABLE_JUDGE_FIELDS['v1']
    )
    return float(all(float(scores.get(field, 0.0)) >= 0.5 for field in fields))


def _regex_extract_scores(xml_text: str) -> dict:
    """
    Robust regex-based extractor used as a fallback (or primary) when the
    judge's <rationale> contains characters that look like XML tags
    (e.g., '<reason>', '<doc [3]>'), which break the standard ET parser.

    Strategy: just find <field>VALUE</field> pairs by name. The score values
    are simple numbers, so this is unambiguous and never false-positives on
    rationale text.
    """
    out = {}
    for f in _SCORE_FIELDS:
        m = re.search(rf"<{f}>\s*([01](?:\.\d+)?)\s*</{f}>", xml_text)
        out[f] = float(m.group(1)) if m else 0.0

    # failure_modes: extract <mode>...</mode> entries inside <failure_modes>...
    fm = []
    fm_block = re.search(r"<failure_modes>(.*?)</failure_modes>", xml_text, re.DOTALL)
    if fm_block:
        for m in re.finditer(r"<mode>\s*(.*?)\s*</mode>", fm_block.group(1), re.DOTALL):
            fm.append(m.group(1).strip())
    out['failure_modes'] = fm

    # rationale: extract everything between the rationale tags (rationale text
    # may itself contain '<...>' that we do NOT want to interpret as XML)
    rat = re.search(r"<rationale>\s*(.*?)\s*</rationale>", xml_text, re.DOTALL)
    out['rationale'] = rat.group(1).strip() if rat else ""

    return out


def parse_judge_xml(xml_text: str) -> dict:
    """
    Parse judge XML output and extract scores.

    Robust to a common failure mode: the judge model's <rationale> often
    references trace tag names (e.g. "doc [3]", "<plan>", "<reason>") which
    confuse a strict XML parser. We try ElementTree first and fall back to
    regex-based field extraction if it fails. Both paths return the same dict
    schema so callers don't have to care which one ran.
    """
    if not xml_text:
        return {**{f: 0.0 for f in _SCORE_FIELDS},
                'failure_modes': ['empty_response'], 'rationale': ''}

    # Try ElementTree first (fast and strict).
    try:
        root = ET.fromstring(xml_text)
        scores = {}
        for f in _SCORE_FIELDS:
            elem = root.find(f)
            scores[f] = float(elem.text.strip()) if (elem is not None and elem.text) else 0.0
        # Failure modes
        modes = []
        fm_elem = root.find('failure_modes')
        if fm_elem is not None:
            for me in fm_elem.findall('mode'):
                if me.text:
                    modes.append(me.text.strip())
        scores['failure_modes'] = modes
        # Rationale
        rat_elem = root.find('rationale')
        scores['rationale'] = (rat_elem.text or "").strip() if rat_elem is not None else ""
        return scores
    except Exception:
        # Fall back to regex extraction (handles angle brackets in rationale).
        try:
            return _regex_extract_scores(xml_text)
        except Exception as e:
            return {**{f: 0.0 for f in _SCORE_FIELDS},
                    'failure_modes': ['parse_error'],
                    'rationale': f"Failed to parse: {str(e)}"}


def _call_judge_api(query: str, docs: str, model_output: str, template_version: str) -> dict:
    """
    Call judge API for a single item, with REAL retries.

    Contract: this function MUST return a parsed-score dict produced by a
    real judge response. It never silently substitutes zero-default scores
    for transient failures. After exhausting the retry budget it raises.

    Returns dict with parsed scores plus telemetry fields:
        _latency_s, _input_tokens, _output_tokens, _total_tokens
    """
    import time as _time

    # Select appropriate prompt formatter based on template version
    if template_version == 'v1':
        prompt_text = format_v1_judge_prompt(query, docs, model_output)
    elif template_version == 'v2':
        prompt_text = format_v2_judge_prompt(query, docs, model_output)
    elif template_version == 'v3':
        prompt_text = format_v3_judge_prompt(query, docs, model_output)
    elif template_version == 'v4':
        prompt_text = format_v4_judge_prompt(query, docs, model_output)
    else:
        prompt_text = format_v1_judge_prompt(query, docs, model_output)

    # Bounded-retry policy: retry for ~5 min, then RAISE so training stops loudly.
    # No silent zero-default fallback — operator wants any persistent judge
    # failure to surface immediately.
    deadline_s = float(os.environ.get('JUDGE_RETRY_DEADLINE_S', '720'))  # 12 min — fits 3× 200s attempts + backoff
    started = _time.time()
    attempt = 0
    raw_response = ""
    parsed_scores = None
    last_err_str = ""
    while True:
        try:
            _t0 = _time.time()
            _model_name = get_judge_model()
            _extra_kwargs = {}
            # Qwen3 series on DashScope require enable_thinking=False; otherwise
            # the API rejects non-streaming calls or pays the full thinking-mode
            # cost (3K+ extra tokens per call). qwen-max/turbo/doubao unaffected.
            if _model_name.lower().startswith('qwen3'):
                _extra_kwargs['extra_body'] = {'enable_thinking': False}
            response = get_judge_client().chat.completions.create(
                model=_model_name,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.0,
                timeout=200.0,  # per-attempt timeout — explicit, no client-level retry
                **_extra_kwargs,
            )
            _latency_s = _time.time() - _t0

            raw_response = response.choices[0].message.content or ""
            raw_response = raw_response.strip()
            if not raw_response:
                raise RuntimeError("Judge returned empty response")

            parsed_scores = parse_judge_xml(raw_response)

            # Treat parse_error as retryable — the model may produce cleaner XML next time.
            if parsed_scores.get('failure_modes') == ['parse_error']:
                raise RuntimeError(f"XML parse failed: {parsed_scores.get('rationale','')[:200]}")

            # The judge occasionally emits an overall label that contradicts its
            # own component scores. Evaluation uses the defined conjunction, so
            # derive it deterministically and retain the raw label for auditing.
            reported_overall = float(parsed_scores.get('overall_consistency', 0.0))
            parsed_scores['judge_reported_overall_consistency'] = reported_overall
            parsed_scores['overall_consistency'] = _strict_overall_consistency(
                parsed_scores, template_version
            )

            parsed_scores['raw_response'] = raw_response
            break  # success
        except Exception as e:
            attempt += 1
            last_err_str = f"{type(e).__name__}: {str(e)[:200]}"

            # Permanent failure: provider-side content filter rejection.
            # DashScope returns BadRequestError 400 with code=data_inspection_failed
            # when the judge input hits its moderation policy. Same input is
            # deterministically rejected — retrying for 12 min then stopping training
            # makes long sweeps brittle. We treat the sample as worst-case unfaithful
            # (all 5 binary checks = 0) so reward signal is conservative, log a clear
            # warning, and continue. Cannot distinguish whether query / docs /
            # completion triggered the filter, so this is a pragmatic engineering
            # compromise (small bias toward 0) rather than bias-free.
            # Other 400s (param errors, model_not_found, etc.) still raise.
            err_str = str(e)
            is_content_filter = (
                type(e).__name__ == 'BadRequestError'
                and ('data_inspection_failed' in err_str
                     or 'inappropriate content' in err_str.lower())
            )
            if is_content_filter:
                print(f"[Judge][CONTENT-FILTER] sample rejected by provider content filter "
                      f"(attempt #{attempt}, prompt={len(prompt_text)} chars). "
                      f"Returning overall_consistency=0 and continuing training. "
                      f"Error: {last_err_str}", flush=True)
                parsed_scores = {
                    'plan_reason_consistency': 0,
                    'receipt_reason_consistency': 0,
                    'reason_answer_consistency': 0,
                    'evidence_grounded_faithfulness': 0,
                    'overall_consistency': 0,
                    'failure_modes': ['content_filter'],
                    'rationale': 'judge input rejected by provider content filter',
                    'raw_response': '',
                }
                _latency_s = 0.0
                # Stub usage so the after-loop telemetry extractor stays simple.
                class _Stub:
                    usage = type('U', (), {'prompt_tokens': 0,
                                           'completion_tokens': 0,
                                           'total_tokens': 0})()
                response = _Stub()
                break  # exit retry loop with valid parsed_scores

            elapsed = _time.time() - started
            if elapsed >= deadline_s:
                # Budget exhausted — RAISE so training stops loudly. No silent fallback.
                msg = (f"[JUDGE-FAIL] retry budget exhausted after {attempt} attempts "
                       f"({elapsed/60:.1f} min). Stopping training. Last error: {last_err_str}")
                print(msg, flush=True)
                raise RuntimeError(msg)
            wait = min(2 ** min(attempt, 6), 60)
            # Throttle log noise: log first 3, every 5th, then every 25th.
            if attempt <= 3 or (attempt <= 50 and attempt % 5 == 0) or attempt % 25 == 0:
                print(f"[Judge][retry #{attempt} @ {elapsed:.0f}s] {last_err_str} "
                      f"→ sleeping {wait}s and retrying")
            _time.sleep(wait)

    # Attach per-call telemetry. Both API and local vLLM servers populate
    # response.usage; guard with getattr in case some backend omits it.
    usage = getattr(response, 'usage', None)
    parsed_scores['_latency_s'] = _latency_s
    parsed_scores['_input_tokens'] = int(getattr(usage, 'prompt_tokens', 0) or 0) if usage else 0
    parsed_scores['_output_tokens'] = int(getattr(usage, 'completion_tokens', 0) or 0) if usage else 0
    parsed_scores['_total_tokens'] = int(getattr(usage, 'total_tokens', 0) or 0) if usage else 0

    # Append to global telemetry (only for real API calls; cache hits skip this).
    stat = {
        "latency_s": _latency_s,
        "input_tokens": parsed_scores['_input_tokens'],
        "output_tokens": parsed_scores['_output_tokens'],
        "total_tokens": parsed_scores['_total_tokens'],
        "wall_time": _time.time(),
        # binary judge breakdown so we can rebuild any per-step distribution offline
        "plan_reason_consistency":        parsed_scores.get('plan_reason_consistency', 0.0),
        "receipt_reason_consistency":     parsed_scores.get('receipt_reason_consistency', 0.0),
        "reason_answer_consistency":      parsed_scores.get('reason_answer_consistency', 0.0),
        "evidence_grounded_faithfulness": parsed_scores.get('evidence_grounded_faithfulness', 0.0),
        "overall_consistency":            parsed_scores.get('overall_consistency', 0.0),
    }
    with _judge_stats_lock:
        _judge_call_stats.append(stat)
        # Optionally persist to disk (set JUDGE_STATS_FILE env var to enable).
        # Each call appends one JSON line. Thread-safe under _judge_stats_lock.
        _stats_file = os.environ.get('JUDGE_STATS_FILE')
        if _stats_file:
            try:
                # Lazy-create the directory so callers don't have to.
                _dir = os.path.dirname(_stats_file)
                if _dir and not os.path.isdir(_dir):
                    os.makedirs(_dir, exist_ok=True)
                with open(_stats_file, "a") as _f:
                    _f.write(json.dumps(stat) + "\n")
            except Exception:
                # Never let telemetry I/O kill a training step
                pass

    return parsed_scores


def _call_judge_batch(items: List[Tuple[str, str, str, str]]) -> List[dict]:
    """
    Call judge API for multiple items concurrently.

    Args:
        items: List of (query, docs, model_output, template_version) tuples

    Returns:
        List of parsed score dicts in the same order as input
    """
    results = [None] * len(items)
    uncached_items = []
    uncached_indices = []

    # First pass: check cache
    for i, (query, docs, model_output, template_version) in enumerate(items):
        cache_key = f"{query}::{docs}::{model_output}::{template_version}"

        if cache_key in _judge_cache:
            results[i] = _judge_cache[cache_key]
        else:
            uncached_items.append((query, docs, model_output, template_version, cache_key))
            uncached_indices.append(i)

    # If all cached, return immediately
    if not uncached_items:
        return results

    # Batch API calls for uncached items.
    # IMPORTANT: do NOT silently substitute zero-default scores for failures.
    # Every call must complete successfully (`_call_judge_api` retries internally).
    # If any call genuinely fails after all retries, we re-raise so training stops
    # loudly rather than corrupting reward signal with phantom zeros.
    _maybe_reload_concurrency()  # hot-reload from control file if changed
    with ThreadPoolExecutor(max_workers=_batch_reward_workers) as executor:
        futures = {}
        for idx, (query, docs, model_output, template_version, cache_key) in enumerate(uncached_items):
            future = executor.submit(_call_judge_api, query, docs, model_output, template_version)
            futures[future] = (uncached_indices[idx], cache_key)

        for future in as_completed(futures):
            i, cache_key = futures[future]
            parsed_scores = future.result()  # Re-raises if call failed beyond retry budget
            _judge_cache[cache_key] = parsed_scores
            results[i] = parsed_scores

    return results


def judge_overall_consistency_reward(completions, question=None, documents=None, messages=None, template_version=None, **kwargs):
    """
    Judge-based reward for overall trace consistency (R_faith).

    This is the primary judge reward that evaluates the overall consistency of the trace.
    Calculates the mean of the applicable individual metrics based on the template version:

    - v1: Mean of 4 metrics (plan_reason, receipt_reason, reason_answer, evidence_faithfulness)
    - v2: Mean of 3 metrics (receipt_reason, reason_answer, evidence_faithfulness)
    - v3: Mean of 3 metrics (plan_reason, reason_answer, evidence_faithfulness)
    - v4: Mean of 2 metrics (reason_answer, evidence_faithfulness)

    Args:
        completions: List of completion strings or list of message dicts
        question: List of questions (optional, extracted from messages if not provided)
        documents: List of document strings (optional, extracted from messages if not provided)
        messages: List of message lists (optional, used to extract question/documents)
        template_version: Template version
        **kwargs: Additional arguments

    Returns:
        List of reward scores (continuous values in [0, 1] based on mean of applicable metrics)
    """
    # Extract question and documents from messages if not provided
    if (question is None or documents is None) and messages is not None:
        question, documents = extract_question_documents_from_messages(messages)

    if question is None or documents is None or len(question) == 0:
        raise ValueError("question and documents are required for judge rewards, and could not be extracted from messages")

    try:
        completion_contents = [completion[0]["content"] for completion in completions]
    except:
        completion_contents = completions

    # Handle template_version flexibly
    if template_version is None:
        # Use global template version if not provided
        template_versions = [get_template_version()] * len(completion_contents)
    elif isinstance(template_version, list):
        template_versions = template_version
    else:
        template_versions = [template_version] * len(completion_contents)

    # Prepare items for batch call
    items = list(zip(question, documents, completion_contents, template_versions))

    # Get judge scores
    all_scores = _call_judge_batch(items)

    # Calculate mean of applicable metrics based on template version
    scores = []
    for score_dict, tv in zip(all_scores, template_versions):
        fields = _APPLICABLE_JUDGE_FIELDS.get(tv, _APPLICABLE_JUDGE_FIELDS['v1'])
        metrics = [score_dict.get(field, 0.0) for field in fields]

        # Calculate mean of applicable metrics
        mean_score = sum(metrics) / len(metrics)
        scores.append(mean_score)

    return scores
