"""Rescore FanOutQA results using per-value matching instead of strict JSON EM.

Gold is a JSON-stringified dict like {"key1":"val1","key2":"val2",...}.
Prediction is a free-text string. For each value in gold, check if it appears
(after normalization) in the prediction. Score = #matched / #total.

Also computes:
- value_em: 1.0 only if ALL values matched (strict per-key)
- value_recall: fraction of gold values found in prediction (lenient)
- value_f1: harmonic mean of precision and recall on value-tokens
"""
import json, re, sys, string
from pathlib import Path
from collections import Counter


def normalize(s: str) -> str:
    s = s.lower()
    # remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # strip punctuation
    exclude = set(string.punctuation + "'`´’")
    s = ''.join(c if c not in exclude else ' ' for c in s)
    s = re.sub(r'_', ' ', s)
    return ' '.join(s.split())


def parse_gold_dict(gt_str):
    """Try to parse as JSON dict. Returns list of value strings, or [gt_str] if not a dict."""
    try:
        d = json.loads(gt_str)
        if isinstance(d, dict):
            # values may be strings, numbers, or lists
            vals = []
            for v in d.values():
                if isinstance(v, list):
                    vals.extend(str(x) for x in v)
                else:
                    vals.append(str(v))
            return vals
    except (json.JSONDecodeError, TypeError):
        pass
    return [gt_str]


def score_one(prediction, gt_str):
    pred_norm = normalize(prediction)
    pred_tokens = set(pred_norm.split())
    gold_values = parse_gold_dict(gt_str)
    n = len(gold_values)
    if n == 0:
        return None
    # value_recall: fraction of values whose normalized form appears as substring
    matched = 0
    matched_value_tokens = set()
    all_value_tokens = set()
    for v in gold_values:
        vnorm = normalize(v)
        if vnorm and vnorm in pred_norm:
            matched += 1
        for t in vnorm.split():
            all_value_tokens.add(t)
        if vnorm in pred_norm:
            for t in vnorm.split():
                matched_value_tokens.add(t)
    value_recall = matched / n
    value_em = 1.0 if matched == n else 0.0
    # token-level f1 over value-content tokens
    common = pred_tokens & all_value_tokens
    p = len(common) / max(1, len(pred_tokens)) if pred_tokens else 0
    r = len(common) / max(1, len(all_value_tokens)) if all_value_tokens else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {'value_em': value_em, 'value_recall': value_recall, 'value_f1': f1, 'n_gold_values': n, 'n_matched': matched}


def rescore(results_jsonl):
    rows = [json.loads(l) for l in open(results_jsonl) if l.strip()]
    n = len(rows)
    sums = {'value_em': 0, 'value_recall': 0, 'value_f1': 0}
    for r in rows:
        gt = r.get('ground_truth', [''])
        gt_str = gt[0] if isinstance(gt, list) and gt else (gt if isinstance(gt, str) else '')
        s = score_one(r.get('prediction', '') or r.get('full_response', ''), gt_str)
        if s is None:
            continue
        for k in sums:
            sums[k] += s[k]
    return {k: sums[k] / n for k in sums}


if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parents[2]
    print(f"{'model':18s}  {'value_em':>10s}  {'value_recall':>12s}  {'value_f1':>10s}")
    for tag in ('base', 'rscraft_sft', 'craft_nojudge', 'craft_full'):
        p = ROOT / f'data/results/ood/{tag}/fanoutqa/results.jsonl'
        if p.exists():
            r = rescore(p)
            print(f"{tag:18s}  {r['value_em']*100:>9.2f}%  {r['value_recall']*100:>11.2f}%  {r['value_f1']*100:>9.2f}%")
