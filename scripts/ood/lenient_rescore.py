"""Lenient rescore: if strict <answer> extraction fails AND the response is truncated,
search the partial response for the gold answer (substring match after normalization).

This recovers samples where the model produced the right answer in <reason>
but got truncated before emitting <answer>...</answer>.

Reports both strict (existing) and lenient EM/F1.
"""
import json, re, string
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

def normalize(s):
    s = s.lower()
    exclude = set(string.punctuation + "'`´’")
    s = ''.join(c if c not in exclude else ' ' for c in s)
    s = re.sub(r'_', ' ', s)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def extract_strict(full_response):
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', full_response, re.DOTALL)
    return (m.group(1).strip() if m else "")

def lenient_em(full_response, gold_answers):
    """1.0 if gold appears as substring in normalized full_response, else 0.0."""
    fr_norm = normalize(full_response)
    for g in gold_answers:
        gn = normalize(g)
        if gn and gn in fr_norm:
            return 1.0
    return 0.0

def f1(pred, gold):
    p_toks = normalize(pred).split(); g_toks = normalize(gold).split()
    if not p_toks or not g_toks: return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    n = sum(common.values())
    if n == 0: return 0.0
    p = n / len(p_toks); r = n / len(g_toks)
    return 2*p*r/(p+r)

def rescore(jsonl_path):
    rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    n = len(rows)
    sums = {'strict_em': 0, 'lenient_em': 0, 'strict_f1': 0, 'lenient_f1': 0}
    for r in rows:
        fr = r.get('full_response','')
        gt = r.get('ground_truth', [''])
        if isinstance(gt, str): gt = [gt]
        # strict
        pred = extract_strict(fr)
        sem = max(1.0 if normalize(pred) == normalize(g) else 0.0 for g in gt) if pred else 0.0
        sf1 = max(f1(pred, g) for g in gt) if pred else 0.0
        # lenient
        if pred:
            lem = sem; lf1 = sf1
        else:
            lem = lenient_em(fr, gt)
            # for f1, use the entire stripped reasoning as proxy
            reason = re.search(r'<reason>(.*?)(?:</reason>|$)', fr, re.DOTALL)
            r_text = reason.group(1) if reason else fr
            lf1 = max(f1(r_text, g) for g in gt)
        sums['strict_em'] += sem
        sums['lenient_em'] += lem
        sums['strict_f1'] += sf1
        sums['lenient_f1'] += lf1
    return {k: sums[k]/n for k in sums}

if __name__ == '__main__':
    print(f"{'tag':18s}  {'dataset':12s}  {'strict_EM':>10s}  {'lenient_EM':>11s}  {'strict_F1':>10s}  {'lenient_F1':>11s}  {'recovered':>10s}")
    print("-" * 90)
    for tag in ('craft_full',):
        for ds in ('hotpotqa', 'musique', '2wiki', 'multihoprag'):
            p = ROOT / f'data/results/tf/{tag}/{ds}/results.jsonl'
            if p.exists():
                r = rescore(p)
                rec = (r['lenient_em'] - r['strict_em']) * 100
                print(f"{tag:18s}  {ds:12s}  {r['strict_em']*100:>9.2f}%  {r['lenient_em']*100:>10.2f}%  {r['strict_f1']*100:>9.2f}%  {r['lenient_f1']*100:>10.2f}%  {rec:>+9.2f}pp")
