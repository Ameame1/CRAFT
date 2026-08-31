"""Build MultiHop-RAG test set in CRAFT v1 format with oracle gold + sampled distractors."""
import json, random
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
SRC = Path("/tmp/multihoprag_dl")
OUT = ROOT / "data/test_ood/multihoprag_v1.jsonl"

# Load
queries = json.load(open(SRC / "MultiHopRAG.json"))
corpus = json.load(open(SRC / "corpus.json"))

# Index corpus by (title, url) for matching evidence
corpus_by_title = {a['title']: a for a in corpus}

# Read CRAFT v1 prompt template from existing test data so format matches.
# The in-dist prompt structure is:
#   [system header + format spec + example + closing] + "**User**:\n\n<query>...</query>\n\n<docs>...</docs>"
# We need everything BEFORE "**User**:" as the prefix, then we append our own User block.
import re
sys_template = json.loads(next(open(ROOT / "data/test/hotpotqa_2000_v1.jsonl"))).get('prompt', [{}])[0].get('content', '')
m = re.search(r'\*\*User\*\*:', sys_template)
prefix_template = sys_template[:m.start()] if m else ""

print(f"loaded {len(queries)} queries, {len(corpus)} articles")
print(f"prefix template (first 250):\n{prefix_template[:250]}\n---")

N_DOCS = 10  # gold + distractors, matching HotpotQA-distractor
N_TRUNC = 1200  # truncate body to 1200 chars to keep prompt under 8K

def trunc(s, n=N_TRUNC):
    s = s.strip()
    return (s[:n] + "...") if len(s) > n else s

skipped = 0
out_rows = []
for qi, q in enumerate(queries):
    # collect unique gold titles from evidence_list
    gold_titles = []
    for ev in q.get('evidence_list', []):
        t = ev.get('title')
        if t and t not in gold_titles and t in corpus_by_title:
            gold_titles.append(t)
    if not gold_titles:
        skipped += 1
        continue
    # cap golds to <=5 to leave room for distractors
    gold_titles = gold_titles[:min(5, N_DOCS-2)]
    n_gold = len(gold_titles)
    # sample distractors
    pool = [a for a in corpus if a['title'] not in gold_titles]
    distractors = random.sample(pool, min(N_DOCS - n_gold, len(pool)))
    docs = [corpus_by_title[t] for t in gold_titles] + distractors
    # shuffle
    random.shuffle(docs)
    supporting_ids = [i for i, d in enumerate(docs) if d['title'] in set(gold_titles)]

    # render docs as 1-indexed "N. Title: body" matching in-dist CRAFT format exactly
    docs_text = ""
    for i, d in enumerate(docs):
        body = trunc(d.get('body', ''))
        docs_text += f"\n{i+1}. {d['title']}: {body}\n"
    supporting_ids_1idx = [i + 1 for i in supporting_ids]

    user_content = f"{prefix_template}**User**:\n\n<query>\n{q['query']}\n</query>\n\n<docs>{docs_text}</docs>"

    out_rows.append({
        "id": f"mhr_{qi}",
        "prompt": [{"role": "user", "content": user_content}],
        "answers": [q.get('answer', '')],
        "supporting_ids": supporting_ids_1idx,
        "template_version": "v1",
        "name": "multihoprag",
        "question_type": q.get("question_type", ""),
    })

print(f"skipped (no resolvable gold): {skipped}")
print(f"output rows: {len(out_rows)}")
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    for r in out_rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# stats
prompt_lens = [len(r['prompt'][0]['content']) for r in out_rows]
import statistics
print(f"prompt char-len: mean={statistics.mean(prompt_lens):.0f} p50={sorted(prompt_lens)[len(prompt_lens)//2]} p95={sorted(prompt_lens)[int(len(prompt_lens)*0.95)]} max={max(prompt_lens)}")
print(f"~tokens (chars/4): mean={statistics.mean(prompt_lens)/4:.0f} p95={sorted(prompt_lens)[int(len(prompt_lens)*0.95)]/4:.0f} max={max(prompt_lens)/4:.0f}")
print(f"\nWritten: {OUT}")
