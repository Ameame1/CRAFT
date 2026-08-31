"""Build FanOutQA test set in CRAFT v1 format. Each pos_doc is multiple wiki pages
concatenated; split on '{{Short description|' / '{{short description|' boundaries,
strip wikitext markup, take lead section, truncate to ~1500 chars/page."""
import json, re, random
from pathlib import Path
import pandas as pd

random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
SRC_PARQUET = Path("/tmp/fanoutqa_paired_dl/train/data_000000.parquet")
OUT = ROOT / "data/test_ood/fanoutqa_v1.jsonl"

df = pd.read_parquet(SRC_PARQUET)
print(f"loaded {len(df)} questions")

# Strip wikitext: templates {{...}}, refs <ref>...</ref>, links [[link|alt]] -> alt or link, headers === ===
def strip_wikitext(t: str) -> str:
    # remove <ref>...</ref>
    t = re.sub(r'<ref[^>]*?>.*?</ref>', '', t, flags=re.DOTALL)
    t = re.sub(r'<ref[^/]*?/>', '', t)
    # remove templates: {{anything}}, including nested (greedy, multi-pass)
    for _ in range(8):
        t2 = re.sub(r'\{\{[^{}]*\}\}', '', t, flags=re.DOTALL)
        if t2 == t: break
        t = t2
    # links: [[link|alt]] -> alt; [[link]] -> link
    t = re.sub(r'\[\[([^\[\]\|]+?)\|([^\[\]]+?)\]\]', r'\2', t)
    t = re.sub(r'\[\[([^\[\]\|]+?)\]\]', r'\1', t)
    # external links: [http://... text] -> text
    t = re.sub(r'\[http[^\s\[\]]+\s+([^\[\]]+?)\]', r'\1', t)
    t = re.sub(r'\[http[^\s\[\]]+\]', '', t)
    # bold/italic: '''text''' or ''text''
    t = re.sub(r"'''([^']+)'''", r'\1', t)
    t = re.sub(r"''([^']+)''", r'\1', t)
    # headers: == foo ==
    t = re.sub(r'^=+\s*([^=]+?)\s*=+$', r'\1', t, flags=re.MULTILINE)
    # collapse whitespace
    t = re.sub(r'\n\s*\n+', '\n\n', t)
    t = re.sub(r'[ \t]+', ' ', t)
    return t.strip()

def get_lead(t: str, max_chars=1500) -> str:
    """Take lead section: stop at first section header == X == ; truncate to max_chars."""
    # If headers exist after stripping, take everything up to first header line
    parts = t.split('\n')
    lead_lines = []
    cum = 0
    for ln in parts:
        if cum > max_chars: break
        # treat as section break if a line is just a noun-ish 1-5 words and is preceded by blank
        # but our regex already collapsed headers; so just truncate by chars
        lead_lines.append(ln)
        cum += len(ln) + 1
    s = '\n'.join(lead_lines)
    return s[:max_chars].strip() + ("..." if len(s) > max_chars else "")

# Detect page boundaries — wiki pages start with {{Short description|...}} or {{Use mdy dates}}
# Easier: split on the literal /\{\{[Ss]hort description\|/ — adjacent text up to next is one page
PAGE_SPLIT = re.compile(r'(?=\{\{[Ss]hort description\|)')

# Read CRAFT v1 prompt template from in-dist test data — full prefix up to **User**:
sys_template = json.loads(next(open(ROOT / "data/test/hotpotqa_2000_v1.jsonl"))).get('prompt', [{}])[0].get('content', '')
m_user = re.search(r'\*\*User\*\*:', sys_template)
prefix_template = sys_template[:m_user.start()] if m_user else ""

out_rows = []
for qi, row in df.iterrows():
    q = row['question']
    a = row['answer']  # often a JSON string with a dict — keep as-is
    pos = row['pos_doc']
    pages = [p for p in PAGE_SPLIT.split(pos) if p.strip()]
    # take up to 8 pages (fits 8 docs × 1500 chars = 12000 char prompt)
    pages = pages[:8]
    if not pages:
        # fallback: treat whole as one page
        pages = [pos]
    docs_text = ""
    for i, p in enumerate(pages):
        s = strip_wikitext(p)
        s = get_lead(s, 1500)
        first_line = s.split('\n')[0][:80] if s else f"page_{i+1}"
        docs_text += f"\n{i+1}. {first_line}: {s[:1200]}\n"
    user_content = f"{prefix_template}**User**:\n\n<query>\n{q}\n</query>\n\n<docs>{docs_text}</docs>"

    out_rows.append({
        "id": f"foq_{qi}",
        "prompt": [{"role": "user", "content": user_content}],
        "answers": [a],
        "supporting_ids": list(range(1, len(pages) + 1)),  # 1-indexed; all pages are gold
        "template_version": "v1",
        "name": "fanoutqa",
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    for r in out_rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# stats
import statistics
prompt_lens = [len(r['prompt'][0]['content']) for r in out_rows]
print(f"output rows: {len(out_rows)}")
print(f"prompt char-len: mean={statistics.mean(prompt_lens):.0f} p50={sorted(prompt_lens)[len(prompt_lens)//2]} p95={sorted(prompt_lens)[int(len(prompt_lens)*0.95)]} max={max(prompt_lens)}")
print(f"~tokens (chars/4): mean={statistics.mean(prompt_lens)/4:.0f} p95={sorted(prompt_lens)[int(len(prompt_lens)*0.95)]/4:.0f} max={max(prompt_lens)/4:.0f}")
print(f"\nWritten: {OUT}")
