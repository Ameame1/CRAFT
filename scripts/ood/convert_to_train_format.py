"""Convert existing v1 test jsonl -> training-format jsonl.

Training format (per data/train/grpo/grpo_25000_v1_messages.jsonl):
  prompt = [
    {"role": "system", "content": SYS_TEMPLATE},
    {"role": "user", "content": "<question>QUESTION</question><documents>[1] doc1...[N] docN</documents>"}
  ]

Where SYS_TEMPLATE is the training-time system message with format-spec placeholders.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Pull the EXACT system message used during training
TRAIN_SAMPLE = json.loads(open(ROOT / "data/train/grpo/grpo_25000_v1_messages.jsonl").readline())
SYS_MSG = TRAIN_SAMPLE['messages'][0]['content']
print(f"system msg len: {len(SYS_MSG)} chars")
print(f"system msg first 100: {SYS_MSG[:100]!r}")
print()

def extract_qd(prompt_content: str):
    """Extract (question, list_of_doc_strings) from a v1-style prompt content.

    The prompt may have the question/docs duplicated (once in prefix, once after **User**:).
    Take the version after **User**: if present, else the only one.
    """
    if "**User**:" in prompt_content:
        body = prompt_content.split("**User**:")[-1]
    else:
        body = prompt_content
    qm = re.search(r'<query>\s*(.*?)\s*</query>', body, re.DOTALL)
    dm = re.search(r'<docs>\s*(.*?)\s*</docs>', body, re.DOTALL)
    if not qm or not dm:
        return None, None
    question = qm.group(1).strip()
    # Parse docs and normalize to training format: each doc is `[N] Title: body`
    # with NO internal newlines (training docs have max 1 newline = trailing).
    # Collapse all internal whitespace runs to single spaces within each doc body.
    docs_block = dm.group(1).strip()
    out_docs = []
    cur_lines = []
    for ln in docs_block.split('\n'):
        m = re.match(r'^(\d+)\. ', ln)
        if m and cur_lines:
            joined = ' '.join(s.strip() for s in cur_lines if s.strip())
            out_docs.append(joined)
            cur_lines = [f"[{m.group(1)}] " + ln[len(m.group(0)):]]
        elif m:
            cur_lines = [f"[{m.group(1)}] " + ln[len(m.group(0)):]]
        else:
            cur_lines.append(ln)
    if cur_lines:
        joined = ' '.join(s.strip() for s in cur_lines if s.strip())
        out_docs.append(joined)
    return question, out_docs

def convert(in_path: Path, out_path: Path):
    rows_in = [json.loads(l) for l in open(in_path) if l.strip()]
    rows_out = []
    skipped = 0
    for r in rows_in:
        prompt_content = r['prompt'][0]['content']
        question, docs = extract_qd(prompt_content)
        if question is None:
            skipped += 1; continue
        # Match training format exactly: docs joined with single `\n`, no trailing `\n` before </documents>
        # Result: <documents>[1] body1\n[2] body2\n...[N] bodyN</documents>
        user_msg = f"<question>{question}</question><documents>" + '\n'.join(docs) + "</documents>"
        new_r = dict(r)
        new_r['prompt'] = [
            {"role": "system", "content": SYS_MSG},
            {"role": "user", "content": user_msg},
        ]
        rows_out.append(new_r)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    # stats
    user_lens = [len(r['prompt'][1]['content']) for r in rows_out]
    user_lens.sort()
    print(f"  {in_path.name} -> {out_path.name}: in={len(rows_in)} out={len(rows_out)} skipped={skipped}")
    print(f"    user msg chars: mean={sum(user_lens)//len(user_lens)} p95={user_lens[int(len(user_lens)*0.95)]} max={user_lens[-1]}  (~tokens={user_lens[-1]//4})")

if __name__ == '__main__':
    pairs = [
        (ROOT / "data/test/hotpotqa_2000_v1.jsonl", ROOT / "data/test_tf/hotpotqa_v1.jsonl"),
        (ROOT / "data/test/musique_2000_v1.jsonl",  ROOT / "data/test_tf/musique_v1.jsonl"),
        (ROOT / "data/test/2wiki_2000_v1.jsonl",    ROOT / "data/test_tf/2wiki_v1.jsonl"),
        (ROOT / "data/test_ood/multihoprag_v1.jsonl", ROOT / "data/test_tf/multihoprag_v1.jsonl"),
    ]
    for src, dst in pairs:
        convert(src, dst)
