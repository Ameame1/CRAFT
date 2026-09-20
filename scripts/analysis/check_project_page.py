#!/usr/bin/env python3
"""Check displayed project-page values against the local manuscript, not raw runs."""

import argparse
import re
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.section = None
        self.tables = []
        self.table = None
        self.row = None
        self.cell = None
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'section':
            self.section = attrs.get('id')
        if tag == 'table' and self.section == 'results':
            self.table = []
        if tag == 'tr' and self.table is not None:
            self.row = []
        if tag in ('td', 'th') and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        self.text.append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(''.join(self.cell).strip())
            self.cell = None
        if tag == 'tr' and self.row is not None:
            self.table.append(self.row)
            self.row = None
        if tag == 'table' and self.table is not None:
            self.tables.append(self.table)
            self.table = None
        if tag == 'section':
            self.section = None


def table(tex, label):
    end = tex.index(r'\label{' + label + '}')
    start = max(tex.rfind(r'\begin{table}', 0, end),
                tex.rfind(r'\begin{table*}', 0, end))
    assert start >= 0, label
    return tex[start:end]


def numbers(cells):
    # Only the first numeric literal in a cell is the score; later ones are deltas.
    return [Decimal(re.search(r'-?\d+(?:\.\d+)?', c).group()) for c in cells]


def check(paper, page):
    tex = paper.read_text()
    html = page.read_text()
    parsed = Page()
    parsed.feed(html)
    text = ' '.join(' '.join(parsed.text).split())
    title = re.search(r'\\title\{([^}]+)\}', tex).group(1)
    assert f'<title>{title}</title>' in html
    assert f'<p class="paper-title">{title}</p>' in html

    main = table(tex, 'tab:main_results_v1')
    rows = [numbers(line.split('&')[2:]) for line in main.splitlines()
            if line.strip().startswith(r'\quad')]
    assert len(rows) == 17 and all(len(row) == 9 for row in rows)
    base, sft, craft = rows[3], rows[12], rows[16]
    api = [max(row[i] for row in rows[6:9]) for i in range(9)]
    expected = [
        [[row[i] for i in (0, 3, 6)] for row in (base, sft, api, craft)],
        [[row[i] for i in (2, 5, 8)] for row in (base, sft, api, craft)],
        [[row[0], row[2]] for row in rows[13:17]],
    ]
    assert len(parsed.tables) == 3
    checked = 0
    for actual, wanted in zip(parsed.tables, expected):
        values = [[Decimal(x) for x in row[1:]] for row in actual[1:]]
        assert values == wanted, (values, wanted)
        checked += sum(map(len, wanted))

    for name, wanted in [('EM_DATA', expected[0]), ('FAITH_DATA', expected[1])]:
        block = re.search(r'const ' + name + r' = \{(.*?)\n\};', html, re.S).group(1)
        actual = []
        for key in ('base', 'sft', 'api', 'craft'):
            seq = re.search(key + r':\[([^]]+)\]', block).group(1)
            actual.append([Decimal(x) for x in seq.split(',')])
        assert actual == wanted, name
        checked += sum(map(len, wanted))
    scale = re.search(r'const SCALE_DATA = \{(.*?)\n\};', html, re.S).group(1)
    for index, key in enumerate(('em', 'faith')):
        actual = numbers(re.search(key + r':\[([^]]+)\]', scale).group(1).split(','))
        assert actual == [row[index] for row in expected[2]], key
        checked += len(actual)

    ablation = table(tex, 'tab:comparison_7b_v1_v5')
    v1 = [numbers(line.split('&')[1:]) for line in ablation.splitlines()
          if line.strip().startswith(r'CRAFT$_{\text{v1}}$')]
    assert len(v1) == 4
    assert v1[0] == base and v1[1] == sft and v1[3] == craft
    deltas = [craft[0] - v1[2][0], craft[2] - v1[2][2],
              craft[6] - api[6], craft[6] - base[6],
              craft[2] - base[2], craft[2] - sft[2]]
    for delta in deltas:
        assert f'{delta:.2f}' in text, delta
    for value in (v1[2][0], v1[2][2], craft[0], api[0]):
        assert f'{value:.2f}' in text, value

    human = table(tex, 'tab:human_validation')
    dimensions = [numbers(line.split('&')[1:]) for line in human.splitlines()
                  if line.strip().startswith(('Plan-Reason &', 'Evidence Gr. &',
                                               'Answer Deriv. &', 'Gold Doc Cit. &'))]
    assert len(dimensions) == 4
    for row in dimensions:
        assert sum(row[:4]) == 500
        assert f'κ {row[-1]:.2f}' in text
    mean_agreement = sum(row[-2] for row in dimensions) / 4
    mean_kappa = sum(row[-1] for row in dimensions) / 4
    assert f'{mean_agreement:.1f}' in text
    assert f'κ {mean_kappa:.2f}' in text
    assert '500' in text
    assert 'fitted simulations' in text and 'counterfactual estimates' in text
    assert 'mean of ten 1,000-example runs' not in html
    print(f'PASS: title; {checked} table/chart score occurrences; 6 deltas; '
          'ablation anchors; human-validation counts and aggregates; provenance labels.')
    print('Scope: manuscript/page consistency only, not checkpoint reproducibility.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--page', type=Path, default=Path('index.html'))
    args = parser.parse_args()
    check(args.paper, args.page)
