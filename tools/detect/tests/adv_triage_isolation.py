#!/usr/bin/env python3
"""The model's suggestions must never become labels. Prove it from the source.

triage_crops.py writes guesses about what each unreviewed crop contains, so
the review queue can be worked one kind of mistake at a time. The standing
constraint is that those guesses are for FILTERING ONLY: they may not be
written into the annotation ledgers, may not reach the crop dataset, and may
not be used as training labels. Only a human verdict is ground truth.

A comment saying so is worth very little. These checks fail the build if any
of the following becomes true:

  t1  any module under tools/ other than the writer and the dashboard
      mentions the triage file
  t2  the triage writer opens a ledger or a dataset directory for writing
  t3  the review page's flag path lets a suggestion reach the ledger record
  t4  a triage record is missing its unverified marks
  t5  the ledgers contain a record that came from the triage file
  t6  the dashboard's suggestion fields are not separate from the verdict one
  t7  the bucket mapping silently disagrees with the ImageNet class order

Run: python tools/detect/tests/adv_triage_isolation.py
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
TOOLS = os.path.join(REPO, 'tools')
TRIAGE = os.path.join(REPO, 'tools', 'detect', 'triage_crops.py')
DASH = os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py')

# WHO IS ALLOWED to know this file exists. An allowlist of builders was the
# first attempt and it rotted immediately: three of its seven names were not
# real files and three real builders (build_detector_negatives.py,
# dedup_crops.py, reserve_acceptance_set.py) were never scanned, so the check
# that was supposed to catch a leak was reading almost nothing.
#
# Inverted, it cannot rot. EVERY python file under tools/ is scanned, and only
# these two may mention the triage file: the tool that writes it and the
# dashboard that filters with it. A new script that reads it fails this test
# until someone adds it here deliberately -- which is the review the rule
# actually needs.
ALLOWED = {'triage_crops.py', 'dashboard.py', 'adv_triage_isolation.py'}
LEDGERS = (os.path.join(REPO, 'data', 'hard_negatives', 'labels.jsonl'),
           os.path.join(REPO, 'data', 'hard_positives', 'labels.jsonl'))
TRIAGE_FILE = os.path.join(REPO, 'data', 'dashboard', 'triage.jsonl')

fails = []


def check(name, ok, detail=''):
    print(('ok   ' if ok else 'FAIL ') + name + (('  ' + detail) if detail
                                                 and not ok else ''))
    if not ok:
        fails.append(name)


def read(p):
    try:
        with open(p, encoding='utf-8') as fh:
            return fh.read()
    except OSError:
        return None


# ── t1 nothing but the writer and the filter knows this file exists ────────
scanned, guilty = 0, []
for base, _, files in os.walk(TOOLS):
    for f in files:
        if not f.endswith('.py') or f in ALLOWED:
            continue
        scanned += 1
        if re.search(r'triage', read(os.path.join(base, f)) or '', re.I):
            guilty.append(os.path.relpath(os.path.join(base, f), REPO))
check(f't1 none of the {scanned} other tools/ modules mention the triage file',
      not guilty, 'mentions triage: ' + ', '.join(guilty))
check('t1b the allowlist names files that exist', all(
    any(f in fs for _, _, fs in os.walk(TOOLS)) for f in ALLOWED),
    'an allowed name matches no file -- the list has rotted')

# ── t2 the writer never opens a ledger or dataset for writing ───────────────
src = read(TRIAGE)
if src is None:
    check('t2 triage_crops.py exists', False, TRIAGE)
else:
    # every open() with a write mode in that file
    writes = re.findall(r"open\(\s*([^)]*?),\s*['\"]([wa][^'\"]*)['\"]", src)
    bad = [t for t, _ in writes
           if 'args.out' not in t and 'OUT_FILE' not in t]
    check('t2 the writer only ever writes its own output file', not bad,
          'other write targets: ' + '; '.join(bad))
    check('t2b the writer names no ledger path', 'labels.jsonl' not in
          src.split('def judged_names')[1].split('def ')[0].replace(
              "'labels.jsonl'", '') if 'def judged_names' in src else True,
          'labels.jsonl referenced outside the read-only helper')

# ── t3 a suggestion cannot ride into a ledger record ────────────────────────
dash = read(DASH)
if dash is None:
    check('t3 dashboard.py readable', False)
else:
    fn = ''
    m = re.search(r'\ndef flag_crop\(.*?\n(?=\ndef )', dash, re.S)
    if m:
        fn = m.group(0)
    check('t3 the flag writer never touches a suggestion',
          bool(fn) and not re.search(r"\bsg\b|triage|suggest", fn),
          'flag_crop mentions the suggestion')

    # ── t6 the payload keeps guesses under their own keys ───────────────────
    check("t6 suggestion fields are separate from 'label'",
          "'sg': sg.get('b', '')" in dash and "'label'" not in
          dash.split("'sg': sg.get('b', '')")[1].split('})')[0],
          'a suggestion is written into the verdict field')

# ── t4 every record marks itself unverified ─────────────────────────────────
recs = []
if os.path.exists(TRIAGE_FILE):
    with open(TRIAGE_FILE) as fh:
        for ln in fh:
            try:
                recs.append(json.loads(ln))
            except ValueError:
                pass
if recs:
    bad = [r for r in recs if r.get('unverified') is not True
           or r.get('source') != 'model_suggestion']
    check(f't4 all {len(recs):,} records marked unverified', not bad,
          f'{len(bad)} records missing the marks')
else:
    print('ok   t4 no predictions written yet (nothing to verify)')

# ── t5 no ledger record came from the triage file ───────────────────────────
led_bad = []
for p in LEDGERS:
    try:
        with open(p) as fh:
            for i, ln in enumerate(fh, 1):
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if not isinstance(r, dict):
                    continue
                if (r.get('source') == 'model_suggestion'
                        or r.get('unverified') is True
                        or 'bucket' in r or 'sg' in r):
                    led_bad.append(f'{os.path.basename(os.path.dirname(p))}:{i}')
    except OSError:
        pass
check('t5 no ledger record carries a model suggestion', not led_bad,
      'suspect records: ' + ', '.join(led_bad[:5]))

# ── t7 the bucket edges still match the ImageNet class order ────────────────
if src:
    sys.path.insert(0, os.path.dirname(TRIAGE))
    # the keys are named constants, not literals, so count the class names
    # the block pins -- one per boundary the buckets depend on
    blk = src.split('EDGE = {')[1].split('}')[0] if 'EDGE = {' in src else ''
    edges = re.findall(r"'([^']+)'", blk)
    check('t7 bucket edges are asserted, not assumed', len(edges) == 4,
          f'found {len(edges)} edge assertions: {edges}')
    check('t7b the writer verifies them before writing',
          'ImageNet class order is not what the buckets' in src,
          'no runtime check of the class order')

print()
if fails:
    raise SystemExit(f'{len(fails)} isolation check(s) FAILED: '
                     + ', '.join(fails))
print('model suggestions are structurally separated from the training data')
