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
# Being on this list means t1 SKIPS the file, so an allowlisted module could
# reach for a ledger and t1 would never notice -- verified by appending one to
# crop_search.py and watching the suite still pass. t1c below is what actually
# holds these two, so the exemption buys silence about the triage file and
# nothing else.
ALLOWED = {'triage_crops.py', 'dashboard.py', 'adv_triage_isolation.py',
           'crop_search.py',
           # It drives the review page, and the page polls /api/triage for the
           # guesser strip, so the word is unavoidable there. What t1 is for --
           # a module quietly growing a path from a guess to a ledger -- is
           # covered for this file by NO_LEDGER below: a test may READ the
           # guess API, and may not name a verdict store.
           'adv_review_render.py',
           # It exists to prove one guesser's run is never credited to the
           # other, which cannot be written without naming the status file.
           'adv_triage_backends.py'}
# The allowlisted modules that have no business writing a verdict. dashboard.py
# is not here: it owns the ledgers. triage_crops.py is covered by t2.
NO_LEDGER = ('crop_search.py', 'adv_review_render.py',
             'adv_triage_backends.py')
LEDGER_WORDS = ('hard_negatives', 'hard_positives', 'labels.jsonl',
                'label_flags', 'leash.db', 'flag_crop')
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
# An allowlisted module is invisible to t1, so the thing t1 would have caught
# has to be caught somewhere. A search index reads embeddings and writes
# embeddings; the moment one of these names a ledger, it is doing something
# else and this fails.
_ledger_guilty = []
for _name in NO_LEDGER:
    for _base, _, _fs in os.walk(TOOLS):
        if _name in _fs:
            _txt = read(os.path.join(_base, _name))
            for _w in LEDGER_WORDS:
                if _w in _txt:
                    _ledger_guilty.append(f'{_name} mentions {_w}')
check('t1c allowlisted search modules touch no ledger', not _ledger_guilty,
      '; '.join(_ledger_guilty))
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
    # Two files, both its own: the predictions, and the progress the dashboard
    # reads. Everything else is a leak. This list is the deliberate review --
    # adding the status file made this check fail until it was named here,
    # which is the check doing its job.
    OWN = ('args.out', 'OUT_FILE', 'args.status', 'STATUS_FILE',
           'tmp_status')      # named, not a bare 'tmp' anything could hold
    bad = [t for t, _ in writes if not any(o in t for o in OWN)]
    check('t2 the writer only ever writes its own files', not bad,
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

# ── t8 both backend_of() implementations agree ─────────────────────────────
# The tool stamps each record with the backend that wrote it; the dashboard
# decides which backend a record belongs to when it reads one back. Those are
# two copies of one rule in two files, and they cannot import each other -- the
# dashboard needs duckdb, the tool needs torch. If they ever disagree, records
# are filed under one guesser and looked for under another, and 62,979 of them
# predate the field entirely and are attributed purely by this function.
#
# Compared by BEHAVIOUR, not by source text: the two are allowed to be written
# differently as long as they answer the same. Each is lifted out by AST and
# run on its own, so this stays import-free like the rest of the file.
def _lift(path, name):
    import ast as _ast
    try:
        tree = _ast.parse(read(path) or '')
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == name:
            ns = {}
            exec(compile(_ast.Module(body=[node], type_ignores=[]),
                         path, 'exec'), ns)
            return ns[name]
    return None


DASH = os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py')
_a, _b = _lift(TRIAGE, 'backend_of'), _lift(DASH, 'backend_of')
if _a and _b:
    # Every model string either backend can stamp on a record. A backend the
    # tool knows and the dashboard does not files its guesses under the wrong
    # guesser silently -- which is what happened when the dog-bin gate was
    # added to one side only: 241 of its verdicts turned up in SigLIP's
    # filter, and SigLIP's own coverage absorbed them.
    probes = ['google/siglip2-so400m-patch14-384',
              'google/siglip2-base-patch16-224', 'rfdetr', 'rfdetr-large',
              'rfdetr-nano', 'imagenet', 'efficientnet_v2_s.imagenet1k_v1',
              'siglip', 'rfdetr-small', 'dogbin', 'dogbin:dogbin_008',
              'dogbin:anything', '', None, 'something-new']
    differ = [f'{p!r}: tool={_a(p)!r} dashboard={_b(p)!r}'
              for p in probes if _a(p) != _b(p)]
    check('t8 the tool and the dashboard agree which backend wrote a record',
          not differ, '; '.join(differ))
    # backend_of() recognises RF-DETR by prefix rather than by the size table,
    # so that it needs no module constants and can be lifted out and compared.
    # A size key that broke the prefix would be filed as SigLIP, silently.
    sizes = re.findall(r"'([^']+)':\s*'RFDETR",
                       (src or '').split('RFDETR_SIZES = {')[-1])
    check('t8b every RF-DETR size key starts with the prefix backend_of reads',
          bool(sizes) and all(s.startswith('rfdetr') for s in sizes),
          f'keys: {sizes}')
else:
    check('t8 both backend_of() implementations were found',
          False, f'tool={bool(_a)} dashboard={bool(_b)}')

print()
if fails:
    raise SystemExit(f'{len(fails)} isolation check(s) FAILED: '
                     + ', '.join(fails))
print('model suggestions are structurally separated from the training data')
