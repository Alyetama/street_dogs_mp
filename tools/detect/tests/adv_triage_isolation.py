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
  t5  any store a human decision lands in holds something the model wrote --
      the flag ledgers, the looked-at-and-kept ledger, the hand-drawn
      geometry, both audit ledgers and what they export, the leash verdicts
      and the label flags; or a store grows that is in none of those lists
  t6  the dashboard's suggestion fields are not separate from the verdict one
  t7  the bucket mapping silently disagrees with the ImageNet class order

WHICH ASSERTION t5d MAKES ABOUT A BOX, AND WHY IT IS NOT THE OTHER ONE.
A correction whose four numbers equal the detector's box at the ledger's own
precision is nobody's geometry: it is Reset box putting the model's box back.
The tempting rule is "no such record may be written", and it is wrong. The
ledger is append-only and last write wins, so writing the detector's box IS
how a reviewer withdraws a box they drew earlier -- refusing the write would
leave the withdrawn box in force, and only the reviewer knows which they
meant. Four such records are on disk across three boxes, and every one of the
three has a history holding real corrections too. So the rule is the other
one: such a record must stay tellable from a real correction, and the only
thing that can tell them apart is the geometry itself. t5c keeps the fields
that make that comparison possible and t5d makes it, failing when a box is on
record in the corrections store by the detector's geometry ALONE -- no human
framing anywhere in its history. That is the one state a consumer cannot read
as anything but "a person framed this".

Run: python tools/detect/tests/adv_triage_isolation.py

t5d needs duckdb to read the predictions store, so run it under the env the
dashboard runs on; it says so and skips loudly elsewhere. Everything else
runs anywhere. Point ISOLATION_STORES at a repo root holding COPIES of the
stores to scan them somewhere else -- which is the only safe way to prove
these checks bite, since proving it means writing a poisoned verdict and a
probe session has already left fake ones in the live audit ledger.
"""

import json
import os
import re
import sqlite3
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
                'label_flags', 'leash.db', 'flag_crop',
                # the five that grew after this list was written, and that t5
                # went blind to for the same reason
                'box_corrections', 'boxes.jsonl', 'reviewed.jsonl',
                'verdicts.jsonl', 'audit_finds')

# The stores are read from here rather than from REPO so that a poisoned COPY
# can be scanned instead of the real thing. Proving any of t5 bites means
# writing a record that must never exist, and a probe session has already put
# fake verdicts into the live audit ledger by doing that in place; there is no
# command to take one back out.
STORES = os.environ.get('ISOLATION_STORES') or REPO
DATA = os.path.join(STORES, 'data')

# EVERY file a human verdict or a hand-drawn box reaches, and the model's own
# file it must stay out of. t5 used to name two of these -- the two
# labels.jsonl -- while ten more grew up beside them, so the check that was
# supposed to catch a model's opinion reaching the training data was reading
# two of the twelve places one can land, and the geometry store was not among
# them. t5g walks data/ and fails on any store that is in neither list, so the
# next one cannot go quiet the same way: classifying it is the review the rule
# actually needs.
HUMAN_STORES = ('hard_negatives/labels.jsonl',      # flag verdicts
                'hard_positives/labels.jsonl',
                'hard_negatives/reviewed.jsonl',    # looked at and kept
                'box_corrections/boxes.jsonl',      # hand-drawn geometry
                'fn_audit/verdicts.jsonl',          # audit answers
                'leash_audit/verdicts.jsonl',
                # not answers, but the denominator of every rate the audit
                # reports -- a model writing here inflates what it was asked
                'fn_audit/drawn.jsonl',
                'leash_audit/drawn.jsonl',
                'audit_finds/manifest.jsonl',       # the exported dataset
                'audit_finds_leash/manifest.jsonl')
LEASH_DB = 'leash_labels/leash.db'
FLAGS_DB = 'label_flags/label_flags.db'
MODEL_OWN = ('dashboard/triage.jsonl',)
BOXES = 'box_corrections/boxes.jsonl'
# The surfaces a person can answer from. leash_store stamps every row with the
# one it was written through, which is the whole point of the column: a batch
# import of the leash model's own parquet predictions arrives naming itself.
HUMAN_SOURCES = ('review_page',)
TRIAGE_FILE = os.path.join(DATA, 'dashboard', 'triage.jsonl')

fails = []
skipped = []


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


def records(rel):
    """[(line number, record)] from one jsonl store. Missing reads as empty.

    A fresh clone has none of these and a store nobody has written to yet is
    an empty file, so absence is not a failure anywhere below -- what would be
    a failure is a store that exists and is never opened, which is what t5g
    is for.
    """
    out = []
    try:
        with open(os.path.join(DATA, rel), encoding='utf-8') as fh:
            for i, ln in enumerate(fh, 1):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    out.append((i, r))
    except OSError:
        pass
    return out


def rows(rel, sql):
    """Query one sqlite store read-only, or None if it is not there."""
    p = os.path.join(DATA, rel)
    if not os.path.exists(p):
        return None
    con = sqlite3.connect('file:%s?mode=ro' % p, uri=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def wrote_it(r):
    """True if the model wrote this record, whatever file it turned up in.

    The triage writer's own stamps, so a record that travelled from
    triage.jsonl into a ledger is recognisable there by the marks t4 makes it
    carry -- which is the point of making it carry them.
    """
    return (r.get('source') == 'model_suggestion'
            or r.get('unverified') is True
            or 'bucket' in r or 'sg' in r)


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

# ── t5 nothing the model wrote is in a store a human decision lands in ─────
led_bad, n_recs = [], 0
for _rel in HUMAN_STORES:
    for _i, _r in records(_rel):
        n_recs += 1
        if wrote_it(_r):
            led_bad.append(f'{_rel}:{_i}')
check(f't5 none of the {n_recs:,} records in the {len(HUMAN_STORES)} human '
      f'stores carries a model suggestion', not led_bad,
      'suspect records: ' + ', '.join(led_bad[:5]))

# ── t5b the two sqlite stores hold human answers only ──────────────────────
db_bad = []
_leash = rows(LEASH_DB, 'SELECT crop, source FROM leash')
for _crop, _source in _leash or ():
    if _source not in HUMAN_SOURCES:
        db_bad.append(f'leash.db {_crop}: source {_source!r}')
# A flag whose should_be is the class the file is already filed under records
# no human decision at all -- it is the dataset's own label handed back as a
# correction, the same shape as the detector's box in the corrections ledger,
# and the training page counts it as a person disagreeing with the model.
_flags = rows(FLAGS_DB, 'SELECT file, class_was, should_be FROM flags')
for _file, _was, _be in _flags or ():
    if not _be or _be == _was:
        db_bad.append(f'label_flags.db {_file}: should_be {_be!r}')
check(f't5b the {len(_leash or ())} leash verdicts and {len(_flags or ())} '
      f'label flags on record are human answers', not db_bad,
      '; '.join(db_bad[:5]))

# ── t5c the corrections ledger still says which box, and where ─────────────
# The only thing that separates a hand-drawn box from the detector's own is
# the geometry, and comparing them needs the key to look the prediction up
# under. Drop det_idx, or start writing normalised coordinates, and t5d has
# nothing to compare -- it would pass on every record forever without ever
# reading one.
_boxes = records(BOXES)
_shape = [f'{BOXES}:{i}' for i, r in _boxes
          if not (str(r.get('image_id') or '').isdigit()
                  and isinstance(r.get('det_idx'), int)
                  and all(isinstance(r.get(k), (int, float))
                          for k in ('x1', 'y1', 'x2', 'y2')))]
check(f't5c all {len(_boxes):,} corrections carry the key and the four '
      f'numbers a comparison needs', not _shape, '; '.join(_shape[:5]))

# ── t5d a box on record by the detector's own geometry alone ───────────────
_hist = {}
for _i, _r in _boxes:
    if f'{BOXES}:{_i}' not in _shape:
        _hist.setdefault((str(_r['image_id']), int(_r['det_idx'])),
                         []).append((_r['x1'], _r['y1'], _r['x2'], _r['y2']))


def detector_boxes(keys):
    """{(image_id, det_idx): (x1, y1, x2, y2)} from the predictions store.

    Rounded the way save_box() rounds, because two decimals is the precision
    the corrections ledger speaks. Comparing at full float precision instead
    calls a byte-for-byte revert 'different' and finds two of the four
    reverts on disk rather than all four.
    """
    sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
    import duckdb
    import store as st
    det = st._sql_src(st._store_globs(st.get_detect_root(STORES), 'det'))
    ids = sorted({k[0] for k in keys})
    con = duckdb.connect()
    try:
        got = con.execute(
            'SELECT CAST(image_id AS VARCHAR), det_idx, x1, y1, x2, y2 FROM '
            + det + ' WHERE CAST(image_id AS VARCHAR) IN ('
            + ','.join('?' * len(ids)) + ')', ids).fetchall()
    finally:
        con.close()
    return {(a, int(b)): tuple(round(float(v), 2) for v in (c, d, e, f))
            for a, b, c, d, e, f in got}


_model, _why = {}, ''
if _hist:
    try:
        _model = detector_boxes(_hist)
    except Exception as _e:
        _why = str(_e)
if _why:
    # loud, and named as a skip -- a silent abort here is indistinguishable
    # from a pass, which is the failure mode two of this suite's guards were
    # already shipping
    skipped.append('t5d')
    print('SKIP t5d could not read the predictions store, so no correction '
          'was compared against the box it corrects: ' + _why)
else:
    _lone = sorted(k for k, v in _hist.items()
                   if k in _model and all(b == _model[k] for b in v))
    check('t5d no box is in the corrections store on the detector\'s own '
          'geometry alone', not _lone,
          'model geometry wearing a correction\'s clothes: '
          + ', '.join('%s#%d' % k for k in _lone[:5]))
    # Without this the whole check goes quiet the day the predictions store
    # moves: every key stops resolving, nothing is compared, and t5d prints
    # ok because it found no counterexample among the zero it looked at.
    _seen = [k for k in _hist if k in _model]
    check(f't5d-b the comparison read the detector back for {len(_seen):,} of '
          f'the {len(_hist):,} corrected boxes', bool(_seen) or not _hist,
          'not one correction resolved a prediction -- nothing was compared')

# ── t5e the review page banks no verdict a person did not make ────────────
# Paging away IS the positive verdict on everything still on screen, and that
# is the right rule for the queue. In audit mode the page walks a FIXED list
# of crops already judged, so the same code retires crops nobody answered for
# -- including one whose annotation was just removed, which the button
# promises puts it back in the queue. Both banking paths have to refuse.
if dash is not None:
    # empty when the name is gone, which fails rather than going blind on a
    # rename -- an extraction that quietly matches nothing is the shape of
    # guard this repo has been bitten by
    _ms = (dash.split('function markSeen(){', 1)[-1].split('\n}', 1)[0]
           if 'function markSeen(){' in dash else '')
    _beacon = "addEventListener('pagehide'"
    _pg = (dash.split(_beacon, 1)[-1].split('\n});', 1)[0]
           if _beacon in dash else '')
    _unguarded = [n for n, b in (('markSeen()', _ms),
                                 ('the pagehide beacon', _pg))
                  if "mode==='audit'" not in b]
    check('t5e neither banking path records a verdict in audit mode',
          not _unguarded, ' and '.join(_unguarded) + ' banks unjudged crops')

# ── t5f the audit ledgers hold answers a person could click ────────────────
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
import fn_audit as fa                                          # noqa: E402
vd_bad, ex_bad, n_vd = [], [], 0
for _stage in sorted(fa.STAGES):
    _sp = fa.STAGES[_stage]
    _rel = _sp['audit_dir'] + '/verdicts.jsonl'
    for _i, _r in records(_rel):
        n_vd += 1
        # a null verdict is read as a withdrawal, not as a third answer
        if _r.get('verdict') is not None and fa.verdict_of(
                _r.get('verdict'), _stage) is None:
            vd_bad.append(f'{_rel}:{_i} {_r.get("verdict")!r}')
    # The exported dataset carries the model's score and band beside every
    # row, as provenance. The label must still be the person's answer and
    # nothing else -- a label taken from p_dog would train the next model on
    # the current one's opinion, which is the whole thing this file is about.
    for _i, _r in records(_sp['dataset'] + '/manifest.jsonl'):
        if _r.get('label') != fa.verdict_of(_r.get('verdict'), _stage):
            ex_bad.append(f'{_sp["dataset"]}:{_i} labelled '
                          f'{_r.get("label")!r} on verdict '
                          f'{_r.get("verdict")!r}')
check(f't5f all {n_vd:,} audit verdicts are answers the page offers',
      not vd_bad, '; '.join(vd_bad[:5]))
check('t5f-b every exported label is its own verdict', not ex_bad,
      '; '.join(ex_bad[:5]))

# ── t5g every store on disk is one this file has classified ────────────────
# Two levels only: below that are the crops and the full frames, tens of
# thousands of jpgs and no ledger among them.
_on_disk = []
for _d in sorted(os.listdir(DATA) if os.path.isdir(DATA) else ()):
    _p = os.path.join(DATA, _d)
    if os.path.isfile(_p):
        if _d.endswith(('.jsonl', '.db')):
            _on_disk.append(_d)
        continue
    try:
        _kids = sorted(os.listdir(_p))
    except OSError:
        continue
    _on_disk += [_d + '/' + f for f in _kids if f.endswith(('.jsonl', '.db'))]
_known = HUMAN_STORES + MODEL_OWN + (LEASH_DB, FLAGS_DB)
_unclassified = [s for s in _on_disk if s not in _known]
check(f't5g all {len(_on_disk)} stores under data/ are classified',
      not _unclassified, 'nobody has said whether these hold human '
      'decisions: ' + ', '.join(_unclassified[:5]))

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
if skipped:
    # said again at the bottom because the one line in the middle of thirty is
    # exactly where a check that never ran gets read as one that passed
    print('NOT CHECKED: ' + ', '.join(skipped))
print('model suggestions are structurally separated from the training data')
