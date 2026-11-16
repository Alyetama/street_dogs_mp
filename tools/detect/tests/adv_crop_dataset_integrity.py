#!/usr/bin/env python3
"""The crop datasets: built only from human labels, cut the way production
cuts, and split so that nothing measured was ever trained on.

Every defect this file guards was measured on a real dataset first:

  D1  a Label Studio crop's name could not be parsed, so 2,933 crops of
      dogbin_20261012 had no sequence -- outside the cap, invisible to the
      leak check, 6.77% of val leaked
  D2  the leash build excluded the DOGBIN acceptance set, by image_id only,
      so all 312 leash acceptance images would have gone back into training
  D3  15k gate and 2.8k leash audit verdicts were in no dataset
  D4  four framings in one dataset, and 2,190 crops cut from 1280-px copies
      because the build's interpreter had no S3 client and said "no bucket"
  D5  a 64-px floor dropped the small boxes production judges all the time
  D6  the same box under two names, sometimes with two labels
  D7  no frozen test set drawn from the human verdicts
  D8  the checks that should have caught all of it could not fail

and, from the review of the first fix, the holes it left:

  * test crops that were byte twins of images the deployed model trained on,
    under another image_id and sequence (12 of 1,852), and re-encoded copies
    of reserved crops reaching val -- held out by PIXELS now, not by name;
  * one box "not_dog" on the gate audit and "unleashed" on the leash audit,
    trained both ways; the annotate page's boxes and "no dog here" frames
    read by neither crop family; a leash call cut from the raw detector box
    beside its own redraw; a Label Studio box joined against a redraw
    instead of the detector's box; repeat leash calls crashing the build;
  * a frame that went unreadable mid-build, a source that read as empty and
    a --reserve that replaced the acceptance set, each passing as clean;
  * collector refusals that could be deleted with the suite still green --
    they now each have a stubbed collect() case (frames wider than 1280 px,
    so a thumbnail cannot pass for the original) and a MUTANT.

and, from the leash audit of leash_20261115_18b269 (2026-11-15):

  * the leash family trained on reviewers' leash-inclusive redraws (270
    audit_edited rows, 12 corrected leash calls) where production cuts the
    detector box -- the leash rows are cut from the detector box now, the
    redraw kept as metadata, while the gate keeps redraw-wins;
  * one detector box double-boxed by the store under two det keys trained
    twice, or under two labels -- joined on pool_box IoU now;
  * creator blocks and a size-predicts-the-label skew -- the opt-in
    --max-per-creator and --balance-size-bands, asserted on plan and disk.

Synthetic frames in a temp directory; the stores, the catalog and the drives
are stood in for -- the catalog lookups too (resolve_sequences and
resolve_capture), so the suite reads nothing under data/ and runs in a
checkout without one. Each check is a function returning a list of
failures, and the MUTANTS section re-runs a check with the OLD behaviour
patched back in and requires it to FAIL -- a check that passes either way proves nothing.

The optional real-data smoke (skipped loudly when the data is not there)
runs reserve_test_set --dry-run against the live ledgers.

Run: python tools/detect/tests/adv_crop_dataset_integrity.py   (mp14 env)
"""
import ast
import collections
import contextlib
import importlib.util
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DETECT = os.path.join(REPO, 'tools', 'detect')
sys.path.insert(0, DETECT)

import crop_sources as cs                     # noqa: E402
import rebuild_crop_dataset as rcd            # noqa: E402
import reserve_test_set as rts                # noqa: E402
import gate_store                             # noqa: E402

OLD_NAME_RE = re.compile(r'^(?:flag_|pos_)?(?:le|un|no|nd)?_?(\d{6,})[_.]')
# Spelled in two halves: the isolation guards fail any tools/ module that
# spells the suggestion tier's store out, and this test only needs to prove
# such a path is REFUSED.
SUGGESTION_STORE = 'llm' + '_annotations'

TMP = tempfile.mkdtemp(prefix='adv_crop_integrity_')


class _C:
    """A duckdb connection that is never queried (every reader is stubbed)."""
    def close(self):
        pass


def _img(path, w=48, h=40, seed=None):
    """A noise image: distinct bytes and a distinct dHash for every seed."""
    from PIL import Image
    import numpy as np
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(a).save(path, 'JPEG', quality=95)
    return path


_SEED = [1000]


def row(source='gate_audit', label='dog', iid='100000000001', det='d0',
        seq='S1', box=(10, 10, 50, 50), framing=None, path=None,
        seq_status='manifest'):
    """A manifest row with a real file behind it."""
    _SEED[0] += 1
    uid = cs.uid_for(source, iid, det)
    path = path or _img(os.path.join(TMP, 'crops', label, uid),
                        seed=_SEED[0])
    sha, dh, wh = cs.describe(path)
    return {'uid': uid, 'path': path, 'label': label, 'family': 'dogbin',
            'source': source, 'image_id': iid, 'det_key': det,
            'sequence': seq, 'seq_status': seq_status,
            'box': list(box) if box else None, 'frame': None,
            'framing': framing or ('audit_full' if source.endswith('audit')
                                   else 'cut_crops'),
            'crop_wh': wh, 'sha256': sha, 'dhash': dh, 'band': None,
            'p_model': None, 'aimed': False,
            'provenance': {'ledger': os.path.join(REPO, 'data', 'fn_audit',
                                                  'verdicts.jsonl'),
                           'key_or_task_id': iid, 'by': 't', 'ts': 0}}


def write_manifest(rows, name):
    p = os.path.join(TMP, name)
    with open(p, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    return p


# What the stand-in catalog knows of each image's capture ({image_id:
# {creator, ts, lon, lat}}); a check sets the ones it needs. Everything else
# has no capture, which joins no twin -- the real catalog is never opened.
CAPTURES = {}
_REAL_CAPTURE = rcd.resolve_capture


def _stub_capture(ids, repo, py):
    return {str(i): dict(CAPTURES[str(i)]) for i in ids
            if str(i) in CAPTURES}


def run_rebuild(argv):
    """(exit text or None, stdout). SystemExit is caught and returned.

    HERMETIC: manifest mode looks every image's capture up in
    <repo>/data/catalog.duckdb (resolve_capture, for the twin rule), and so
    does load_reserves. Unstubbed, D1/D2/D7/D8's builds read the 33,170
    live ground_animals parquets (~6.5 s a lookup) and fail outright in a
    checkout with no catalog -- so the lookup is stood in for here, the way
    resolve_sequences is, unless a check put its own stub in place.
    """
    buf = io.StringIO()
    stub = rcd.resolve_capture is _REAL_CAPTURE
    if stub:
        rcd.resolve_capture = _stub_capture
    try:
        with contextlib.redirect_stdout(buf):
            rcd.main(argv)
        return None, buf.getvalue()
    except SystemExit as e:
        return str(e), buf.getvalue()
    finally:
        if stub:
            rcd.resolve_capture = _REAL_CAPTURE


# ── D1 parsing ──────────────────────────────────────────────────────────────

NAMES = {
    'pos_ls_1000628200890316_0.jpg': '1000628200890316',
    'flag_ls_1000628200890316_10.jpg': '1000628200890316',
    'ls_1000628200890316_b3.jpg': '1000628200890316',
    'ga_1000628200890316_d0.jpg': '1000628200890316',
    'le_2519462714862865_0.jpg': '2519462714862865',
    'no_1097638894341354_0.jpg': '1097638894341354',
    'flag_892156714766703_1.jpg': '892156714766703',
    '892156714766703_1.jpg': '892156714766703',
}


def check_parse(parse=None):
    parse = parse or cs.parse_name
    bad = []
    for name, want in NAMES.items():
        got = parse(name)
        if not got or got[0] != want:
            bad.append('D1 %s parsed to %r, want image_id %s'
                       % (name, got, want))
    got = parse('1786008818912_128242676254461_094.jpg')
    if got and got[0] == '1786008818912':
        bad.append('D1 a leash-call crop name resolves to its TIMESTAMP '
                   '1786008818912 as the image_id')
    if parse('README.md') is not None:
        bad.append('D1 a non-crop name parsed')
    return bad


def check_parse_fallback_agrees():
    bad = []
    for name in list(NAMES) + ['1786008818912_128242676254461_094.jpg',
                               'x.jpg', 'hn_123456_d4.jpg']:
        if cs.parse_name(name) != rcd._parse_name_fallback(name):
            bad.append('D1 rebuild\'s fallback parser disagrees with '
                       'crop_sources on %s' % (name,))
    return bad


def check_legacy_unparseable_fails():
    src = os.path.join(TMP, 'legacy_src')
    _img(os.path.join(src, 'train', 'dog', '123456789_0.jpg'), seed=1)
    _img(os.path.join(src, 'train', 'dog', 'mystery_crop.jpg'), seed=2)
    real = rcd.resolve_sequences
    rcd.resolve_sequences = lambda ids, repo, py: {i: 'S' for i in ids}
    try:
        err, _ = run_rebuild(['--src', src, '--out',
                              os.path.join(TMP, 'legacy_out'), '--no-reserve'])
    finally:
        rcd.resolve_sequences = real
    if not err or 'no image_id' not in err:
        return ['D1 a legacy crop whose name gives no image_id did not fail '
                'the build (got %r) -- it becomes a sequence of its own' %
                (err,)]
    return []


# ── D1 cap and leak ─────────────────────────────────────────────────────────

def check_cap(cap_fn=None):
    cap_fn = cap_fn or rcd.cap_rows
    rows = [row('ls_box', 'not_dog', '2000000000%02d' % k, 'b0', 'HERD')
            for k in range(10)]
    kept, _ = cap_fn(rows, 3, random.Random(0))
    n = collections.Counter((r['sequence'], r['label']) for r in kept)
    if max(n.values()) > 3:
        return ['D1 ten Label Studio crops of one sequence kept %d past a cap '
                'of 3' % (max(n.values()),)]
    return []


def _pairs(n=40, ls_seq=None):
    """n sequences, each one audit crop + one Label Studio crop."""
    rows = []
    for k in range(n):
        s = 'P%03d' % k
        rows.append(row('gate_audit', 'dog', '30000000%04d' % k, 'd0', s))
        r = row('ls_box', 'dog', '31000000%04d' % k, 'b0',
                ls_seq(k) if ls_seq else s)
        r['true_sequence'] = s
        rows.append(r)
    return rows


def check_leak(ls_seq=None):
    rows = _pairs(ls_seq=ls_seq)
    g = rcd.split_groups(rows, [])
    sp, _ = rcd.assign_splits(rows, g, 0.5, random.Random(1))
    side = collections.defaultdict(set)
    for r in rows:
        side[r.get('true_sequence') or r['sequence']].add(sp[g[r['uid']]])
    leaked = [s for s, v in side.items() if len(v) > 1]
    if leaked:
        return ['D1 %d sequences have a Label Studio crop on the other side '
                'of the split from its sequence-mate' % (len(leaked),)]
    return []


# ── D2 reserves ─────────────────────────────────────────────────────────────

def check_reserve_by_sequence(reserved_sets=None):
    bad = []
    rows = [row('gate_audit', 'dog', '400000000001', 'd0', 'RS'),
            row('gate_audit', 'dog', '400000000002', 'd0', 'OK'),
            row('gate_audit', 'not_dog', '400000000003', 'd0', 'OK2')]
    m = write_manifest(rows, 'm_reserve.jsonl')
    res = os.path.join(TMP, 'reserve.json')
    with open(res, 'w') as fh:
        json.dump({'image_ids': ['499999999999'], 'sequences': ['RS']}, fh)
    out = os.path.join(TMP, 'out_reserve_%d' % (random.randrange(1 << 30),))
    real = rcd.reserved_sets
    if reserved_sets:
        rcd.reserved_sets = reserved_sets
    try:
        err, _ = run_rebuild(['--manifest', m, '--out', out, '--reserve', res,
                              '--hamming', '0', '--execute'])
    finally:
        rcd.reserved_sets = real
    if err:
        return ['D2 the reserve build failed: %s' % (err,)]
    with open(os.path.join(out, 'crops.jsonl')) as fh:
        got = {json.loads(line)['image_id'] for line in fh}
    if '400000000001' in got:
        bad.append('D2 a crop that shares only a SEQUENCE with a reserved '
                   'image is in the dataset')
    return bad


def check_no_reserve_refused():
    m = write_manifest([row('gate_audit', 'dog', '410000000001')],
                       'm_norev.jsonl')
    err, _ = run_rebuild(['--manifest', m, '--out',
                          os.path.join(TMP, 'out_norev'), '--execute'])
    if not err or '--no-reserve' not in err:
        return ['D2 a rebuild with no --reserve did not refuse: %r' % (err,)]
    return []


def check_leash_argv_reserves(families=None):
    """The leash build hands rebuild the LEASH reserves, and not dogbin's."""
    import build_dataset as bd
    tmp = os.path.join(TMP, 'bd_root')
    data = os.path.join(TMP, 'bd_data')
    os.makedirs(os.path.join(tmp, 'leash_v2'), exist_ok=True)
    os.makedirs(data, exist_ok=True)
    for n in ('leash_acceptance_set.json', 'leash_test_v1.json',
              'dogbin_acceptance_set.json', 'dogbin_test_v1.json'):
        with open(os.path.join(data, n), 'w') as fh:
            json.dump({'image_ids': ['1']}, fh)
    calls = []

    def fake_run(self, name, argv, cwd=None, env=None):
        calls.append((name, argv))
        self.steps.append({'name': name, 'argv': argv, 'exit_code': 0,
                           'seconds': 0.0})
        if name == 'rebuild_crop_dataset':
            out = argv[argv.index('--out') + 1]
            for sp in ('train', 'val'):
                os.makedirs(os.path.join(out, sp, 'leashed'), exist_ok=True)
        return 0
    saved = (bd.Runner.run, bd.D, os.environ.get('TRAINING_ROOT'),
             bd.FAMILIES['leash'].get('reserves'))
    bd.Runner.run = fake_run
    bd.D = lambda *p: os.path.join(data, *p)
    os.environ['TRAINING_ROOT'] = tmp
    if families:
        bd.FAMILIES['leash']['reserves'] = families
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            bd.build('leash', out=os.path.join(TMP, 'bd_out_%d' % len(
                os.listdir(TMP))), no_export=True, no_embed_dedup=True,
                detect_root=os.path.join(TMP, 'nostore'), allow_no_ls=True)
    except SystemExit as e:
        return ['D2 the stubbed leash build refused: %s' % (e,)]
    finally:
        bd.Runner.run, bd.D = saved[0], saved[1]
        if saved[2] is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = saved[2]
        bd.FAMILIES['leash']['reserves'] = saved[3]
    argv = next((a for n, a in calls if n == 'rebuild_crop_dataset'), [])
    got = [os.path.basename(argv[i + 1]) for i, x in enumerate(argv)
           if x == '--reserve']
    bad = []
    if sorted(got) != ['leash_acceptance_set.json', 'leash_test_v1.json']:
        bad.append('D2 the leash build reserves %r, not the leash acceptance '
                   'and test sets' % (got,))
    if any('dogbin' in g for g in got):
        bad.append('D2 the leash build holds out the DOGBIN reserve')
    return bad


# ── D3 coverage ─────────────────────────────────────────────────────────────

def _audit_fixture(stage, answers):
    """An audit directory for `stage`: verdicts, full/ crops, one edited/
    crop (key ...2#0, redrawn to (2, 3, 20, 30)) and a pool parquet."""
    base = os.path.join(TMP, 'audit_fixture_' + stage)
    paths = {'out': base, 'pool': os.path.join(base, 'pool.parquet'),
             'verdicts': os.path.join(base, 'verdicts.jsonl'),
             'full': os.path.join(base, 'full'),
             'pages': os.path.join(base, 'pages')}
    os.makedirs(paths['pages'], exist_ok=True)
    yes, no = answers
    vs = [('500000000001#0', yes), ('500000000002#0', no),
          ('500000000003#0', 'unsure'), ('500000000004#1', yes),
          ('500000000005#0', yes), ('500000000005#0', None)]
    with open(paths['verdicts'], 'w') as fh:
        for k, v in vs:
            fh.write(json.dumps({'key': k, 'verdict': v, 'band': 3,
                                 'p_dog': 0.3, 'seq': 'Q' + k[:12]}) + '\n')
    for k in ('500000000001_0', '500000000002_0', '500000000003_0',
              '500000000005_0'):
        _img(os.path.join(paths['full'], k + '.jpg'), seed=hash(k) % 999)
    _img(os.path.join(base, 'edited', '500000000002_0.jpg'), seed=7)
    import duckdb
    duckdb.connect().execute(
        "COPY (SELECT * FROM (VALUES ('500000000001', 0, 1.0, 2.0, 30.0, 40.0,"
        " 'c', 'd'), ('500000000002', 0, 1.0, 2.0, 30.0, 40.0, 'c', 'd')) "
        "t(image_id, det_idx, x1, y1, x2, y2, cell, drive)) TO '%s' "
        "(FORMAT PARQUET)" % paths['pool'])
    return paths


def check_audit_coverage():
    """Every labelled verdict becomes a row, and WHICH crop of a redrawn box
    trains is the family's rule: the gate trains on the redraw (edited/),
    the leash model on the detector box production cuts (full/), keeping
    the redraw as provenance.redraw_box (user decision 2026-11-15)."""
    import fn_audit as fa
    fx = {'leash': _audit_fixture('leash', ('leashed', 'unleashed')),
          'gate': _audit_fixture('gate', ('dog', 'not_dog'))}
    real_paths, real_corr = fa.paths, cs.read_audit_corrections
    fa.paths = lambda stage='gate': dict(fx[stage])
    cs.read_audit_corrections = lambda stage, path=None: {
        ('500000000002', 0): (2.0, 3.0, 20.0, 30.0)}
    try:
        got = {st: cs.audit_candidates(st) for st in fx}
    finally:
        fa.paths, cs.read_audit_corrections = real_paths, real_corr
    bad = []
    for st, (rows, counts) in sorted(got.items()):
        keys = {r['provenance']['key_or_task_id']: r for r in rows}
        labelled = {'500000000001#0', '500000000002#0', '500000000004#1'}
        accounted = len(keys) + counts.get('audit_missing_crop', 0)
        if accounted != len(labelled):
            bad.append('D3 %s: %d labelled verdicts became %d rows + %d '
                       'counted drops' % (st, len(labelled), len(keys),
                                          counts.get('audit_missing_crop', 0)))
        if '500000000003#0' in keys:
            bad.append('D3 %s: an "unsure" answer became a training row'
                       % (st,))
        if '500000000005#0' in keys:
            bad.append('D3 %s: a withdrawn verdict (null) became a training '
                       'row' % (st,))
        r2 = keys.get('500000000002#0') or {}
        if st == 'gate':
            if r2.get('framing') != 'audit_edited' or 'edited' not in \
                    r2.get('src_path', ''):
                bad.append('D3 gate: a redrawn box did not win over the '
                           'detector\'s full/ crop: %r'
                           % (r2.get('src_path'),))
            elif r2['box'] != [2.0, 3.0, 20.0, 30.0]:
                bad.append('D3 gate: an edited crop does not carry its '
                           'redrawn box')
        else:
            if r2.get('framing') != 'audit_full' or 'full' not in \
                    r2.get('src_path', ''):
                bad.append('D3 leash: a redraw (edited/) trains instead of '
                           'the detector box production cuts: %r'
                           % (r2.get('src_path'),))
            elif r2['box'] != [1.0, 2.0, 30.0, 40.0] or \
                    r2['box'] != r2.get('pool_box'):
                bad.append('D3 leash: the crop\'s box is not its pool_box '
                           '(%r vs %r)' % (r2['box'], r2.get('pool_box')))
            elif r2['provenance'].get('redraw_box') != [2.0, 3.0, 20.0, 30.0]:
                bad.append('D3 leash: the redraw is not kept as metadata '
                           '(provenance.redraw_box %r)'
                           % (r2['provenance'].get('redraw_box'),))
            if counts.get('edited_not_used_detector_box_wins') != 1:
                bad.append('D3 leash: the redraw not trained on is not '
                           'counted: %r' % (dict(counts),))
    return bad


# ── D4 framing ──────────────────────────────────────────────────────────────

def _frame(w=400, h=300, name='frame.jpg'):
    return _img(os.path.join(TMP, 'frames', name), w=w, h=h, seed=w * h)


def _cut(box, source='hard_negative', **extra):
    fr = _frame()
    r = dict({'source': source, 'label': 'not_dog', 'box': list(box),
              'uid': 'hn_600000000001_d0.jpg'}, **extra)
    _, wh, res = cs.cut_frame((fr, [(0, r)], os.path.join(TMP, 'cut')))
    return res[0], r


def check_framing_matches_production():
    from PIL import Image
    box = (100, 100, 150, 140)
    (_, dst, why), _ = _cut(box)
    if not dst:
        return ['D4 a plain box was not cut: %s' % (why,)]
    with Image.open(_frame()) as im:
        im.load()
        want = gate_store.cut_crops(im.convert('RGB'),
                                    [(0,) + box])[0][1].size
    with Image.open(dst) as got:
        if got.size != want:
            return ['D4 crop is %r, production cuts %r' % (got.size, want)]
    return []


def check_constant_is_imported():
    from PIL import Image
    box = (100, 100, 150, 140)
    (_, d1, _), _ = _cut(box)
    with Image.open(d1) as im:
        a = im.size
    real = gate_store.PAD_FRAC
    gate_store.PAD_FRAC = 0.2
    try:
        (_, d2, _), _ = _cut(box)
        with Image.open(d2) as im:
            b = im.size
    finally:
        gate_store.PAD_FRAC = real
    if a == b:
        return ['D4 changing gate_store.PAD_FRAC did not change the cut -- '
                'the margin is a copy, not production\'s']
    return []


LITERAL_RE = re.compile(r'(?<![\d.])0\.1[25](?!\d)')


def check_no_copied_constants(sources=None):
    bad = []
    for name, text in (sources or [
            (n, open(os.path.join(DETECT, n)).read())
            for n in ('crop_sources.py', 'harvest_flagged.py')]):
        for ln, line in enumerate(text.splitlines(), 1):
            if LITERAL_RE.search(line):
                bad.append('D4 %s:%d carries a framing literal: %s'
                           % (name, ln, line.strip()[:80]))
    return bad


def check_resized_ls_frame_refused():
    bad = []
    small = _img(os.path.join(TMP, 'frames', 'ls1280.jpg'), w=128, h=72,
                 seed=3)
    if cs._full_width(small, (4000, 2250)):
        bad.append('D4 a 1280-style copy was accepted as a 4000-px original')
    fr = small
    r = {'source': 'ls_box', 'label': 'leashed', 'pct': (10, 10, 20, 20),
         'ls_orig': (4000, 2250), 'uid': 'ls_600000000002_b0.jpg'}
    _, _, res = cs.cut_frame((fr, [(0, r)], os.path.join(TMP, 'cut')))
    if res[0][1] is not None or res[0][2] != 'ls_size_mismatch':
        bad.append('D4 a Label Studio box was cut from a frame that is not '
                   'the size Label Studio recorded: %r' % (res[0],))
    return bad


def check_missing_boto3_fatal():
    if importlib.util.find_spec('boto3') is not None:
        print('  (D4 boto3 check skipped: this interpreter HAS boto3)')
        return []
    got = cs.fetch_ls_frames(['https://h/x/y/700000000001.jpg'],
                             os.path.join(TMP, 'lsframes'), sys.executable,
                             os.path.join(TMP, 'no.env'))
    if '__error__' not in got:
        return ['D4 an interpreter without boto3 did not report an error: %r'
                % (got,)]
    # ...and collect() turns that into a refusal, not "no bucket"
    saved = (cs.frames_for, cs.drive_roots, cs.exclusions, cs._duck,
             cs.audit_candidates, cs.leash_call_candidates,
             cs.frame_label_candidates, cs.other_family_vetoes)
    cs.frames_for = lambda ids, root, con=None: {}
    cs.drive_roots = lambda: {}
    cs.exclusions = lambda fam, root=None: ({}, {})
    cs._duck = lambda memory='12GB': _C()
    cs.audit_candidates = lambda stage, con=None: ([], collections.Counter())
    cs.leash_call_candidates = lambda root, **k: ([], collections.Counter(),
                                                  [])
    cs.frame_label_candidates = lambda path=None: ([], {},
                                                   collections.Counter())
    cs.other_family_vetoes = lambda fam, root, con=None: (
        [], collections.Counter())
    task = {'id': 1, 'image': 'https://h/x/y/700000000001.jpg',
            'label': [{'x': 1, 'y': 1, 'width': 5, 'height': 5,
                       'rectanglelabels': ['leashed dog'],
                       'original_width': 4000, 'original_height': 2250}]}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cs.collect('leash', os.path.join(TMP, 'stage_boto'), [task],
                       'export.json', detect_root='x', roots={},
                       s3_python=sys.executable,
                       env_file=os.path.join(TMP, 'no.env'),
                       frames_dir=os.path.join(TMP, 'lsframes'),
                       training_root=TMP, say=lambda *_: None,
                       allow_empty_sources=('leash_audit', 'leash_call'))
        return ['D4 a build whose S3 helper cannot import boto3 carried on '
                'without the frame instead of failing']
    except SystemExit as e:
        if 'bucket' not in str(e):
            return ['D4 the refusal does not say the bucket was unreachable: '
                    '%s' % (e,)]
        return []
    finally:
        (cs.frames_for, cs.drive_roots, cs.exclusions, cs._duck,
         cs.audit_candidates, cs.leash_call_candidates,
         cs.frame_label_candidates, cs.other_family_vetoes) = saved


# ── D5 small boxes ──────────────────────────────────────────────────────────

def check_small_boxes_cut():
    from PIL import Image
    bad = []
    for box, want in (((200, 200, 220, 220), None), ((50, 50, 53, 53), 11)):
        (_, dst, why), _ = _cut(box)
        if not dst:
            bad.append('D5 a %dx%d box was not cut (%s)'
                       % (box[2] - box[0], box[3] - box[1], why))
            continue
        with Image.open(dst) as im:
            if want and im.size != (want, want):
                bad.append('D5 a 3x3 box came out %r, production gives '
                           '%dx%d' % (im.size, want, want))
    tree = ast.parse(open(os.path.join(DETECT, 'harvest_flagged.py')).read())
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == \
                'add_argument' and n.args and \
                getattr(n.args[0], 'value', '') == '--min-size':
            dflt = [k.value.value for k in n.keywords if k.arg == 'default']
            if dflt != [0]:
                bad.append('D5 harvest_flagged --min-size defaults to %r, '
                           'not 0' % (dflt,))
    return bad


# ── D6 dedup ────────────────────────────────────────────────────────────────

def check_dedup(dedup=None):
    dedup = dedup or rcd.dedup_rows
    bad = []
    # byte-identical, opposite labels -> both dropped
    a = row('leash_audit', 'dog', '800000000001', 'd0', 'A')
    b = row('leash_audit', 'not_dog', '800000000002', 'd0', 'B',
            path=_copy(a['path'], 'not_dog', 'la_800000000002_d0.jpg'))
    b['sha256'] = a['sha256']
    kept, _, conf, _, _ = dedup([a, b], 0)
    if kept:
        bad.append('D6 byte-identical crops with opposite labels survived: '
                   '%r' % ([r['uid'] for r in kept],))
    # same bytes, same label -> one kept, both sequences one group
    c = row('gate_audit', 'dog', '800000000003', 'd0', 'C')
    d = row('gate_audit', 'dog', '800000000004', 'd0', 'D',
            path=_copy(c['path'], 'dog', 'ga_800000000004_d0.jpg'))
    e = row('gate_audit', 'dog', '800000000005', 'd0', 'D')
    kept, unions, _, _, _ = dedup([c, d, e], 0)
    if len([r for r in kept if r['uid'] in (c['uid'], d['uid'])]) != 1:
        bad.append('D6 two byte-identical same-label crops were not '
                   'collapsed to one')
    g = rcd.split_groups(kept, unions)
    if len({g[r['uid']] for r in kept}) != 1:
        bad.append('D6 the sequences of byte-identical crops are not one '
                   'split group')
    # same (iid, det), opposite labels -> dropped
    f = row('hard_negative', 'not_dog', '800000000006', 'd2', 'F')
    h = row('gate_audit', 'dog', '800000000006', 'd2', 'F')
    kept, _, conf, _, _ = dedup([f, h], 0)
    if kept or not conf:
        bad.append('D6 one detector box labelled both ways survived')
    # a drawn box over an audit box, same label -> one row
    i = row('gate_audit', 'dog', '800000000007', 'd0', 'I',
            box=(100, 100, 200, 200))
    j = row('ls_box', 'dog', '800000000007', 'b0', 'I',
            box=(110, 110, 210, 210))
    kept, _, _, _, _ = dedup([i, j], 0)
    if len(kept) != 1:
        bad.append('D6 a drawn box at IoU ~0.68 to an audit box with the '
                   'same label stayed two rows')
    # IoU ambiguity -> the image goes
    k1 = row('gate_audit', 'dog', '800000000008', 'd0', 'K',
             box=(100, 100, 200, 200))
    k2 = row('hard_positive', 'dog', '800000000008', 'd1', 'K',
             box=(105, 105, 205, 205))
    k3 = row('ls_box', 'dog', '800000000008', 'b0', 'K',
             box=(102, 102, 202, 202))
    kept, _, conf, _, _ = dedup([k1, k2, k3], 0)
    if any(r['image_id'] == '800000000008' for r in kept):
        bad.append('D6 a drawn box matching two detector boxes did not drop '
                   'the image')
    # cross-class dHash 0 -> both
    m1 = row('gate_audit', 'dog', '800000000009', 'd0', 'M')
    m2 = row('gate_audit', 'not_dog', '800000000010', 'd0', 'N')
    m2['dhash'] = m1['dhash']
    # a different size, so only the distance-0 rule can catch it
    m2['crop_wh'] = [m1['crop_wh'][0] * 3, m1['crop_wh'][1] * 3]
    kept, _, conf, _, _ = dedup([m1, m2], 0)
    if kept:
        bad.append('D6 dHash-identical crops with opposite labels survived')
    # cross-class, dHash 3 apart, the same size -> one picture, both go
    n1 = row('gate_audit', 'dog', '800000000011', 'd0', 'N1')
    n2 = row('gate_audit', 'not_dog', '800000000012', 'd0', 'N2')
    n2['dhash'] = n1['dhash'] ^ 0b10101
    n2['crop_wh'] = [n1['crop_wh'][0] + 1, n1['crop_wh'][1]]
    kept, _, conf, _, _ = dedup([n1, n2], 0)
    if kept:
        bad.append('D6 a same-size crop 3 dHash bits from one with the '
                   'opposite label survived')
    n2['crop_wh'] = [n1['crop_wh'][0] * 3, n1['crop_wh'][1] * 3]
    kept, _, _, _, _ = dedup([n1, n2], 0)
    if len(kept) != 2:
        bad.append('D6 two crops of very different sizes that merely share a '
                   'dHash neighbourhood were dropped as one picture')
    return bad


def _copy(src, cls, name):
    dst = os.path.join(TMP, 'crops', cls, name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return dst


# ── D7 the reserve tool ─────────────────────────────────────────────────────

def _verdicts():
    out = []
    for k in range(60):
        out.append({'key': '9000000000%02d#0' % k, '_label':
                    'dog' if k % 2 else 'not_dog', 'band': k % 3,
                    'p_dog': .5, 'seq': 'V%02d' % k, 'ts': k})
    out[0]['seq'] = 'img:900000000000'
    return out


def check_reserve_tool(dataset_ids=None):
    bad = []
    dataset_ids = dataset_ids or rts.dataset_ids
    vs = _verdicts()
    crop_dir = os.path.join(TMP, 'rts_full')
    for v in vs:
        _img(os.path.join(crop_dir, v['key'].replace('#', '_') + '.jpg'),
             seed=int(v['key'][:12]) % 9999)
    comp = os.path.join(TMP, 'rts_compare')
    _img(os.path.join(comp, 'train', 'dog', 'pos_ls_900000000003_0.jpg'),
         seed=11)
    try:
        cids, _ = dataset_ids(comp)
    except SystemExit as e:
        return ['D7 the compare dataset could not be read: %s' % (e,)]
    catalog = {v['key'][:12]: v['seq'] for v in vs}
    catalog['900000000005'] = 'V05'
    cseqs = {'V07'}                       # a compare-dataset sequence
    cands, why = rts.eligible('dogbin', vs, {'900000000001#0'}, crop_dir,
                              set(), cids, cseqs, set(), set(), catalog,
                              ('dog', 'not_dog'))
    keys = {c['key'] for c in cands}
    for k, what in (('900000000001#0', 'an aimed draw'),
                    ('900000000000#0', 'an img: sequence'),
                    ('900000000003#0', 'an image in the compare dataset '
                                       '(an ls_-named crop)'),
                    ('900000000007#0', 'a compare-dataset sequence')):
        if k in keys:
            bad.append('D7 %s is eligible for the test set' % (what,))
    frac = 0.3
    st = rts.draw(cands, frac, 5)
    for (band, lab), (el, rs) in st.items():
        want = int(round(frac * len(el)))
        if len(el) >= 5:
            want = max(1, want)
        if len(rs) != want:
            bad.append('D7 stratum %s/%s reserved %d of %d, want %d'
                       % (band, lab, len(rs), len(el), want))
    if rts.draw(cands, frac, 5) != st:
        bad.append('D7 the same seed drew a different test set')
    out = os.path.join(TMP, 'rts_out.json')
    with open(out, 'w') as fh:
        json.dump({'keys': ['x'], 'created_at': 'then'}, fh)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rts.main(['--family', 'leash', '--out', out])
        bad.append('D7 an existing reservation was overwritten without '
                   '--force')
    except SystemExit as e:
        if 'already reserves' not in str(e):
            bad.append('D7 the overwrite refusal said: %s' % (e,))
    src = open(os.path.join(DETECT, 'reserve_test_set.py')).read()
    if "'band_populations'" not in src:
        bad.append('D7 the reservation records no band_populations')
    return bad


def _old_dataset_ids(path):
    ids = set()
    for base, _, files in os.walk(path):
        for f in files:
            m = OLD_NAME_RE.match(f)
            if m:
                ids.add(m.group(1))
    return ids, len(ids)


# ── D8 the assertions ───────────────────────────────────────────────────────

def _plan():
    rows = []
    for k, (sp, seq) in enumerate((('train', 'T1'), ('train', 'T2'),
                                   ('val', 'V1'), ('val', 'V2'))):
        r = row('gate_audit', 'dog' if k % 2 else 'not_dog',
                '95000000000%d' % k, 'd0', seq)
        r['split'], r['group'] = sp, seq
        rows.append(r)
    return rows


def check_assertions():
    bad = []
    if rcd.check_rows(_plan()):
        bad.append('D8 a clean plan fails: %s' % (rcd.check_rows(_plan()),))
    cases = []
    p = _plan()
    p[2]['sequence'] = 'T1'
    cases.append(('a sequence in train and val', p, ()))
    p = _plan()
    p[2]['group'] = 'T1'
    cases.append(('a split group in train and val', p, ()))
    cases.append(('a reserved train sequence', _plan(),
                  [{'image_ids': ['1'], 'sequences': ['T2']}]))
    p = _plan()
    p[0]['source'] = 'gate_store'
    cases.append(('a model verdict as the source', p, ()))
    p = _plan()
    p[0]['provenance']['ledger'] = os.path.join(REPO, 'data', 'gate',
                                                'gate-00001.parquet')
    cases.append(('provenance under data/gate/', p, ()))
    p = _plan()
    p[0]['provenance']['ledger'] = os.path.join(REPO, 'data',
                                                SUGGESTION_STORE, 'x.jsonl')
    cases.append(('provenance in the suggestion tier\'s store', p, ()))
    p = _plan()
    p[2]['seq_status'], p[2]['sequence'] = 'noseq', 'noseq:x'
    cases.append(('a sequence-less crop in val (with --keep-noseq)', p, (),
                  {'allow_noseq': True}))
    p = _plan()
    p[0]['seq_status'], p[0]['sequence'] = 'noseq', 'noseq:x'
    cases.append(('a sequence-less crop without --keep-noseq', p, ()))
    p = _plan()
    for r in p[:3]:
        r['seq_status'], r['sequence'], r['split'] = 'noseq', 'noseq:' + \
            r['image_id'], 'train'
        r['group'] = r['sequence']
    cases.append(('3 of 4 crops sequence-less (limit 50%)', p, (),
                  {'allow_noseq': True}))
    p = _plan()
    p[1]['framing'] = 'pad015'
    cases.append(('a crop framed some other way', p, ()))
    p = _plan()
    p[1]['sha256'] = p[0]['sha256']
    cases.append(('the same bytes twice', p, ()))
    # the per-(sequence, class) cap: four of one class in one sequence
    p = _plan()
    for k in range(4):
        r = row('gate_audit', 'dog', '95100000000%d' % k, 'd0', 'T1')
        r['split'], r['group'] = 'train', 'T1'
        p.append(r)
    cases.append(('4 crops of one (sequence, class) under a cap of 3', p, ()))
    p = _plan()
    p[0]['image_id'] = ''
    cases.append(('a crop with no image_id', p, ()))
    p = _plan()
    p[0]['provenance'] = {}
    cases.append(('a crop with no provenance', p, ()))
    p = _plan()
    p[1]['image_id'], p[1]['det_key'] = p[0]['image_id'], p[0]['det_key']
    p[1]['sequence'] = p[1]['group'] = p[0]['sequence']
    p[1]['split'] = p[0]['split']
    cases.append(('one (image_id, det_key) twice, different bytes', p, ()))
    cases.append(('a surviving member of a label conflict', _plan(), (),
                  {'conflict_uids': {_plan()[0]['uid']}}))
    p = _plan()
    cases.append(('a copy of a reserved crop under another name', p, (),
                  {'reserved_crops': {'hamming': 2,
                                      'sha256': [p[3]['sha256']]}}))
    p = _plan()
    cases.append(('a near-copy (dHash 1) of a reserved crop', p, (),
                  {'reserved_crops': {'hamming': 2,
                                      'dhash': [p[2]['dhash'] ^ 1]}}))
    for case in cases:
        what, plan, res = case[:3]
        kw = case[3] if len(case) > 3 else {}
        if not rcd.check_rows(plan, res, 3, 0.5, **kw):
            bad.append('D8 check_rows passed a plan with %s' % (what,))
    # the manifest itself: a row with no sequence never reaches a split
    m = row('gate_audit', 'dog', '952000000001', 'd0', 'Z')
    m['sequence'] = None
    err, _ = run_rebuild(['--manifest', write_manifest([m], 'm_noseq.jsonl'),
                          '--out', os.path.join(TMP, 'out_noseq_pre'),
                          '--no-reserve'])
    if not err or 'no sequence' not in err:
        bad.append('D8 a manifest row with no sequence was not refused: %r'
                   % (err,))
    # on disk: a file in no manifest row
    out = os.path.join(TMP, 'verify_ds')
    plan = _plan()
    for r in plan:
        d = os.path.join(out, r['split'], r['label'])
        os.makedirs(d, exist_ok=True)
        shutil.copy2(r['path'], os.path.join(d, r['uid']))
    with open(os.path.join(out, 'crops.jsonl'), 'w') as fh:
        for r in plan:
            fh.write(json.dumps(r) + '\n')
    with open(os.path.join(out, 'rebuild_manifest.json'), 'w') as fh:
        json.dump({'max_per_sequence': 3, 'max_noseq_frac': 0.5,
                   'no_reserve': True}, fh)
    if rcd.verify_dataset(out):
        bad.append('D8 verify fails a clean dataset: %s'
                   % (rcd.verify_dataset(out),))
    # ...and every row-level property, re-checked on the files: a reserved
    # sequence on disk fails verify even though the build's own plan passed
    if not rcd.verify_dataset(out, reserves=[
            {'image_ids': ['1'], 'sequences': [plan[0]['sequence']]}]):
        bad.append('D8 verify passed a dataset holding a reserved sequence '
                   '-- the on-disk check_rows is not run')
    _img(os.path.join(out, 'train', 'dog', 'ga_959999999999_d0.jpg'), seed=5)
    if not rcd.verify_dataset(out):
        bad.append('D8 verify passed a dataset holding a file in no '
                   'manifest row')
    rep = rcd.size_report(plan, {'quantiles': {'p50': 88}})
    want = {'%s/%s' % (r['split'], r['label']) for r in plan}
    if set(rep['splits']) != want:
        bad.append('D8 the size report lacks %s'
                   % (sorted(want - set(rep['splits'])),))
    return bad


def check_manifest_end_to_end():
    """A whole manifest-mode build: writes, verifies, and says so."""
    rows = []
    for k in range(30):
        rows.append(row('gate_audit', 'dog' if k % 3 else 'not_dog',
                        '96000000%04d' % k, 'd0', 'E%02d' % (k // 2)))
    m = write_manifest(rows, 'm_e2e.jsonl')
    out = os.path.join(TMP, 'out_e2e')
    err, text = run_rebuild(['--manifest', m, '--out', out, '--no-reserve',
                             '--hamming', '0', '--execute'])
    if err:
        return ['D8 a clean manifest build failed: %s' % (err,)]
    if 'assertions on disk: all' not in text:
        return ['D8 the build did not verify what it wrote']
    err, _ = run_rebuild(['--verify-only', out])
    return ['D8 --verify-only fails a fresh build: %s' % (err,)] if err else []


# ── the collector itself, sources stubbed ───────────────────────────────────
#
# collect() is where most of the refusals live, and a refusal nobody makes
# fire is a refusal that can be deleted. Every source reader is stood in for
# with rows built here; the frames are real files on a stand-in drive, WIDER
# THAN 1280 px so a frame quietly downscaled to the old thumbnail size cannot
# pass for the original; the cut, the framing check, the vetoes and the
# sequence check are the real ones.

DRV = 'drv'
CELL = 'cellA'


def _frame_on_drive(iid, w=1600, h=900, seed=0):
    from PIL import Image
    import numpy as np
    rng = np.random.default_rng(seed or int(iid) % 9973)
    small = Image.fromarray(rng.integers(0, 255, (30, 40, 3), dtype=np.uint8))
    p = os.path.join(TMP, DRV, CELL, 'ground_animal_images', iid + '.jpg')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    small.resize((w, h), Image.BICUBIC).save(p, 'JPEG', quality=90)
    return p


def _size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def _prov(*parts):
    return {'ledger': os.path.join(REPO, 'data', *parts),
            'key_or_task_id': 'k', 'by': 't', 'ts': 0}


def _audit_row(iid, label, box, stage='gate', pad=None, verdict_seq=None):
    """An audit row whose full/ crop the page cut -- the production way, or
    with a `pad` fraction of the long side the way the old builds did."""
    frame = os.path.join(TMP, DRV, CELL, 'ground_animal_images', iid + '.jpg')
    im = cs._open_rgb(frame)
    if pad is None:
        crop = gate_store.cut_crops(im, [(0,) + tuple(box)])[0][1]
    else:
        x1, y1, x2, y2 = box
        m = pad * max(x2 - x1, y2 - y1)
        crop = im.crop((int(x1 - m), int(y1 - m), int(x2 + m), int(y2 + m)))
    src = os.path.join(TMP, 'auditfull', '%s_0.jpg' % iid)
    os.makedirs(os.path.dirname(src), exist_ok=True)
    crop.save(src, 'JPEG', quality=95)
    return {'source': 'gate_audit' if stage == 'gate' else 'leash_audit',
            'label': label, 'image_id': iid, 'det_key': 'd0',
            'box': list(box), 'pool_box': list(box),
            'cell_drive': (CELL, DRV), 'src_path': src,
            'framing': 'audit_full', 'band': 1, 'p_model': .5,
            'aimed': False, 'verdict_seq': verdict_seq,
            'provenance': _prov('fn_audit', 'verdicts.jsonl')}


def _det_row(source, label, iid, box, det='d0'):
    return {'source': source, 'label': label, 'image_id': iid,
            'det_key': det, 'box': list(box), 'pool_box': list(box),
            'framing': 'cut_crops',
            'provenance': _prov('hard_negatives', 'labels.jsonl')}


def run_collect(family, by_source, tasks=None, store_wh=None, seqs=None,
                excl=None, cross=(), empty=None, allow_empty=None,
                allow_missing=0, extra_frames=()):
    """(refusal text or None, rows, conflicts) of a stubbed collect()."""
    stage = os.path.join(TMP, 'stage_%d' % random.randrange(1 << 30))
    names = ('frames_for', 'drive_roots', 'exclusions', '_duck',
             'audit_candidates', 'flag_candidates', 'leash_call_candidates',
             'frame_label_candidates', 'other_family_vetoes')
    saved = {n: getattr(cs, n) for n in names}
    saved_seq = rcd.resolve_sequences
    store_wh = store_wh or {}
    have = set(extra_frames)
    for lst in by_source.values():
        have.update(r['image_id'] for r in lst)
    for t in tasks or ():
        have.add(os.path.basename(t['image']).split('.')[0])

    def frames_for(ids, root, con=None):
        out = {}
        for i in ids:
            p = os.path.join(TMP, DRV, CELL, 'ground_animal_images',
                             i + '.jpg')
            if i in have and os.path.isfile(p):
                try:
                    w, h = store_wh.get(i) or _size(p)
                except OSError:
                    w, h = 1600, 900
                out[i] = (CELL, DRV, w, h)
        return out

    def rows_of(src):
        return [dict(x) for x in by_source.get(src, ())]

    def counted(r):
        return r, collections.Counter(rows=len(r))
    cs.frames_for = frames_for
    cs.drive_roots = lambda: {DRV: os.path.join(TMP, DRV)}
    cs.exclusions = lambda fam, root=None: (dict(excl or {}), {})
    cs._duck = lambda memory='12GB': _C()
    cs.audit_candidates = lambda stage, con=None: counted(rows_of(
        'gate_audit' if stage == 'gate' else 'leash_audit'))
    cs.flag_candidates = lambda label, root, **k: counted(rows_of(
        'hard_negative' if label == 'false_positive' else 'hard_positive'))
    cs.leash_call_candidates = lambda root, **k: counted(
        rows_of('leash_call')) + ([],)
    cs.frame_label_candidates = lambda path=None: (
        rows_of('frame_label'), dict(empty or {}),
        collections.Counter(rows=len(by_source.get('frame_label', ()))))
    cs.other_family_vetoes = lambda fam, root, con=None: (
        list(cross), collections.Counter())
    rcd.resolve_sequences = lambda ids, repo, py: {
        i: (seqs or {}).get(i, 'S' + i) for i in ids}
    if allow_empty is None:
        allow_empty = [s for s in cs.FAMILY_SOURCES[family]
                       if not by_source.get(s)
                       and not (s == 'ls_box' and tasks)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rows, _counts, conf = cs.collect(
                family, stage, tasks, 'export.json', detect_root='x',
                workers=2, frames_dir=os.path.join(TMP, 'lsframes_c'),
                allow_missing_frames=allow_missing, training_root=TMP,
                allow_empty_sources=allow_empty, say=lambda *_: None)
        return None, rows, conf
    except SystemExit as e:
        return str(e), [], []
    finally:
        for n, f in saved.items():
            setattr(cs, n, f)
        rcd.resolve_sequences = saved_seq


BOX = (400.0, 300.0, 520.0, 420.0)


def _ids(rows):
    return {r['uid'] for r in rows}


def check_collect_clean():
    """The baseline every refusal below is measured against: it must pass."""
    _frame_on_drive('970000000001')
    _frame_on_drive('970000000002')
    err, rows, _ = run_collect('dogbin', {
        'hard_negative': [_det_row('hard_negative', 'not_dog',
                                   '970000000001', BOX)],
        'gate_audit': [_audit_row('970000000002', 'dog', BOX)]})
    if err:
        return ['D4 a clean stubbed collection refused: %s' % (err,)]
    if len(rows) != 2:
        return ['D4 a clean stubbed collection kept %d of 2 rows'
                % (len(rows),)]
    from PIL import Image
    hn = next(r for r in rows if r['source'] == 'hard_negative')
    with Image.open(hn['frame']['path']) as im:
        want = gate_store.cut_crops(im.convert('RGB'),
                                    [(0,) + BOX])[0][1].size
    if tuple(hn['crop_wh']) != tuple(want):
        return ['D4 the cut crop is %r, production cuts %r on the full frame'
                % (hn['crop_wh'], want)]
    return []


def check_collect_missing_frame():
    err, _, _ = run_collect('dogbin', {'hard_negative': [
        _det_row('hard_negative', 'not_dog', '970000000099', BOX)]})
    if not err or 'full-resolution frame' not in err:
        return ['D4 a box whose frame is on no drive did not fail the build '
                '(%r)' % (err,)]
    return []


def check_collect_unreadable_frame():
    p = _frame_on_drive('970000000003')
    with open(p, 'r+b') as fh:
        fh.truncate(200)
    err, _, _ = run_collect('dogbin', {'hard_negative': [
        _det_row('hard_negative', 'not_dog', '970000000003', BOX)]},
        store_wh={'970000000003': (1600, 900)})
    if not err or 'readable full-resolution frame' not in err:
        return ['D4 a frame that is on disk but cannot be decoded (a drive '
                'dropping mid-build) was dropped instead of failing: %r'
                % (err,)]
    return []


def check_collect_frame_size_vs_store():
    """A detector box on a frame decoded at another size than the store's."""
    _frame_on_drive('970000000004')
    err, _, _ = run_collect('dogbin', {'hard_negative': [
        _det_row('hard_negative', 'not_dog', '970000000004', BOX)]},
        store_wh={'970000000004': (3200, 1800)})
    if not err or 'orig_w' not in err:
        return ['D4 a detector box was cut from a frame whose size is not '
                'the store\'s orig_w x orig_h: %r' % (err,)]
    return []


def check_collect_ls_size_mismatch():
    _frame_on_drive('970000000005')
    task = {'id': 5, 'image': 's3://b/p/970000000005.jpg',
            'label': [{'x': 20, 'y': 20, 'width': 10, 'height': 10,
                       'rectanglelabels': ['unleashed dog'],
                       'original_width': 4000, 'original_height': 2250}]}
    err, _, _ = run_collect('leash', {}, tasks=[task])
    if not err or 'size' not in err:
        return ['D4 a Label Studio box on a frame of another size than Label '
                'Studio recorded did not fail the build: %r' % (err,)]
    return []


def check_collect_seq_disagreement():
    _frame_on_drive('970000000006')
    err, _, _ = run_collect('dogbin', {'gate_audit': [
        _audit_row('970000000006', 'dog', BOX, verdict_seq='VERDICT')]},
        seqs={'970000000006': 'CATALOG'})
    if not err or 'disagrees' not in err:
        return ['D1 a verdict whose sequence the catalog contradicts did not '
                'fail the build: %r' % (err,)]
    return []


def check_collect_framing_check():
    _frame_on_drive('970000000007')
    err, _, _ = run_collect('dogbin', {'gate_audit': [
        _audit_row('970000000007', 'dog', BOX, pad=0.3)]})
    if not err or 'framed' not in err:
        return ['D4 an audit crop cut with a 0.3 margin passed as '
                'production-framed: %r' % (err,)]
    return []


def check_collect_empty_source():
    _frame_on_drive('970000000008')
    err, _, _ = run_collect('leash', {'leash_audit': [
        _audit_row('970000000008', 'leashed', BOX, stage='leash')]},
        allow_empty=[])
    bad = []
    if not err or 'leash_call' not in err or 'ls_box' not in err:
        bad.append('D3 a leash build with no Label Studio rows and no leash '
                   'calls did not refuse: %r' % (err,))
    err, rows, _ = run_collect('leash', {'leash_audit': [
        _audit_row('970000000008', 'leashed', BOX, stage='leash')]},
        allow_empty=['ls_box', 'leash_call'])
    if err or len(rows) != 1:
        bad.append('D3 --allow-empty-source did not let the build through: '
                   '%r' % (err,))
    return bad


def check_collect_exclusions_and_vetoes():
    """label_flags / conflict exclusions, the LS "other animal" veto, the
    other family's verdicts and an empty frame each take their row out."""
    bad = []
    for i in range(10, 16):
        _frame_on_drive('9700000000%02d' % i)
    # excluded image
    err, rows, _ = run_collect('dogbin', {'hard_negative': [
        _det_row('hard_negative', 'not_dog', '970000000010', BOX),
        _det_row('hard_negative', 'not_dog', '970000000011', BOX)]},
        excl={'970000000010': 'label_flags'})
    if err or any(r['image_id'] == '970000000010' for r in rows):
        bad.append('D6 a label-flagged image is in the collection (%r)'
                   % (err,))
    # leash: LS says "other animal" over the same box
    task = {'id': 12, 'image': 's3://b/p/970000000012.jpg',
            'label': [{'x': 25, 'y': 33.3, 'width': 7.5, 'height': 13.3,
                       'rectanglelabels': ['other animal'],
                       'original_width': 1600, 'original_height': 900}]}
    err, rows, conf = run_collect('leash', {'leash_audit': [
        _audit_row('970000000012', 'unleashed', BOX, stage='leash')]},
        tasks=[task], allow_empty=['ls_box', 'leash_call'])
    if err or rows:
        bad.append('D6 a leash verdict on a box Label Studio calls "other '
                   'animal" survived (%r)' % (err,))
    # leash: the gate audit says not_dog on the same detector box
    veto = {'image_id': '970000000013', 'det_key': 'd0', 'box': list(BOX),
            'label': 'not_dog', 'source': 'gate_audit',
            'uid': 'ga_970000000013_d0.jpg'}
    err, rows, conf = run_collect('leash', {'leash_audit': [
        _audit_row('970000000013', 'unleashed', BOX, stage='leash')]},
        cross=[veto], allow_empty=['ls_box', 'leash_call'])
    if err or rows or not any(c['kind'] == 'leash_vs_gate_not_dog'
                              for c in conf):
        bad.append('D6 an "unleashed" box the gate audit called not_dog '
                   'trained the leash model (%r)' % (err,))
    # dogbin: a leash verdict says the not_dog box is a dog
    veto = dict(veto, image_id='970000000014', label='unleashed',
                source='leash_audit', uid='la_970000000014_d0.jpg')
    err, rows, conf = run_collect('dogbin', {'gate_audit': [
        _audit_row('970000000014', 'not_dog', BOX)]}, cross=[veto])
    if err or rows:
        bad.append('D6 a not_dog box the leash audit called "unleashed" '
                   'trained the dog-bin (%r)' % (err,))
    # a dog row on a frame a person declared empty
    err, rows, conf = run_collect('dogbin', {'hard_positive': [
        _det_row('hard_positive', 'dog', '970000000015', BOX)]},
        empty={'970000000015': {'by': 't', 'at': 1}})
    if err or rows:
        bad.append('D6 a dog crop on a frame the annotate page says has no '
                   'dog survived (%r)' % (err,))
    return bad


def check_exclusion_records():
    """exclusions() itself reads label_flags; the leash conflict list is
    RETIRED -- consulted and digested, never applied.

    data/leash_label_conflicts.json named 19 crops on 10 images that
    "disagreed" between two Label Studio lineages; its own commit (de27f267)
    called that a detection-index misalignment, and side by side every dog
    in all 10 images carries the same label in both. Applied, it dropped 34
    good ls_box rows -- 10 of the only 13 Label Studio scenes with a leashed
    and an unleashed dog together. So the image it names must NOT be
    excluded, its sha256 must still be recorded (a reader of the bundle can
    see the file was there), and a status line must say it was retired.
    """
    import types
    fake = types.ModuleType('label_flags')
    fake.DB_PATH = os.path.join(TMP, 'lf.db')
    fake.flagged_files = lambda: {'x': {'image_id': '971000000001'}}
    conflicts = os.path.join(TMP, 'excl_data', 'leash_label_conflicts.json')
    os.makedirs(os.path.dirname(conflicts), exist_ok=True)
    with open(conflicts, 'w') as fh:
        json.dump(['le_971000000002_0.jpg'], fh)
    saved_mod, saved_d = sys.modules.get('label_flags'), cs.D
    sys.modules['label_flags'] = fake
    cs.D = lambda *p: os.path.join(TMP, 'excl_data', *p)
    try:
        got, digests = cs.exclusions('leash', os.path.join(TMP, 'no_root'))
    finally:
        cs.D = saved_d
        if saved_mod is None:
            sys.modules.pop('label_flags', None)
        else:
            sys.modules['label_flags'] = saved_mod
    bad = []
    if got.get('971000000001') != 'label_flags':
        bad.append('D6 exclusions() ignores a label-flagged image')
    if '971000000002' in got:
        bad.append('D6 exclusions() still applies the retired leash label '
                   'conflicts (%r)' % (got.get('971000000002'),))
    if digests.get('leash_label_conflicts') != cs.sha256_file(conflicts):
        bad.append('D6 the retired conflicts file is not digested: %r'
                   % (digests.get('leash_label_conflicts'),))
    if 'retired' not in str(digests.get('leash_label_conflicts:status')):
        bad.append('D6 no status line says the conflicts file was retired: '
                   '%r' % (digests.get('leash_label_conflicts:status'),))
    return bad


def check_leash_calls():
    """The DETECTOR's box is cut and the review page's redraw kept as
    provenance.redraw_box (the leash rule, user decision 2026-11-15: 12 of
    the 13 leash calls collected that day had a redraw, and production cuts
    the detector box); an audit:leash record is never read as one; repeat
    calls collapse, disagreeing calls drop."""
    import types
    import harvest_flagged as hf
    db = os.path.join(TMP, 'leash.db')
    open(db, 'w').close()
    got = [('1786000000001_972000000001_090.jpg', 972000000001, 'leashed',
            0.90, 1, 100, 'a'),
           ('1786000000002_972000000001_090.jpg', 972000000001, 'leashed',
            0.90, 2, 200, 'b'),
           ('1786000000003_972000000002_080.jpg', 972000000002, 'leashed',
            0.80, 3, 300, 'a'),
           ('1786000000004_972000000002_080.jpg', 972000000002, 'unleashed',
            0.80, 4, 400, 'b')]

    class _Q:
        def execute(self, *_):
            return self

        def fetchall(self):
            return list(got)

        def close(self):
            pass
    fake = types.ModuleType('leash_store')
    fake.DB_PATH = db
    fake.connect = lambda: _Q()
    corr = os.path.join(TMP, 'corr.jsonl')
    with open(corr, 'w') as fh:
        fh.write(json.dumps({'image_id': '972000000001', 'det_idx': 0,
                             'x1': 1, 'y1': 2, 'x2': 300, 'y2': 400,
                             'crop': '1786000000001_972000000001_090.jpg'})
                 + '\n')
        fh.write(json.dumps({'image_id': '972000000001', 'det_idx': 0,
                             'x1': 0, 'y1': 0, 'x2': 900, 'y2': 900,
                             'crop': 'audit:leash'}) + '\n')
    saved = (sys.modules.get('leash_store'), hf.boxes_for)
    sys.modules['leash_store'] = fake
    hf.boxes_for = lambda ids, root, python_bin=None: {
        i: [(0, 10.0, 20.0, 110.0, 220.0, 0.9 if i.endswith('1') else 0.8)]
        for i in ids}
    try:
        rows, c, conf = cs.leash_call_candidates('x', corrections=corr)
    finally:
        hf.boxes_for = saved[1]
        if saved[0] is None:
            sys.modules.pop('leash_store', None)
        else:
            sys.modules['leash_store'] = saved[0]
    bad = []
    one = [r for r in rows if r['image_id'] == '972000000001']
    if len(one) != 1:
        bad.append('D6 two agreeing calls on one box became %d rows'
                   % (len(one),))
    elif one[0]['pool_box'] != [10.0, 20.0, 110.0, 220.0]:
        bad.append('D6 a leash call does not carry its detector box')
    elif one[0]['box'] != one[0]['pool_box']:
        bad.append('D3 a leash call is cut from box %r, not its detector box '
                   '%r (production cuts the detector box)'
                   % (one[0]['box'], one[0]['pool_box']))
    elif one[0]['provenance'].get('redraw_box') != [1.0, 2.0, 300.0, 400.0]:
        bad.append('D3 a leash call\'s review-page redraw is not kept as '
                   'provenance.redraw_box (%r; an audit:leash record must '
                   'not stand in for it either)'
                   % (one[0]['provenance'].get('redraw_box'),))
    if any(r['image_id'] == '972000000002' for r in rows) or not conf:
        bad.append('D6 a box called both leashed and unleashed was kept')
    return bad


def check_review_corrections_skip_audit():
    import harvest_flagged as hf
    p = os.path.join(TMP, 'corr2.jsonl')
    with open(p, 'w') as fh:
        fh.write(json.dumps({'image_id': '973000000001', 'det_idx': 0,
                             'x1': 1, 'y1': 1, 'x2': 5, 'y2': 5,
                             'crop': 'x.jpg'}) + '\n')
        fh.write(json.dumps({'image_id': '973000000001', 'det_idx': 0,
                             'x1': 0, 'y1': 0, 'x2': 99, 'y2': 99,
                             'crop': 'audit:leash'}) + '\n')
        fh.write(json.dumps({'image_id': '973000000002', 'det_idx': 1,
                             'x1': 0, 'y1': 0, 'x2': 99, 'y2': 99,
                             'crop': 'audit:gate'}) + '\n')
    got = hf.read_corrections(p)
    if got != {('973000000001', 0): (1.0, 1.0, 5.0, 5.0)}:
        return ['D3 the review page\'s corrections reader takes the audit '
                'pages\' redraws too: %r' % (got,)]
    return []


def check_frame_labels():
    p = os.path.join(TMP, 'frame_labels.jsonl')
    with open(p, 'w') as fh:
        for r in ({'image_id': '974000000001', 'boxes': [[1, 2, 30, 40]],
                   'empty': False, 'w': 100, 'h': 80, 'at': 1, 'by': 'a',
                   'batch': 'b1'},
                  {'image_id': '974000000001', 'boxes': [[5, 5, 50, 60],
                                                         [60, 10, 90, 40]],
                   'empty': False, 'w': 100, 'h': 80, 'at': 2, 'by': 'a',
                   'batch': 'recheck_1'},
                  {'image_id': '974000000002', 'boxes': [], 'empty': True,
                   'w': 100, 'h': 80, 'at': 3, 'by': 'b', 'batch': 'b1'}):
            fh.write(json.dumps(r) + '\n')
        fh.write('{"torn')
    rows, empty, _ = cs.frame_label_candidates(p)
    bad = []
    boxes = sorted(tuple(r['box']) for r in rows)
    if boxes != [(5.0, 5.0, 50.0, 60.0), (60.0, 10.0, 90.0, 40.0)]:
        bad.append('D3 the annotate page\'s LATEST save is not the label: %r'
                   % (boxes,))
    if any(r['label'] != 'dog' or r['source'] != 'frame_label'
           for r in rows):
        bad.append('D3 an annotate-page box is not a frame_label dog')
    if set(empty) != {'974000000002'}:
        bad.append('D6 a frame declared empty is not returned as one')
    # and the cut refuses a frame of another size than the page recorded
    fr = _img(os.path.join(TMP, 'frames', 'fl.jpg'), w=120, h=80, seed=9)
    r = dict(rows[0], uid='fl_974000000001_b0.jpg')
    _, _, res = cs.cut_frame((fr, [(0, r)], os.path.join(TMP, 'cut')))
    if res[0][2] != 'fl_size_mismatch':
        bad.append('D4 an annotate-page box was cut from a frame of another '
                   'size: %r' % (res[0],))
    return bad


def check_dedup_joins_detector_box(strip_pool=False):
    """A drawn box joins the DETECTOR's box, not a redraw of it; a drawn dog
    box over a not_dog detector box is a conflict."""
    bad = []
    det = (100, 100, 200, 200)
    ed = row('gate_audit', 'dog', '975000000001', 'd0', 'J',
             box=(60, 60, 320, 320), framing='audit_edited')
    ed['pool_box'] = None if strip_pool else list(det)
    ls = row('ls_box', 'dog', '975000000001', 'b0', 'J',
             box=(102, 101, 201, 203))
    kept, _, _, _, _ = rcd.dedup_rows([ed, ls], 0)
    if len(kept) != 1:
        bad.append('D6 a Label Studio box at IoU 0.97 with detector box d0 '
                   'was not joined to it because the audit row carries a '
                   'redraw')
    hn = row('hard_negative', 'not_dog', '975000000002', 'd0', 'K', box=det)
    hn['pool_box'] = None if strip_pool else list(det)
    hn['box'] = list(det) if not strip_pool else [0, 0, 400, 400]
    fl = row('frame_label', 'dog', '975000000002', 'b0', 'K',
             box=(105, 100, 210, 205))
    kept, _, conf, _, _ = rcd.dedup_rows([hn, fl], 0)
    if kept or not conf:
        bad.append('D6 a dog box drawn on the annotate page over a not_dog '
                   'detector box did not drop both as a conflict')
    a = row('ls_box', 'dog', '975000000003', 'b0', 'L', box=det)
    b = row('frame_label', 'dog', '975000000003', 'b0', 'L',
            box=(101, 99, 199, 202))
    kept, _, _, _, _ = rcd.dedup_rows([a, b], 0)
    if len(kept) != 1:
        bad.append('D6 a Label Studio box and an annotate-page box of one dog '
                   'stayed two rows')
    return bad


def _leash(r):
    r['family'] = 'leash'
    return r


def check_dedup_rank_by_family(rank=None):
    """One box, two agreeing rows: the dog-bin keeps the redraw, the leash
    model the detector box's crop (audit_full)."""
    saved = rcd._rank
    if rank:
        rcd._rank = rank
    try:
        bad = []
        for fam, a, b, want in (
                ('dogbin', 'dog', 'dog', 'audit_edited'),
                ('leash', 'leashed', 'leashed', 'audit_full')):
            iid = '975100000001' if fam == 'dogbin' else '975100000002'
            full = row('gate_audit' if fam == 'dogbin' else 'leash_audit', a,
                       iid, 'd0', 'RK', box=(100, 100, 200, 200),
                       framing='audit_full')
            ed = row('gate_audit' if fam == 'dogbin' else 'leash_audit', b,
                     iid, 'd0', 'RK', box=(60, 60, 260, 260),
                     framing='audit_edited')
            ed['uid'] = ed['uid'].replace('.jpg', '_e.jpg')
            for r in (full, ed):
                r['pool_box'] = [100, 100, 200, 200]
                r['family'] = fam
            kept, _, _, _, _ = rcd.dedup_rows([ed, full], 0)
            if len(kept) != 1 or kept[0]['framing'] != want:
                bad.append('D6 %s: of a redraw and the detector box\'s crop '
                           'of one box, %r survived, want %s'
                           % (fam, [r['framing'] for r in kept], want))
        # a Label Studio box of the same dog (IoU 0.575 with the detector
        # box, leash-inclusive): the dog-bin keeps the drawn crop, the leash
        # model the detector's -- audit_full over ls_box AND over a leash
        # call of the same box
        for fam, src, lab, want in (
                ('dogbin', 'gate_audit', 'dog', 'ls_box'),
                ('leash', 'leash_audit', 'leashed', 'leash_audit')):
            iid = '975100000003' if fam == 'dogbin' else '975100000004'
            full = row(src, lab, iid, 'd0', 'RL', box=(100, 100, 200, 200),
                       framing='audit_full')
            full['pool_box'] = [100, 100, 200, 200]
            ls = row('ls_box', lab, iid, 'b0', 'RL', box=(95, 95, 215, 240))
            ls['pool_box'] = None
            group = [ls, full]
            if fam == 'leash':
                call = row('leash_call', lab, iid, 'd0', 'RL',
                           box=(100, 100, 200, 200))
                call['pool_box'] = [100, 100, 200, 200]
                group.append(call)
            for r in group:
                r['family'] = fam
            kept, _, _, _, cnt = rcd.dedup_rows(group, 0)
            if len(kept) != 1 or kept[0]['source'] != want:
                bad.append('D6 %s: of a Label Studio crop and the detector '
                           'box\'s crop of one box, %r survived, want %s'
                           % (fam, [r['source'] for r in kept], want))
        bad += _check_collapse_by_family()
        return bad
    finally:
        rcd._rank = saved


def _check_collapse_by_family():
    """dHash (step 3) and embedding (step 4) collapses of same-class crops:
    the dog-bin keeps the largest file; leash keeps the detector-box crop
    even when a drawn crop of the dog is the larger file (a leash-inclusive
    box usually is)."""
    bad = []
    for step in ('dhash', 'embedding'):
        for fam, src, lab, want in (
                ('dogbin', 'gate_audit', 'dog', 'ls_box'),
                ('leash', 'leash_audit', 'leashed', 'leash_audit')):
            tag = '%d%d' % (step == 'dhash', fam == 'leash')
            small = row(src, lab, '97510001%s01' % tag, 'd0', 'CS' + tag,
                        framing='audit_full')
            small['pool_box'] = list(small['box'])
            big_path = _img(os.path.join(TMP, 'crops_big', tag + '.jpg'),
                            w=160, h=140, seed=_SEED[0] + 7)
            big = row('ls_box', lab, '97510001%s02' % tag, 'b0', 'CT' + tag,
                      path=big_path)
            big['pool_box'] = None
            assert os.path.getsize(big['path']) > \
                os.path.getsize(small['path'])
            for r in (small, big):
                r['family'] = fam
            if step == 'dhash':
                big['dhash'] = small['dhash']
                kept, _, _, _, _ = rcd.dedup_rows([small, big], 6)
            else:
                kept, _, _, _, _ = rcd.dedup_rows(
                    [small, big], 0, emb_groups=[[small['path'],
                                                  big['path']]])
            if len(kept) != 1 or kept[0]['source'] != want:
                bad.append('D6 %s: the %s collapse of a detector crop and a '
                           'larger drawn crop kept %r, want %s'
                           % (fam, step, [r['source'] for r in kept], want))
    return bad


def _phase2_rank(row):
    # leash: only the redraw ranks last; a Label Studio box still outranks
    # the detector's crop (the rank before the 2026-11-15 review)
    if row.get('framing') == 'audit_edited':
        return 5 if rcd.family_of(row) == 'leash' else 0
    if row.get('source') in cs.DRAWN_SOURCES:
        return 1
    if row.get('framing') == 'audit_full':
        return 2
    if row.get('source') in ('hard_negative', 'hard_positive'):
        return 3
    return 4


def _size_only_collapse(members):
    return max(members, key=lambda r: (rcd._size_of(r), r['uid']))


def _old_rank(row):
    # redraw-wins for every family -- the rule before 2026-11-15
    if row.get('framing') == 'audit_edited':
        return 0
    if row.get('source') in cs.DRAWN_SOURCES:
        return 1
    if row.get('framing') == 'audit_full':
        return 2
    return 3


def check_dedup_overlapping_det_keys(dedup=None):
    """Two det keys on one image whose POOL boxes overlap at IoU >= LS_IOU
    are one box (the gen-1 store double-boxes): two labels drop both, one
    label keeps one, a drawn box over the pair joins it; a redraw or a
    distant box does not join, and check_rows refuses the pair."""
    dedup = dedup or rcd.dedup_rows
    bad = []
    a_box, b_box = [100.0, 100.0, 200.0, 200.0], [103.0, 102.0, 204.0, 201.0]
    assert cs.iou(a_box, b_box) >= cs.LS_IOU

    def det(source, label, iid, dk, pool, box=None, framing=None):
        r = _leash(row(source, label, iid, dk, 'OV' + iid[-2:],
                       box=box or pool, framing=framing))
        r['pool_box'] = list(pool) if pool else None
        return r
    # 1. two labels, two keys, one box -> both dropped, as a conflict
    x = det('leash_audit', 'unleashed', '975200000001', 'd1', a_box)
    y = det('leash_call', 'leashed', '975200000001', 'd0', b_box)
    kept, _, conf, _, cnt = dedup([x, y], 0)
    if kept or not any(c['kind'] == 'same_box_two_labels_overlap'
                       for c in conf):
        bad.append('D6 one detector box under two det keys and two labels '
                   'trained (kept %r)' % ([r['uid'] for r in kept],))
    # 2. one label -> one row
    x = det('leash_audit', 'leashed', '975200000002', 'd1', a_box)
    y = det('leash_call', 'leashed', '975200000002', 'd0', b_box)
    kept, _, _, _, cnt = dedup([x, y], 0)
    if len(kept) != 1 or not cnt.get('detector_keys_joined_by_overlap'):
        bad.append('D6 one detector box under two det keys stayed %d rows'
                   % (len(kept),))
    # 3. a drawn box over the double-boxed pair joins it -- not "ambiguous"
    x = det('leash_audit', 'leashed', '975200000003', 'd1', a_box)
    y = det('leash_call', 'leashed', '975200000003', 'd0', b_box)
    z = _leash(row('ls_box', 'leashed', '975200000003', 'b0', 'OV03',
                   box=(101, 101, 202, 202)))
    kept, _, conf, _, _ = dedup([x, y, z], 0)
    if len(kept) != 1:
        bad.append('D6 a drawn box over one double-boxed detector box did not '
                   'join it (kept %d, conflicts %r)'
                   % (len(kept), [c['kind'] for c in conf]))
    # 4. two dogs side by side (IoU < LS_IOU) stay two
    x = det('leash_audit', 'leashed', '975200000004', 'd0', a_box)
    y = det('leash_audit', 'unleashed', '975200000004', 'd1',
            [180.0, 100.0, 280.0, 200.0])
    kept, _, conf, _, _ = dedup([x, y], 0)
    if len(kept) != 2:
        bad.append('D6 two detector boxes at IoU %.2f were joined as one'
                   % cs.iou(a_box, [180.0, 100.0, 280.0, 200.0]))
    # 5. the join is on the POOL box only: a row with no pool_box whose
    # (redrawn) box overlaps is matched by key alone
    x = det('leash_audit', 'leashed', '975200000005', 'd0', a_box)
    y = det('gate_audit', 'unleashed', '975200000005', 'd1', None,
            box=b_box, framing='audit_full')
    kept, _, _, _, _ = dedup([x, y], 0)
    if len(kept) != 2:
        bad.append('D6 a row with no pool_box was joined on its own box')
    # 6. the assertion: such a pair on a plan fails check_rows
    x = det('leash_audit', 'leashed', '975200000006', 'd1', a_box)
    y = det('leash_call', 'leashed', '975200000006', 'd0', b_box)
    for r in (x, y):
        r['split'], r['group'] = 'train', r['sequence']
    if not any('one box' in f for f in rcd.check_rows([x, y])):
        bad.append('D8 check_rows passed one detector box under two det keys')
    return bad


def _keyonly_dedup(rows, hamming=6, emb_groups=None, iou_thr=0.5):
    """dedup_rows with the det-key join by key only (before 2026-11-15)."""
    f = _mutated(rcd.dedup_rows,
                 'if _iou(lst[i][1], lst[j][1]) >= iou_thr and',
                 'if False and')
    return f(rows, hamming, emb_groups, iou_thr)


def _mutated(fn, old, new):
    """fn re-compiled from its own source with `old` -> `new`, in a copy of
    its module's namespace (the real function is left alone)."""
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(fn))
    if src.count(old) != 1:
        raise AssertionError('mutant anchor %r not found once in %s'
                             % (old, fn.__name__))
    ns = dict(fn.__globals__)
    exec(compile(src.replace(old, new), fn.__code__.co_filename, 'exec'), ns)
    return ns[fn.__name__]


def check_leash_framing_assertion(check=None):
    """Assertion 11 on a plan: no leash redraw, no leash detector row cut
    from a box other than its pool_box -- and the gate's redraws pass."""
    check = check or rcd.check_rows
    bad = []

    def plan():
        out = []
        for k, (src, lab, sp) in enumerate((
                ('leash_audit', 'leashed', 'train'),
                ('leash_call', 'unleashed', 'train'),
                ('leash_audit', 'unleashed', 'val'),
                ('ls_box', 'leashed', 'val'))):
            r = _leash(row(src, lab, '97530000000%d' % k,
                           'b0' if src == 'ls_box' else 'd0', 'LF%d' % k,
                           box=(10, 10, 50, 50)))
            r['pool_box'] = None if src == 'ls_box' else [10, 10, 50, 50]
            r['split'], r['group'] = sp, r['sequence']
            out.append(r)
        return out
    if check(plan()):
        bad.append('D8 a clean leash plan fails: %s' % (check(plan()),))
    p = plan()
    p[0]['framing'] = 'audit_edited'
    if not check(p):
        bad.append('D8 check_rows passed a leash row framed audit_edited')
    p = plan()
    p[1]['box'] = [5, 5, 80, 90]
    if not check(p):
        bad.append('D8 check_rows passed a leash call cut from a redraw, not '
                   'its pool_box')
    p = plan()
    p[2]['box'] = [5, 5, 80, 90]
    if not check(p):
        bad.append('D8 check_rows passed a leash audit row cut from a box '
                   'other than its pool_box')
    g = row('gate_audit', 'dog', '975300000009', 'd0', 'LF9',
            box=(5, 5, 80, 90), framing='audit_edited')
    g['pool_box'] = [10, 10, 50, 50]
    g['split'], g['group'] = 'train', 'LF9'
    if check(plan() + [g]):
        bad.append('D8 the leash framing rule fails a dog-bin redraw, which '
                   'is the gate\'s rule: %s' % (check(plan() + [g]),))
    return bad


def _creator_rows():
    """Creator cA: 8 dog crops in sequence S1 (4 images x 2) and one each in
    S2-S4, 2 not_dog; creator cB: 3 dog; 7 rows with no creator (1 dog, 6
    not_dog -- more of one class than the cap, so treating "no creator" as
    one creator would cut them)."""
    rows = []
    k = [0]

    def mk(label, seq, creator, iid=None):
        k[0] += 1
        iid = iid or '9754000%05d' % k[0]
        r = row('gate_audit', label, iid, 'd%d' % k[0], seq)
        r['capture'] = ({'creator': creator, 'ts': 1e6 * k[0], 'lon': 0.0,
                         'lat': 0.0} if creator else None)
        r['split'], r['group'] = 'train', seq
        rows.append(r)
    for i in range(4):
        iid = '97540009%04d' % i
        mk('dog', 'S1', 'cA', iid)
        mk('dog', 'S1', 'cA', iid)
    for seq in ('S2', 'S3', 'S4'):
        mk('dog', seq, 'cA')
    mk('not_dog', 'S5', 'cA')
    mk('not_dog', 'S6', 'cA')
    for _ in range(3):
        mk('dog', 'S7', 'cB')
    mk('dog', 'S8', None)
    for i in range(6):
        mk('not_dog', 'S9%d' % i, None)
    return rows


def _image_tier_rows():
    """Creator cC, one sequence, two images with two dog crops each -- the
    det keys chosen so that in plain sha256(0:uid) order BOTH of image A's
    crops come first. A cap of 2 keeps one of each image only if the image
    tier of _spread_order runs."""
    def key(iid, d):
        return rcd._hash_key(0, cs.uid_for('gate_audit', iid, d))
    ia, ib = '975410000001', '975410000002'
    da = sorted(('d%d' % i for i in range(40)), key=lambda d: key(ia, d))[:2]
    top = max(key(ia, d) for d in da)
    db = [d for d in ('d%d' % i for i in range(400)) if key(ib, d) > top][:2]
    out = []
    for iid, dets in ((ia, da), (ib, db)):
        for d in dets:
            r = row('gate_audit', 'dog', iid, d, 'S10')
            r['capture'] = {'creator': 'cC', 'ts': 5e6, 'lon': 0.0,
                            'lat': 0.0}
            r['split'], r['group'] = 'train', 'S10'
            out.append(r)
    return out


def check_creator_cap(cap_fn=None):
    """At most N per (creator, class), distinct sequences first, rows with no
    creator untouched, the same answer twice -- and check_rows holds it."""
    cap_fn = cap_fn or rcd.cap_creators
    rows = _creator_rows()
    bad = []
    kept, dropped, by_creator = cap_fn(rows, 4, 0)
    per = collections.Counter((rcd.creator_of(r), r['label']) for r in kept)
    if per.get(('cA', 'dog')) != 4:
        bad.append('D9 creator cA kept %r dog crops under a cap of 4'
                   % (per.get(('cA', 'dog')),))
    if per.get(('cA', 'not_dog')) != 2 or per.get(('cB', 'dog')) != 3:
        bad.append('D9 the creator cap took rows from a (creator, class) '
                   'under it: %r' % (dict(per),))
    seqs = sorted(r['sequence'] for r in kept
                  if rcd.creator_of(r) == 'cA' and r['label'] == 'dog')
    if seqs != ['S1', 'S2', 'S3', 'S4']:
        bad.append('D9 the creator cap kept %r -- not one crop of each of '
                   'its four sequences first' % (seqs,))
    if sum(1 for r in kept if not rcd.creator_of(r)) != 7:
        bad.append('D9 a row with no creator was capped (%d of 7 kept)'
                   % (sum(1 for r in kept if not rcd.creator_of(r)),))
    if dict(dropped) != {'dog': 7} or dict(by_creator) != {'cA': 7}:
        bad.append('D9 the creator cap reports %r / %r, want 7 cA dog rows'
                   % (dict(dropped), dict(by_creator)))
    again, _, _ = cap_fn(_creator_rows_like(rows), 4, 0)
    if sorted(r['uid'] for r in again) != sorted(r['uid'] for r in kept):
        bad.append('D9 the creator cap is not deterministic for a seed')
    if rcd.check_rows(kept, max_per_creator=4):
        bad.append('D8 check_rows fails a creator-capped plan: %s'
                   % (rcd.check_rows(kept, max_per_creator=4),))
    if not rcd.check_rows(rows, max_per_creator=4):
        bad.append('D8 check_rows passed 11 crops of one (creator, class) '
                   'under a cap of 4')
    # within a sequence, one crop per image before a second of any
    tier = _image_tier_rows()
    kept, _, _ = cap_fn(tier, 2, 0)
    if sorted({r['image_id'] for r in kept}) != \
            sorted({r['image_id'] for r in tier}):
        bad.append('D9 a cap of 2 on two images of one sequence kept %r -- '
                   'not one crop of each image first'
                   % (sorted(r['uid'] for r in kept),))
    # a creator with no capture TIME is still a creator: a NULL captured_at
    # must not exempt an image from the cap (it joins no twin, though)
    cap = rcd._capture_of(None, None, 'C1')
    if (cap or {}).get('creator') != 'C1' or rcd.capture_known(cap):
        bad.append('D9 a creator with no captured_at reads as %r' % (cap,))
    else:
        nots = []
        for i in range(6):
            r = row('gate_audit', 'dog', '9754200000%02d' % i, 'd0',
                    'NT%d' % i)
            r['capture'] = dict(cap)
            r['split'], r['group'] = 'train', r['sequence']
            nots.append(r)
        kept, _, _ = cap_fn(nots, 4, 0)
        if len(kept) != 4:
            bad.append('D9 6 crops of creator C1 with no captured_at kept %d '
                       'under a cap of 4' % (len(kept),))
        if rcd.capture_twins([(r['image_id'], r['capture']) for r in nots]):
            bad.append('D9 captures with no time were paired as twins')
    return bad


def _creator_rows_like(rows):
    return [dict(r) for r in reversed(rows)]


def _hash_only_spread(rows, seed):
    return sorted(rows, key=lambda r: (rcd._hash_key(seed, r['uid']),
                                       r['uid']))


def _sequence_only_spread(rows, seed):
    """_spread_order without its image tier: the k-th row of each sequence
    first, but in plain hash order within a sequence."""
    by_seq = collections.defaultdict(list)
    for r in rows:
        by_seq[str(r['sequence'])].append(r)
    keyed = []
    for lst in by_seq.values():
        lst = _hash_only_spread(lst, seed)
        for k, r in enumerate(lst):
            keyed.append(((k, rcd._hash_key(seed, r['uid']), r['uid']), r))
    keyed.sort(key=lambda t: t[0])
    return [r for _, r in keyed]


def _band_rows(split='train', tag='0', mix=((2, 8), (6, 4))):
    """(rows, pos rate): by default <=64 px 2 dog / 8 not_dog, >192 px 6 dog
    / 4 not_dog, so the split's dog rate is 0.4 and size predicts the label;
    `mix` sets the (dog, not_dog) counts of the two bands. Crops are
    NON-square: their long side sits in another band. One more row has no
    crop_wh: it is kept and counted, never banded."""
    rows = []
    k = [0]

    def mk(label, wh):
        k[0] += 1
        r = row('gate_audit', label, '9755%s%07d' % (tag, k[0]), 'd0',
                'B%s%03d' % (tag, k[0]))
        r['crop_wh'] = list(wh)
        r['split'], r['group'] = split, r['sequence']
        rows.append(r)
    (sp, sn), (bp, bn) = mix
    for _ in range(sp):
        mk('dog', (50, 230))
    for _ in range(sn):
        mk('not_dog', (50, 230))
    for _ in range(bp):
        mk('dog', (230, 200))
    for _ in range(bn):
        mk('not_dog', (230, 200))
    mk('not_dog', (60, 60))
    rows[-1]['crop_wh'] = None
    return rows, (sp + bp) / float(sp + sn + bp + bn)


def check_band_balance(balance=None):
    """Within each split and each short-side band the dog rate becomes THAT
    split's own rate -- 0.4 in train, 0.5 in val (not 0.5 everywhere, not
    the 0.45 of both together) -- whole splits only, rows with no crop_wh
    kept and counted, the drops reported per (split, band, class), and
    check_rows holds it."""
    balance = balance or rcd.balance_bands
    tr, rate_tr = _band_rows('train', '0')
    va, rate_va = _band_rows('val', '1', mix=((3, 7), (7, 3)))
    rates = {'train': rate_tr, 'val': rate_va}
    bad = []
    kept, rep = balance(tr + va, 'dog', 0)
    for sp in ('train', 'val'):
        rate = rates[sp]
        got = ((rep.get('splits') or {}).get(sp) or {})
        if got.get('target_pos_rate') is None or \
                abs(got['target_pos_rate'] - rate) > 1e-9:
            bad.append('D9 %s balanced to %r, want its own rate %.2f'
                       % (sp, got.get('target_pos_rate'), rate))
        if got.get('no_crop_wh') != 1:
            bad.append('D9 %s reports %r row(s) with no crop_wh, want 1'
                       % (sp, got.get('no_crop_wh')))
        nowh = [r for r in kept if r['split'] == sp and not r.get('crop_wh')]
        if len(nowh) != 1:
            bad.append('D9 %s kept %d row(s) with no crop_wh, want 1'
                       % (sp, len(nowh)))
        cnt = collections.defaultdict(collections.Counter)
        for r in kept:
            if r['split'] == sp and r.get('crop_wh'):
                cnt[rcd.size_band(min(r['crop_wh']))][r['label']] += 1
        for b, c in sorted(cnt.items()):
            n = c['dog'] + c['not_dog']
            if abs(c['dog'] - rate * n) > 1.0:
                bad.append('D9 %s band %s keeps %d dog of %d, want a dog rate '
                           'of %.1f (the split\'s own)' % (sp, b, c['dog'], n,
                                                             rate))
        if set(cnt) != {'<=64', '>192'}:
            bad.append('D9 %s: bands %r -- the band is not taken on the SHORT '
                       'side' % (sp, sorted(cnt)))
    moved = {r['uid'] for r in kept} - {r['uid'] for r in tr + va}
    if moved or any(r['split'] != ('train' if r['uid'] in {x['uid'] for x in
                                                           tr} else 'val')
                    for r in kept):
        bad.append('D9 the band balance moved rows between splits')
    wants = {'train': {'<=64': {'not_dog': 5}, '>192': {'dog': 3}},
             'val': {'<=64': {'not_dog': 4}, '>192': {'dog': 4}}}
    for sp, want in wants.items():
        got = ((rep.get('splits') or {}).get(sp) or {}).get('dropped')
        if got != want:
            bad.append('D9 %s drops reported as %r, want %r' % (sp, got, want))
    targets = {'positive': 'dog', 'targets': rates}
    if rcd.check_rows(kept, band_balance=targets):
        bad.append('D8 check_rows fails a band-balanced plan: %s'
                   % (rcd.check_rows(kept, band_balance=targets),))
    if not rcd.check_rows(tr + va, band_balance=targets):
        bad.append('D8 check_rows passed bands at dog rate 0.2 / 0.6 and '
                   '0.3 / 0.7 against targets of 0.4 / 0.5')
    return bad


def _rebalance_manifest():
    """30 dog-bin crops, one sequence each: creator cA holds 10 dogs, small
    crops are mostly not_dog and big ones mostly dog."""
    rows = []
    CAPTURES.clear()
    for k in range(30):
        label = 'dog' if k % 3 else 'not_dog'
        iid = '97560000%04d' % k
        r = row('gate_audit', label, iid, 'd0', 'RB%02d' % k)
        big = (label == 'dog') != (k % 7 == 0)
        r['crop_wh'] = [210, 220] if big else [50, 60]
        creator = 'cA' if label == 'dog' and k < 16 else 'c%d' % k
        CAPTURES[iid] = {'creator': creator, 'ts': 1e6 * k, 'lon': 0.0,
                         'lat': 0.0}
        rows.append(r)
    return write_manifest(rows, 'm_rebalance.jsonl')


def check_rebalanced_build(plan_fn=None):
    """--max-per-creator and --balance-size-bands through a whole manifest
    build: recorded as the 'rebalanced' variant, every drop counted,
    re-asserted on disk -- and the plain build of the same manifest is
    'plain' and drops nothing for them. Either flag ALONE is a 'rebalanced'
    build too, recording only its own setting; and a build that did not ask
    for 12 / 13 says so ("not requested") instead of "all 14 hold"."""
    saved = rcd.rebalance_plan
    if plan_fn:
        rcd.rebalance_plan = plan_fn
    try:
        m = _rebalance_manifest()
        tag = random.randrange(1 << 30)
        outs, texts = {}, {}
        for name, extra in (('plain', []),
                            ('reb', ['--max-per-creator', '3',
                                     '--balance-size-bands']),
                            ('reb2', ['--max-per-creator', '3',
                                      '--balance-size-bands']),
                            ('cap', ['--max-per-creator', '3']),
                            ('band', ['--balance-size-bands'])):
            out = os.path.join(TMP, 'out_rb_%s_%d' % (name, tag))
            err, text = run_rebuild(['--manifest', m, '--out', out,
                                     '--no-reserve', '--hamming', '0',
                                     '--val-frac', '0.3', '--execute']
                                    + extra)
            if err:
                return ['D9 the %s build failed: %s' % (name, err)]
            outs[name], texts[name] = out, text
    finally:
        rcd.rebalance_plan = saved
    bad = []
    n = rcd.ASSERTIONS
    for name, want, n_req in (('plain', 'all %d assertions hold (2 not '
                               'requested)' % (n - 2,), 2),
                              ('cap', 'all %d assertions hold (1 not '
                               'requested)' % (n - 1,), 1),
                              ('band', 'all %d assertions hold (1 not '
                               'requested)' % (n - 1,), 1),
                              ('reb', 'all %d assertions hold\n' % (n,), 0)):
        t = texts[name]
        if t.count('assertions on the plan: ' + want) != 1 or \
                t.count('assertions on disk: ' + want) != 1 or \
                t.count('not requested: a ') != 2 * n_req:
            bad.append('D9 the %s build does not report %r on the plan and '
                       'on disk: %r' % (name, want, [
                           x for x in t.splitlines()
                           if 'assertions' in x or 'not requested' in x]))
    for name, cap_want, bands_want in (('cap', 3, False),
                                       ('band', None, True)):
        mm = _read_json(os.path.join(outs[name], 'rebuild_manifest.json'))
        rb = mm.get('rebalance') or {}
        if mm.get('variant') != 'rebalanced' or \
                rb.get('variant') != 'rebalanced' or \
                rb.get('max_per_creator') != cap_want or \
                rb.get('balance_size_bands') is not bands_want or \
                ('creator_cap' in rb) != bool(cap_want) or \
                ('band_balance' in rb) != bands_want:
            bad.append('D9 the %s-only build is recorded as %r with %r'
                       % (name, mm.get('variant'),
                          {k: rb.get(k) for k in (
                              'variant', 'max_per_creator',
                              'balance_size_bands')}))

    def man(o):
        with open(os.path.join(o, 'rebuild_manifest.json')) as fh:
            return json.load(fh)

    def uids(o):
        with open(os.path.join(o, 'crops.jsonl')) as fh:
            return sorted(json.loads(line)['uid'] for line in fh)
    pm, rm = man(outs['plain']), man(outs['reb'])
    if pm.get('variant') != 'plain' or rm.get('variant') != 'rebalanced':
        bad.append('D9 the manifests say %r / %r, not plain / rebalanced'
                   % (pm.get('variant'), rm.get('variant')))
    reb = rm.get('rebalance') or {}
    if reb.get('max_per_creator') != 3 or not reb.get('balance_size_bands'):
        bad.append('D9 the rebalance settings are not recorded: %r'
                   % ({k: reb.get(k) for k in ('max_per_creator',
                                               'balance_size_bands')},))
    if not (reb.get('creator_cap') or {}).get('dropped_total') or \
            not (reb.get('band_balance') or {}).get('dropped_total'):
        bad.append('D9 the rebalanced build dropped nothing (creator %r, '
                   'bands %r)' % ((reb.get('creator_cap') or {})
                                  .get('dropped_total'),
                                  (reb.get('band_balance') or {})
                                  .get('dropped_total')))
    if len(uids(outs['reb'])) + reb.get('creator_cap', {}).get(
            'dropped_total', 0) + reb.get('band_balance', {}).get(
            'dropped_total', 0) != len(uids(outs['plain'])):
        bad.append('D9 the rebalanced build is not the plain one minus what '
                   'its record says it dropped')
    if (pm.get('rebalance') or {}).get('rows_before') != \
            (pm.get('rebalance') or {}).get('rows_after'):
        bad.append('D9 the plain build dropped rows for a rebalance')
    if uids(outs['reb']) != uids(outs['reb2']):
        bad.append('D9 two rebalanced builds of one manifest and seed differ')
    err, _ = run_rebuild(['--verify-only', outs['reb']])
    if err:
        bad.append('D9 --verify-only fails a rebalanced build: %s' % (err,))
    # on disk: the record asks for a cap the files break
    p = os.path.join(outs['reb'], 'rebuild_manifest.json')
    doc = man(outs['reb'])
    doc['rebalance']['max_per_creator'] = 1
    with open(p, 'w') as fh:
        json.dump(doc, fh)
    if not rcd.verify_dataset(outs['reb']):
        bad.append('D9 verify passed a dataset over the creator cap its '
                   'record asks for')
    doc['rebalance']['max_per_creator'] = 3
    for d in doc['rebalance']['band_balance']['splits'].values():
        if d.get('target_pos_rate') is not None:
            d['target_pos_rate'] = 0.05
    with open(p, 'w') as fh:
        json.dump(doc, fh)
    if not rcd.verify_dataset(outs['reb']):
        bad.append('D9 verify passed size bands off the rate its record '
                   'asks for')
    return bad


def _read_json(p):
    with open(p) as fh:
        return json.load(fh)


def _no_rebalance(plan, *a, **k):
    rec = {'variant': 'rebalanced', 'max_per_creator': a[0] or None,
           'balance_size_bands': bool(a[1]), 'rows_before': len(plan),
           'rows_after': len(plan)}
    return plan, rec


def check_reserved_near_copies(near=None):
    """A re-encoded copy of a reserved crop under another image_id and
    sequence never reaches train or val."""
    from PIL import Image
    saved = rcd.near_hashes
    if near:
        rcd.near_hashes = near
    try:
        held = row('gate_audit', 'dog', '976000000001', 'd0', 'RSEQ')
        twin_p = os.path.join(TMP, 'crops', 'dog', 'ga_976000000002_d0.jpg')
        with Image.open(held['path']) as im:
            im.save(twin_p, 'JPEG', quality=80)
        twin = row('gate_audit', 'dog', '976000000002', 'd0', 'OTHER',
                   path=twin_p)
        far = row('gate_audit', 'not_dog', '976000000003', 'd0', 'FAR')
        far2 = row('gate_audit', 'dog', '976000000005', 'd0', 'FAR2')
        # a crop the reserve FILE records (sha + dhash), not in the manifest
        rec_p = os.path.join(TMP, 'crops', 'dog', 'ga_976000000004_d0.jpg')
        with Image.open(far['path']) as im:
            im.save(rec_p, 'JPEG', quality=70)
        rec_dh = rcd.dhash(rec_p)
        res = os.path.join(TMP, 'reserve_twins.json')
        with open(res, 'w') as fh:
            json.dump({'image_ids': ['976000000001', '976000000004'],
                       'sequences': ['RSEQ'],
                       'items': [{'key': '976000000004#0', 'sha256': 'f' * 64,
                                  'dhash': rec_dh}]}, fh)
        out = os.path.join(TMP, 'out_twins_%d' % random.randrange(1 << 30))
        err, _ = run_rebuild(['--manifest', write_manifest(
            [held, twin, far, far2], 'm_twins.jsonl'), '--out', out,
            '--reserve', res, '--hamming', '0', '--execute'])
    finally:
        rcd.near_hashes = saved
    if err:
        return ['D7 the reserve twin build refused: %s' % (err,)]
    with open(os.path.join(out, 'crops.jsonl')) as fh:
        got = {json.loads(line)['image_id'] for line in fh}
    bad = []
    if rcd.dhash(twin_p) == held['dhash'] and twin['sha256'] == held['sha256']:
        bad.append('D7 (test) the re-encoded twin is byte-identical')
    if '976000000002' in got:
        bad.append('D7 a re-encoded copy of a reserved crop, under another '
                   'image_id and sequence, is in the dataset')
    if '976000000003' in got:
        bad.append('D7 a near-copy of a crop the reserve FILE records is in '
                   'the dataset')
    if '976000000005' not in got:
        bad.append('D7 an unrelated crop was dropped as a reserve twin')
    return bad


def check_noseq_dropped():
    bad = []
    rows = [row('gate_audit', 'dog', '977000000001', 'd0', 'N1'),
            row('gate_audit', 'not_dog', '977000000002', 'd0', 'N2'),
            row('gate_audit', 'dog', '977000000003', 'd0',
                'noseq:977000000003', seq_status='noseq')]
    # enough sequence-less crops of both classes that, unpinned, some of
    # them would be drawn into val
    for k in range(4, 16):
        rows.append(row('gate_audit', 'dog' if k % 2 else 'not_dog',
                        '9770000000%02d' % k, 'd0', 'noseq:9770000000%02d' % k,
                        seq_status='noseq'))
    m = write_manifest(rows, 'm_noseq2.jsonl')
    out = os.path.join(TMP, 'out_noseq_a')
    err, _ = run_rebuild(['--manifest', m, '--out', out, '--no-reserve',
                          '--hamming', '0', '--max-noseq-frac', '0.95',
                          '--execute'])
    if err:
        return ['D1 the noseq build refused: %s' % (err,)]
    got = {json.loads(line)['image_id']: json.loads(line)['split']
           for line in open(os.path.join(out, 'crops.jsonl'))}
    if '977000000003' in got:
        bad.append('D1 a crop with no resolved sequence is in the dataset '
                   'without --keep-noseq')
    out = os.path.join(TMP, 'out_noseq_b')
    err, _ = run_rebuild(['--manifest', m, '--out', out, '--no-reserve',
                          '--hamming', '0', '--max-noseq-frac', '0.95',
                          '--keep-noseq', '--execute'])
    if err:
        return bad + ['D1 the --keep-noseq build refused: %s' % (err,)]
    got = {json.loads(line)['image_id']: json.loads(line)['split']
           for line in open(os.path.join(out, 'crops.jsonl'))}
    off = sorted(i for i, sp in got.items()
                 if i not in ('977000000001', '977000000002') and sp != 'train')
    if '977000000003' not in got or off:
        bad.append('D1 --keep-noseq did not keep every sequence-less crop '
                   'pinned to train: %r' % (off,))
    return bad


def check_reserve_twins(hash_files=None, near=None):
    """The test-set draw keeps out twins of what the deployed model saw."""
    from PIL import Image
    saved = (rts.hash_files, rts._near)
    if hash_files:
        rts.hash_files = hash_files
    if near:
        rts._near = near
    try:
        crop_dir = os.path.join(TMP, 'rts_twin_full')
        vs = []
        for k in range(12):
            key = '9780000000%02d#0' % k
            vs.append({'key': key, '_label': 'dog', 'band': 0, 'p_dog': .5,
                       'seq': 'W%02d' % k, 'ts': k})
            _img(os.path.join(crop_dir, key.replace('#', '_') + '.jpg'),
                 seed=4000 + k)
        p = lambda k: os.path.join(crop_dir, '9780000000%02d_0.jpg' % k)
        shutil.copy2(p(0), p(1))                    # byte twin of a compared
        with Image.open(p(2)) as im:                # re-encoded twin of one
            im.save(p(3), 'JPEG', quality=75)
        shutil.copy2(p(5), p(6))                    # two eligible twins
        comp = os.path.join(TMP, 'rts_twin_compare', 'train', 'dog')
        os.makedirs(comp, exist_ok=True)
        shutil.copy2(p(7), os.path.join(comp, 'pos_ls_978999999999_0.jpg'))
        catalog = {v['key'][:12]: v['seq'] for v in vs}
        refs = list(rts.hash_files([os.path.join(comp, f)
                                    for f in os.listdir(comp)]).values())
        cands, why = rts.eligible(
            'dogbin', vs, set(), crop_dir, set(),
            {'978000000000', '978000000002'}, set(), set(), set(), catalog,
            ('dog', 'not_dog'), ref_hashes=refs)
    finally:
        rts.hash_files, rts._near = saved
    keys = {c['key'] for c in cands}
    bad = []
    for k, what in (('978000000001#0', 'a byte-identical copy of a compared '
                     'image\'s crop'),
                    ('978000000003#0', 'a re-encoded copy of a compared '
                     'image\'s crop'),
                    ('978000000007#0', 'a copy of a compared dataset FILE')):
        if k in keys:
            bad.append('D7 %s is eligible for the test set' % (what,))
    if '978000000004#0' not in keys:
        bad.append('D7 an unrelated crop was ruled out as a twin')
    rts.twin_units(cands)
    u = {c['key']: c['unit'] for c in cands}
    if '978000000005#0' in u and u.get('978000000005#0') != \
            u.get('978000000006#0'):
        bad.append('D7 two eligible twins in different sequences are not one '
                   'draw unit')
    st = rts.draw(cands, 0.5, 5)
    drawn = sum(len(el) for el, _ in st.values())
    if drawn != len(set(u.values())):
        bad.append('D7 the draw is over %d units, not the %d twin-joined '
                   'ones -- twins can be split between test and train'
                   % (drawn, len(set(u.values()))))
    return bad


def check_family_reserves_add():
    import build_dataset as bd
    data = os.path.join(TMP, 'fr_data')
    os.makedirs(data, exist_ok=True)
    for n in ('leash_acceptance_set.json', 'leash_test_v1.json', 'x.json',
              'other_test.json'):
        with open(os.path.join(data, n), 'w') as fh:
            json.dump({'image_ids': ['1']}, fh)
    saved = bd.D
    bd.D = lambda *p: os.path.join(data, *p)
    try:
        a = [os.path.basename(x) for x in bd.family_reserves(
            'leash', [os.path.join(data, 'x.json')])]
        b = [os.path.basename(x) for x in bd.family_reserves(
            'leash', None, test_set=os.path.join(data, 'other_test.json'))]
    finally:
        bd.D = saved
    bad = []
    if 'leash_acceptance_set.json' not in a or 'leash_test_v1.json' not in a:
        bad.append('D2 --reserve REPLACED the acceptance and test sets: %r'
                   % (a,))
    if b != ['leash_acceptance_set.json', 'other_test.json']:
        bad.append('D2 --test-set dropped the acceptance set or kept the '
                   'default test set: %r' % (b,))
    return bad


def check_no_ls_refused():
    import build_dataset as bd
    tmp = os.path.join(TMP, 'bd_root2')
    data = os.path.join(TMP, 'bd_data2')
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(data, exist_ok=True)
    for n in ('leash_acceptance_set.json', 'leash_test_v1.json'):
        with open(os.path.join(data, n), 'w') as fh:
            json.dump({'image_ids': ['1']}, fh)
    saved = (bd.D, os.environ.get('TRAINING_ROOT'))
    bd.D = lambda *p: os.path.join(data, *p)
    os.environ['TRAINING_ROOT'] = tmp
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            bd.build('leash', out=os.path.join(TMP, 'bd_out_nols'),
                     no_export=True, no_embed_dedup=True,
                     detect_root=os.path.join(TMP, 'nostore'))
        return ['D3 a leash build with --no-export and no export in hand '
                'did not refuse']
    except SystemExit as e:
        if '--allow-no-ls' not in str(e):
            return ['D3 the no-export refusal does not name --allow-no-ls: '
                    '%s' % (e,)]
        return []
    finally:
        bd.D = saved[0]
        if saved[1] is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = saved[1]


# ── the optional real-data smoke ────────────────────────────────────────────

def smoke_reserve_counts():
    want = {'dogbin': 9263, 'leash': 2095}
    if not os.path.isfile(os.path.join(REPO, 'data', 'fn_audit',
                                       'verdicts.jsonl')):
        print('  (real-data smoke skipped: no audit ledger here)')
        return []
    bad = []
    for fam, n in want.items():
        p = subprocess.run([sys.executable,
                            os.path.join(DETECT, 'reserve_test_set.py'),
                            '--family', fam, '--dry-run'],
                           capture_output=True, text=True, timeout=600)
        m = re.search(r'-> eligible (\d+)', p.stdout or '')
        if p.returncode != 0 or not m:
            print('  (real-data smoke for %s skipped: %s)'
                  % (fam, (p.stderr or p.stdout).strip()[-200:]))
            continue
        got = int(m.group(1))
        if abs(got - n) > 0.05 * n:
            bad.append('SMOKE %s eligible %d is not within 5%% of the '
                       'measured %d' % (fam, got, n))
    return bad


# ── mutants: the old behaviour must make the checks fail ────────────────────

def _old_parse(name):
    m = OLD_NAME_RE.match(os.path.basename(name))
    return (m.group(1), None) if m else None


def _cap_per_crop(rows, cap, rng):
    # the nofile: singleton world -- every crop its own (sequence, class)
    return list(rows), collections.Counter()


def _ids_only(reserves):
    ids, _ = _REAL_RESERVED(reserves)
    return ids, set()


_REAL_RESERVED = rcd.reserved_sets


def _patched(obj, name, value, fn):
    """Run fn() with obj.name replaced by value, then put it back."""
    real = getattr(obj, name)
    setattr(obj, name, value)
    try:
        return fn()
    finally:
        setattr(obj, name, real)


def _old_read_corrections(path):
    # the reader before it skipped the audit pages' records
    out = {}
    for ln in open(path):
        r = json.loads(ln)
        out[(str(r['image_id']), int(r.get('det_idx') or 0))] = (
            float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']))
    return out


def _identity_veto(rows, *a, **k):
    return rows, []

MUTANTS = [
    ('D1 old NAME_RE', lambda: check_parse(_old_parse)),
    ('D1 nofile: per-crop sequences',
     lambda: check_leak(ls_seq=lambda k: 'nofile:%d' % k)),
    ('D1 cap keyed per crop', lambda: check_cap(_cap_per_crop)),
    ('D2 image_id-only exclusion',
     lambda: check_reserve_by_sequence(_ids_only)),
    ('D2 leash build given the dogbin reserve',
     lambda: check_leash_argv_reserves(('dogbin_acceptance_set.json',))),
    ('D4 a copied 0.15 margin',
     lambda: check_no_copied_constants([('m.py', 'pad = 0.15 * w')])),
    ('D7 old parser on the compare dataset',
     lambda: check_reserve_tool(_old_dataset_ids)),
    ('D4 the audit framing check a no-op',
     lambda: _patched(cs, 'framing_check_audit',
                      lambda rows, **k: {'checked': 0},
                      check_collect_framing_check)),
    ('D6 the other family\'s verdicts ignored',
     lambda: _patched(cs, 'apply_box_vetoes', _identity_veto,
                      check_collect_exclusions_and_vetoes)),
    ('D6 frames declared empty ignored',
     lambda: _patched(cs, 'apply_empty_frames', _identity_veto,
                      check_collect_exclusions_and_vetoes)),
    ('D6 the Label Studio "other animal" veto off',
     lambda: _patched(cs, 'apply_vetoes', _identity_veto,
                      check_collect_exclusions_and_vetoes)),
    ('D6 a drawn box joined against a redraw',
     lambda: check_dedup_joins_detector_box(strip_pool=True)),
    ('D7 reserved crops held out by name only',
     lambda: check_reserved_near_copies(near=lambda *a: set())),
    ('D7 test-set twins of the compared dataset not looked for',
     lambda: check_reserve_twins(
         hash_files=lambda paths, workers=8: {p: ('sha:' + p, None)
                                              for p in paths},
         near=lambda *a: False)),
    ('D3 the audit pages\' redraws read as review-page ones',
     lambda: _patched(__import__('harvest_flagged'), 'read_corrections',
                      _old_read_corrections,
                      check_review_corrections_skip_audit)),
    ('D3 a leash call\'s redraw not kept as metadata',
     lambda: _patched(__import__('harvest_flagged'), 'read_corrections',
                      lambda path: {}, check_leash_calls)),
    ('D3 leash calls cut from the review-page redraw (the old rule)',
     lambda: _patched(cs, 'leash_call_candidates', _mutated(
         cs.leash_call_candidates, 'box = list(det_box)',
         'box = [float(v) for v in fixed] if fixed else list(det_box)'),
         check_leash_calls)),
    ('D3 leash audit: the redraw wins (the old rule)',
     lambda: _patched(cs, 'EDITED_WINS', {'gate': True, 'leash': True},
                      check_audit_coverage)),
    ('D3 gate audit: the detector box wins',
     lambda: _patched(cs, 'EDITED_WINS', {'gate': False, 'leash': False},
                      check_audit_coverage)),
    ('D6 the leash dedup keeps the redraw (redraw-wins for every family)',
     lambda: check_dedup_rank_by_family(_old_rank)),
    ('D6 differing det keys joined by key only',
     lambda: check_dedup_overlapping_det_keys(_keyonly_dedup)),
    ('D8 the leash framing assertion off',
     lambda: check_leash_framing_assertion(_mutated(
         rcd.check_rows, "lf = [r for r in rows if family_of(r) == 'leash']",
         'lf = []'))),
    ('D9 no creator cap',
     lambda: check_creator_cap(lambda rows, cap, seed: (
         list(rows), collections.Counter(), collections.Counter()))),
    ('D9 creator cap in plain hash order (not sequences first)',
     lambda: _patched(rcd, '_spread_order', _hash_only_spread,
                      check_creator_cap)),
    ('D9 no band balance',
     lambda: check_band_balance(lambda rows, pos, seed: (list(rows), {}))),
    ('D9 bands taken on the long side',
     lambda: check_band_balance(_long_side_balance)),
    ('D9 bands balanced to 0.5, not the split\'s rate',
     lambda: check_band_balance(_mutated(
         rcd.balance_bands,
         "'target_pos_rate': (n_pos / len(banded)) if banded else None",
         "'target_pos_rate': 0.5 if banded else None"))),
    ('D9 the rebalance not run by the build',
     lambda: check_rebalanced_build(_no_rebalance)),
    ('D6 leash: a Label Studio crop outranks the detector box (phase-2 rank)',
     lambda: check_dedup_rank_by_family(_phase2_rank)),
    ('D6 leash: the dHash / embedding collapse keeps the largest file',
     lambda: _patched(rcd, '_collapse_best', _size_only_collapse,
                      check_dedup_rank_by_family)),
    ('D9 a creator with no captured_at reads as no creator',
     lambda: _patched(rcd, '_capture_of', _mutated(
         rcd._capture_of, 'if not creator:',
         'if not creator or ts is None:'), check_creator_cap)),
    ('D9 rows with no creator capped as one creator',
     lambda: check_creator_cap(_mutated(
         rcd.cap_creators, 'cr = creator_of(r)',
         "cr = creator_of(r) or 'NONE'"))),
    ('D9 the creator cap without the image tier',
     lambda: _patched(rcd, '_spread_order', _sequence_only_spread,
                      check_creator_cap)),
    ('D9 bands balanced to the rate of all splits together',
     lambda: check_band_balance(_mutated(
         rcd.balance_bands,
         "'target_pos_rate': (n_pos / len(banded)) if banded else None",
         "'target_pos_rate': (sum(1 for x in rows if _band_of(x) and "
         "x['label'] == pos) / float(sum(1 for x in rows if _band_of(x)))) "
         "if banded else None"))),
    ('D9 rows with no crop_wh dropped by the band balance',
     lambda: check_band_balance(_mutated(
         rcd.balance_bands,
         'kept += [r for r in lst if not _band_of(r)]', 'pass'))),
    ('D9 a creator-cap-only build recorded as plain',
     lambda: check_rebalanced_build(_mutated(
         rcd.rebalance_plan, 'if max_per_creator or balance_size_bands:',
         'if balance_size_bands:'))),
    ('D9 a plain build claims assertions 12 and 13 held',
     lambda: _patched(rcd, 'rebalance_not_requested', lambda rec: [],
                      check_rebalanced_build)),
]


def _long_side_balance(rows, pos, seed):
    """balance_bands with each crop banded by its LONG side."""
    real = rcd._band_of

    def long_band(r):
        wh = r.get('crop_wh')
        return rcd.size_band(max(wh)) if wh else None
    rcd._band_of = long_band
    try:
        return rcd.balance_bands(rows, pos, seed)
    finally:
        rcd._band_of = real


def main():
    checks = [check_parse, check_parse_fallback_agrees,
              check_legacy_unparseable_fails, check_cap, check_leak,
              check_reserve_by_sequence, check_no_reserve_refused,
              check_leash_argv_reserves, check_audit_coverage,
              check_framing_matches_production, check_constant_is_imported,
              check_no_copied_constants, check_resized_ls_frame_refused,
              check_missing_boto3_fatal, check_small_boxes_cut, check_dedup,
              check_reserve_tool, check_assertions,
              check_manifest_end_to_end, check_collect_clean,
              check_collect_missing_frame, check_collect_unreadable_frame,
              check_collect_frame_size_vs_store,
              check_collect_ls_size_mismatch, check_collect_seq_disagreement,
              check_collect_framing_check, check_collect_empty_source,
              check_collect_exclusions_and_vetoes, check_exclusion_records,
              check_leash_calls, check_review_corrections_skip_audit,
              check_frame_labels, check_dedup_joins_detector_box,
              check_dedup_rank_by_family, check_dedup_overlapping_det_keys,
              check_leash_framing_assertion, check_creator_cap,
              check_band_balance, check_rebalanced_build,
              check_reserved_near_copies, check_noseq_dropped,
              check_reserve_twins, check_family_reserves_add,
              check_no_ls_refused, smoke_reserve_counts]
    bad = []
    try:
        for fn in checks:
            try:
                bad += fn()
            except Exception as e:        # noqa: BLE001 - report, not die
                bad.append('%s threw %s: %s' % (fn.__name__,
                                                type(e).__name__, e))
        for name, fn in MUTANTS:
            try:
                got = fn()
            except Exception as e:        # noqa: BLE001
                got = ['threw %s' % (type(e).__name__,)]
            if not got:
                bad.append('MUTANT %s: the check still passes with the old '
                           'behaviour put back -- it proves nothing' % (name,))
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    for b in bad:
        print('FAIL ' + b)
    if bad:
        return 1
    print('crop datasets: one parser, production framing, human labels only, '
          'reserves held by sequence, duplicates and conflicts resolved '
          'across sources, and every assertion can fail (%d checks, %d '
          'mutants)' % (len(checks), len(MUTANTS)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
