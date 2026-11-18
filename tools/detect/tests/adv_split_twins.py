#!/usr/bin/env python3
"""Capture twins and a split that one row cannot re-deal.

Two defects of the crop-dataset build, both measured on leash_20261115_18b269
(2026-11-15) before they were fixed:

  T1  THE SAME MOMENT IN TWO SEQUENCES. The same dog, shot by the same
      creator at the same moment, can sit in two Mapillary sequences (a
      multi-camera rig, a split upload). Nothing joined them: 4 val crops had
      a twin in train at 0 s / 0-6 m, and 10 of the 312 leash acceptance
      images had one in train. Crop dHash (8+ bits apart) and the embedding
      (cos 0.88 against a 0.93 join) both missed them, because one twin is a
      Label Studio box that takes in the leash and the other the detector's.
      Now rebuild_crop_dataset.capture_twins -- same creator, <= 60 s and
      <= 30 m, or <= 10 s when a geometry is missing -- joins their sequences
      into one split group, drops a twin of a reserved image, and asserts
      both (assertion 10). reserve_test_set.twin_units calls the same helper
      for future draws, and capture_draw_units makes a capture twin of a
      compared or acceptance image ineligible. A build with no capture data
      (every build before this) cannot be checked: --verify-only said "all 11
      hold" on leash_20261115_18b269 with those 4 val twins in it, so it now
      reports assertion 10 as skipped. The wiring (manifest_main into
      split_groups, reserve_test_set.main into both helpers) is tested end
      to end, not just the helpers.

  T2  ONE ROW RE-DEALT 700. cap_rows and assign_splits drew from one shared
      random.Random, so one verdict more or less moved 626-705 of 3,743 rows
      across train/val. Now both are ordered by sha256(seed:item), and 20
      random single-row removals from the 3,735 crops of the audit's
      reproduced leash stage (cap 5, seed 0) moved [0 x7, 1 x13] rows.

Synthetic rows and frames in a temp directory; the catalog lookups are stood
in for. Each check is a function returning a list of failures; the MUTANTS
section puts each old behaviour back and requires its check to FAIL.

Run: python tools/detect/tests/adv_split_twins.py   (mp14 env)
"""
import collections
import contextlib
import io
import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))

import rebuild_crop_dataset as rcd            # noqa: E402
import reserve_test_set as rts                # noqa: E402

TMP = tempfile.mkdtemp(prefix='adv_split_twins_')
T0 = 1_700_000_040.0           # a whole minute: bucket edges are testable
LON, LAT = 85.3240, 27.7172    # Kathmandu; 1e-5 deg of latitude ~ 1.11 m


def cap(creator='c1', dt=0.0, north_m=0.0, geo=True):
    """A capture `dt` seconds after T0 and `north_m` metres north."""
    return {'creator': creator, 'ts': T0 + dt,
            'lon': LON if geo else None,
            'lat': (LAT + north_m / 111195.0) if geo else None}


_SEED = [0]


def _img(path):
    from PIL import Image
    import numpy as np
    _SEED[0] += 1
    a = np.random.default_rng(_SEED[0]).integers(0, 255, (40, 48, 3),
                                                 dtype=np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(a).save(path, 'JPEG', quality=95)
    return path


def row(iid, seq, label='dog', capture=None, with_file=False):
    uid = 'ga_%s_d0.jpg' % (iid,)
    r = {'uid': uid, 'image_id': str(iid), 'det_key': 'd0',
         'sequence': seq, 'seq_status': 'manifest', 'label': label,
         'source': 'gate_audit', 'framing': 'audit_full', 'family': 'dogbin',
         'box': [10, 10, 50, 50], 'pool_box': [10, 10, 50, 50],
         'aimed': False, 'band': None, 'p_model': None, 'frame': None,
         'provenance': {'ledger': os.path.join(REPO, 'data', 'fn_audit',
                                               'verdicts.jsonl'),
                        'key_or_task_id': iid, 'by': 't', 'ts': 0}}
    if capture is not None:
        r['capture'] = capture
    if with_file:
        p = _img(os.path.join(TMP, 'crops', label, uid))
        r['path'] = p
        r['sha256'] = rcd.sha256_file(p)
        r['dhash'] = rcd.dhash(p)
        r['crop_wh'] = [48, 40]
    return r


def _filler(n, start=0, rng=None):
    """n unrelated single-image sequences, a creator each, both labels."""
    rng = rng or random.Random(3)
    return [row('9%011d' % (start + k), 'F%04d' % (start + k),
                rng.choice(('dog', 'not_dog')),
                cap('f%d' % (start + k), dt=1000.0 * k))
            for k in range(n)]


# ── T1 the rule ─────────────────────────────────────────────────────────────

def check_twins_joined():
    """Same creator 5 s / 3 m apart, other sequence: one group, one split."""
    rows = _filler(40) + [row('100000000001', 'SA', 'dog', cap('c1', 0, 0)),
                          row('100000000002', 'SB', 'dog', cap('c1', 5, 3))]
    g = rcd.split_groups(rows, [])
    gid = {r['image_id']: g[r['uid']] for r in rows}
    bad = []
    if gid['100000000001'] != gid['100000000002']:
        bad.append('T1 two sequences by one creator 5 s / 3 m apart are two '
                   'split groups')
    sp, _ = rcd.assign_splits(rows, g, 0.5, 0)
    if sp[gid['100000000001']] != sp[gid['100000000002']]:
        bad.append('T1 capture twins landed in two splits')
    return bad


def check_not_twins():
    """200 s apart, another creator, 40 m apart: never joined."""
    bad = []
    for what, other in (('200 s apart', cap('c1', 200, 0)),
                        ('another creator at the same moment',
                         cap('c2', 0, 0)),
                        ('40 m apart', cap('c1', 5, 40)),
                        ('61 s apart', cap('c1', 61, 0))):
        rows = [row('100000000011', 'SA', 'dog', cap('c1', 0, 0)),
                row('100000000012', 'SB', 'dog', other)]
        g = rcd.split_groups(rows, [])
        if len(set(g.values())) != 2:
            bad.append('T1 %s were joined as twins' % (what,))
    return bad


def check_missing_geometry():
    """No geometry: joined within 10 s on time alone, not at 30 s."""
    bad = []
    for dt, want in ((8, 1), (30, 2)):
        rows = [row('100000000021', 'SA', 'dog', cap('c1', 0, geo=False)),
                row('100000000022', 'SB', 'dog', cap('c1', dt, 0))]
        n = len(set(rcd.split_groups(rows, []).values()))
        if n != want:
            bad.append('T1 a capture with no geometry %d s from its creator\'s '
                       'other one gave %d group(s), want %d' % (dt, n, want))
    return bad


def check_bucket_edge():
    """Twins 2 s apart across a bucket edge (T0+59 and T0+61) are joined;
    and 20,000 captures by one creator do not take an all-pairs pass."""
    import time
    bad = []
    pairs = rcd.capture_twins([('a', cap('c1', 59)), ('b', cap('c1', 61))])
    if pairs != [('a', 'b')]:
        bad.append('T1 twins 2 s apart across a minute boundary were missed '
                   '(got %r)' % (pairs,))
    items = [('n%05d' % k, cap('c1', 120.0 * k)) for k in range(20000)]
    t = time.time()
    got = rcd.capture_twins(items)
    if got:
        bad.append('T1 captures 120 s apart were joined')
    if time.time() - t > 5:
        bad.append('T1 capture_twins took %.1fs on 20,000 captures -- '
                   'all-pairs, not bucketed' % (time.time() - t,))
    return bad


def check_assertion():
    """check_rows fails a plan where twins straddle, or a row twins a
    reserved image; passes the same plan with the twins on one side."""
    bad = []
    a = row('100000000031', 'SA', 'dog', cap('c1', 0, 0))
    b = row('100000000032', 'SB', 'dog', cap('c1', 4, 2))
    plan = [dict(a, split='train', group='SA'), dict(b, split='val',
                                                     group='SB')]
    if not any('capture-twin' in f for f in rcd.check_rows(plan)):
        bad.append('T1 check_rows passed capture twins in train and val')
    plan[1]['split'] = 'train'
    if rcd.check_rows(plan):
        bad.append('T1 check_rows failed a clean plan: %s'
                   % (rcd.check_rows(plan),))
    res = [{'image_ids': ['100000000039'], 'sequences': ['SR'],
            'captures': {'100000000039': cap('c1', 2, 1)}}]
    if not any('reserved image' in f for f in rcd.check_rows(plan, res)):
        bad.append('T1 check_rows passed a row that is a capture twin of a '
                   'reserved image')
    old = [dict(r) for r in plan]
    for r in old:
        r.pop('capture')
    if rcd.check_rows(old, res):
        bad.append('T1 a build from before captures (rows with none) fails '
                   'the twin assertion it could not have known about')
    if not rcd.twin_check_skipped(old, res):
        bad.append('T1 rows with no capture data are not reported as an '
                   'unchecked twin assertion')
    if not rcd.twin_check_skipped(plan, [{'image_ids': ['100000000039']}]):
        bad.append('T1 reserves with no captures are not reported as an '
                   'unchecked twin assertion')
    if rcd.twin_check_skipped(plan, res):
        bad.append('T1 a fully checkable plan is reported as skipped')
    return bad


def _stub_capture(caps):
    def fake(ids, repo, py):
        return {str(i): caps[str(i)] for i in ids if str(i) in caps}
    return fake


def check_reserve_twin_dropped():
    """End to end: a capture twin of a reserved image never reaches a split,
    is counted in the manifest, and the on-disk check holds."""
    rows = []
    rng = random.Random(5)
    for k in range(30):
        rows.append(row('2%011d' % k, 'Q%03d' % k,
                        rng.choice(('dog', 'not_dog')),
                        with_file=True))
    twin = row('300000000001', 'ST', 'dog', with_file=True)
    mate = row('300000000002', 'ST', 'dog', with_file=True)  # 10 min later
    # a twin pair INSIDE the dataset, in two sequences: manifest_main must
    # hand its union to split_groups
    pa = row('300000000011', 'SX', 'dog', with_file=True)
    pb = row('300000000012', 'SY', 'not_dog', with_file=True)
    rows += [twin, mate, pa, pb]
    caps = {r['image_id']: cap('f' + r['image_id'], 0) for r in rows}
    caps['300000000001'] = cap('c9', 5, 3)
    caps['300000000002'] = cap('c9', 600, 3)
    caps['300000000011'] = cap('c8', 0, 0)
    caps['300000000012'] = cap('c8', 4, 2)
    caps['300000000099'] = cap('c9', 0, 0)       # the reserved image
    man = os.path.join(TMP, 'manifest.jsonl')
    with open(man, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    res = os.path.join(TMP, 'reserve.json')
    with open(res, 'w') as fh:
        json.dump({'image_ids': ['300000000099'], 'sequences': ['SR']}, fh)
    out = os.path.join(TMP, 'out_reserve')
    shutil.rmtree(out, ignore_errors=True)
    saved = rcd.resolve_capture
    rcd.resolve_capture = _stub_capture(caps)
    buf = io.StringIO()
    err = None
    try:
        with contextlib.redirect_stdout(buf):
            rcd.main(['--manifest', man, '--out', out, '--reserve', res,
                      '--val-frac', '0.4', '--execute'])
    except SystemExit as e:
        err = str(e)
    finally:
        rcd.resolve_capture = saved
    if err:
        return ['T1 the build refused (%s): %s' % (err, buf.getvalue()[-400:])]
    bad = []
    with open(os.path.join(out, 'crops.jsonl')) as fh:
        on = {r['uid']: r for r in (json.loads(line) for line in fh
                                    if line.strip())}
    got = set(on)
    a, b = on.get(pa['uid']), on.get(pb['uid'])
    if not a or not b:
        bad.append('T1 a twin pair inside the dataset was dropped')
    elif a['group'] != b['group'] or a['split'] != b['split']:
        bad.append('T1 the build did not join a twin pair in two sequences '
                   '(groups %s/%s, splits %s/%s) -- manifest_main does not '
                   'pass the twin unions to split_groups'
                   % (a['group'], b['group'], a['split'], b['split']))
    if twin['uid'] in got:
        bad.append('T1 a capture twin of a reserved image was trained on')
    if mate['uid'] not in got:
        bad.append('T1 the twin\'s sequence-mate 10 min later was dropped '
                   'too -- the rule is per image, not per sequence')
    with open(os.path.join(out, 'rebuild_manifest.json')) as fh:
        ct = json.load(fh).get('capture_twins') or {}
    if ct.get('reserve_twin_rows_dropped') != 1 or \
            ct.get('reserve_twin_uids') != [twin['uid']]:
        bad.append('T1 the manifest does not record the dropped reserve '
                   'twin: %r' % ({k: ct.get(k) for k in (
                       'reserve_twin_rows_dropped', 'reserve_twin_uids')},))
    if ct.get('twin_sequence_unions') != 1:
        bad.append('T1 the manifest records %r twin sequence unions, want 1'
                   % (ct.get('twin_sequence_unions'),))
    for k in ('twin_sequence_unions', 'largest_group_rows',
              'largest_group_sequences', 'images_without_capture'):
        if k not in ct:
            bad.append('T1 the manifest does not record %s' % (k,))
    skipped = []
    fails = rcd.verify_dataset(out, skipped=skipped)
    if fails:
        bad.append('T1 the on-disk check fails a clean build: %s' % (fails,))
    if skipped:
        bad.append('T1 a build with captures reports the twin assertion as '
                   'skipped: %s' % (skipped,))
    # and the on-disk check SEES a twin of a reserved image put back by hand
    with open(os.path.join(out, 'crops.jsonl')) as fh:
        lines = [json.loads(x) for x in fh if x.strip()]
    lines[0]['capture'] = caps['300000000001']
    with open(os.path.join(out, 'crops.jsonl'), 'w') as fh:
        for x in lines:
            fh.write(json.dumps(x) + '\n')
    if not any('reserved image' in f for f in rcd.verify_dataset(out)):
        bad.append('T1 the on-disk check passed a twin of a reserved image')
    bad += _check_old_build_skip(out, lines)
    return bad


def _check_old_build_skip(out, lines):
    """A build from before captures (no `capture` on any row, no `captures`
    on any reserve) passes assertion 10 vacuously; --verify-only must say it
    was SKIPPED and count it out, not print "all 11 hold"."""
    with open(os.path.join(out, 'crops.jsonl'), 'w') as fh:
        for x in lines:
            fh.write(json.dumps({k: v for k, v in x.items()
                                 if k != 'capture'}) + '\n')
    rp = os.path.join(out, 'reserves.json')
    with open(rp) as fh:
        res = json.load(fh)
    with open(rp, 'w') as fh:
        json.dump([{k: v for k, v in r.items()
                    if k not in ('captures', 'captures_missing')}
                   for r in res], fh)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rcd.main(['--verify-only', out])
    except SystemExit as e:
        return ['T1 --verify-only refused an old-style build: %s %s'
                % (e, buf.getvalue()[-400:])]
    text = buf.getvalue()
    # a plain build: 12 and 13 (the opt-in rebalance) are not requested
    want = ('all %d assertions hold (1 not checkable, 2 not requested)'
            % (rcd.ASSERTIONS - 3,))
    bad = []
    if want not in text or text.count('SKIP assertion 10') != 1:
        bad.append('T1 --verify-only on a build with no capture data claims '
                   'every assertion held: %r' % (text.strip()[-300:],))
    return bad


def check_test_draw_units():
    """reserve_test_set.twin_units: capture twins share a draw unit when
    captures are given (future draws), and not when they are not (the
    frozen v1 draws, which had no captures)."""
    cands = [{'key': '400000000001#0', 'image_id': '400000000001',
              'seq': 'UA', 'sha256': 'a', 'dhash': 0},
             {'key': '400000000002#0', 'image_id': '400000000002',
              'seq': 'UB', 'sha256': 'b', 'dhash': (1 << 64) - 1},
             {'key': '400000000003#0', 'image_id': '400000000003',
              'seq': 'UC', 'sha256': 'c', 'dhash': 0x5555555555555555}]
    caps = {'400000000001': cap('c1', 0, 0), '400000000002': cap('c1', 3, 2),
            '400000000003': cap('c2', 0, 0)}
    bad = []
    u = {c['key']: c['unit'] for c in rts.twin_units(
        [dict(c) for c in cands], captures=caps)}
    if u['400000000001#0'] != u['400000000002#0']:
        bad.append('T1 a future test draw splits capture twins between test '
                   'and train')
    if u['400000000003#0'] == u['400000000001#0']:
        bad.append('T1 another creator\'s capture joined a draw unit')
    u0 = {c['key']: c['unit'] for c in rts.twin_units(
        [dict(c) for c in cands])}
    if len(set(u0.values())) != 3:
        bad.append('T1 twin_units without captures changed: the frozen '
                   'draws would no longer reproduce')
    return bad


def check_test_draw_main():
    """reserve_test_set.main --dry-run, ledgers and catalog stood in for:
    a capture twin of a COMPARED image is not eligible, and two candidates
    that are capture twins of each other are one draw unit -- through main,
    not just through twin_units."""
    import types
    import crop_sources
    import build_dataset
    acc_p = os.path.join(TMP, 'acc.json')
    with open(acc_p, 'w') as fh:
        json.dump({'image_ids': ['700000000098']}, fh)

    def cand(i, seq):
        return {'key': '%s#0' % i, 'image_id': i, 'label': 'leashed',
                'band': 0, 'p_model': 0.5, 'seq': seq, 'ts': 0,
                'sha256': i, 'dhash': int(i) * 0x9E3779B97F4A7C15 % (1 << 64)}
    cands = [cand('700000000001', 'VA'), cand('700000000002', 'VB'),
             cand('700000000003', 'VC')]
    cands += [cand('70000000%04d' % k, 'W%03d' % k) for k in range(10, 20)]
    caps = {c['image_id']: cap('w' + c['image_id'], 0) for c in cands}
    caps['700000000001'] = cap('c5', 0, 0)
    caps['700000000002'] = cap('c5', 3, 2)       # twin of 001: one unit
    caps['700000000003'] = cap('c6', 0, 0)
    caps['700000000099'] = cap('c6', 4, 1)       # compared image: twin of 003
    fa = types.ModuleType('fn_audit')
    fa.paths = lambda stage, *, gen: {'full': TMP, 'verdicts': acc_p}
    fa.spec = lambda stage: {'answers': ['leashed', 'unleashed'],
                             'model': 'm'}
    fa.GenError = type('GenError', (Exception,), {})
    fa.ctx = lambda stage, gen: types.SimpleNamespace(
        stage=stage, gen=gen, pool_id='g1-legacy', P=fa.paths(stage, gen=gen),
        meta={'model': 'm', 'detect_root': None}, frozen=True)
    fa.read_verdicts = lambda stage=None, *, gen: []
    fa.verdict_of = lambda v, stage: v
    fa.targeted = lambda stage, *, gen: set()
    patches = [
        (sys.modules, 'fn_audit', fa),
        (crop_sources, 'exclusions', lambda fam, root: (set(), {})),
        (build_dataset, 'training_root', lambda: TMP),
        (rts, 'dataset_ids', lambda c: ({'700000000099'}, 1)),
        (rts, 'dataset_files', lambda c: []),
        (rts, 'hash_files', lambda paths, *a, **k: {}),
        (rts, 'eligible', lambda *a, **k: (
            [dict(c) for c in cands],
            collections.Counter(eligible=len(cands)))),
        (rcd, 'resolve_sequences', lambda ids, repo, py: {}),
        (rcd, 'resolve_capture', _stub_capture(caps)),
    ]
    old = []
    for obj, name, val in patches:
        if obj is sys.modules:
            old.append((obj, name, sys.modules.get(name)))
            sys.modules[name] = val
        else:
            old.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = rts.main(['--family', 'leash', '--gen', '1', '--dry-run',
                           '--compare', os.path.join(TMP, 'cmp'),
                           '--acceptance', acc_p])
    except SystemExit as e:
        rc = 'exit %s' % (e,)
    finally:
        for obj, name, val in reversed(old):
            if obj is sys.modules:
                if val is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = val
            else:
                setattr(obj, name, val)
    text = buf.getvalue()
    if rc != 0:
        return ['T1 reserve_test_set --dry-run failed (%s): %s'
                % (rc, text[-400:])]
    bad = []
    if 'capture_twin_of_compare_or_acceptance' not in text or \
            'eligible %d' % (len(cands) - 1,) not in text:
        bad.append('T1 a capture twin of a compared image stayed eligible '
                   'for the test draw')
    # 12 candidates left; 001 and 002 are one unit -> 11
    if 'in 11 draw units' not in text:
        bad.append('T1 main draws capture twins as two units: %r'
                   % ([x for x in text.splitlines()
                       if x.startswith('twins:')],))
    return bad


# ── T2 one row moves one group ──────────────────────────────────────────────

def _plan(rows, cap_fn=None, split_fn=None, seed=0):
    cap_fn = cap_fn or rcd.cap_rows
    split_fn = split_fn or rcd.assign_splits
    kept, _ = cap_fn(rows, 2, seed)
    g = rcd.split_groups(kept, [])
    sp, _ = split_fn(kept, g, 0.2, seed)
    return {r['uid']: (sp[g[r['uid']]], g[r['uid']]) for r in kept}


def _stability_rows():
    rng = random.Random(11)
    rows = []
    for k in range(160):
        for j in range(rng.choice((1, 1, 2, 3, 4))):
            rows.append(row('5%07d%04d' % (k, j), 'G%04d' % k,
                            rng.choice(('dog', 'not_dog')),
                            cap('g%d' % k, 3600.0 * j)))
    return rows


def check_stable_split(plan=None):
    """Removing any one of 40 rows moves the rows of at most one group."""
    plan = plan or _plan
    rows = _stability_rows()
    base = plan(rows)
    worst = 0
    for drop in random.Random(2).sample(rows, 40):
        got = plan([r for r in rows if r['uid'] != drop['uid']])
        moved = {got[u][1] for u in got if u in base and
                 got[u][0] != base[u][0]}
        worst = max(worst, len(moved))
    if worst > 1:
        return ['T2 one row removed moved the split of %d groups -- the '
                'split re-deals on every new verdict' % (worst,)]
    return []


def check_stable_cap(cap_fn=None):
    """Removing a row of one (sequence, class) bucket changes no other
    bucket's survivors."""
    cap_fn = cap_fn or rcd.cap_rows
    rows = [row('6%07d%04d' % (k, j), 'H%03d' % k, 'dog')
            for k in range(30) for j in range(6)]
    base = {r['uid'] for r in cap_fn(rows, 2, 0)[0]}
    gone = rows[0]
    got = {r['uid'] for r in cap_fn([r for r in rows if r is not gone], 2,
                                    0)[0]}
    other = {u for u in base ^ got if not u.startswith('ga_6%07d' % 0)}
    if other:
        return ['T2 removing one row changed which rows survive the cap in '
                '%d other bucket(s)' % (len({u[:11] for u in other}),)]
    return []


def check_seed_still_matters():
    rows = _stability_rows()
    a = {u: v[0] for u, v in _plan(rows, seed=0).items()}
    b = {u: v[0] for u, v in _plan(rows, seed=1).items()}
    if a == b:
        return ['T2 the split does not depend on the seed']
    return []


# ── mutants: the old behaviour must make the checks fail ────────────────────

def _old_cap(rows, cap, rng):
    per = collections.defaultdict(list)
    for r in rows:
        per[(r['sequence'], r['label'])].append(r)
    kept, dropped = [], collections.Counter()
    for k in sorted(per):
        lst = sorted(per[k], key=lambda r: r['uid'])
        rng.shuffle(lst)
        if len(lst) > cap:
            dropped[k[1]] += len(lst) - cap
            lst = lst[:cap]
        kept += lst
    kept.sort(key=lambda r: r['uid'])
    return kept, dropped


def _old_assign(rows, group_of, val_frac, rng):
    items = collections.defaultdict(list)
    for r in rows:
        items[group_of[r['uid']]].append(r)
    order = sorted(items)
    rng.shuffle(order)
    want = collections.Counter()
    for r in rows:
        want[r['label']] += val_frac
    got = collections.Counter()
    split = {}
    for g in order:
        cnt = collections.Counter(r['label'] for r in items[g])
        if any(got[c] < want[c] for c in cnt):
            split[g] = 'val'
            got.update(cnt)
        else:
            split[g] = 'train'
    return split, set()


def _old_shared_rng_plan(rows):
    rng = random.Random(0)          # ONE generator for both, as before
    kept, _ = _old_cap(rows, 2, rng)
    g = rcd.split_groups(kept, [])
    sp, _ = _old_assign(kept, g, 0.2, rng)
    return {r['uid']: (sp[g[r['uid']]], g[r['uid']]) for r in kept}


def _within_bucket_only(items, seconds=60, **k):
    b = collections.defaultdict(list)
    for n, c in items:
        if c:
            b[(c['creator'], int(c['ts'] // seconds))].append((n, c))
    out = set()
    for lst in b.values():
        for i, (ni, ci) in enumerate(lst):
            for nj, cj in lst[i + 1:]:
                if ni != nj and rcd.is_capture_twin(ci, cj):
                    out.add(tuple(sorted((ni, nj))))
    return sorted(out)


def _geo_required(a, b, *args, **k):
    if not a or not b or None in (a.get('lon'), b.get('lon')):
        return False
    return _REAL_IS_TWIN(a, b, *args, **k)


_REAL_IS_TWIN = rcd.is_capture_twin
_REAL_SPLIT = rcd.split_groups
_REAL_TWIN_UNITS = rts.twin_units
_REAL_DRAW_UNITS = rts.capture_draw_units


def _patched(obj, name, value, fn):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        return fn()
    finally:
        setattr(obj, name, old)


MUTANTS = [
    ('T1 the split joins no capture twins',
     lambda: _patched(rcd, 'capture_twin_unions', lambda rows: ([], []),
                      check_twins_joined)),
    ('T1 twins looked for within a bucket only',
     lambda: _patched(rcd, 'capture_twins', _within_bucket_only,
                      check_bucket_edge)),
    ('T1 no twin without a geometry',
     lambda: _patched(rcd, 'is_capture_twin', _geo_required,
                      check_missing_geometry)),
    ('T1 the twin assertion a no-op',
     lambda: _patched(rcd, 'capture_twins', lambda items, **k: [],
                      check_assertion)),
    ('T1 reserve twins kept',
     lambda: _patched(rcd, 'reserve_twin_ids', lambda rows, res: {},
                      check_reserve_twin_dropped)),
    ('T1 the test draw ignores capture twins',
     lambda: _patched(rcd, 'capture_twins', lambda items, **k: [],
                      check_test_draw_units)),
    ('T1 manifest_main does not pass the twin unions to split_groups',
     lambda: _patched(rcd, 'split_groups',
                      lambda rows, unions, tw=None: _REAL_SPLIT(rows, unions,
                                                                ()),
                      check_reserve_twin_dropped)),
    ('T1 --verify-only claims a skipped twin check held',
     lambda: _patched(rcd, 'twin_check_skipped', lambda rows, res=(): [],
                      check_reserve_twin_dropped)),
    ('T1 reserve_test_set.main draws without the captures',
     lambda: _patched(rts, 'twin_units',
                      lambda cands, dist=rts.TWIN_DHASH, captures=None:
                      _REAL_TWIN_UNITS(cands, dist),
                      check_test_draw_main)),
    ('T1 reserve_test_set.main does not fence compared images\' twins',
     lambda: _patched(rts, 'capture_draw_units',
                      lambda c, f, w, p, repo=rts.REPO:
                      _REAL_DRAW_UNITS(c, (), w, p, repo),
                      check_test_draw_main)),
    ('T2 one shared rng for the cap and the split',
     lambda: check_stable_split(_old_shared_rng_plan)),
    ('T2 the split shuffled as one list',
     lambda: check_stable_split(lambda rows: _plan(
         rows, split_fn=lambda k, g, f, s: _old_assign(
             k, g, f, random.Random(s))))),
    ('T2 the cap shuffled with one rng',
     lambda: check_stable_cap(lambda rows, cap, s: _old_cap(
         rows, cap, random.Random(s)))),
]


def main():
    checks = [check_twins_joined, check_not_twins, check_missing_geometry,
              check_bucket_edge, check_assertion, check_reserve_twin_dropped,
              check_test_draw_units, check_test_draw_main, check_stable_split, check_stable_cap,
              check_seed_still_matters]
    bad = []
    try:
        for fn in checks:
            try:
                got = fn()
            except Exception as e:        # noqa: BLE001 - report, not die
                got = ['threw %s: %s' % (type(e).__name__, e)]
            for b in got:
                bad.append('%s: %s' % (fn.__name__, b))
            print(('FAIL ' if got else 'ok   ') + fn.__name__)
        for name, fn in MUTANTS:
            try:
                got = fn()
            except Exception as e:        # noqa: BLE001
                got = ['threw %s' % (type(e).__name__,)]
            if not got:
                bad.append('MUTANT %s: the check still passes with the old '
                           'behaviour put back -- it proves nothing' % (name,))
            print(('ok   ' if got else 'FAIL ') + 'mutant ' + name)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    for b in bad:
        print('FAIL ' + b)
    if bad:
        return 1
    print('split twins: capture twins join one split group, never cross a '
          'reserve, and one row moves at most one group (%d checks, %d '
          'mutants)' % (len(checks), len(MUTANTS)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
