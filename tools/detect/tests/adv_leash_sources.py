#!/usr/bin/env python3
"""The leash crop sources: four holes the leash_20261115_18b269 audit found
in crop_sources.py, each measured on that build first.

  L1  apply_box_vetoes matched two DETECTOR keys only when they were equal.
      The gen-1 store double-boxes, so the gate audit's not_dog on d0 and the
      leash audit's "unleashed" on d1 of the same object (IoU 0.889, a bird
      on 2020931115336701) both stood, and the bird trained as an unleashed
      dog. Differing d-keys are now one box when the row's POOL box (the
      detector's, never the redraw) overlaps the veto at IoU >= LS_IOU, and
      that drop is its own conflict kind ('..._overlap').
  L2  apply_vetoes (Label Studio "other animal"/cow) tested only `box`. For a
      redrawn row that is the leash-inclusive redraw, median 1.6x the
      detector box, which can fall under LS_IOU against the very box it
      redrew. Both boxes are tried now.
  L3  data/leash_label_conflicts.json excluded 10 whole images (34 good
      ls_box rows, the only mixed leashed/unleashed scenes) on the strength
      of an index misalignment between two old lineages. Retired: nothing is
      excluded, the digest is still recorded.
  L4  framing_check_audit wanted an exact size match. The audit page ledgers
      a redraw to 2 decimals and cuts edited/ from the unrounded floats, so 2
      of 270 edited/ crops are 1 px off; the seeded sample decided whether a
      build aborted. edited/ may now differ by <= 1 px per axis (counted and
      reported); full/ stays exact. collect() keeps the result as
      counts['framing_check'], so the tolerated uids reach the bundle.

Synthetic rows and frames in a temp directory; nothing real is read. Each
check returns a list of failures, and every MUTANT re-runs a check with the
old (or a wrong) behaviour put back and requires it to FAIL.

Run: python tools/detect/tests/adv_leash_sources.py   (mp14 env)
"""
import inspect
import json
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DETECT = os.path.join(REPO, 'tools', 'detect')
sys.path.insert(0, DETECT)

import crop_sources as cs                     # noqa: E402

TMP = tempfile.mkdtemp(prefix='adv_leash_sources_')
DOGS = ('leashed', 'unleashed')
KIND = 'leash_vs_gate_not_dog'

# the two real store boxes of 2020931115336701, IoU 0.889
REAL_D0 = [1791.0, 3354.0, 1891.0, 3474.0]
REAL_D1 = [1787.0, 3350.0, 1892.0, 3478.0]


def _row(iid, key, label, pool_box, box=None, source='leash_audit',
         framing='audit_full'):
    return {'uid': cs.uid_for(source, iid, key), 'image_id': iid,
            'det_key': key, 'label': label, 'source': source,
            'pool_box': list(pool_box) if pool_box else None,
            'box': list(box or pool_box), 'framing': framing}


def _veto(iid, key, box, label='not_dog', source='gate_audit'):
    return {'image_id': iid, 'det_key': key, 'box': list(box),
            'label': label, 'source': source,
            'uid': cs.uid_for(source, iid, key), 'by': 't', 'ts': 1}


# ── L1: differing detector keys on one object ───────────────────────────────

def check_box_veto_overlap(fn=None):
    fn = fn or cs.apply_box_vetoes
    bad = []
    if cs.iou(REAL_D0, REAL_D1) < 0.85:
        return ['L1 the fixture boxes no longer overlap the way the real '
                'pair does (%.3f)' % cs.iou(REAL_D0, REAL_D1)]
    # d1 "unleashed" vs gate not_dog on d0, IoU 0.889: one box
    row = _row('2020931115336701', 'd1', 'unleashed', REAL_D1)
    kept, conf = fn([row], [_veto('2020931115336701', 'd0', REAL_D0)],
                    DOGS, KIND)
    if kept:
        bad.append('L1 a leash row whose detector box overlaps a gate '
                   'not_dog box at IoU 0.89 under another det key survived')
    elif not conf or conf[0].get('kind') != KIND + '_overlap':
        bad.append('L1 the overlap drop was not recorded as %s_overlap: %r'
                   % (KIND, [c.get('kind') for c in conf]))
    elif {m['uid'] for m in conf[0]['members']} != {
            row['uid'], cs.uid_for('gate_audit', '2020931115336701', 'd0')}:
        bad.append('L1 the overlap conflict does not name both records: %r'
                   % (conf[0]['members'],))
    # a different dog beside it, IoU ~0.33 < LS_IOU: two boxes, kept
    near = [100.0, 100.0, 200.0, 200.0]
    side = [150.0, 100.0, 250.0, 200.0]
    assert cs.iou(near, side) < cs.LS_IOU
    kept, conf = fn([_row('970100000001', 'd1', 'leashed', side)],
                    [_veto('970100000001', 'd0', near)], DOGS, KIND)
    if len(kept) != 1 or conf:
        bad.append('L1 a leash row under LS_IOU of a not_dog box on another '
                   'det key was dropped (IoU %.2f)' % cs.iou(near, side))
    # equal keys are one box whatever the geometry, and keep the plain kind
    kept, conf = fn([_row('970100000002', 'd3', 'leashed', near)],
                    [_veto('970100000002', 'd3', [900, 900, 950, 950])],
                    DOGS, KIND)
    if kept or not conf or conf[0].get('kind') != KIND:
        bad.append('L1 an equal-key veto no longer drops its row as %s (%r)'
                   % (KIND, [c.get('kind') for c in conf]))
    # the overlap is the DETECTOR's box, never the redraw: a leash-inclusive
    # redraw that swallows the neighbouring not_dog box is still another box
    redraw = [95.0, 95.0, 215.0, 205.0]
    assert cs.iou(redraw, near) >= cs.LS_IOU > cs.iou(side, near)
    kept, conf = fn([_row('970100000003', 'd1', 'unleashed', side, redraw,
                          framing='audit_edited')],
                    [_veto('970100000003', 'd0', near)], DOGS, KIND)
    if len(kept) != 1 or conf:
        bad.append('L1 a redraw (not the detector box) was joined against '
                   'another det key\'s not_dog box')
    # a label outside drop_labels is never touched
    kept, _ = fn([_row('970100000004', 'd1', 'not_dog', REAL_D1)],
                 [_veto('970100000004', 'd0', REAL_D0)], DOGS, KIND)
    if len(kept) != 1:
        bad.append('L1 a row whose label is not in drop_labels was dropped')
    # a drawn (b-key) row still joins by overlap, as before
    kept, conf = fn([_row('970100000005', 'b0', 'leashed', None, REAL_D1,
                          source='ls_box', framing='cut_crops')],
                    [_veto('970100000005', 'd0', REAL_D0)], DOGS, KIND)
    if kept or not conf or conf[0].get('kind') != KIND:
        bad.append('L1 a drawn box over a not_dog detector box survived')
    # one row hit by an equal-key veto AND a differing-key overlap: not an
    # overlap-only drop, so the plain kind, and both vetoes named
    row = _row('970100000006', 'd3', 'leashed', REAL_D1)
    vs = [_veto('970100000006', 'd3', REAL_D1),
          _veto('970100000006', 'd0', REAL_D0)]
    kept, conf = fn([row], vs, DOGS, KIND)
    if kept or not conf or conf[0].get('kind') != KIND:
        bad.append('L1 a row hit by an equal-key veto as well as an overlap '
                   'was not recorded under the plain kind %s: %r'
                   % (KIND, [c.get('kind') for c in conf]))
    elif {m['uid'] for m in conf[0]['members']} != (
            {row['uid']} | {v['uid'] for v in vs}):
        bad.append('L1 a row hit by two vetoes does not name both: %r'
                   % (conf[0]['members'],))
    # a d-keyed row WITHOUT a pool box is matched by key only: its box may
    # be a redraw, which is never the detector's box
    kept, conf = fn([_row('970100000007', 'd1', 'unleashed', None, redraw,
                          framing='audit_edited')],
                    [_veto('970100000007', 'd0', near)], DOGS, KIND)
    if len(kept) != 1 or conf:
        bad.append('L1 a d-keyed row with no pool box was joined on its '
                   '(possibly redrawn) box against another det key')
    return bad


def _old_apply_box_vetoes(rows, vetoes, drop_labels, kind):
    # before L1: differing d-keys never compared
    out_keep, out_conf = [], []
    for row in rows:
        hits = []
        if row['label'] in drop_labels:
            rk = str(row.get('det_key') or '')
            rb = row.get('pool_box') or row.get('box')
            for v in vetoes:
                if v['image_id'] != row['image_id']:
                    continue
                vk = str(v.get('det_key') or '')
                if rk.startswith('d') and vk.startswith('d'):
                    if rk == vk:
                        hits.append(v)
                elif rb and v.get('box') and cs.iou(rb, v['box']) >= cs.LS_IOU:
                    hits.append(v)
        if hits:
            out_conf.append({'kind': kind, 'members': [{'uid': row['uid']}]})
            continue
        out_keep.append(row)
    return out_keep, out_conf


def _mutated(fn, old, new):
    """fn re-compiled from its own source with `old` -> `new`, in a copy of
    its module's namespace (so the real function is left alone)."""
    src = inspect.getsource(fn)
    if src.count(old) != 1:
        raise AssertionError('mutant anchor %r not found once in %s'
                             % (old, fn.__name__))
    ns = dict(fn.__globals__)
    exec(compile(src.replace(old, new), fn.__code__.co_filename, 'exec'), ns)
    return ns[fn.__name__]


# ── L2: the Label Studio "other animal" veto tries both boxes ───────────────

def _ls_veto(iid, pct, orig=(1000, 1000)):
    return {'image_id': iid, 'det_key': 'b0', 'pct': pct,
            'ls_label': 'other animal', 'orig': orig}


def check_ls_veto_both_boxes(fn=None):
    fn = fn or cs.apply_vetoes
    bad = []
    pct = (10.0, 10.0, 10.0, 10.0)               # -> [100,100,200,200] px
    det = [100.0, 100.0, 200.0, 200.0]
    # an enlarged (leash-inclusive) redraw over a detector box LS calls
    # "other animal": the redraw alone is IoU 0.25
    redraw = [60.0, 60.0, 240.0, 280.0]
    assert cs.iou(redraw, det) < cs.LS_IOU
    kept, conf = fn([_row('970200000001', 'd0', 'leashed', det, redraw,
                          framing='audit_edited')],
                    [_ls_veto('970200000001', pct)])
    if kept or not conf:
        bad.append('L2 an audit_edited redraw whose detector box sits under '
                   'an LS "other animal" box survived (redraw IoU %.2f)'
                   % cs.iou(redraw, det))
    # a correction of a misplaced detector box: the box matches, the pool
    # box does not -- still dropped
    off = [180.0, 180.0, 300.0, 300.0]
    assert cs.iou(off, det) < cs.LS_IOU
    kept, conf = fn([_row('970200000002', 'd0', 'unleashed', off, det,
                          source='leash_call', framing='cut_crops')],
                    [_ls_veto('970200000002', pct)])
    if kept or not conf:
        bad.append('L2 a corrected box over an LS "other animal" box '
                   'survived because its detector box did not overlap')
    # neither box overlapping: kept
    kept, conf = fn([_row('970200000003', 'd0', 'leashed', off,
                          [400, 400, 500, 500])],
                    [_ls_veto('970200000003', pct)])
    if len(kept) != 1 or conf:
        bad.append('L2 a row overlapping no LS veto box was dropped')
    # a row with no pool box (a Label Studio row) still works on its box
    kept, conf = fn([_row('970200000004', 'b1', 'leashed', None, det,
                          source='ls_box', framing='cut_crops')],
                    [_ls_veto('970200000004', pct)])
    if kept or not conf:
        bad.append('L2 an ls_box row on the "other animal" box survived')
    return bad


def _old_apply_vetoes(rows, vetoes):
    keep, conf = [], []
    for row in rows:
        hits = []
        for v in vetoes:
            if v['image_id'] != row['image_id']:
                continue
            vb = cs.ls_pixel_box(v['pct'], *v['orig'])
            if row.get('box') and cs.iou(row['box'], vb) >= cs.LS_IOU:
                hits.append(v)
        if hits:
            conf.append({'kind': 'leash_vs_ls_not_dog'})
            continue
        keep.append(row)
    return keep, conf


def _pool_only_apply_vetoes(rows, vetoes):
    # the "simpler alternative": pool_box or box, never both
    return _old_apply_vetoes(
        [dict(r, box=r.get('pool_box') or r.get('box')) for r in rows],
        vetoes)


# ── L3: the leash label conflicts file is retired ───────────────────────────

def check_conflicts_retired(fn=None):
    fn = fn or cs.exclusions
    fake = types.ModuleType('label_flags')
    fake.DB_PATH = os.path.join(TMP, 'lf.db')
    fake.flagged_files = lambda: {'x': {'image_id': '970300000001'}}
    data = os.path.join(TMP, 'excl_data')
    os.makedirs(data, exist_ok=True)
    conflicts = os.path.join(data, 'leash_label_conflicts.json')
    with open(conflicts, 'w') as fh:
        json.dump(['970300000002_0.jpg', '970300000002_1.jpg',
                   '970300000003_0.jpg'], fh)
    saved_mod, saved_d = sys.modules.get('label_flags'), cs.D
    sys.modules['label_flags'] = fake
    cs.D = lambda *p: os.path.join(data, *p)
    try:
        got, digests = fn('leash', os.path.join(TMP, 'no_root'))
        got_dogbin, _ = fn('dogbin', os.path.join(TMP, 'no_root'))
    finally:
        cs.D = saved_d
        if saved_mod is None:
            sys.modules.pop('label_flags', None)
        else:
            sys.modules['label_flags'] = saved_mod
    bad = []
    if got.get('970300000001') != 'label_flags' \
            or got_dogbin.get('970300000001') != 'label_flags':
        bad.append('L3 exclusions() stopped honouring label_flags')
    if '970300000002' in got or '970300000003' in got:
        bad.append('L3 the retired leash_label_conflicts.json still excludes '
                   'images: %r' % ({k: v for k, v in got.items()
                                    if k != '970300000001'},))
    if digests.get('leash_label_conflicts') != cs.sha256_file(conflicts):
        bad.append('L3 the conflicts file is no longer digested (%r) -- a '
                   'reader cannot see it was consulted'
                   % (digests.get('leash_label_conflicts'),))
    status = str(digests.get('leash_label_conflicts:status') or '')
    if 'retired' not in status or '2 image_id' not in status:
        bad.append('L3 the bundle does not say the conflicts file was '
                   'retired, nor how many image_ids it names: %r' % (status,))
    return bad


def _old_exclusions(family, training_root=None):
    # before L3: the conflict list applied, as whole-image exclusions
    out, digests = _REAL_EXCLUSIONS(family, training_root)
    if family == 'leash':
        p = cs.D('leash_label_conflicts.json')
        with open(p) as fh:
            for name in json.load(fh) or []:
                out.setdefault(cs.image_id_of(name), 'leash_label_conflicts')
    return out, digests


_REAL_EXCLUSIONS = cs.exclusions


# ── L4: the audit framing check's 1-px edited/ tolerance ────────────────────

def _jpeg(path, wh):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new('RGB', wh, (90, 90, 90)).save(path, quality=90)
    return path


def _framing(fn, framing, have, got):
    """(error or None, result) of fn over one audit row whose file is `have`
    and whose re-cut is `got`."""
    frame = _jpeg(os.path.join(TMP, 'frame.jpg'), (64, 64))
    crop = _jpeg(os.path.join(TMP, 'crops', '%s_%dx%d.jpg'
                              % (framing, have[0], have[1])), have)
    row = {'uid': 'la_970400000001_d0.jpg', 'framing': framing,
           'box': [1.0, 1.0, 20.0, 20.0], 'path': crop,
           'frame': {'path': frame}}
    g = fn.__globals__
    saved = g['recut_size']
    g['recut_size'] = lambda path, box: tuple(got)
    try:
        return None, fn([row], say=lambda *_: None)
    except SystemExit as e:
        return str(e), None
    finally:
        g['recut_size'] = saved


def check_framing_tolerance(fn=None):
    fn = fn or cs.framing_check_audit
    bad = []
    err, res = _framing(fn, 'audit_edited', (252, 303), (252, 304))
    if err:
        bad.append('L4 an edited/ crop 1 px off its re-cut (the redraw '
                   'rounding of la_2000627870347514_d0) aborted the build: '
                   '%s' % (err,))
    elif not res or res.get('tolerated') != 1:
        bad.append('L4 a tolerated 1-px edited/ mismatch was not counted: %r'
                   % (res,))
    err, res = _framing(fn, 'audit_edited', (333, 223), (333, 223))
    if err or not res or res.get('tolerated'):
        bad.append('L4 an exact edited/ match was counted as tolerated or '
                   'failed: %r %r' % (err, res))
    err, _ = _framing(fn, 'audit_edited', (252, 302), (252, 304))
    if not err or 'framed' not in err:
        bad.append('L4 an edited/ crop 2 px off its re-cut passed as '
                   'production-framed')
    err, _ = _framing(fn, 'audit_full', (252, 303), (252, 304))
    if not err or 'framed' not in err:
        bad.append('L4 a full/ crop 1 px off its re-cut passed -- full/ is '
                   'cut from the detector box and must match exactly')
    return bad


def check_framing_result_stored(src=None):
    """collect() keeps framing_check_audit's result in counts, the only way
    the tolerated uids reach the bundle (collect_counts.json)."""
    src = src if src is not None else inspect.getsource(cs.collect)
    if "counts['framing_check'] = framing_check_audit(" not in src:
        return ["L4 collect() runs the framing check but does not store its "
                "result as counts['framing_check'] -- the tolerated uids "
                "never reach collect_counts.json"]
    return []


def _with_tol(px):
    def run():
        saved = cs.EDITED_TOL_PX
        cs.EDITED_TOL_PX = px
        try:
            return check_framing_tolerance()
        finally:
            cs.EDITED_TOL_PX = saved
    return run


MUTANTS = [
    ('L1 differing detector keys compared by key only',
     lambda: check_box_veto_overlap(_old_apply_box_vetoes)),
    ('L1 the d/d overlap taken on the redraw, not the detector box',
     lambda: check_box_veto_overlap(_mutated(
         cs.apply_box_vetoes, "iou(row['pool_box'], v['box'])",
         "iou(row['box'], v['box'])"))),
    ('L1 overlap drops recorded under the plain kind',
     lambda: check_box_veto_overlap(_mutated(
         cs.apply_box_vetoes, "kind + '_overlap' if", "kind if"))),
    ('L2 the "other animal" veto on box only',
     lambda: check_ls_veto_both_boxes(_old_apply_vetoes)),
    ('L2 the "other animal" veto on pool_box-or-box only',
     lambda: check_ls_veto_both_boxes(_pool_only_apply_vetoes)),
    ('L3 the conflicts file applied again',
     lambda: check_conflicts_retired(_old_exclusions)),
    ('L4 the framing check exact for edited/ too', _with_tol(0)),
    ('L4 the framing check tolerating 2 px', _with_tol(2)),
    ('L4 the 1-px tolerance applied to full/ as well',
     lambda: check_framing_tolerance(_mutated(
         cs.framing_check_audit, "row['framing'] == 'audit_edited' and got",
         "got"))),
    ('L1 \'_overlap\' when any hit (not every hit) is an overlap',
     lambda: check_box_veto_overlap(_mutated(
         cs.apply_box_vetoes, "if n_overlap == len(hits)", "if n_overlap"))),
    ('L1 the d/d overlap falling back to the redraw when no pool box',
     lambda: check_box_veto_overlap(_mutated(
         cs.apply_box_vetoes,
         "elif (row.get('pool_box') and v.get('box')\n"
         "                          and iou(row['pool_box'], v['box'])",
         "elif (rb and v.get('box')\n"
         "                          and iou(rb, v['box'])"))),
    ('L4 collect() dropping the framing result',
     lambda: check_framing_result_stored(inspect.getsource(cs.collect).replace(
         "counts['framing_check'] = framing_check_audit(",
         "framing_check_audit("))),
]


def main():
    checks = [check_box_veto_overlap, check_ls_veto_both_boxes,
              check_conflicts_retired, check_framing_tolerance,
              check_framing_result_stored]
    bad = []
    try:
        for fn in checks:
            try:
                got = fn()
            except Exception as e:        # noqa: BLE001 - report, not die
                got = ['%s threw %s: %s' % (fn.__name__, type(e).__name__, e)]
            bad += got
            print(('FAIL ' if got else 'ok   ') + fn.__name__)
        for name, fn in MUTANTS:
            try:
                got = fn()
            except Exception as e:        # noqa: BLE001
                got = ['threw %s: %s' % (type(e).__name__, e)]
            if not got:
                bad.append('MUTANT %s: the check still passes with the old '
                           'behaviour put back -- it proves nothing' % (name,))
                print('FAIL mutant ' + name)
            else:
                print('ok   mutant ' + name)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    for b in bad:
        print('FAIL ' + b)
    if bad:
        return 1
    print('leash sources: duplicate detector boxes vetoed by overlap, the '
          '"other animal" veto on both boxes, the conflicts file retired, '
          'edited/ framing within 1 px (%d checks, %d mutants)'
          % (len(checks), len(MUTANTS)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
