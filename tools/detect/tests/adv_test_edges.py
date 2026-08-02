"""
Adversarial edge tests for store.py beyond the builder self-test:

  * image_id past UINT64 fails BEFORE any bytes hit disk (no partial part)
  * det_idx 256+ (the §5.3 UINT8 vs §4.6 max_det=300 contradiction) is a
    loud CommitError at intake, not an ArrowInvalid at commit; det_idx 255
    with n_det=256 still commits (boundary)
  * two GOOD overlapping committed parts: first adopted, overlapper deleted
  * tiling_resume on an empty dir and a nonexistent dir -> ([], False)
  * a dir holding parts of two shards requires shard_idx (ValueError)
  * repair=False reports but never deletes
  * _state.json keeps the 'len' of earlier shards across rebuilds
  * a det-only orphan file IS visible to the invariants glob (orphan_dets)
    until repaired -- then invariants pass again

Run:  <python-with-pyarrow> tools/detect/tests/adv_test_edges.py
Exits non-zero on any failure. Throwaway DETECT_ROOT under TMPDIR.
"""

import json
import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import store  # noqa: E402

FAILURES = []
PASSES = []


def check(name, fn):
    try:
        fn()
    except Exception:
        FAILURES.append(name)
        print(f'FAIL  {name}')
        traceback.print_exc()
    else:
        PASSES.append(name)
        print(f'ok    {name}')


ROOT = tempfile.mkdtemp(prefix='adv_edges_')
BASE = 10_000_000_000_000_000
GEN, REGION = 1, 'Europe'


def img_row(iid, n_det=0, max_conf=None):
    return {
        'image_id': iid,
        'drive': 1,
        'status': 0,
        'n_det': n_det,
        'max_conf': max_conf,
        'orig_w': 100,
        'orig_h': 100,
        'reduce': 1,
        'guards': 0,
        'ts_off': 1,
        'run_id': 1,
    }


def det_row(iid, j):
    return {
        'image_id': iid,
        'det_idx': j,
        'conf': 0.5,
        'x1': 1.0,
        'y1': 1.0,
        'x2': 2.0,
        'y2': 2.0,
        'run_id': 1,
    }


def test_image_id_overflow_writes_nothing():
    w = store.Writer(detect_root=ROOT)
    sw = w.open_shard(GEN, REGION, 'C_ovf', 'lynx', 0, 0, 1)
    sw.add_image(img_row(2**64))  # one past UINT64
    try:
        sw.commit(1)
        raise AssertionError('2**64 image_id committed')
    except (store.StoreError, Exception) as exc:
        assert not isinstance(exc, AssertionError)
    pdir = store.pair_dir(ROOT, GEN, REGION, 'C_ovf', 'lynx')
    left = [n for n in os.listdir(pdir) if n.endswith('.parquet')]
    assert left == [], f'partial part left on disk: {left}'


def test_det_idx_uint8_ceiling():
    w = store.Writer(detect_root=ROOT)
    sw = w.open_shard(GEN, REGION, 'C_didx', 'lynx', 0, 0, 1)
    try:
        sw.add_detections([det_row(BASE + 1, 256)])
        raise AssertionError('det_idx=256 accepted')
    except store.CommitError as exc:
        assert 'UINT8' in str(exc), exc
    try:
        sw.add_detections([{k: v for k, v in det_row(BASE + 1, 0).items()
                            if k != 'det_idx'}])
        raise AssertionError('missing det_idx accepted')
    except store.CommitError:
        pass
    # Boundary: 256 boxes (det_idx 0..255) is the most the schema can hold
    # and must commit cleanly.
    sw.add_image(img_row(BASE + 1, n_det=256, max_conf=0.5))
    sw.add_detections([det_row(BASE + 1, j) for j in range(256)])
    img_path, det_path = sw.commit(1)
    assert os.path.exists(img_path) and os.path.exists(det_path)


def test_overlapping_good_parts():
    w = store.Writer(detect_root=ROOT)
    cell = 'C_olap'
    sw = w.open_shard(GEN, REGION, cell, 'lynx', 0, 0, 60)
    for i in range(60):
        sw.add_image(img_row(BASE + 100 + i))
    sw.commit(60)
    # A second, overlapping committed range [40, 100) -- possible only via
    # operator error / a divergent writer, but resume must resolve it:
    # earlier coverage stands, the overlapper is deleted and redone.
    sw = w.open_shard(GEN, REGION, cell, 'lynx', 0, 40, 100)
    for i in range(60):
        sw.add_image(img_row(BASE + 5000 + i))
    sw.commit(100)
    pdir = store.pair_dir(ROOT, GEN, REGION, cell, 'lynx')
    parts, done = store.tiling_resume(pdir, 100, shard_idx=0)
    assert parts == [(0, 60)] and not done, (parts, done)
    names = sorted(n for n in os.listdir(pdir) if n.endswith('.parquet'))
    assert names == [
        store.part_basename(0, 0, 60, 'det'),
        store.part_basename(0, 0, 60, 'img'),
    ], names


def test_empty_and_missing_dirs():
    empty = os.path.join(ROOT, 'emptydir')
    os.makedirs(empty, exist_ok=True)
    assert store.tiling_resume(empty, 10) == ([], False)
    assert store.tiling_resume(os.path.join(ROOT, 'nosuch'), 10) == ([],
                                                                     False)


def test_multi_shard_dir_needs_shard_idx():
    w = store.Writer(detect_root=ROOT)
    cell = 'C_multi'
    for sid in (0, 1):
        sw = w.open_shard(GEN, REGION, cell, 'lynx', sid, 0, 5)
        for i in range(5):
            sw.add_image(img_row(BASE + 200 + sid * 100 + i))
        sw.commit(5)
    pdir = store.pair_dir(ROOT, GEN, REGION, cell, 'lynx')
    try:
        store.tiling_resume(pdir, 5)
        raise AssertionError('ambiguous shard_idx not rejected')
    except ValueError:
        pass
    assert store.tiling_resume(pdir, 5, shard_idx=1) == ([(0, 5)], True)
    # _state.json keeps shard 0's len after shard 1's commit rebuilt it.
    state = json.load(open(os.path.join(pdir, store._STATE_NAME)))
    assert state['shards']['0'].get('len') == 5, state
    assert state['shards']['1'].get('len') == 5, state


def test_repair_false_never_deletes():
    cell = 'C_norepair'
    w = store.Writer(detect_root=ROOT)
    sw = w.open_shard(GEN, REGION, cell, 'lynx', 0, 0, 5)
    for i in range(5):
        sw.add_image(img_row(BASE + 350 + i))
    sw.commit(5)
    pdir = store.pair_dir(ROOT, GEN, REGION, cell, 'lynx')
    # Bad-footer part (row count contradicts name) + det-only orphan + .tmp.
    for kind in ('img', 'det'):
        shutil.copyfile(os.path.join(pdir, store.part_basename(0, 0, 5,
                                                               kind)),
                        os.path.join(pdir, store.part_basename(0, 5, 9,
                                                               kind)))
    shutil.copyfile(os.path.join(pdir, store.part_basename(0, 0, 5, 'det')),
                    os.path.join(pdir, store.part_basename(1, 0, 5, 'det')))
    tmp = os.path.join(pdir,
                       store.part_basename(0, 9, 12, 'img') + '.tmp')
    open(tmp, 'wb').write(b'garbage')
    before = sorted(os.listdir(pdir))
    parts, done = store.tiling_resume(pdir, 5, shard_idx=0, repair=False)
    assert parts == [(0, 5)] and done  # bad parts reported, not adopted
    assert sorted(os.listdir(pdir)) == before, 'repair=False deleted files'
    # And with repair on, everything invalid goes away.
    parts, done = store.tiling_resume(pdir, 5, shard_idx=0)
    store.tiling_resume(pdir, 5, shard_idx=1)
    left = sorted(n for n in os.listdir(pdir) if n != store._STATE_NAME)
    assert left == [
        store.part_basename(0, 0, 5, 'det'),
        store.part_basename(0, 0, 5, 'img'),
    ], left


def test_orphan_det_poisons_invariants_until_repaired():
    cell = 'C_orph'
    w = store.Writer(detect_root=ROOT)
    sw = w.open_shard(GEN, REGION, cell, 'lynx', 0, 0, 1)
    iid = BASE + 400
    sw.add_image(img_row(iid, n_det=1, max_conf=0.5))
    sw.add_detections([det_row(iid, 0)])
    img_path, det_path = sw.commit(1)
    pdir = store.pair_dir(ROOT, GEN, REGION, cell, 'lynx')
    # Crash between the det and img renames of a LATER part: det-only file
    # whose image never got an img row anywhere.
    orphan = os.path.join(pdir, store.part_basename(1, 0, 1, 'det'))
    shutil.copyfile(det_path, orphan)
    os.remove(img_path)  # now even the original image row is gone
    try:
        store.invariants(ROOT)
        raise AssertionError('orphan detections not caught')
    except store.InvariantError as exc:
        assert 'orphan_dets' in str(exc), exc
    # tiling_resume repairs both the orphan and the now-half shard 0.
    store.tiling_resume(pdir, 1, shard_idx=0)
    store.tiling_resume(pdir, 1, shard_idx=1)
    assert sorted(os.listdir(pdir)) == [store._STATE_NAME]
    store.invariants(ROOT)  # clean again


def main():
    print(f'adv edge tests in {ROOT}')
    store.ensure_bootstrap(ROOT)
    try:
        check('image_id_overflow_writes_nothing',
              test_image_id_overflow_writes_nothing)
        check('det_idx_uint8_ceiling', test_det_idx_uint8_ceiling)
        check('overlapping_good_parts', test_overlapping_good_parts)
        check('empty_and_missing_dirs', test_empty_and_missing_dirs)
        check('multi_shard_dir_needs_shard_idx',
              test_multi_shard_dir_needs_shard_idx)
        check('repair_false_never_deletes', test_repair_false_never_deletes)
        check('orphan_det_poisons_invariants_until_repaired',
              test_orphan_det_poisons_invariants_until_repaired)
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
    print(f'\n{len(PASSES)} passed, {len(FAILURES)} failed')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.exit(main())
