"""
Adversarial test: compact() crash-recovery (§5.7).

The §5.7 deletion loop can crash at ANY point after the compacted pair's
renames: with no parts deleted, some deleted, or all deleted but sidecars
left. Rerunning compact() must finish the job in every case -- and must
REFUSE when the compacted pair does not contain the surviving parts (i.e.
new parts landed after compaction, so the pair is stale).

The original implementation verified the compacted row count against the
sum of the SURVIVING parts, so a crash mid-deletion wedged the cell forever
(every rerun raised StoreError) while queries double-counted the surviving
rows.

Run:  <python-with-pyarrow> tools/detect/tests/adv_test_compact_recovery.py
Exits non-zero on any failure. Throwaway DETECT_ROOT under TMPDIR.
"""

import json
import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pyarrow.parquet as pq  # noqa: E402

import store  # noqa: E402

FAILURES = []


def check(name, fn):
    try:
        fn()
    except Exception:
        FAILURES.append(name)
        print(f'FAIL  {name}')
        traceback.print_exc()
    else:
        print(f'ok    {name}')


ROOT = tempfile.mkdtemp(prefix='adv_compact_')
BASE = 10_000_000_000_000_000
GEN, REGION, CELL = 1, 'Asia', 'Asia_0_0_5_5'
KEEP = os.path.join(ROOT, '_keep')  # saved part copies, outside the store


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
        'ts_off': iid % 1000,
        'run_id': 1,
    }


def det_row(iid, j=0):
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


def build_cell(w):
    for d, drive in enumerate(('lynx', 'bobcat')):
        sw = w.open_shard(GEN, REGION, CELL, drive, 0, 0, 10)
        for i in range(10):
            iid = BASE + d * 1000 + i
            if i % 5 == 0:
                sw.add_image(img_row(iid, n_det=1, max_conf=0.5))
                sw.add_detections([det_row(iid)])
            else:
                sw.add_image(img_row(iid))
        sw.commit(10)


def test_recovery_mid_deletion():
    """Crash after SOME parts were deleted: rerun finishes, no wedge."""
    store.ensure_bootstrap(ROOT)
    w = store.Writer(detect_root=ROOT)
    build_cell(w)

    pdir = store.pair_dir(ROOT, GEN, REGION, CELL, 'lynx')
    img_p = os.path.join(pdir, store.part_basename(0, 0, 10, 'img'))
    det_p = os.path.join(pdir, store.part_basename(0, 0, 10, 'det'))
    os.makedirs(KEEP, exist_ok=True)
    shutil.copyfile(img_p, os.path.join(KEEP, 'p.img'))
    shutil.copyfile(det_p, os.path.join(KEEP, 'p.det'))

    out_img, out_det = store.compact(GEN, CELL, ROOT)
    n_img = pq.ParquetFile(out_img).metadata.num_rows
    assert n_img == 20, n_img

    # Simulate the crash-mid-deletion state: compacted pair present, ONE
    # surviving part pair back on disk.
    shutil.copyfile(os.path.join(KEEP, 'p.img'), img_p)
    shutil.copyfile(os.path.join(KEEP, 'p.det'), det_p)
    out2 = store.compact(GEN, CELL, ROOT)  # must NOT raise
    assert out2 == (out_img, out_det), out2
    assert not os.path.exists(img_p) and not os.path.exists(det_p)
    assert pq.ParquetFile(out_img).metadata.num_rows == 20  # untouched
    store.invariants(ROOT)  # no double-count left behind
    _, bad = store.verify(ROOT)
    assert bad == [], bad


def test_refuses_stale_compact():
    """A part with image_ids UNKNOWN to the compacted pair must survive."""
    w = store.Writer(detect_root=ROOT)
    sw = w.open_shard(GEN, REGION, CELL, 'lynx', 1, 0, 3)
    for i in range(3):
        sw.add_image(img_row(BASE + 50_000 + i))
    new_img, new_det = sw.commit(3)
    try:
        store.compact(GEN, CELL, ROOT)
        raise AssertionError('stale compacted pair was not refused')
    except store.StoreError as exc:
        assert 'does not contain' in str(exc), exc
    # Nothing deleted: the new part and the compacted pair both survive.
    assert os.path.exists(new_img) and os.path.exists(new_det)
    cell_dir = os.path.dirname(
        store.pair_dir(ROOT, GEN, REGION, CELL, 'lynx'))
    assert os.path.exists(os.path.join(cell_dir, 'drive=_merged', 'compact.img.parquet'))
    # Clean up the extra part so later checks see a consistent store.
    os.remove(new_img)
    os.remove(new_det)


def test_stale_sidecar_cleanup():
    """Crash after all deletions but before sidecar removal: rerun cleans."""
    pdir = store.pair_dir(ROOT, GEN, REGION, CELL, 'bobcat')
    os.makedirs(pdir, exist_ok=True)
    state = os.path.join(pdir, store._STATE_NAME)
    with open(state, 'w') as f:
        json.dump({'shards': {'0': {'parts': [[0, 10]], 'len': 10}}}, f)
    out = store.compact(GEN, CELL, ROOT)
    assert out is not None
    assert not os.path.exists(state), 'stale _state.json not cleaned'


def main():
    print(f'adv compact-recovery test in {ROOT}')
    try:
        check('recovery_mid_deletion', test_recovery_mid_deletion)
        check('refuses_stale_compact', test_refuses_stale_compact)
        check('stale_sidecar_cleanup', test_stale_sidecar_cleanup)
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
    print(f'\n{3 - len(FAILURES)} passed, {len(FAILURES)} failed')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.exit(main())
