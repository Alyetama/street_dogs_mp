"""
Self-test for tools/detect/store.py -- the §5/§6.3 crash windows and
accounting invariants, as plain asserts (no pytest).

Covers, per the build plan (§9 step 8):
  * byte-exact §5.3 schemas + encodings (DELTA_BINARY_PACKED, zstd, no dict)
  * partial commit + continuation and exact-tiling resume (§5.2, §6.3)
  * a SIGKILL-style torn .tmp left behind is ignored and repaired
  * NaN -> NULL mapping: the §5.4 threshold query returns the NULL answer
  * the three §5.4 invariants (+ progress identity) pass over the store
  * a truncated COMMITTED parquet is caught by verify() (§5.6)
  * the _bootstrap partition makes empty-store queries work (§5.2)
  * §5.5 guards: det-row hard abort, free-space hard abort, soft alarm,
    n_det==max_det flag
  * errors table round-trip; §5.7 compact

Run:  <python-with-pyarrow> tools/detect/tests/test_store.py
Exits non-zero on any failure. Uses a throwaway DETECT_ROOT under TMPDIR.
"""

import math
import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

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


# ---------------------------------------------------------------- fixtures


def img_row(iid, n_det=0, status=0, max_conf=None, guards=0):
    return {
        'image_id': iid,
        'drive': 1,
        'status': status,
        'n_det': n_det,
        'max_conf': max_conf,
        'orig_w': 4080,
        'orig_h': 3072,
        'reduce': 2,
        'guards': guards,
        'ts_off': iid % 100000,
        'run_id': 1,
    }


def det_rows(iid, n, conf0=0.9):
    return [{
        'image_id': iid,
        'det_idx': j,
        'conf': conf0 - 0.01 * j,
        'x1': 1.0,
        'y1': 2.0,
        'x2': 30.0,
        'y2': 40.0,
        'run_id': 1,
    } for j in range(n)]


def fill_shard(sw, ids, positive_every=3, nan_negatives=False):
    """Feed a ShardWriter: every Nth image positive, one error row."""
    for k, iid in enumerate(ids):
        if k == 1:
            sw.add_image(img_row(iid, status=store.STATUS_DECODE_ERROR))
        elif k % positive_every == 0:
            n = 1 + (k % 3)
            sw.add_image(img_row(iid, n_det=n, max_conf=0.9))
            sw.add_detections(det_rows(iid, n))
        else:
            # numpy-style NaN sentinel for a negative: writer must map to
            # NULL (§5.4).
            mc = float('nan') if nan_negatives else None
            sw.add_image(img_row(iid, n_det=0, max_conf=mc))


ROOT = tempfile.mkdtemp(prefix='store_test_')
GEN, REGION, CELL, DRIVE = 1, 'Europe', 'Europe_10_45_15_50', 'lynx'
BASE = 10_000_000_000_000_000  # 17-digit ids like production


def writer(**kw):
    return store.Writer(detect_root=ROOT, **kw)


# ------------------------------------------------------------------ tests


def test_bootstrap_empty_store():
    store.ensure_bootstrap(ROOT)
    store.ensure_bootstrap(ROOT)  # idempotent
    prog = store.invariants(ROOT)  # would raise IOException without it
    assert prog == {
        'scanned': 0,
        'positive': 0,
        'negative': 0,
        'errored': 0,
        'boxes': 0,
    }, prog


def test_schema_byte_exact_and_encodings():
    w = writer()
    sw = w.open_shard(GEN, REGION, CELL, DRIVE, 0, 0, 60)
    ids = [BASE + i * 7 for i in range(60)]
    fill_shard(sw, ids, nan_negatives=True)
    img_path, det_path = sw.commit(60)

    img = pq.read_table(img_path)
    det = pq.read_table(det_path)
    # Byte-exact dtypes via pyarrow inspection (§5.3), incl. nullability.
    exp_img = [('image_id', pa.uint64(), False), ('drive', pa.uint8(), False),
               ('status', pa.uint8(), False), ('n_det', pa.uint16(), False),
               ('max_conf', pa.float32(), True),
               ('orig_w', pa.uint16(), False), ('orig_h', pa.uint16(), False),
               ('reduce', pa.uint8(), False), ('guards', pa.uint16(), False),
               ('ts_off', pa.uint32(), False), ('run_id', pa.uint16(), False),
               ('shard_idx', pa.uint32(), False),
               ('model_sha8', pa.string(), True)]
    exp_det = [('image_id', pa.uint64(), False),
               ('det_idx', pa.uint8(), False), ('conf', pa.float32(), False),
               ('x1', pa.float32(), False), ('y1', pa.float32(), False),
               ('x2', pa.float32(), False), ('y2', pa.float32(), False),
               ('run_id', pa.uint16(), False),
               ('shard_idx', pa.uint32(), False),
               ('leash_class', pa.uint8(), True),
               ('leash_conf', pa.float32(), True),
               ('model_sha8', pa.string(), True)]
    for tbl, exp in ((img, exp_img), (det, exp_det)):
        got = [(f.name, f.type, f.nullable) for f in tbl.schema]
        assert got == exp, f'schema mismatch:\n got {got}\n exp {exp}'

    # Encodings + compression from the parquet metadata (§5.3).
    for path, delta_cols in ((img_path, {'image_id',
                                         'ts_off'}), (det_path, {'image_id'})):
        md = pq.ParquetFile(path).metadata
        assert md.num_row_groups >= 1
        rg = md.row_group(0)
        assert rg.num_rows <= store.ROW_GROUP_SIZE
        for ci in range(rg.num_columns):
            col = rg.column(ci)
            name = col.path_in_schema
            assert col.compression == 'ZSTD', (name, col.compression)
            encs = set(col.encodings)
            assert 'PLAIN_DICTIONARY' not in encs and 'RLE_DICTIONARY' \
                not in encs, (name, encs)  # use_dictionary=False
            if name in delta_cols:
                assert 'DELTA_BINARY_PACKED' in encs, (name, encs)
    # Rows sorted by image_id within the part (§5.3).
    col = img.column('image_id').to_pylist()
    assert col == sorted(col)
    dcol = det.column('image_id').to_pylist()
    assert dcol == sorted(dcol)


def test_nan_maps_to_null():
    # The part written above used NaN sentinels for negatives; stored value
    # must be SQL NULL so 'max_conf >= t' filters them (§5.4).
    imgs = pq.read_table(
        os.path.join(store.pair_dir(ROOT, GEN, REGION, CELL, DRIVE),
                     store.part_basename(0, 0, 60, 'img')))
    n_det = imgs.column('n_det').to_pylist()
    max_conf = imgs.column('max_conf').to_pylist()
    assert any(n == 0 for n in n_det)
    for n, mc in zip(n_det, max_conf):
        if n == 0:
            assert mc is None, f'negative stored max_conf={mc}, want NULL'
        else:
            assert mc is not None and not math.isnan(mc)
    # §5.4 threshold query returns the NULL-based answer: only genuinely
    # positive images pass the filter, no NaN>=t=TRUE inflation.
    img_src = store._sql_src(store._store_globs(ROOT, 'img'))
    res = store._run_queries({
        'thresh':
        f'SELECT count(*) FROM {img_src} WHERE max_conf >= 0.25',
        'pos':
        f'SELECT count(*) FROM {img_src} WHERE n_det > 0',
    })
    assert res['thresh'][0][0] == res['pos'][0][0], res


def test_partial_commit_continuation_and_tiling_resume():
    shard_len = 100
    w = writer()
    # Partial commit (graceful stop at 60), then continuation (§5.2 lines
    # s00008.p000000_001536 / p001536_004000 in miniature).
    sw = w.open_shard(GEN, REGION, CELL, DRIVE, 1, 0, shard_len)
    ids = [BASE + 1000 + i for i in range(shard_len)]
    fill_shard(sw, ids[:60])
    sw.commit(60)
    assert sw.cur_start == 60
    fill_shard(sw, ids[60:])
    sw.commit(100)

    pdir = store.pair_dir(ROOT, GEN, REGION, CELL, DRIVE)
    # SIGKILL-style torn .tmp left behind mid-write: garbage content at a
    # .tmp name. Must be invisible to resume/queries and cleaned by repair.
    torn = os.path.join(pdir, 's00001.p000100_000200.img.parquet.tmp')
    with open(torn, 'wb') as f:
        f.write(b'PAR1garbage-not-a-footer')
    parts, done = store.tiling_resume(pdir, shard_len, shard_idx=1)
    assert parts == [(0, 60), (60, 100)], parts
    assert done
    assert not os.path.exists(torn), 'repair should remove torn .tmp'
    # Missing tail -> not done, no gaps invented.
    parts, done = store.tiling_resume(pdir, 120, shard_idx=1)
    assert parts == [(0, 60), (60, 100)] and not done
    # _state.json is a listdir-reconstructible index over exactly these
    # parts (§6.3).
    import json
    state = json.load(open(os.path.join(pdir, store._STATE_NAME)))
    assert state['shards']['1']['parts'] == [[0, 60], [60, 100]], state


def test_tiling_rejects_bad_footer_and_orphans():
    pdir = store.pair_dir(ROOT, GEN, REGION, CELL, DRIVE)
    # A part whose img footer num_rows contradicts its declared range: the
    # "adopt a 1,536-row file as a 4,000-image shard" bug (§5.2). Fake it
    # by copying the 60-row img/det pair of shard 0 to a name claiming 50.
    for kind in ('img', 'det'):
        shutil.copyfile(
            os.path.join(pdir, store.part_basename(0, 0, 60, kind)),
            os.path.join(pdir, store.part_basename(2, 0, 50, kind)))
    parts, done = store.tiling_resume(pdir, 50, shard_idx=2)
    assert parts == [] and not done
    assert not os.path.exists(
        os.path.join(pdir, store.part_basename(2, 0, 50, 'img')))
    # Crash between det and img rename (§6.3 step 1 vs 2): det-only orphan
    # is uncommitted -> deleted, range redone.
    shutil.copyfile(os.path.join(pdir, store.part_basename(0, 0, 60, 'det')),
                    os.path.join(pdir, store.part_basename(3, 0, 60, 'det')))
    parts, done = store.tiling_resume(pdir, 60, shard_idx=3)
    assert parts == [] and not done
    assert not os.path.exists(
        os.path.join(pdir, store.part_basename(3, 0, 60, 'det')))


def test_commit_validation():
    w = writer()

    def expect(exc, fn):
        try:
            fn()
        except exc:
            return
        raise AssertionError(f'{fn} did not raise {exc.__name__}')

    sw = w.open_shard(GEN, REGION, CELL, DRIVE, 9, 0, 10)
    expect(store.CommitError, lambda: sw.add_image({
        **img_row(1), 'status': None
    }))
    expect(store.CommitError, lambda: sw.add_image({
        **img_row(2), 'n_det': None
    }))
    # NaN with n_det>0 is corrupt, not mappable (§5.4).
    expect(store.CommitError,
           lambda: sw.add_image(img_row(3, n_det=2, max_conf=float('nan'))))
    # non-NULL max_conf on a negative is equally corrupt.
    expect(store.CommitError,
           lambda: sw.add_image(img_row(4, n_det=0, max_conf=0.3)))
    sw.add_image(img_row(5, n_det=1, max_conf=0.5))
    expect(store.CommitError, lambda: sw.add_image(img_row(5)))  # dup id
    # n_det=1 declared but 0 det rows buffered -> the local det<->img check.
    sw2 = w.open_shard(GEN, REGION, CELL, DRIVE, 9, 0, 1)
    sw2.add_image(img_row(6, n_det=1, max_conf=0.5))
    expect(store.CommitError, lambda: sw2.commit(1))
    # Wrong row count for the declared positional range (§6.1).
    sw3 = w.open_shard(GEN, REGION, CELL, DRIVE, 9, 0, 5)
    sw3.add_image(img_row(7, n_det=0))
    expect(store.CommitError, lambda: sw3.commit(5))
    # Refuse overwrite of a committed part (§5.2 immutability).
    sw4 = w.open_shard(GEN, REGION, CELL, DRIVE, 0, 0, 60)
    for i in range(60):
        sw4.add_image(img_row(BASE + 900_000 + i))
    expect(store.CommitError, lambda: sw4.commit(60))


def test_guards():
    # Hard abort: cumulative det rows past the ceiling (§5.5).
    w = writer()
    w.seed_det_rows(store.DET_ROWS_MAX)
    sw = w.open_shard(GEN, REGION, CELL, DRIVE, 4, 0, 1)
    sw.add_image(img_row(BASE + 1, n_det=1, max_conf=0.9))
    sw.add_detections(det_rows(BASE + 1, 1))
    try:
        sw.commit(1)
        raise AssertionError('det-row ceiling did not abort')
    except store.StoreGuardAbort:
        pass
    assert not os.path.exists(
        os.path.join(store.pair_dir(ROOT, GEN, REGION, CELL, DRIVE),
                     store.part_basename(4, 0, 1, 'img')))
    # Hard abort: free space below the floor (floor raised above reality).
    w2 = writer(free_min_bytes=10**18)
    sw = w2.open_shard(GEN, REGION, CELL, DRIVE, 4, 0, 1)
    sw.add_image(img_row(BASE + 2))
    try:
        sw.commit(1)
        raise AssertionError('free-space floor did not abort')
    except store.StoreGuardAbort:
        pass
    # Soft alarm: trailing boxes/img > 0.60 over the (shrunk) window.
    w3 = writer(soft_window=10, soft_ratio=store.SOFT_BOXES_PER_IMG)
    sw = w3.open_shard(GEN, REGION, CELL, DRIVE, 5, 0, 10)
    for i in range(10):
        iid = BASE + 100_000 + i
        sw.add_image(img_row(iid, n_det=2, max_conf=0.8))
        sw.add_detections(det_rows(iid, 2))
    sw.commit(10)
    assert w3.soft_alarm, 'soft alarm should fire at 2.0 boxes/img'
    # Flag n_det == max_det (§5.5) as a guards bit.
    w4 = writer(max_det=3)
    sw = w4.open_shard(GEN, REGION, CELL, DRIVE, 6, 0, 1)
    iid = BASE + 200_000
    sw.add_image(img_row(iid, n_det=3, max_conf=0.7))
    sw.add_detections(det_rows(iid, 3))
    img_path, _ = sw.commit(1)
    g = pq.read_table(img_path).column('guards').to_pylist()[0]
    assert g & store.GUARD_NDET_MAXED, hex(g)


def test_invariants_pass_and_catch():
    prog = store.invariants(ROOT)  # three §5.4 assertions inside
    assert prog['scanned'] == prog['positive'] + prog['negative'] + \
        prog['errored']
    assert prog['scanned'] > 0 and prog['boxes'] > 0
    # Now inject a duplicate image_id (copy a committed pair to a new part
    # name in another shard) and require the invariants to catch it.
    pdir = store.pair_dir(ROOT, GEN, REGION, CELL, DRIVE)
    for kind in ('img', 'det'):
        shutil.copyfile(
            os.path.join(pdir, store.part_basename(0, 0, 60, kind)),
            os.path.join(pdir, store.part_basename(7, 0, 60, kind)))
    try:
        store.invariants(ROOT)
        raise AssertionError('duplicate image_ids not caught')
    except store.InvariantError as exc:
        assert 'dup_image_ids' in str(exc)
    finally:
        for kind in ('img', 'det'):
            os.remove(os.path.join(pdir, store.part_basename(7, 0, 60, kind)))
    store.invariants(ROOT)  # clean again


def test_verify_catches_truncation():
    n_ok, bad = store.verify(ROOT)
    assert bad == [], bad
    assert n_ok >= 6  # bootstrap pair + committed parts
    # Truncate a COMMITTED det parquet by 200 bytes -- unreadable and
    # glob-poisoning per §5.6 -- and require verify() to catch it.
    pdir = store.pair_dir(ROOT, GEN, REGION, CELL, DRIVE)
    victim = os.path.join(pdir, store.part_basename(0, 0, 60, 'det'))
    orig = open(victim, 'rb').read()
    assert len(orig) > 200
    try:
        with open(victim, 'wb') as f:
            f.write(orig[:-200])
        n_ok2, bad = store.verify(ROOT)
        assert [p for p, _ in bad] == [victim], bad
        assert n_ok2 == n_ok - 1
    finally:
        with open(victim, 'wb') as f:
            f.write(orig)
    _, bad = store.verify(ROOT)
    assert bad == []


def test_errors_table():
    w = writer()
    rows = [{
        'image_id': BASE + 5,
        'status': store.STATUS_READ_ERROR,
        'drive': 1,
        'path': '/x/y.jpg',
        'exc_type': 'OSError',
        'msg': 'boom',
        'ts_off': 12,
        'run_id': 1,
    }]
    p = w.write_errors(GEN, REGION, CELL, DRIVE, 0, rows)
    p2 = w.write_errors(GEN, REGION, CELL, DRIVE, 0,
                        [{
                            **rows[0], 'image_id': BASE + 4
                        }])
    assert p == p2
    t = pq.read_table(p)
    assert [(f.name, f.type) for f in t.schema] == \
        [(f.name, f.type) for f in store.ERR_SCHEMA]
    assert t.num_rows == 2  # merged across the two shard-part flushes
    assert t.column('image_id').to_pylist() == [BASE + 4, BASE + 5]


def test_compact():
    # Fresh cell across two drives, one committed shard each (§5.7).
    cell = 'Asia_70_20_75_25'
    w = writer()
    tot_img = tot_det = 0
    for d, drive in enumerate(('lynx', 'bobcat')):
        sw = w.open_shard(GEN, 'Asia', cell, drive, 0, 0, 20)
        for i in range(20):
            iid = BASE + 500_000 + d * 1000 + i
            if i % 4 == 0:
                sw.add_image(img_row(iid, n_det=1, max_conf=0.6))
                sw.add_detections(det_rows(iid, 1))
                tot_det += 1
            else:
                sw.add_image(img_row(iid))
            tot_img += 1
        sw.commit(20)
    out_img, out_det = store.compact(GEN, cell, ROOT)
    assert pq.ParquetFile(out_img).metadata.num_rows == tot_img
    assert pq.ParquetFile(out_det).metadata.num_rows == tot_det
    ids = pq.read_table(out_img).column('image_id').to_pylist()
    assert ids == sorted(ids)
    # Parts + sidecars gone; store still passes invariants + verify.
    cell_dir = os.path.dirname(store.pair_dir(ROOT, GEN, 'Asia', cell, 'lynx'))
    leftovers = [
        os.path.join(dp, n) for dp, _, ns in os.walk(cell_dir) for n in ns
    ]
    assert sorted(leftovers) == sorted([out_img, out_det]), leftovers
    out2 = store.compact(GEN, cell, ROOT)  # idempotent
    assert out2 == (out_img, out_det)
    store.invariants(ROOT)
    _, bad = store.verify(ROOT)
    assert bad == []


def main():
    print(f'store self-test in {ROOT}')
    try:
        check('bootstrap_empty_store', test_bootstrap_empty_store)
        check('schema_byte_exact_and_encodings',
              test_schema_byte_exact_and_encodings)
        check('nan_maps_to_null', test_nan_maps_to_null)
        check('partial_commit_continuation_and_tiling_resume',
              test_partial_commit_continuation_and_tiling_resume)
        check('tiling_rejects_bad_footer_and_orphans',
              test_tiling_rejects_bad_footer_and_orphans)
        check('commit_validation', test_commit_validation)
        check('guards', test_guards)
        check('invariants_pass_and_catch', test_invariants_pass_and_catch)
        check('verify_catches_truncation', test_verify_catches_truncation)
        check('errors_table', test_errors_table)
        check('compact', test_compact)
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
    print(f'\n{len(PASSES)} passed, {len(FAILURES)} failed')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.exit(main())
