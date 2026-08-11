#!/usr/bin/env python3
"""Rows are not detections, and a shard index is a promise. Prove both.

Four defects in the detect pipeline had one root between them: something
counted ROWS of a store that keys on (image_id, det_idx, cell, drive) and read
the answer as a count of things. A frame the harvest wrote into several cells
carries every one of its detections once per placement -- corpus-legitimate,
documented in store.unique_src(), and reported by store.invariants() as a
property that must never fail -- so a row count is 2-16x a detection count on
those frames and every consumer that does not collapse them is wrong.

  p1  harvest_flagged.boxes_for() saw one box 16 times, could not tell which
      of the 16 the reviewer had flagged, and filed the frame as ambiguous.
      117 human verdicts on the live ledgers were being dropped out of every
      dataset built afterwards.
  p2  build_detector_negatives.detections_per_image() read a one-detection
      frame as a two-detection one and excluded it from the background set
      with a printed reason about a sibling detection that does not exist.

The other two are about a name meaning the same thing twice:

  p3  gate_store cut its shards over a job list that --limit had already
      truncated, so a trial run wrote a SHORT shard under the index a full run
      gives 20,000 images, and done_shards() skipped that index forever.
  p4  build_sqldb re-INSERTed a changed part file without deleting the rows it
      had already loaded from that path, and left the rows of a part that had
      been compacted away or dropped-and-redone. A comment claimed the
      protection; nothing implemented it.

And two guards that certified what they had not measured:

  p5  rebuild_crop_dataset's straddle check intersected a set with its own
      complement -- empty for every possible input, printed "shared 0" under
      every dataset it ever built.
  p6  reserve_acceptance_set stamped every reservation with a date literal, so
      the refusal that protects an acceptance number reported a fixed date as
      the date of the reservation it was protecting.
  p7  sweep.py documented a `compact` subcommand it never registered.
  p8  build_detector_negatives excluded val by image_id and never by sequence.

Everything below runs against fixtures in a temp directory. Nothing here
reads or writes a store under data/.

Run: python tools/detect/tests/adv_detect_accounting.py
Needs duckdb + pyarrow (mp14).
"""

import ast
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DETECT = os.path.join(REPO, 'tools', 'detect')
if DETECT not in sys.path:
    sys.path.insert(0, DETECT)

fails = []


def check(name, ok, detail=''):
    print(('ok   ' if ok else 'FAIL ') + name
          + (('  ' + detail) if detail and not ok else ''))
    if not ok:
        fails.append(name)


def src_of(name):
    with open(os.path.join(DETECT, name), encoding='utf-8') as fh:
        return fh.read()


# ── a fixture store with a cell twin in it ─────────────────────────────────
def fake_store(root, placements=(('r', 'c', 'd1'), ('r', 'c2', 'd1'))):
    """A store where image 111 sits in every placement given.

    Two placements, one detection each in the parquet, is the shape the live
    store really holds: 118 of the 2,778 flagged frames have a det row count
    that differs from their distinct det_idx count.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import store

    def rows(iid, dets):
        n = len(dets)
        img = pa.table({
            'image_id': pa.array([iid] * 1, pa.uint64()),
            'drive': pa.array([1], pa.uint8()),
            'status': pa.array([1], pa.uint8()),
            'n_det': pa.array([n], pa.uint16()),
            'max_conf': pa.array([0.5], pa.float32()),
            'orig_w': pa.array([100], pa.uint16()),
            'orig_h': pa.array([100], pa.uint16()),
            'reduce': pa.array([0], pa.uint8()),
            'guards': pa.array([0], pa.uint16()),
            'ts_off': pa.array([0], pa.uint32()),
            'run_id': pa.array([1], pa.uint16()),
            'shard_idx': pa.array([0], pa.uint32()),
            'model_sha8': pa.array(['ab'])}, schema=store.IMG_SCHEMA)
        det = pa.table({
            'image_id': pa.array([iid] * n, pa.uint64()),
            'det_idx': pa.array([d for d, _ in dets], pa.uint8()),
            'conf': pa.array([c for _, c in dets], pa.float32()),
            'x1': pa.array([10.0] * n, pa.float32()),
            'y1': pa.array([10.0] * n, pa.float32()),
            'x2': pa.array([90.0] * n, pa.float32()),
            'y2': pa.array([90.0] * n, pa.float32()),
            'run_id': pa.array([1] * n, pa.uint16()),
            'shard_idx': pa.array([0] * n, pa.uint32()),
            'leash_class': pa.array([None] * n, pa.uint8()),
            'leash_conf': pa.array([None] * n, pa.float32()),
            'model_sha8': pa.array(['ab'] * n)}, schema=store.DET_SCHEMA)
        return img, det

    boot = os.path.join(root, '_bootstrap', 'gen=0001', 'region=r',
                        'cell=boot', 'drive=d1')
    os.makedirs(boot, exist_ok=True)
    img, det = rows(999, [(0, 0.9)])
    pq.write_table(img, os.path.join(boot, 'img.parquet'))
    pq.write_table(det, os.path.join(boot, 'det.parquet'))
    for region, cell, drive in placements:
        d = os.path.join(root, 'shards', 'gen=0001', f'region={region}',
                         f'cell={cell}', f'drive={drive}')
        os.makedirs(d, exist_ok=True)
        # ONE detection on this frame. Repeated per placement, which is what
        # makes a row count lie.
        img, det = rows(111, [(0, 0.42)])
        pq.write_table(img, os.path.join(d, 's00000.p000000_000001.img.parquet'))
        pq.write_table(det, os.path.join(d, 's00000.p000000_000001.det.parquet'))
        # and a genuinely two-detection frame, to prove the fix does not
        # simply collapse everything to one
        img, det = rows(222, [(0, 0.30), (1, 0.70)])
        pq.write_table(img, os.path.join(d, 's00000.p000001_000002.img.parquet'))
        pq.write_table(det, os.path.join(d, 's00000.p000001_000002.det.parquet'))


def p1_p2(tmp):
    root = os.path.join(tmp, 'store')
    fake_store(root)
    import harvest_flagged as hf
    boxes = hf.boxes_for(['111', '222'], root)
    twin = boxes.get('111', [])
    real = boxes.get('222', [])
    check('p1 boxes_for returns one row per detection, not per cell placement',
          len(twin) == 1 and {d[0] for d in real} == {0, 1}
          and len(real) == 2,
          f'image 111 came back {len(twin)} time(s) for its single '
          f'detection, image 222 as {len(real)} row(s) for its two')
    # the whole point: a flag carrying that box's confidence must resolve to
    # exactly one box, which is what harvest_flagged's matcher requires
    hit = [d for d in twin if abs(d[5] - 0.42) <= 0.006]
    check('p1b a flag on a twinned frame resolves to exactly one box',
          len(hit) == 1,
          f'{len(hit)} boxes matched the flagged confidence -- the harvest '
          f'calls that ambiguous and drops the human verdict')

    import store
    import build_detector_negatives as bdn
    real_root = store.get_detect_root
    store.get_detect_root = lambda *a, **k: root
    try:
        per = bdn.detections_per_image({'111', '222'}, REPO)
    finally:
        store.get_detect_root = real_root
    check('p2 detections_per_image counts detections, not store rows',
          per.get('111') == 1 and per.get('222') == 2,
          f'a one-detection frame in two cells read as {per.get("111")} '
          f'detections; a two-detection frame read as {per.get("222")}')


def p3():
    import gate_store as gs
    # 165 shards of 20,000 with a 12,062 tail, which is the live gate plan
    shards = [[('i', 1, 'd', [1, 2])] * 20000 for _ in range(2)]
    shards.append([('i', 1, 'd', [1, 2])] * 12062)
    check('p3 stop_after leaves the shard boundaries alone',
          gs.stop_after(shards, set(), 0) is None
          and gs.stop_after(shards, set(), 5000) == 1
          and gs.stop_after(shards, {0}, 5000) == 2
          and gs.stop_after(shards, set(), 10 ** 9) == 3,
          f'{gs.stop_after(shards, set(), 5000)!r} / '
          f'{gs.stop_after(shards, {0}, 5000)!r}')
    short = gs.refuse_short_shard(0, 3407, 165, 20000)
    check('p3b a short shard under a non-final index is refused',
          bool(short) and '16,593' in short,
          f'committed silently: {short!r}')
    check('p3c a full shard and the true last shard are allowed',
          gs.refuse_short_shard(0, 20000, 165, 20000) == ''
          and gs.refuse_short_shard(164, 12062, 165, 20000) == '',
          'the refusal fires on a shard that is fine')
    # p3b alone cannot hold this: a truncated work list makes jobs, shards and
    # "the last shard" short TOGETHER, and p3b then sees a short shard that is
    # the last one it knows about. Only the count recorded at plan time can
    # tell, which is what check_job_count compares against.
    check('p3d a job list shorter than the plan is refused outright',
          bool(gs.check_job_count(3407, 3292062))
          and gs.check_job_count(3292062, 3292062) == ''
          and gs.check_job_count(3407, None) == '',
          f'{gs.check_job_count(3407, 3292062)!r}')
    tree = ast.parse(src_of('gate_store.py'))
    run = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                and n.name == 'run'), None)
    called = {getattr(c.func, 'id', '') for c in ast.walk(run)
              if isinstance(c, ast.Call)} if run else set()
    check('p3e run() actually asks both questions before it writes',
          {'check_job_count', 'refuse_short_shard'} <= called,
          f'run() calls {sorted(called & {"check_job_count", "refuse_short_shard"})}'
          f' -- a protection nothing calls is not one')


def p4(tmp):
    """A refresh must not count a reloaded or vanished part twice."""
    import types
    import duckdb
    import build_sqldb as bs
    root = os.path.join(tmp, 'sqlstore')
    fake_store(root, placements=(('r', 'c', 'd1'),))
    db = os.path.join(tmp, 'refresh.duckdb')
    real_root = bs.detect_root
    bs.detect_root = lambda: root
    args = types.SimpleNamespace(db=db, memory='1GB', threads=2)

    def counts():
        con = duckdb.connect(db, read_only=True)
        try:
            return (con.execute('SELECT count(*) FROM images').fetchone()[0],
                    con.execute('SELECT count(*) FROM detections')
                    .fetchone()[0])
        finally:
            con.close()

    import contextlib
    import io as _io

    def build():
        with contextlib.redirect_stdout(_io.StringIO()):
            bs.build(args)

    try:
        build()
        first = counts()
        cell = os.path.join(root, 'shards', 'gen=0001', 'region=r',
                            'cell=c', 'drive=d1')
        # (a) tiling_resume dropped a torn part and the run redid the range:
        #     the name is deterministic on [start,end), so the SAME path comes
        #     back with different bytes
        for k in ('img', 'det'):
            p = os.path.join(cell, f's00000.p000000_000001.{k}.parquet')
            os.utime(p, (0, 0))
        build()
        reloaded = counts()
        # (b) compaction: the parts are deleted and one merged pair appears
        #     under drive=_merged, which part_files() globs like any other
        merged = os.path.join(root, 'shards', 'gen=0001', 'region=r',
                              'cell=c', 'drive=_merged')
        os.makedirs(merged, exist_ok=True)
        for k in ('img', 'det'):
            shutil.copy(os.path.join(cell, f's00000.p000000_000001.{k}.parquet'),
                        os.path.join(merged, f'compact.{k}.parquet'))
        shutil.rmtree(cell)
        build()
        compacted = counts()
    finally:
        bs.detect_root = real_root
    check('p4 a re-committed part does not double-count',
          reloaded == first, f'{first} became {reloaded}')
    # the merged pair holds only the first part's rows here, so the row count
    # is allowed to fall -- what must not happen is the deleted parts' rows
    # surviving alongside it
    check('p4b a compacted cell does not leave its parts behind',
          compacted[0] <= first[0] and compacted[1] <= first[1],
          f'{first} became {compacted} after the parts were deleted')


def p5():
    from rebuild_crop_dataset import straddling_sequences
    seq = {'a_1.jpg': 'S1', 'a_2.jpg': 'S1', 'b_1.jpg': 'S2'}
    clean = [('train', 'dog', 'p', 'a_1.jpg'), ('train', 'dog', 'p', 'a_2.jpg'),
             ('val', 'dog', 'p', 'b_1.jpg')]
    leak = [('train', 'dog', 'p', 'a_1.jpg'), ('val', 'dog', 'p', 'a_2.jpg'),
            ('val', 'dog', 'p', 'b_1.jpg')]
    _, _, ok_shared = straddling_sequences(clean, seq.get)
    _, _, bad_shared = straddling_sequences(leak, seq.get)
    check('p5 the straddle check passes a per-sequence split', not ok_shared,
          f'it found {ok_shared} in a split that does not straddle')
    check('p5b the straddle check CAN fail -- it finds a split sequence',
          bad_shared == {'S1'},
          'two crops of one sequence on opposite sides went unnoticed, which '
          'is the state the old complement-of-a-set check could not detect')


def p6():
    tree = ast.parse(src_of('reserve_acceptance_set.py'))
    lit = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for k, v in zip(node.keys, node.values):
            if (isinstance(k, ast.Constant) and k.value == 'created'
                    and isinstance(v, ast.Constant)):
                lit.append(f'line {v.lineno}: {v.value!r}')
    check("p6 a reservation's 'created' is computed, not a literal",
          not lit, 'a fixed date is stamped on every reservation ever '
          'drawn: ' + '; '.join(lit))


def p7():
    tree = ast.parse(src_of('sweep.py'))
    subs = [n.args[0].value for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, 'attr', '') == 'add_parser'
            and n.args and isinstance(n.args[0], ast.Constant)]
    check('p7 sweep registers the compact subcommand it documents',
          'compact' in subs,
          f'store.compact() is the documented post-run step and argparse '
          f'answers "invalid choice"; registered: {sorted(subs)}')
    doc = ast.get_docstring(tree) or ''
    check('p7b the usage block promises no --gen on verify/invariants',
          'invariants | compact --gen' not in doc
          and 'invariants --gen' not in doc,
          'both take no arguments -- argparse answers "unrecognized '
          'arguments: --gen"')


def p8():
    tree = ast.parse(src_of('build_detector_negatives.py'))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    keys = {k.value for n in ast.walk(tree) if isinstance(n, ast.Dict)
            for k in n.keys if isinstance(k, ast.Constant)}
    check('p8 backgrounds are excluded from val by SEQUENCE, not image_id',
          'val_seqs' in names and 'val_sequence_skipped' in keys,
          'nothing here resolves the val split\'s sequences, so a background '
          'may be frame N of a pass whose frame N+2 is a labelled val dog')


def main():
    tmp = tempfile.mkdtemp(prefix='detacct_')
    try:
        p1_p2(tmp)
        p3()
        p4(tmp)
        p5()
        p6()
        p7()
        p8()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if fails:
        print(f'\n{len(fails)} accounting check(s) FAILED: '
              + ', '.join(fails))
        return 1
    print('\nrows are collapsed to detections, shard indices mean one thing, '
          'and\nthe split guard can fail')
    return 0


if __name__ == '__main__':
    sys.exit(main())
