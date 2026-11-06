#!/usr/bin/env python3
"""Duplicate boxes, taken out at count time: NMS at IoU 0.7 over the store.

The sweep stored its boxes after NMS at IoU 0.90, on purpose -- 0.7 cannot be
undone once it has run, and 0.90 holds every box 0.7 would keep
(docs/DETECTION_RUN_STRATEGY.md, Q4: "Downstream counts must then re-apply
NMS"). Nothing ever did. So about one stored box in six is a second box on an
object that already has one, and every count taken off the boxes -- dogs on
the map, the leash split, boxes per country -- counted those objects about
1.2 times.

This module works out, once per change to the store, which boxes a greedy NMS
at IoU 0.7 would have dropped, and writes them to one small sidecar that every
count anti-joins. The store, the gate and leash ledgers, and every human
verdict keyed by (image_id, det_idx) stay exactly as they are: nothing is
deleted or renumbered, so a later change of threshold is one rebuild.

TWO SUPPRESSION SETS, because one rule cannot serve both kinds of count:

  sup_gate  within the gate's call: a box is only ever dropped by a
            higher-confidence box the gate put in the same class (dog, not a
            dog, or not judged). EVERY count on the dashboard uses this one --
            boxes, dogs and leash calls alike -- so boxes >= dogs >= leashed
            nest and a rate never divides one unit by another.
  sup_box   class-agnostic, over every box: what a sweep at IoU 0.7 would
            have stored. Nothing reads it yet; it is here for a count of the
            detector's own boxes with no classifier involved.

Agnostic NMS before the gate would let a box the gate REJECTED (a person and
dog together, a bin) remove a box the gate called a dog: 32,108 gate-kept
dogs, and 20,799 frames that hold a dog would have read as dog-free. Within
the gate's call, a dog is only ever removed by another dog box covering it,
which is the duplicate the count wants gone -- and no frame loses its last
dog, because the frame's surest dog box always survives.

Greedy order is det_idx (the store writes them surest first; ties are already
broken). A box is suppressed when its IoU with a KEPT box of its group is
above IoU (strictly, as torchvision does). Cell twins and re-committed shards
store the same (image_id, det_idx) more than once with identical geometry;
they are collapsed first, or a box would suppress its own copy.

    python tools/detect/box_nms.py refresh [--force]
    python tools/detect/box_nms.py status

Outputs (both written atomically): data/dashboard/box_nms.parquet -- one row
per suppressed box: image_id UBIGINT, det_idx UTINYINT, sup_box, sup_gate,
by_box, by_gate (the det_idx that suppressed it, NULL where not) -- and
data/dashboard/box_nms.json, the meta: the signature of the det shards and
gate shards it was computed from, the rule, and the counts. A consumer that
finds the meta stale (current()) must count raw and say so; it must never
apply a suppression set computed from another store.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, 'data', 'dashboard')
GATE_DIR = os.path.join(REPO, 'data', 'gate')
PARQUET = 'box_nms.parquet'
META = 'box_nms.json'

SCHEMA = 1
IOU = 0.7
P_DOG_MIN = 0.5        # the gate's own cut, as map_layers and explore use it
DOG, NOT_DOG, UNJUDGED = 1, 0, 2
RULE = ('greedy NMS in det_idx order, suppressed when IoU > 0.7 with a kept '
        'box: sup_box over every box, sup_gate only within the gate call '
        '(dog / not a dog / not judged)')
# (image_id, det_idx) packed into one uint64 as image_id * _IDX_MUL + det_idx.
# det_idx is a uint8 in the store (a frame holds at most 256 boxes), so the
# multiplier is 256 -- not 32, which wraps a 33rd box onto the NEXT image and
# suppresses a box that is no duplicate of anything. Today's largest image id
# (5.87e16) times 256 is 1.50e19, inside uint64; greedy() refuses anything
# that is not.
_IDX_MUL = 256


def _detect_path():
    d = os.path.join(REPO, 'tools', 'detect')
    if d not in sys.path:
        sys.path.insert(0, d)


def det_root_default():
    _detect_path()
    import store as _store
    return _store.get_detect_root()


def det_files(det_root):
    """Every det shard, sorted. The bootstrap partition holds no rows and is
    left out of the signature; the query reads it like every other reader."""
    return sorted(glob.glob(os.path.join(det_root, 'shards', '**',
                                         '*.det.parquet'), recursive=True))


def gate_files(gate_dir):
    # `gate-*`, never `*`: work.parquet sits in the same directory
    return sorted(glob.glob(os.path.join(gate_dir, 'gate-*.parquet')))


def _stat_sig(paths):
    # the whole path: shard file names repeat across cells and drives
    ent = []
    for p in paths:
        try:
            st = os.stat(p)
            ent.append((p, st.st_mtime_ns, st.st_size))
        except OSError:
            ent.append((p, None, None))
    return hashlib.sha256(json.dumps(ent).encode()).hexdigest()


def signatures(det_root, gate_dir):
    """(det_sig, gate_sig, sig). sig also covers the rule and threshold, so a
    change of either rebuilds instead of reading as current."""
    d = _stat_sig(det_files(det_root))
    g = _stat_sig(gate_files(gate_dir))
    s = hashlib.sha256(json.dumps([SCHEMA, IOU, P_DOG_MIN, RULE, d, g])
                       .encode()).hexdigest()
    return d, g, s


def read_meta(out_dir=None):
    try:
        with open(os.path.join(out_dir or OUT_DIR, META)) as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError):
        return None


def _parquet_ok(path, meta):
    """The sidecar is readable whole and holds the rows its meta says. A
    truncated or half-copied parquet with an intact meta would otherwise
    read as current and then crash every consumer (or, worse, read short)."""
    try:
        import pyarrow.parquet as pq
        n = pq.read_table(path, columns=['image_id', 'det_idx']).num_rows
    except Exception:
        return False
    return isinstance(meta.get('rows'), int) and n == meta['rows']


def current(out_dir=None, det_root=None, gate_dir=None):
    """The meta, if the sidecar was computed from the store and ledger as they
    are now -- else None, and the caller counts raw and says so."""
    out_dir = out_dir or OUT_DIR
    meta = read_meta(out_dir)
    if not meta or meta.get('schema') != SCHEMA:
        return None
    if not _parquet_ok(os.path.join(out_dir, PARQUET), meta):
        return None
    try:
        det_root = det_root or det_root_default()
    except Exception:
        return None
    _, _, sig = signatures(det_root, gate_dir or GATE_DIR)
    if meta.get('sig') != sig:
        return None
    meta = dict(meta)
    meta['path'] = os.path.join(out_dir, PARQUET)
    return meta


def suppressed_sql(meta, kind, id_type='VARCHAR'):
    """A SQL relation (image_id, det_idx) of the boxes `kind` ('gate' or
    'box') suppresses, with image_id cast to the caller's type: the ledgers
    keep it as VARCHAR, the store as UBIGINT, and a join across the two types
    matches nothing without saying so."""
    if kind not in ('gate', 'box'):
        raise ValueError(kind)
    path = meta['path'].replace("'", "''")
    return (f"(SELECT CAST(image_id AS {id_type}) AS image_id, "
            f"CAST(det_idx AS INTEGER) AS det_idx "
            f"FROM read_parquet('{path}') WHERE sup_{kind})")


def dedup_note(meta):
    """What a payload records about the suppression it applied (or None)."""
    if not meta:
        return None
    return {'iou': meta['iou'], 'rule': meta['rule'], 'sig': meta['sig'],
            'built_at': meta.get('built_at')}


def greedy(img, i, j, n_levels):
    """Greedy NMS over within-frame pairs that overlap past the threshold.

    img, i, j: one entry per pair (image_id, the surer det_idx, the other);
    only pairs above the threshold are passed. Returns (keys, sup, by): the
    boxes that appear in any pair, whether each is suppressed, and by which
    det_idx (255 where kept). A box at det_idx r is decided by the boxes
    before it, all of which are final once levels 0..r-1 are done -- so one
    vectorised pass per level is exact greedy NMS, chains included (A drops B,
    so B cannot drop C).
    """
    import numpy as np
    img = np.asarray(img, dtype=np.uint64)
    i = np.asarray(i, dtype=np.uint64)
    j = np.asarray(j, dtype=np.uint64)
    if len(img) and (int(img.max()) >= (2**64 - 1) // _IDX_MUL
                     or int(max(i.max(), j.max())) >= _IDX_MUL):
        # a key that wraps names a different box: refuse rather than drop it
        raise ValueError('an (image_id, det_idx) pair does not fit the key '
                         f'packing (multiplier {_IDX_MUL})')
    ki = img * np.uint64(_IDX_MUL) + i
    kj = img * np.uint64(_IDX_MUL) + j
    keys = np.unique(np.concatenate([ki, kj]))
    pi = np.searchsorted(keys, ki)
    pj = np.searchsorted(keys, kj)
    sup = np.zeros(len(keys), dtype=bool)
    by = np.full(len(keys), 255, dtype=np.int64)
    jj = j.astype(np.int64)
    for r in range(1, n_levels):
        sel = (jj == r) & ~sup[pi]
        if not sel.any():
            continue
        np.minimum.at(by, pj[sel], i[sel].astype(np.int64))
        sup[pj[sel]] = True
    return keys, sup, by


def build(det_root=None, gate_dir=None, out_dir=None, force=False,
          memory='6GB', threads=4):
    """Recompute the sidecar if the store, ledger or rule moved."""
    import fcntl

    t0 = time.time()
    det_root = det_root or det_root_default()
    gate_dir = gate_dir or GATE_DIR
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    def fresh(sig):
        old = read_meta(out_dir)
        return (old and old.get('sig') == sig
                and _parquet_ok(os.path.join(out_dir, PARQUET), old))

    dsig, gsig, sig = signatures(det_root, gate_dir)
    if not force and fresh(sig):
        return {'built': False, 'reason': 'signature match', 'sig': sig,
                'secs': time.time() - t0}
    # Two builders can meet: the dashboard's hourly build and the explore
    # rollup both refresh the sidecar. One lock round the whole build, and a
    # second look once it is held, so the one that waited finds the work done
    # instead of redoing it -- and neither renames the other's temp file.
    lock = open(os.path.join(out_dir, PARQUET + '.lock'), 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        dsig, gsig, sig = signatures(det_root, gate_dir)
        if not force and fresh(sig):
            return {'built': False, 'reason': 'signature match', 'sig': sig,
                    'secs': time.time() - t0}
        return _build_locked(det_root, gate_dir, out_dir, dsig, gsig, sig,
                             memory, threads, t0)
    finally:
        lock.close()


def _build_locked(det_root, gate_dir, out_dir, dsig, gsig, sig, memory,
                  threads, t0):
    import duckdb
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    _detect_path()
    import store as _store
    det = _store._sql_src(_store._store_globs(det_root, 'det'))
    gfiles = gate_files(gate_dir)
    con = duckdb.connect()
    try:
        con.execute(f'PRAGMA threads={int(threads)}')
        con.execute(f"SET memory_limit='{memory}'")
        # ONE ROW PER BOX. Cell twins and re-commits repeat (image_id,
        # det_idx) with identical geometry; collapsed here, the surest copy's
        # geometry is taken whole (never min/max per coordinate, which would
        # build a box that no row holds).
        con.execute(f"""
            CREATE TEMP TABLE b AS
            SELECT image_id, det_idx, max(conf) conf,
                   arg_max([x1, y1, x2, y2], conf) xy
            FROM {det} GROUP BY 1, 2""")
        n_boxes, = con.execute('SELECT count(*) FROM b').fetchone()
        if gfiles:
            con.execute("""
                CREATE TEMP TABLE lab AS
                SELECT TRY_CAST(image_id AS UBIGINT) image_id,
                       CAST(det_idx AS UTINYINT) det_idx,
                       max(p_dog) p
                FROM read_parquet(?) GROUP BY 1, 2""", [gfiles])
        else:
            con.execute('CREATE TEMP TABLE lab(image_id UBIGINT, '
                        'det_idx UTINYINT, p FLOAT)')
        con.execute(f"""
            CREATE TEMP TABLE m AS
            SELECT b.image_id, b.det_idx, b.xy[1] x1, b.xy[2] y1,
                   b.xy[3] x2, b.xy[4] y2,
                   CASE WHEN l.p IS NULL THEN {UNJUDGED}
                        WHEN l.p >= {P_DOG_MIN} THEN {DOG}
                        ELSE {NOT_DOG} END AS g
            FROM b LEFT JOIN lab l USING (image_id, det_idx)
            WHERE b.image_id IN (SELECT image_id FROM b
                                 GROUP BY 1 HAVING count(*) > 1)""")
        con.execute(f"""
            CREATE TEMP TABLE pr AS
            SELECT a.image_id, a.det_idx i, c.det_idx j, a.g = c.g same
            FROM m a JOIN m c
              ON a.image_id = c.image_id AND a.det_idx < c.det_idx
            WHERE (greatest(0, least(a.x2, c.x2) - greatest(a.x1, c.x1))
                   * greatest(0, least(a.y2, c.y2) - greatest(a.y1, c.y1)))
                  > {IOU} * ((a.x2 - a.x1) * (a.y2 - a.y1)
                     + (c.x2 - c.x1) * (c.y2 - c.y1)
                     - greatest(0, least(a.x2, c.x2) - greatest(a.x1, c.x1))
                       * greatest(0, least(a.y2, c.y2) - greatest(a.y1, c.y1)))
            """)
        pr = con.execute('SELECT image_id, i, j, same FROM pr').fetchnumpy()
        levels = int(con.execute('SELECT coalesce(max(det_idx), 0) + 1 '
                                 'FROM b').fetchone()[0])
        if levels > _IDX_MUL:
            raise RuntimeError(f'det_idx reaches {levels - 1}, beyond the '
                               f'key packing ({_IDX_MUL} per image)')
        img, i, j, same = (pr['image_id'], pr['i'], pr['j'],
                           np.asarray(pr['same'], dtype=bool))
        kb, sb, bb = greedy(img, i, j, levels)
        kg, sg, bg = greedy(img[same], i[same], j[same], levels)
        keys = np.union1d(kb[sb], kg[sg])
        in_b = np.isin(keys, kb[sb])
        in_g = np.isin(keys, kg[sg])
        by_b = np.full(len(keys), -1, dtype=np.int64)
        by_g = np.full(len(keys), -1, dtype=np.int64)
        if len(keys):
            pos = np.searchsorted(kb, keys[in_b])
            by_b[in_b] = bb[pos]
            pos = np.searchsorted(kg, keys[in_g])
            by_g[in_g] = bg[pos]
        tbl = pa.table({
            'image_id': pa.array(keys // np.uint64(_IDX_MUL),
                                 type=pa.uint64()),
            'det_idx': pa.array((keys % np.uint64(_IDX_MUL))
                                .astype(np.uint8), type=pa.uint8()),
            'sup_box': pa.array(in_b),
            'sup_gate': pa.array(in_g),
            'by_box': pa.array(np.where(in_b, by_b, 0).astype(np.uint8),
                               mask=~in_b, type=pa.uint8()),
            'by_gate': pa.array(np.where(in_g, by_g, 0).astype(np.uint8),
                                mask=~in_g, type=pa.uint8()),
        })
        ppath = os.path.join(out_dir, PARQUET)
        tmp = f'{ppath}.{os.getpid()}.tmp'
        try:
            pq.write_table(tbl, tmp, compression='zstd')
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        con.register('s', tbl)
        # the counts the meta reports, each taken off the same tables the
        # consumers will join, so a number here is a number on the page
        dogs, dogs_sup, frames_dog = con.execute(f"""
            SELECT count(*) FILTER (WHERE l.p >= {P_DOG_MIN}),
                   count(*) FILTER (WHERE l.p >= {P_DOG_MIN} AND s.sup_gate),
                   count(DISTINCT l.image_id) FILTER (WHERE l.p >= {P_DOG_MIN})
            FROM lab l LEFT JOIN s USING (image_id, det_idx)""").fetchone()
        frames_dog_after, = con.execute(f"""
            SELECT count(DISTINCT l.image_id) FROM lab l
            LEFT JOIN s USING (image_id, det_idx)
            WHERE l.p >= {P_DOG_MIN} AND NOT coalesce(s.sup_gate, false)
            """).fetchone()
    finally:
        con.close()
    meta = {
        'schema': SCHEMA, 'iou': IOU, 'p_dog_min': P_DOG_MIN, 'rule': RULE,
        'sig': sig, 'det_sig': dsig, 'gate_sig': gsig,
        'det_files': len(det_files(det_root)), 'gate_files': len(gfiles),
        'rows': int(len(keys)),
        'built_at': time.strftime('%Y-%m-%d %H:%M'),
        'seconds': round(time.time() - t0, 1),
        # model output throughout: the detector's boxes, the gate's calls
        'source': 'detector boxes + gate calls (model output)',
        'counts': {
            'boxes': int(n_boxes),
            # through the gate set: the box count every page shows
            'boxes_kept_gate': int(n_boxes - int(in_g.sum())),
            # class-agnostic: what a sweep at 0.7 would have stored; no page
            # shows it
            'boxes_kept': int(n_boxes - int(in_b.sum())),
            'gate_dogs': int(dogs or 0),
            'gate_dogs_kept': int((dogs or 0) - (dogs_sup or 0)),
            'frames_with_dog': int(frames_dog or 0),
            'frames_with_dog_kept': int(frames_dog_after or 0),
            'suppressed_box': int(in_b.sum()),
            'suppressed_gate': int(in_g.sum()),
        },
    }
    os.replace(tmp, ppath)
    mpath = os.path.join(out_dir, META)
    mtmp = f'{mpath}.{os.getpid()}.tmp'
    with open(mtmp, 'w') as fh:
        json.dump(meta, fh, indent=1)
    os.replace(mtmp, mpath)
    return {'built': True, 'reason': 'rebuilt', 'sig': sig,
            'secs': time.time() - t0, 'counts': meta['counts']}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=('refresh', 'status'))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--det-root', default=None)
    ap.add_argument('--gate-dir', default=None)
    ap.add_argument('--out-dir', default=None)
    a = ap.parse_args(argv)
    if a.cmd == 'status':
        meta = current(a.out_dir, a.det_root, a.gate_dir)
        if meta is None:
            print('box_nms: no current sidecar (counts will be raw)')
            return 1
        print(json.dumps(meta['counts'], indent=1))
        return 0
    info = build(a.det_root, a.gate_dir, a.out_dir, force=a.force)
    if not info['built']:
        print(f"box_nms: skipped ({info['reason']}) in {info['secs']:.2f}s")
        return 0
    c = info['counts']
    print(f"box_nms: built in {info['secs']:.1f}s -- boxes {c['boxes']:,} -> "
          f"{c['boxes_kept_gate']:,} (class-agnostic {c['boxes_kept']:,}); "
          f"gate dogs {c['gate_dogs']:,} -> "
          f"{c['gate_dogs_kept']:,}; frames with a dog "
          f"{c['frames_with_dog']:,} -> {c['frames_with_dog_kept']:,}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
