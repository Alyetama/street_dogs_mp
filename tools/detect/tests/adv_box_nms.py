#!/usr/bin/env python3
"""Duplicate boxes taken out at count time must be exactly greedy NMS, within
the gate's call, and every count must go through the same set -- or say it
did not.

tools/detect/box_nms.py writes the boxes an NMS at IoU 0.7 drops; the map
layers and the explore rollup anti-join them. Each way that goes quietly
wrong is checked against a store small enough to work out by hand:

  * the rule: a chain (A drops B, B overlaps C) keeps C; IoU exactly 0.7 is
    kept (strict, as torchvision); a box the gate called a dog is never
    dropped by a box it did not; cell twins never suppress their own copy;
    the suppressor recorded is the kept box that did it.
  * the meta: its counts are the counts a consumer will get, and no frame
    that holds a dog loses its last one.
  * freshness: a second build skips; a gate shard that moves, a missing
    parquet, an older schema or a different threshold all make the sidecar
    not current -- and a consumer handed a stale one must count raw.
  * the consumers: map_layers drops the same boxes from the dog and the leash
    counts, records the rule under 'dedup', counts raw with 'dedup' null
    when there is no current sidecar, and rebuilds (not skips) when one
    appears. The dashboard runs the sidecar before the layers and its map
    captions say which kind of count they show.

Standalone: python tools/detect/tests/adv_box_nms.py
"""

import json
import os
import re
import shutil
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))

import duckdb  # noqa: E402

import box_nms  # noqa: E402
import map_layers as ml  # noqa: E402

fails = []


def check(name, ok, detail=''):
    print(('ok   ' if ok else 'FAIL ') + name
          + (('  ' + str(detail)) if detail and not ok else ''))
    if not ok:
        fails.append(name)


# (image_id, det_idx, conf, x1, y1, x2, y2). det_idx order is conf order.
BOXES = [
    # 101: a chain. IoU(0,1)=0.8, IoU(1,2)=0.8, IoU(0,2)=0.6 -> greedy keeps
    # 0 and 2; "drop anything overlapping a surer box" would drop 2 as well
    (101, 0, 0.9, 0, 0, 10, 10), (101, 1, 0.8, 1, 0, 11, 10),
    (101, 2, 0.7, 2.5, 0, 12.5, 10),
    # 102: IoU exactly 0.7 (inter 70, union 100) -> kept
    (102, 0, 0.9, 0, 0, 10, 10), (102, 1, 0.8, 0, 0, 10, 7),
    # 103: a dog box over a box the gate rejected (IoU 0.9): agnostic NMS
    # drops the rejected one, the gate set drops nothing
    (103, 0, 0.9, 0, 0, 10, 10), (103, 1, 0.8, 0, 0, 10, 9),
    # 104: the REJECTED box is the surer one; the dog box under it must
    # survive the gate set (the frame keeps its dog) and fall to agnostic
    (104, 0, 0.9, 0, 0, 10, 10), (104, 1, 0.8, 0, 0, 10, 9),
    # 105: three dogs, 0 and 2 overlap (IoU 0.9), 1 elsewhere -> 2 drops,
    # suppressed BY 0
    (105, 0, 0.9, 0, 0, 10, 10), (105, 1, 0.8, 50, 50, 60, 60),
    (105, 2, 0.7, 0, 0, 10, 9),
    # 106: a box the gate never judged under a dog box -> different group
    (106, 0, 0.9, 0, 0, 10, 10), (106, 1, 0.8, 0, 0, 10, 9),
    # 107: one box, stored under TWO cells (a cell twin): never suppressed
    (107, 0, 0.9, 0, 0, 10, 10),
    # 108: two dogs side by side, no overlap -> both kept
    (108, 0, 0.9, 0, 0, 10, 10), (108, 1, 0.8, 20, 0, 30, 10),
    # 109: the 101 chain again, but its head is a box the gate REJECTED:
    # within the gate call the two dogs are their own chain (2 drops, by 1);
    # agnostic drops 1 (by 0) and so keeps 2. "Agnostic NMS, then keep only
    # same-class suppressions" gets this frame wrong both ways.
    (109, 0, 0.9, 0, 0, 10, 10), (109, 1, 0.8, 1, 0, 11, 10),
    (109, 2, 0.7, 2.5, 0, 12.5, 10),
    # 110: an unjudged box over a not-dog box: two different groups, so the
    # gate set drops nothing (merging "not a dog" with "never judged" would)
    (110, 0, 0.9, 0, 0, 10, 10), (110, 1, 0.8, 0, 0, 10, 9),
    # 111: box 2 overlaps BOTH kept boxes 0 and 1 (IoU 0.82 each; 0 and 1
    # overlap 0.67 so both stay): the suppressor recorded is the surer, 0
    (111, 0, 0.9, 0, 0, 10, 10), (111, 1, 0.8, 2, 0, 12, 10),
    (111, 2, 0.7, 1, 0, 11, 10),
]
# 112: a frame of 34 boxes -- det_idx 0..32 apart, 33 a copy of 0 -- and 113,
# the NEXT image id, with two boxes apart. A key packing that wraps det_idx
# 33 onto image 113 drops 113's second box instead of 112's duplicate.
BOXES += [(112, k, round(0.9 - 0.01 * k, 3), 20 * k, 0, 20 * k + 10, 10)
          for k in range(33)]
BOXES += [(112, 33, 0.5, 0, 0, 10, 9),
          (113, 0, 0.9, 0, 0, 10, 10), (113, 1, 0.8, 50, 0, 60, 10)]
TWIN = (107, 0, 0.9, 0, 0, 10, 10)
# gate: (image_id, det_idx, p_dog); 106/1 deliberately absent
GATE = [(101, 0, 0.9), (101, 1, 0.9), (101, 2, 0.9),
        (102, 0, 0.9), (102, 1, 0.9),
        (103, 0, 0.9), (103, 1, 0.1),
        (104, 0, 0.1), (104, 1, 0.9),
        (105, 0, 0.9), (105, 1, 0.9), (105, 2, 0.9),
        (106, 0, 0.9),
        (107, 0, 0.9),
        (108, 0, 0.9), (108, 1, 0.9),
        (109, 0, 0.1), (109, 1, 0.9), (109, 2, 0.9),
        (110, 1, 0.1),                      # 110/0 never judged
        (111, 0, 0.9), (111, 1, 0.9), (111, 2, 0.9)]
GATE += [(112, k, 0.9) for k in range(34)] + [(113, 0, 0.9), (113, 1, 0.9)]
# leash rows sit on gate dogs only: (image_id, det_idx, p_leashed)
LEASH = [(101, 0, 0.9), (101, 1, 0.9), (101, 2, 0.1),
         (102, 0, 0.9), (102, 1, 0.9), (103, 0, 0.9), (104, 1, 0.1),
         (105, 0, 0.9), (105, 1, 0.1), (105, 2, 0.9), (106, 0, 0.9),
         (107, 0, 0.1), (108, 0, 0.9), (108, 1, 0.9),
         (109, 1, 0.1), (109, 2, 0.9),
         (111, 0, 0.9), (111, 1, 0.9), (111, 2, 0.1),
         (112, 0, 0.9), (112, 33, 0.9), (113, 1, 0.1)]

# worked out by hand from the geometry above
EXPECT_GATE = {(101, 1): 0, (105, 2): 0, (109, 2): 1, (111, 2): 0,
               (112, 33): 0}                         # key -> by_gate
EXPECT_BOX = {(101, 1): 0, (103, 1): 0, (104, 1): 0, (105, 2): 0,
              (106, 1): 0, (109, 1): 0, (110, 1): 0, (111, 2): 0,
              (112, 33): 0}                          # key -> by_box


def build_store(tmp):
    det = os.path.join(tmp, 'detect')
    con = duckdb.connect()
    sch = ('image_id UBIGINT, det_idx UTINYINT, conf FLOAT, x1 FLOAT, '
           'y1 FLOAT, x2 FLOAT, y2 FLOAT')

    def write(path, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        con.execute(f'CREATE OR REPLACE TEMP TABLE t({sch})')
        if rows:
            con.executemany('INSERT INTO t VALUES (?, ?, ?, ?, ?, ?, ?)',
                            rows)
        con.execute(f"COPY t TO '{path}' (FORMAT PARQUET)")

    boot = os.path.join(det, '_bootstrap', 'gen=_bootstrap',
                        'region=_bootstrap', 'cell=_bootstrap',
                        'drive=_bootstrap', 'det.parquet')
    write(boot, [])
    sh = os.path.join(det, 'shards', 'gen=0001', 'region=R')
    write(os.path.join(sh, 'cell=C1', 'drive=d1', 's00000.p0_9.det.parquet'),
          [b for b in BOXES if b[0] <= 104])
    write(os.path.join(sh, 'cell=C2', 'drive=d1', 's00000.p0_9.det.parquet'),
          [b for b in BOXES if b[0] > 104])
    write(os.path.join(sh, 'cell=C3', 'drive=d2', 's00000.p0_9.det.parquet'),
          [TWIN])
    # the image rows the explore rollup reads beside the boxes
    isch = 'image_id UBIGINT, n_det USMALLINT, max_conf FLOAT'

    def write_img(path, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        con.execute(f'CREATE OR REPLACE TEMP TABLE ti({isch})')
        if rows:
            con.executemany('INSERT INTO ti VALUES (?, ?, ?)', rows)
        con.execute(f"COPY ti TO '{path}' (FORMAT PARQUET)")

    def img_rows(boxes):
        by = {}
        for b in boxes:
            n, c = by.get(b[0], (0, 0.0))
            by[b[0]] = (n + 1, max(c, b[2]))
        return [(i, n, c) for i, (n, c) in sorted(by.items())]

    write_img(boot.replace('det.parquet', 'img.parquet'), [])
    write_img(os.path.join(sh, 'cell=C1', 'drive=d1',
                           's00000.p0_9.img.parquet'),
              img_rows([b for b in BOXES if b[0] <= 104]))
    write_img(os.path.join(sh, 'cell=C2', 'drive=d1',
                           's00000.p0_9.img.parquet'),
              img_rows([b for b in BOXES if b[0] > 104]))
    write_img(os.path.join(sh, 'cell=C3', 'drive=d2',
                           's00000.p0_9.img.parquet'), img_rows([TWIN]))
    gate = os.path.join(tmp, 'gate')
    leash = os.path.join(tmp, 'leash')
    os.makedirs(gate)
    os.makedirs(leash)
    con.execute('CREATE TEMP TABLE g(image_id VARCHAR, det_idx INTEGER, '
                'label VARCHAR, p_dog FLOAT, model VARCHAR)')
    con.executemany('INSERT INTO g VALUES (?, ?, ?, ?, ?)',
                    [(str(i), d, 'dog' if p >= 0.5 else 'not_dog', p,
                      'dogbin_test') for i, d, p in GATE])
    con.execute(f"COPY (SELECT * FROM g WHERE image_id < '104') TO "
                f"'{os.path.join(gate, 'gate-00000.parquet')}' "
                f"(FORMAT PARQUET)")
    con.execute(f"COPY (SELECT * FROM g WHERE image_id >= '104') TO "
                f"'{os.path.join(gate, 'gate-00001.parquet')}' "
                f"(FORMAT PARQUET)")
    # a worklist beside the shards, with another schema, as production has
    con.execute(f"COPY (SELECT 1 AS x) TO "
                f"'{os.path.join(gate, 'work.parquet')}' (FORMAT PARQUET)")
    con.execute('CREATE TEMP TABLE le(image_id VARCHAR, det_idx INTEGER, '
                'p_leashed FLOAT, model VARCHAR)')
    con.executemany('INSERT INTO le VALUES (?, ?, ?, ?)',
                    [(str(i), d, p, 'leash_test') for i, d, p in LEASH])
    con.execute(f"COPY le TO '{os.path.join(leash, 'leash-00000.parquet')}'"
                f" (FORMAT PARQUET)")
    # harvest manifests + catalog snapshot for map_layers: every image at
    # one point, so located == total
    man = os.path.join(tmp, 'grid_runs', 'm1.parquet')
    os.makedirs(os.path.dirname(man))
    con.execute('CREATE TEMP TABLE m(image_id BIGINT, '
                'computed_geometry VARCHAR)')
    con.executemany('INSERT INTO m VALUES (?, ?)',
                    [(i, json.dumps({'type': 'Point',
                                     'coordinates': [0.1, 0.1]}))
                     for i in sorted({b[0] for b in BOXES})])
    con.execute(f"COPY m TO '{man}' (FORMAT PARQUET)")
    st = os.stat(man)
    snap = os.path.join(tmp, 'catalog.parquet')
    con.execute('CREATE TEMP TABLE snap(path VARCHAR, region VARCHAR, '
                'kind VARCHAR, mtime DOUBLE, size_bytes BIGINT)')
    con.execute('INSERT INTO snap VALUES (?, ?, ?, ?, ?)',
                [man, 'Testland', 'ground_animals', float(st.st_mtime),
                 int(st.st_size)])
    con.execute(f"COPY snap TO '{snap}' (FORMAT PARQUET)")
    con.close()
    return dict(det_root=det, gate_dir=gate, leash_dir=leash, snapshot=snap,
                nms_dir=os.path.join(tmp, 'nms'))


def read_sup(nms_dir):
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT image_id, det_idx, sup_box, sup_gate, by_box, by_gate "
        f"FROM read_parquet('{os.path.join(nms_dir, box_nms.PARQUET)}')"
    ).fetchall()
    con.close()
    return rows


def rule_checks(fx):
    try:
        info = box_nms.build(fx['det_root'], fx['gate_dir'], fx['nms_dir'],
                             force=True)
    except Exception as e:
        # a fixture frame of 34 boxes is legal; a build that cannot hold it
        # has to be fixed, and says so here rather than as a traceback
        info = {'built': False, 'error': repr(e)}
    check('b0 a forced build builds (a 34-box frame included)',
          info.get('built'), info)
    if not info.get('built'):
        return False
    rows = read_sup(fx['nms_dir'])
    gate = {(int(r[0]), int(r[1])): r[5] for r in rows if r[3]}
    box = {(int(r[0]), int(r[1])): r[4] for r in rows if r[2]}
    check('b1 greedy NMS within the gate call drops exactly the duplicate '
          'dog boxes (chain keeps its third box; IoU 0.7 exactly is kept; '
          'twins and side-by-side dogs untouched)',
          set(gate) == set(EXPECT_GATE),
          f'got {sorted(gate)} want {sorted(EXPECT_GATE)}')
    check('b1b a box the gate called a dog is never dropped by one it did '
          'not call a dog (or never judged), and a not-dog box is never '
          'dropped by an unjudged one',
          (104, 1) not in gate and (106, 1) not in gate
          and (103, 1) not in gate and (110, 1) not in gate, sorted(gate))
    check('b1c a dog chain headed by a rejected box is NMS within the gate '
          'call, not agnostic NMS filtered afterwards',
          gate.get((109, 2)) == 1 and (109, 1) not in gate
          and box.get((109, 1)) == 0 and (109, 2) not in box,
          (gate.get((109, 2)), box.get((109, 1))))
    check('b1d a frame of 34 boxes drops its own duplicate (det_idx 33), '
          'never a box of the next image',
          (112, 33) in gate and (113, 1) not in gate and (113, 1) not in box,
          sorted(k for k in gate if k[0] in (112, 113)))
    check('b2 class-agnostic NMS drops every overlapping lower box, across '
          'gate calls', set(box) == set(EXPECT_BOX),
          f'got {sorted(box)} want {sorted(EXPECT_BOX)}')
    check('b3 the suppressor recorded is the kept box that did it',
          all(gate.get(k) == v for k, v in EXPECT_GATE.items())
          and all(box.get(k) == v for k, v in EXPECT_BOX.items()),
          f'gate {gate} box {box}')
    # whichever order the pairs arrive in, two kept suppressors -> the surer
    try:
        a = box_nms.greedy([7, 7], [0, 1], [2, 2], 3)
        b = box_nms.greedy([7, 7], [1, 0], [2, 2], 3)
        by_a = dict(zip(a[0].tolist(), a[2].tolist()))
        by_b = dict(zip(b[0].tolist(), b[2].tolist()))
        k2 = 7 * box_nms._IDX_MUL + 2
        ok = by_a.get(k2) == 0 and by_b.get(k2) == 0
    except Exception as e:
        ok, by_a, by_b = False, repr(e), ''
    check('b3b with two kept suppressors the surer one is recorded, in any '
          'pair order', ok, (by_a, by_b))
    try:
        box_nms.greedy([7], [0], [box_nms._IDX_MUL], box_nms._IDX_MUL + 1)
        refused = False
    except ValueError:
        refused = True
    check('b3c a det_idx the key packing cannot hold is refused, not wrapped',
          refused)
    check('b4 a cell twin never suppresses its own copy',
          (107, 0) not in gate and (107, 0) not in box)
    meta = box_nms.read_meta(fx['nms_dir'])
    c = (meta or {}).get('counts', {})
    n_boxes = len({(b[0], b[1]) for b in BOXES})
    dogs = sum(1 for g in GATE if g[2] >= 0.5)
    frames = len({g[0] for g in GATE if g[2] >= 0.5})
    check('b5 meta counts are the counts a consumer gets',
          c.get('boxes') == n_boxes
          and c.get('boxes_kept') == n_boxes - len(EXPECT_BOX)
          and c.get('boxes_kept_gate') == n_boxes - len(EXPECT_GATE)
          and c.get('gate_dogs') == dogs
          and c.get('gate_dogs_kept') == dogs - len(EXPECT_GATE)
          and c.get('suppressed_gate') == len(EXPECT_GATE)
          and c.get('suppressed_box') == len(EXPECT_BOX), c)
    check('b6 no frame that holds a dog loses its last one',
          c.get('frames_with_dog') == frames
          and c.get('frames_with_dog_kept') == frames, c)


def fresh_checks(fx):
    cur = box_nms.current(fx['nms_dir'], fx['det_root'], fx['gate_dir'])
    check('f1 a sidecar built from this store and ledger is current',
          cur is not None)
    again = box_nms.build(fx['det_root'], fx['gate_dir'], fx['nms_dir'])
    check('f2 a second build skips when nothing moved',
          not again.get('built') and again.get('reason') == 'signature match',
          again)
    # a gate shard moving (a re-run of one shard) makes it stale
    g0 = os.path.join(fx['gate_dir'], 'gate-00000.parquet')
    st = os.stat(g0)
    os.utime(g0, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    check('f3 a gate shard that moved makes the sidecar NOT current',
          box_nms.current(fx['nms_dir'], fx['det_root'], fx['gate_dir'])
          is None)
    reb = box_nms.build(fx['det_root'], fx['gate_dir'], fx['nms_dir'])
    check('f4 ... and the next build rebuilds', reb.get('built'), reb)
    # a det shard moving does the same
    d0 = sorted(box_nms.det_files(fx['det_root']))[0]
    st = os.stat(d0)
    os.utime(d0, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    check('f5 a det shard that moved makes the sidecar NOT current',
          box_nms.current(fx['nms_dir'], fx['det_root'], fx['gate_dir'])
          is None)
    box_nms.build(fx['det_root'], fx['gate_dir'], fx['nms_dir'])
    # another threshold is another sidecar
    saved = box_nms.IOU
    try:
        box_nms.IOU = 0.5
        check('f6 a different IoU makes the sidecar NOT current',
              box_nms.current(fx['nms_dir'], fx['det_root'], fx['gate_dir'])
              is None)
    finally:
        box_nms.IOU = saved
    p = os.path.join(fx['nms_dir'], box_nms.PARQUET)
    shutil.move(p, p + '.away')
    check('f7 a missing parquet makes the sidecar NOT current',
          box_nms.current(fx['nms_dir'], fx['det_root'], fx['gate_dir'])
          is None)
    shutil.move(p + '.away', p)
    mp = os.path.join(fx['nms_dir'], box_nms.META)
    doc = json.load(open(mp))
    doc['schema'] = box_nms.SCHEMA + 1
    json.dump(doc, open(mp, 'w'))
    check('f8 a meta of another schema is NOT current',
          box_nms.current(fx['nms_dir'], fx['det_root'], fx['gate_dir'])
          is None)
    box_nms.build(fx['det_root'], fx['gate_dir'], fx['nms_dir'], force=True)
    # a damaged parquet under an intact meta: not current, and rebuilt
    p = os.path.join(fx['nms_dir'], box_nms.PARQUET)
    raw = open(p, 'rb').read()
    open(p, 'wb').write(raw[:len(raw) // 2])
    check('f10 a truncated sidecar under an intact meta is NOT current '
          '(consumers count raw instead of crashing)',
          box_nms.current(fx['nms_dir'], fx['det_root'], fx['gate_dir'])
          is None)
    reb = box_nms.build(fx['det_root'], fx['gate_dir'], fx['nms_dir'])
    check('f11 ... and the next build rebuilds it rather than skipping',
          reb.get('built') and box_nms.current(
              fx['nms_dir'], fx['det_root'], fx['gate_dir']) is not None, reb)
    # two builders at once (the dashboard and the explore rollup)
    import threading
    errs = []

    def run():
        try:
            box_nms.build(fx['det_root'], fx['gate_dir'], fx['nms_dir'],
                          force=True)
        except Exception as e:
            errs.append(repr(e))
    ts = [threading.Thread(target=run) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    left = [f for f in os.listdir(fx['nms_dir']) if f.endswith('.tmp')]
    check('f12 builders running at once neither crash nor leave temp files, '
          'and the result is current',
          not errs and not left and box_nms.current(
              fx['nms_dir'], fx['det_root'], fx['gate_dir']) is not None,
          (errs, left))
    cur = box_nms.current(fx['nms_dir'], fx['det_root'], fx['gate_dir'])
    con = duckdb.connect()
    n = con.execute(
        f"SELECT count(*) FROM read_parquet('{fx['gate_dir']}/gate-*.parquet')"
        f" g ANTI JOIN {box_nms.suppressed_sql(cur, 'gate')} s "
        f"ON s.image_id = g.image_id AND s.det_idx = g.det_idx").fetchone()[0]
    con.close()
    check('f9 the relation joins the ledgers\' VARCHAR ids (a type mismatch '
          'would match nothing and drop nothing)',
          n == len(GATE) - len(EXPECT_GATE), n)


def layer_checks(fx, tmp):
    out = os.path.join(tmp, 'out')
    kw = dict(gate_dir=fx['gate_dir'], leash_dir=fx['leash_dir'],
              snapshot=fx['snapshot'], out_dir=out,
              hide_regions=frozenset(), det_root=fx['det_root'])
    raw = ml.refresh(force=True, nms_dir=os.path.join(tmp, 'no_sidecar'),
                     **kw)
    dogs = json.load(open(os.path.join(out, 'map_layer_dogs.json')))
    leash = json.load(open(os.path.join(out, 'map_layer_leash.json')))
    n_dogs = sum(1 for g in GATE if g[2] >= 0.5)
    check('m1 with no current sidecar the layers count raw and say so',
          raw.get('built') and dogs['total'] == n_dogs
          and dogs.get('dedup') is None and leash.get('dedup') is None,
          (raw, dogs.get('total'), dogs.get('dedup')))
    warm = ml.refresh(nms_dir=fx['nms_dir'], **kw)
    check('m2 a sidecar appearing rebuilds the layers (the signature covers '
          'it)', warm.get('built'), warm)
    dogs = json.load(open(os.path.join(out, 'map_layer_dogs.json')))
    leash = json.load(open(os.path.join(out, 'map_layer_leash.json')))
    want_dogs = n_dogs - len(EXPECT_GATE)
    kept = [r for r in LEASH if (r[0], r[1]) not in EXPECT_GATE]
    want_l = sum(1 for r in kept if r[2] >= 0.5)
    want_u = sum(1 for r in kept if r[2] < 0.5)
    check('m3 dog crops lose exactly the gate set\'s duplicates',
          dogs['total'] == want_dogs and dogs['unlocated'] == 0,
          (dogs['total'], want_dogs, dogs['unlocated']))
    check('m4 the frame count does not move',
          dogs['images'] == len({g[0] for g in GATE if g[2] >= 0.5}),
          dogs['images'])
    check('m5 leash calls lose the same boxes (leashed and unleashed alike, '
          'so the ratio stays one unit)',
          leash['leashed_total'] == want_l and leash['loose_total'] == want_u,
          (leash['leashed_total'], want_l, leash['loose_total'], want_u))
    check('m6 both payloads record the rule they went through',
          (dogs.get('dedup') or {}).get('iou') == box_nms.IOU
          and (leash.get('dedup') or {}).get('sig')
          == box_nms.read_meta(fx['nms_dir'])['sig'],
          (dogs.get('dedup'), leash.get('dedup')))
    skip = ml.refresh(nms_dir=fx['nms_dir'], **kw)
    check('m7 ... and skips once it has', not skip.get('built')
          and skip.get('reason') == 'signature match', skip)
    g1 = os.path.join(fx['gate_dir'], 'gate-00001.parquet')
    st = os.stat(g1)
    os.utime(g1, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    stale = ml.refresh(nms_dir=fx['nms_dir'], **kw)
    dogs = json.load(open(os.path.join(out, 'map_layer_dogs.json')))
    check('m8 a sidecar gone stale is not applied: the layers fall back to '
          'raw and say so', stale.get('built') and dogs['total'] == n_dogs
          and dogs.get('dedup') is None, (stale, dogs.get('total')))


def explore_checks(fx):
    """explore_index.py is not in every checkout; where it is, its gate,
    leash and box counts must go through the same set."""
    if not os.path.exists(os.path.join(REPO, 'tools', 'dashboard',
                                       'explore_index.py')):
        print('skip explore checks (explore_index.py is not in this checkout)')
        return
    try:
        import explore_index as ei
    except Exception as e:
        check('e0 explore_index imports', False, repr(e))
        return
    box_nms.build(fx['det_root'], fx['gate_dir'], fx['nms_dir'], force=True)
    cur = box_nms.current(fx['nms_dir'], fx['det_root'], fx['gate_dir'])
    con = duckdb.connect()
    con.execute('CREATE TABLE ic(image_id UBIGINT, cc VARCHAR)')
    con.executemany('INSERT INTO ic VALUES (?, ?)',
                    [(i, 'TST') for i in sorted({b[0] for b in BOXES})])
    try:
        g, _ = ei._ledger_rollup(con, fx['gate_dir'], 'gate-*.parquet',
                                 'p_dog', 0.5, 'gate', cur, box_nms)
        lr, _ = ei._ledger_rollup(con, fx['leash_dir'], 'leash-*.parquet',
                                  'p_leashed', 0.5, 'leash', cur, box_nms)
    except Exception as e:
        check('e1 explore: the ledger rollups run with a sidecar', False,
              repr(e))
        return
    t = g.get('TST', {})
    lt = lr.get('TST', {})
    kept = [r for r in LEASH if (r[0], r[1]) not in EXPECT_GATE]
    check('e1 explore: judged and kept dogs both lose the gate set',
          t.get('gate_judged') == len(GATE) - len(EXPECT_GATE)
          and t.get('gate_kept') == sum(1 for x in GATE if x[2] >= 0.5)
          - len(EXPECT_GATE), t)
    check('e2 explore: leash judged and leashed lose the same boxes',
          lt.get('leash_judged') == len(kept)
          and lt.get('leash_kept') == sum(1 for r in kept if r[2] >= 0.5),
          lt)
    import store as _store
    saved = _store.get_detect_root
    try:
        _store.get_detect_root = lambda repo_root=None: fx['det_root']
        d = ei.detect_rollup(con, cur, box_nms)
    except Exception as e:
        check('e3 explore: detect_rollup runs on the fixture store', False,
              repr(e))
        return
    finally:
        _store.get_detect_root = saved
    n_boxes = len({(b[0], b[1]) for b in BOXES})
    row = d.get('TST', {})
    check('e3 explore: boxes are counted through the same (gate) set, so '
          'boxes >= judged >= dogs nest',
          row.get('det_boxes') == n_boxes - len(EXPECT_GATE)
          and sum(row.get('det_conf', [])) == row.get('det_boxes')
          and row.get('det_boxes', 0) >= t.get('gate_judged', 0), row)
    con.close()
    src = open(ei.__file__).read()
    b = src.find('\ndef build(')
    body = src[b:src.find('\ndef ', b + 10)]
    check('e4 explore build() refreshes the sidecar and passes it to the '
          'detector, gate and leash rollups, and records the rule',
          '_box_nms(refresh=True)' in body
          and bool(re.search(r'detect_rollup\(con, nms, bn\)', body))
          and len(re.findall(r"_ledger_rollup\([^)]*nms, bn\)", body,
                             re.S)) == 2
          and "'dedup': bn.dedup_note(nms) if nms else None" in body)
    page = open(os.path.join(REPO, 'tools', 'dashboard',
                             'explore_page.py')).read()
    check('e5 the explore page passes the rule through and its footer says '
          'which kind of count it shows',
          "'dedup': doc.get('dedup')" in page and page.count('j.dedup') >= 2)
    # stale(): a CURRENT sidecar the summary was not built through -> stale;
    # a stale or missing sidecar is not a reason to rebuild into raw
    summ = os.path.join(tempfile.mkdtemp(prefix='adv_box_nms_e_'),
                        'explore.json')
    saved = (ei.SUMMARY, ei._box_nms)
    try:
        ei.SUMMARY = summ
        json.dump({'dedup': {'sig': 'old'}}, open(summ, 'w'))
        ei._box_nms = lambda refresh=False: ({'sig': 'new'}, box_nms)
        s1 = ei.stale(3600)
        ei._box_nms = lambda refresh=False: (None, box_nms)
        s2 = ei.stale(3600)
        json.dump({'dedup': {'sig': 'new'}}, open(summ, 'w'))
        ei._box_nms = lambda refresh=False: ({'sig': 'new'}, box_nms)
        s3 = ei.stale(3600)
    finally:
        ei.SUMMARY, ei._box_nms = saved
        shutil.rmtree(os.path.dirname(summ), ignore_errors=True)
    check('e6 explore is stale for a new current sidecar, not for a stale '
          'one, and not when it already went through the current one',
          s1 is True and s2 is False and s3 is False, (s1, s2, s3))


def wiring_checks():
    src = open(os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py')).read()
    b = src.find('\ndef build(args):')
    body = src[b:src.find('\ndef ', b + 10)]
    i, j = body.find("'box_nms.py'"), body.find('ml.refresh()')
    check('w1 the dashboard build refreshes the sidecar before the map '
          'layers', 0 <= i < j
          and bool(re.search(r"subprocess\.run\(\[sys\.executable, "
                             r"nms_py, 'refresh'\]", body))
          and 0 <= body.find('nms_ok = nms_refresh()') < j, (i, j))
    check('w2 both map captions say whether duplicates were removed',
          len(re.findall(r"statsEl\.textContent=[^\n]*dedupNote\(d[gl]\)",
                         src)) == 2)


def main():
    tmp = tempfile.mkdtemp(prefix='adv_box_nms_')
    try:
        fx = build_store(tmp)
        if rule_checks(fx) is False:
            # everything after builds the sidecar again and would only
            # repeat the same failure as a traceback
            print('stopping: the sidecar does not build on the fixture')
            return 1
        fresh_checks(fx)
        layer_checks(fx, tmp)
        explore_checks(fx)
        wiring_checks()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{len(fails)} failed' if fails else '\nall passed')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
