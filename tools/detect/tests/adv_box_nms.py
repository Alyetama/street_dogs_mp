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
  * the conf floor (count_conf_min in best_models.json, keyed on the
    store's model_sha8): a box at or under its engine's floor is out of
    both sets, the boundary included; candidates count, other projects do
    not; a box with no model_sha8 or an engine with no floor is untouched;
    the floor is in the signature, the meta and the payloads' dedup note;
    a store with no model_sha8 column (gen 1) writes the very parquet bytes
    it wrote before, under the very signature when no floor is configured.
    Each guard is put back wrong by a mutant that must be caught.

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

# DASH_PY=<copy> checks an edited copy of the watched dashboard.py instead
# of the live file (saving the live one restarts the dashboard)
DASH_PY = os.environ.get('DASH_PY') or os.path.join(
    REPO, 'tools', 'dashboard', 'dashboard.py')

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


def _js_fn(src, name, end):
    """The source of JS function `name` out of a page template."""
    i = src.find(f'function {name}(')
    j = src.find(end, i)
    return src[i:j + len(end)] if i >= 0 and j > i else None


def _node(js):
    import subprocess
    r = subprocess.run(['node', '-e', js], capture_output=True, text=True,
                       timeout=30)
    return json.loads(r.stdout) if r.returncode == 0 else {'err': r.stderr}


FLOOR_CASES = [{'dedup': {'iou': 0.7, 'count_conf_min': {'047b6268': 0.25}}},
               {'dedup': {'iou': 0.7, 'count_conf_min': {'a': 0.25,
                                                         'b': 0.5,
                                                         'c': 0.25}}},
               {'dedup': {'iou': 0.7, 'count_conf_min': {}}},
               {'dedup': {'iou': 0.7}},       # a payload from before the floor
               {'dedup': None}, None]


def caption_checks(dash_src, page_src, mlm, tag=''):
    """Every caption that reports the sidecar's counts says the conf floor
    when one was applied, and says there is none on a raw fallback -- on
    gen 2 most of what the sidecar removes is under the floor, and a page
    that called all of it 'duplicates' (or a raw page that blamed a 28%
    jump on duplicates alone) would misstate both. Returns True if all
    passed."""
    import contextlib
    import io
    n0 = len(fails)
    fn = _js_fn(dash_src, 'dedupNote', '\n    }\n')
    got = (_node(fn + 'console.log(JSON.stringify(' + json.dumps(FLOOR_CASES)
                 + '.map(dedupNote)))') if fn else {'err': 'no dedupNote'})
    ok = isinstance(got, list) and len(got) == 6
    check(f'{tag}w3 map captions: the floor applied is named (once per '
          'value), none is claimed where none was applied, and a raw layer '
          'says it has no floor either',
          ok and 'IoU 0.7' in got[0] and 'conf <= 0.25 ' in got[0]
          and 'conf <= 0.25/0.5 ' in got[1]
          and 'conf' not in got[2] and 'conf' not in got[3]
          and all('NOT removed' in x and 'no conf floor' in x
                  for x in got[4:]), got)
    fn = _js_fn(page_src, 'floorNote', '\n}\n')
    fn = fn and (_js_fn(page_src, 'esc', '[c]})}\n') or '') + fn
    got = (_node(fn + 'console.log(JSON.stringify([floorNote({a:0.25}),'
                 'floorNote({}),floorNote(undefined),floorNote({a:0.25,'
                 'b:0.25})]))') if fn else {'err': 'no floorNote'})
    check(f'{tag}w4 explore footer: the floor is named after the duplicates '
          'and the raw footer says no floor was applied',
          isinstance(got, list) and '&le; 0.25 ' in got[0]
          and got[1] == '' and got[2] == '' and got[3] == got[0]
          and 'floorNote(j.dedup.count_conf_min)' in page_src
          and 'no conf floor was applied' in page_src, got)
    outs = []
    for info in ({'dedup': True, 'count_conf_min': {'a': 0.25}},
                 {'dedup': True, 'count_conf_min': {}},
                 {'dedup': False, 'count_conf_min': {}}):
        info = dict(info, built=True, secs=1.0, manifests_found=1,
                    manifests_named=1,
                    dogs={'total': 1, 'images': 1, 'unlocated': 0,
                          'cells': 1},
                    leash={'leashed': 1, 'loose': 0, 'unlocated': 0,
                           'cells': 1})
        saved = (mlm.refresh, sys.argv)
        buf = io.StringIO()
        try:
            mlm.refresh = lambda *a, _i=info, **k: _i
            sys.argv = ['map_layers.py']
            with contextlib.redirect_stdout(buf):
                mlm.main()
        except Exception as e:
            buf.write(repr(e))
        finally:
            mlm.refresh, sys.argv = saved
        outs.append(buf.getvalue().splitlines()[0] if buf.getvalue() else '')
    check(f'{tag}w5 the map_layers CLI line names the floor, and a raw '
          'build says it has none',
          'conf <= 0.25 left out' in outs[0] and 'conf' not in outs[1]
          and 'RAW' in outs[2] and 'no conf floor' in outs[2], outs)
    return len(fails) == n0


def caption_mutants(dash_src, page_src):
    """Each caption put back wrong must fail caption_checks."""
    import contextlib
    import importlib.util
    import io
    ml_src = open(ml.__file__).read()
    mutants = [
        ('d', 'the map caption drops the floor',
         "(v.length?' · boxes at conf <= '+v.sort().join('/')+' left out':'')",
         "''"),
        ('d', 'the raw map caption claims nothing about the floor',
         "'duplicate boxes NOT removed, no conf floor applied'",
         "'duplicate boxes NOT removed'"),
        ('d', 'one floor per engine, repeated',
         "if(v.indexOf(f[k])<0)v.push(f[k])", "v.push(f[k])"),
        ('p', 'the explore footer drops the floor',
         "return v.length?', and boxes at conf &le; '", "return 0?'"),
        ('p', 'the explore footer never asks for the floor',
         "floorNote(j.dedup.count_conf_min)+'.'", "'.'"),
        ('m', 'the CLI line drops the floor',
         "               if floor else '')", "               if 0 else '')"),
    ]
    for k, (which, name, old, new) in enumerate(mutants):
        src = {'d': dash_src, 'p': page_src, 'm': ml_src}[which]
        if src.count(old) != 1:
            check(f'wm mutant "{name}" applies', False,
                  f'target text found {src.count(old)} times')
            continue
        d, p, m = dash_src, page_src, ml
        if which == 'd':
            d = src.replace(old, new)
        elif which == 'p':
            p = src.replace(old, new)
        else:
            path = os.path.join(tempfile.mkdtemp(prefix='adv_box_nms_wm_'),
                                'ml_mut.py')
            with open(path, 'w') as fh:
                fh.write(src.replace(old, new))
            spec = importlib.util.spec_from_file_location('ml_mut', path)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)
        saved = list(fails)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                ok = caption_checks(d, p, m, 'mut: ')
            except Exception:
                ok = False
        fails[:] = saved
        check(f'wm mutant "{name}" is caught', not ok)


# ---- the count-time conf floor ---------------------------------------
ENG = {k: k * 8 + '0' * 56 for k in 'abcde'}   # sha256_engine per engine
# (image_id, det_idx, conf, x1, y1, x2, y2, engine); det_idx is conf order
CBOXES = [
    # 201: 0 sure; 1 just over a's floor (0.26 > 0.25) kept; 2 exactly AT
    # the floor -> out (the review counted conf > 0.25)
    (201, 0, 0.9, 0, 0, 10, 10, 'a'), (201, 1, 0.26, 20, 0, 30, 10, 'a'),
    (201, 2, 0.25, 50, 0, 60, 10, 'a'),
    # 202: every dog under the floor, 1 also an NMS duplicate of 0 -> the
    # frame loses its dog (NMS alone never does that)
    (202, 0, 0.2, 0, 0, 10, 10, 'a'), (202, 1, 0.1, 0, 0, 10, 9, 'a'),
    # 203: engine b has no floor -> counted at any conf
    (203, 0, 0.1, 0, 0, 10, 10, 'b'),
    # 205: engine c is a CANDIDATE with floor 0.5 -> 1 (0.4) is out
    (205, 0, 0.9, 0, 0, 10, 10, 'c'), (205, 1, 0.4, 50, 0, 60, 10, 'c'),
    # 206: engine a, over its floor, stored under two cells (a twin)
    (206, 0, 0.3, 0, 0, 10, 10, 'a'),
    # 207: engine e's floor belongs to ANOTHER project -> not applied
    (207, 0, 0.5, 0, 0, 10, 10, 'e'),
]
# 204 sits in a shard written before model_sha8 existed: NULL -> untouched
CBOX_OLD = (204, 0, 0.1, 0, 0, 10, 10)
CTWIN = (206, 0, 0.3, 0, 0, 10, 10, 'a')
EXPECT_CONF = {(201, 2), (202, 0), (202, 1), (205, 1)}
EXPECT_CONF_NMS = {(202, 1): 0}          # dropped by NMS as well, by box 0
CUT = {ENG['a'][:8]: 0.25, ENG['c'][:8]: 0.5, ENG['d'][:8]: 0.3}


def write_best(path, **over):
    """A best_models.json: a is the detector's best, b/c/d candidates (b
    without a floor, d an engine the store never ran), e in another
    project with a floor that must not leak into the detector's counts."""
    det = {'best': {'run': 'A', 'sha256_engine': ENG['a'],
                    'count_conf_min': 0.25},
           'candidates': [{'run': 'B', 'sha256_engine': ENG['b']},
                          {'run': 'C', 'sha256_engine': ENG['c'],
                           'count_conf_min': 0.5},
                          {'run': 'D', 'sha256_engine': ENG['d'],
                           'count_conf_min': 0.3}]}
    det.update(over)
    with open(path, 'w') as fh:
        json.dump({'projects': {
            'dogdetection': det,
            'dog-bin': {'best': {'run': 'E', 'sha256_engine': ENG['e'],
                                 'count_conf_min': 0.9}}}}, fh)
    return path


def build_conf_store(tmp):
    det = os.path.join(tmp, 'detect')
    con = duckdb.connect()

    def write(path, sch, rows, n):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        con.execute(f'CREATE OR REPLACE TEMP TABLE t({sch})')
        if rows:
            con.executemany(f'INSERT INTO t VALUES ({", ".join("?" * n)})',
                            rows)
        con.execute(f"COPY t TO '{path}' (FORMAT PARQUET)")

    sch = ('image_id UBIGINT, det_idx UTINYINT, conf FLOAT, x1 FLOAT, '
           'y1 FLOAT, x2 FLOAT, y2 FLOAT')
    isch = 'image_id UBIGINT, n_det USMALLINT, max_conf FLOAT'
    boot = os.path.join(det, '_bootstrap', 'gen=_bootstrap',
                        'region=_bootstrap', 'cell=_bootstrap',
                        'drive=_bootstrap', 'det.parquet')
    write(boot, sch, [], 7)
    write(boot.replace('det.parquet', 'img.parquet'), isch, [], 3)
    sh = os.path.join(det, 'shards', 'gen=0002', 'region=R')

    def rows(bs):
        return [b[:7] + (ENG[b[7]][:8],) for b in bs]

    def img_rows(bs):
        by = {}
        for b in bs:
            n, c = by.get(b[0], (0, 0.0))
            by[b[0]] = (n + 1, max(c, b[2]))
        return [(i, n, c) for i, (n, c) in sorted(by.items())]

    for cell, bs, with_sha in (('C1', CBOXES, True), ('C2', [CTWIN], True),
                               ('C3', [CBOX_OLD], False)):
        d = os.path.join(sh, f'cell={cell}', 'drive=d1')
        if with_sha:
            write(os.path.join(d, 's00000.p0_9.det.parquet'),
                  sch + ', model_sha8 VARCHAR', rows(bs), 8)
        else:
            write(os.path.join(d, 's00000.p0_9.det.parquet'), sch, bs, 7)
        write(os.path.join(d, 's00000.p0_9.img.parquet'), isch,
              img_rows(bs), 3)
    allb = [b[:7] for b in CBOXES] + [CBOX_OLD]
    gate = os.path.join(tmp, 'gate')
    leash = os.path.join(tmp, 'leash')
    os.makedirs(gate)
    os.makedirs(leash)
    write(os.path.join(gate, 'gate-00000.parquet'),
          'image_id VARCHAR, det_idx INTEGER, label VARCHAR, p_dog FLOAT, '
          'model VARCHAR',
          [(str(b[0]), b[1], 'dog', 0.9, 'dogbin_test') for b in allb], 5)
    write(os.path.join(leash, 'leash-00000.parquet'),
          'image_id VARCHAR, det_idx INTEGER, p_leashed FLOAT, model VARCHAR',
          [(str(b[0]), b[1], 0.9 if b[1] % 2 == 0 else 0.1, 'leash_test')
           for b in allb], 4)
    man = os.path.join(tmp, 'grid_runs', 'm1.parquet')
    write(man, 'image_id BIGINT, computed_geometry VARCHAR',
          [(i, json.dumps({'type': 'Point', 'coordinates': [0.1, 0.1]}))
           for i in sorted({b[0] for b in allb})], 2)
    st = os.stat(man)
    snap = os.path.join(tmp, 'catalog.parquet')
    write(snap, 'path VARCHAR, region VARCHAR, kind VARCHAR, mtime DOUBLE, '
          'size_bytes BIGINT',
          [(man, 'Testland', 'ground_animals', float(st.st_mtime),
            int(st.st_size))], 5)
    con.close()
    return dict(det_root=det, gate_dir=gate, leash_dir=leash, snapshot=snap,
                nms_dir=os.path.join(tmp, 'nms'), best=os.path.join(
                    tmp, 'best_models.json'), allb=allb)


def conf_checks(bn, fx, legacy, tmp, tag='', mlm=None):
    """Every conf-floor check against module `bn` (the real one or a
    mutant), and map_layers module `mlm` (default the real one). Returns
    True when all of them passed."""
    n0 = len(fails)
    mlm = mlm or ml
    bn.BEST_MODELS = write_best(fx['best'])
    try:
        info = bn.build(fx['det_root'], fx['gate_dir'], fx['nms_dir'],
                        force=True)
    except Exception as e:
        info = {'built': False, 'error': repr(e)}
    check(f'{tag}c0 the fixture with a model_sha8 column builds',
          info.get('built'), info)
    if not info.get('built'):
        return False
    rows = read_sup(fx['nms_dir'])
    gate = {(int(r[0]), int(r[1])): r[5] for r in rows if r[3]}
    box = {(int(r[0]), int(r[1])): r[4] for r in rows if r[2]}
    check(f'{tag}c1 a box at or under its engine\'s floor is out of the gate '
          'set -- the boundary included, a candidate\'s floor included; a '
          'box with no model_sha8, an engine with no floor, or a floor from '
          'another project, is not',
          set(gate) == EXPECT_CONF, f'got {sorted(gate)} '
          f'want {sorted(EXPECT_CONF)}')
    check(f'{tag}c2 ... and out of the class-agnostic set too',
          set(box) == EXPECT_CONF, sorted(box))
    check(f'{tag}c3 by_* names an NMS suppressor only: NULL for a box the '
          'floor alone took out',
          all(gate.get(k) == EXPECT_CONF_NMS.get(k) for k in EXPECT_CONF)
          and all(box.get(k) == EXPECT_CONF_NMS.get(k) for k in EXPECT_CONF),
          (gate, box))
    meta = bn.read_meta(fx['nms_dir']) or {}
    c = meta.get('counts', {})
    n = len(fx['allb'])
    frames = len({b[0] for b in fx['allb']})
    lost = {k[0] for k in EXPECT_CONF} - {
        b[0] for b in fx['allb'] if (b[0], b[1]) not in EXPECT_CONF}
    check(f'{tag}c4 the meta counts are the counts through the floor',
          c.get('boxes') == n
          and c.get('boxes_kept_gate') == n - len(EXPECT_CONF)
          and c.get('boxes_kept') == n - len(EXPECT_CONF)
          and c.get('gate_dogs_kept') == n - len(EXPECT_CONF)
          and c.get('suppressed_conf') == len(EXPECT_CONF)
          and c.get('gate_dogs_below_conf') == len(EXPECT_CONF)
          and c.get('frames_with_dog_kept') == frames - len(lost), c)
    applied = {ENG['a'][:8]: 0.25, ENG['c'][:8]: 0.5}
    note = bn.dedup_note(bn.current(fx['nms_dir'], fx['det_root'],
                                    fx['gate_dir']))
    check(f'{tag}c5 the meta and the payloads\' dedup note name the floors '
          'applied (and the meta the ones configured)',
          meta.get('count_conf_min') == applied
          and meta.get('count_conf_min_config') == CUT
          and (note or {}).get('count_conf_min') == applied,
          (meta.get('count_conf_min'), meta.get('count_conf_min_config'),
           note))
    # another floor is another sidecar
    write_best(fx['best'], best={'run': 'A', 'sha256_engine': ENG['a'],
                                 'count_conf_min': 0.3})
    stale = bn.current(fx['nms_dir'], fx['det_root'], fx['gate_dir'])
    reb = bn.build(fx['det_root'], fx['gate_dir'], fx['nms_dir'])
    rows = read_sup(fx['nms_dir'])
    check(f'{tag}c6 a changed floor makes the sidecar NOT current, and the '
          'next build rebuilds it with the new floor -- compared at the '
          'store\'s float32 precision, so a conf stored as 0.3 is AT a 0.3 '
          'floor, not over it',
          stale is None and reb.get('built')
          and (206, 0) in {(int(r[0]), int(r[1])) for r in rows if r[3]},
          (stale is None, reb.get('built')))
    # an unreadable best_models.json: not current (raw, said so), no build
    with open(fx['best'], 'w') as fh:
        fh.write('{not json')
    try:
        bn.build(fx['det_root'], fx['gate_dir'], fx['nms_dir'], force=True)
        refused = False
    except ValueError:
        refused = True
    check(f'{tag}c7 an unreadable best_models.json: the sidecar is not '
          'current and a build refuses rather than dropping the floor',
          refused and bn.current(fx['nms_dir'], fx['det_root'],
                                 fx['gate_dir']) is None)
    bad = []
    for name, over in (
            ('a floor with no engine to key it',
             {'best': {'run': 'A', 'count_conf_min': 0.25}}),
            ('a floor outside [0, 1)',
             {'best': {'run': 'A', 'sha256_engine': ENG['a'],
                       'count_conf_min': 1.5}}),
            ('two floors for one engine',
             {'candidates': [{'run': 'A2', 'sha256_engine': ENG['a'],
                              'count_conf_min': 0.4}]})):
        try:
            bn.conf_cutoffs(write_best(fx['best'], **over))
            bad.append(name)
        except ValueError:
            pass
    check(f'{tag}c8 a floor that cannot be applied as written is refused',
          not bad, bad)
    absent = os.path.join(tmp, 'no_such_best_models.json')
    check(f'{tag}c9 no best_models.json (a checkout without data/) is no '
          'floor, not an error', bn.conf_cutoffs(absent) == {})
    # GEN 1: no model_sha8 column anywhere. With floors configured it
    # writes the bytes it writes with none, and with none configured its
    # signature is the one every sidecar before the floor carried.
    outs = []
    for k, best in enumerate((write_best(fx['best']), absent)):
        bn.BEST_MODELS = best
        out = os.path.join(tmp, f'legacy_{k}')
        bn.build(legacy['det_root'], legacy['gate_dir'], out, force=True)
        outs.append(open(os.path.join(out, bn.PARQUET), 'rb').read())
    d = box_nms._stat_sig(box_nms.det_files(legacy['det_root']))
    g = box_nms._stat_sig(box_nms.gate_files(legacy['gate_dir']))
    old = __import__('hashlib').sha256(json.dumps(
        [bn.SCHEMA, bn.IOU, bn.P_DOG_MIN, bn.RULE, d, g]).encode()
    ).hexdigest()
    lmeta = bn.read_meta(os.path.join(tmp, 'legacy_1')) or {}
    check(f'{tag}c10 a store with no model_sha8 column writes the same '
          'parquet bytes with floors configured as with none, and with none '
          'its signature is the pre-floor one',
          outs[0] == outs[1] and lmeta.get('sig') == old
          and bn.read_meta(os.path.join(tmp, 'legacy_0'))
          .get('counts', {}).get('suppressed_conf') == 0,
          (outs[0] == outs[1], lmeta.get('sig') == old))
    # the consumers, unchanged, take the floor through sup_gate
    bn.BEST_MODELS = write_best(fx['best'])
    bn.build(fx['det_root'], fx['gate_dir'], fx['nms_dir'], force=True)
    cur = bn.current(fx['nms_dir'], fx['det_root'], fx['gate_dir'])
    saved = mlm._box_nms
    out = os.path.join(tmp, 'layers')
    try:
        # map_layers reloads box_nms by name; hand it this module as is
        mlm._box_nms = lambda nd, dr, gd: (bn.current(nd, dr, gd), bn)
        rinfo = mlm.refresh(force=True, gate_dir=fx['gate_dir'],
                           leash_dir=fx['leash_dir'],
                           snapshot=fx['snapshot'], out_dir=out,
                           hide_regions=frozenset(), nms_dir=fx['nms_dir'],
                           det_root=fx['det_root'])
        dogs = json.load(open(os.path.join(out, 'map_layer_dogs.json')))
        lj = json.load(open(os.path.join(out, 'map_layer_leash.json')))
    except Exception as e:
        dogs, lj, rinfo = {'error': repr(e)}, {}, {}
    finally:
        mlm._box_nms = saved
    kept = [b for b in fx['allb'] if (b[0], b[1]) not in EXPECT_CONF]
    check(f'{tag}c11 map layers: dogs and leash calls lose the boxes under '
          'the floor, through the sidecar alone',
          dogs.get('total') == len(kept)
          and lj.get('leashed_total', 0) + lj.get('loose_total', 0)
          == len(kept)
          and (dogs.get('dedup') or {}).get('count_conf_min') == applied
          and rinfo.get('count_conf_min') == applied,
          (dogs.get('total'), len(kept), dogs.get('dedup')))
    if cur and os.path.exists(os.path.join(REPO, 'tools', 'dashboard',
                                           'explore_index.py')):
        import explore_index as ei
        explore_floor_checks(ei, bn, fx, cur, kept, tag)
    return len(fails) == n0


def refresh_mutant(tmp, legacy):
    """map_layers.refresh() that stops handing the floor on (the CLI line
    reads it) must fail conf_checks' c11."""
    import contextlib
    import importlib.util
    import io
    src = open(ml.__file__).read()
    old = ("            'count_conf_min': (nms or {}).get('count_conf_min') "
           "or {},\n")
    name = 'refresh() does not hand the floor to the CLI'
    if src.count(old) != 1:
        check(f'rm mutant "{name}" applies', False, src.count(old))
        return
    os.makedirs(tmp)
    path = os.path.join(tmp, 'ml_mut.py')
    with open(path, 'w') as fh:
        fh.write(src.replace(old, ''))
    spec = importlib.util.spec_from_file_location('ml_mut', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    fx = build_conf_store(os.path.join(tmp, 'fx'))
    saved, saved_f = box_nms.BEST_MODELS, list(fails)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            ok = conf_checks(box_nms, fx, legacy, tmp, 'mut: ', m)
        except Exception:
            ok = False
    fails[:] = saved_f
    box_nms.BEST_MODELS = saved
    check(f'rm mutant "{name}" is caught', not ok)


def explore_floor_checks(ei, bn, fx, cur, kept, tag=''):
    """explore_index (module `ei`, the real one or a mutant) through the
    floor: judged, kept and boxes lose what it took; the frames the detector
    PROCESSED do not; the mean surest confidence is over the boxes kept --
    a frame whose every box is under the floor has no surest box left."""
    import store as _store
    n0 = len(fails)
    con = duckdb.connect()
    con.execute('CREATE TABLE ic(image_id UBIGINT, cc VARCHAR)')
    con.executemany('INSERT INTO ic VALUES (?, ?)',
                    [(i, 'TST') for i in sorted({b[0]
                                                 for b in fx['allb']})])
    sg = _store.get_detect_root
    try:
        _store.get_detect_root = lambda repo_root=None: fx['det_root']
        g, _ = ei._ledger_rollup(con, fx['gate_dir'], 'gate-*.parquet',
                                 'p_dog', 0.5, 'gate', cur, bn)
        dr = ei.detect_rollup(con, cur, bn)
    except Exception as e:
        g, dr = {'error': repr(e)}, {}
    finally:
        _store.get_detect_root = sg
        con.close()
    t, r = g.get('TST', {}), dr.get('TST', {})
    check(f'{tag}c12 explore: judged, kept and boxes all lose the boxes '
          'under the floor',
          t.get('gate_judged') == len(kept)
          and t.get('gate_kept') == len(kept)
          and r.get('det_boxes') == len(kept), (t, r))
    top = {}
    for b in kept:
        top[b[0]] = max(top.get(b[0], 0.0), b[2])
    want = round(sum(top.values()) / len(top), 4)
    check(f'{tag}c13 explore: frames processed keep every frame (202 lost '
          'its boxes, not its processing); the mean surest confidence is '
          'over the boxes kept',
          r.get('det_images') == len({b[0] for b in fx['allb']})
          and abs((r.get('det_mean_max_conf') or 0) - want) < 1e-4,
          (r.get('det_images'), r.get('det_mean_max_conf'), want))
    return len(fails) == n0


def explore_mutants(tmp, legacy):
    """explore_index put back wrong must fail explore_floor_checks."""
    import contextlib
    import importlib.util
    import io
    path = os.path.join(REPO, 'tools', 'dashboard', 'explore_index.py')
    if not os.path.exists(path):
        return
    src = open(path).read()
    mutants = [
        ('the mean surest conf left on the raw image table',
         "            row['det_mean_max_conf'] = round(float(maxc or 0), 4)\n",
         ''),
        ('frames processed counted off the boxes kept',
         "        per[cc] = {'det_images': int(imgs or 0),",
         "        per[cc] = {'det_images': int(dets or 0),"),
    ]
    fx = build_conf_store(os.path.join(tmp, 'fx'))
    saved = box_nms.BEST_MODELS
    try:
        box_nms.BEST_MODELS = write_best(fx['best'])
        box_nms.build(fx['det_root'], fx['gate_dir'], fx['nms_dir'],
                      force=True)
        cur = box_nms.current(fx['nms_dir'], fx['det_root'], fx['gate_dir'])
    finally:
        box_nms.BEST_MODELS = saved
    kept = [b for b in fx['allb'] if (b[0], b[1]) not in EXPECT_CONF]
    for k, (name, old, new) in enumerate(mutants):
        if src.count(old) != 1:
            check(f'em mutant "{name}" applies', False,
                  f'target text found {src.count(old)} times')
            continue
        mp = os.path.join(tmp, f'ei_mut{k}.py')
        with open(mp, 'w') as fh:
            fh.write(src.replace(old, new))
        spec = importlib.util.spec_from_file_location(f'ei_mut{k}', mp)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            check(f'em mutant "{name}" loads', False, repr(e))
            continue
        saved_f = list(fails)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                ok = explore_floor_checks(mod, box_nms, fx, cur, kept,
                                          'mut: ')
            except Exception:
                ok = False
        fails[:] = saved_f
        check(f'em mutant "{name}" is caught', not ok)


def conf_mutants(tmp, legacy):
    """Each guard put back wrong must fail conf_checks."""
    import contextlib
    import importlib.util
    import io
    src = open(box_nms.__file__).read()
    mutants = [
        ('the floor left out of sup_gate',
         "'sup_gate': pa.array(in_g | in_c)", "'sup_gate': pa.array(in_g)"),
        ('the floor left out of sup_box',
         "'sup_box': pa.array(in_b | in_c)", "'sup_box': pa.array(in_b)"),
        ('the boundary counted (conf < floor, not <=)',
         'WHERE b.conf <= cut.cmin', 'WHERE b.conf < cut.cmin'),
        ('one floor for every box, whatever drew it',
         'FROM b JOIN cut USING (sha8)\n            WHERE',
         'FROM b, cut\n            WHERE'),
        ('model_sha8 never read',
         "if 'model_sha8' in cols", "if False"),
        ('the floor compared as a double (float32 0.3 reads > 0.3)',
         "cut(sha8 VARCHAR, cmin FLOAT)", "cut(sha8 VARCHAR, cmin DOUBLE)"),
        ('candidates ignored',
         "[proj.get('best')] + list(proj.get('candidates') or [])",
         "[proj.get('best')]"),
        ('another project\'s floors read',
         ".get(DET_PROJECT) or {}", ".get('dog-bin') or {}"),
        ('the floor left out of the signature',
         '    if cut:\n        parts.append', '    if False:\n        '
         'parts.append'),
        ('the signature moved for a store with no floor',
         '    if cut:\n        parts.append', '    if True:\n        '
         'parts.append'),
        ('the dedup note drops the floor',
         "'count_conf_min': meta.get('count_conf_min') or {}}",
         "'count_conf_min': {}}"),
        ('a floor out of range accepted', 'or not 0 <= c < 1):',
         'or False):'),
        ('the meta box count ignores the floor',
         "'boxes_kept_gate': int(n_boxes - int((in_g | in_c).sum()))",
         "'boxes_kept_gate': int(n_boxes - int(in_g.sum()))"),
        ('an unreadable best_models.json read as no floor',
         '    except FileNotFoundError:\n        return {}',
         '    except (FileNotFoundError, ValueError):\n        return {}'),
    ]
    for k, (name, old, new) in enumerate(mutants):
        if src.count(old) != 1:
            check(f'cm mutant "{name}" applies', False,
                  f'target text found {src.count(old)} times')
            continue
        d = os.path.join(tmp, f'mut{k}')
        os.makedirs(d)
        path = os.path.join(d, 'box_nms_mut.py')
        with open(path, 'w') as fh:
            fh.write(src.replace(old, new))
        spec = importlib.util.spec_from_file_location('box_nms_mut', path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            mod.REPO = REPO
            fx = build_conf_store(os.path.join(d, 'fx'))
            saved = list(fails)
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    ok = conf_checks(mod, fx, legacy, d, 'mut: ')
                except Exception:
                    ok = False
            fails[:] = saved
        except Exception as e:
            check(f'cm mutant "{name}" loads', False, repr(e))
            continue
        check(f'cm mutant "{name}" is caught', not ok)


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
        dash_src = open(DASH_PY).read()
        page_py = os.path.join(REPO, 'tools', 'dashboard', 'explore_page.py')
        page_src = open(page_py).read() if os.path.exists(page_py) else ''
        if not shutil.which('node'):
            print('SKIP w3-w5: node not on PATH')
        elif 'count_conf_min' not in (_js_fn(dash_src, 'dedupNote',
                                             '\n    }\n') or ''):
            print('SKIP w3-w5: this dashboard.py has no conf-floor caption '
                  'yet (install the switch copy, or run with DASH_PY=<copy>)')
        elif not page_src:
            print('SKIP w3-w5: explore_page.py is not in this checkout')
        else:
            caption_checks(dash_src, page_src, ml)
            caption_mutants(dash_src, page_src)
        ctmp = os.path.join(tmp, 'conf')
        cfx = build_conf_store(ctmp)
        saved = box_nms.BEST_MODELS
        try:
            conf_checks(box_nms, cfx, fx, ctmp)
            conf_mutants(os.path.join(tmp, 'conf_mut'), fx)
            explore_mutants(os.path.join(tmp, 'explore_mut'), fx)
            refresh_mutant(os.path.join(tmp, 'refresh_mut'), fx)
        finally:
            box_nms.BEST_MODELS = saved
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{len(fails)} failed' if fails else '\nall passed')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
