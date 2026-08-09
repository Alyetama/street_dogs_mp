#!/usr/bin/env python3
"""Build dogdet_v3: the detector dataset, re-split by sequence and fed the
confirmed near-misses.

    python tools/detect/build_dogdet_v3.py --out <dogs_detection>/dogdet_v3

WHY A REBUILD. Measured on dogdet_v2 before this was written: 63.1% of val
frames (309/490) sit in a sequence that also appears in train. Mapillary
frames come one second apart down one road, so those val frames are near
duplicates of training frames -- the same defect measured at 70.8% in
leash_binary_v1, now in the detector's own split. The retrain this feeds is
being selected on RECALL, and recall read off near-duplicates of the training
set is not a measurement.

WHAT GOES IN, beyond dogdet_v2:

  * hard_positives -- detections at conf as low as 0.05 that a person
    confirmed are real dogs. These are the recall frontier: dogs the current
    model very nearly missed, on the exact imagery it runs against.
  * audit verdicts of 'dog' -- the same thing found from the other end, boxes
    a person confirmed while auditing the gate. Human answers both; the
    model's own opinion is never a label.
  * box_corrections -- hand-drawn geometry overriding the detector's, applied
    wherever an added frame carries one.

ONLY SINGLE-DETECTION FRAMES ARE ADDED, the same rule build_detector_negatives
applies to backgrounds, for the same reason read the other way: a frame gets
labels for the boxes a person confirmed and silence everywhere else, and
silence trains "no dog here". On a frame whose only detection is the confirmed
one, that silence is as true as the sweep could make it; on a multi-detection
frame it is a lie about the unconfirmed boxes.

THE HOLDOUT. Frames of old val whose sequences never appeared in old train
are the only frames the promoted model (train-30) provably never learned
from. They stay in val, listed in the manifest as `holdout`, so the old and
new model can be compared on ground neither trained on. Leaked old-val frames
move to TRAIN (their sequences live there already; val is the only place they
do harm).

Whole sequences move, never frames. Unresolved-sequence frames are pinned to
train -- absence of a sequence is not a sequence, and a frame that cannot
prove it is unrelated to train does not belong in val.
"""

import argparse
import glob
import json
import os
import random
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def _training_root():
    """The training repo. Env first, then the dashboard's own config -- the
    same answer gate_control resolves the promoted weights against -- so this
    works in a bare shell and in the sweep of guards alike."""
    got = os.environ.get('TRAINING_ROOT')
    if got:
        return got
    try:
        sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
        import dashboard as dash
        got = dash.training_root()
    except Exception:
        got = ''
    return got or os.path.dirname(os.path.dirname(REPO))

TRAINING_ROOT = _training_root()
V2 = os.path.join(TRAINING_ROOT, 'dogdet_v2')
HARD_POS = os.path.join(REPO, 'data', 'hard_positives', 'labels.jsonl')
AUDIT_VERDICTS = os.path.join(REPO, 'data', 'fn_audit', 'verdicts.jsonl')
CORRECTIONS = os.path.join(REPO, 'data', 'box_corrections', 'boxes.jsonl')


def _grid_roots():
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    import dashboard as dash
    return dash._grid_roots()


def resolve_sequences(ids, con):
    """{image_id: sequence} off the harvest manifests, NULLs dropped.

    min() per id, not any_value: a duplicated frame must resolve to the same
    sequence on every run or the split moves under the seed.
    """
    mans = []
    for root in _grid_roots().values():
        mans += glob.glob(os.path.join(root, '*', 'all_data_*.parquet'))
    con.execute("CREATE TEMP TABLE need(image_id VARCHAR)")
    con.executemany("INSERT INTO need VALUES (?)", [(i,) for i in ids])
    con.execute(f"""
        CREATE TEMP TABLE seqs AS
        SELECT image_id, min(seq) AS seq FROM (
            SELECT CAST(m.image_id AS VARCHAR) image_id,
                   CAST(m."sequence" AS VARCHAR) seq
            FROM read_parquet({mans!r}, union_by_name=true) m
            SEMI JOIN need n ON n.image_id = CAST(m.image_id AS VARCHAR)
            WHERE m."sequence" IS NOT NULL) GROUP BY 1""")
    return dict(con.execute("SELECT image_id, seq FROM seqs").fetchall())


def frame_meta(ids, con):
    """{image_id: (n_det, orig_w, orig_h, cell, drive)} from the sweep store."""
    sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
    import store
    root = store.get_detect_root()
    img = store._sql_src(store._store_globs(root, 'img'))
    det = store._sql_src(store._store_globs(root, 'det'))
    con.execute("CREATE TEMP TABLE want(image_id VARCHAR)")
    con.executemany("INSERT INTO want VALUES (?)", [(i,) for i in ids])
    meta = {}
    for iid, n, w, h, cell, drive in con.execute(f"""
            SELECT CAST(i.image_id AS VARCHAR), any_value(i.n_det),
                   any_value(i.orig_w), any_value(i.orig_h),
                   any_value(i.cell), any_value(i.drive)
            FROM {img} i SEMI JOIN want t
              ON t.image_id = CAST(i.image_id AS VARCHAR)
            GROUP BY 1""").fetchall():
        meta[iid] = (int(n or 0), int(w or 0), int(h or 0), cell, drive)
    boxes = {}
    for iid, di, x1, y1, x2, y2 in con.execute(f"""
            SELECT CAST(d.image_id AS VARCHAR), d.det_idx,
                   any_value(d.x1), any_value(d.y1),
                   any_value(d.x2), any_value(d.y2)
            FROM {det} d SEMI JOIN want t
              ON t.image_id = CAST(d.image_id AS VARCHAR)
            GROUP BY 1, 2""").fetchall():
        boxes[(iid, int(di))] = (float(x1), float(y1), float(x2), float(y2))
    return meta, boxes


def confirmed_positive_ids():
    """Frames a HUMAN said contain a real dog, from both ledgers."""
    out = set()
    try:
        for line in open(HARD_POS):
            if line.strip():
                d = json.loads(line)
                if d.get('label') == 'true_positive':
                    out.add(str(d['image_id']))
    except OSError:
        pass
    try:
        sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
        import fn_audit as fa
        for v in fa.read_verdicts(stage='gate'):
            if fa.verdict_of(v.get('verdict'), 'gate') == 'dog':
                out.add(str(v['key']).split('#')[0])
    except Exception:
        pass
    return out


def corrections():
    out = {}
    try:
        for line in open(CORRECTIONS):
            if line.strip():
                d = json.loads(line)
                out[(str(d['image_id']), int(d.get('det_idx') or 0))] = (
                    float(d['x1']), float(d['y1']),
                    float(d['x2']), float(d['y2']))
    except OSError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=os.path.join(TRAINING_ROOT,
                                                  'dogdet_v3'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-target', type=int, default=480,
                    help='promote whole train sequences to val until val '
                         'reaches roughly this many frames')
    ap.add_argument('--memory', default='6GB')
    a = ap.parse_args()
    import duckdb
    rng = random.Random(a.seed)

    train_ids = sorted(f[:-4] for f in os.listdir(f'{V2}/images/train'))
    val_ids = sorted(f[:-4] for f in os.listdir(f'{V2}/images/val'))
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{a.memory}'")
    con.execute("SET preserve_insertion_order=false")

    print(f'{len(train_ids)} train + {len(val_ids)} val in dogdet_v2',
          flush=True)
    adds = confirmed_positive_ids()
    have = set(train_ids) | set(val_ids)
    adds -= have
    print(f'{len(adds)} confirmed-positive frames not yet in the dataset',
          flush=True)

    seq = resolve_sequences(sorted(have | adds), con)
    tr_seq = {seq[i] for i in train_ids if i in seq}

    # ── the split ───────────────────────────────────────────────────────────
    holdout = [i for i in val_ids
               if i in seq and seq[i] not in tr_seq]
    leaked = [i for i in val_ids if i in seq and seq[i] in tr_seq]
    unresolved_val = [i for i in val_ids if i not in seq]
    print(f'holdout (train-30-blind val): {len(holdout)}  ·  '
          f'leaked val -> train: {len(leaked)}  ·  '
          f'unresolved val -> train: {len(unresolved_val)}', flush=True)

    new_train = set(train_ids) | set(leaked) | set(unresolved_val)
    new_val = set(holdout)
    val_seqs = {seq[i] for i in new_val}

    # promote whole sequences until val is a real benchmark again
    by_seq = {}
    for i in sorted(new_train):
        if i in seq:
            by_seq.setdefault(seq[i], []).append(i)
    candidates = sorted(by_seq)
    rng.shuffle(candidates)
    promoted = []
    for sq in candidates:
        if len(new_val) >= a.val_target:
            break
        if sq in val_seqs:
            continue
        members = by_seq[sq]
        new_val.update(members)
        new_train.difference_update(members)
        val_seqs.add(sq)
        promoted.append(sq)
    print(f'promoted {len(promoted)} sequences '
          f'({len(new_val) - len(holdout)} frames) from train to val',
          flush=True)

    # ── the additions ───────────────────────────────────────────────────────
    meta, boxes = frame_meta(sorted(adds), con)
    fixes = corrections()
    roots = _grid_roots()
    added, skipped = [], {'multi_detection': 0, 'val_sequence': 0,
                          'no_meta': 0, 'no_file': 0}
    for iid in sorted(adds):
        m = meta.get(iid)
        if not m or not m[1] or not m[2]:
            skipped['no_meta'] += 1
            continue
        n_det, w, h, cell, drive = m
        if n_det != 1:
            skipped['multi_detection'] += 1
            continue
        if seq.get(iid) in val_seqs:
            skipped['val_sequence'] += 1
            continue
        src = os.path.join(roots.get(drive, ''), cell,
                           'ground_animal_images', f'{iid}.jpg')
        if not os.path.exists(src):
            skipped['no_file'] += 1
            continue
        box = fixes.get((iid, 0)) or boxes.get((iid, 0))
        if not box:
            skipped['no_meta'] += 1
            continue
        added.append((iid, src, box, (w, h)))
    print(f'adding {len(added)} frames to train, skipped {skipped}',
          flush=True)

    # ── write it ────────────────────────────────────────────────────────────
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    def put(src_img, src_lbl, split, iid):
        di = os.path.join(a.out, 'images', split, f'{iid}.jpg')
        dl = os.path.join(a.out, 'labels', split, f'{iid}.txt')
        for s, d in ((src_img, di), (src_lbl, dl)):
            if os.path.exists(d):
                os.remove(d)
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)

    old_split = {i: 'train' for i in train_ids}
    old_split.update({i: 'val' for i in val_ids})
    n_bg = {'train': 0, 'val': 0}
    for iid in sorted(new_train | new_val):
        split = 'train' if iid in new_train else 'val'
        was = old_split[iid]
        put(os.path.join(V2, 'images', was, f'{iid}.jpg'),
            os.path.join(V2, 'labels', was, f'{iid}.txt'), split, iid)
        if os.path.getsize(os.path.join(V2, 'labels', was,
                                        f'{iid}.txt')) == 0:
            n_bg[split] += 1

    for iid, src, (x1, y1, x2, y2), (w, h) in added:
        dst = os.path.join(a.out, 'images', 'train', f'{iid}.jpg')
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        with open(os.path.join(a.out, 'labels', 'train',
                               f'{iid}.txt'), 'w') as fh:
            fh.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')

    with open(os.path.join(a.out, 'dataset.yaml'), 'w') as fh:
        fh.write(f'path: {a.out}\ntrain: images/train\nval: images/val\n\n'
                 f'names:\n  0: target\n')

    n_train = len(new_train) + len(added)
    manifest = {
        'src': V2, 'seed': a.seed,
        'train': n_train, 'val': len(new_val),
        'backgrounds_train': n_bg['train'], 'backgrounds_val': n_bg['val'],
        'holdout': sorted(holdout),
        'leaked_val_moved_to_train': len(leaked),
        'unresolved_pinned_to_train': len(unresolved_val),
        'promoted_sequences': len(promoted),
        'added_confirmed_positives': len(added),
        'added_ids': sorted(i for i, *_ in added),
        'additions_skipped': skipped,
        'corrections_applied': sum(1 for i, *_ in added
                                   if (i, 0) in fixes),
        'sequences': {i: seq.get(i) for i in sorted(new_train | new_val
                                                    | {x[0] for x in added})},
    }
    with open(os.path.join(a.out, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=1)
    print(f"dogdet_v3: {n_train} train ({n_bg['train']} bg) / "
          f"{len(new_val)} val ({n_bg['val']} bg), "
          f"holdout {len(holdout)}, +{len(added)} confirmed positives "
          f"({manifest['corrections_applied']} with corrected boxes)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
