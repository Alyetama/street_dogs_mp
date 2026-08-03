#!/usr/bin/env python3
"""
Turn dashboard verdicts into full-resolution training crops.

``--label false_positive`` (default) harvests the hard NEGATIVES -- crops the
reviewer marked "not a dog". ``--label true_positive`` harvests the hard
POSITIVES -- low-confidence detections the reviewer confirmed really are dogs,
which for a gate tuned on recall are the expensive ones to miss.

Boxes corrected by hand in the review page override the detector geometry
(``--corrections``); without that the editor would be decorative.

The dashboard's "flag as false positive" button records the image_id of a
detection the user judged not to be a dog. The thumbnail it copied is only
~160 px -- fine for the UI, too small and too lossy for training. This tool
re-cuts each flagged detection from the ORIGINAL full-resolution jpg, using
the exact box the sweep stored.

That is possible because the predictions store already holds, for every
detection, ``x1,y1,x2,y2`` in ORIGINAL pixels (spec section 5.3). So a flag
only has to persist an image_id; the geometry is looked up here.

Output is laid out for ``yolo classify``, so the crops can be dropped straight
into an existing dataset's negative class:

    <out>/not_dog/<image_id>_<det_idx>.jpg

Why these are worth more than annotator negatives: every one of them fooled
the detector at inference time on real corpus imagery. Boxes an annotator drew
are mostly distant specks (median short side 36 px vs 208 px for dogs); these
sit in the same size range as true detections, so a classifier trained on them
cannot fall back on "small means not a dog".

    python tools/detect/harvest_flagged.py --out <home>/dogs_detection/hard_negatives
    python tools/detect/harvest_flagged.py --append-to <dogbin_v3>/train/not_dog

READ-ONLY on the sweep store and on every image drive.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import store  # noqa: E402

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGERS = {
    'false_positive': os.path.join(REPO, 'data', 'hard_negatives',
                                   'labels.jsonl'),
    'true_positive': os.path.join(REPO, 'data', 'hard_positives',
                                  'labels.jsonl'),
}
FLAGS = LEDGERS['false_positive']
# Boxes the reviewer corrected by hand in the dashboard. They are the whole
# point of the editor: without reading them here, every correction is
# discarded and the crop is cut from the detector's original box.
CORRECTIONS = os.path.join(REPO, 'data', 'box_corrections', 'boxes.jsonl')


def read_corrections(path):
    """{(image_id, det_idx): (x1, y1, x2, y2)} -- last write wins."""
    out = {}
    try:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if not isinstance(r, dict) or not r.get('image_id'):
                    continue
                try:
                    out[(str(r['image_id']), int(r.get('det_idx') or 0))] = (
                        float(r['x1']), float(r['y1']),
                        float(r['x2']), float(r['y2']))
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    return out


def read_flags(path, want='false_positive'):
    """{image_id: record} for one label, last write wins."""
    out = {}
    try:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if r.get('label') == want and r.get('image_id'):
                    out[str(r['image_id'])] = r
    except OSError:
        pass
    return out


def boxes_for(image_ids, detect_root, python_bin=None):
    """{image_id: [(det_idx, x1, y1, x2, y2, conf)]} from the parquet store."""
    if not image_ids:
        return {}
    ids = ','.join("'%s'" % i for i in image_ids)
    det = store._sql_src(store._store_globs(detect_root, 'det'))
    sql = (f"SELECT CAST(image_id AS VARCHAR), det_idx, x1, y1, x2, y2, conf "
           f"FROM {det} WHERE CAST(image_id AS VARCHAR) IN ({ids})")
    rows = store._run_queries({'q': sql})['q']
    out = {}
    for iid, di, x1, y1, x2, y2, conf in rows:
        out.setdefault(str(iid), []).append(
            (int(di), float(x1), float(y1), float(x2), float(y2), float(conf)))
    return out


def cells_for(image_ids, detect_root):
    """{image_id: (cell, drive)} -- lets us build the jpg path directly.

    The sweep already recorded which (cell, drive) each image came from, so
    a flagged image's file is at <root_of_drive>/<cell>/ground_animal_images/
    <image_id>.jpg. That is the whole reason not to enumerate the corpus.
    """
    if not image_ids:
        return {}
    ids = ','.join("'%s'" % i for i in image_ids)
    img = store._sql_src(store._store_globs(detect_root, 'img'))
    sql = (f"SELECT CAST(image_id AS VARCHAR), any_value(cell), "
           f"any_value(drive) FROM {img} "
           f"WHERE CAST(image_id AS VARCHAR) IN ({ids}) GROUP BY image_id")
    return {str(i): (c, d) for i, c, d in store._run_queries({'q': sql})['q']}


def roots_by_drive(roots):
    """{drive label: grid_runs root} -- drive label is the mount basename."""
    out = {}
    for r in roots:
        parts = os.path.abspath(r).split(os.sep)
        # /media/<user>/<drive>/... or /home/<user>/<drive>/...
        for i, p in enumerate(parts):
            if p in ('media', 'home') and i + 2 < len(parts):
                out.setdefault(parts[i + 2], r)
                break
    return out


def index_images(roots):
    """{image_id: path} across every grid_runs root (cell tree or flat)."""
    idx = {}
    for root in roots:
        for d in glob.glob(os.path.join(root, '*', 'ground_animal_images')):
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if e.name.endswith('.jpg'):
                            idx.setdefault(e.name[:-4], e.path)
            except OSError:
                continue
    return idx


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--label',
                   default='false_positive',
                   choices=sorted(LEDGERS),
                   help='which verdict to harvest. false_positive -> hard '
                   'negatives (not_dog); true_positive -> the low-confidence '
                   'real dogs the reviewer confirmed (dog).')
    p.add_argument('--flags',
                   help='ledger to read (defaults to the one for --label)')
    p.add_argument('--corrections',
                   default=CORRECTIONS,
                   help='hand-corrected boxes from the review page; they '
                   'override the detector geometry. Pass "" to ignore.')
    p.add_argument('--out', help='write <out>/<class>/*.jpg')
    p.add_argument('--append-to',
                   help='write crops directly into this existing class dir '
                   '(e.g. dogbin_v3/train/not_dog)')
    p.add_argument('--roots-file',
                   default=os.path.join(REPO, 'data', 'catalog_dirs.txt'))
    p.add_argument('--pad',
                   type=float,
                   default=0.15,
                   help='context padding as a fraction of box size, matching '
                   'build_crop_dataset.py so crops are comparable')
    p.add_argument('--min-size',
                   type=int,
                   default=64,
                   help='skip crops whose short side is under this')
    p.add_argument('--max-conf',
                   type=float,
                   help='only harvest flags below this detector confidence')
    p.add_argument('--all-detections',
                   action='store_true',
                   help='re-cut EVERY detection in a flagged image. Off by '
                   'default: the user flagged one crop, and 94 of 314 flagged '
                   'images carry 2-12 detections, so the others may well be '
                   'real dogs -- harvesting them would put dogs into the '
                   'negative class and poison the very gate this feeds.')
    p.add_argument('--conf-tol',
                   type=float,
                   default=0.006,
                   help='tolerance when matching a flag back to its detection '
                   'by confidence (the ledger rounds conf to 2 decimals)')
    p.add_argument('--scan-corpus',
                   action='store_true',
                   help='if an image cannot be placed from the store, fall '
                   'back to enumerating every jpg on every root (32.5M '
                   'entries, minutes of I/O the running sweep needs)')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()

    import cv2

    ledger = args.flags or LEDGERS[args.label]
    cls_dir = 'dog' if args.label == 'true_positive' else 'not_dog'
    flags = read_flags(ledger, args.label)
    if not flags:
        print(f'no {args.label} flags in {ledger}')
        return 0
    if args.max_conf is not None:
        flags = {
            k: v
            for k, v in flags.items()
            if float(v.get('conf', 1.0)) < args.max_conf
        }
    print(f'{len(flags):,} flagged detections')

    dst = args.append_to or (os.path.join(args.out, cls_dir)
                             if args.out else None)
    if not dst:
        print('pass --out or --append-to', file=sys.stderr)
        return 2

    boxes = boxes_for(list(flags), store.get_detect_root())
    print(f'  {len(boxes):,} found in the predictions store')
    missing_store = [i for i in flags if i not in boxes]
    if missing_store:
        print(f'  {len(missing_store):,} not in the store yet (the sweep may '
              f'not have committed that shard); re-run later')

    # Keep only the detection the user actually flagged. The flag persists the
    # box's confidence, so match on it; images whose flag cannot be resolved to
    # exactly one detection are skipped rather than guessed at.
    if not args.all_detections:
        kept, ambiguous, unmatched = {}, 0, 0
        for iid, dets in boxes.items():
            want = flags[iid].get('conf')
            if want is None:
                kept[iid] = dets
                continue
            hit = [d for d in dets if abs(d[5] - float(want)) <= args.conf_tol]
            if len(hit) == 1:
                kept[iid] = hit
            elif not hit:
                unmatched += 1
            else:
                # several boxes at the same confidence -- cannot tell which was
                # flagged, and a wrong guess is a dog labelled not-a-dog
                ambiguous += 1
        n_multi = sum(1 for d in boxes.values() if len(d) > 1)
        print(f'  {n_multi:,} flagged images carry >1 detection; matching each '
              f'flag to its own box by confidence')
        print(f'  kept {sum(len(v) for v in kept.values()):,} of '
              f'{sum(len(v) for v in boxes.values()):,} detections'
              f'  (skipped {ambiguous:,} ambiguous, {unmatched:,} unmatched)')
        boxes = kept

    # Hand-corrected geometry wins over the detector's. Without this the
    # review page's box editor would be decorative: the user drags the box,
    # and the crop is still cut where the model guessed.
    corr = read_corrections(args.corrections) if args.corrections else {}
    n_corr = 0
    if corr:
        for iid, dets in boxes.items():
            for i, d in enumerate(dets):
                c = corr.get((iid, d[0]))
                if c:
                    dets[i] = (d[0], c[0], c[1], c[2], c[3], d[5])
                    n_corr += 1
        print(f'  {n_corr:,} box(es) replaced by a reviewer correction '
              f'({len(corr):,} in the ledger)')

    roots = [
        ln.strip() for ln in open(args.roots_file)
        if ln.strip() and not ln.startswith('#')
    ]
    # Resolve each flagged image straight to its file via the (cell, drive)
    # the sweep recorded. The old path built a {image_id: path} dict over
    # every jpg in the corpus -- 32.58M entries, ~12 min of drive I/O stolen
    # from the running sweep and several GB of RAM, to look up ~300 ids.
    # --scan-corpus keeps that fallback for images the store has no row for.
    idx = {}
    placed = cells_for(list(boxes), store.get_detect_root())
    by_drive = roots_by_drive(roots)
    for iid, (cell, drive) in placed.items():
        root = by_drive.get(drive)
        if not root:
            continue
        p = os.path.join(root, cell, 'ground_animal_images', f'{iid}.jpg')
        if os.path.exists(p):
            idx[iid] = p
    print(f'  resolved {len(idx):,}/{len(boxes):,} jpgs directly from the '
          f'store\'s (cell, drive) -- no corpus enumeration')
    if args.scan_corpus and len(idx) < len(boxes):
        print(f'  {len(boxes) - len(idx):,} unresolved -- falling back to a '
              f'full corpus scan (slow)')
        idx.update({k: v for k, v in index_images(roots).items()
                    if k not in idx})
    n_ok = n_small = n_noimg = 0
    if args.execute:
        os.makedirs(dst, exist_ok=True)
    for iid, dets in boxes.items():
        path = idx.get(iid)
        if not path:
            n_noimg += 1
            continue
        img = cv2.imread(path) if args.execute else None
        for di, x1, y1, x2, y2, conf in dets:
            w, h = x2 - x1, y2 - y1
            if min(w, h) < args.min_size:
                n_small += 1
                continue
            n_ok += 1
            if not args.execute or img is None:
                continue
            pad = args.pad * max(w, h)
            a = int(max(0, x1 - pad))
            b = int(max(0, y1 - pad))
            c = int(min(img.shape[1], x2 + pad))
            d = int(min(img.shape[0], y2 + pad))
            if c - a < 8 or d - b < 8:
                continue
            cv2.imwrite(os.path.join(dst, f'{iid}_{di}.jpg'), img[b:d, a:c],
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    verb = 'wrote' if args.execute else 'would write'
    kind = 'positive' if args.label == 'true_positive' else 'negative'
    print(f'\n{verb} {n_ok:,} full-res {kind} crops -> {dst}')
    if n_small:
        print(f'  {n_small:,} skipped under {args.min_size}px')
    if n_noimg:
        print(f'  {n_noimg:,} source images not found on any root')
    if not args.execute:
        print('\nnothing written. re-run with --execute')
    return 0


if __name__ == '__main__':
    sys.exit(main())
