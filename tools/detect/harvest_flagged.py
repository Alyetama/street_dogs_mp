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

Framing is production's: gate_store.cut_crops, imported, on the PIL-decoded
full-resolution frame -- the dataset builder (build_dataset.py, through
crop_sources.py) cuts the same way and no longer runs this tool; it stays for
cutting a ledger by hand.

Output is laid out for ``yolo classify``, so the crops can be dropped straight
into an existing dataset's negative class:

    <out>/not_dog/<image_id>_<det_idx>.jpg

Why these are worth more than annotator negatives: every one of them fooled
the detector at inference time on real corpus imagery. Boxes an annotator drew
are mostly distant specks (median short side 36 px vs 208 px for dogs); these
sit in the same size range as true detections, so a classifier trained on them
cannot fall back on "small means not a dog".

    python tools/detect/harvest_flagged.py --out <datasets>/hard_negatives
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
from fn_audit import GEN_FIELD, gen_of  # noqa: E402

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


def read_corrections(path, *, gen, counts=None):
    """{(image_id, det_idx): (x1, y1, x2, y2)} -- the REVIEW page's redraws.

    Only redraws made on detector generation `gen` -- the generation of the
    store the caller joins them against (crop_sources.join_gen). A record's
    `gen` (absent: gen 1, fn_audit.gen_of) says whose det_idx it carries; one
    of another generation would move a box nobody redrew, so it is skipped
    and counted in `counts['other_gen']`. Last write wins, within `gen`.

    Records whose `crop` starts with 'audit:' are skipped:
    the audit pages write their redraws into the same file under the same
    (image_id, det_idx) key, and they mean something else. A leash-audit
    redraw takes in the leash and often the person holding it (92% of them
    enlarge the box for exactly that); applied to a review-page flag it would
    turn a dog-bin hard positive into a crop of a dog, a leash and a person.
    The audit redraws are read by crop_sources.read_audit_corrections, per
    stage, for the audit rows they belong to.
    """
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
                if str(r.get('crop') or '').startswith('audit:'):
                    continue
                if gen_of(r.get(GEN_FIELD)) != gen:
                    if counts is not None:
                        counts['other_gen'] += 1
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


def read_flags(path, want='false_positive', *, gen, counts=None):
    """{image_id: record} for one label, last write wins -- among the flags
    made on detector generation `gen` only.

    A flag names its box by (image_id, conf), and a conf means a box only
    within one generation: a flag of another (absent `gen`: gen 1,
    fn_audit.gen_of) is skipped and counted in `counts['other_gen']`.
    """
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
                if not isinstance(r, dict):
                    continue
                if r.get('label') == want and r.get('image_id'):
                    if gen_of(r.get(GEN_FIELD)) != gen:
                        if counts is not None:
                            counts['other_gen'] += 1
                        continue
                    out[str(r['image_id'])] = r
    except OSError:
        pass
    return out


def boxes_for(image_ids, detect_root, python_bin=None):
    """{image_id: [(det_idx, x1, y1, x2, y2, conf)]} from the parquet store.

    ONE ROW PER DETECTION, not per stored row. The store keys on
    (image_id, det_idx, cell, drive): a frame the harvest wrote into several
    cells has its detections repeated once per placement, identical det_idx,
    identical geometry, identical conf (store.unique_src() documents why the
    twins exist and why any one copy is as good as another). Left in, the
    conf-matching below sees the SAME box 2-16 times, len(hit) > 1, and files
    a perfectly unambiguous flag under `ambiguous` -- 117 of 136 rejections
    on the live ledgers were one detection counted several times, and each
    one is a human verdict dropped out of every dataset built afterwards.

    NOT store.unique_src(): that collapses to one row per IMAGE, which would
    keep a single detection per frame and throw the rest away. The partition
    here has to be (image_id, det_idx), the same one gate_store.plan() uses.
    """
    if not image_ids:
        return {}
    ids = ','.join("'%s'" % i for i in image_ids)
    det = store._sql_src(store._store_globs(detect_root, 'det'))
    sql = (f"SELECT CAST(image_id AS VARCHAR), det_idx, x1, y1, x2, y2, conf "
           f"FROM {det} WHERE CAST(image_id AS VARCHAR) IN ({ids}) "
           # deterministic survivor, so two runs cut the same crop
           f"QUALIFY row_number() OVER (PARTITION BY image_id, det_idx "
           f"ORDER BY cell, drive) = 1")
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
    # NO --pad. The crop is framed by gate_store.cut_crops, imported, so it is
    # framed the way production frames the box it will be judging. The old
    # default here was a wider margin than production's own, which meant
    # every harvested crop showed the model more context than any crop it
    # is ever asked about. Passing the
    # old flag is an error rather than silently ignored.
    p.add_argument('--pad', help=argparse.SUPPRESS)
    p.add_argument('--min-size',
                   type=int,
                   default=0,
                   help='skip boxes whose short side is under this (default '
                   '0: keep everything gate_store.cut_crops keeps, which is '
                   'what production judges -- the old 64-px floor dropped '
                   '657 false positives and 202 true positives, while 35%% '
                   'of the gen-2 boxes are under 48 px)')
    p.add_argument('--detect-root',
                   help='the store whose det_idx and confidences the ledger '
                   'was recorded against (default: the gen-1 store '
                   'data/ledger_detect_root.txt pins; a disagreeing '
                   '$DETECT_ROOT is refused). A store whose shards are not '
                   'only gen=0001 is refused unless --any-generation')
    p.add_argument('--any-generation',
                   action='store_true',
                   help='accept a --detect-root that holds another detector '
                   'generation. Only for a ledger recorded against THAT '
                   'store: every existing ledger carries gen-1 det_idx and '
                   'confidences, which match nothing (or the wrong box) in '
                   'another generation. Either way only rows tagged with '
                   'the store\'s own generation (`gen`; untagged = 1) are '
                   'read.')
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
    if args.pad is not None:
        p.error('--pad is gone: crops are framed by gate_store.cut_crops '
                '(PAD_FRAC, PAD_PX), the production framing')

    import gate_store
    import crop_sources
    from PIL import Image
    if args.detect_root:
        detect_root = args.detect_root
        why = crop_sources.ledger_store_refusal(detect_root)
        if why and not args.any_generation:
            raise SystemExit('--detect-root refused: %s. Pass '
                             '--any-generation only if the ledger was '
                             'recorded against this store.' % why)
    else:
        detect_root = crop_sources.default_detect_root()
    # the flags and redraws made on this store's detector generation, only
    gen = crop_sources.join_gen(detect_root)

    ledger = args.flags or LEDGERS[args.label]
    cls_dir = 'dog' if args.label == 'true_positive' else 'not_dog'
    skipped = {'other_gen': 0}
    flags = read_flags(ledger, args.label, gen=gen, counts=skipped)
    if skipped['other_gen']:
        print(f'  {skipped["other_gen"]:,} {args.label} flag(s) of another '
              f'detector generation skipped (this store is gen {gen})')
    if not flags:
        print(f'no gen-{gen} {args.label} flags in {ledger}')
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

    boxes = boxes_for(list(flags), detect_root)
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
                # A flag without a confidence cannot be matched to its box.
                # Keeping every detection here -- which this branch used to do
                # -- silently bypasses the guard the --all-detections help
                # text describes: 94 of 314 flagged images carry 2-12
                # detections, and the siblings are often real dogs. One
                # detection is fine (nothing to disambiguate); more is
                # ambiguous, same as several boxes at equal confidence.
                if len(dets) == 1:
                    kept[iid] = dets
                else:
                    ambiguous += 1
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
    corr = (read_corrections(args.corrections, gen=gen)
            if args.corrections else {})
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
    placed = cells_for(list(boxes), detect_root)
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
        img = None
        if args.execute:
            try:
                img = Image.open(path)
                img.load()
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            except Exception:             # noqa: BLE001 - counted below
                img = None
        keep = []
        for di, x1, y1, x2, y2, conf in dets:
            w, h = x2 - x1, y2 - y1
            if min(w, h) < args.min_size:
                n_small += 1
                continue
            n_ok += 1
            keep.append((di, x1, y1, x2, y2))
        if not args.execute or img is None:
            continue
        # production's own cutter: same margin, same MIN_SIDE, full resolution
        for di, crop in gate_store.cut_crops(img, keep):
            crop.save(os.path.join(dst, f'{iid}_{di}.jpg'), 'JPEG',
                      quality=95)
        img.close()
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
