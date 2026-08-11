#!/usr/bin/env python3
"""Add reviewer-flagged false positives to the detector set as background images.

The detector finds dogs. Every crop flagged "not a dog" in the review page is
therefore a real detector error -- a goat, a cow, a bin bag it boxed as a dog --
and the frames those came from are the highest-value negatives available:
the model already got them wrong, on the exact imagery it runs against.

YOLO learns a negative from an image with an EMPTY label file. So this copies
the full frame and writes a zero-byte .txt beside it.

Three constraints shape what actually gets added.

ONLY FULLY-FLAGGED FRAMES. A frame with four detections where the reviewer
flagged one says nothing about the other three. Adding it whole, unlabelled,
tells the detector there is no dog in a frame that may well contain three --
which trains it to MISS dogs, the one error this project cannot recover from.
Measured on the current ledger: 842 of 1,177 flagged frames carry exactly one
detection and are safe; 335 carry 2-44 and are not.

BACKGROUNDS ARE CAPPED. The set already contains 209 empty labels in 1,985
train images -- 10.5%, at the top of ultralytics' recommended 0-10%. Adding all
842 would make it 37% and buy precision with recall. --target-frac decides how
far past the current ratio to go; the default 0.20 adds roughly 235.

VAL IS NOT TOUCHED, AND NOT LEAKED INTO. Backgrounds go to train only, so mAP
and recall stay directly comparable to train-30 rather than being measured
against a moved goalpost. "Not in val" is decided by SEQUENCE, not by
image_id: frames a second apart are near-duplicates, and a background drawn
from a val frame's own pass both teaches "no dog on this street" and hands
the model the val scene in advance.

    python tools/detect/build_detector_negatives.py \\
        --src <detector dataset> --out <new dataset> --execute

READ-ONLY on --src and on the flag ledger.
"""

import argparse
import collections
import json
import os
import random
import re
import shutil
import sys

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ID_RE = re.compile(r'\d{6,}')


def ids_in(d):
    out = set()
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for f in names:
        m = ID_RE.search(f)
        if m:
            out.add(m.group(0))
    return out


def flagged_ids(ledger):
    out = {}
    try:
        with open(ledger) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if r.get('image_id') and r.get('label') == 'false_positive':
                    out[str(r['image_id'])] = r
    except OSError:
        pass
    return out


def detections_per_image(ids, repo):
    """{image_id: n detections} from the predictions store.

    count(DISTINCT det_idx), not count(*). The store keys on
    (image_id, det_idx, cell, drive) and a frame the harvest wrote into two
    cells carries every one of its detections twice -- so count(*) reads a
    single-detection frame as a two-detection one and the caller excludes it
    as "2+ detections with the others unverified", about a sibling that does
    not exist. Measured over the flagged ids on the live store: 2,062 have
    exactly one real detection, 1,977 have exactly one ROW; the row test
    silently loses 85 of the highest-value negatives there are.

    Not store.unique_src() either -- that keeps one row per IMAGE, which
    would report 1 for every frame and make this test meaningless.
    """
    sys.path.insert(0, os.path.join(repo, 'tools', 'detect'))
    import duckdb
    import store as _store
    root = _store.get_detect_root()
    src = _store._sql_src(_store._store_globs(root, 'det'))
    con = duckdb.connect()
    try:
        lst = "','".join(sorted(ids))
        rows = con.execute(
            f"SELECT CAST(image_id AS VARCHAR), count(DISTINCT det_idx) "
            f"FROM {src} WHERE CAST(image_id AS VARCHAR) IN ('{lst}') "
            f"GROUP BY 1"
        ).fetchall()
    finally:
        con.close()
    return {i: n for i, n in rows}


def resolve_originals(ids, roots_file, repo):
    """{image_id: path to the ORIGINAL full jpg} via the store's (cell, drive).

    Same route harvest_flagged.py uses: the sweep already recorded which cell
    and drive each image came from, so this is a path join rather than a walk
    over 32.5M files.
    """
    sys.path.insert(0, os.path.join(repo, 'tools', 'detect'))
    import store as _store
    from harvest_flagged import cells_for, roots_by_drive
    try:
        roots = [ln.strip() for ln in open(roots_file)
                 if ln.strip() and not ln.startswith('#')]
    except OSError:
        raise SystemExit(f'no roots file at {roots_file}')
    placed = cells_for(list(ids), _store.get_detect_root())
    by_drive = roots_by_drive(roots)
    out = {}
    for iid, (cell, drive) in placed.items():
        root = by_drive.get(drive)
        if not root:
            continue
        p = os.path.join(root, cell, 'ground_animal_images', f'{iid}.jpg')
        if os.path.exists(p):
            out[iid] = p
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True,
                    help='existing detector dataset (images/ and labels/)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--roots-file',
                    default=os.path.join(REPO, 'data', 'catalog_dirs.txt'),
                    help='grid_runs roots, one per line -- the ORIGINAL jpgs')
    ap.add_argument('--ledger',
                    default=os.path.join(REPO, 'data', 'hard_negatives',
                                         'labels.jsonl'))
    ap.add_argument('--target-frac', type=float, default=0.20,
                    help='background share of the TRAIN split to aim for. The '
                         'source is already at 0.105; ultralytics suggests '
                         '0-10%%, so past ~0.25 expect precision to rise and '
                         'recall to fall -- the wrong trade on a one-pass sweep')
    ap.add_argument('--max-per-sequence', type=int, default=2,
                    help='cap frames kept per Mapillary sequence, so one long '
                         'pass past one goat cannot dominate the negatives')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--duckdb-python',
                    default=os.environ.get('DUCKDB_PYTHON') or sys.executable)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tr_img = os.path.join(args.src, 'images', 'train')
    tr_lab = os.path.join(args.src, 'labels', 'train')
    if not os.path.isdir(tr_img):
        raise SystemExit(f'no images/train under {args.src}')
    n_train = len(os.listdir(tr_img))
    n_bg = sum(1 for f in os.listdir(tr_lab)
               if f.endswith('.txt')
               and os.path.getsize(os.path.join(tr_lab, f)) == 0)
    print(f'source: {n_train} train images, {n_bg} already background '
          f'({n_bg / max(n_train, 1):.1%})')

    flags = flagged_ids(args.ledger)
    print(f'{len(flags)} flagged false positives in the ledger')

    # 1. never contradict an existing label
    in_src = ids_in(tr_img) | ids_in(os.path.join(args.src, 'images', 'val'))
    cand = {i for i in flags if i not in in_src}
    print(f'  {len(flags) - len(cand)} already in the detector set '
          f'-> {len(cand)} candidates')

    # 2. only frames whose every detection was flagged
    per = detections_per_image(cand, REPO)
    safe = {i for i in cand if per.get(i) == 1}
    print(f'  {len(cand) - len(safe)} carry 2+ detections with the others '
          f'unverified -> {len(safe)} safe')

    # 3. the ORIGINAL jpg, resolved through the store's (cell, drive)
    #
    # NOT data/hard_negatives/full/. Those frames are preview renders: the
    # sweep draws the detection box onto them (sweep.py, cv2.rectangle) and
    # renders from the 1280 letterbox, so they carry a burned-in yellow
    # rectangle and grey padding. Training on them would teach the detector
    # about the annotation overlay instead of the scene -- and the negative
    # would be perfectly correlated with the artefact, which is the worst
    # possible shortcut to hand a model.
    have = resolve_originals(safe, args.roots_file, REPO)
    print(f'  {len(safe) - len(have)} could not be resolved to an original '
          f'jpg -> {len(have)} usable')

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rebuild_crop_dataset import resolve_sequences

    # 3b. nothing from a sequence val already grades on
    #
    # Step 1 excluded val by IMAGE_ID, which is the split rule this project
    # measured at 63-71% leakage and abandoned: Mapillary frames a second
    # apart are near-duplicates, so frame N as a background in train and
    # frame N+2 as a labelled dog in val means val is scoring a scene the
    # model has partly memorised -- and has been told contains nothing.
    # build_dogdet_v3.py already refuses this; the rule is the same here.
    val_ids = ids_in(os.path.join(args.src, 'images', 'val'))
    val_seqs = {s for s in resolve_sequences(
        val_ids, REPO, args.duckdb_python).values() if s}
    seq = resolve_sequences(set(have), REPO, args.duckdb_python)
    leaked = {i for i in have if seq.get(i) in val_seqs}
    for i in leaked:
        have.pop(i, None)
    print(f'  {len(leaked)} share a sequence with a val frame -> '
          f'{len(have)} left ({len(val_seqs)} val sequences on record)')

    # 4. one long pass past one goat must not dominate
    by_seq = collections.defaultdict(list)
    for i in sorted(have):
        by_seq[seq.get(i) or f'noseq:{i}'].append(i)
    capped = []
    dropped_cap = 0
    for s in sorted(by_seq):
        lst = sorted(by_seq[s])
        rng.shuffle(lst)
        if len(lst) > args.max_per_sequence:
            dropped_cap += len(lst) - args.max_per_sequence
            lst = lst[:args.max_per_sequence]
        capped += lst
    print(f'  {dropped_cap} over the per-sequence cap of '
          f'{args.max_per_sequence} -> {len(capped)} after capping')

    # 5. how many to actually add, from the target background share
    #    (n_bg + x) / (n_train + x) = frac
    f = args.target_frac
    want = int(round((f * n_train - n_bg) / (1 - f))) if f < 1 else len(capped)
    want = max(0, min(want, len(capped)))
    capped.sort()
    rng.shuffle(capped)
    chosen = sorted(capped[:want])
    final_bg = n_bg + len(chosen)
    final_n = n_train + len(chosen)
    print(f'\nadding {len(chosen)} background frames to TRAIN only')
    print(f'  background share {n_bg}/{n_train} = {n_bg/n_train:.1%}'
          f'  ->  {final_bg}/{final_n} = {final_bg/final_n:.1%}')
    print('  val is untouched, so recall and mAP stay comparable to the '
          'source run')

    if not args.execute:
        print('\nnothing written. re-run with --execute')
        return 0

    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    for sub in ('images', 'labels'):
        for split in ('train', 'val'):
            s = os.path.join(args.src, sub, split)
            d = os.path.join(args.out, sub, split)
            for f in os.listdir(s):
                shutil.copy2(os.path.join(s, f), os.path.join(d, f))
    for iid in chosen:
        shutil.copy2(have[iid],
                     os.path.join(args.out, 'images', 'train', f'{iid}.jpg'))
        open(os.path.join(args.out, 'labels', 'train', f'{iid}.txt'),
             'w').close()

    yml = os.path.join(args.out, 'dataset.yaml')
    with open(yml, 'w') as fh:
        fh.write(f'path: {os.path.abspath(args.out)}\n'
                 'train: images/train\nval: images/val\n\nnames:\n  0: target\n')
    man = {
        'src': args.src,
        'flagged_in_ledger': len(flags),
        'already_in_src': len(flags) - len(cand),
        'multi_detection_skipped': len(cand) - len(safe),
        'unresolved_to_original': len(safe) - len(have) - len(leaked),
        'val_sequence_skipped': len(leaked),
        'over_sequence_cap': dropped_cap,
        'added_backgrounds': len(chosen),
        'target_frac': args.target_frac,
        'background_share_before': round(n_bg / n_train, 4),
        'background_share_after': round(final_bg / final_n, 4),
        'train_images_after': final_n,
        'val_untouched': True,
        'seed': args.seed,
        'max_per_sequence': args.max_per_sequence,
        'added_image_ids': chosen,
    }
    with open(os.path.join(args.out, 'negatives_manifest.json'), 'w') as fh:
        json.dump(man, fh, indent=1)
    print(f'\nwrote {final_n} train / '
          f'{len(os.listdir(os.path.join(args.out, "images", "val")))} val '
          f'-> {args.out}')
    print(f'manifest: {os.path.join(args.out, "negatives_manifest.json")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
