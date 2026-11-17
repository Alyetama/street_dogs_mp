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
# The store the flags are matched against. main() sets it from --detect-root;
# unset, detect_root() resolves the pinned ledger root.
DETECT_ROOT = None


def _crop_sources():
    sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
    import crop_sources
    return crop_sources


def detect_root():
    """The store the flags were made on: --detect-root, else the pinned
    ledger root (crop_sources.default_detect_root), NEVER the live one.

    A flag is an image_id and a gen-1 conf. "Every detection on this frame
    was flagged" is a statement about the gen-1 boxes; asked of the live
    store after the switch, 1,725 single-detection frames have no gen-2
    detection at all and drop out, and frames newly single in gen 2 go in
    as "no dog here" on a box nobody looked at.
    """
    global DETECT_ROOT
    if not DETECT_ROOT:
        DETECT_ROOT = _crop_sources().default_detect_root()
    return DETECT_ROOT


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
    """{image_id: n detections} from the predictions store (detect_root()).

    Given a mapping {image_id: flag row} rather than bare ids, a frame with
    ONE detection whose conf is not within CONF_TOL (0.006) of the flag's --
    or whose flag carries no conf -- comes back as None, not 1: its lone box
    is not the one the person flagged, so nobody vouched for it and an empty
    label would call it "no dog here" on nobody's word.

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
    root = detect_root()
    src = _store._sql_src(_store._store_globs(root, 'det'))
    con = duckdb.connect()
    try:
        lst = "','".join(sorted(ids))
        rows = con.execute(
            f"SELECT CAST(image_id AS VARCHAR), count(DISTINCT det_idx), "
            f"min(conf), max(conf) "
            f"FROM {src} WHERE CAST(image_id AS VARCHAR) IN ('{lst}') "
            f"GROUP BY 1"
        ).fetchall()
    finally:
        con.close()
    out = {i: n for i, n, _, _ in rows}
    if isinstance(ids, dict):
        tol = _crop_sources().CONF_TOL
        for i, n, lo, hi in rows:
            want = (ids.get(i) or {}).get('conf')
            if n == 1 and (want is None or lo is None
                           or abs(float(lo) - float(want)) > tol
                           or abs(float(hi) - float(want)) > tol):
                out[i] = None
    return out


def resolve_originals(ids, roots_file, repo):
    """{image_id: path to the ORIGINAL full jpg} via the store's (cell, drive).

    Same route harvest_flagged.py uses: the sweep already recorded which cell
    and drive each image came from, so this is a path join rather than a walk
    over 32.5M files.
    """
    sys.path.insert(0, os.path.join(repo, 'tools', 'detect'))
    from harvest_flagged import cells_for, roots_by_drive
    try:
        roots = [ln.strip() for ln in open(roots_file)
                 if ln.strip() and not ln.startswith('#')]
    except OSError:
        raise SystemExit(f'no roots file at {roots_file}')
    placed = cells_for(list(ids), detect_root())
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


def main(argv=None):
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
    ap.add_argument('--hidden-dogs', default=None,
                    help='the hidden-dog store (default: '
                         'data/hidden_dogs/flagged.json). A frame it flags is '
                         'never made a background: an independent model sees '
                         'a dog in it')
    ap.add_argument('--no-hidden-dogs', action='store_true',
                    help='ignore the hidden-dog store')
    ap.add_argument('--detect-root', default=None,
                    help='the store the flags are matched against (default: '
                         'the pinned ledger root, data/ledger_detect_root.txt'
                         '). The flags carry gen-1 confs; another store '
                         'matches them against boxes nobody flagged, so one '
                         'whose shards are not only gen=0001 is refused '
                         'unless --any-generation')
    ap.add_argument('--any-generation', action='store_true',
                    help='accept a --detect-root that holds another detector '
                         'generation. Only for flags recorded against THAT '
                         'store: every existing flag carries a gen-1 conf.')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args(argv)
    global DETECT_ROOT
    DETECT_ROOT = None
    if args.detect_root:
        if not os.path.isdir(os.path.join(args.detect_root, 'shards')):
            raise SystemExit(f'{args.detect_root} has no shards/ -- it is '
                             f'not a detector store')
        why = _crop_sources().ledger_store_refusal(args.detect_root)
        if why and not args.any_generation:
            raise SystemExit(f'--detect-root refused: {why}. Pass '
                             f'--any-generation only if the flags were '
                             f'recorded against this store.')
        DETECT_ROOT = os.path.abspath(args.detect_root)

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

    # 2. only frames whose every detection was flagged: ONE detection, and
    #    that one the flagged box (its conf within 0.006 of the flag's)
    per = detections_per_image({i: flags[i] for i in cand}, REPO)
    safe = {i for i in cand if per.get(i) == 1}
    not_flagged = sorted(i for i in cand if i in per and per[i] is None)
    print(f'  {len(not_flagged)} carry one detection that is not the flagged '
          f'box (conf off by more than 0.006, or none on the flag)')
    print(f'  {len(cand) - len(safe) - len(not_flagged)} carry 2+ detections '
          f'with the others unverified, or none -> {len(safe)} safe')

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

    # 3c. nor from a test set, or a sequence one measures on -- the same rule
    # as val, for the label batches kept out of training (frame_labels.py)
    import frame_labels
    test = frame_labels.test_sets()
    in_test = {i for i in have
               if i in test['ids'] or seq.get(i) in test['seqs']}
    for i in in_test:
        have.pop(i, None)
    print(f'  {len(in_test)} belong to a test set or share its sequence -> '
          f'{len(have)} left ({", ".join(test["batches"]) or "no test sets"})')

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
    # 6. never a background where an independent model sees a dog
    #
    # Dropped AFTER the draw, not before it, and not refilled: the frames
    # drawn are the ones that were audited, and a refill would bring in
    # backgrounds nobody has looked for dogs in.
    import hidden_dogs
    hidden_path = None if args.no_hidden_dogs else (args.hidden_dogs
                                                    or hidden_dogs.STORE)
    flagged = hidden_dogs.read(hidden_path) if hidden_path else {}
    chosen, hidden_out = hidden_dogs.screen(chosen, flagged)
    print(f'  {len(hidden_out)} drawn frame(s) hold a dog an independent '
          f'model sees ({len(flagged)} flagged on record) -> left out, not '
          f'replaced')
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
        'multi_detection_skipped': len(cand) - len(safe) - len(not_flagged),
        'lone_detection_not_flagged': len(not_flagged),
        # the store the flags were matched against
        'detect_root': DETECT_ROOT,
        'unresolved_to_original': (len(safe) - len(have) - len(leaked) -
                                   len(in_test)),
        'val_sequence_skipped': len(leaked),
        'test_set_skipped': sorted(in_test),
        'test_sets': test['batches'],
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
        'hidden_dogs': {
            'store': (os.path.relpath(hidden_path, REPO)
                      if hidden_path else None),
            'flagged_frames': len(flagged),
            'left_out': sorted(hidden_out),
        },
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
