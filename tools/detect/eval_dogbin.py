#!/usr/bin/env python3
"""
Evaluate the binary dog / not_dog gate as a GATE, not as a classifier.

Top-1 accuracy is the wrong headline for this model. The val split is
~4.8:1 dog-heavy, so "always say dog" already scores ~82.8%: a number in the
high 80s would mean the model had learnt nothing. What matters is the
threshold curve -- the gate sits in front of the leash classifier, so the
question is:

    how many not-dog crops can I throw away before I start throwing away
    real dogs?

So this reports ROC AUC and average precision, then a table of operating
points anchored on DOG RECALL (the thing that is expensive to lose), giving
the not-dog rejection rate achievable at each. It also scores the dashboard's
flagged false positives separately: those are negatives drawn from the real
sweep distribution rather than from an annotator's crops, which is the
distribution the gate will actually meet in production.

Wilson intervals throughout -- val has only ~104 negatives, and a rejection
rate quoted to three decimals off 104 samples would be false precision.

    python tools/detect/eval_dogbin.py --weights <run>/weights/best.pt
"""

import argparse
import json
import math
import os
import re
import sys

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def load_dir(d):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.lower().endswith(exts))


def id_of_name(path):
    """First 6+ digit run in the basename -- the image_id for every crop
    naming scheme this pipeline uses (flag_<id>_<det>, <id>_<det>,
    <ts>_<id>_<conf> would yield the ts, but acceptance crops are harvested
    <id>_<det> names, where the id comes first)."""
    m = re.search(r'\d{6,}', os.path.basename(path))
    return m.group(0) if m else None


def drop_trained_on(paths, data_root):
    """(held_out, contaminated): flagged crops the model has never seen.

    The whole point of the sweep-negative table is that it measures the gate on
    a distribution it was NOT fit to. But rebuild_crop_dataset.py folds these
    same flagged detections in as --extra-negatives, so by default the two sets
    overlap and the headline is partly a train-set score.

    Nobody noticed because the two naming schemes never collide:

        dataset : flag_<image_id>_<det_idx>.jpg
        flag dir: <ts>_<image_id>_<conf>.jpg

    A filename comparison returns zero overlap. Matching on the image_id inside
    the name found 360 of 1072 -- 297 in train, 63 in val -- and removing them
    moved rejection at t=0.5 from 0.7740 [0.747,0.799] to 0.7032 [0.669,0.736],
    two intervals that do not touch. The contaminated median P(dog) was 4.4x
    lower than the honest one, which is the memorisation showing through.

    Excluded crops are returned so the caller reports the count. Silence here
    would restore exactly the bug this function exists to prevent.
    """
    if not data_root:
        return paths, []
    # Every long digit run in the name, not a single fixed pattern. Crops of
    # the same detection reach this tool under at least three schemes --
    #     flag_<image_id>_<det>.jpg     dataset member
    #     <ts>_<image_id>_<conf>.jpg    dashboard thumbnail
    #     <image_id>_<det>.jpg          harvest_flagged output
    # -- and a regex written for one silently fails to parse the others, which
    # means silently NOT excluding them. That is the whole bug: a guard that
    # cannot read the filename reports zero contamination and looks like it
    # passed. Over-matching is the safe direction here; a wrongly excluded
    # crop costs one sample, a wrongly included one costs the conclusion.
    def id_candidates(name):
        return set(re.findall(r'\d{6,}', os.path.basename(name)))

    seen = set()
    for split in ('train', 'val'):
        for cls in ('not_dog', 'dog'):
            try:
                names = os.listdir(os.path.join(data_root, split, cls))
            except OSError:
                continue
            for f in names:
                seen |= id_candidates(f)
    if not seen:
        return paths, []
    keep, bad = [], []
    for p in paths:
        (bad if (id_candidates(p) & seen) else keep).append(p)
    return keep, bad


def readable(paths):
    """(good, bad): paths PIL can actually decode.

    The flag ledger records a crop as copied before anything verifies the
    bytes, so a truncated write leaves an entry pointing at a stub. One such
    3-byte file used to abort the whole evaluation inside ultralytics with
    PIL.UnidentifiedImageError, after the val table had already been printed
    but before the number that decides promotion. Losing a 40-minute run to
    one bad file out of 1023 is not a reasonable failure mode.

    Dropped files are RETURNED, not swallowed: they shrink the denominator of
    a rejection rate, so the caller has to say how many went missing.
    """
    from PIL import Image
    good, bad = [], []
    for p in paths:
        try:
            Image.open(p).verify()
            good.append(p)
        except Exception:
            bad.append(p)
    return good, bad


def dog_prob(model, paths, batch=64, imgsz=640, device=0):
    """P(dog) for each path, in order."""
    out = []
    names = model.names
    dog_i = [i for i, n in names.items() if str(n).lower() == 'dog']
    if not dog_i:
        raise SystemExit(f'no "dog" class in model.names={names}')
    dog_i = dog_i[0]
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        for r in model.predict(chunk, imgsz=imgsz, verbose=False,
                               device=device):
            out.append(float(r.probs.data[dog_i]))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True)
    # no machine path in a tracked file: $DOGBIN_DATASET, else pass --data
    ap.add_argument('--data', default=os.environ.get('DOGBIN_DATASET'),
                    help='dataset root with <split>/dog and <split>/not_dog '
                         '(or set $DOGBIN_DATASET)')
    ap.add_argument('--split', default='val')
    ap.add_argument('--hard-negatives',
                    default=os.path.join(REPO, 'data', 'hard_negatives',
                                         'crops'),
                    help='dashboard-flagged false positives: negatives from '
                         'the REAL sweep distribution')
    ap.add_argument('--acceptance-set',
                    default=os.path.join(REPO, 'data',
                                         'dogbin_acceptance_set.json'),
                    help='reserved image_ids (reserve_acceptance_set.py); '
                         'scored as their own table, at full resolution')
    ap.add_argument('--acceptance-crops',
                    default=os.path.join(REPO, 'data', 'harvest', 'v4', 'fp',
                                         'not_dog'),
                    help='full-resolution harvest dir the reserved ids are '
                         'drawn from')
    ap.add_argument('--imgsz', type=int, default=640)
    # small by default on purpose: this tool is meant to be run WHILE the
    # 32.5M-image sweep holds most of the GPU, and a batch of 64 OOMs there
    ap.add_argument('--batch', type=int, default=8)
    # the GPU is usually busy with the sweep AND a training run; --device cpu
    # is slower but never contends for memory that a multi-hour job needs
    ap.add_argument('--device', default='0')
    args = ap.parse_args()

    from ultralytics import YOLO
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score

    if not args.data:
        raise SystemExit('pass --data or set $DOGBIN_DATASET')
    model = YOLO(args.weights)
    print(f'model   : {args.weights}')
    print(f'classes : {model.names}')

    root = os.path.join(args.data, args.split)
    pos = load_dir(os.path.join(root, 'dog'))
    neg = load_dir(os.path.join(root, 'not_dog'))
    if not pos or not neg:
        raise SystemExit(f'need both classes under {root}')
    prior = len(pos) / (len(pos) + len(neg))
    print(f'\n{args.split}: {len(pos)} dog / {len(neg)} not_dog '
          f'({len(pos)/max(len(neg),1):.1f}:1)')
    print(f'majority-class baseline (always "dog"): {prior:.4f}  '
          f'<- any top-1 near this means nothing was learnt')

    p = np.array(dog_prob(model, pos, args.batch, args.imgsz, args.device))
    n = np.array(dog_prob(model, neg, args.batch, args.imgsz, args.device))
    y = np.r_[np.ones_like(p), np.zeros_like(n)]
    s = np.r_[p, n]

    auc = roc_auc_score(y, s)
    ap_dog = average_precision_score(y, s)
    ap_neg = average_precision_score(1 - y, -s)
    top1 = ((p >= 0.5).sum() + (n < 0.5).sum()) / len(s)
    print(f'\nROC AUC              : {auc:.4f}')
    print(f'AP (dog, positive)   : {ap_dog:.4f}')
    print(f'AP (not_dog)         : {ap_neg:.4f}   '
          f'(chance = {1 - prior:.4f})')
    lo, hi = wilson(int((p >= .5).sum() + (n < .5).sum()), len(s))
    print(f'top-1 @0.5           : {top1:.4f}  [{lo:.3f}, {hi:.3f}]  '
          f'vs {prior:.4f} baseline')

    # ---- the gate table: what does each dog-recall level cost/buy? ----
    print('\nGate operating points -- threshold on P(dog), keep if >= t')
    print(f'{"dog recall":>10} {"thresh":>7} {"not_dog rejected":>17} '
          f'{"95% CI":>16} {"dogs lost":>10}')
    for target in (1.00, 0.999, 0.995, 0.99, 0.98, 0.95, 0.90):
        # largest threshold that still keeps >= target of the dogs
        k = math.floor((1 - target) * len(p))
        t = float(np.sort(p)[k]) if k < len(p) else 0.0
        kept_dog = int((p >= t).sum())
        rej_neg = int((n < t).sum())
        rlo, rhi = wilson(rej_neg, len(n))
        print(f'{kept_dog/len(p):>10.4f} {t:>7.4f} '
              f'{rej_neg/len(n):>16.4f} '
              f'[{rlo:.3f},{rhi:.3f}]{"":>2} {len(p)-kept_dog:>10}')

    # ---- the honest test: negatives from the live sweep ----
    hn = load_dir(args.hard_negatives)
    hn, contaminated = drop_trained_on(hn, args.data)
    if contaminated:
        print(f'\nEXCLUDED {len(contaminated)} flagged crop(s) that are IN the '
              f'dataset at {os.path.basename(args.data or "")} -- the model was '
              f'fit or early-stopped on them, so scoring them here would be a '
              f'train-set score reported as a held-out one.')
    hn, unreadable = readable(hn)
    if unreadable:
        print(f'\nSKIPPED {len(unreadable)} unreadable crop(s) '
              f'(truncated or non-image); they are excluded from the counts '
              f'below:')
        # NOT `for p in ...`: p is the positive-score array, and rebinding it
        # to a filename here would poison every later use in this function.
        for bad_path in unreadable[:5]:
            print(f'    {os.path.basename(bad_path)}  '
                  f'({os.path.getsize(bad_path)} bytes)')
        if len(unreadable) > 5:
            print(f'    ... and {len(unreadable) - 5} more')
    if hn:
        h = np.array(dog_prob(model, hn, args.batch, args.imgsz, args.device))
        print(f'\nFlagged false positives from the live sweep: {len(hn)} crops')
        print('  (all are not_dog by construction -- you flagged them)')
        for t in (0.5, 0.9, 0.95, 0.99):
            rej = int((h < t).sum())
            lo, hi = wilson(rej, len(h))
            print(f'  rejected at t={t:<5}: {rej:>4}/{len(h)} = '
                  f'{rej/len(h):.4f}  [{lo:.3f}, {hi:.3f}]')
        print(f'  median P(dog) on these: {float(np.median(h)):.4f}')
        curated = float(np.median(n))
        print(f'  median P(dog) on curated val negatives: {curated:.4f}')
        if float(np.median(h)) > curated + 0.05:
            print('  -> the gate is measurably WEAKER on real sweep negatives '
                  'than on annotator negatives; the curated number is '
                  'optimistic for deployment')
    else:
        print(f'\n(no crops under {args.hard_negatives} -- skipped the '
              f'real-distribution check)')

    # ---- THE number: the reserved acceptance set, at full resolution ------
    # Everything above is diagnostics. This table is the one a promotion may
    # cite: negatives the model has never seen in ANY form (the ids are
    # excluded from every split by rebuild_crop_dataset.py), scored from the
    # full-resolution harvest -- not the ~160px dashboard thumbnails the
    # --hard-negatives dir holds, which are a different distribution from
    # anything the gate meets in production.
    acc_ids = set()
    if args.acceptance_set and os.path.exists(args.acceptance_set):
        with open(args.acceptance_set) as fh:
            acc_ids = set(json.load(fh).get('image_ids') or [])
    if acc_ids and args.acceptance_crops and os.path.isdir(args.acceptance_crops):
        acc = [f for f in load_dir(args.acceptance_crops)
               if id_of_name(f) in acc_ids]
        acc, contaminated = drop_trained_on(acc, args.data)
        acc, _ = readable(acc)
        if contaminated:
            print(f'\nWARNING: {len(contaminated)} reserved crop(s) are in '
                  f'the dataset -- the reservation is not protecting this '
                  f'model. Their scores are excluded, but investigate.')
        if acc:
            a = np.array(dog_prob(model, acc, args.batch, args.imgsz,
                                  args.device))
            print(f'\nACCEPTANCE SET (reserved, never trained on, full-res): '
                  f'{len(acc)} of {len(acc_ids)} reserved ids')
            for t in (0.5, 0.9, 0.95, 0.99):
                rej = int((a < t).sum())
                lo, hi = wilson(rej, len(a))
                print(f'  rejected at t={t:<5}: {rej:>4}/{len(a)} = '
                      f'{rej/len(a):.4f}  [{lo:.3f}, {hi:.3f}]')
            print(f'  median P(dog): {float(np.median(a)):.4f}')
        else:
            print(f'\n(no reserved crops found under {args.acceptance_crops})')
    elif acc_ids:
        print(f'\n(acceptance set has {len(acc_ids)} ids but no crop dir at '
              f'{args.acceptance_crops} -- pass --acceptance-crops)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
