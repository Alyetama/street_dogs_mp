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
import math
import os
import sys


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
    ap.add_argument('--data', default='<home>/dogs_detection/leash_binary_v1')
    ap.add_argument('--split', default='val')
    ap.add_argument('--hard-negatives',
                    default='<mounts>/crucial/street_dogs_mp_crucial/'
                            'data/hard_negatives/crops',
                    help='dashboard-flagged false positives: negatives from '
                         'the REAL sweep distribution')
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
    return 0


if __name__ == '__main__':
    sys.exit(main())
