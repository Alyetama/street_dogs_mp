#!/usr/bin/env python3
"""Evaluate the leash classifier -- leashed vs unleashed -- on reserved crops.

NOT a copy of eval_dogbin.py with the words swapped. That tool is shaped by
the gate's cost asymmetry: a missed dog is unrecoverable, a surviving false
positive is one click, so every number there is anchored on dog recall and the
question is "how much can I throw away before I lose a dog?".

Leashed vs unleashed has no such asymmetry. Both errors are the same kind of
mistake -- a mislabelled animal in the final dataset -- so the headline is
BALANCED ACCURACY (the mean of the two class recalls), not top-1. Top-1 on a
1.3:1 split rewards a model that leans toward the majority class; balanced
accuracy does not.

The numbers that decide anything come from the reserved acceptance set:
image_ids withheld by reserve_acceptance_set.py BEFORE leash_v2 was split, so
they are absent from train AND val. The val split still drives early stopping,
which makes it a tuning set, not an independent one.

    python tools/detect/eval_leash.py --weights <run>/weights/best.pt

Prints, for the val split and again for the reserved set: balanced accuracy,
per-class recall with Wilson intervals, the confusion matrix, ROC AUC, and the
threshold that maximises balanced accuracy. Exits 1 if no reserved crops could
be scored -- a run without that table cannot support a promotion.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_dogbin import (  # noqa: E402  -- one implementation of each guard
    REPO, drop_trained_on, id_of_name, load_dir, readable, wilson,
)

POS, NEG = 'leashed', 'unleashed'


def class_prob(model, paths, pos_name, batch, imgsz, device):
    """P(pos_name) per path, in order."""
    idx = [i for i, n in model.names.items()
           if str(n).lower() == pos_name.lower()]
    if not idx:
        raise SystemExit(f'no "{pos_name}" class in model.names={model.names}')
    i0, out = idx[0], []
    for i in range(0, len(paths), batch):
        for r in model.predict(paths[i:i + batch], imgsz=imgsz,
                               verbose=False, device=device):
            out.append(float(r.probs.data[i0]))
    return out


def report(tag, p_pos, p_neg, np):
    """p_pos/p_neg: P(leashed) for the leashed and unleashed crops."""
    from sklearn.metrics import roc_auc_score
    n_pos, n_neg = len(p_pos), len(p_neg)
    if not n_pos or not n_neg:
        print(f'\n{tag}: need both classes (have {n_pos} {POS}, '
              f'{n_neg} {NEG}) -- skipped')
        return None
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
    s = np.r_[p_pos, p_neg]
    base = max(n_pos, n_neg) / (n_pos + n_neg)
    print(f'\n{tag}: {n_pos} {POS} / {n_neg} {NEG}   '
          f'majority baseline {base:.4f}')

    def at(t):
        tp = int((p_pos >= t).sum())
        tn = int((p_neg < t).sum())
        return tp, n_pos - tp, tn, n_neg - tn

    tp, fn, tn, fp = at(0.5)
    rec_p, rec_n = tp / n_pos, tn / n_neg
    lo_p, hi_p = wilson(tp, n_pos)
    lo_n, hi_n = wilson(tn, n_neg)
    bal = (rec_p + rec_n) / 2
    top1 = (tp + tn) / (n_pos + n_neg)
    print(f'  BALANCED ACCURACY @0.5 : {bal:.4f}   <- the headline')
    print(f'  top-1 @0.5             : {top1:.4f}   '
          f'(vs {base:.4f} baseline)')
    print(f'  recall {POS:<12}    : {rec_p:.4f}  [{lo_p:.3f}, {hi_p:.3f}]  '
          f'({tp}/{n_pos})')
    print(f'  recall {NEG:<12}    : {rec_n:.4f}  [{lo_n:.3f}, {hi_n:.3f}]  '
          f'({tn}/{n_neg})')
    print(f'  ROC AUC                : {roc_auc_score(y, s):.4f}')
    print(f'  confusion @0.5         : '
          f'{POS}->{POS} {tp}, {POS}->{NEG} {fn}, '
          f'{NEG}->{NEG} {tn}, {NEG}->{POS} {fp}')

    # Where the model is actually best, rather than at whatever 0.5 happens to
    # give. With no cost asymmetry the operating point is a free choice, so the
    # honest thing is to report the best available and the cost of moving.
    ts = sorted(set(np.round(np.r_[p_pos, p_neg], 4).tolist()))
    best_t, best_b = 0.5, bal
    for t in ts:
        a, _, b, _ = at(t)
        v = (a / n_pos + b / n_neg) / 2
        if v > best_b:
            best_b, best_t = v, t
    if best_t != 0.5:
        tp2, _, tn2, _ = at(best_t)
        print(f'  best balanced accuracy : {best_b:.4f} at t={best_t:.4f}  '
              f'(+{best_b - bal:.4f} over 0.5; recall {tp2/n_pos:.4f}/'
              f'{tn2/n_neg:.4f})')
    return {'balanced_accuracy': round(bal, 4), 'top1': round(top1, 4),
            'roc_auc': round(float(roc_auc_score(y, s)), 4),
            f'recall_{POS}': round(rec_p, 4), f'recall_{NEG}': round(rec_n, 4),
            'n': n_pos + n_neg}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data', default=os.environ.get('LEASH_DATASET')
                    or os.path.join(REPO, '..', 'leash_v2'),
                    help='dataset root with <split>/leashed and '
                         '<split>/unleashed (or set $LEASH_DATASET)')
    ap.add_argument('--split', default='val')
    ap.add_argument('--acceptance-set',
                    default=os.path.join(REPO, 'data',
                                         'leash_acceptance_set.json'))
    ap.add_argument('--acceptance-crops',
                    default=os.path.join(REPO, 'data', 'harvest',
                                         'leash_src', 'train'),
                    help='<dir>/leashed and <dir>/unleashed holding the '
                         'reserved crops at full resolution')
    ap.add_argument('--trained-on', nargs='*', default=(),
                    help='EXTRA dataset roots this model saw, beyond --data. '
                         'The reservation protects models trained on --data; a '
                         'model trained on something ELSE may have seen every '
                         'reserved id, and scoring it here would report a '
                         'train-set number with nothing to flag it. Measured: '
                         'leash_03_jun_26-3 scores 0.9334 balanced accuracy on '
                         'a set whose 312 ids are all in its own training data.')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='0')
    ap.add_argument('--json', help='also write the metrics here')
    args = ap.parse_args()

    from ultralytics import YOLO
    import numpy as np

    model = YOLO(args.weights)
    print(f'model   : {args.weights}')
    print(f'classes : {model.names}')

    def score(d):
        ps = load_dir(d)
        ps, _ = readable(ps)
        return np.array(class_prob(model, ps, POS, args.batch, args.imgsz,
                                   args.device)) if ps else np.array([])

    out = {}
    root = os.path.join(args.data, args.split)
    v = report(f'{args.split} split (drives early stopping -- a tuning set, '
               f'not an independent one)',
               score(os.path.join(root, POS)),
               score(os.path.join(root, NEG)), np)
    if v:
        out['val'] = v

    # ---- the reserved set: never trained on, never early-stopped on --------
    try:
        with open(args.acceptance_set) as fh:
            held = set(json.load(fh).get('image_ids') or [])
    except (OSError, ValueError) as e:
        print(f'\nNO ACCEPTANCE NUMBER PRODUCED: cannot read '
              f'{args.acceptance_set} ({e}). Nothing here can support a '
              f'promotion.')
        return 1
    if not held:
        print(f'\nNO ACCEPTANCE NUMBER PRODUCED: {args.acceptance_set} '
              f'reserves no image_ids.')
        return 1

    got = {}
    for cls in (POS, NEG):
        ps = [f for f in load_dir(os.path.join(args.acceptance_crops, cls))
              if id_of_name(f) in held]
        ps, contaminated = drop_trained_on(ps, args.data)
        for extra in args.trained_on:
            ps, more = drop_trained_on(ps, extra)
            contaminated += more
        if contaminated:
            print(f'\nWARNING: {len(contaminated)} reserved {cls} crop(s) are '
                  f'in the dataset -- the reservation is not protecting this '
                  f'model. Excluded from the scores below; investigate.')
        ps, bad = readable(ps)
        if bad:
            print(f'\nSKIPPED {len(bad)} unreadable reserved {cls} crop(s):')
            for b in bad[:5]:
                print(f'    {os.path.basename(b)}  ({os.path.getsize(b)} bytes)')
        got[cls] = np.array(class_prob(model, ps, POS, args.batch, args.imgsz,
                                       args.device)) if ps else np.array([])

    a = report('ACCEPTANCE SET (reserved before the split; never trained on, '
               'never early-stopped on)', got[POS], got[NEG], np)
    if not a:
        print('\nNO ACCEPTANCE NUMBER PRODUCED: no reserved crops scored.')
        return 1
    out['acceptance'] = a
    ids = len({id_of_name(f) for cls in (POS, NEG)
               for f in load_dir(os.path.join(args.acceptance_crops, cls))
               if id_of_name(f) in held})
    print(f'  reserved ids: {ids} of {len(held)} had a crop here')

    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(out, fh, indent=1)
        print(f'\nmetrics -> {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
