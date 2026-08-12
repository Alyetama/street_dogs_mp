#!/usr/bin/env python3
"""Which boxes one detector finds and another misses, and whether NMS ate them.

    python tools/detect/miss_diff.py --data <dataset> --a <weights> --b <weights>

WHY. A recall difference between two detectors is usually reported as two
numbers, and two numbers cannot tell you whether the weaker model never
proposed the box or proposed it and then threw it away. Those have different
fixes -- one is training, the other is a threshold -- and the second is free.

So every ground-truth box is matched against both models on the SAME frames,
the disagreement is counted pairwise (a box both models miss says nothing about
which is better), and then the losing model is re-run with NMS loosened and the
detection cap raised. A miss that comes back under looser NMS was a suppression,
not a failure to see.

Pairwise matters. Comparing 0.857 against 0.922 over 154 boxes hides that the
two models agree on 136 of them; the question is only ever about the ones where
they differ, and there an exact binomial on the discordant pairs is the whole
test.
"""

import argparse
import os
import sys

IOU = 0.5


def gt_boxes(lbl, w, h):
    out = []
    try:
        txt = open(lbl).read().strip()
    except OSError:
        return out
    for ln in txt.splitlines():
        p = ln.split()
        if len(p) != 5:
            continue
        _, cx, cy, bw, bh = (float(x) for x in p)
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    u = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)
    return i / u if u > 0 else 0.0


def hits(model, ip, gt, conf, nms_iou, max_det):
    """Which ground-truth boxes this model finds, greedily, one prediction each."""
    r = model.predict(ip, conf=conf, iou=nms_iou, max_det=max_det,
                      verbose=False, device=0)[0]
    preds = sorted(zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist()),
                   key=lambda x: -x[0])
    found, used = set(), set()
    for _, box in preds:
        best, bi = 0.0, -1
        for i, g in enumerate(gt):
            if i in used:
                continue
            v = iou(box, g)
            if v > best:
                best, bi = v, i
        if bi >= 0 and best >= IOU:
            used.add(bi)
            found.add(bi)
    return found, r.orig_shape


def frames(root):
    idir = os.path.join(root, 'images', 'val')
    for name in sorted(os.listdir(idir)):
        yield (name, os.path.join(idir, name),
               os.path.join(root, 'labels', 'val',
                            name.rsplit('.', 1)[0] + '.txt'))


def exact_p(k, n):
    """Two-sided exact binomial on the discordant pairs, p=0.5 under the null."""
    import math
    if not n:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', required=True,
                    help='dataset root holding images/val and labels/val')
    ap.add_argument('--a', required=True, help='weights under test')
    ap.add_argument('--b', required=True, help='weights to compare against')
    ap.add_argument('--conf', type=float, default=0.05,
                    help='the threshold the sweep actually deploys')
    ap.add_argument('--nms-iou', type=float, default=0.7,
                    help="ultralytics' default")
    ap.add_argument('--max-det', type=int, default=300)
    ap.add_argument('--recover', default='0.9,0.95',
                    help='NMS IoU values to retry A misses at; higher keeps '
                         'more overlapping boxes')
    a = ap.parse_args()

    from ultralytics import YOLO
    A, B = YOLO(a.a), YOLO(a.b)
    rows = []
    for name, ip, lp in frames(a.data):
        _, shape = hits(A, ip, [], a.conf, a.nms_iou, a.max_det)
        h, w = shape
        gt = gt_boxes(lp, w, h)
        if not gt:
            continue
        fa, _ = hits(A, ip, gt, a.conf, a.nms_iou, a.max_det)
        fb, _ = hits(B, ip, gt, a.conf, a.nms_iou, a.max_det)
        for i in range(len(gt)):
            rows.append({'name': name, 'i': i, 'n': len(gt), 'ip': ip,
                         'gt': gt[i], 'a': i in fa, 'b': i in fb})

    print(f'{len(rows)} ground-truth boxes over '
          f'{len({r["name"] for r in rows})} frames, conf {a.conf}\n')
    for label, sub in (('all boxes', rows),
                       ('multi-object frames', [r for r in rows if r['n'] > 1])):
        ao = [r for r in sub if r['a'] and not r['b']]
        bo = [r for r in sub if r['b'] and not r['a']]
        d = len(ao) + len(bo)
        print(f'  {label:<22} n={len(sub):>4}  A-only misses {len(bo):>3}  '
              f'B-only misses {len(ao):>3}  p={exact_p(len(ao), d):.3f}')

    lost = [r for r in rows if r['b'] and not r['a']]
    if not lost:
        print('\nA misses nothing B finds; nothing to recover.')
        return 0
    print(f'\nCan looser NMS recover the {len(lost)} boxes A loses?')
    print(f'  {"nms_iou":>8} {"max_det":>8}   recovered')
    best = 0
    for v in [float(x) for x in a.recover.split(',')] + [a.nms_iou]:
        for md in (a.max_det, max(a.max_det, 1000)):
            got = 0
            for r in lost:
                f, _ = hits(A, r['ip'], [r['gt']], a.conf, v, md)
                got += (0 in f)
            best = max(best, got)
            tag = '  (baseline)' if v == a.nms_iou and md == a.max_det else ''
            print(f'  {v:>8.2f} {md:>8}   {got:>3}/{len(lost)}{tag}')
    # The verdict has to follow the table. Printing the suppression sentence
    # unconditionally made a run that recovered NOTHING read as though it had
    # found the cause, which is the opposite of what the numbers said.
    if best:
        print(f'\n{best} of {len(lost)} come back under looser NMS: those were '
              f'suppressed rather than unseen, which is a threshold to tune '
              f'and not a retrain.')
    else:
        print(f'\nNone of the {len(lost)} come back at any setting tried, so A '
              f'never proposed them. Nothing about NMS or the detection cap '
              f'will recover these; the difference is in the model.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
