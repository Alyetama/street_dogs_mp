#!/usr/bin/env python3
"""
Score a finished classify run against its own val split and keep the mistakes.

    python tools/detect/run_mistakes.py --run dog-bin/dogbin_008
    python tools/detect/run_mistakes.py --all           # every scorable run
    python tools/detect/run_mistakes.py --run ... --device cuda

The confusion matrix on the dashboard says how many crops went each way. It
cannot say WHICH, and which is the only form the answer is useful in: "31 real
not-dogs were called dog" is a number to worry about, and the thirty-one
pictures are a thing to act on -- they are usually a kind, and the kind is
what tells you what the next dataset needs.

So this writes, per run, every crop the model got wrong, with what it said and
how sure it was. Sorted by confidence, because a confident mistake is worth
more than a hesitant one: the model is not undecided about those, it is wrong
about them, and whatever it has learnt to see there is learnt firmly.

Read-only on the datasets and the run directories. Writes one JSON per run
under data/mistakes/, which the dashboard reads -- never the model, so no
render ever waits for inference.
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, 'data', 'mistakes')
# how many to keep per run; the tail of a long list is never looked at, and
# the file is read on every render of the run's panel
KEEP = 240


def out_path(run_key):
    """data/mistakes/<project>__<name>.json for a project/name key."""
    return os.path.join(OUT_DIR, run_key.replace('/', '__') + '.json')


def discover(root):
    """[(key, weights, dataset)] for classify runs that can be re-scored."""
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    import training_tracker
    out = []
    for r in training_tracker.collect(root):
        if r.get('task') != 'classify':
            continue
        w = os.path.join(r['dir'], 'weights', 'best.pt')
        ds = r.get('data') or ''
        if os.path.exists(w) and os.path.isdir(os.path.join(ds, 'val')):
            out.append((f"{r['project']}/{r['name']}", w, ds))
    return out


def score(weights, dataset, split='val', device='cpu', imgsz=640, batch=16):
    """[(rel path, true, pred, p_pred, p_true)] for every crop it got wrong."""
    from ultralytics import YOLO
    from PIL import Image
    root = os.path.join(dataset, split)
    # ultralytics assigns class indices by SORTED directory name, so the truth
    # has to be read the same way or every label is quietly off by one
    classes = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    files = []
    for ci, cname in enumerate(classes):
        d = os.path.join(root, cname)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                files.append((os.path.join(d, f), ci))
    model = YOLO(weights)
    names = model.names or {}
    # the model's own class order wins if it has one; a dataset rebuilt with a
    # class renamed would otherwise be scored against the wrong names
    if names:
        classes = [names[i] for i in sorted(names)]
    wrong, n = [], 0
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        ims = [Image.open(p).convert('RGB') for p, _ in chunk]
        res = model.predict(ims, imgsz=imgsz, device=device, verbose=False)
        for (path, truth), r in zip(chunk, res):
            n += 1
            probs = r.probs
            pred = int(probs.top1)
            if pred == truth:
                continue
            data = probs.data.tolist()
            wrong.append({
                'file': os.path.relpath(path, dataset),
                'true': classes[truth] if truth < len(classes) else str(truth),
                'pred': classes[pred] if pred < len(classes) else str(pred),
                'p': round(float(data[pred]), 4),
                'p_true': round(float(data[truth]), 4)})
        for im in ims:
            im.close()
    # most confident first: those are the ones the model is sure about and
    # wrong about, which is where a dataset gap shows itself
    wrong.sort(key=lambda x: -x['p'])
    return classes, n, wrong


def run_one(key, weights, dataset, split, device, imgsz):
    classes, n, wrong = score(weights, dataset, split, device, imgsz)
    doc = {'run': key, 'dataset': dataset, 'split': split,
           'classes': classes, 'n': n, 'wrong': len(wrong),
           'accuracy': round((n - len(wrong)) / n, 4) if n else None,
           'device': device, 'imgsz': imgsz,
           'truncated': len(wrong) > KEEP,
           'items': wrong[:KEEP]}
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = out_path(key) + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, out_path(key))
    print(f'  {key}: {len(wrong)} wrong of {n} '
          f'({(n - len(wrong)) / n:.1%} correct) -> '
          f'{os.path.relpath(out_path(key), REPO)}')
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', help='project/name, as the dashboard shows it')
    ap.add_argument('--all', action='store_true',
                    help='every classify run whose weights and dataset are '
                         'both still on disk')
    ap.add_argument('--root', default=os.environ.get('TRAINING_ROOT', ''),
                    help='where the runs live (default $TRAINING_ROOT)')
    ap.add_argument('--split', default='val')
    ap.add_argument('--device', default='cpu',
                    help="cpu or cuda. cpu by default: this box trains on the "
                         "GPU, and a debugging aid should never be the reason "
                         "a training run runs out of memory")
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--force', action='store_true',
                    help='rescore runs that already have a file')
    a = ap.parse_args(argv)

    root = a.root
    if not root:
        try:
            sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
            import dashboard
            root = dashboard.training_root()
        except Exception:
            root = ''
    if not root:
        print('no training root: pass --root or set $TRAINING_ROOT',
              file=sys.stderr)
        return 1

    found = discover(root)
    if a.run:
        found = [f for f in found if f[0] == a.run]
        if not found:
            print(f'{a.run}: not a classify run with weights and a dataset '
                  f'still on disk', file=sys.stderr)
            return 1
    elif not a.all:
        print(f'{len(found)} run(s) can be scored:')
        for key, _, ds in found:
            done = 'scored' if os.path.exists(out_path(key)) else '-'
            print(f'  {key:30s} {os.path.basename(ds):14s} {done}')
        print('\npass --run <project/name> or --all')
        return 0

    for key, w, ds in found:
        if os.path.exists(out_path(key)) and not a.force:
            print(f'  {key}: already scored (--force to redo)')
            continue
        try:
            run_one(key, w, ds, a.split, a.device, a.imgsz)
        except Exception as e:
            print(f'  {key}: {type(e).__name__}: {e}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
