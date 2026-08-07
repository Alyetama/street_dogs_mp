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


def resolve_data(raw, run_dir, root):
    """An absolute dataset path, given what args.yaml recorded.

    A run launched from the dataset's own directory records `dataset.yaml`
    and nothing else, so the string alone does not say where it is. Tried
    against the run's own directory and the training root before giving up --
    the alternative is refusing to score every detector on this box.
    """
    raw = (raw or '').strip()
    if not raw:
        return ''
    if os.path.isabs(raw):
        return raw if os.path.exists(raw) else ''
    for base in (run_dir, os.path.dirname(run_dir),
                 os.path.dirname(os.path.dirname(run_dir)), root):
        cand = os.path.join(base or '', raw)
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return ''


def discover(root):
    """[(key, task, weights, dataset)] for runs that can be re-scored."""
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    import training_tracker
    out = []
    for r in training_tracker.collect(root):
        task = r.get('task')
        if task not in ('classify', 'detect'):
            continue
        w = os.path.join(r['dir'], 'weights', 'best.pt')
        if not os.path.exists(w):
            continue
        ds = resolve_data(r.get('data'), r['dir'], root)
        if not ds:
            continue
        if task == 'classify':
            ok = os.path.isdir(os.path.join(ds, 'val'))
        else:
            # A bare `dataset.yaml` resolves against several bases, and one of
            # them can turn up an unrelated file of that name -- which is
            # worse than finding nothing, because it would score a detector
            # against somebody else's data. The split has to actually be
            # there, with images in it, before the run counts as scorable.
            ok = _val_images(ds) > 0
        if ok:
            out.append((f"{r['project']}/{r['name']}", task, w, ds))
    return out


def _spec(yaml_path):
    """The few flat keys a dataset yaml carries: path, train, val, test.

    Hand-read rather than parsed, because the DASHBOARD asks this question too
    -- it calls _val_images() to decide whether a detector can be scored -- and
    the dashboard's interpreter has no PyYAML. With the import inside a
    try/except returning 0, a missing parser was indistinguishable from an
    empty split: every detect run looked unscorable, forever and silently, in
    the one environment where the answer mattered. training_tracker.read_args
    reads ultralytics' args.yaml the same way and for the same reason.

    Only top-level `key: value` lines are wanted. `names:` is a nested block
    and is deliberately not understood; a line that is not a scalar assignment
    at column zero is skipped.
    """
    out = {}
    try:
        import yaml as _yaml
        with open(yaml_path) as fh:
            got = _yaml.safe_load(fh)
        return got if isinstance(got, dict) else {}
    except ImportError:
        pass
    except (OSError, ValueError):
        return {}
    try:
        with open(yaml_path) as fh:
            for ln in fh:
                if not ln[:1].strip() or ln.lstrip().startswith('#'):
                    continue          # indented (nested) or a comment
                k, sep, v = ln.partition(':')
                if not sep or not k.strip():
                    continue
                v = v.split('#')[0].strip().strip('"').strip("'")
                if v:
                    out[k.strip()] = v
    except OSError:
        return {}
    return out


def _val_images(yaml_path, split='val'):
    """How many images the val split of this dataset yaml actually has."""
    try:
        spec = _spec(yaml_path)
        base = spec.get('path') or os.path.dirname(yaml_path)
        if not os.path.isabs(base):
            base = os.path.join(os.path.dirname(yaml_path), base)
        d = os.path.join(base, spec.get(split) or f'images/{split}')
        return sum(1 for f in os.listdir(d)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    except Exception:
        return 0


def _iou(a, b):
    """Intersection over union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def score_detect(weights, yaml_path, split='val', device='cpu', imgsz=640,
                 conf=0.25, iou_hit=0.5):
    """Boxes a detector invented, and boxes it missed.

    A classifier is wrong about a whole picture, so the picture is the
    evidence. A detector is wrong about a REGION, so the region is: an
    invented box is cropped from where it fired, a missed one from where the
    label says it should have. Matching is greedy by confidence at IoU 0.5,
    the threshold the metrics on this page already use.
    """
    from ultralytics import YOLO
    import yaml as _yaml
    with open(yaml_path) as fh:
        spec = _yaml.safe_load(fh) or {}
    base = spec.get('path') or os.path.dirname(yaml_path)
    if not os.path.isabs(base):
        base = os.path.join(os.path.dirname(yaml_path), base)
    img_dir = os.path.join(base, spec.get(split) or f'images/{split}')
    if not os.path.isdir(img_dir):
        raise SystemExit(f'no {split} images at {img_dir}')
    # YOLO's own convention: labels sit beside images with the directory
    # renamed, so a dataset laid out any other way is not one this can read
    lab_dir = img_dir.replace(os.sep + 'images' + os.sep,
                              os.sep + 'labels' + os.sep)
    names = spec.get('names') or {}
    classes = [names[i] for i in sorted(names)] if isinstance(names, dict) \
        else list(names)

    from PIL import Image
    model = YOLO(weights)
    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    wrong, n_gt, n_hit = [], 0, 0
    for f in files:
        ipath = os.path.join(img_dir, f)
        with Image.open(ipath) as im:
            W, H = im.size
        gt = []
        lp = os.path.join(lab_dir, os.path.splitext(f)[0] + '.txt')
        try:
            for ln in open(lp):
                parts = ln.split()
                if len(parts) < 5:
                    continue
                c = int(float(parts[0]))
                cx, cy, bw, bh = (float(x) for x in parts[1:5])
                gt.append((c, [(cx - bw / 2) * W, (cy - bh / 2) * H,
                               (cx + bw / 2) * W, (cy + bh / 2) * H]))
        except OSError:
            pass
        n_gt += len(gt)
        res = model.predict(ipath, imgsz=imgsz, device=device, conf=conf,
                            verbose=False)[0]
        preds = []
        for b in res.boxes:
            preds.append((int(b.cls.item()), float(b.conf.item()),
                          [float(v) for v in b.xyxy[0].tolist()]))
        preds.sort(key=lambda x: -x[1])
        used = set()
        for pc, pconf, pbox in preds:
            best, at = 0.0, -1
            for gi, (gc, gbox) in enumerate(gt):
                if gi in used or gc != pc:
                    continue
                v = _iou(pbox, gbox)
                if v > best:
                    best, at = v, gi
            if best >= iou_hit and at >= 0:
                used.add(at)
                n_hit += 1
            else:
                wrong.append({'file': os.path.relpath(ipath, base),
                              'kind': 'invented', 'p': round(pconf, 4),
                              'cls': classes[pc] if pc < len(classes) else str(pc),
                              'box': [round(v, 1) for v in pbox]})
        for gi, (gc, gbox) in enumerate(gt):
            if gi not in used:
                wrong.append({'file': os.path.relpath(ipath, base),
                              'kind': 'missed', 'p': None,
                              'cls': classes[gc] if gc < len(classes) else str(gc),
                              'box': [round(v, 1) for v in gbox]})
    # invented boxes first and by confidence, misses after: a confident
    # invention is the most informative thing here, and a miss has no
    # confidence to sort by
    wrong.sort(key=lambda x: (x['kind'] != 'invented', -(x['p'] or 0)))
    return classes, base, n_gt, n_hit, wrong


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


def run_one(key, task, weights, dataset, split, device, imgsz):
    if task == 'detect':
        classes, base, n_gt, n_hit, wrong = score_detect(
            weights, dataset, split, device, imgsz)
        doc = {'run': key, 'task': 'detect', 'dataset': base,
               'yaml': dataset, 'split': split, 'classes': classes,
               'n': n_gt, 'hit': n_hit, 'wrong': len(wrong),
               # recall, not accuracy: a detector has no denominator that
               # counts the things it correctly did not fire on
               'recall': round(n_hit / n_gt, 4) if n_gt else None,
               'device': device, 'imgsz': imgsz,
               'truncated': len(wrong) > KEEP, 'items': wrong[:KEEP]}
    else:
        classes, n, wrong = score(weights, dataset, split, device, imgsz)
        doc = {'run': key, 'task': 'classify', 'dataset': dataset,
               'split': split, 'classes': classes, 'n': n,
               'wrong': len(wrong),
               'accuracy': round((n - len(wrong)) / n, 4) if n else None,
               'device': device, 'imgsz': imgsz,
               'truncated': len(wrong) > KEEP, 'items': wrong[:KEEP]}
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = out_path(key) + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, out_path(key))
    if doc['task'] == 'detect':
        print(f"  {key}: {doc['wrong']} wrong "
              f"({sum(1 for i in doc['items'] if i['kind'] == 'invented')} "
              f"invented, {sum(1 for i in doc['items'] if i['kind'] == 'missed')}"
              f" missed of {doc['n']} labelled) -> "
              f"{os.path.relpath(out_path(key), REPO)}")
    else:
        n = doc['n']
        print(f"  {key}: {doc['wrong']} wrong of {n} "
              f"({(n - doc['wrong']) / n:.1%} correct) -> "
              f"{os.path.relpath(out_path(key), REPO)}")
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
            print(f'{a.run}: not a run with weights and a dataset still on '
                  f'disk', file=sys.stderr)
            return 1
    elif not a.all:
        print(f'{len(found)} run(s) can be scored:')
        for key, task, _, ds in found:
            done = 'scored' if os.path.exists(out_path(key)) else '-'
            print(f'  {key:30s} {task:9s} {os.path.basename(ds):16s} {done}')
        print('\npass --run <project/name> or --all')
        return 0

    for key, task, w, ds in found:
        if os.path.exists(out_path(key)) and not a.force:
            print(f'  {key}: already scored (--force to redo)')
            continue
        try:
            run_one(key, task, w, ds, a.split, a.device, a.imgsz)
        except Exception as e:
            print(f'  {key}: {type(e).__name__}: {e}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
