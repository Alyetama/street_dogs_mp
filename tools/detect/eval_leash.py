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

THE FROZEN TEST SET (--test-set data/leash_test_v1.json)

    python tools/detect/eval_leash.py --test-set data/leash_test_v1.json \\
        --weights <new>/weights/best.pt --compare <old>/weights/best.pt

leash_test_v1 is 628 leash-audit boxes withheld by key, image_id AND sequence
before any current split was cut, and its crops are the audit's full/ files:
the detector box cut with gate_store.cut_crops, which is what production
judges. Until this mode nothing could score a model on it -- the acceptance
path above reads old-framing harvest crops -- so every comparison needed an
ad-hoc script, and an ad-hoc script is how a model ends up scored on edited/
instead of full/, on a re-cut file, or at a stale imgsz. Here:

  * each item is found by its KEY (<image_id>#<det_idx> -> <id>_<det>.jpg)
    under REPO/data/leash_audit/full, the stored absolute crops_from only as
    a fallback, and only that key is scored -- never every crop sharing the
    image_id. A missing file or one whose sha256 is not the frozen value
    stops the run: a test set that silently shrinks or changes is not the
    test set the last number was measured on.
  * the label is items[].label, the human verdict -- not a class folder.
  * the crop file IS the production crop, so it is predicted as it is, at the
    checkpoint's own train imgsz (gate_store.model_imgsz), with no re-pad.
  * items[].p_model and band_populations are NOT used. p_model is the legacy
    224-px leash_v2_001 audit score that defined the sampling bands; against
    the human labels it has AUC 0.489 on these 628, so reading it as "what
    the old model says" would put leash_v2_001 at chance. The headline is
    the unweighted balanced accuracy of a fresh prediction.

Every metric is also reported on slices computed at eval time from repo data
-- nothing is written into the frozen file:

  redraw   71 of the 628 (68 leashed) carry an 'audit:leash' redraw in
           data/box_corrections/boxes.jsonl; for 32 of them the redraw has
           IoU < 0.5 with the pool's detector box, and the full/ crop shows a
           lone dog while only the redraw takes in the leash or the handler.
           Those labels are likely unsupported by the pixels: a known
           ceiling on leashed recall, reported apart, not excluded.
  size     crop short side <=64 / 64-96 / 96-128 / 128-192 / >192 px. The
           training set's leashed rate runs 0.17 at <=64 px to ~0.60 above
           192 px while the test set's is 0.50-0.61 in every band, so a model
           can learn "small means unleashed" and val will not show it.
  region   from the grid cell of the box in data/leash_audit/pool.parquet. A
           lookup that gives each item its region's training leashed rate
           reaches AUC 0.797 on this test set, so overall accuracy rewards a
           model that leans on scene and geography. The AGAINST-PRIOR slice
           (leashed in South/Southeast Asia, South America, Africa, Central
           America; unleashed in Europe, North America, Australia) is where
           that shortcut fails, and the only place it shows.

With --compare the checkpoints are scored on one common item set, printed
side by side, and every pair gets an exact McNemar test on per-item
correctness at 0.5 and a paired bootstrap 95% interval on the difference in
balanced accuracy, so "X beats Y" is a test result, not two numbers that
happen to differ. A JSON report (with every item's prediction) goes to
--out, by default into the first checkpoint's run directory
(<run>/<test set>_eval.json, i.e. under the training root -- pass --out to
keep it elsewhere).
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
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


# ── the frozen test set ────────────────────────────────────────────────────
# Repo-relative, never the absolute path the doc was written with: the repo
# is moved between drives, and the stored crops_from is only a fallback.

AUDIT_DIR = os.path.join(REPO, 'data', 'leash_audit')
AUDIT_FULL = os.path.join(AUDIT_DIR, 'full')
AUDIT_POOL = os.path.join(AUDIT_DIR, 'pool.parquet')
CORRECTIONS = os.path.join(REPO, 'data', 'box_corrections', 'boxes.jsonl')
THRESHOLD = 0.5          # gate_store labels p >= 0.5 as the positive class
REDRAW_LOW_IOU = 0.5     # below this the redraw is a different crop
# (upper bound inclusive, name): the bands the leash audit measured the
# training set's size skew in -- rebuild_crop_dataset.SIZE_BANDS, the one
# definition the build's size diagnostic and --balance-size-bands use too
from rebuild_crop_dataset import SIZE_BANDS, size_band  # noqa: E402
SHORT_BANDS = tuple((float('inf') if hi is None else hi, name)
                    for _lo, hi, name in SIZE_BANDS)
# Regions whose training leashed rate is high / low. An item AGAINST its
# region's prior is the case a scene-and-geography shortcut gets wrong:
# 122 of the 628 test items, 534 of 2,994 train rows.
PRIOR_LEASHED = frozenset({'Europe', 'North_America', 'Australia'})
PRIOR_UNLEASHED = frozenset({'South_Asia', 'Southeast_Asia', 'South_America',
                             'Africa', 'Central_America_and_Caribbean'})
# A grid cell is <region>_<lon0>_<lat0>_<lon1>_<lat1>, e.g.
# 'Africa_-5_15_0_20' or 'Central_America_and_Caribbean_-80_15_-75_20'.
_CELL_RE = re.compile(r'^(.+?)(?:_-?\d+(?:\.\d+)?){4}$')


def region_of_cell(cell):
    m = _CELL_RE.match(cell) if isinstance(cell, str) else None
    return m.group(1) if m else None


def key_file(key):
    """'1002054879356240#0' -> '1002054879356240_0.jpg', the audit's name."""
    return key.replace('#', '_') + '.jpg'


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_test_set(path, full_dir=AUDIT_FULL):
    """(doc, items): the frozen set, every item resolved and verified.

    items are dicts {key, image_id, det_idx, label, path, fallback}. Refuses
    -- SystemExit listing what is wrong -- on a doc for another family, a
    duplicate key, a label that is neither class, a crop found nowhere, or a
    crop whose sha256 is not the frozen one. All problems are collected first
    so one run shows them all. When the doc carries keys[], items[] must
    hold exactly those keys.
    """
    with open(path) as fh:
        doc = json.load(fh)
    fam = doc.get('family') or doc.get('stage')
    if fam != 'leash':
        raise SystemExit(f'{path}: family {fam!r}, not a leash test set')
    raw = doc.get('items') or []
    if not raw:
        raise SystemExit(f'{path}: no items[]')
    alt = doc.get('crops_from')
    items, problems, seen = [], [], set()
    for it in raw:
        key, label, want = it.get('key'), it.get('label'), it.get('sha256')
        if not isinstance(key, str) or '#' not in key:
            problems.append(f'item without a <image_id>#<det_idx> key: {it!r}')
            continue
        if key in seen:
            problems.append(f'{key}: listed twice')
            continue
        seen.add(key)
        if label not in (POS, NEG):
            problems.append(f'{key}: label {label!r}')
            continue
        if not want:
            problems.append(f'{key}: no frozen sha256')
            continue
        name = key_file(key)
        p, fallback = os.path.join(full_dir, name), False
        if not os.path.isfile(p) and alt:
            q = os.path.join(alt, name)
            if os.path.isfile(q):
                p, fallback = q, True
        if not os.path.isfile(p):
            problems.append(f'{key}: missing ({name} in neither {full_dir} '
                            f'nor crops_from {alt})')
            continue
        got = _sha256(p)
        if got != want:
            problems.append(f'{key}: sha256 {got[:12]} at {p}, frozen '
                            f'{want[:12]} -- the crop changed since the set '
                            f'was frozen')
            continue
        iid, det = key.split('#', 1)
        try:
            det = int(det)
        except ValueError:
            problems.append(f'{key}: det_idx is not an integer')
            continue
        items.append({'key': key, 'image_id': iid, 'det_idx': det,
                      'label': label, 'path': p, 'fallback': fallback})
    # keys[] is the membership the set was frozen with (628 = items[] in
    # leash_test_v1). Every remaining sha still matches when an entry is cut
    # from items[], so only this comparison catches a doc that shrank.
    frozen = doc.get('keys')
    if frozen is not None:
        listed = {it.get('key') for it in raw}
        lost = sorted(set(frozen) - listed)
        extra = sorted(k for k in listed - set(frozen) if isinstance(k, str))
        if lost:
            problems.append(f'{len(lost)} key(s) in keys[] have no items[] '
                            f'entry: {", ".join(lost[:5])}')
        if extra:
            problems.append(f'{len(extra)} items[] key(s) not in keys[]: '
                            f'{", ".join(extra[:5])}')
    if problems:
        head = '\n  '.join(problems[:20])
        more = (f'\n  ... and {len(problems) - 20} more'
                if len(problems) > 20 else '')
        raise SystemExit(f'REFUSED: {len(problems)} of {len(raw)} test items '
                         f'cannot be scored as frozen:\n  {head}{more}')
    return doc, items


def _pool_boxes(pool_path, items):
    """{(image_id, det_idx): (x1, y1, x2, y2, cell)} from the audit pool.

    pyarrow, not duckdb: this runs in the ultralytics env, which has no duckdb.
    """
    import pyarrow.parquet as pq
    ids = sorted({it['image_id'] for it in items})
    t = pq.read_table(pool_path,
                      columns=['image_id', 'det_idx', 'x1', 'y1', 'x2', 'y2',
                               'cell'],
                      filters=[('image_id', 'in', ids)])
    out = {}
    for r in t.to_pylist():
        out.setdefault((str(r['image_id']), int(r['det_idx'])),
                       (r['x1'], r['y1'], r['x2'], r['y2'], r['cell']))
    return out


def annotate_items(items, pool_path=AUDIT_POOL, corrections_path=CORRECTIONS):
    """Add short_side, region, redrawn and redraw_iou to each item, in place.

    Computed here from the crop file, the audit pool and the corrections
    ledger -- never written into the frozen doc. A value that cannot be
    derived is None and the item lands in no slice of that kind; the count
    of those is returned so the report can say so.
    """
    from PIL import Image
    from crop_sources import iou, read_audit_corrections
    for it in items:
        with Image.open(it['path']) as im:
            it['short_side'] = min(im.size)
    try:
        pool = _pool_boxes(pool_path, items)
    except Exception as e:                  # noqa: BLE001 - say, then go on
        print(f'WARNING: cannot read the audit pool {pool_path} ({e}) -- no '
              f'region slices, no redraw IoU')
        pool = {}
    # read_audit_corrections gives {} for a missing ledger without a word,
    # and boxes.jsonl is gitignored: on a fresh clone every item would read
    # as 'no audit redraw' and the 71 redraws would vanish from the report
    if os.path.isfile(corrections_path):
        redraws = read_audit_corrections('leash', corrections_path)
    else:
        print(f'WARNING: no corrections ledger at {corrections_path} -- '
              f'redraws unknown, no redraw slices')
        redraws = None
    unknown = {'region': 0, 'redrawn': 0, 'redraw_iou': 0}
    for it in items:
        k = (it['image_id'], it['det_idx'])
        row = pool.get(k)
        it['region'] = region_of_cell(row[4]) if row else None
        it['redrawn'] = None if redraws is None else k in redraws
        it['redraw_iou'] = None
        if it['redrawn'] and row:
            it['redraw_iou'] = round(iou(redraws[k], row[:4]), 4)
        unknown['region'] += it['region'] is None
        unknown['redrawn'] += it['redrawn'] is None
        unknown['redraw_iou'] += (bool(it['redrawn'])
                                  and it['redraw_iou'] is None)
    return unknown


def short_band(s):
    return size_band(s)


def against_prior(it):
    """True / False against / with the regional prior; None outside it."""
    r = it.get('region')
    if r in PRIOR_LEASHED:
        return it['label'] == NEG
    if r in PRIOR_UNLEASHED:
        return it['label'] == POS
    return None


def slice_index(items):
    """[(group, slice, [item index])] -- every slice the report prints."""
    out = [('all', 'all', list(range(len(items))))]
    # 'is False', not 'not': redrawn None (no ledger) is in neither slice
    out.append(('redraw', 'no audit redraw',
                [i for i, it in enumerate(items) if it['redrawn'] is False]))
    out.append(('redraw', 'audit redraw',
                [i for i, it in enumerate(items) if it['redrawn']]))
    out.append(('redraw', f'redraw IoU < {REDRAW_LOW_IOU} (leash likely '
                f'outside crop)',
                [i for i, it in enumerate(items)
                 if it['redraw_iou'] is not None
                 and it['redraw_iou'] < REDRAW_LOW_IOU]))
    for _, name in SHORT_BANDS:
        out.append(('short side', name,
                    [i for i, it in enumerate(items)
                     if short_band(it['short_side']) == name]))
    out.append(('prior', 'against regional prior',
                [i for i, it in enumerate(items) if against_prior(it)]))
    out.append(('prior', 'with regional prior',
                [i for i, it in enumerate(items)
                 if against_prior(it) is False]))
    for r in sorted({it['region'] for it in items if it['region']}):
        out.append(('region', r, [i for i, it in enumerate(items)
                                  if it['region'] == r]))
    return out


def rank_auc(y, s):
    """Mann-Whitney ROC AUC with tied scores sharing their mean rank; None
    with one class. Here rather than sklearn's so the test env needs none."""
    n1 = sum(y)
    n0 = len(y) - n1
    if not n1 or not n0:
        return None
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    r1 = sum(r for r, t in zip(ranks, y) if t)
    return (r1 - n1 * (n1 + 1) / 2) / (n1 * n0)


def metrics(y, p, t=THRESHOLD):
    """y: 1 = leashed; p: P(leashed). Headline numbers at t, one class allowed
    (balanced accuracy and AUC are then None, the recall that exists stays)."""
    n = len(y)
    n_pos = sum(y)
    n_neg = n - n_pos
    tp = sum(1 for a, b in zip(y, p) if a and b >= t)
    tn = sum(1 for a, b in zip(y, p) if not a and b < t)
    rp = tp / n_pos if n_pos else None
    rn = tn / n_neg if n_neg else None
    auc = rank_auc(y, p)

    def r4(v):
        return None if v is None else round(v, 4)
    return {
        'n': n, f'n_{POS}': n_pos, f'n_{NEG}': n_neg,
        'balanced_accuracy': r4((rp + rn) / 2) if n_pos and n_neg else None,
        'accuracy': r4((tp + tn) / n) if n else None,
        f'recall_{POS}': r4(rp),
        f'recall_{POS}_ci95': [r4(v) for v in wilson(tp, n_pos)]
        if n_pos else None,
        f'recall_{NEG}': r4(rn),
        f'recall_{NEG}_ci95': [r4(v) for v in wilson(tn, n_neg)]
        if n_neg else None,
        'auc': r4(auc),
        'confusion': {'tp': tp, 'fn': n_pos - tp, 'tn': tn,
                      'fp': n_neg - tn},
    }


def mcnemar_exact(ok_a, ok_b):
    """Two-sided exact McNemar on paired per-item correctness."""
    a_only = sum(1 for a, b in zip(ok_a, ok_b) if a and not b)
    b_only = sum(1 for a, b in zip(ok_a, ok_b) if b and not a)
    n = a_only + b_only
    if not n:
        return {'a_only': 0, 'b_only': 0, 'p': 1.0}
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {'a_only': a_only, 'b_only': b_only,
            'p': round(min(1.0, 2 * tail), 6)}


def _ba(y, p, idx, t=THRESHOLD):
    tp = tn = n1 = 0
    for i in idx:
        if y[i]:
            n1 += 1
            tp += p[i] >= t
        else:
            tn += p[i] < t
    n0 = len(idx) - n1
    if not n1 or not n0:
        return None
    return (tp / n1 + tn / n0) / 2


def bootstrap_ba_diff(y, pa, pb, reps=2000, seed=20261115):
    """Paired bootstrap 95% interval on BA(a) - BA(b), items resampled with
    replacement. Fixed seed: the same scores give the same interval."""
    rng = random.Random(seed)
    n = len(y)
    diffs = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        a, b = _ba(y, pa, idx), _ba(y, pb, idx)
        if a is not None and b is not None:
            diffs.append(a - b)
    if not diffs:
        return None
    diffs.sort()
    lo = diffs[int(0.025 * (len(diffs) - 1))]
    hi = diffs[int(0.975 * (len(diffs) - 1))]
    return [round(lo, 4), round(hi, 4)]


def run_test_set(doc, items, models, predict, trained_on=()):
    """Score each model on the annotated items; the report as a dict.

    models: [{'name', 'weights', 'imgsz', 'data_roots'}]; predict(model,
    paths) -> [P(leashed)] in order -- a parameter so the test can stand a
    predictor in without loading a checkpoint. Items any model may have
    trained on (drop_trained_on against its own dataset and --trained-on)
    are excluded from EVERY model, so the table and the paired tests are on
    one item set.
    """
    contaminated = {}
    for m in models:
        paths = [it['path'] for it in items]
        bad = []
        for root in list(m.get('data_roots') or ()) + list(trained_on):
            paths, more = drop_trained_on(paths, root)
            bad += more
        if bad:
            contaminated[m['name']] = sorted(bad)
    dirty = {p for v in contaminated.values() for p in v}
    keep = [it for it in items if it['path'] not in dirty]
    if not keep:
        raise SystemExit('NO TEST NUMBER PRODUCED: every item is in a '
                         "model's training data")
    y = [1 if it['label'] == POS else 0 for it in keep]
    slices = slice_index(keep)
    scores, per_model = {}, {}
    for m in models:
        s = [float(v) for v in predict(m, [it['path'] for it in keep])]
        if len(s) != len(keep):
            raise SystemExit(f"{m['name']}: {len(s)} scores for "
                             f'{len(keep)} items')
        scores[m['name']] = s
        sl = {}
        for group, name, idx in slices:
            sl.setdefault(group, {})[name] = metrics([y[i] for i in idx],
                                                     [s[i] for i in idx])
        per_model[m['name']] = {
            'weights': m['weights'], 'imgsz': m['imgsz'],
            'excluded_trained_on': [os.path.basename(p) for p in
                                    contaminated.get(m['name'], [])],
            'headline': sl['all']['all'], 'slices': sl}
    paired = []
    names = [m['name'] for m in models]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pa, pb = scores[a], scores[b]
            ok_a = [(v >= THRESHOLD) == bool(t) for v, t in zip(pa, y)]
            ok_b = [(v >= THRESHOLD) == bool(t) for v, t in zip(pb, y)]
            ba_a = per_model[a]['headline']['balanced_accuracy']
            ba_b = per_model[b]['headline']['balanced_accuracy']
            paired.append({
                'a': a, 'b': b, 'n': len(y), 'ba_a': ba_a, 'ba_b': ba_b,
                'ba_diff': (None if ba_a is None or ba_b is None
                            else round(ba_a - ba_b, 4)),
                'ba_diff_ci95_bootstrap': bootstrap_ba_diff(y, pa, pb),
                'mcnemar_exact': mcnemar_exact(ok_a, ok_b)})
    return {
        'test_set': {'created': doc.get('created'),
                     'n_items': len(items), 'n_scored': len(keep),
                     'n_excluded_trained_on': len(items) - len(keep),
                     'fallback_crops_from': sum(it['fallback']
                                                for it in items)},
        'protocol': {'threshold': THRESHOLD,
                     'crops': 'leash_audit full/ files as frozen '
                              '(sha256-verified), predicted unchanged',
                     'weighting': 'none -- p_model and band_populations '
                                  'are not used'},
        'models': per_model,
        'paired': paired,
        'items': [{'key': it['key'], 'label': it['label'],
                   'short_side': it['short_side'], 'region': it['region'],
                   'redrawn': it['redrawn'], 'redraw_iou': it['redraw_iou'],
                   'p': {n: round(scores[n][k], 5) for n in names}}
                  for k, it in enumerate(keep)],
    }


def _f(v, w=6):
    return f'{v:.4f}'.rjust(w) if isinstance(v, (int, float)) else '-'.rjust(w)


def print_test_report(rep):
    ts = rep['test_set']
    print(f"\nTEST SET: {ts['n_scored']} of {ts['n_items']} items scored "
          f"(excluded as trained on: {ts['n_excluded_trained_on']}; found "
          f"only via crops_from: {ts['fallback_crops_from']})")
    print('  unweighted, threshold 0.5; p_model and band weights not used')
    names = list(rep['models'])
    rl, ru = f'recall_{POS}', f'recall_{NEG}'
    for name in names:
        m = rep['models'][name]
        print(f"\n{name}   imgsz {m['imgsz']}   {m['weights']}")
        print(f"  {'slice':<44} {'n':>4} {'n_lsh':>5}   {'BA':>6} "
              f"{'acc':>6} {'r_lsh':>6} {'r_unl':>6} {'AUC':>6}")
        for group, sl in m['slices'].items():
            for sname, v in sl.items():
                label = sname if group in ('all', sname) else \
                    f'{group}: {sname}'
                print(f"  {label[:44]:<44} {v['n']:>4} {v[f'n_{POS}']:>5}   "
                      f"{_f(v['balanced_accuracy'])} {_f(v['accuracy'])} "
                      f"{_f(v[rl])} {_f(v[ru])} {_f(v['auc'])}")
    if len(names) > 1:
        print('\nSIDE BY SIDE (balanced accuracy / leashed recall / unleashed '
              'recall)')
        head = ''.join(f'{n[:22]:>24}' for n in names)
        print(f"  {'slice':<40}{head}")
        first = rep['models'][names[0]]['slices']
        for group, sl in first.items():
            for sname in sl:
                label = sname if group in ('all', sname) else \
                    f'{group}: {sname}'
                cells = ''
                for n in names:
                    v = rep['models'][n]['slices'][group][sname]
                    cells += (f"{_f(v['balanced_accuracy'])}/"
                              f"{_f(v[rl])}/{_f(v[ru])}").rjust(24)
                print(f'  {label[:40]:<40}{cells}')
        print('\nPAIRED (same items; McNemar exact on correctness at 0.5, '
              'bootstrap 95% CI on the BA difference)')
        for pr in rep['paired']:
            mc = pr['mcnemar_exact']
            ci = pr['ba_diff_ci95_bootstrap']
            ci_s = f'[{ci[0]:+.4f}, {ci[1]:+.4f}]' if ci else '-'
            # None when the scored items hold one class (BA undefined)
            d = pr['ba_diff']
            d_s = '-' if d is None else f'{d:+.4f}'
            print(f"  {pr['a']} - {pr['b']}: BA {_f(pr['ba_a'])} vs "
                  f"{_f(pr['ba_b'])}  diff {d_s} {ci_s}   "
                  f"only-{pr['a']}-right {mc['a_only']}, "
                  f"only-{pr['b']}-right {mc['b_only']}, p={mc['p']:.4g}")


def _run_name(weights, taken):
    d = os.path.dirname(os.path.abspath(weights))
    run = os.path.dirname(d) if os.path.basename(d) == 'weights' else d
    name = f'{os.path.basename(run)}/{os.path.basename(weights)}'
    base, k = name, 2
    while name in taken:
        name, k = f'{base}#{k}', k + 1
    return name


def test_set_main(args):
    """--test-set: load and verify, annotate, score every checkpoint, report.

    Everything a number could silently depend on is checked BEFORE a model is
    loaded, so a refused set costs seconds, not a model load.
    """
    # module globals passed explicitly, so a test can point them elsewhere
    doc, items = load_test_set(args.test_set, AUDIT_FULL)
    unknown = annotate_items(items, AUDIT_POOL, CORRECTIONS)
    print(f'test set: {args.test_set}  ({len(items)} items, sha256 verified)')
    if unknown['region']:
        print(f"  {unknown['region']} item(s) have no region (not in the "
              f'audit pool) -- in no region or prior slice')
    if unknown['redrawn']:
        print(f"  {unknown['redrawn']} item(s) have no redraw record either "
              f'way (no corrections ledger) -- in no redraw slice')
    if unknown['redraw_iou']:
        print(f"  {unknown['redraw_iou']} redrawn item(s) have no pool box "
              f'-- in no redraw-IoU slice')

    from ultralytics import YOLO
    from gate_store import model_imgsz
    weights = [args.weights] + list(args.compare or ())
    models, loaded = [], {}
    for w in weights:
        model = YOLO(w)
        imgsz = args.imgsz or model_imgsz(model)
        if not imgsz:
            raise SystemExit(f'{w}: the checkpoint does not record its train '
                             f'imgsz -- pass --imgsz')
        if args.imgsz:
            print(f'WARNING: --imgsz {args.imgsz} overrides the checkpoint\'s '
                  f'own size for {w}; that is not the production protocol')
        own = ((getattr(model, 'ckpt', None) or {}).get('train_args')
               or {}).get('data')
        roots = [args.data]
        if own and os.path.isdir(own):
            roots.append(own)
        else:
            print(f'WARNING: {w} trained on {own!r}, which is not here -- its '
                  f'own training data cannot be checked against the test '
                  f'set; pass it with --trained-on')
        name = _run_name(w, loaded)
        loaded[name] = model
        models.append({'name': name, 'weights': w, 'imgsz': imgsz,
                       'data_roots': roots})
        print(f'model   : {name}  imgsz {imgsz}  classes {model.names}')

    def predict(m, paths):
        # PIL crops, as gate_store hands predict() its in-memory cut_crops
        # output; the file is that crop, so nothing is re-padded or resized
        from PIL import Image
        model = loaded[m['name']]
        idx = [i for i, n in model.names.items()
               if str(n).lower() == POS]
        if not idx:
            raise SystemExit(f'{m["name"]}: no "{POS}" class in '
                             f'{model.names}')
        out = []
        for i in range(0, len(paths), args.batch):
            ims = []
            for p in paths[i:i + args.batch]:
                with Image.open(p) as im:
                    ims.append(im.convert('RGB'))
            for r in model.predict(ims, imgsz=m['imgsz'], verbose=False,
                                   device=args.device):
                out.append(float(r.probs.data[idx[0]]))
        return out

    rep = run_test_set(doc, items, models, predict, args.trained_on)
    rep['test_set']['path'] = os.path.relpath(os.path.abspath(args.test_set),
                                              REPO)
    rep['test_set']['doc_sha256'] = _sha256(args.test_set)
    for name, n in ((n, len(v['excluded_trained_on']))
                    for n, v in rep['models'].items()):
        if n:
            print(f'\nWARNING: {n} test item(s) are in the training data of '
                  f'{name} -- excluded from every model below; investigate')
    # written BEFORE printing: the scoring is the expensive part (6 min for
    # two models on CPU), and a print that throws must not take it with it
    out = args.out or args.json or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(args.weights))),
        os.path.splitext(os.path.basename(args.test_set))[0] + '_eval.json')
    with open(out, 'w') as fh:
        json.dump(rep, fh, indent=1)
    print_test_report(rep)
    print(f'\nreport -> {out}')
    return 0


def main(argv=None):
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
    # default: the size the model was TRAINED at, read from its checkpoint --
    # the same lookup tools/detect/gate_store.py predicts with, so an
    # acceptance number and a store verdict are made at one size
    ap.add_argument('--imgsz', type=int, default=0)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='0')
    ap.add_argument('--json', help='also write the metrics here')
    ap.add_argument('--test-set', metavar='PATH',
                    help='score on a frozen test set (data/leash_test_v1.json)'
                         ' instead of val and the acceptance set; see above')
    ap.add_argument('--compare', nargs='+', metavar='WEIGHTS',
                    help='--test-set only: more checkpoints, scored on the '
                         'same items, side by side, with paired tests')
    ap.add_argument('--out', help='--test-set only: the JSON report '
                    '(default: <run dir of --weights>/<test set>_eval.json)')
    args = ap.parse_args(argv)
    if args.test_set:
        return test_set_main(args)
    if args.compare:
        ap.error('--compare needs --test-set')
    if args.out:
        ap.error('--out needs --test-set (the acceptance path writes --json)')

    from ultralytics import YOLO
    import numpy as np

    model = YOLO(args.weights)
    print(f'model   : {args.weights}')
    if not args.imgsz:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from gate_store import model_imgsz
        args.imgsz = model_imgsz(model)
        if not args.imgsz:
            raise SystemExit('the checkpoint does not record its train '
                             'imgsz -- pass --imgsz')
    print(f'imgsz   : {args.imgsz}')
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
