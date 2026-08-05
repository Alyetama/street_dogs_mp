#!/usr/bin/env python3
"""Guess what each unreviewed crop contains, to sort the queue -- never to label it.

WHAT THIS IS FOR. The review queue is undifferentiated: a page of 50 crops
mixes dogs, goats, parked cars and blurred hedges, and the reviewer pays the
same attention to each. A general-purpose classifier cannot decide any of
them, but it can say "these fifty look like vehicles" well enough that a
person can work through one kind of mistake at a time.

WHAT THIS IS NOT. These predictions are not labels, not weak labels, not
pre-annotations, and there is no code path that turns them into any of those.
They are written to their own file, in their own format, with `unverified`
stamped on every record, and NOTHING that builds a training set reads it:

  the ledgers          data/hard_negatives/labels.jsonl, data/hard_positives/
                       labels.jsonl -- written only by a human clicking a
                       verdict, and this tool never opens them for writing
  the crop dataset     built by rebuild_crop_dataset.py from those ledgers
  this file            data/dashboard/triage.jsonl -- read by the review
                       page for filtering, and by nothing else

tools/detect/tests/adv_triage_isolation.py asserts that separation against
the source rather than trusting this comment.

THE MODEL is torchvision's ImageNet-1k EfficientNet-V2-S: 1000 classes that
happen to suit the question -- 118 domestic dog breeds, ~280 other animals,
and 602 inanimate objects. The three buckets are derived from the class
ORDER, which is a property of the ImageNet-1k label set (verified in
BUCKETS below), not from a hand-written list that could drift.

Bucket probability is the SUMMED softmax mass over a bucket's classes, not
the top-1 label's. On a 40px crop of a dog the model often spreads its mass
over a dozen breeds and lands top-1 on something absurd; the mass over
"some dog" is still decisive, and that is the number the filter uses.

    python tools/detect/triage_crops.py --limit 200        # try it
    python tools/detect/triage_crops.py                    # everything unjudged
    python tools/detect/triage_crops.py --device cuda      # when the GPU is idle

Resumable: a crop already in the output is skipped, so this can be run in
short bursts on a busy machine and picked up later.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# Bucket boundaries in the ImageNet-1k class order. These are not guesses:
# index 151 is 'Chihuahua' and 268 'Mexican hairless' (the domestic breeds
# run between them), 397 is 'puffer' -- the last animal -- and 398 'abacus',
# the first artifact. --verify-buckets re-checks all four against the
# model's own category list before writing anything.
DOG_LO, DOG_HI = 151, 268           # inclusive, domestic dogs
ANIMAL_HI = 397                     # inclusive, last animal class
EDGE = {DOG_LO: 'Chihuahua', DOG_HI: 'Mexican hairless',
        ANIMAL_HI: 'puffer', ANIMAL_HI + 1: 'abacus'}

OUT_FILE = os.path.join(REPO, 'data', 'dashboard', 'triage.jsonl')
SCHEMA = 1
MODEL_ID = 'efficientnet_v2_s.imagenet1k_v1'


def bucket_of(i):
    """dog | animal | object, from the class index alone."""
    if DOG_LO <= i <= DOG_HI:
        return 'dog'
    if i <= ANIMAL_HI:
        return 'animal'
    return 'object'


def judged_names(repo):
    """Crop FILENAMES a human has already ruled on.

    Read-only, and only to decide what not to bother predicting. The ledgers
    are the ground truth this tool exists to stay out of.
    """
    out = set()
    for rel in (('data', 'hard_negatives', 'labels.jsonl'),
                ('data', 'hard_positives', 'labels.jsonl')):
        p = os.path.join(repo, *rel)
        try:
            with open(p) as fh:
                for ln in fh:
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue
                    if isinstance(r, dict) and r.get('name'):
                        out.add(r['name'])
        except OSError:
            pass
    return out


def already_done(path):
    """Names already predicted, so a re-run resumes instead of redoing."""
    out = set()
    try:
        with open(path) as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get('name'):
                    out.add(r['name'])
    except OSError:
        pass
    return out


def pool(repo):
    """[(name, dir)] over both review pools, mirroring the dashboard's own."""
    sys.path.insert(0, os.path.join(repo, 'tools', 'dashboard'))
    try:
        import dashboard as dash
        return dash.review_pool_names()
    except Exception:
        # Standalone fallback: the dashboard module pulls in duckdb, which
        # need not exist in whichever env has torch.
        out = []
        for d in (os.path.join(repo, 'data', 'dashboard', 'recent_crops'),
                  os.path.join(repo, 'data', 'dashboard', 'review_set')):
            try:
                out += [(n, d) for n in os.listdir(d) if n.endswith('.jpg')]
            except OSError:
                pass
        return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=OUT_FILE)
    ap.add_argument('--limit', type=int, default=0,
                    help='stop after this many crops (0 = no limit)')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--threads', type=int, default=4,
                    help='CPU threads; the sweep and any training run want '
                         'the rest of them')
    ap.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--include-judged', action='store_true',
                    help='also predict crops already ruled on (for measuring '
                         'the model against human verdicts -- still never '
                         'written back)')
    ap.add_argument('--refresh', action='store_true',
                    help='re-predict crops already in the output file')
    ap.add_argument('--verify-buckets', action='store_true',
                    help='check the bucket edges and exit')
    args = ap.parse_args()

    import torch
    from PIL import Image
    from torchvision.models import (efficientnet_v2_s,
                                    EfficientNet_V2_S_Weights)
    torch.set_num_threads(max(1, args.threads))
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    cats = weights.meta['categories']

    # The buckets are an assertion about the label set, so check it. A
    # different weights enum with a different order would otherwise be
    # silently mis-bucketed into every prediction this tool ever writes.
    bad = [f'{i}: expected {want!r}, got {cats[i]!r}'
           for i, want in EDGE.items() if cats[i] != want]
    if bad:
        raise SystemExit('ImageNet class order is not what the buckets '
                         'assume:\n  ' + '\n  '.join(bad))
    if args.verify_buckets:
        n = {b: sum(1 for i in range(len(cats)) if bucket_of(i) == b)
             for b in ('dog', 'animal', 'object')}
        print(f'bucket edges verified against {len(cats)} classes: {n}')
        return 0

    names = pool(REPO)
    skip = set() if args.refresh else already_done(args.out)
    if not args.include_judged:
        skip |= judged_names(REPO)
    todo = [(n, d) for n, d in names if n not in skip]
    todo.sort()
    if args.limit:
        todo = todo[:args.limit]
    print(f'{len(names):,} crops in the pool, {len(todo):,} to predict '
          f'({len(skip):,} already judged or done)')
    if not todo:
        return 0

    model = efficientnet_v2_s(weights=weights).eval().to(args.device)
    tf = weights.transforms()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ran_at = time.strftime('%Y-%m-%dT%H:%M:%S')
    wrote = unreadable = 0
    t0 = time.time()
    # Appended a batch at a time and flushed, so a run killed halfway leaves
    # every completed batch usable -- this is meant to be run in bursts.
    with open(args.out, 'a') as fh:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            ims, keep = [], []
            for nm, d in chunk:
                try:
                    with Image.open(os.path.join(d, nm)) as im:
                        ims.append(tf(im.convert('RGB')))
                    keep.append(nm)
                except Exception:
                    unreadable += 1
            if not ims:
                continue
            with torch.no_grad():
                p = model(torch.stack(ims).to(args.device)).softmax(1).cpu()
            for j, nm in enumerate(keep):
                probs = p[j]
                # summed mass per bucket, not the top-1's bucket: a small
                # crop scatters its mass across breeds and the sum is the
                # only stable signal
                mass = {'dog': 0.0, 'animal': 0.0, 'object': 0.0}
                for idx in range(len(cats)):
                    mass[bucket_of(idx)] += float(probs[idx])
                top = torch.topk(probs, args.topk)
                best = max(mass, key=mass.get)
                # The name shown on the tile must belong to the bucket the
                # tile was filed under. The overall top-1 need not: mass
                # decides the bucket, so a crop can land in 'dog' on the sum
                # of forty breeds while its single best guess is a rooster,
                # and a chip reading 'cock' on a dog-filtered tile reads as
                # a bug. Take the best class INSIDE the winning bucket.
                in_b = [i for i in range(len(cats)) if bucket_of(i) == best]
                bi = max(in_b, key=lambda i: float(probs[i]))
                fh.write(json.dumps({
                    'schema': SCHEMA,
                    'name': nm,
                    'bucket': best,
                    'p': round(mass[best], 4),
                    'mass': {k: round(v, 4) for k, v in mass.items()},
                    # what to call it: always a member of `bucket`
                    'label': cats[bi],
                    'label_p': round(float(probs[bi]), 4),
                    # the raw top-k too, so a disagreement stays inspectable
                    'top': [[cats[int(t)], round(float(s), 4)]
                            for s, t in zip(top.values, top.indices)],
                    # stamped on every record so nothing downstream can read
                    # one of these and mistake it for a decision
                    'unverified': True,
                    'source': 'model_suggestion',
                    'model': MODEL_ID,
                    'ran_at': ran_at,
                }) + '\n')
                wrote += 1
            fh.flush()
            if (i // args.batch) % 10 == 0:
                el = time.time() - t0
                rate = wrote / el if el else 0
                print(f'  {wrote:,}/{len(todo):,}  {rate:.1f}/s', end='\r',
                      flush=True)
    el = time.time() - t0
    print(f'\n{wrote:,} predictions -> {args.out} in {el:.0f}s '
          f'({wrote / el:.1f}/s)' if el else '')
    if unreadable:
        print(f'  {unreadable:,} unreadable, skipped')
    print('These are suggestions for sorting the queue. Nothing reads this '
          'file except the review page filter.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
