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

THE MODEL is SigLIP 2, zero-shot, asked our own question in our own words
(see PROMPTS). The alternative -- an ImageNet-1k classifier -- is still one
flag away with --model imagenet, and the measured gap between them is large;
the table above PROMPTS has the numbers, taken against the crops a human has
already ruled on rather than against a public benchmark.

Bucket probability is the SUMMED mass over a bucket's labels, not the top
one's. A 40px dog spreads its score over several dog-ish phrasings and can
land top-1 somewhere odd; the mass on "some kind of dog" is still decisive,
and that is what the filter sorts by.

    python tools/detect/triage_crops.py --limit 200        # try it
    python tools/detect/triage_crops.py                    # everything unjudged
    python tools/detect/triage_crops.py --watch 600        # keep it current
    python tools/detect/triage_crops.py --device cuda      # when the GPU is idle
    python tools/detect/triage_crops.py \
        --model google/siglip2-large-patch16-256 --refresh   # slower, better

Changing --model changes what the buckets mean, so re-run with --refresh
after a switch: the reader takes the LAST record for a crop, and a file with
two models' opinions in it would filter by whichever ran last.

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

# ── the zero-shot backend, and why it is the default ────────────────────────
# Measured against the 1,693 crops a human has already ruled on -- the exact
# question this filter helps with -- ranking them by "is this a dog":
#
#   yolo26n-cls        AUC 0.676   423 crops/s     ImageNet-1k, nano tier
#   efficientnet_v2_s  AUC 0.755    11 crops/s     ImageNet-1k
#   siglip2-base       AUC 0.888   4.3 crops/s     zero-shot, our own labels
#   siglip2-large      AUC 0.945   0.7 crops/s     zero-shot (308-crop subset)
#
# The ImageNet models share a ceiling the accuracy number hides: 1000 fixed
# classes, none of which is "an empty road", and no way to abstain -- every
# crop is forced into some label. A zero-shot model is asked OUR question in
# OUR words, which is why the jump is so much larger than the gap in their
# published top-1 scores.
#
# Each prompt declares the bucket it belongs to, so the mapping is a table
# rather than an index range that has to be verified against a class order.
# The short name is what the tile shows.
SIGLIP_DEFAULT = 'google/siglip2-base-patch16-224'
PROMPTS = [
    ('dog', 'dog', 'a photo of a dog'),
    ('dog', 'street dog', 'a street dog lying on the road'),
    ('dog', 'puppy', 'a photo of a puppy'),
    ('dog', 'dog', 'a dog walking on a street'),
    ('animal', 'cow', 'a photo of a cow'),
    ('animal', 'ox', 'an ox pulling a cart'),
    ('animal', 'goat', 'a photo of a goat'),
    ('animal', 'sheep', 'a photo of a sheep'),
    ('animal', 'horse', 'a photo of a horse'),
    ('animal', 'donkey', 'a photo of a donkey'),
    ('animal', 'camel', 'a photo of a camel'),
    ('animal', 'pig', 'a photo of a pig'),
    ('animal', 'cat', 'a photo of a cat'),
    ('animal', 'bird', 'a photo of a bird'),
    ('animal', 'chicken', 'a chicken or other poultry'),
    ('animal', 'monkey', 'a monkey or ape'),
    ('animal', 'deer', 'a deer or antelope'),
    ('animal', 'elephant', 'a photo of an elephant'),
    ('animal', 'wild animal', 'a wild animal in the distance'),
    ('object', 'car', 'a car or truck'),
    ('object', 'motorcycle', 'a motorcycle or scooter'),
    ('object', 'bicycle', 'a bicycle'),
    ('object', 'person', 'a person walking'),
    ('object', 'building', 'a building or wall'),
    ('object', 'road', 'an empty road or pavement'),
    ('object', 'plants', 'a tree, bush or grass'),
    ('object', 'rubbish', 'a pile of rubbish or debris'),
    ('object', 'sign', 'a road sign or street furniture'),
    ('object', 'nothing', 'a blurry photo of nothing in particular'),
]


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

    The key is 'crop'. This read 'name' -- the key THIS tool writes -- so it
    matched nothing and skipped nobody: an exclusion that excluded zero
    crops. It cost little (judged crops leave the review pool anyway, so only
    6 overlapped) but a guard that never fires is not a guard.
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
                    if isinstance(r, dict):
                        # 'crop' is the ledgers' key; 'name' appears only in
                        # this tool's own output
                        nm = r.get('crop') or r.get('name')
                        if nm:
                            out.add(nm)
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
    ap.add_argument('--model', default=SIGLIP_DEFAULT,
                    help="a SigLIP 2 id, or 'imagenet' for the old "
                         'EfficientNet backend. Bigger SigLIP is better and '
                         'much slower: base 4.3 crops/s, large 0.7 (CPU).')
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--include-judged', action='store_true',
                    help='also predict crops already ruled on (for measuring '
                         'the model against human verdicts -- still never '
                         'written back)')
    ap.add_argument('--refresh', action='store_true',
                    help='re-predict crops already in the output file')
    ap.add_argument('--verify-buckets', action='store_true',
                    help='check the bucket edges and exit')
    ap.add_argument('--watch', type=int, default=0, metavar='SECONDS',
                    help='keep going: sleep this long and pick up whatever '
                         'the sweep has written since. The live pool turns '
                         'over at ~2 crops/s, so a single pass leaves the '
                         'queue partly unguessed within the hour.')
    args = ap.parse_args()

    import torch
    from PIL import Image
    torch.set_num_threads(max(1, args.threads))
    imagenet = args.model == 'imagenet'

    if imagenet:
        from torchvision.models import (efficientnet_v2_s,
                                        EfficientNet_V2_S_Weights)
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
        model = efficientnet_v2_s(weights=weights).eval().to(args.device)
        tf = weights.transforms()
        model_id = MODEL_ID
    else:
        from transformers import AutoModel, AutoProcessor
        if args.verify_buckets:
            n = {}
            for bk, _, _ in PROMPTS:
                n[bk] = n.get(bk, 0) + 1
            print(f'{len(PROMPTS)} prompts: {n}')
            return 0
        proc = AutoProcessor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).eval().to(args.device)
        model_id = args.model
        # the text side is fixed, so encode it once for the whole run
        with torch.no_grad():
            tok = proc(text=[p for _, _, p in PROMPTS], padding='max_length',
                       max_length=64, return_tensors='pt').to(args.device)
            tfeat = model.get_text_features(**tok)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    def score(ims):
        """[(bucket_mass, [(bucket, name, p), ...])] for a batch of images."""
        if imagenet:
            batch = torch.stack([tf(im) for im in ims]).to(args.device)
            with torch.no_grad():
                probs = model(batch).softmax(1).cpu()
            out = []
            for row in probs:
                mass = {'dog': 0.0, 'animal': 0.0, 'object': 0.0}
                for idx in range(len(cats)):
                    mass[bucket_of(idx)] += float(row[idx])
                out.append((mass, [(bucket_of(i), cats[i], float(row[i]))
                                   for i in range(len(cats))]))
            return out
        px = proc(images=ims, return_tensors='pt').to(args.device)
        with torch.no_grad():
            ifeat = model.get_image_features(**px)
            ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
            logits = ifeat @ tfeat.T * model.logit_scale.exp() + model.logit_bias
            # SigLIP scores each label independently (sigmoid, not softmax),
            # so normalise across the prompt table to get a share per bucket
            p = torch.sigmoid(logits).cpu()
            p = p / p.sum(1, keepdim=True).clamp(min=1e-9)
        out = []
        for row in p:
            mass = {'dog': 0.0, 'animal': 0.0, 'object': 0.0}
            for j, (bk, _, _) in enumerate(PROMPTS):
                mass[bk] += float(row[j])
            out.append((mass, [(PROMPTS[j][0], PROMPTS[j][1], float(row[j]))
                               for j in range(len(PROMPTS))]))
        return out

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    print(f'model: {model_id}')

    def once(first):
        """One pass over whatever is unpredicted right now."""
        names = pool(REPO)
        # --refresh means "redo them", which must apply to the FIRST pass
        # only: on a later --watch pass it would loop over the same crops
        # forever and never reach the new ones.
        skip = set() if (args.refresh and first) else already_done(args.out)
        if not args.include_judged:
            skip |= judged_names(REPO)
        todo = sorted((n, d) for n, d in names if n not in skip)
        if args.limit:
            todo = todo[:args.limit]
        print(f'{len(names):,} crops in the pool, {len(todo):,} to predict '
              f'({len(skip):,} already judged or done)')
        if not todo:
            return 0
        ran_at = time.strftime('%Y-%m-%dT%H:%M:%S')
        wrote = unreadable = 0
        t0 = time.time()
        # Appended a batch at a time and flushed, so a run killed halfway
        # leaves every completed batch usable -- meant to be run in bursts.
        with open(args.out, 'a') as fh:
            for i in range(0, len(todo), args.batch):
                ims, keep = [], []
                for nm, d in todo[i:i + args.batch]:
                    try:
                        with Image.open(os.path.join(d, nm)) as im:
                            ims.append(im.convert('RGB'))
                        keep.append(nm)
                    except Exception:
                        # the live pool is pruned while this runs; a crop
                        # that vanished mid-pass is not an error
                        unreadable += 1
                if not ims:
                    continue
                scored = score(ims)
                for im in ims:
                    im.close()
                for j, nm in enumerate(keep):
                    mass, per_label = scored[j]
                    best = max(mass, key=mass.get)
                    # The name on the tile must belong to the bucket the tile
                    # was filed under. Mass decides the bucket, so a crop can
                    # land in 'dog' on the sum of many dog-ish labels while
                    # its single best label sits elsewhere, and a chip that
                    # disagrees with the filter that surfaced it reads as a
                    # bug. Best label INSIDE the winning bucket.
                    in_b = [(nm2, pr) for bk, nm2, pr in per_label
                            if bk == best]
                    gname, gp = max(in_b, key=lambda t: t[1])
                    top = sorted(per_label, key=lambda t: -t[2])[:args.topk]
                    fh.write(json.dumps({
                        'schema': SCHEMA,
                        'name': nm,
                        'bucket': best,
                        'p': round(mass[best], 4),
                        'mass': {k: round(v, 4) for k, v in mass.items()},
                        # 'guess', never 'label': the ledgers call the HUMAN
                        # verdict 'label', and a file that must never be
                        # mistaken for a ledger should not borrow its key.
                        'guess': gname,
                        'guess_p': round(gp, 4),
                        # the raw top-k too, so a disagreement is inspectable
                        'top': [[t[1], round(t[2], 4)] for t in top],
                        # stamped on every record so nothing downstream can
                        # read one of these and mistake it for a decision
                        'unverified': True,
                        'source': 'model_suggestion',
                        'model': model_id,
                        'ran_at': ran_at,                    }) + '\n')
                    wrote += 1
                fh.flush()
                if (i // args.batch) % 10 == 0:
                    el = time.time() - t0
                    print(f'  {wrote:,}/{len(todo):,}  '
                          f'{wrote / el if el else 0:.1f}/s', end='\r',
                          flush=True)
        el = time.time() - t0
        print(f'\n{wrote:,} predictions -> {args.out} in {el:.0f}s '
              f'({wrote / el:.1f}/s)' if el else f'\n{wrote:,} predictions')
        if unreadable:
            print(f'  {unreadable:,} vanished or unreadable, skipped')
        return wrote

    once(True)
    if args.watch:
        print(f'watching: another pass every {args.watch}s, Ctrl-C to stop')
        try:
            while True:
                time.sleep(args.watch)
                once(False)
        except KeyboardInterrupt:
            print('\nstopped')
    print('These are suggestions for sorting the queue. Nothing reads this '
          'file except the review page filter.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
