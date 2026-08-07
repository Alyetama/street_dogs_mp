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

Bucket probability is the MEAN mass over a bucket's labels, not the sum and
not the top one's. Averaging is what makes the buckets comparable: a 40px dog
spreads its score over several dog-ish phrasings, so the top label alone is
unreliable, but SUMMING hands the decision to whichever bucket has the most
prompts. With 4 dog, 15 animal and 10 object phrasings and the scores
normalised across the table, an image the model has no opinion about scores
15/29 for animal and 4/29 for dog -- it lands in "other animal" for no reason
but the size of the list.

Measured on dogbin_v5's val split (342 dog, 300 not_dog, hand-labelled):
summing put 12.3% real dogs in the "other animal" bucket and found 87.4% of
the dogs; averaging put 5.7% there and found 93.9%. Same model, same prompts.

    python tools/detect/triage_crops.py --limit 200        # try it
    python tools/detect/triage_crops.py                    # everything unjudged
    python tools/detect/triage_crops.py --watch 600        # keep it current
    python tools/detect/triage_crops.py --device cuda      # when the GPU is idle
    python tools/detect/triage_crops.py --refresh \
        --model google/siglip2-so400m-patch14-384 --device cuda   # the best one

The default is the base model because it is the one that runs anywhere: 23
crops/s on a CPU. siglip2-so400m-patch14-384 is materially better on the crops
this queue is hardest on -- dark, small, low-contrast dogs -- taking the
"other animal" bucket from 5.7% real dogs down to 2.0% and dog recall from
93.9% to 97.7% on the same split. It costs a GPU to be practical: 24 crops/s
on an RTX 5080 against 1.0 on the CPU, so a full pass is 16 minutes rather
than six and a half hours.

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
# Progress, for the dashboard. Separate from the predictions so a reader that
# only wants "is it running" never has to parse a 5,000-line file, and so a
# run that dies leaves a last known position behind rather than silence.
STATUS_FILE = os.path.join(REPO, 'data', 'dashboard', 'triage_status.json')
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
# so400m, not base. Measured on the same 1,693 human-judged crops: base put
# 5.7% of real dogs in 'other animals', so400m 2.0%, and 'most of the other
# animals are dogs' was the complaint that started this.
#
# The dashboard can name a model in its config and does, but the DEFAULT is
# what anyone running this by hand gets, and the two have to agree: crop
# vectors carry the model that made them, so one pass launched without
# --model quietly replaced a store of so400m vectors with base ones, leaving
# them in a different space from every search word already encoded. Base is
# still the right choice on a machine with no GPU -- pass SIGLIP_FAST.
SIGLIP_DEFAULT = 'google/siglip2-so400m-patch14-384'
SIGLIP_FAST = 'google/siglip2-base-patch16-224'
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


# How many phrasings each bucket has, so a bucket's score can be an average
# rather than a sum. Derived, never hand-written: a table edited without
# updating a constant beside it is a silent reweighting of every guess.
BUCKET_N = {}
for _b, _, _ in PROMPTS:
    BUCKET_N[_b] = BUCKET_N.get(_b, 0) + 1


# ── RF-DETR, the second opinion ─────────────────────────────────────────────
# A COCO detector rather than a zero-shot classifier, so it fails differently:
# it names a concrete class or says nothing, where SigLIP always has an
# opinion. That is the whole point of having two.
#
# IT IS THE WEAKER ONE ON THIS DATA, and by a lot. Measured against the 120
# crops a human has confirmed are dogs:
#
#     SigLIP so400m                            ~98%  called dog
#     RF-DETR medium, crop upscaled + padded    56%
#     RF-DETR large,  same                      57%  (and 21% of not-dogs
#                                                     wrongly called dog)
#     RF-DETR medium, whole frame, dog ANYWHERE 24%
#
# The reason is in the data: the pool's crops have a median long side of 35px
# and 78% are under 64px. A detector needs pixels on target; an embedding
# model degrades gracefully. Running it on the full frame is worse still --
# that is the small-object regime, and a 35px dog in a 1280x640 street scene
# is exactly what detectors miss. So the crop, upscaled, is the best of the
# three, and the dashboard shows the recall next to the choice.
#
# Where it EARNS its place: it calls only 5-8% of confirmed not-dogs 'dog',
# and it puts a fifth to a third of them in a named animal class. For finding
# more cows and horses to annotate, a detector's concrete label beats a
# bucket.
RFDETR_SIZES = {'rfdetr': 'RFDETRMedium', 'rfdetr-small': 'RFDETRSmall',
                'rfdetr-nano': 'RFDETRNano', 'rfdetr-large': 'RFDETRLarge'}
# Best of the sixteen crop preprocessings measured; see the table above.
# Upscaling gives the detector pixels, padding gives the object a plausible
# share of the frame, and both matter: no upscale and no pad is 27.5%.
RFDETR_UPSCALE = 160
RFDETR_PAD = 1.5
RFDETR_THRESHOLD = 0.25
# COCO's animals, minus the dog. The bucket rule is a table over class NAMES,
# not indices -- an index table silently re-buckets everything if the class
# order ever moves, which is the trap the ImageNet backend checks for.
COCO_ANIMALS = ('cat', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'bird')


def coco_bucket(name):
    """Which of the three buckets a COCO class belongs to."""
    if name == 'dog':
        return 'dog'
    return 'animal' if name in COCO_ANIMALS else 'object'


def backend_of(model_id):
    """Which backend produced a record, given the model it names.

    Records written before backends existed carry only `model`, so the answer
    has to be derivable from it -- 52,000 of them would otherwise vanish from
    a filter that asks for the backend by name.
    """
    # Deliberately free of module constants: the dashboard has to answer this
    # question identically about records it did not write, it cannot import
    # this file (different environments), and a guard compares the two by
    # running them side by side. Every RFDETR_SIZES key starts with 'rfdetr',
    # which the same guard checks.
    m = str(model_id or '')
    if m.startswith('rfdetr'):
        return 'rfdetr'
    if m == 'imagenet' or m.startswith('efficientnet'):
        return 'imagenet'
    return 'siglip'


def _oom(exc):
    """Is this the GPU telling us there is no room?"""
    return ('out of memory' in str(exc).lower()
            or type(exc).__name__ == 'OutOfMemoryError')


def _place(model, device, no_fallback=False):
    """Put a model on `device`, dropping to the CPU if the GPU has no room.

    This tool shares a card with whatever the box is training, and it is the
    least important thing on it. Dying when a training run takes the GPU was
    wrong twice over: the queue silently stopped being guessed, and the strip
    read "not running" with no reason, so it looked like the dashboard had
    failed rather than the graphics card being full. Slower beats stopped --
    the base model does 23 crops/s on a CPU.
    """
    import torch
    try:
        return model.to(device), device
    except Exception as e:                      # noqa: BLE001 - re-raised below
        if no_fallback or not _oom(e) or device == 'cpu':
            raise
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        print('CUDA is out of memory -- something else on this box has the '
              'card. Running on the CPU instead.', flush=True)
        return model.to('cpu'), 'cpu'


VEC_FILE = os.path.join(REPO, 'data', 'dashboard', 'triage_vecs.npz')


def vectored_names(path=None, model_id=None):
    """Crops that already have a vector FROM THIS MODEL.

    A prediction and a vector are written by the same forward pass but they
    are not the same fact, and treating them as one is what made free-text
    search useless: a crop predicted before this file existed is 'done' by the
    prediction ledger and has no vector, forever.

    The model has to match or the answer is worse than useless. A store full
    of base-model vectors would otherwise tell an so400m run that the whole
    pool is covered, it would embed nothing, and the search would stay broken
    in the one way the reader cannot see -- vectors and words in different
    spaces, every score meaningless.
    """
    import numpy as np
    try:
        d = np.load(path or VEC_FILE, allow_pickle=False)
        if model_id is not None and str(d['model']) != str(model_id):
            return set()
        return {str(x) for x in d['names']}
    except Exception:
        return set()


def save_vectors(names, rows, model_id, path=None, live=None):
    """Merge this pass's crop vectors into the store, newest winning.

    One row per crop, L2-normalised, float16 -- 1152 dims for so400m is 2.3 kB
    a crop, so the whole live pool is about 12 MB. The model id is written
    with them because a vector from one model cannot be compared with text
    encoded by another; a reader that finds a mismatch should ignore the file
    rather than return confident nonsense.

    ``live`` is the pool as it stands, and vectors for crops outside it are
    dropped. The store describes a queue that rotates -- 3,000 crops, turned
    over in under an hour on a working harvest -- so without this it fills
    with crops that were deleted hours ago. Measured: 4,513 vectors, 3,010
    crops in the pool, and NOT ONE in both. Every search ranked a set disjoint
    from the queue it was ordering, which looks exactly like a model returning
    nonsense.
    """
    if not names:
        return 0
    import numpy as np
    path = path or VEC_FILE
    keep_n, keep_v = [], None
    try:
        old = np.load(path, allow_pickle=False)
        if str(old['model']) == str(model_id):
            keep_n = [str(x) for x in old['names']]
            keep_v = old['vecs']
    except Exception:
        pass
    merged = {}
    if keep_v is not None:
        for i, nm in enumerate(keep_n):
            if live is None or nm in live:
                merged[nm] = keep_v[i]
    for i, nm in enumerate(names):
        merged[nm] = rows[i]
    nm_all = sorted(merged)
    arr = np.stack([merged[n] for n in nm_all]).astype('float16')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp.npz'
    np.savez(tmp, names=np.array(nm_all), vecs=arr,
             model=np.array(str(model_id)))
    os.replace(tmp, path)
    return len(nm_all)


def _owner_alive(path):
    """Is another LIVE run already publishing here?

    Two runs share one status file, so a short one-off finishing would write
    'not running' over a --watch run that is merely between passes, and the
    dashboard would call it stopped until the next batch. A run only reports
    its own end if nobody else owns the file.
    """
    try:
        with open(path) as fh:
            doc = json.load(fh)
        pid = int(doc.get('pid') or 0)
        if pid <= 0 or pid == os.getpid():
            return False
        os.kill(pid, 0)
        return bool(doc.get('running'))
    except Exception:
        return False


def write_status(path, **kw):
    """Publish where this run has got to. Atomic, because the dashboard
    polls it while it is being written, and a half-written JSON on the wire
    reads as 'no run' -- which is a lie about a run that is going fine."""
    kw.setdefault('schema', 1)
    kw.setdefault('pid', os.getpid())
    kw['updated'] = time.time()
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp_status = path + '.tmp'
        with open(tmp_status, 'w') as fh:
            json.dump(kw, fh)
        os.replace(tmp_status, path)
    except OSError:
        pass          # progress reporting must never break the run


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


def already_done(path, backend=None):
    """Names already predicted BY THIS BACKEND, so a re-run resumes.

    Per backend, because the two are meant to be comparable: switching to
    RF-DETR must re-guess the pool rather than inherit SigLIP's answers, and
    switching back must not throw SigLIP's away. One file still, one record
    per (crop, backend) -- the reader keeps the last line for each.
    """
    out = set()
    try:
        with open(path) as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if not isinstance(r, dict) or not r.get('name'):
                    continue
                if backend and backend_of(
                        r.get('backend') or r.get('model')) != backend:
                    continue
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
    # 'auto' rather than 'cpu': the dashboard's Run button passes no --device,
    # so a cpu default meant the button ran the big model on a CPU at under
    # 1 crop/s over a pool that rotates in an hour -- it could never catch up.
    # 'cuda' cannot be the default either, since this repo is public and a
    # clone without a card would only crash. _place still steps aside when a
    # training run has the card.
    ap.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
    ap.add_argument('--no-vectors', action='store_true',
                    help='do not keep the crop embeddings the model already '
                         'produces (they are what makes free-text search work)')
    ap.add_argument('--no-cpu-fallback', action='store_true',
                    help='fail instead of dropping to the CPU when the GPU is '
                         'full (default is to fall back and keep going)')
    ap.add_argument('--model', default=SIGLIP_DEFAULT,
                    help="a SigLIP 2 id, 'rfdetr' (or "
                         + '/'.join(sorted(k for k in RFDETR_SIZES
                                           if k != 'rfdetr'))
                         + ") for the COCO detector, or 'imagenet' for the "
                         'old EfficientNet. Bigger SigLIP is better and much '
                         f'slower on a CPU; {SIGLIP_FAST} is the one to pass '
                         'on a machine with no GPU (measured: 253 crops/s for '
                         'the default on this card, 4.3/s for base on the '
                         'CPU). RF-DETR needs its own environment -- it wants '
                         'transformers>=5, which the SigLIP backend does not '
                         'run on.')
    ap.add_argument('--topk', type=int, default=3)
    ap.add_argument('--include-judged', action='store_true',
                    help='also predict crops already ruled on (for measuring '
                         'the model against human verdicts -- still never '
                         'written back)')
    ap.add_argument('--refresh', action='store_true',
                    help='re-predict crops already in the output file')
    ap.add_argument('--verify-buckets', action='store_true',
                    help='check the bucket edges and exit')
    ap.add_argument('--status', default=STATUS_FILE,
                    help='where to publish progress for the dashboard')
    ap.add_argument('--watch', type=int, default=0, metavar='SECONDS',
                    help='keep going: sleep this long and pick up whatever '
                         'the sweep has written since. The live pool turns '
                         'over at ~2 crops/s, so a single pass leaves the '
                         'queue partly unguessed within the hour.')
    args = ap.parse_args()

    import torch
    from PIL import Image
    torch.set_num_threads(max(1, args.threads))
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if args.device == 'cpu' and args.model == SIGLIP_DEFAULT:
            # Measured on this box: so400m does 17.7 crops/s on the card and
            # well under 1 on a CPU, against a pool that turns over in an
            # hour. Silently taking three days per pass is not a default
            # anyone would choose, so say which knob changes it.
            print('no CUDA device -- the default model is far too slow on a '
                  f'CPU for a pool this size. Consider --model {SIGLIP_FAST}.',
                  flush=True)
    imagenet = args.model == 'imagenet'
    rfdetr = args.model in RFDETR_SIZES
    # Only SigLIP puts images and text in one space, so only SigLIP can leave
    # behind the vectors free-text search runs on. Naming the capability, not
    # the backend, so the next one added has to answer the question.
    embeds = not imagenet and not rfdetr
    BACKEND = backend_of(args.model)

    if rfdetr:
        import rfdetr as _rf
        from rfdetr.assets.coco_classes import COCO_CLASSES
        # The bucket rule is an assertion about the class list, so check it --
        # a release that renamed or dropped a class would otherwise re-bucket
        # every guess this tool ever writes, in silence.
        # COCO_CLASSES is a {class_id: name} dict, not a list -- membership
        # has to be tested against the NAMES or every class reads as missing.
        known = set(COCO_CLASSES.values())
        missing = [c for c in ('dog',) + COCO_ANIMALS if c not in known]
        if missing:
            raise SystemExit('COCO class names are not what the buckets '
                             'assume, missing: ' + ', '.join(missing))
        if args.verify_buckets:
            n = {}
            for c in COCO_CLASSES.values():
                n[coco_bucket(c)] = n.get(coco_bucket(c), 0) + 1
            print(f'{len(COCO_CLASSES)} COCO classes bucketed: {n}')
            print('  animals: ' + ', '.join(sorted(
                c for c in known if coco_bucket(c) == 'animal')))
            return 0
        model = getattr(_rf, RFDETR_SIZES[args.model])()
        # rfdetr owns its own placement; there is no .to() to fall back with,
        # so the shared _place() helper does not apply here.
        model_id = args.model
        cats = COCO_CLASSES
    elif imagenet:
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
        model = efficientnet_v2_s(weights=weights).eval()
        model, load_dev = _place(model, args.device, args.no_cpu_fallback)
        args.device = load_dev
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
        model = AutoModel.from_pretrained(args.model).eval()
        # The GPU may already be full when this starts -- a training run takes
        # the card and holds it for hours. Placing the model is where that is
        # discovered, so it is where the decision to step aside belongs.
        model, load_dev = _place(model, args.device, args.no_cpu_fallback)
        args.device = load_dev
        model_id = args.model
        # the text side is fixed, so encode it once for the whole run
        with torch.no_grad():
            tok = proc(text=[p for _, _, p in PROMPTS], padding='max_length',
                       max_length=64, return_tensors='pt').to(args.device)
            tfeat = model.get_text_features(**tok)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    # Which card this run is actually on. It can change mid-run: something
    # else may take the GPU after we have started, and the sensible answer to
    # that is to keep guessing on the CPU rather than to stop.
    DEV = {'device': args.device, 'fell_back': False}
    VEC = {'last': None}

    def _to_cpu():
        """Move the model, and the cached text features with it, to the CPU."""
        nonlocal model, tfeat
        DEV['device'] = 'cpu'
        DEV['fell_back'] = True
        if rfdetr:
            # rfdetr wraps its own module and places it itself; there is no
            # .to() here to move, so stepping aside is not on offer. Say so
            # rather than raise an AttributeError inside an OOM handler.
            raise RuntimeError(
                'the GPU is full and the RF-DETR backend cannot fall back to '
                'the CPU -- stop whatever has the card, or run the SigLIP '
                'backend instead')
        model = model.to('cpu')
        if embeds:
            with torch.no_grad():
                tok = proc(text=[q[2] for q in PROMPTS], padding='max_length',
                           max_length=64, return_tensors='pt')
                tfeat = model.get_text_features(**tok)
                tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def score(ims):
        """[(bucket_mass, ...)] for a batch, on whichever device still has room.

        A training run taking the card mid-pass used to end this process. It is
        the least important thing on the GPU, so it steps aside instead: the
        queue keeps being guessed, just slower.
        """
        try:
            return _score_on(ims)
        except Exception as e:
            if args.no_cpu_fallback or DEV['fell_back'] or not _oom(e):
                raise
            print('\nCUDA is out of memory -- something else on this box has '
                  'the card. Falling back to the CPU and carrying on.',
                  flush=True)
            _to_cpu()
            return _score_on(ims)

    def _rf_prep(im):
        """Upscale, then centre on a larger canvas.

        Both halves earned their place by measurement. Upscaling gives the
        detector pixels to work with -- the median crop's long side is 35px --
        and padding gives the object a share of the frame closer to what a
        scene-trained detector expects. Neither alone: raw crops score 27.5%
        on confirmed dogs, upscaled-only 36.7%, padded-only 49.2%, both 55.8%.
        """
        w, h = im.size
        if max(w, h) < RFDETR_UPSCALE:
            s = RFDETR_UPSCALE / max(w, h)
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))),
                           Image.BICUBIC)
        w, h = im.size
        canvas = Image.new('RGB', (int(w * RFDETR_PAD), int(h * RFDETR_PAD)),
                           (114, 114, 114))
        canvas.paste(im, ((canvas.width - w) // 2, (canvas.height - h) // 2))
        return canvas

    def _score_on(ims):
        if rfdetr:
            out = []
            for im in ims:
                det = model.predict(_rf_prep(im), threshold=RFDETR_THRESHOLD)
                # A bucket's score is the BEST detection in it, not the sum:
                # three low-confidence cows are not evidence of one cow, and
                # summing would let a crowded frame outvote a clear call.
                mass = {'dog': 0.0, 'animal': 0.0, 'object': 0.0}
                per = []
                for k in range(len(det)):
                    # .get, not []: cats is a dict keyed by COCO id and a
                    # class_id outside it must not take the pass down
                    nm = cats.get(int(det.class_id[k]))
                    if not nm:
                        continue
                    p = float(det.confidence[k])
                    b = coco_bucket(nm)
                    mass[b] = max(mass[b], p)
                    per.append((b, nm, p))
                out.append((mass, per))
            return out
        if imagenet:
            batch = torch.stack([tf(im) for im in ims]).to(DEV['device'])
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
        px = proc(images=ims, return_tensors='pt').to(DEV['device'])
        with torch.no_grad():
            ifeat = model.get_image_features(**px)
            ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
            # Kept, because it is already computed and it is the only thing
            # that makes an arbitrary typed word searchable later. Discarding
            # it means re-running the model per query; keeping it makes a
            # query a dot product.
            VEC['last'] = ifeat.detach().cpu().to(torch.float16).numpy()
            logits = ifeat @ tfeat.T * model.logit_scale.exp() + model.logit_bias
            # SigLIP scores each label independently (sigmoid, not softmax),
            # so normalise across the prompt table to get a share per bucket
            p = torch.sigmoid(logits).cpu()
            p = p / p.sum(1, keepdim=True).clamp(min=1e-9)
        out = []
        for row in p:
            tot = {'dog': 0.0, 'animal': 0.0, 'object': 0.0}
            for j, (bk, _, _) in enumerate(PROMPTS):
                tot[bk] += float(row[j])
            # per-prompt average, so a bucket cannot win on prompt count alone
            mass = {b: tot[b] / max(1, BUCKET_N[b]) for b in tot}
            out.append((mass, [(PROMPTS[j][0], PROMPTS[j][1], float(row[j]))
                               for j in range(len(PROMPTS))]))
        return out

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    print(f'model: {model_id}')

    started = time.time()
    passes = [0]
    vec_names, vec_rows = [], []

    def once(first):
        """One pass over whatever is unpredicted right now."""
        passes[0] += 1
        names = pool(REPO)
        pool_names = {n for n, _ in names}
        # --refresh means "redo them", which must apply to the FIRST pass
        # only: on a later --watch pass it would loop over the same crops
        # forever and never reach the new ones.
        skip = (set() if (args.refresh and first)
                else already_done(args.out, backend=BACKEND))
        # A prediction is not a vector. Skipping on the prediction alone left
        # every crop scored before the vectors existed permanently unsearchable
        # -- and since the same forward pass produces both, re-doing one is the
        # whole cost of getting the other.
        # Only a run that can actually produce a vector is allowed to re-do a
        # crop for the sake of one. The ImageNet and RF-DETR backends produce
        # none, so either would have found the whole pool 'owing', re-predicted
        # every crop, written no vectors, and found the same debt again on the
        # next --watch pass -- forever.
        owed = (set() if (args.no_vectors or not embeds)
                else pool_names - vectored_names(model_id=model_id))
        skip -= owed
        if not args.include_judged:
            skip |= judged_names(REPO)
        todo = sorted((n, d) for n, d in names if n not in skip)
        if args.limit:
            todo = todo[:args.limit]
        print(f'{len(names):,} crops in the pool, {len(todo):,} to predict '
              f'({len(skip):,} already judged or done'
              + (f'; {len(owed):,} owe a vector' if owed else '') + ')')
        if not todo:
            write_status(args.status, backend=BACKEND, running=bool(args.watch), model=model_id,
                         done=0, total=0, rate=0, started=started,
                         passes=passes[0], watch=args.watch, idle=True)
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
                if VEC.get('last') is not None and len(VEC['last']) == len(keep):
                    for j, nm in enumerate(keep):
                        vec_names.append(nm)
                        vec_rows.append(VEC['last'][j])
                    VEC['last'] = None
                for im in ims:
                    im.close()
                for j, nm in enumerate(keep):
                    mass, per_label = scored[j]
                    if not per_label or not any(mass.values()):
                        # A detector is allowed to find nothing, and that is
                        # an answer -- 'no guess yet' is already a filter on
                        # the page. Written anyway, with a bucket the reader
                        # does not recognise, so the crop counts as done and
                        # is not re-run on every pass forever.
                        fh.write(json.dumps({
                            'schema': SCHEMA, 'name': nm, 'bucket': 'none',
                            'p': 0.0, 'guess': 'nothing detected',
                            'unverified': True, 'source': 'model_suggestion',
                            'backend': BACKEND, 'model': model_id,
                            'ran_at': ran_at}) + '\n')
                        wrote += 1
                        continue
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
                        # which guesser said it, so two can disagree about the
                        # same crop and the page can filter by one of them
                        'backend': BACKEND,
                        'model': model_id,
                        'ran_at': ran_at,                    }) + '\n')
                    wrote += 1
                fh.flush()
                el = time.time() - t0
                write_status(args.status, backend=BACKEND, running=True, model=model_id,
                             done=wrote, total=len(todo),
                             rate=round(wrote / el, 2) if el else 0,
                             started=started, unreadable=unreadable,
                             passes=passes[0], watch=args.watch)
                if (i // args.batch) % 10 == 0:
                    print(f'  {wrote:,}/{len(todo):,}  '
                          f'{wrote / el if el else 0:.1f}/s', end='\r',
                          flush=True)
        el = time.time() - t0
        write_status(args.status, backend=BACKEND, running=bool(args.watch), model=model_id,
                     done=wrote, total=len(todo),
                     rate=round(wrote / el, 2) if el else 0,
                     started=started, unreadable=unreadable, passes=passes[0],
                     watch=args.watch, idle=bool(args.watch))
        print(f'\n{wrote:,} predictions -> {args.out} in {el:.0f}s '
              f'({wrote / el:.1f}/s)' if el else f'\n{wrote:,} predictions')
        if unreadable:
            print(f'  {unreadable:,} vanished or unreadable, skipped')
        if vec_rows and not args.no_vectors:
            try:
                # `live` is this pass's pool, so vectors for crops that have
                # since rotated out go with them
                total = save_vectors(vec_names, vec_rows, model_id,
                                     live=pool_names)
                print(f'  {len(vec_rows):,} crop vectors kept '
                      f'({total:,} searchable)')
            except Exception as e:
                print(f'  vectors not written: {type(e).__name__}: {e}')
            vec_names.clear(); vec_rows.clear()
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
    # ...unless a different live run owns the file; see _owner_alive
    if not _owner_alive(args.status):
        write_status(args.status, backend=BACKEND, running=False, model=model_id, done=0,
                     total=0, rate=0, started=started, passes=passes[0],
                     watch=args.watch, finished=True)
    print('These are suggestions for sorting the queue. Nothing reads this '
          'file except the review page filter.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
