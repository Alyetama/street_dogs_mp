#!/usr/bin/env python3
"""
Make a typed word searchable over the review queue.

    python tools/detect/crop_search.py --add cat goat "plastic bag"
    python tools/detect/crop_search.py --seed        # the usual confusables
    python tools/detect/crop_search.py --query cat   # try it from the shell
    python tools/detect/crop_search.py               # what is searchable

WHY NOT YOLO-WORLD OR YOLOE. Both are open-vocabulary DETECTORS: they find and
localise objects inside a scene. The review queue is not scenes -- a detector
already cut every crop out, and what is left is a small picture of one thing.
Asking a detector to find the cat in a 200px picture of a cat re-solves a
solved problem, and it costs a full detection pass per crop for every new word
typed. YOLOE's prompt-free mode has the opposite problem for this job: it
returns ITS vocabulary, not the word you typed.

What the queue needs is retrieval, and the model already running over it is a
retrieval model. SigLIP 2 puts images and text in one space, so the crop
vectors triage_crops.py now keeps turn any word into a dot product: one text
encode, then a matmul over the pool. Measured on 600 crops before building
this -- the top hit for cow, goat, person and bird was right.

This script owns the text half. Encoding needs the model, which the dashboard
does not have, so vectors are computed here and cached; the dashboard only
ever multiplies.
"""

import argparse
import contextlib
import fcntl
import os
import sys

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'data', 'dashboard')
VEC_FILE = os.path.join(OUT, 'triage_vecs.npz')
TXT_FILE = os.path.join(OUT, 'search_terms.npz')
# A prompt, not a bare noun: SigLIP was trained on captions, and "a photo of a
# cat" sits closer to a photograph of one than "cat" does.
TEMPLATE = 'a photo of {}'
# What a dog detector actually trips over, so the common words are instant.
SEED = [
    'a cat', 'a cow', 'an ox', 'a goat', 'a sheep', 'a horse', 'a donkey',
    'a camel', 'a pig', 'a chicken', 'a bird', 'a monkey', 'a deer',
    'a fox', 'a wolf', 'a bear', 'a rat', 'a squirrel', 'a lizard',
    'a person walking', 'a child', 'a person sitting on the ground',
    'a motorcycle', 'a bicycle', 'a car', 'a rickshaw', 'a cart',
    'a plastic bag', 'a pile of rubbish', 'a rock', 'a bush', 'a tree',
    'a shadow on the road', 'a statue of an animal', 'a stuffed toy',
    'a road sign', 'a bollard', 'a fire hydrant', 'a puddle',
    'a dog lying down', 'a puppy', 'a dog on a leash',
]


@contextlib.contextmanager
def _locked(target):
    """Hold an exclusive lock beside `target` for the read-modify-write."""
    lock = target + '.lock'
    fh = open(lock, 'w')
    try:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()


def load_terms(path=None):
    """{term: vector} already encoded, and the model they belong to."""
    import numpy as np
    path = path or TXT_FILE
    try:
        d = np.load(path, allow_pickle=False)
        return ({str(t): d['vecs'][i] for i, t in enumerate(d['terms'])},
                str(d['model']))
    except Exception:
        return {}, ''


def crop_model(path=None):
    """Which model the crop vectors came from, or '' if there are none."""
    import numpy as np
    try:
        d = np.load(path or VEC_FILE, allow_pickle=False)
        return str(d['model'])
    except Exception:
        return ''


def add(terms, model_id=None, path=None, quiet=False):
    """Encode and cache these terms. Returns how many were encoded.

    Vectors are only ever compared with crop vectors from the same model, so
    a model change throws the cache away rather than silently mixing two
    spaces -- the numbers would still come out, and they would be meaningless.
    The WORDS survive that: they are re-encoded under the new model, because
    losing the vocabulary would leave the page's suggestion list empty every
    time somebody changed the guesser's model.

    Serialised on a lock file. The dashboard spawns one of these per unknown
    word typed, each does load - modify - save, and two overlapping runs lose
    whichever term was written first. Measured with two concurrent adds: 'a
    cow' vanished, and the caller's own 300-second retry guard then refused to
    ask for it again.
    """
    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor
    want = [t.strip() for t in terms if t and t.strip()]
    if not want:
        return 0
    mid = model_id or crop_model()
    if not mid:
        raise SystemExit('no crop vectors yet -- run triage_crops.py first')
    out = path or TXT_FILE
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with _locked(out):
        # read INSIDE the lock: another run may have written since we were
        # spawned, and the whole point is that its terms survive ours
        have, have_model = load_terms(path)
        if have_model and have_model != mid:
            want = sorted(set(want) | set(have))
            have = {}
        todo = [t for t in want if t not in have]
        if not todo:
            return 0
        proc = AutoProcessor.from_pretrained(mid)
        model = AutoModel.from_pretrained(mid).eval()
        with torch.no_grad():
            tok = proc(text=[TEMPLATE.format(t) for t in todo],
                       padding='max_length', max_length=64,
                       return_tensors='pt')
            v = model.get_text_features(**tok)
            v = v / v.norm(dim=-1, keepdim=True)
        v = v.cpu().to(torch.float16).numpy()
        for i, t in enumerate(todo):
            have[t] = v[i]
        keys = sorted(have)
        tmp = out + '.tmp.npz'
        np.savez(tmp, terms=np.array(keys),
                 vecs=np.stack([have[k] for k in keys]).astype('float16'),
                 model=np.array(mid))
        os.replace(tmp, out)
    if not quiet:
        for t in todo:
            print(f'  + {t}')
    return len(todo)


def search(term, k=24, path=None):
    """[(crop name, score)] for a cached term, best first."""
    import numpy as np
    terms, tmodel = load_terms(path)
    if term not in terms:
        return None
    try:
        d = np.load(VEC_FILE, allow_pickle=False)
    except Exception:
        return []
    if str(d['model']) != tmodel:
        return []
    sims = d['vecs'].astype('float32') @ terms[term].astype('float32')
    order = np.argsort(-sims)[:k]
    return [(str(d['names'][i]), float(sims[i])) for i in order]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--add', nargs='+', metavar='TERM')
    ap.add_argument('--seed', action='store_true',
                    help='encode the usual dog-confusables')
    ap.add_argument('--query', metavar='TERM')
    ap.add_argument('-k', type=int, default=12)
    a = ap.parse_args(argv)

    if a.seed:
        n = add(SEED)
        print(f'{n} new term(s); {len(load_terms()[0])} searchable')
        return 0
    if a.add:
        n = add(a.add)
        print(f'{n} new term(s); {len(load_terms()[0])} searchable')
        return 0
    if a.query:
        if a.query not in load_terms()[0]:
            add([a.query])
        hits = search(a.query, a.k)
        if hits is None:
            print('could not encode that term')
            return 1
        print(f'{len(hits)} best match(es) for {a.query!r}:')
        for nm, sc in hits:
            print(f'  {sc:.3f}  {nm}')
        return 0

    terms, tmodel = load_terms()
    cm = crop_model()
    print(f'crop vectors : {"none" if not cm else cm}')
    print(f'terms cached : {len(terms)}' + ('' if tmodel == cm or not terms
                                            else '  (STALE: different model)'))
    for t in sorted(terms):
        print(f'  {t}')
    if not terms:
        print('\nnothing searchable yet -- try --seed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
