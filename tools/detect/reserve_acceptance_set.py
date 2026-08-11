#!/usr/bin/env python3
"""Reserve flagged false positives as a permanent, never-trained acceptance set.

dogbin_v3 was accepted on a test it had partly trained on. The dashboard's
flagged false positives were folded into the dataset as extra negatives, and
eval_dogbin.py then scored that same flag directory: 360 of 1072 crops were
dataset members, 297 of them in train. Rejection at t=0.5 read 0.7740 when the
honest figure was 0.7032, two Wilson intervals that do not overlap.

The fix is not a smarter check at evaluation time -- it is holding a slice out
BEFORE the dataset is built, once, and writing the decision down. This tool
does that. The output is a small JSON id list that every later rebuild reads
and excludes, so the reservation survives dataset regeneration, renaming, and
whoever runs it next.

    python tools/detect/reserve_acceptance_set.py --crops data/harvest/v4/fp \\
        --clusters clusters.json --frac 0.30

Groups are the union of THREE relations, and a group is assigned whole:

  sequence         consecutive frames of one recording session
  near-duplicate   the embedding/dHash clusters from dedup_crops.py
  image_id         one photo can yield several detections

Splitting on image_id alone would put frame 7 of a sequence in the acceptance
set and frame 8 in train, which is the leak this file exists to prevent, one
level down.

Refuses to overwrite an existing reservation without --force: silently
re-drawing it would let a model be accepted on crops a previous run trained on.
"""

import argparse
import json
import os
import random
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(REPO, 'data', 'dogbin_acceptance_set.json')
NAME_RE = re.compile(r'^(?:flag_|pos_)?(?:le|un|no|nd)?_?(\d{6,})[_.]')


def image_id_of(fname):
    m = NAME_RE.match(os.path.basename(fname))
    return m.group(1) if m else None


def union_find_groups(keys, edges):
    p = {k: k for k in keys}

    def find(x):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    for a, b in edges:
        if a in p and b in p:
            ra, rb = find(a), find(b)
            if ra != rb:
                p[ra] = rb
    g = {}
    for k in keys:
        g.setdefault(find(k), []).append(k)
    return list(g.values())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--crops', required=True,
                    help='directory of harvested flagged crops (recursed)')
    ap.add_argument('--clusters',
                    help='clusters.json from dedup_crops.py, so a duplicate '
                         'pair cannot straddle the reservation boundary')
    ap.add_argument('--frac', type=float, default=0.30,
                    help='fraction of GROUPS to reserve')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--python', default=os.environ.get('DUCKDB_PYTHON')
                    or sys.executable,
                    help='interpreter with duckdb, for the sequence lookup')
    ap.add_argument('--force', action='store_true',
                    help='redraw even though a reservation already exists')
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        with open(args.out) as f:
            prev = json.load(f)
        raise SystemExit(
            f'{args.out} already reserves {len(prev.get("image_ids", []))} '
            f'image_ids (drawn {prev.get("created")}).\n'
            'Re-drawing would let a model be accepted on crops an earlier run '
            'trained on. Pass --force only if no model has been trained '
            'against the current reservation.')

    files = []
    for root, _, names in os.walk(args.crops):
        files += [os.path.join(root, n) for n in names
                  if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
    ids = {}
    for f in files:
        i = image_id_of(f)
        if i:
            ids.setdefault(i, []).append(f)
    if not ids:
        raise SystemExit(f'no parseable crops under {args.crops}')
    print(f'{len(files)} crops, {len(ids)} distinct image_ids')

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rebuild_crop_dataset import resolve_sequences
    seq = resolve_sequences(list(ids), REPO, args.python)
    print(f'sequences resolved: {len(seq)}/{len(ids)}')

    # edges: same sequence, then same duplicate cluster
    edges = []
    by_seq = {}
    for i in ids:
        s = seq.get(i)
        if s:
            by_seq.setdefault(s, []).append(i)
    for members in by_seq.values():
        for k in range(1, len(members)):
            edges.append((members[0], members[k]))
    n_seq_edges = len(edges)

    if args.clusters:
        with open(args.clusters) as f:
            cl = json.load(f)
        for g in cl.get('groups', []):
            gid = [image_id_of(p) for p in g]
            gid = [x for x in gid if x in ids]
            for k in range(1, len(gid)):
                edges.append((gid[0], gid[k]))
    print(f'edges: {n_seq_edges} sequence, {len(edges)-n_seq_edges} duplicate')

    groups = union_find_groups(list(ids), edges)
    groups.sort(key=lambda g: (-len(g), min(g)))       # deterministic
    rng = random.Random(args.seed)
    order = list(range(len(groups)))
    rng.shuffle(order)

    target = int(round(args.frac * len(ids)))
    held, held_ids = [], set()
    for gi in order:
        if len(held_ids) >= target:
            break
        held.append(gi)
        held_ids.update(groups[gi])

    out = {
        # WHEN THIS DRAW HAPPENED, not when the code was written. It was a
        # string literal, so every reservation this tool has ever produced
        # claims 2026-08-03 -- including the date the --force refusal above
        # prints back at whoever is deciding whether a reservation predates
        # the model it is about to accept. That is the one question this file
        # exists to let a human answer, and a constant cannot answer it.
        'created': time.strftime('%Y-%m-%d'),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'purpose': 'Permanent acceptance set for the dog-bin gate. These '
                   'image_ids must NEVER enter any training or validation '
                   'split. rebuild_crop_dataset.py --exclude-ids reads this '
                   'file; eval_dogbin.py scores against it.',
        'source': os.path.relpath(args.crops, REPO),
        'frac_requested': args.frac,
        'seed': args.seed,
        'groups_total': len(groups),
        'groups_reserved': len(held),
        'image_ids': sorted(held_ids),
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'\nreserved {len(held_ids)}/{len(ids)} image_ids '
          f'({len(held_ids)/len(ids)*100:.1f}%) in {len(held)} whole groups')
    print(f'trainable remainder: {len(ids)-len(held_ids)}')
    print(f'-> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
