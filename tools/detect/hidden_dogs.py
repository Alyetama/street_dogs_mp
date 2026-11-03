#!/usr/bin/env python3
"""Frames where an independent model sees a dog nobody labelled.

THE PROBLEM. A confirmed-positive addition is one box a person agreed with,
and the detector build writes it as a frame holding exactly that one dog -- so
every other dog in the frame is taught as "not a dog". A background from the
negatives builder is a whole frame taught as "no dog at all". Measured
2026-10-14 with the stock COCO yolo26x on native-resolution tiles (class dog,
conf >= 0.5): 437 of the 6,312 additions in dogdet_20260923_0cf56d (6.9%) and
62 of its 1,947 added backgrounds (3.2%) hold a dog box that overlaps no
label -- and, on a background, not the object the reviewer rejected. About
seven in eight of a sample of the additions' flags were real dogs by eye,
many of them a second dog beside the confirmed one.

WHAT THIS IS, AND IS NOT. The store is model output, tier `unverified`. It is
never a label: nothing here is written into a human store, and its boxes are
never trained on. It only keeps the builds from teaching a silence that is in
doubt: build_dogdet_v3.py leaves a flagged addition out, and
build_detector_negatives.py never makes a flagged frame a background. The
fix for a flagged frame is a person boxing every dog in it on the Annotate
page -- `batch` writes the label batch -- and a whole-frame label supersedes
the flag, because the build already prefers a frame label to an addition.

REVERSIBLE. Both builds read the store by default, record in their manifest
what it made them skip, and take --no-hidden-dogs to ignore it. Writing the
store keeps the file it replaces under data/hidden_dogs/superseded/.

    python tools/detect/hidden_dogs.py import --flags <json> \\
        --sizes <jsonl> --role addition --source '<what produced it>'
    python tools/detect/hidden_dogs.py batch --name hidden_dogs_001
"""

import argparse
import json
import os
import random
import shutil
import sys
import time

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.path.join(REPO, 'data', 'hidden_dogs', 'flagged.json')
BATCHES = os.path.join(REPO, 'data', 'label_batches')
TIER = 'unverified'
ROLES = ('addition', 'background')


def read(path=STORE):
    """{image_id: record} for every flagged frame. A missing store is none.

    A store that is there but cannot be read is an error, not an empty set: a
    build that silently skipped nothing would teach the silence this store
    exists to keep out, and say nothing about it.
    """
    try:
        fh = open(path)
    except FileNotFoundError:
        return {}
    with fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or doc.get('tier') != TIER:
        raise ValueError(
            f'{path} is not a hidden-dog store (tier '
            f'{doc.get("tier") if isinstance(doc, dict) else "?"})')
    frames = doc.get('frames')
    if not isinstance(frames, dict):
        raise ValueError(f'{path} has no frames')
    return {str(k): v for k, v in frames.items()}


def screen(items, flagged, key=lambda x: x):
    """(kept, skipped): `items` split by whether the store flags the image id
    `key` reads off each one."""
    kept, skipped = [], []
    for x in items:
        (skipped if str(key(x)) in flagged else kept).append(x)
    return kept, skipped


def _write(doc, path):
    """Replace the store, keeping the file it replaces."""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(path):
        keep = os.path.join(d, 'superseded')
        os.makedirs(keep, exist_ok=True)
        shutil.copy2(
            path,
            os.path.join(
                keep,
                os.path.basename(path)[:-len('.json')] +
                time.strftime('.%Y%m%d-%H%M%S') + '.json'))
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def import_flags(flags, sizes, role, source, path=STORE, now=None):
    """Add flagged frames to the store; returns how many it now holds.

    `flags` is [[image_id, [[x1, y1, x2, y2, conf, ...], ...]], ...] in the
    frame's own pixels, `sizes` {image_id: (w, h)} for those pixels. A frame
    imported again is replaced, so a re-run of the same audit is idempotent.
    """
    if role not in ROLES:
        raise ValueError(f'role must be one of {ROLES}, not {role!r}')
    if not source:
        raise ValueError('a flag says where it came from')
    frames = read(path)
    at = int(now if now is not None else time.time())
    for iid, boxes in flags:
        iid = str(iid)
        if iid not in sizes:
            raise ValueError(f'{iid}: no frame size to read its boxes against')
        w, h = sizes[iid]
        frames[iid] = {
            'role':
            role,
            'boxes': [[round(float(v), 1)
                       for v in b[:4]] + [round(float(b[4]), 4)]
                      for b in boxes],
            'w':
            int(w),
            'h':
            int(h),
            'source':
            source,
            'at':
            at,
        }
    _write({'tier': TIER, 'schema': 1, 'frames': frames}, path)
    return len(frames)


def _frame_meta(ids):
    """{image_id: (n_det, w, h, cell, drive)}, {image_id: sequence}, roots --
    the detector build's own readers, so a batch finds the same jpg it would."""
    import duckdb
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import build_dogdet_v3 as v3
    con = duckdb.connect()
    meta, _ = v3.frame_meta(ids, con)
    return meta, v3.resolve_sequences(ids, con), v3._grid_roots()


def plan_batch(flagged, labelled, meta, seqs, roots, role=None, seed=0):
    """(rows, seeds) for a label batch of the flagged frames nobody has
    labelled whole yet. The seed is the flag's most confident box: somewhere
    to look first on the Annotate page, never a label."""
    rows, seeds = [], {}
    for iid in sorted(flagged):
        rec = flagged[iid]
        if role and rec.get('role') != role:
            continue
        if iid in labelled:
            continue
        m = meta.get(iid)
        if not m or m[4] not in roots or not m[3]:
            continue
        jpg = os.path.join(roots[m[4]], m[3], 'ground_animal_images',
                           f'{iid}.jpg')
        if not os.path.isfile(jpg):
            continue
        best = max(rec['boxes'], key=lambda b: b[4])
        rows.append({
            'image_id':
            iid,
            'seq':
            seqs.get(iid),
            'w':
            rec['w'],
            'h':
            rec['h'],
            'cell':
            m[3],
            'drive':
            m[4],
            'jpg':
            jpg,
            'role':
            rec['role'],
            'why': ('an independent model sees a dog nobody labelled ('
                    f'{len(rec["boxes"])} box(es), {TIER})'),
        })
        seeds[iid] = {
            'seed_id': iid,
            'box': best[:4],
            'w': rec['w'],
            'h': rec['h'],
            'tier': TIER,
        }
    random.Random(seed).shuffle(rows)
    return rows, seeds


def write_batch(name, rows, seeds, summary, batches=BATCHES):
    """A new batch directory. An existing one is never written into: a batch
    somebody has started labelling is theirs."""
    d = os.path.join(batches, name)
    os.makedirs(d)  # FileExistsError, never exist_ok
    with open(os.path.join(d, 'manifest.jsonl'), 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    with open(os.path.join(d, 'seeds.json'), 'w') as fh:
        json.dump(seeds, fh)
    with open(os.path.join(d, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    im = sub.add_parser('import', help='add flagged frames to the store')
    im.add_argument('--flags',
                    required=True,
                    help='JSON list of [image_id, [[x1,y1,x2,y2,conf], ...]]')
    im.add_argument('--sizes',
                    required=True,
                    help='JSONL with image_id, W, H for the same frames')
    im.add_argument('--role', required=True, choices=ROLES)
    im.add_argument('--source',
                    required=True,
                    help='the model, threshold and rule that flagged them')
    im.add_argument('--store', default=STORE)
    ba = sub.add_parser('batch', help='a label batch of the flagged frames')
    ba.add_argument('--name', required=True)
    ba.add_argument('--role', choices=ROLES, default=None)
    ba.add_argument('--seed', type=int, default=0)
    ba.add_argument('--store', default=STORE)
    a = ap.parse_args(argv)

    if a.cmd == 'import':
        with open(a.flags) as fh:
            flags = json.load(fh)
        sizes = {}
        with open(a.sizes) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    sizes[str(r['image_id'])] = (r['W'], r['H'])
        n = import_flags(flags, sizes, a.role, a.source, a.store)
        print(f'{len(flags)} {a.role} frame(s) imported; the store holds {n}')
        return 0

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import frame_labels
    flagged = read(a.store)
    labelled = frame_labels.read()
    want = sorted(i for i in flagged
                  if not a.role or flagged[i].get('role') == a.role)
    meta, seqs, roots = _frame_meta(want)
    rows, seeds = plan_batch(flagged, labelled, meta, seqs, roots, a.role,
                             a.seed)
    summary = {
        'kind':
        'hidden_dogs',
        'batch':
        a.name,
        'planned_at':
        int(time.time()),
        'asked':
        len(want),
        'picked':
        len(rows),
        'already_labelled':
        sum(1 for i in want if i in labelled),
        'role':
        a.role or 'all',
        'source':
        'tools/detect/hidden_dogs.py batch',
        'seed_tier':
        TIER,
        'command': ['tools/detect/hidden_dogs.py'] +
        list(argv if argv is not None else sys.argv[1:]),
    }
    d = write_batch(a.name, rows, seeds, summary)
    print(f'{len(rows)} of {len(want)} flagged frame(s) -> {d}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
