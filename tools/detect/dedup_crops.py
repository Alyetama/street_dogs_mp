#!/usr/bin/env python3
"""Find visually near-duplicate crops using image embeddings, not just hashes.

The crop datasets are cut from Mapillary street imagery, where the same animal
is photographed repeatedly as the camera drives past. Three mechanisms already
try to stop those duplicates crossing the train/val boundary, and each has a
blind spot:

  sequence id   groups frames from one recording session, but a dog can be
                captured by two different sequences (two passes, two cameras),
                and one sequence can run 87 minutes across a whole city.
  dHash         a 64-bit structural hash of a 9x8 grayscale thumbnail. Catches
                near-identical framings. Blind to the same subject at a
                different distance, angle, or crop boundary -- which is what
                consecutive frames with camera motion actually look like.
  size floor    unrelated to duplication.

This adds the missing one: a real image embedding. Crops are encoded with
ImageNet-pretrained EfficientNet-V2-S (penultimate layer, 1280-d), L2
normalised, and compared by cosine similarity.

Deliberately NOT the gate model's own features: dogbin is trained to collapse
everything except dogness, so two different dogs land in the same place. A
general-purpose encoder keeps the visual content that makes duplicates
duplicates.

    # calibrate first -- look at real pairs before trusting a threshold
    python tools/detect/dedup_crops.py calibrate <dir> [<dir> ...] --out sheets/

    # then cluster
    python tools/detect/dedup_crops.py cluster <dir> [<dir> ...] \
        --threshold 0.94 --out clusters.json

Clusters are the union of the embedding graph and the dHash graph, so a pair
caught by either mechanism is caught. Read-only on the crops.
"""

import argparse
import glob
import json
import os
import re
import sys

BATCH = 64
IMGSZ = 384


def crop_paths(dirs):
    out = []
    for d in dirs:
        if os.path.isfile(d):
            out.append(d)
            continue
        for ext in ('jpg', 'jpeg', 'png', 'webp'):
            out += glob.glob(os.path.join(d, '**', f'*.{ext}'), recursive=True)
    return sorted(set(out))


def embed(paths, device='cuda', batch=BATCH):
    """(paths_ok, NxD float32 L2-normalised embeddings)."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

    # 'auto' exists for the caller that is not a person: the dataset build
    # runs this from the dashboard's build lane, where a training job may
    # already own the GPU and a hard 'cuda' would be a crash on the one box
    # that has no GPU at all. A few thousand crops on CPU is minutes, and a
    # build is already the slow path.
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    w = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=w)
    model.classifier = torch.nn.Identity()
    model.eval().to(device)
    tf = w.transforms()

    ok, vecs = [], []
    for i in range(0, len(paths), batch):
        chunk, imgs = paths[i:i + batch], []
        for p in chunk:
            try:
                imgs.append((p, tf(Image.open(p).convert('RGB'))))
            except Exception:
                pass                      # unreadable: reported by the caller
        if not imgs:
            continue
        x = torch.stack([t for _, t in imgs]).to(device)
        with torch.no_grad():
            f = model(x).float()
        f = torch.nn.functional.normalize(f, dim=1)
        vecs.append(f.cpu().numpy())
        ok += [p for p, _ in imgs]
        if (i // batch) % 20 == 0:
            print(f'  embedded {len(ok):,}/{len(paths):,}', file=sys.stderr)
    if not vecs:
        return [], np.zeros((0, 1280), 'float32')
    return ok, np.concatenate(vecs).astype('float32')


def dhash(path, size=8):
    """64-bit difference hash -- same definition rebuild_crop_dataset.py uses."""
    from PIL import Image
    try:
        im = Image.open(path).convert('L').resize((size + 1, size),
                                                  Image.BILINEAR)
    except Exception:
        return None
    px = list(im.getdata())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | (1 if row[c] < row[c + 1] else 0)
    return bits


def pairs_above(emb, thr, block=2048):
    """[(i, j, sim)] with i<j and cosine >= thr, computed blockwise.

    Blockwise because the full matrix is O(n^2): at 40k crops that is 6.4 GB
    of float32, and this runs while a 32.5M-image sweep owns the machine.
    """
    import numpy as np
    n = len(emb)
    out = []
    for a in range(0, n, block):
        A = emb[a:a + block]
        for b in range(a, n, block):
            B = emb[b:b + block]
            S = A @ B.T
            ii, jj = np.where(S >= thr)
            for i, j in zip(ii, jj):
                gi, gj = a + int(i), b + int(j)
                if gi < gj:
                    out.append((gi, gj, float(S[i, j])))
    return out


def union_find(n, edges):
    p = list(range(n))

    def find(x):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            p[ri] = rj
    g = {}
    for i in range(n):
        g.setdefault(find(i), []).append(i)
    return list(g.values())


def cmd_calibrate(args):
    """Write contact sheets of real pairs per similarity band.

    A threshold picked from a number alone is a guess. These sheets exist so
    the operating point is chosen by looking at the pairs it would collapse:
    too low and two different dogs get merged, too high and the duplicates
    this tool exists to remove survive.
    """
    from PIL import Image
    import numpy as np
    paths = crop_paths(args.dirs)
    print(f'{len(paths):,} crops', file=sys.stderr)
    ok, emb = embed(paths, args.device)
    bands = [(0.99, 1.01), (0.97, 0.99), (0.95, 0.97),
             (0.93, 0.95), (0.90, 0.93), (0.85, 0.90)]
    allp = pairs_above(emb, 0.85)
    print(f'{len(allp):,} pairs at cosine >= 0.85', file=sys.stderr)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)
    summary = []
    for lo, hi in bands:
        sel = [p for p in allp if lo <= p[2] < hi]
        summary.append({'band': f'{lo:.2f}-{hi:.2f}', 'pairs': len(sel)})
        if not sel:
            continue
        take = [sel[i] for i in rng.choice(len(sel),
                                           min(args.samples, len(sel)),
                                           replace=False)]
        cell = 200
        sheet = Image.new('RGB', (cell * 2 + 30, cell * len(take)), 'white')
        for r, (i, j, s) in enumerate(take):
            for c, idx in enumerate((i, j)):
                try:
                    im = Image.open(ok[idx]).convert('RGB')
                    im.thumbnail((cell, cell))
                    sheet.paste(im, (c * (cell + 30), r * cell))
                except Exception:
                    pass
        f = os.path.join(args.out, f'band_{lo:.2f}_{hi:.2f}.jpg')
        sheet.save(f, quality=88)
        print(f'  {lo:.2f}-{hi:.2f}: {len(sel):>6} pairs -> {f}',
              file=sys.stderr)
    print(json.dumps({'total_crops': len(ok), 'bands': summary}, indent=1))
    return 0


def cmd_cluster(args):
    paths = crop_paths(args.dirs)
    print(f'{len(paths):,} crops', file=sys.stderr)
    ok, emb = embed(paths, args.device)
    unreadable = sorted(set(paths) - set(ok))
    idx = {p: i for i, p in enumerate(ok)}

    ep = pairs_above(emb, args.threshold)
    edges = [(i, j) for i, j, _ in ep]
    print(f'  embedding pairs >= {args.threshold}: {len(edges):,}',
          file=sys.stderr)

    # dHash, as an independent second opinion -- it catches exact re-encodes
    # whose embeddings drift, and costs nothing next to the forward pass.
    hedges = 0
    if args.hamming >= 0:
        buckets = {}
        for p in ok:
            h = dhash(p)
            if h is not None:
                buckets.setdefault(h, []).append(idx[p])
        for v in buckets.values():
            for k in range(1, len(v)):
                edges.append((v[0], v[k]))
                hedges += 1
        print(f'  dHash exact-match pairs: {hedges:,}', file=sys.stderr)

    groups = union_find(len(ok), edges)
    dup = [g for g in groups if len(g) > 1]
    out = {
        'threshold': args.threshold,
        'crops': len(ok),
        'unreadable': unreadable,
        'clusters': len(groups),
        'multi_clusters': len(dup),
        'removable': sum(len(g) - 1 for g in dup),
        'groups': [[ok[i] for i in g] for g in sorted(dup, key=len,
                                                      reverse=True)],
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"{out['crops']:,} crops -> {out['clusters']:,} clusters; "
          f"{out['removable']:,} removable duplicates in "
          f"{out['multi_clusters']:,} groups -> {args.out}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    c = sub.add_parser('calibrate', help='contact sheets per similarity band')
    c.add_argument('dirs', nargs='+')
    c.add_argument('--out', default='dedup_sheets')
    c.add_argument('--samples', type=int, default=8)
    c.add_argument('--device', default='cuda')
    c.set_defaults(func=cmd_calibrate)

    k = sub.add_parser('cluster', help='write near-duplicate clusters')
    k.add_argument('dirs', nargs='+')
    k.add_argument('--threshold', type=float, default=0.94,
                   help='cosine similarity at which two crops are the same '
                        'subject. Calibrate before changing it.')
    k.add_argument('--hamming', type=int, default=0,
                   help='also join exact dHash matches (0 = exact only, '
                        '-1 = skip dHash entirely)')
    k.add_argument('--out', default='clusters.json')
    k.add_argument('--device', default='cuda')
    k.set_defaults(func=cmd_cluster)

    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
