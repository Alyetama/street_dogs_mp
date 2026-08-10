#!/usr/bin/env python3
"""Choose crops to review from the whole store, instead of whatever was passing.

The review page reads a ROLLING pool: the sweep writes at most 2 crops/s and
keeps the newest 3,000 by count. Against 1.24M positives that is 0.24% held at
any moment, and everything else is pruned unreviewed. It is a fine live
monitor and a poor way to build a training set -- what reaches a human is
decided by when they happened to look.

This picks instead. The store knows every detection's confidence, geometry and
cell, so a set can be selected on purpose:

  BY CONFIDENCE BAND. The gate's errors live where the detector was unsure, so
  --conf-min/--conf-max aims at a band. Stratified across it in deciles, so one
  narrow slice cannot dominate.

  BIG ENOUGH TO JUDGE. --min-px drops boxes no person can call, the same floor
  the review queue applies. Default 48.

  SPREAD OVER GROUND. One cell can hold thousands of frames of one street.
  --per-cell caps how many come from any single cell, and no image gives up
  more than one crop -- the review page serves one per image and retires the
  rest with it.

  NOTHING ALREADY SPOKEN FOR. Excludes every image_id already flagged, already
  kept, already in the crop dataset, and every id reserved into an acceptance
  set -- a reserved crop that came back through review would quietly rejoin
  the training data it was withheld from.

Crops are cut from the ORIGINAL jpg at native box resolution, the same way
harvest_flagged.py does, not from the 1280 letterbox the preview pool uses.

    python tools/detect/build_review_set.py --n 2000            # plan only
    python tools/detect/build_review_set.py --n 2000 --execute

Writes only into the review-set directory. READ-ONLY on the store and on the
grid_runs originals.
"""

import argparse
import collections
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

# The review page's own pattern: <ms>_<image_id>_<conf*100>.jpg. Matching it
# exactly is what lets a harvested crop use every existing mechanism -- the
# flag ledgers, /hq, the box editor, the country filter -- with no new code.
NAME = '{ms}_{iid}_{c2:03d}.jpg'
_ID_RE = re.compile(r'^\d{6,24}$')


def default_dir():
    """Inside the dashboard's static root, so the existing file server hands
    these out at /review_set/<name> with no new route -- the same reason the
    live preview pool lives where it does."""
    return os.path.join(REPO, 'data', 'dashboard', 'review_set')


def judged_ids(repo):
    """Every image_id already spoken for, from every ledger that decides one.

    Each ledger says how many ids it brought, and one that is not where this
    looks for it says so on stderr instead of quietly contributing nothing.
    The kept ledger was read from data/dashboard/seen.jsonl, a file that has
    never existed anywhere in this repo -- the dashboard banks kept crops in
    data/hard_negatives/reviewed.jsonl -- so the largest of the three ledgers,
    8,196 ids against 2,653 and 125, excluded nothing at all. The only symptom
    was a set that came up short: a picked-and-already-kept crop was written,
    counted in n_written, and then dropped at serve time by the review page's
    own judged-or-seen filter.
    """
    out = set()

    def note(rel, ids):
        p = os.path.join(*rel)
        if ids is None:
            print(f'WARNING: no ledger at {p}; it is excluding nothing',
                  file=sys.stderr)
            return
        print(f'  {len(ids):,} ids from {p}')
        out.update(ids)

    for rel in (('data', 'hard_negatives', 'labels.jsonl'),
                ('data', 'hard_positives', 'labels.jsonl'),
                ('data', 'hard_negatives', 'reviewed.jsonl')):
        ids = set()
        try:
            with open(os.path.join(repo, *rel)) as fh:
                for ln in fh:
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        continue
                    if isinstance(r, dict) and r.get('image_id'):
                        ids.add(str(r['image_id']))
        except OSError:
            ids = None
        note(rel, ids)
    # reserved acceptance sets: these ids are withheld from training on
    # purpose, and re-reviewing one is how it finds its way back in
    for rel in (('data', 'dogbin_acceptance_set.json'),
                ('data', 'leash_acceptance_set.json')):
        try:
            with open(os.path.join(repo, *rel)) as fh:
                ids = {str(i) for i in (json.load(fh).get('image_ids') or [])}
        except (OSError, ValueError):
            ids = None
        note(rel, ids)
    return out


def dataset_ids(root):
    """image_ids already in the crop dataset, from the filenames."""
    out = set()
    if not root or not os.path.isdir(root):
        return out
    for base, _, files in os.walk(root):
        for f in files:
            m = re.search(r'\d{6,}', f)
            if m:
                out.add(m.group(0))
    return out


def candidates(root, args, exclude):
    """[(image_id, det_idx, conf, cell, drive, side)] passing every filter."""
    import duckdb
    import store
    img = store._sql_src(store._store_globs(root, 'img'))
    det = store._sql_src(store._store_globs(root, 'det'))
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory}'")
    # One pass. The size floor and the confidence band are applied in SQL so
    # the exclusion set never has to hold a million rows.
    rows = con.execute(f"""
        SELECT CAST(d.image_id AS VARCHAR), d.det_idx, d.conf,
               i.cell, i.drive, least(d.x2 - d.x1, d.y2 - d.y1) AS side
        FROM {det} d
        JOIN {img} i ON i.image_id = d.image_id AND i.gen = d.gen
                    AND i.cell = d.cell
        WHERE d.gen = ? AND d.conf >= ? AND d.conf <= ?
          AND least(d.x2 - d.x1, d.y2 - d.y1) >= ?
    """, [f'{args.gen:04d}', args.conf_min, args.conf_max,
          args.min_px]).fetchall()
    con.close()
    return [r for r in rows if r[0] not in exclude]


def choose(cands, args, rng):
    """Stratify over confidence deciles, capped per cell and one per image.

    A flat sample of a band is dominated by wherever the sweep spent its time.
    Deciles keep the band's shape, and the per-cell cap keeps one long street
    from filling the set.

    One crop per image_id, because that is all the review page can serve: it
    emits a single crop per image and either verdict then retires the whole
    image. A second detection from the same frame is cut, counted in
    n_written, and never shown to anyone -- 39 of the 2,000 in the set on
    disk are exactly that.
    """
    by_dec = collections.defaultdict(list)
    span = max(1e-9, args.conf_max - args.conf_min)
    for c in cands:
        d = min(9, int((c[2] - args.conf_min) / span * 10))
        by_dec[d].append(c)
    picked, per_cell, taken = [], collections.Counter(), set()
    want_each = max(1, args.n // max(1, len(by_dec)))
    for d in sorted(by_dec):
        pool = by_dec[d]
        pool.sort(key=lambda r: (r[0], r[1]))     # deterministic before shuffle
        rng.shuffle(pool)
        took = 0
        for c in pool:
            if took >= want_each or len(picked) >= args.n:
                break
            if per_cell[c[3]] >= args.per_cell or c[0] in taken:
                continue
            per_cell[c[3]] += 1
            taken.add(c[0])
            picked.append(c)
            took += 1
    # top up from whatever is left if a decile ran dry
    if len(picked) < args.n:
        rest = [c for c in cands if c[0] not in taken]
        rest.sort(key=lambda r: (r[0], r[1]))
        rng.shuffle(rest)
        for c in rest:
            if len(picked) >= args.n:
                break
            if per_cell[c[3]] >= args.per_cell or c[0] in taken:
                continue
            per_cell[c[3]] += 1
            taken.add(c[0])
            picked.append(c)
    return picked


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=2000)
    ap.add_argument('--gen', type=int, default=1)
    ap.add_argument('--conf-min', type=float, default=0.10)
    ap.add_argument('--conf-max', type=float, default=0.90)
    ap.add_argument('--min-px', type=int, default=48,
                    help='shorter box side in ORIGINAL pixels; the review '
                         'queue applies the same floor')
    ap.add_argument('--per-cell', type=int, default=6)
    ap.add_argument('--out', default=None)
    ap.add_argument('--roots-file',
                    default=os.path.join(REPO, 'data', 'catalog_dirs.txt'))
    ap.add_argument('--dataset', default=os.environ.get('DOGBIN_DATASET', ''),
                    help='crop dataset whose ids to exclude')
    ap.add_argument('--memory', default='6GB')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()

    import store
    root = store.get_detect_root()
    out = args.out or default_dir()
    rng = random.Random(args.seed)

    exclude = judged_ids(REPO)
    # the dataset and the set's own directory are two more exclusions that go
    # quiet when they are not where they are meant to be, so they report the
    # same way the ledgers do
    if not args.dataset:
        print('  no crop dataset named (--dataset, or DOGBIN_DATASET), so '
              'none of its ids are excluded')
    elif not os.path.isdir(args.dataset):
        print(f'WARNING: no crop dataset at {args.dataset}; it is excluding '
              f'nothing', file=sys.stderr)
    else:
        ds = dataset_ids(args.dataset)
        print(f'  {len(ds):,} ids from the crop dataset')
        exclude |= ds
    # already harvested into this set on an earlier run
    harvested = set()
    if os.path.isdir(out):
        for f in os.listdir(out):
            m = re.match(r'^\d{10,}_(\d{6,})_\d{3}\.jpg$', f)
            if m:
                harvested.add(m.group(1))
    print(f'  {len(harvested):,} ids already harvested into the set')
    exclude |= harvested
    # distinct, not the sum of the lines above: a kept crop is usually in a
    # flag ledger too
    print(f'{len(exclude):,} distinct image_ids already spoken for (judged, '
          f'kept, in the dataset, reserved, or already harvested)')

    cands = candidates(root, args, exclude)
    print(f'{len(cands):,} detections pass conf {args.conf_min}-'
          f'{args.conf_max} and >= {args.min_px}px, on unjudged images')
    if not cands:
        print('nothing to choose from')
        return 1
    picked = choose(cands, args, rng)
    print(f'{len(picked):,} chosen, <= {args.per_cell} per cell, '
          f'stratified over 10 confidence deciles')
    dec = collections.Counter(
        min(9, int((c[2] - args.conf_min)
                   / max(1e-9, args.conf_max - args.conf_min) * 10))
        for c in picked)
    print('  by decile: ' + ' '.join(f'{d}:{dec[d]}' for d in sorted(dec)))
    print(f'  cells: {len({c[3] for c in picked}):,}')

    if not args.execute:
        print('\nnothing written. re-run with --execute')
        return 0

    # PIL, not cv2: this tool needs duckdb AND an image decoder in one
    # interpreter, and no env on this machine has cv2 and duckdb together.
    # PIL is in the one that has duckdb, and a crop-and-save needs nothing cv2
    # offers over it.
    from PIL import Image
    from harvest_flagged import roots_by_drive
    try:
        roots = [ln.strip() for ln in open(args.roots_file)
                 if ln.strip() and not ln.startswith('#')]
    except OSError:
        raise SystemExit(f'no roots file at {args.roots_file}')
    by_drive = roots_by_drive(roots)
    os.makedirs(out, exist_ok=True)

    # boxes for the chosen detections, in original pixels. Pinned to the same
    # gen the candidates were drawn from: this dict is keyed on
    # (image_id, det_idx), which a second generation's row for the same
    # detection would overwrite, and the crop would then be cut at a box the
    # chosen conf never belonged to.
    import duckdb
    det = store._sql_src(store._store_globs(root, 'det'))
    ids = ','.join("'%s'" % i for i in sorted({c[0] for c in picked}))
    con = duckdb.connect()
    box = {(str(i), int(d)): (a, b, cc, e) for i, d, a, b, cc, e in con.execute(
        f"SELECT CAST(image_id AS VARCHAR), det_idx, x1, y1, x2, y2 "
        f"FROM {det} WHERE gen = ? AND CAST(image_id AS VARCHAR) IN ({ids})",
        [f'{args.gen:04d}']).fetchall()}
    con.close()

    ms = int(time.time() * 1000)
    wrote, missing, unreadable = 0, 0, 0
    for iid, di, conf, cell, drive, side in picked:
        b = box.get((iid, int(di)))
        r = by_drive.get(drive)
        if not b or not r:
            missing += 1
            continue
        src = os.path.join(r, cell, 'ground_animal_images', f'{iid}.jpg')
        if not os.path.exists(src):
            missing += 1
            continue
        try:
            im = Image.open(src)
            im.load()
            if im.mode != 'RGB':
                im = im.convert('RGB')
        except Exception:
            unreadable += 1
            continue
        w, h = im.size
        x1, y1, x2, y2 = (int(v) for v in b)
        # the same 12% margin the live preview writer uses, so a harvested
        # crop and a pool crop of the same box frame the animal alike
        pad = int(0.12 * max(x2 - x1, y2 - y1)) + 4
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 - x1 < 8 or y2 - y1 < 8:
            missing += 1
            continue
        name = NAME.format(ms=ms + wrote, iid=iid, c2=int(round(conf * 100)))
        im.crop((x1, y1, x2, y2)).save(os.path.join(out, name), quality=92)
        wrote += 1
        if wrote % 200 == 0:
            print(f'  {wrote:,}/{len(picked):,}', end='\r')

    man = {'built_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'gen': args.gen,
           'conf_min': args.conf_min, 'conf_max': args.conf_max,
           'min_px': args.min_px, 'per_cell': args.per_cell, 'n_asked': args.n,
           'n_written': wrote, 'candidates': len(cands),
           'excluded_ids': len(exclude), 'seed': args.seed}
    with open(os.path.join(out, 'review_set.json'), 'w') as fh:
        json.dump(man, fh, indent=1)
    print(f'\nwrote {wrote:,} crops to {out}')
    if missing or unreadable:
        print(f'  skipped {missing:,} with no original on disk, '
              f'{unreadable:,} unreadable')
    print('Point the dashboard at it with review_extra_dir to review them.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
