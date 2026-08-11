#!/usr/bin/env python3
"""
Build an inscope-style worklist of ground-animal images that have a parquet row
but NO jpg on any drive, so ``backfill_missing.py`` can re-fetch their metadata
(and with it a FRESH thumb url) and download them.

Why this exists
---------------
``backfill_missing.py --download-only`` replays the ``thumb_original_url``
stored in the parquet. Mapillary thumb urls are signed and expire, and a row
whose url was never captured (or has since been blanked) stores NULL -- which
that path skips outright. Either way the jpg never lands. Feeding the same ids
back through the normal ``--inscope`` path re-queries the entity API per image,
so every download uses a url minted seconds earlier.

What it writes
--------------
``<out>/<Parent_Region>.parquet`` with the two columns the backfill's inscope
reader needs -- ``image_id`` and ``safe_region_id`` (the cell) -- sorted by cell
so the stream stays cell-contiguous (the backfill buffers per cell and flushes
on change; unsorted input would shred it into tiny parquets).

One row per (image_id, CELL), NOT per image_id
----------------------------------------------
An image on a 5-degree boundary is returned by the bbox query of both adjacent
cells and therefore has a ground_animals row, and a jpg slot, in each of them.
Both rows are missing work: the jpg is looked for under ``<cell>/
ground_animal_images/``, and ``prune_unrecoverable.py`` consumes this same
worklist per cell to decide which manifest rows to drop. De-duplicating
globally here would leave the second cell's row unfetched and unprunable
forever, so the duplication is deliberate. What must never happen is reporting
the row count as a count of images -- the summary below prints both.

Run it, then hand ``--out`` to the backfill as ``--inscope`` together with
``--no-skip-extracted`` (these ids ARE in the region's all_data parquets, so the
default already-extracted filter would drop every one of them).

    python tools/catalog/build_missing_worklist.py \
        --data-dirs /media/.../crucial/grid_runs /media/.../weasel/grid_runs \
        --image-dirs /media/.../lynx/grid_runs /media/.../bobcat/grid_runs \
        --out data/missing_worklist

READ-ONLY on all data drives: it only lists directories and reads the
``image_id``/``thumb_original_url`` columns of existing parquets.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import polars as pl


def cell_dirs(data_dirs, region=None):
    """{cell: [dirs holding its ground_animals parquets]} across data roots."""
    out = defaultdict(list)
    for root in data_dirs:
        pat = f'{region}_*' if region else '*'
        for d in glob.glob(os.path.join(root, pat)):
            if not os.path.isdir(d):
                continue
            if glob.glob(os.path.join(d, 'ground_animals_*.parquet')):
                out[os.path.basename(d)].append(d)
    return out


def jpgs_on_disk(image_dirs, cell):
    """Every image id already downloaded for ``cell``, across all image roots."""
    have = set()
    for root in image_dirs:
        d = os.path.join(root, cell, 'ground_animal_images')
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.name.endswith('.jpg'):
                        have.add(e.name[:-4])
        except OSError:
            pass
    return have


def missing_for_cell(dirs, have):
    """(missing ids, n_rows_with_url) for one cell.

    ``n_rows_with_url`` counts how many of the missing ones still carry a stored
    url -- purely diagnostic: a high count means --download-only would have
    worked, a low one means the url refresh below is doing the real work.
    """
    ids, with_url = set(), 0
    for d in dirs:
        for pq in glob.glob(os.path.join(d, 'ground_animals_*.parquet')):
            try:
                df = pl.read_parquet(
                    pq, columns=['image_id', 'thumb_original_url'])
            except Exception:
                continue
            for row in df.iter_rows(named=True):
                iid = str(row['image_id'])
                if iid in have or iid in ids:
                    continue
                ids.add(iid)
                if row['thumb_original_url']:
                    with_url += 1
    return ids, with_url


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dirs',
                   nargs='+',
                   required=True,
                   help='grid_runs roots holding the ground_animals parquets.')
    p.add_argument('--image-dirs',
                   nargs='+',
                   required=True,
                   help='EVERY grid_runs root that may hold jpgs. Missing one '
                   'here would re-download images you already have.')
    p.add_argument('--out',
                   default='data/missing_worklist',
                   help='Directory for the <Region>.parquet worklists.')
    p.add_argument('--region', help='Only this parent region (default: all).')
    p.add_argument('--min-cell',
                   type=int,
                   default=1,
                   help='Skip cells with fewer than N missing (default 1).')
    args = p.parse_args()

    cells = cell_dirs(args.data_dirs, args.region)
    if not cells:
        print('no cells found -- check --data-dirs / --region',
              file=sys.stderr)
        return 1
    print(f'scanning {len(cells):,} cells for images with no jpg on disk...')

    per_region = defaultdict(list)
    stats = defaultdict(lambda: [0, 0])  # region -> [rows, with_url]
    all_ids = set()
    for n, (cell, dirs) in enumerate(sorted(cells.items()), 1):
        have = jpgs_on_disk(args.image_dirs, cell)
        ids, with_url = missing_for_cell(dirs, have)
        if len(ids) < args.min_cell:
            continue
        region = cell.rsplit('_', 4)[0]
        per_region[region] += [(i, cell) for i in sorted(ids)]
        stats[region][0] += len(ids)
        stats[region][1] += with_url
        all_ids.update(ids)
        print(f'[{n}/{len(cells)}] {cell:<40} missing {len(ids):>7,}',
              flush=True)

    os.makedirs(args.out, exist_ok=True)
    total = 0
    region_ids = {}
    for region, rows in sorted(per_region.items()):
        rows.sort(key=lambda r: r[1])  # cell-contiguous, as required
        pl.DataFrame({
            'image_id': [r[0] for r in rows],
            'safe_region_id': [r[1] for r in rows]
        }).write_parquet(os.path.join(args.out, f'{region}.parquet'))
        total += len(rows)
        region_ids[region] = len({r[0] for r in rows})

    summary = {
        r: {
            # rows == (image_id, cell) pairs; images == distinct image_ids.
            # A boundary image has a row in each of the two cells it falls in.
            'rows': s[0],
            'images': region_ids.get(r, 0),
            'still_has_stored_url': s[1]
        }
        for r, s in stats.items()
    }
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, sort_keys=True)

    print(f'\n{"region":<32}{"rows":>10}{"images":>10}{"has stored url":>16}')
    for r, s in sorted(stats.items(), key=lambda x: -x[1][0]):
        print(f'{r:<32}{s[0]:>10,}{region_ids.get(r, 0):>10,}{s[1]:>16,}')
    print(f'{"TOTAL":<32}{total:>10,}{len(all_ids):>10,}')
    if total != len(all_ids):
        print(f'\n{total - len(all_ids):,} of the rows are the SECOND cell of '
              f'an image that straddles a 5-degree boundary. Both rows are '
              f'real work (a jpg per cell, a manifest row per cell); "images" '
              f'is the download count, "rows" is the work count.')
    print(f'\nworklists -> {args.out}/<Region>.parquet')
    print('feed to: backfill_missing.py --inscope '
          f'{args.out} --no-skip-extracted ...')
    return 0


if __name__ == '__main__':
    sys.exit(main())
