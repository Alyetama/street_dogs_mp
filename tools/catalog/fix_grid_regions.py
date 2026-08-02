#!/usr/bin/env python3
"""
Apply the region-audit corrections: rename mis-filed cells on every data root
and fix the grid CSV.

``original_global_grid_5deg.csv`` labels regions with coarse lat/lon boxes, and
several are drawn wrong -- the ``Africa`` box reaches lon 55 / lat 40 and is
tested before ``Middle East`` (12 cells, lon 55..65), so Kuwait, Riyadh, Baghdad
and Doha are all filed as Africa. ``audit_grid_regions.py`` finds these by
intersecting each cell with Natural Earth polygons; this applies its verdict.

Only ``MISASSIGNED`` rows are acted on -- cells whose land is on a different
CONTINENT than the label implies. Rows marked ``taxonomy`` (right continent,
different sub-label, e.g. European Russia as Europe vs Russia & North Asia) and
``straddles`` (no single label correct at 5 degrees) are deliberately left
alone: they are conventions to choose, not errors to fix.

The cell name is embedded in more than the directory::

    <root>/<cell>/all_data_<cell>_NNN.parquet
    <root>/<cell>/ground_animals_<cell>_NNN.parquet
    <root>/<cell>/validated_images_<cell>.txt
    <root>/<cell>/<cell>_tiles.png

so every filename containing the old cell name is renamed too. Missing that
would leave `all_data_Africa_45_25_50_30_000.parquet` sitting inside
`Middle_East_45_25_50_30/`, invisible to every tool that globs
``all_data_{cell}_*.parquet``.

Everything is ``os.rename`` within a single filesystem: atomic, no copy, no
window where data exists in neither place. A JSON journal of every rename is
written so the whole migration can be reversed with --undo.

Dry run by default.

    python tools/catalog/fix_grid_regions.py --roots <grid_runs> ...      # plan
    python tools/catalog/fix_grid_regions.py --roots <grid_runs> ... --execute
    python tools/catalog/fix_grid_regions.py --undo runs/region_fix_<ts>.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd


def safe_region(name):
    """Region name as it appears in a cell directory name."""
    return name.replace('&', 'and').replace(' ', '_')


def target_cell(old_cell, new_region):
    """``Africa_45_25_50_30`` + ``Middle East`` -> ``Middle_East_45_25_50_30``.

    The bbox is the last four underscore-separated fields; everything before is
    the (possibly underscored) region name, so split from the right.
    """
    bbox = old_cell.rsplit('_', 4)[1:]
    return safe_region(new_region) + '_' + '_'.join(bbox)


def plan(audit_csv, roots, only_verdict='MISASSIGNED', min_land_km2=100.0):
    """[(root, old_dir, new_dir, [(old_file, new_file), ...])] plus collisions.

    Cells whose land area is under ``min_land_km2`` are skipped: in a cell that
    is 99.99% ocean the "dominant region" is decided by a few square km of reef,
    which is far too thin a basis for renaming. Those cells hold no imagery
    anyway -- imagery follows land.
    """
    df = pd.read_csv(audit_csv)
    df = df[df.verdict == only_verdict]
    if 'land_frac' in df.columns:
        # cell is 5deg x 5deg; ~111 km per degree at the equator
        km2 = df.land_frac * (5 * 111)**2
        skipped = df[km2 < min_land_km2]
        for _, r in skipped.iterrows():
            print(
                f'  skip (only {r.land_frac * (5 * 111) ** 2:.0f} km2 land): '
                f'{r.cell} -> {r.dominant}')
        df = df[km2 >= min_land_km2]
    steps, collisions = [], []
    for _, r in df.iterrows():
        old, new = r.cell, target_cell(r.cell, r.dominant)
        if old == new:
            continue
        for root in roots:
            od = os.path.join(root, old)
            if not os.path.isdir(od):
                continue
            nd = os.path.join(root, new)
            if os.path.exists(nd):
                collisions.append((od, nd))
                continue
            files = []
            for f in sorted(os.listdir(od)):
                if old in f:
                    files.append((f, f.replace(old, new)))
            steps.append((root, od, nd, files))
    return steps, collisions


def apply_steps(steps, journal_path, execute):
    """Rename files first, then the directory. Journal each completed action."""
    done = []
    for root, od, nd, files in steps:
        for f_old, f_new in files:
            src, dst = os.path.join(od, f_old), os.path.join(od, f_new)
            if execute:
                if os.path.exists(dst):
                    raise FileExistsError(dst)
                os.rename(src, dst)
                done.append(['file', src, dst])
        if execute:
            if os.path.exists(nd):
                raise FileExistsError(nd)
            os.rename(od, nd)
            done.append(['dir', od, nd])
            # Flush after each cell: a kill mid-run must still be undoable.
            with open(journal_path, 'w') as fh:
                json.dump(done, fh)
    return done


def undo(journal_path):
    """Reverse a journal, newest action first."""
    with open(journal_path) as fh:
        done = json.load(fh)
    for kind, src, dst in reversed(done):
        if os.path.exists(dst) and not os.path.exists(src):
            os.rename(dst, src)
    print(f'reversed {len(done)} renames from {journal_path}')


def fix_grid_csv(path, audit_csv, execute):
    """Rewrite the region column for the mis-assigned cells."""
    grid = pd.read_csv(path)
    aud = pd.read_csv(audit_csv)
    aud = aud[aud.verdict == 'MISASSIGNED']
    key = {}
    for _, r in aud.iterrows():
        lon, lat, _, _ = r.cell.rsplit('_', 4)[1:]
        key[(int(lon), int(lat))] = r.dominant
    n = 0
    for i, g in grid.iterrows():
        want = key.get((g.sw_lon, g.sw_lat))
        if want and want != g.region:
            grid.at[i, 'region'] = want
            n += 1
    if execute and n:
        grid.to_csv(path, index=False)
    return n


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--audit', default='data/geo/region_audit.csv')
    p.add_argument('--grid', default='original_global_grid_5deg.csv')
    p.add_argument('--roots',
                   nargs='+',
                   help='Every grid_runs root. Omitting one silently leaves '
                   'that drive on the old names.')
    p.add_argument('--min-land-km2',
                   type=float,
                   default=100.0,
                   help='Skip cells with less land than this; their dominant '
                   'region rests on a sliver of reef (default 100).')
    p.add_argument('--journal-dir', default='runs')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--undo', help='Journal file to reverse.')
    p.add_argument('--stamp',
                   default='manual',
                   help='Journal filename suffix (no clock access here).')
    args = p.parse_args()

    if args.undo:
        undo(args.undo)
        return 0
    if not args.roots:
        print('--roots is required', file=sys.stderr)
        return 2

    steps, collisions = plan(args.audit,
                             args.roots,
                             min_land_km2=args.min_land_km2)
    nfiles = sum(len(f) for _, _, _, f in steps)
    by_root = defaultdict(int)
    for root, *_ in steps:
        by_root[root] += 1

    print(f"{'EXECUTING' if args.execute else 'DRY RUN'} · "
          f"{len(steps)} cell dirs · {nfiles} files\n")
    for root, n in sorted(by_root.items()):
        print(f'  {n:>4} dirs  {root}')
    if collisions:
        print(f'\n!! {len(collisions)} COLLISIONS (target exists) — aborting:')
        for od, nd in collisions[:10]:
            print(f'   {od}\n-> {nd}')
        return 1
    print('\nsample:')
    for root, od, nd, files in steps[:5]:
        print(f'  {os.path.basename(od)} -> {os.path.basename(nd)} '
              f'({len(files)} files)  [{root.split("/")[3]}]')

    ngrid = fix_grid_csv(args.grid, args.audit, args.execute)
    print(f'\ngrid CSV rows relabelled: {ngrid}'
          f'{"" if args.execute else " (not written)"}')

    if not args.execute:
        print('\nnothing changed. re-run with --execute')
        return 0

    os.makedirs(args.journal_dir, exist_ok=True)
    jp = os.path.join(args.journal_dir, f'region_fix_{args.stamp}.json')
    done = apply_steps(steps, jp, True)
    with open(jp, 'w') as fh:
        json.dump(done, fh)
    print(f'\n{len(done)} renames done · journal {jp}')
    print('  reverse with: --undo ' + jp)
    print('\nNOW STALE, regenerate:')
    print('  python tools/catalog/catalog.py refresh && '
          'python tools/catalog/catalog.py images')
    print('  coverage_missing* shards and data/missing_worklist/ are keyed by '
          'the old parent regions')
    return 0


if __name__ == '__main__':
    sys.exit(main())
