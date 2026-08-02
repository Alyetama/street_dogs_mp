#!/usr/bin/env python3
"""
Apply the region-audit corrections: rename mis-filed cells on every data root
and fix the grid CSV — atomically enough that CSV and disk can never drift.

``original_global_grid_5deg.csv`` labels regions with coarse lat/lon boxes, and
several were drawn wrong -- the ``Africa`` box reached lon 55 / lat 40 and was
tested before ``Middle East`` (12 cells, lon 55..65), so Kuwait, Riyadh, Baghdad
and Doha were all filed as Africa. ``audit_grid_regions.py`` finds such cells by
intersecting each cell with Natural Earth polygons; this applies its verdict.

Only ``MISASSIGNED`` rows are acted on. ``taxonomy`` (right continent, different
sub-label) and ``straddles`` (no single correct label at 5 degrees) are left
alone deliberately.

One eligibility decision, applied everywhere
--------------------------------------------
The 2026-08-01 run relabelled 54 CSV rows but renamed only 51 cells' dirs,
because the land-area guard was applied to the dir plan and NOT to the CSV
rewrite -- leaving 3 cells where tools deriving dir names from the CSV missed
the data on disk (one of those cells held 25,530 rows and 172 jpgs; the guard
skipped it only through a rounding bug, since fixed). Now ``eligible()``
computes the acted-on set ONCE, from the audit's full-precision equal-area
``land_km2`` column, and both the dir plan and the CSV rewrite consume that
same set.

The cell name is embedded in filenames (``all_data_<cell>_NNN.parquet``,
``validated_images_<cell>.txt``, ``<cell>_tiles.png``), so files are renamed
along with the directory; missing that leaves parquets invisible to every glob.

Journal & undo
--------------
Every action -- file rename, dir rename, AND each CSV row change -- is written
to the journal BEFORE it is executed (write-ahead), so a crash at any point
leaves a journal that ``--undo`` can replay. ``--undo`` reverses newest-first,
skips actions that were never performed (idempotent), restores the CSV rows,
and reports exactly what it reversed vs skipped. Note: file entries record
paths as they were at execution time (inside the OLD dir name, because files
are renamed before their directory); undo replays in reverse order, so the dir
is restored first and those paths become valid again.

Reconcile
---------
``--reconcile`` ignores the audit and instead diffs DISK against the CSV: any
cell directory whose region prefix disagrees with the CSV's (sanitized) region
for that bbox is renamed to match the CSV. This heals historical drift.

Dry run by default.

    python tools/catalog/fix_grid_regions.py --roots <grid_runs> ...      # plan
    python tools/catalog/fix_grid_regions.py --roots <grid_runs> ... --execute
    python tools/catalog/fix_grid_regions.py --roots <grid_runs> ... --reconcile [--execute]
    python tools/catalog/fix_grid_regions.py --undo runs/region_fix_<stamp>.json
"""

import argparse
import glob
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

    The bbox is the last four underscore-separated fields (coordinates may be
    negative, which is why the split is from the right).
    """
    bbox = old_cell.rsplit('_', 4)[1:]
    return safe_region(new_region) + '_' + '_'.join(bbox)


def eligible(audit_csv, min_land_km2):
    """The acted-on set: MISASSIGNED rows passing the land guard.

    Uses the audit's equal-area ``land_km2`` column (0.1 km2 precision). (The previous
    version reconstructed km2 from a 3-decimal ``land_frac``, which rounded a
    120 km2 cell to zero and silently skipped it.) Both the dir plan and the
    CSV rewrite must consume THIS set and nothing else.
    """
    df = pd.read_csv(audit_csv)
    df = df[df.verdict == 'MISASSIGNED'].copy()
    if 'land_km2' not in df.columns:
        raise SystemExit(
            'audit CSV has no land_km2 column -- re-run '
            'audit_grid_regions.py first (older audits carried '
            'only a rounded land_frac, which caused false skips).')
    skipped = df[df.land_km2 < min_land_km2]
    for _, r in skipped.iterrows():
        print(f'  skip (only {r.land_km2:.0f} km2 land): '
              f'{r.cell} -> {r.dominant}')
    return df[df.land_km2 >= min_land_km2]


def steps_for(renames, roots):
    """[(root, old_dir, new_dir, [(old_file, new_file), ...])], collisions.

    ``renames`` is [(old_cell, new_cell)].
    """
    steps, collisions = [], []
    for old, new in renames:
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
            files = [(f, f.replace(old, new)) for f in sorted(os.listdir(od))
                     if old in f]
            steps.append((root, od, nd, files))
    return steps, collisions


def reconcile_renames(grid_csv, roots):
    """[(old_cell, new_cell)] for every dir whose prefix disagrees with the CSV."""
    grid = pd.read_csv(grid_csv)
    want = {
        (int(g.sw_lon), int(g.sw_lat)): safe_region(g.region)
        for _, g in grid.iterrows()
    }
    out, seen = [], set()
    for root in roots:
        for d in glob.glob(os.path.join(root, '*')):
            cell = os.path.basename(d)
            parts = cell.rsplit('_', 4)
            if len(parts) != 5 or not os.path.isdir(d):
                continue
            try:
                key = (int(parts[1]), int(parts[2]))
            except ValueError:
                continue
            expect = want.get(key)
            if expect and parts[0] != expect and cell not in seen:
                seen.add(cell)
                out.append((cell, expect + '_' + '_'.join(parts[1:])))
    return out


def journal_flush(path, done):
    """Atomic AND durable: fsync file and directory so the write-ahead entry
    cannot be reordered after its rename by a power loss."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(done, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    dfd = os.open(os.path.dirname(os.path.abspath(path)) or '.', os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def apply_steps(steps, csv_changes, grid_path, journal_path, execute):
    """Write-ahead: journal each batch of renames BEFORE performing them.

    ``csv_changes`` is [(sw_lon, sw_lat, old_region, new_region)]; journalled
    up front so --undo can restore the CSV as well as the file system.
    """
    done = []
    for lon, lat, old, new in csv_changes:
        done.append(['csv', grid_path, f'{lon},{lat}', old, new])
    if execute:
        journal_flush(journal_path, done)
    for root, od, nd, files in steps:
        batch = ([['file', os.path.join(od, a),
                   os.path.join(od, b)] for a, b in files] + [['dir', od, nd]])
        if execute:
            done.extend(batch)
            journal_flush(journal_path, done)  # intent recorded first
            for kind, src, dst in batch:
                if os.path.exists(dst):
                    raise FileExistsError(dst)
                os.rename(src, dst)
    return done


def undo(journal_path):
    """Reverse a journal newest-first; restore CSV rows; report honestly."""
    with open(journal_path) as fh:
        done = json.load(fh)
    reversed_n, skipped = 0, []
    csv_restores = defaultdict(list)
    for entry in reversed(done):
        if entry[0] == 'csv':
            _, grid_path, key, old, new = entry
            csv_restores[grid_path].append((key, old, new))
            continue
        kind, src, dst = entry
        if os.path.exists(dst) and not os.path.exists(src):
            os.rename(dst, src)
            reversed_n += 1
        else:
            skipped.append(f'{kind}: {dst}')
    for grid_path, rows in csv_restores.items():
        grid = pd.read_csv(grid_path)
        n = 0
        for key, old, new in rows:
            lon, lat = (int(x) for x in key.split(','))
            m = (grid.sw_lon == lon) & (grid.sw_lat == lat) & \
                (grid.region == new)
            if m.any():
                grid.loc[m, 'region'] = old
                n += 1
        if n:
            grid.to_csv(grid_path, index=False)
        print(f'  CSV: restored {n}/{len(rows)} rows in {grid_path}')
    print(f'reversed {reversed_n} renames from {journal_path}')
    if skipped:
        print(f'  skipped {len(skipped)} never-performed or already-reversed '
              'actions (expected after a crash, given write-ahead entries):')
        for x in skipped[:10]:
            print(f'    {x}')


def csv_changes_for(grid_csv, eligible_df):
    """[(sw_lon, sw_lat, old_region, new_region)] for the eligible set."""
    grid = pd.read_csv(grid_csv)
    key = {}
    for _, r in eligible_df.iterrows():
        lon, lat, _, _ = r.cell.rsplit('_', 4)[1:]
        key[(int(lon), int(lat))] = r.dominant
    out = []
    for _, g in grid.iterrows():
        want = key.get((int(g.sw_lon), int(g.sw_lat)))
        if want and want != g.region:
            out.append((int(g.sw_lon), int(g.sw_lat), g.region, want))
    return out


def write_csv_changes(grid_csv, changes):
    grid = pd.read_csv(grid_csv)
    for lon, lat, old, new in changes:
        m = (grid.sw_lon == lon) & (grid.sw_lat == lat)
        grid.loc[m, 'region'] = new
    grid.to_csv(grid_csv, index=False)


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
                   'region rests on a sliver of reef (default 100). Applied '
                   'to BOTH the dir renames and the CSV rewrite.')
    p.add_argument('--reconcile',
                   action='store_true',
                   help='Ignore the audit; rename any dir whose region prefix '
                   'disagrees with the CSV for its bbox (heals drift).')
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

    if args.reconcile:
        renames = reconcile_renames(args.grid, args.roots)
        csv_changes = []  # reconcile aligns disk TO the CSV
        mode = 'RECONCILE (disk -> CSV)'
    else:
        elig = eligible(args.audit, args.min_land_km2)
        renames = [(r.cell, target_cell(r.cell, r.dominant))
                   for _, r in elig.iterrows()]
        csv_changes = csv_changes_for(args.grid, elig)
        mode = 'AUDIT'

    steps, collisions = steps_for(renames, args.roots)
    nfiles = sum(len(f) for _, _, _, f in steps)
    by_root = defaultdict(int)
    for root, *_ in steps:
        by_root[root] += 1

    print(f"{mode} · {'EXECUTING' if args.execute else 'DRY RUN'} · "
          f"{len(steps)} cell dirs · {nfiles} files · "
          f"{len(csv_changes)} CSV rows\n")
    for root, n in sorted(by_root.items()):
        print(f'  {n:>4} dirs  {root}')
    if collisions:
        print(
            f'\n!! {len(collisions)} COLLISIONS (target exists) -- aborting:')
        for od, nd in collisions[:10]:
            print(f'   {od}\n-> {nd}')
        return 1
    print('\nsample:')
    for root, od, nd, files in steps[:5]:
        print(f'  {os.path.basename(od)} -> {os.path.basename(nd)} '
              f'({len(files)} files)')

    if not args.execute:
        print('\nnothing changed. re-run with --execute')
        return 0

    os.makedirs(args.journal_dir, exist_ok=True)
    jp = os.path.join(args.journal_dir, f'region_fix_{args.stamp}.json')
    if os.path.exists(jp):
        print(f'!! journal {jp} exists -- pick a different --stamp',
              file=sys.stderr)
        return 1
    done = apply_steps(steps, csv_changes, args.grid, jp, True)
    if csv_changes:
        write_csv_changes(args.grid, csv_changes)
    journal_flush(jp, done)
    print(f'\n{len(done)} journalled actions done · journal {jp}')
    print('  reverse with: --undo ' + jp)
    print('\nNOW STALE, regenerate:')
    print('  python tools/catalog/catalog.py refresh && '
          'python tools/catalog/catalog.py images')
    print('  coverage_missing* shards and data/missing_worklist/ are keyed by '
          'the old cell names (file names AND safe_region_id row values)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
