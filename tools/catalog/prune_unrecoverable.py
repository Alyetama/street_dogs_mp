#!/usr/bin/env python3
"""
Tombstone the images Mapillary no longer serves, then drop their rows from the
ground-animal manifests.

90,831 image_ids have a ``ground_animals_*.parquet`` row but no jpg on any
drive, and two independent recovery attempts failed: replaying the stored
``thumb_original_url`` (99.8% of them are NULL), and re-querying the entity API
per id for a fresh url (HTTP 200 with full metadata but no thumb url of any
size). They are gone upstream, so the manifest overstates what can ever exist
and every completeness metric is permanently short.

What this does, in order:
  1. writes ``<out>/<Region>.parquet`` tombstones FIRST -- the record exists
     before anything is modified;
  2. re-verifies, at prune time, that each id still has no jpg on ANY image
     root (a re-download between worklist and prune must not be pruned);
  3. per affected file: renames the original to ``<name>.pre_prune`` (same
     filesystem, instant, reversible), writes the pruned copy via
     tmp + fsync + rename, and asserts ``old_rows == new_rows + removed``;
  4. journals every action write-ahead so ``--undo`` restores exactly.

Deliberately NOT touched:
  * ``all_data_*.parquet`` -- the full metadata record of what Mapillary
    showed us. Its rows stay; only the ground-animal manifest is corrected.
  * the ``.pre_prune`` originals -- kept until explicitly deleted by the user.

Dry run by default.

    python tools/catalog/prune_unrecoverable.py --worklist data/missing_worklist_after
    python tools/catalog/prune_unrecoverable.py --worklist ... --execute
    python tools/catalog/prune_unrecoverable.py --undo runs/prune_<stamp>.json
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import polars as pl

REASON = 'no_url_from_entity_api'


def read_lines(path):
    out = []
    try:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln and not ln.startswith('#'):
                    out.append(ln.rstrip('/'))
    except OSError:
        pass
    return out


def load_worklist(wl_dir):
    """{cell: set(image_id)} plus {region: n} from the worklist shards."""
    by_cell = defaultdict(set)
    per_region = {}
    for p in sorted(glob.glob(os.path.join(wl_dir, '*.parquet'))):
        region = os.path.basename(p)[:-len('.parquet')]
        df = pl.read_parquet(p, columns=['image_id', 'safe_region_id'])
        per_region[region] = df.height
        for iid, cell in df.iter_rows():
            by_cell[cell].add(str(iid))
    return by_cell, per_region


def jpgs_on_disk(image_roots, cell):
    have = set()
    for root in image_roots:
        d = os.path.join(root, cell, 'ground_animal_images')
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.name.endswith('.jpg'):
                        have.add(e.name[:-4])
        except OSError:
            pass
    return have


def durable_write(df, final):
    tmp = final + '.tmp'
    df.write_parquet(tmp)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, final)
    dfd = os.open(os.path.dirname(final) or '.', os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def journal_flush(path, entries):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(entries, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def undo(journal):
    with open(journal) as f:
        acts = json.load(f)
    restored = skipped = 0
    for kind, a, b in reversed(acts):
        if kind != 'rename':
            continue
        # a = original path, b = <a>.pre_prune
        if os.path.exists(b):
            if os.path.exists(a):
                os.remove(a)  # the pruned replacement
            os.rename(b, a)
            restored += 1
        else:
            skipped += 1
    print(f'restored {restored} originals, skipped {skipped}')


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--worklist',
                   default='data/missing_worklist_after',
                   help='dir of <Region>.parquet shards listing the ids')
    p.add_argument('--data-dirs',
                   nargs='+',
                   help='grid_runs roots holding ground_animals parquets '
                   '(default: every root in data/catalog_dirs.txt)')
    p.add_argument('--image-dirs',
                   nargs='+',
                   help='EVERY root that may hold jpgs (default: same file). '
                   'Omitting one would prune an id you actually have.')
    p.add_argument('--out', default='data/missing_unrecoverable')
    p.add_argument('--journal-dir', default='runs')
    p.add_argument('--stamp', default='manual')
    p.add_argument('--region', help='only this parent region')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--undo')
    args = p.parse_args()

    if args.undo:
        undo(args.undo)
        return 0

    roots = read_lines('data/catalog_dirs.txt')
    data_dirs = args.data_dirs or roots
    image_dirs = args.image_dirs or roots
    if not data_dirs:
        print('no roots -- pass --data-dirs/--image-dirs', file=sys.stderr)
        return 2

    by_cell, per_region = load_worklist(args.worklist)
    if args.region:
        by_cell = {
            c: v
            for c, v in by_cell.items() if c.rsplit('_', 4)[0] == args.region
        }
    total_ids = sum(len(v) for v in by_cell.values())
    print(f'{total_ids:,} unrecoverable ids across {len(by_cell)} cells')

    # ---- 1. tombstones FIRST -------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    tomb = defaultdict(list)
    for cell, ids in by_cell.items():
        region = cell.rsplit('_', 4)[0]
        for iid in sorted(ids):
            tomb[region].append((iid, cell))
    if args.execute:
        for region, rows in sorted(tomb.items()):
            durable_write(
                pl.DataFrame({
                    'image_id': [r[0] for r in rows],
                    'cell': [r[1] for r in rows],
                    'region': [region] * len(rows),
                    'reason': [REASON] * len(rows),
                    'attempts':
                    ['stored_url_replay;entity_api_refetch'] * len(rows),
                }), os.path.join(args.out, f'{region}.parquet'))
        print(f'tombstones written -> {args.out}/<Region>.parquet')
    else:
        print(f'would write tombstones for {len(tomb)} regions -> {args.out}')

    # ---- 2/3. prune ground_animals only --------------------------------
    acts = []
    jp = os.path.join(args.journal_dir, f'prune_{args.stamp}.json')
    if args.execute:
        os.makedirs(args.journal_dir, exist_ok=True)
    n_files = n_removed = n_recovered = 0
    for cell in sorted(by_cell):
        ids = by_cell[cell]
        have = jpgs_on_disk(image_dirs, cell)
        prunable = ids - have  # step 2: re-verify at prune time
        recovered = ids & have
        if recovered:
            n_recovered += len(recovered)
            print(f'  {cell}: {len(recovered)} ids have jpgs now -- KEEPING')
        if not prunable:
            continue
        for root in data_dirs:
            for f in sorted(
                    glob.glob(
                        os.path.join(root, cell, 'ground_animals_*.parquet'))):
                if f.endswith('.pre_prune'):
                    continue
                try:
                    df = pl.read_parquet(f)
                except Exception as e:
                    print(f'  !! unreadable, skipping: {f} ({e})')
                    continue
                mask = df['image_id'].cast(pl.Utf8).is_in(list(prunable))
                hit = int(mask.sum())
                if not hit:
                    continue
                out = df.filter(~mask)
                assert df.height == out.height + hit, f'row math failed {f}'
                n_files += 1
                n_removed += hit
                if args.execute:
                    bak = f + '.pre_prune'
                    acts.append(['rename', f, bak])
                    journal_flush(jp, acts)  # write-ahead
                    os.rename(f, bak)
                    durable_write(out, f)
    print(f'\n{"PRUNED" if args.execute else "WOULD PRUNE"}: '
          f'{n_removed:,} rows from {n_files:,} ground_animals files')
    if n_recovered:
        print(f'  {n_recovered:,} ids skipped (jpg present now)')
    if args.execute:
        print(f'journal {jp}  --  reverse with --undo {jp}')
        print('originals kept as *.pre_prune (delete only when satisfied)')
        print('\nnow: catalog.py refresh && catalog.py images')
    else:
        print('\nnothing changed. re-run with --execute')
    return 0


if __name__ == '__main__':
    sys.exit(main())
