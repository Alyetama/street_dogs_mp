"""
Drop permanently-dead images from the ground_animals manifest parquets.

"Dead" = image_ids in data/dead_images.txt: ground-animal detections whose source
media Mapillary no longer serves (verified unrecoverable via single/bulk/bbox/
sequence API paths). They inflate the "missing" gap (% Images Downloaded < 100)
even though the images can never be downloaded.

This removes those rows from every ground_animals_<region>_*.parquet, but FIRST
archives the removed rows (with provenance columns) to a separate parquet so they
can be restored later if needed.

Safety:
  * Phase 1 reads all manifests, collects the dead rows, and writes the archive.
    Nothing is modified yet. If anything fails here, originals are untouched.
  * Phase 2 rewrites only the affected parquets, atomically (write .tmp then
    os.replace). Re-running is idempotent (already-clean files are skipped).
  * Default is DRY-RUN; pass --execute to actually write.

Usage (run from the repo root):
  python tools/repair/drop_dead_from_manifest.py --dirs DIR1 DIR2 ...            # dry-run
  python tools/repair/drop_dead_from_manifest.py --dirs DIR1 DIR2 ... --execute  # do it

Restore later (if ever needed):
  import polars as pl
  arc = pl.read_parquet('data/dead_manifest_rows.parquet')
  # group by _source_file / _region and concat back into the originals,
  # dropping the _region/_source_file helper columns first.
"""

import argparse
import glob
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dirs', nargs='+', required=True,
                   help='Base grid_runs directories to search for parquets.')
    p.add_argument('--dead-file', default='data/dead_images.txt')
    p.add_argument('--archive', default='data/dead_manifest_rows.parquet',
                   help='Where to save the removed rows (the safety net).')
    p.add_argument('--execute', action='store_true',
                   help='Actually write the archive and rewrite manifests '
                        '(default: dry-run).')
    p.add_argument('-w', '--workers', type=int, default=24)
    args = p.parse_args()

    if not os.path.exists(args.dead_file):
        print(f"[fatal] dead file not found: {args.dead_file}")
        sys.exit(1)

    dead = {ln.strip() for ln in open(args.dead_file) if ln.strip()}
    print(f"[info] {len(dead):,} dead image_ids loaded from {args.dead_file}")

    files = []
    for d in args.dirs:
        files.extend(glob.glob(os.path.join(d, '*', 'ground_animals_*.parquet')))
    files = sorted(set(files))
    print(f"[info] scanning {len(files):,} ground_animals parquet files...")

    # --- Detection pass: read only the image_id column (fast/lazy), in
    #     parallel, to find which files contain dead rows, so the heavy full
    #     reads below only touch the affected files. ---
    def detect(f):
        try:
            ids = pl.scan_parquet(f).select(
                pl.col('image_id').cast(pl.Utf8)).collect()['image_id']
        except Exception as e:
            return f, None, str(e)
        return f, int(ids.is_in(dead).sum()), None

    affected = []          # (path, dropped_count)
    per_parent = Counter()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f, n_dead, err in (fut.result() for fut in as_completed(
                ex.submit(detect, f) for f in files)):
            done += 1
            if err is not None:
                print(f"  [warn] unreadable, skipping: {f} ({err})")
            elif n_dead:
                affected.append((f, n_dead))
                parent = os.path.basename(os.path.dirname(f)).rsplit('_', 4)[0]
                per_parent[parent] += n_dead
            if done % 2000 == 0:
                print(f"    ...scanned {done:,}/{len(files):,}")

    total_dropped = sum(a[1] for a in affected)
    print(f"\n[summary] affected parquet files: {len(affected):,}")
    print(f"[summary] total rows to drop:      {total_dropped:,}")
    print("[summary] by parent region:")
    for parent, n in sorted(per_parent.items(), key=lambda x: -x[1]):
        print(f"    {parent:<34} {n:>10,}")

    if total_dropped == 0:
        print("\n[done] nothing to drop.")
        return

    if not args.execute:
        print("\n[DRY-RUN] no files written. Re-run with --execute to apply.")
        return

    # ---- Phase 1: write the archive FIRST (safety net) ----
    # Some parquets store all-null columns (e.g. thumb_*_url) with a Null dtype
    # that clashes with String in other files, so coerce every column that
    # SHOULD be text to Utf8 before concatenating.
    text_cols = ['image_id', 'captured_at', 'sequence', 'camera_type', 'make',
                 'model', 'thumb_256_url', 'thumb_1024_url', 'thumb_2048_url',
                 'thumb_original_url', 'computed_geometry', 'creator',
                 'detections']
    dead_frames = []
    for f, _n in affected:
        df = pl.read_parquet(f)
        region = os.path.basename(os.path.dirname(f))
        df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in text_cols
                              if c in df.columns])
        dead_frames.append(
            df.filter(df['image_id'].is_in(dead)).with_columns([
                pl.lit(region).alias('_region'),
                pl.lit(os.path.basename(f)).alias('_source_file'),
            ]))
    archive_df = pl.concat(dead_frames, how='vertical_relaxed')
    # If an archive already exists from a previous run, merge and de-dup so we
    # never lose previously-archived rows.
    if os.path.exists(args.archive):
        try:
            prev = pl.read_parquet(args.archive)
            archive_df = pl.concat([prev, archive_df], how='vertical_relaxed')
        except Exception as e:
            print(f"  [warn] could not read existing archive ({e}); "
                  f"writing fresh.")
    archive_df = archive_df.unique(subset=['image_id', '_source_file'],
                                   keep='first')
    tmp_arc = args.archive + '.tmp'
    archive_df.write_parquet(tmp_arc, compression='zstd')
    os.replace(tmp_arc, args.archive)
    print(f"\n[archive] {archive_df.height:,} rows saved to {args.archive}")

    # ---- Phase 2: rewrite affected manifests (atomic) ----
    rewritten = 0
    for f, _drop in affected:
        df = pl.read_parquet(f)
        kept = df.filter(~df['image_id'].cast(pl.Utf8).is_in(dead))
        tmp = f + '.tmp'
        kept.write_parquet(tmp, compression='zstd')
        os.replace(tmp, f)
        rewritten += 1
        if rewritten % 200 == 0:
            print(f"    ...rewritten {rewritten:,}/{len(affected):,}")

    print(f"\n[done] rewrote {rewritten:,} manifest files; "
          f"dropped {total_dropped:,} dead rows (archived in {args.archive}).")


if __name__ == '__main__':
    main()
