"""
Rebuild the ground_animals manifest from images on disk.

Problem
-------
A region's progress can show MORE "Images Collected" than "Ground Animals"
(i.e. % Images Downloaded > 100). The two numbers come from different places:

  * Ground Animals  = unique image_id in ground_animals_<region>_*.parquet
  * Images Collected = number of <image_id>.jpg files in ground_animal_images/

A .jpg is only ever written for an image_id that was in a ground_animals
parquet at download time, so on-disk images should be a SUBSET of the manifest.
When they are not, the manifest is *under-counting*: ground_animals parquet
chunks were lost (cleanup_offending_regions.py deletes parquet rows but KEEPS
images; the corruption handler deletes checkpoints; partial/aborted re-runs),
while the .jpg files and the all_data parquet survived. Because the sub-grids
are already marked .completed_/.empty_, a plain re-run skips them, so the
ground_animals rows are never regenerated and the images become "orphans".

This script treats the IMAGES as the source of truth and rebuilds the missing
manifest rows:

  1. SCAN  - for every region, find orphan image_ids
             (jpg on disk, no row in any ground_animals parquet).
  2. REBUILD - re-query Mapillary detections for each orphan and keep only the
             ones still classified as `animal--ground-animal` (the exact filter
             the main pipeline uses). For kept images, reconstruct the
             ground_animals row:
               * if the image_id is still in the region's all_data parquet, its
                 fully-formatted row is reused verbatim (no extra API call);
               * otherwise the metadata is re-fetched from the Graph API and a
                 row is built with the same schema/serialization the pipeline
                 uses.
             Recovered rows are written to
             ground_animals_<region>_recovered_<NNN>.parquet next to the
             existing chunks, so progress_tracker.py picks them up and the
             count rises to match the images on disk.

Schema parity
-------------
ground_animals_*.parquet has the SAME 17-column schema as all_data_*.parquet
(a ground_animals row is just an all_data row whose image was classified as a
ground animal). Recovered chunks reproduce that schema exactly, including the
captured_at ms->'%Y-%m-%d %H:%M:%S' conversion and JSON-string columns.

Usage (run from the repo root)
------------------------------
  # 1) Find orphans and write a manifest:
  python tools/repair/rebuild_manifest_from_images.py scan original_global_grid_5deg.csv \
      --dirs grid_runs /mnt/hdd/grid_runs ... \
      --out data/manifests/orphan_manifest.csv

  # 2) Rebuild the manifest rows for everything in the orphan manifest:
  python tools/repair/rebuild_manifest_from_images.py rebuild original_global_grid_5deg.csv \
      --dirs ... --manifest data/manifests/orphan_manifest.csv -w 24

  # Or scan + rebuild in one shot, optionally scoped to one parent region:
  python tools/repair/rebuild_manifest_from_images.py all original_global_grid_5deg.csv \
      --dirs ... --region "South Asia" -w 24

Detections and metadata are fetched from the token-gated Graph API (MLY_KEY*
in .env, rotated across calls), so no proxies are needed.
"""

import argparse
import csv
import glob
import itertools
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import orjson
import polars as pl
import requests
from dotenv import dotenv_values
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

csv.field_size_limit(sys.maxsize)

# Final ground_animals / all_data column order and dtypes.
GA_COLUMNS = [
    'image_id', 'captured_at', 'sequence', 'is_pano', 'camera_type',
    'computed_compass_angle', 'height', 'width', 'make', 'model',
    'thumb_256_url', 'thumb_1024_url', 'thumb_2048_url', 'thumb_original_url',
    'computed_geometry', 'creator', 'detections'
]
GA_DTYPES = {
    'image_id': pl.Utf8,
    'captured_at': pl.Utf8,          # already a formatted string in the parquet
    'sequence': pl.Utf8,
    'is_pano': pl.Boolean,
    'camera_type': pl.Utf8,
    'computed_compass_angle': pl.Float64,
    'height': pl.Int64,
    'width': pl.Int64,
    'make': pl.Utf8,
    'model': pl.Utf8,
    'thumb_256_url': pl.Utf8,
    'thumb_1024_url': pl.Utf8,
    'thumb_2048_url': pl.Utf8,
    'thumb_original_url': pl.Utf8,
    'computed_geometry': pl.Utf8,
    'creator': pl.Utf8,
    'detections': pl.Utf8,
}

# Same metadata fields the main pipeline requests for each image.
META_FIELDS = ('id,computed_geometry,captured_at,sequence,is_pano,camera_type,'
               'computed_compass_angle,creator,height,width,detections,make,'
               'model,thumb_256_url,thumb_1024_url,thumb_2048_url,'
               'thumb_original_url')

ANIMAL_VALUE = 'animal--ground-animal'


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def sanitize_folder_name(name):
    safe_name = name.replace('&', 'and')
    return re.sub(r'[^\w\-_\.]', '_', safe_name).strip('_')


def load_api_keys(env_path):
    env = dotenv_values(env_path)
    keys, seen = [], set()
    base = env.get('MLY_KEY')
    if base:
        keys.append(base)
        seen.add(base)
    for name, val in env.items():
        if re.fullmatch(r'MLY_KEY_\d+', name) and val and val not in seen:
            keys.append(val)
            seen.add(val)
    return keys


def build_session():
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.3,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=['GET'])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50,
                          pool_maxsize=50)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    return s


class KeyRotator:
    def __init__(self, keys):
        self._cycle = itertools.cycle(keys) if keys else None
        self._lock = threading.Lock()

    def next(self):
        if not self._cycle:
            return None
        with self._lock:
            return next(self._cycle)


def region_id_of(row):
    parent = row['region']
    uid = (f"{parent}_{row['sw_lon']}_{row['sw_lat']}_"
           f"{row['ne_lon']}_{row['ne_lat']}")
    return parent, uid, sanitize_folder_name(uid)


def find_region_context(safe_id, dirs):
    """Locate a region across dirs.

    Returns (parquet_dir, animal_files, all_data_files, image_dirs) where
    parquet_dir is the dir that holds the existing ground_animals chunks (the
    place recovered chunks should be written).
    """
    animal_files, all_data_files, image_dirs = [], [], []
    parquet_dir = None
    for d in dirs:
        rd = os.path.join(d, safe_id)
        if not os.path.isdir(rd):
            continue
        af = glob.glob(os.path.join(rd, f'ground_animals_{safe_id}_*.parquet'))
        adf = glob.glob(os.path.join(rd, f'all_data_{safe_id}_*.parquet'))
        animal_files.extend(af)
        all_data_files.extend(adf)
        if parquet_dir is None and (af or adf):
            parquet_dir = rd
        img = os.path.join(rd, 'ground_animal_images')
        if os.path.isdir(img):
            image_dirs.append(img)
    if parquet_dir is None and image_dirs:
        # No parquet chunks anywhere; fall back to the first dir that has the
        # region at all so recovered rows still land somewhere sensible.
        for d in dirs:
            rd = os.path.join(d, safe_id)
            if os.path.isdir(rd):
                parquet_dir = rd
                break
    return parquet_dir, animal_files, all_data_files, image_dirs


def disk_image_ids(image_dirs):
    ids = set()
    for img_dir in image_dirs:
        try:
            for entry in os.scandir(img_dir):
                if entry.is_file() and entry.name.lower().endswith('.jpg'):
                    ids.add(entry.name[:-4])
        except FileNotFoundError:
            pass
    return ids


def parquet_image_ids(files):
    if not files:
        return set()
    try:
        return set(map(str, pl.scan_parquet(list(set(files)))
                       .select('image_id').collect()['image_id'].to_list()))
    except Exception:
        return set()


# --------------------------------------------------------------------------- #
# Phase 1: SCAN - find orphan image_ids per region
# --------------------------------------------------------------------------- #
def scan_region(row, dirs):
    parent, _uid, safe_id = region_id_of(row)
    parquet_dir, animal_files, _all_files, image_dirs = find_region_context(
        safe_id, dirs)
    if not image_dirs:
        return None
    ga_ids = parquet_image_ids(animal_files)
    disk_ids = disk_image_ids(image_dirs)
    orphans = disk_ids - ga_ids
    if not orphans:
        return None
    return [{
        'image_id': iid,
        'parent_region': parent,
        'safe_region_id': safe_id,
    } for iid in orphans]


def cmd_scan(args):
    df_grid = pl.read_csv(args.grid_csv)
    rows = list(df_grid.iter_rows(named=True))
    if args.region:
        rows = [r for r in rows if r['region'] == args.region]
        print(f"[scan] filtered to {len(rows)} grid rows for "
              f"'{args.region}'.")

    manifest, per_region = [], {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(scan_region, r, args.dirs): r for r in rows}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Scanning regions"):
            res = fut.result()
            if res:
                manifest.extend(res)
                pr = res[0]['parent_region']
                per_region[pr] = per_region.get(pr, 0) + len(res)

    # Global de-dup (an image_id can appear under overlapping grid cells).
    seen, deduped = set(), []
    for r in manifest:
        if r['image_id'] in seen:
            continue
        seen.add(r['image_id'])
        deduped.append(r)

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'parent_region',
                                          'safe_region_id'])
        w.writeheader()
        w.writerows(deduped)

    print("\n[scan] orphan images by region:")
    for pr in sorted(per_region):
        print(f"    {pr:<32} {per_region[pr]:>12,}")
    print(f"\n[scan] TOTAL orphans (unique): {len(deduped):,}")
    print(f"[scan] manifest written -> {args.out}")
    return deduped


# --------------------------------------------------------------------------- #
# Phase 2: REBUILD - re-classify orphans and write recovered chunks
# --------------------------------------------------------------------------- #
def is_ground_animal(image_id, session, key_rotator):
    """(is_animal, is_dead). is_dead means the id no longer resolves."""
    key = key_rotator.next()
    url = (f'https://graph.mapillary.com/{image_id}/detections'
           f'?access_token={key}&fields=value')
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code in (400, 404):
            return False, True
        resp.raise_for_status()
        data = resp.json().get('data', [])
        return (any(d.get('value') == ANIMAL_VALUE for d in data), False)
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else None
        return False, code in (400, 404)
    except Exception:
        return False, False


def fetch_metadata(image_id, session, key_rotator):
    key = key_rotator.next()
    url = (f'https://graph.mapillary.com/{image_id}'
           f'?access_token={key}&fields={META_FIELDS}')
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code in (400, 404):
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def row_from_metadata(meta):
    """Build a raw record dict (pre-typing) from a Graph API metadata object."""
    rec = {'image_id': str(meta.get('id'))}
    for col in ('captured_at', 'sequence', 'is_pano', 'camera_type',
                'computed_compass_angle', 'height', 'width', 'make', 'model',
                'thumb_256_url', 'thumb_1024_url', 'thumb_2048_url',
                'thumb_original_url'):
        rec[col] = meta.get(col)
    cg = meta.get('computed_geometry')
    rec['computed_geometry'] = orjson.dumps(cg).decode('utf-8') if cg else None
    cr = meta.get('creator')
    rec['creator'] = orjson.dumps(cr).decode('utf-8') if cr else None
    det = meta.get('detections')
    if isinstance(det, dict):
        det = det.get('data', [])
    rec['detections'] = orjson.dumps(det).decode('utf-8') if det else None
    return rec


def coerce_local(df):
    """Take rows lifted from all_data and force the canonical schema/order."""
    for col in GA_COLUMNS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))
    df = df.with_columns([pl.col(c).cast(GA_DTYPES[c], strict=False)
                          for c in GA_COLUMNS])
    return df.select(GA_COLUMNS)


def build_api_df(records):
    """Build a canonical-schema DataFrame from re-fetched metadata records,
    matching the pipeline's captured_at ms->string conversion."""
    if not records:
        return pl.DataFrame(schema=GA_DTYPES)
    df = pl.DataFrame(records, infer_schema_length=None)
    if 'captured_at' in df.columns:
        df = df.with_columns(
            pl.col('captured_at').cast(pl.Int64, strict=False)
            .cast(pl.Datetime("ms")).dt.strftime('%Y-%m-%d %H:%M:%S'))
    return coerce_local(df)


def existing_recovered_ids(parquet_dir, safe_id):
    files = glob.glob(os.path.join(
        parquet_dir, f'ground_animals_{safe_id}_recovered_*.parquet'))
    return parquet_image_ids(files), files


def next_recovered_path(parquet_dir, safe_id, existing_files):
    idx = len(existing_files)
    return os.path.join(
        parquet_dir, f'ground_animals_{safe_id}_recovered_{idx:03d}.parquet')


def rebuild_region(safe_id, parent, orphan_ids, dirs, key_rotator, args,
                   tls):
    """Re-classify a region's orphans and write a recovered parquet chunk.

    Returns a stats dict.
    """
    stats = {'region': safe_id, 'orphans': len(orphan_ids), 'recovered': 0,
             'not_animal': 0, 'dead': 0, 'failed': 0, 'skipped': 0}

    parquet_dir, _af, all_files, _img = find_region_context(safe_id, dirs)
    if parquet_dir is None:
        stats['failed'] = len(orphan_ids)
        return stats

    already, existing_files = existing_recovered_ids(parquet_dir, safe_id)
    todo = [iid for iid in orphan_ids if iid not in already]
    stats['skipped'] = len(orphan_ids) - len(todo)
    if not todo:
        return stats

    all_ids = parquet_image_ids(all_files)
    todo_set = set(todo)
    in_ad = todo_set & all_ids
    not_ad = todo_set - all_ids

    def get_session():
        s = getattr(tls, 'session', None)
        if s is None:
            s = build_session()
            tls.session = s
        return s

    # --- gate: which orphans are still ground animals? ---
    animal_ids, dead, failed = set(), 0, 0

    def gate(iid):
        return (iid, ) + is_ground_animal(iid, get_session(), key_rotator)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in tqdm(as_completed([ex.submit(gate, i) for i in todo]),
                        total=len(todo), desc=f"{safe_id[:38]} gate",
                        leave=False, mininterval=2.0):
            iid, is_animal, is_dead = fut.result()
            if is_animal:
                animal_ids.add(iid)
            elif is_dead:
                dead += 1
            else:
                stats['not_animal'] += 1
    stats['dead'] = dead

    # --- recover rows ---
    local_ids = list(in_ad & animal_ids)
    api_ids = list(not_ad & animal_ids)

    frames = []
    if local_ids:
        local_df = (pl.scan_parquet(list(set(all_files)))
                    .filter(pl.col('image_id').cast(pl.Utf8).is_in(local_ids))
                    .collect()
                    .unique(subset=['image_id'], keep='first'))
        frames.append(coerce_local(local_df))

    if api_ids:
        records = []

        def meta(iid):
            return iid, fetch_metadata(iid, get_session(), key_rotator)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in tqdm(as_completed([ex.submit(meta, i) for i in api_ids]),
                            total=len(api_ids), desc=f"{safe_id[:38]} meta",
                            leave=False, mininterval=2.0):
                iid, m = fut.result()
                if m:
                    records.append(row_from_metadata(m))
                else:
                    stats['failed'] += 1
        if records:
            frames.append(build_api_df(records))

    if not frames:
        return stats

    recovered = pl.concat(frames, how='vertical').unique(
        subset=['image_id'], keep='first')

    if recovered.height == 0:
        return stats

    out_path = next_recovered_path(parquet_dir, safe_id, existing_files)
    if not args.dry_run:
        recovered.write_parquet(out_path, compression='zstd')
    stats['recovered'] = recovered.height
    stats['out_path'] = out_path
    return stats


def load_manifest(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def cmd_rebuild(args, manifest=None):
    if manifest is None:
        manifest = load_manifest(args.manifest)

    keys = load_api_keys(args.env)
    if not keys:
        print("[rebuild] FATAL: no MLY_KEY* tokens found in", args.env)
        return
    key_rotator = KeyRotator(keys)
    print(f"[rebuild] {len(manifest):,} orphan rows | {len(keys)} API key(s) | "
          f"{args.workers} workers"
          f"{' | DRY-RUN (no parquet written)' if args.dry_run else ''}")

    # Group orphans by region.
    by_region = {}
    region_parent = {}
    for r in manifest:
        sid = r['safe_region_id']
        by_region.setdefault(sid, []).append(r['image_id'])
        region_parent[sid] = r['parent_region']

    done_path = args.done_out
    done = set()
    if done_path and os.path.exists(done_path):
        with open(done_path) as f:
            done = {ln.strip() for ln in f if ln.strip()}
        print(f"[rebuild] resuming: {len(done)} regions already complete.")

    tls = threading.local()
    totals = {'recovered': 0, 'not_animal': 0, 'dead': 0, 'failed': 0,
              'skipped': 0, 'orphans': 0}

    region_items = [(sid, ids) for sid, ids in by_region.items()
                    if sid not in done]
    for sid, ids in tqdm(region_items, desc="Regions"):
        stats = rebuild_region(sid, region_parent[sid], ids, args.dirs,
                               key_rotator, args, tls)
        for k in totals:
            totals[k] += stats.get(k, 0)
        msg = (f"  {sid[:50]:<50} orphans={stats['orphans']:>7,} "
               f"recovered={stats['recovered']:>7,} "
               f"not_animal={stats['not_animal']:>5,} dead={stats['dead']:>5,} "
               f"failed={stats['failed']:>5,}")
        tqdm.write(msg)
        if done_path and not args.dry_run and stats.get('failed', 0) == 0:
            with open(done_path, 'a') as f:
                f.write(sid + '\n')

    print("\n[rebuild] done.")
    print(f"    recovered rows written : {totals['recovered']:,}")
    print(f"    no longer animal       : {totals['not_animal']:,}")
    print(f"    dead (gone from API)   : {totals['dead']:,}")
    print(f"    metadata fetch failed  : {totals['failed']:,}")
    print(f"    already recovered      : {totals['skipped']:,}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)

    def add_common(p):
        p.add_argument('grid_csv')
        p.add_argument('--dirs', nargs='+', required=True)
        p.add_argument('--region', default=None,
                       help='Restrict to one parent region (e.g. "Europe").')
        p.add_argument('-w', '--workers', type=int, default=16)
        p.add_argument('--env', default='.env',
                       help='Path to .env holding MLY_KEY* tokens.')

    def add_rebuild(p):
        p.add_argument('--done-out', default='data/rebuild_done_regions.txt',
                       help='Append-only log of fully-rebuilt regions.')
        p.add_argument('--dry-run', action='store_true',
                       help='Classify but do not write recovered parquets.')

    p_scan = sub.add_parser('scan', help='Find orphans, write manifest.')
    add_common(p_scan)
    p_scan.add_argument('--out', default='data/manifests/orphan_manifest.csv')

    p_reb = sub.add_parser('rebuild', help='Rebuild rows from a manifest.')
    add_common(p_reb)
    p_reb.add_argument('--manifest', required=True)
    add_rebuild(p_reb)

    p_all = sub.add_parser('all', help='Scan then rebuild in one run.')
    add_common(p_all)
    p_all.add_argument('--out', default='data/manifests/orphan_manifest.csv')
    add_rebuild(p_all)

    args = parser.parse_args()

    if args.mode == 'scan':
        cmd_scan(args)
    elif args.mode == 'rebuild':
        cmd_rebuild(args)
    elif args.mode == 'all':
        manifest = cmd_scan(args)
        if manifest:
            print()
            cmd_rebuild(args, manifest=manifest)


if __name__ == '__main__':
    main()
