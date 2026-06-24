"""
Standalone finder + downloader for MISSING ground-animal images.

Background
----------
Each region's `ground_animals_<region>_*.parquet` lists every animal image_id
together with a `thumb_original_url`. Those URLs are *signed* fbcdn links that
EXPIRE (note the `oe=` / `oh=` params). Once expired, a plain re-run of
`batch_chunks_mp_api_v3.py` cannot re-download them, and because the regions are
already marked `.completed_/.empty_`, the main script skips them entirely - so
the on-disk gaps never get filled. That is why "nothing new downloaded".

This script:
  1. SCANS every region across all data dirs and finds image_ids that are in the
     parquet but have no .jpg on disk (the true "missing" set).
  2. Writes a manifest file listing all missing images.
  3. DOWNLOADS them: it tries the stored URL first, and on any failure fetches a
     FRESH `thumb_original_url` from the Mapillary Graph API (rotating across all
     MLY_KEY* tokens) and retries. Images the API reports as gone (400/404) are
     recorded as permanently dead.

Usage (run from the repo root)
------------------------------
  # 1) Find missing images and write the manifest only:
  python tools/repair/fetch_missing_images.py scan original_global_grid_5deg.csv \
      --dirs grid_runs /mnt/hdd/grid_runs ... \
      --out data/manifests/missing_manifest.csv

  # 2) Download everything in the manifest:
  python tools/repair/fetch_missing_images.py download \
      --manifest data/manifests/missing_manifest.csv --proxy-file proxies.txt -w 24

  # Or do both in one shot:
  python tools/repair/fetch_missing_images.py all original_global_grid_5deg.csv --dirs ... \
      --out data/manifests/missing_manifest.csv --proxy-file proxies.txt -w 24

  # Restrict the scan to one parent region:
  python tools/repair/fetch_missing_images.py scan original_global_grid_5deg.csv --dirs ... \
      --region "Europe" --out data/manifests/missing_europe.csv
"""

import argparse
import csv
import itertools
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

load_dotenv()

csv.field_size_limit(sys.maxsize)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def sanitize_folder_name(name):
    safe_name = name.replace('&', 'and')
    return re.sub(r'[^\w\-_\.]', '_', safe_name).strip('_')


def load_api_keys():
    """Collect MLY_KEY plus every MLY_KEY_<n> from the environment."""
    keys = []
    base = os.environ.get('MLY_KEY')
    if base:
        keys.append(base)
    for name, val in os.environ.items():
        if re.fullmatch(r'MLY_KEY_\d+', name) and val:
            keys.append(val)
    # de-dup, preserve order
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def load_proxies(proxy_file):
    proxies = []
    if not proxy_file or not os.path.exists(proxy_file):
        return proxies
    with open(proxy_file) as pf:
        for line in pf:
            line = line.strip()
            if not line:
                continue
            parts = line.split(':')
            if len(parts) == 4:
                ip, port, user, pw = parts
                proxies.append(f"http://{user}:{pw}@{ip}:{port}")
            elif len(parts) == 2:
                ip, port = parts
                proxies.append(f"http://{ip}:{port}")
            elif line.startswith(('http', 'socks')):
                proxies.append(line)
    return proxies


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


def is_valid_jpeg(filepath):
    try:
        if os.path.getsize(filepath) < 100:
            return False
        with open(filepath, 'rb') as f:
            return f.read(2) == b'\xff\xd8'
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Phase 1: SCAN - find image_ids that are in the parquet but not on disk
# --------------------------------------------------------------------------- #
def scan_region_row(row, dirs):
    parent_region = row['region']
    unique_region_id = (f"{parent_region}_{row['sw_lon']}_{row['sw_lat']}_"
                        f"{row['ne_lon']}_{row['ne_lat']}")
    safe_region_id = sanitize_folder_name(unique_region_id)

    region_dirs = []
    for d in dirs:
        p = os.path.join(d, safe_region_id)
        if os.path.isdir(p):
            region_dirs.append(p)
    if not region_dirs:
        return None

    animal_files = []
    image_dirs = []
    for r_dir in region_dirs:
        try:
            for entry in os.scandir(r_dir):
                if (entry.is_file()
                        and entry.name.startswith(f'ground_animals_{safe_region_id}_')
                        and entry.name.endswith('.parquet')):
                    animal_files.append(entry.path)
        except Exception:
            pass
        img_dir = os.path.join(r_dir, 'ground_animal_images')
        if os.path.isdir(img_dir):
            image_dirs.append(img_dir)

    if not animal_files:
        return None

    # Map of image_id -> stored thumb_original_url (last one wins)
    id_to_url = {}
    try:
        df = (pl.scan_parquet(list(set(animal_files)))
              .select(['image_id', 'thumb_original_url'])
              .unique(subset=['image_id'], keep='last')
              .collect())
        for iid, url in zip(df['image_id'].to_list(),
                            df['thumb_original_url'].to_list()):
            id_to_url[str(iid)] = url
    except Exception:
        return None

    # Image_ids already present on disk (union across all region dirs).
    disk_ids = set()
    for img_dir in image_dirs:
        try:
            for entry in os.scandir(img_dir):
                if entry.is_file() and entry.name.lower().endswith('.jpg'):
                    disk_ids.add(entry.name[:-4])
        except Exception:
            pass

    missing_ids = set(id_to_url.keys()) - disk_ids
    if not missing_ids:
        return None

    # Write target into the first region dir's image folder.
    target_dir = os.path.join(region_dirs[0], 'ground_animal_images')

    rows = []
    for iid in missing_ids:
        rows.append({
            'image_id': iid,
            'parent_region': parent_region,
            'safe_region_id': safe_region_id,
            'target_dir': target_dir,
            'stored_url': id_to_url[iid] or '',
        })
    return rows


def cmd_scan(args):
    keys = load_api_keys()
    print(f"[scan] {len(keys)} API key(s) available for the download phase.")

    df_grid = pl.read_csv(args.grid_csv)
    rows = list(df_grid.iter_rows(named=True))
    if args.region:
        rows = [r for r in rows if r['region'] == args.region]
        print(f"[scan] filtered to {len(rows)} grid rows for region "
              f"'{args.region}'.")

    manifest_rows = []
    per_region = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(scan_region_row, r, args.dirs): r for r in rows}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Scanning regions"):
            res = fut.result()
            if res:
                manifest_rows.extend(res)
                pr = res[0]['parent_region']
                per_region[pr] = per_region.get(pr, 0) + len(res)

    # Global de-dup by image_id (same id can appear in overlapping grid cells).
    seen = set()
    deduped = []
    for r in manifest_rows:
        if r['image_id'] in seen:
            continue
        seen.add(r['image_id'])
        deduped.append(r)

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'parent_region',
                                          'safe_region_id', 'target_dir',
                                          'stored_url'])
        w.writeheader()
        w.writerows(deduped)

    print(f"\n[scan] missing images by region:")
    for pr in sorted(per_region):
        print(f"    {pr:<32} {per_region[pr]:>10,}")
    print(f"\n[scan] TOTAL missing (unique): {len(deduped):,}")
    print(f"[scan] manifest written -> {args.out}")
    return deduped


# --------------------------------------------------------------------------- #
# Phase 2: DOWNLOAD - stored URL first, then fresh URL from the Graph API
# --------------------------------------------------------------------------- #
class KeyRotator:
    def __init__(self, keys):
        self._cycle = itertools.cycle(keys) if keys else None
        self._lock = threading.Lock()

    def next(self):
        if not self._cycle:
            return None
        with self._lock:
            return next(self._cycle)


# Preference order: original first, then progressively smaller thumbs so we can
# still recover an image when the original is no longer served.
THUMB_FIELDS = ['thumb_original_url', 'thumb_2048_url', 'thumb_1024_url',
                'thumb_256_url']


def fetch_fresh_url(image_id, session, key_rotator):
    """Ask the Graph API for a current (signed) thumbnail URL.

    Returns (url, is_dead). is_dead=True means the image is permanently gone:
    either the id no longer resolves (400/404), or it resolves but the API
    serves no thumbnail of any size (the media is unavailable)."""
    key = key_rotator.next()
    if not key:
        return None, False
    url = (f'https://graph.mapillary.com/{image_id}'
           f'?access_token={key}&fields={",".join(THUMB_FIELDS)}')
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code in (400, 404):
            return None, True
        resp.raise_for_status()
        data = resp.json()
        for field in THUMB_FIELDS:
            if data.get(field):
                return data[field], False
        # 200 but no media of any size -> permanently unavailable.
        return None, True
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response is not None else None
        if code in (400, 404):
            return None, True
        return None, False
    except Exception:
        return None, False


def download_one(task, session, proxies, key_rotator, dead_set, dead_lock):
    image_id = task['image_id']
    target_dir = task['target_dir']
    filepath = os.path.join(target_dir, f"{image_id}.jpg")

    if os.path.exists(filepath) and is_valid_jpeg(filepath):
        return image_id, 'already_present', filepath
    with dead_lock:
        if image_id in dead_set:
            return image_id, 'dead', filepath

    os.makedirs(target_dir, exist_ok=True)

    def attempt(target_url):
        for _ in range(3 if proxies else 2):
            proxy_dict = None
            if proxies:
                p = random.choice(proxies)
                proxy_dict = {'http': p, 'https': p}
            try:
                resp = session.get(target_url, timeout=(5, 20),
                                   proxies=proxy_dict)
                resp.raise_for_status()
                tmp = filepath + '.part'
                with open(tmp, 'wb') as fh:
                    fh.write(resp.content)
                if resp.content[:2] == b'\xff\xd8':
                    os.replace(tmp, filepath)
                    return True
                os.remove(tmp)
            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else None
                if code in (400, 401, 403, 404):
                    return False  # signed-url expired/forbidden -> need fresh
            except Exception:
                pass
        return False

    # 1) Try the stored (possibly expired) URL.
    if task.get('stored_url') and attempt(task['stored_url']):
        return image_id, 'stored', filepath

    # 2) Fetch a fresh signed URL from the API and retry.
    fresh_url, is_dead = fetch_fresh_url(image_id, session, key_rotator)
    if is_dead:
        with dead_lock:
            dead_set.add(image_id)
        return image_id, 'dead', filepath
    if fresh_url and attempt(fresh_url):
        return image_id, 'recovered', filepath

    return image_id, 'failed', filepath


def load_manifest(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def cmd_download(args, tasks=None):
    if tasks is None:
        tasks = load_manifest(args.manifest)
    keys = load_api_keys()
    if not keys:
        print("[download] WARNING: no MLY_KEY* tokens found - expired URLs "
              "cannot be refreshed.")
    proxies = load_proxies(args.proxy_file)
    print(f"[download] {len(tasks):,} images | {len(keys)} API key(s) | "
          f"{len(proxies)} proxies | {args.workers} workers")

    key_rotator = KeyRotator(keys)
    dead_set = set()
    dead_lock = threading.Lock()

    # Resume: skip ids already recorded dead in a previous run.
    if args.dead_out and os.path.exists(args.dead_out):
        with open(args.dead_out) as f:
            for line in f:
                dead_set.add(line.strip())
        print(f"[download] loaded {len(dead_set):,} known-dead ids "
              f"(will be skipped).")

    counts = {'stored': 0, 'recovered': 0, 'already_present': 0,
              'dead': 0, 'failed': 0}
    failed_rows = []
    dead_fh = open(args.dead_out, 'a') if args.dead_out else None

    # Thread-local sessions so connections are reused per worker.
    tls = threading.local()

    def get_session():
        s = getattr(tls, 'session', None)
        if s is None:
            s = build_session()
            tls.session = s
        return s

    def work(task):
        return download_one(task, get_session(), proxies, key_rotator,
                            dead_set, dead_lock)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, t): t for t in tasks}
        pbar = tqdm(as_completed(futures), total=len(futures),
                    desc="Downloading")
        for fut in pbar:
            task = futures[fut]
            try:
                image_id, status, _ = fut.result()
            except Exception:
                status, image_id = 'failed', task['image_id']
            counts[status] = counts.get(status, 0) + 1
            if status == 'failed':
                failed_rows.append(task)
            elif status == 'dead' and dead_fh:
                dead_fh.write(f"{image_id}\n")
            pbar.set_postfix(ok=counts['stored'] + counts['recovered'],
                             dead=counts['dead'], fail=counts['failed'])

    if dead_fh:
        dead_fh.close()

    if failed_rows:
        with open(args.failed_out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(failed_rows[0].keys()))
            w.writeheader()
            w.writerows(failed_rows)

    print("\n[download] done.")
    print(f"    downloaded (stored url)  : {counts['stored']:,}")
    print(f"    recovered (fresh url)    : {counts['recovered']:,}")
    print(f"    already present          : {counts['already_present']:,}")
    print(f"    permanently dead (gone)  : {counts['dead']:,}")
    print(f"    still failing (transient): {counts['failed']:,}")
    if failed_rows:
        print(f"    -> retry these later: {args.failed_out}")
    if args.dead_out:
        print(f"    -> dead ids logged:  {args.dead_out}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)

    def add_scan_args(p):
        p.add_argument('grid_csv')
        p.add_argument('--dirs', nargs='+', required=True)
        p.add_argument('--region', default=None,
                       help='Restrict to one parent region (e.g. "Europe").')
        p.add_argument('--out', default='data/manifests/missing_manifest.csv')
        p.add_argument('-w', '--workers', type=int, default=16)

    def add_dl_args(p):
        p.add_argument('--proxy-file', default=None)
        p.add_argument('--dead-out', default='data/dead_images.txt',
                       help='Append-only log of permanently gone image_ids.')
        p.add_argument('--failed-out', default='data/manifests/still_missing.csv',
                       help='Manifest of transiently failed images to retry.')

    p_scan = sub.add_parser('scan', help='Find missing images, write manifest.')
    add_scan_args(p_scan)

    p_dl = sub.add_parser('download', help='Download from an existing manifest.')
    p_dl.add_argument('--manifest', required=True)
    p_dl.add_argument('-w', '--workers', type=int, default=24)
    add_dl_args(p_dl)

    p_all = sub.add_parser('all', help='Scan then download in one run.')
    add_scan_args(p_all)  # provides shared -w/--workers
    add_dl_args(p_all)

    args = parser.parse_args()

    if args.mode == 'scan':
        cmd_scan(args)
    elif args.mode == 'download':
        cmd_download(args)
    elif args.mode == 'all':
        tasks = cmd_scan(args)
        if tasks:
            print()
            cmd_download(args, tasks=tasks)


if __name__ == '__main__':
    main()
