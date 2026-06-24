"""
Sample the in-scope missing set and probe the Graph API to size the backfill
BEFORE building it.

Reads `coverage_missing_inscope/<Parent>.parquet` (the datefilter output), takes
a spread sample of image_ids (every k-th row, so it covers all cells, not just
the first), and for each fetches `/detections` to learn:
  * how many are still LIVE vs gone (400/404), and
  * what fraction are ground animals (a detection value == 'animal--ground-animal').
Then extrapolates those fractions to the full set.

    python validate_missing_sample.py --inscope coverage_missing_inscope \
        --region Europe -n 2000 [--proxies proxies.txt]
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__)))))

from coverage_audit import (KeyRotator, ProxyPool, build_session, load_keys,
                            make_progress)

GROUND = 'animal--ground-animal'


def resolve_files(inscope, region):
    """Resolve the inscope arg to a list of parquet files.

    Args:
        inscope: A parquet file or a directory of ``<Parent>.parquet``.
        region: Optional parent-region name to restrict to.

    Returns:
        List of existing parquet paths.
    """
    if os.path.isdir(inscope):
        if region:
            files = [os.path.join(inscope, f'{region}.parquet')]
        else:
            files = sorted(glob.glob(os.path.join(inscope, '*.parquet')))
    else:
        files = [inscope]
    return [f for f in files if os.path.exists(f)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inscope', default='coverage_missing_inscope')
    ap.add_argument('--region', default=None,
                    help='Parent region to sample (default: all parents).')
    ap.add_argument('-n', '--sample', type=int, default=2000)
    ap.add_argument('-w', '--workers', type=int, default=64)
    ap.add_argument('--env', default='.env')
    ap.add_argument('--proxies', default=None)
    args = ap.parse_args()

    files = resolve_files(args.inscope, args.region)
    if not files:
        raise SystemExit(f"no inscope parquet found under {args.inscope}")

    lf = pl.scan_parquet(files)
    total = int(lf.select(pl.len()).collect()['len'][0])
    if total == 0:
        raise SystemExit("inscope set is empty")
    stride = max(1, total // args.sample)
    ids = (lf.select('image_id').gather_every(stride).limit(
        args.sample).collect()['image_id'].to_list())
    n = len(ids)
    print(f"total in-scope {total:,} · sampling {n:,} (every {stride:,}th)")

    rot = KeyRotator(load_keys(args.env))
    session = build_session(pool=args.workers + 32)
    pool = ProxyPool(args.proxies) if args.proxies else None

    def check(iid):
        """Probe one image's detections; classify it.

        Returns:
            ``(status, is_ground_animal)`` with status in
            {'live', 'gone', 'error'}.
        """
        url = (f'https://graph.mapillary.com/{iid}/detections'
               f'?access_token={rot.next()}&fields=value')
        pid, proxies = (None, None)
        if pool is not None:
            pid, proxies = pool.acquire()
        try:
            r = session.get(url, timeout=15, proxies=proxies)
            if r.status_code in (400, 404):
                return 'gone', False
            if r.status_code != 200:
                return 'error', False
            data = r.json().get('data', [])
            return 'live', any(d.get('value') == GROUND for d in data)
        except Exception:
            return 'error', False
        finally:
            if pool is not None:
                pool.release(pid)

    live = gone = error = ga = 0
    with make_progress() as progress, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        task = progress.add_task("[bold]probing detections", total=n)
        for status, is_ga in (f.result()
                              for f in as_completed(ex.submit(check, i)
                                                    for i in ids)):
            if status == 'live':
                live += 1
                ga += int(is_ga)
            elif status == 'gone':
                gone += 1
            else:
                error += 1
            progress.advance(task)

    print(f"\nsample of {n:,}:")
    print(f"  live           {live:>7,}  ({live / n:6.1%})")
    print(f"  gone (400/404) {gone:>7,}  ({gone / n:6.1%})")
    print(f"  error/transient{error:>7,}  ({error / n:6.1%})")
    print(f"  ground-animal  {ga:>7,}  "
          f"({ga / n:6.1%} of sample · "
          f"{(ga / live if live else 0):6.1%} of live)")
    print(f"\nextrapolated to {total:,} in-scope:")
    print(f"  ~live           {round(total * live / n):>14,}")
    print(f"  ~ground-animals {round(total * ga / n):>14,}   "
          f"<- real backfill download volume")


if __name__ == '__main__':
    main()
