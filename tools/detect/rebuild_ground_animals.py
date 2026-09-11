#!/usr/bin/env python3
"""Rebuild a ground_animals parquet from its all_data counterpart.

    python tools/detect/rebuild_ground_animals.py scan
    python tools/detect/rebuild_ground_animals.py verify
    python tools/detect/rebuild_ground_animals.py rebuild --dry-run
    python tools/detect/rebuild_ground_animals.py rebuild

WHY THIS EXISTS. On 2026-09-09 two ground_animals files were overwritten by a
tool of this project's own making:

    grid_runs/East_Asia_135_35_140_40/ground_animals_..._backfill_s1_046.parquet
    grid_runs/South_Asia_70_20_75_25/ground_animals_..._014.parquet

The cause was a DuckDB binding rule. In

    copy (select ... from read_parquet(?) ...) to ?

the COPY TARGET binds parameter #1, not the last one, so passing
[source, sidecar, tmp] made the SOURCE the output file. Both files now hold
unrelated content. Roughly 27,987 rows were lost -- the difference between a
plan taken before the damage (32,049,453 ids) and one after (32,021,466) over
the same 33,170 files.

Every COPY in this file therefore inlines its target through _sql_path() and
passes only reads as parameters. That is not a style choice; it is the bug.

HOW A ground_animals FILE IS DERIVED. batch_chunks_mp_api.py writes
all_data_<cell>_NNN.parquet for every image in a part, then writes
ground_animals_<cell>_NNN.parquet as the subset whose image_id appears in
that run's animal set -- images carrying a detection valued
`animal--ground-animal`. Same schema, same order, zstd.

The detections column stored in all_data holds only detection IDs, so the
filter cannot be recomputed locally. It can be re-asked: the Graph API
returns the values in bulk via fields=id,detections{value}, ~100 ids per
request. That is what this tool does.

PROVE IT BEFORE TRUSTING IT. `verify` rebuilds a file that was never damaged
and compares it to what is on disk -- same ids, same rows, same column order.
If that does not come back identical, the method is wrong and `rebuild` must
not be run.
"""

import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / 'data' / 'catalog.duckdb'

API = 'https://graph.mapillary.com/images'
FIELDS = 'id,detections{value}'
ANIMAL = 'animal--ground-animal'
BATCH = 100
WORKERS = 64

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_camera_params import (
    AllKeysDead,
    ApiError,
    KeyPool,  # noqa: E402
    _bar,
    _err_from_body,
    _pyarrow,
    _sql_path,
    keys)


def _duckdb():
    try:
        import duckdb
    except ImportError:
        sys.exit('duckdb missing -- run this under the mp14 env')
    return duckdb


def harvest_schema(con, path):
    return [
        c[0] for c in con.execute("select * from read_parquet(?) limit 0",
                                  [path]).description
    ]


def scan(args):
    """Report files whose schema is not their cell's schema. Exit 0 either way.

    Separate from _find_damaged so the subcommand returns nothing: main()
    hands its result to sys.exit, and a returned list would both print itself
    and exit non-zero on a clean repo.
    """
    _find_damaged()


def _find_damaged():
    duckdb = _duckdb()
    cat = duckdb.connect(str(CATALOG), read_only=True)
    rows = cat.execute(
        "select f.path from files f join drives d "
        "  on d.drive = f.drive and d.online "
        "where f.kind = 'ground_animals' order by f.path").fetchall()
    con = duckdb.connect()
    expected = None
    bad = []
    for (p, ) in rows:
        cols = harvest_schema(con, p)
        if expected is None:
            expected = cols
        if cols != expected:
            bad.append((p, cols))
    print(f'{len(rows):,} ground_animals files, expected schema {expected}')
    if not bad:
        print('all schemas match -- nothing damaged')
        return []
    print(f'\n{len(bad)} file(s) do NOT match:')
    for p, cols in bad:
        n = con.execute("select count(*) from read_parquet(?)",
                        [p]).fetchone()[0]
        ad = counterpart(p)
        print(f'  {Path(p).name}')
        print(f'    columns  : {cols}')
        print(f'    rows     : {n:,}')
        print(
            f'    all_data : {"present" if Path(ad).exists() else "MISSING"}')
    return [p for p, _ in bad]


def counterpart(ga_path):
    """The all_data file a ground_animals file was filtered out of."""
    p = Path(ga_path)
    return str(p.with_name(p.name.replace('ground_animals_', 'all_data_', 1)))


def fetch_animal_ids(ids, pool, desc):
    """Which of these image_ids carry an animal--ground-animal detection.

    Returns (animal_ids, answered_count). An id the API does not answer for is
    NOT silently treated as "no animal" -- the caller compares answered against
    requested and refuses to write a file built on an incomplete answer.
    """
    chunks = [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]
    animals, answered = set(), 0
    lock = threading.Lock()

    def one(k, chunk):
        q = urllib.parse.urlencode({
            'image_ids': ','.join(chunk),
            'fields': FIELDS
        })
        for attempt in range(6):
            ki = pool.pick(k + attempt)
            req = urllib.request.Request(
                f'{API}?{q}',
                headers={'Authorization': f'OAuth {pool.key(ki)}'})
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    body = json.loads(r.read().decode())
                if 'data' not in body:
                    raise _err_from_body(200, json.dumps(body))
                return body['data']
            except urllib.error.HTTPError as e:
                err = _err_from_body(e.code, e.read().decode(errors='replace'))
                if err.is_auth and pool.retire(ki) == 0:
                    raise AllKeysDead(err.message)
            except ApiError as e:
                if e.is_auth and pool.retire(ki) == 0:
                    raise AllKeysDead(e.message)
            except Exception:
                pass
        return None

    bar = _bar(len(chunks), desc)
    try:
        with ThreadPoolExecutor(WORKERS) as ex:
            futs = [ex.submit(one, k, c) for k, c in enumerate(chunks)]
            for f in as_completed(futs):
                got = f.result()
                bar.update(1)
                if got is None:
                    continue
                with lock:
                    answered += len(got)
                    for d in got:
                        dets = (d.get('detections') or {}).get('data') or []
                        if any(x.get('value') == ANIMAL for x in dets):
                            animals.add(str(d['id']))
    finally:
        bar.close()
    return animals, answered


def build(con, pool, ga_path, out_path, label):
    """Write the ground_animals subset of ga_path's all_data counterpart."""
    ad = counterpart(ga_path)
    if not Path(ad).exists():
        return None, f'no all_data counterpart at {Path(ad).name}'
    ids = [
        str(r[0]) for r in con.execute("select image_id from read_parquet(?)",
                                       [ad]).fetchall()
    ]
    animals, answered = fetch_animal_ids(ids, pool, label)
    if answered < len(ids):
        return None, (f'API answered for {answered:,} of {len(ids):,} ids -- '
                      'an incomplete answer would silently drop animals, so '
                      'nothing was written')
    cols = harvest_schema(con, ad)
    pa = _pyarrow()
    tb = pa.table({'image_id': pa.array(sorted(animals), type=pa.string())})
    con.register('keep', tb)
    tmp = Path(out_path).with_name('.' + Path(out_path).name + '.rebuild')
    try:
        con.execute(
            f'copy (select {", ".join(cols)} from read_parquet(?) '
            'where image_id in (select image_id from keep)) '
            f'to {_sql_path(tmp)} (format parquet, compression zstd)', [ad])
        n = con.execute("select count(*) from read_parquet(?)",
                        [str(tmp)]).fetchone()[0]
        os.replace(tmp, out_path)
    finally:
        con.unregister('keep')
        if tmp.exists():
            tmp.unlink()
    return n, None


def verify(args):
    """Rebuild an UNDAMAGED file and prove it comes back identical."""
    duckdb = _duckdb()
    cat = duckdb.connect(str(CATALOG), read_only=True)
    con = duckdb.connect()
    pool = KeyPool(keys())
    truth = args.file or cat.execute(
        "select f.path from files f join drives d "
        "  on d.drive = f.drive and d.online "
        "where f.kind = 'ground_animals' and f.n_rows between 50 and 400 "
        "order by f.n_rows limit 1").fetchone()[0]
    print(f'control file: {Path(truth).name}')
    out = REPO / 'data' / 'camera' / '.verify_rebuild.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    n, err = build(con, pool, truth, out, 'verify')
    if err:
        sys.exit(f'could not rebuild the control: {err}')
    cmp_ = con.execute(
        "select (select count(*) from read_parquet(?)) as truth_rows, "
        "       (select count(*) from read_parquet(?)) as rebuilt_rows, "
        "       (select count(*) from (select image_id from read_parquet(?) "
        "          except select image_id from read_parquet(?))) as missing, "
        "       (select count(*) from (select image_id from read_parquet(?) "
        "          except select image_id from read_parquet(?))) as extra",
        [truth, str(out), truth,
         str(out), str(out), truth]).fetchone()
    print(f'  truth {cmp_[0]:,} rows   rebuilt {cmp_[1]:,} rows')
    print(f'  ids missing from rebuild : {cmp_[2]}')
    print(f'  ids the rebuild invented : {cmp_[3]}')
    same_cols = harvest_schema(con, truth) == harvest_schema(con, str(out))
    print(f'  column order identical   : {same_cols}')
    out.unlink()
    ok = cmp_[0] == cmp_[1] and cmp_[2] == 0 and cmp_[3] == 0 and same_cols
    print('\n' + ('METHOD VERIFIED -- rebuild is safe to run'
                  if ok else 'METHOD FAILED -- do NOT run rebuild'))
    return 0 if ok else 1


def rebuild(args):
    duckdb = _duckdb()
    con = duckdb.connect()
    pool = KeyPool(keys())
    targets = args.file and [args.file] or _find_damaged()
    if not targets:
        return
    print(f'\n{len(targets)} file(s) to rebuild' +
          ('  (DRY RUN)' if args.dry_run else ''))
    for p in targets:
        if args.dry_run:
            ad = counterpart(p)
            n = con.execute("select count(*) from read_parquet(?)",
                            [ad]).fetchone()[0]
            print(f'  {Path(p).name}: would query {n:,} ids from '
                  f'{Path(ad).name}')
            continue
        n, err = build(con, pool, p, p, Path(p).name[:28])
        if err:
            print(f'  FAILED {Path(p).name}: {err}')
        else:
            print(f'  rebuilt {Path(p).name}: {n:,} rows')
    if not args.dry_run:
        print('\nrun tools/catalog/catalog.py refresh -- sizes changed')


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('scan', help='find files whose schema is wrong')
    s.set_defaults(fn=scan)
    v = sub.add_parser('verify', help='rebuild an undamaged file and compare')
    v.add_argument('--file', help='control file to use')
    v.set_defaults(fn=verify)
    r = sub.add_parser('rebuild', help='rebuild the damaged files')
    r.add_argument('--file', help='rebuild just this one')
    r.add_argument('--dry-run', action='store_true')
    r.set_defaults(fn=rebuild)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
