"""
A DuckDB-backed inventory of every Parquet file across all data drives.

Tracks one row per Parquet file (path, drive, kind, region, bbox, row count,
size, mtime) so you can answer "what do we have, where, how much" instantly --
even when a drive is offline -- without re-globbing tens of thousands of files.

READ-ONLY on the data: it reads only Parquet footer metadata (row counts, never
a full scan) and the directory layout. Region and bounding box come for free
from the cell directory name (e.g. ``Europe_10_45_15_50``). The only thing it
writes is its own catalog (``--db`` DuckDB file + a ``catalog.parquet`` snapshot).

    python tools/catalog/catalog.py refresh           # build / update the catalog
    python tools/catalog/catalog.py summary           # totals by drive / kind / region
    python tools/catalog/catalog.py sql "SELECT region, sum(n_rows) FROM files
                                          WHERE kind='ground_animals' GROUP BY 1
                                          ORDER BY 2 DESC LIMIT 20"

Refresh is incremental: a file is re-read only when its size or mtime changed.
Drives that aren't mounted are left untouched (their rows stay in the catalog,
marked offline); files deleted from a mounted drive are pruned.

Which roots to scan: pass ``--dirs <path> ...``, or list them (one per line) in
a local ``data/catalog_dirs.txt`` (gitignored; override with ``--dirs-file``).
No drive paths are hardcoded here -- the default is just ``grid_runs``.
"""

import argparse
import functools
import os
import re
from datetime import datetime

import duckdb

DEFAULT_DIRS = ['grid_runs']
DIRS_FILE = 'data/catalog_dirs.txt'

CELL_RE = re.compile(
    r'^(?P<region>.+?)_(?P<sw_lon>-?\d+(?:\.\d+)?)_(?P<sw_lat>-?\d+(?:\.\d+)?)'
    r'_(?P<ne_lon>-?\d+(?:\.\d+)?)_(?P<ne_lat>-?\d+(?:\.\d+)?)$')

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
  path VARCHAR PRIMARY KEY, drive VARCHAR, kind VARCHAR, variant VARCHAR,
  region VARCHAR, cell VARCHAR,
  sw_lon DOUBLE, sw_lat DOUBLE, ne_lon DOUBLE, ne_lat DOUBLE,
  n_rows BIGINT, size_bytes BIGINT, mtime DOUBLE, indexed_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS drives (
  drive VARCHAR, base_path VARCHAR PRIMARY KEY, online BOOLEAN,
  n_files BIGINT, n_rows BIGINT, last_refresh TIMESTAMP
);
CREATE TABLE IF NOT EXISTS images (
  cell VARCHAR, drive VARCHAR, region VARCHAR,
  n_images BIGINT, bytes BIGINT, scanned_at TIMESTAMP, mtime DOUBLE,
  PRIMARY KEY (cell, drive)
);
CREATE TABLE IF NOT EXISTS cell_images (
  cell VARCHAR PRIMARY KEY, region VARCHAR,
  n_unique BIGINT, n_drives INT, deduped BOOLEAN, updated TIMESTAMP
);
"""


def resolve_dirs(args):
    """Resolve which grid_runs roots to scan, without hardcoding any paths.

    Precedence: an explicit ``--dirs`` wins; otherwise the lines of
    ``--dirs-file`` (a gitignored local config, one path per line, ``#``
    comments allowed) if it exists; otherwise the generic ``DEFAULT_DIRS``.

    Args:
        args: Parsed args carrying ``dirs`` and ``dirs_file``.

    Returns:
        A list of base directory paths.
    """
    if args.dirs:
        return args.dirs
    path = getattr(args, 'dirs_file', None)
    if path and os.path.exists(path):
        with open(path) as f:
            dirs = [ln.strip() for ln in f
                    if ln.strip() and not ln.startswith('#')]
        if dirs:
            return dirs
    return DEFAULT_DIRS


@functools.lru_cache(maxsize=None)
def drive_of(path):
    """Return an env-agnostic label for the filesystem a path lives on.

    Walks up to the path's mount point (where the device id changes) and
    returns that directory's name, so files group "by drive" without any
    hardcoded mount layout. Falls back to the path's own basename if it
    can't be stat'd (e.g. an offline drive).

    Args:
        path: Any path on the drive.

    Returns:
        The mount-point directory name (or a basename fallback).
    """
    try:
        p = os.path.realpath(path)
        dev = os.stat(p).st_dev
        while True:
            parent = os.path.dirname(p)
            if parent == p or os.stat(parent).st_dev != dev:
                return os.path.basename(p) or '?'
            p = parent
    except OSError:
        return os.path.basename(os.path.normpath(path)) or '?'


def classify(fname):
    """Return ``(kind, variant)`` parsed from a Parquet file name."""
    kind = ('all_data' if fname.startswith('all_data_') else
            'ground_animals' if fname.startswith('ground_animals_') else 'other')
    variant = ('backfill' if '_backfill_' in fname else
               'recovered' if '_recovered_' in fname else 'main')
    return kind, variant


def parse_cell(cell):
    """Return ``(region, sw_lon, sw_lat, ne_lon, ne_lat)`` from a cell dir name."""
    m = CELL_RE.match(cell)
    if not m:
        return cell, None, None, None, None
    g = m.group
    return (g('region'), float(g('sw_lon')), float(g('sw_lat')),
            float(g('ne_lon')), float(g('ne_lat')))


def scan_dir(base):
    """Yield ``(path, size, mtime)`` for every ``<base>/<cell>/*.parquet``."""
    with os.scandir(base) as cells:
        for cell in cells:
            if not cell.is_dir():
                continue
            try:
                with os.scandir(cell.path) as it:
                    for e in it:
                        if e.name.endswith('.parquet'):
                            st = e.stat()
                            yield e.path, st.st_size, st.st_mtime
            except OSError:
                continue


def scan_images(base, with_size, prev=None):
    """Yield ``(cell, n_images, bytes, mtime)`` per cell's ground_animal_images.

    Incremental: a cell whose ``ground_animal_images`` dir mtime matches its
    previously-scanned mtime is reused without re-reading its files -- adding or
    removing a jpg always bumps the dir mtime, so counts stay correct. This
    turns a full tens-of-millions-of-files readdir into one ``stat`` per cell
    plus a readdir of only the cells that changed. ``prev`` is
    ``{cell: (n_images, bytes, mtime)}`` from the last scan.

    Byte totals require a stat per file and are gathered only when ``with_size``.
    """
    prev = prev or {}
    with os.scandir(base) as cells:
        for cell in cells:
            if not cell.is_dir():
                continue
            imgdir = os.path.join(cell.path, 'ground_animal_images')
            try:
                mt = os.stat(imgdir).st_mtime
            except OSError:
                continue
            p = prev.get(cell.name)
            if p and p[2] == mt and (not with_size or p[1] is not None):
                if p[0]:                       # unchanged since last scan -> reuse
                    yield cell.name, p[0], p[1], mt
                continue
            try:
                n = b = 0
                with os.scandir(imgdir) as it:
                    for e in it:
                        if e.name.endswith('.jpg'):
                            n += 1
                            if with_size:
                                b += e.stat().st_size
                if n:
                    yield cell.name, n, (b if with_size else None), mt
            except OSError:
                continue


def num_rows_batch(con, paths, chunk=4000):
    """Map each path to its Parquet footer row count (no data scan)."""
    out = {}
    for i in range(0, len(paths), chunk):
        part = paths[i:i + chunk]
        for fn, nr in con.execute(
                "SELECT file_name, num_rows FROM parquet_file_metadata(?)",
                [part]).fetchall():
            out[fn] = nr
        print(f"    read footers {min(i + chunk, len(paths)):,}/{len(paths):,}",
              end='\r', flush=True)
    if paths:
        print()
    return out


def cmd_refresh(args):
    """Scan the data dirs and build/update the catalog incrementally."""
    dirs = resolve_dirs(args)
    con = duckdb.connect(args.db)
    con.execute(SCHEMA)
    existing = {p: (s, m) for p, s, m in
                con.execute("SELECT path, size_bytes, mtime FROM files")
                .fetchall()}
    seen, changed = set(), []
    online_drives = set()
    now = datetime.now()

    for base in dirs:
        if not os.path.isdir(base):
            dr = (con.execute("SELECT any_value(drive) FROM files "
                              "WHERE path LIKE ?", [base + '%']).fetchone()[0]
                  or os.path.basename(base.rstrip('/')) or '?')
            print(f"  ⦸ offline: {base}")
            con.execute(
                "INSERT OR REPLACE INTO drives VALUES (?,?,?,?,?,?)",
                [dr, base, False,
                 con.execute("SELECT count(*) FROM files WHERE path LIKE ?",
                             [base + '%']).fetchone()[0],
                 con.execute("SELECT coalesce(sum(n_rows),0) FROM files "
                             "WHERE path LIKE ?", [base + '%']).fetchone()[0],
                 now])
            continue
        dr = drive_of(base)
        online_drives.add(dr)
        print(f"  scanning {base} ...")
        n_here = 0
        for path, size, mtime in scan_dir(base):
            seen.add(path)
            n_here += 1
            if existing.get(path) != (size, mtime):
                changed.append((path, size, mtime))
        print(f"    {n_here:,} parquet files ({len(changed):,} new/changed so far)")

    if changed:
        print(f"  reading footers for {len(changed):,} new/changed files ...")
        rows = num_rows_batch(con, [c[0] for c in changed])
        recs = []
        for path, size, mtime in changed:
            fname = os.path.basename(path)
            cell = os.path.basename(os.path.dirname(path))
            kind, variant = classify(fname)
            region, a, b, c, d = parse_cell(cell)
            recs.append((path, drive_of(os.path.dirname(path)), kind, variant,
                         region, cell, a, b, c, d, rows.get(path), size, mtime,
                         now))
        con.executemany("DELETE FROM files WHERE path = ?",
                        [(r[0],) for r in recs])
        con.executemany(
            "INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", recs)

    con.execute("CREATE OR REPLACE TEMP TABLE _seen(path VARCHAR)")
    if seen:
        con.executemany("INSERT INTO _seen VALUES (?)", [(p,) for p in seen])
    if online_drives:
        ph = ','.join('?' * len(online_drives))
        pruned = con.execute(
            f"SELECT count(*) FROM files WHERE drive IN ({ph}) "
            "AND path NOT IN (SELECT path FROM _seen)",
            list(online_drives)).fetchone()[0]
        con.execute(
            f"DELETE FROM files WHERE drive IN ({ph}) "
            "AND path NOT IN (SELECT path FROM _seen)", list(online_drives))
        if pruned:
            print(f"  pruned {pruned:,} files removed from mounted drives")

    for base in dirs:
        if os.path.isdir(base):
            dr = drive_of(base)
            nf, nr = con.execute(
                "SELECT count(*), coalesce(sum(n_rows),0) FROM files "
                "WHERE path LIKE ?", [base + '%']).fetchone()
            con.execute("INSERT OR REPLACE INTO drives VALUES (?,?,?,?,?,?)",
                        [dr, base, True, nf, nr, now])

    snap = os.path.join(os.path.dirname(os.path.abspath(args.db)) or '.',
                        'catalog.parquet')
    con.execute(f"COPY (SELECT * FROM files) TO '{snap}' (FORMAT parquet)")
    tot = con.execute("SELECT count(*), coalesce(sum(n_rows),0) FROM files"
                      ).fetchone()
    con.close()
    print(f"\n✓ catalog: {tot[0]:,} files · {tot[1]:,} rows · {args.db} "
          f"(+ snapshot {snap})")


def cmd_images(args):
    """Inventory downloaded jpgs per cell (counts; byte totals with --with-size).

    Incremental: cells whose image dir is unchanged since the last scan are
    reused via their stored mtime, so a re-scan is seconds instead of minutes.
    """
    dirs = resolve_dirs(args)
    con = duckdb.connect(args.db)
    con.execute(SCHEMA)
    try:
        con.execute("ALTER TABLE images ADD COLUMN IF NOT EXISTS mtime DOUBLE")
    except Exception:
        pass
    now = datetime.now()
    online_drives, seen = set(), set()
    for base in dirs:
        if not os.path.isdir(base):
            print(f"  ⦸ offline: {base}")
            continue
        d = drive_of(base)
        online_drives.add(d)
        prev = {r[0]: (r[1], r[2], r[3]) for r in con.execute(
            "SELECT cell, n_images, bytes, mtime FROM images WHERE drive=?",
            [d]).fetchall()}
        print(f"  scanning images under {base} ...")
        recs, n, reused = [], 0, 0
        for cell, cnt, b, mt in scan_images(base, args.with_size, prev):
            region = parse_cell(cell)[0]
            recs.append((cell, d, region, cnt, b, now, mt))
            seen.add((cell, d))
            n += cnt
            pc = prev.get(cell)
            if pc and pc[2] == mt:
                reused += 1
        if recs:
            con.executemany("DELETE FROM images WHERE cell=? AND drive=?",
                            [(r[0], r[1]) for r in recs])
            con.executemany("INSERT INTO images VALUES (?,?,?,?,?,?,?)", recs)
        rs = f" ({reused:,} reused via mtime)" if reused else ""
        print(f"    {len(recs):,} cells · {n:,} images{rs}")
    if online_drives and seen:
        con.execute("CREATE OR REPLACE TEMP TABLE _seen(cell VARCHAR, drive VARCHAR)")
        con.executemany("INSERT INTO _seen VALUES (?,?)", list(seen))
        ph = ','.join('?' * len(online_drives))
        con.execute(
            f"DELETE FROM images WHERE drive IN ({ph}) AND (cell,drive) NOT IN "
            "(SELECT cell,drive FROM _seen)", list(online_drives))
    # deduped per-cell totals: a cell's images can be split across drives
    # (complementary backfill) or duplicated (subset re-download). Sum the
    # per-drive counts, then correct any cell on >1 ONLINE drive by unioning
    # its actual image IDs from disk (cheap — only a handful of cells collide).
    root_by_drive = {drive_of(b): b for b in dirs if os.path.isdir(b)}
    con.execute("DELETE FROM cell_images")
    con.execute(
        "INSERT INTO cell_images SELECT cell, any_value(region), "
        "sum(n_images), count(*), count(*)=1, ? FROM images GROUP BY cell",
        [now])
    collided = con.execute("SELECT cell, list(drive) FROM images GROUP BY cell "
                           "HAVING count(*) > 1").fetchall()
    fixed = 0
    for cell, drvs in collided:
        if not all(d in root_by_drive for d in drvs):
            continue  # an involved drive is offline; keep the summed estimate
        ids = set()
        for d in drvs:
            p = os.path.join(root_by_drive[d], cell, 'ground_animal_images')
            try:
                ids |= {e.name for e in os.scandir(p) if e.name.endswith('.jpg')}
            except OSError:
                pass
        con.execute("UPDATE cell_images SET n_unique=?, deduped=true "
                    "WHERE cell=?", [len(ids), cell])
        fixed += 1
    if collided:
        print(f"  deduped {fixed}/{len(collided)} cross-drive cells "
              f"(union of image IDs)")
    tot = con.execute("SELECT count(*), coalesce(sum(n_images),0), "
                      "coalesce(sum(bytes),0) FROM images").fetchone()
    con.close()
    sz = f" · {tot[2]/1e12:.2f} TB" if args.with_size and tot[2] else ""
    print(f"\n✓ images: {tot[1]:,} jpgs across {tot[0]:,} cells{sz}")


def _table(rows, headers):
    """Print a simple aligned text table."""
    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    w = [max(len(str(x)) for x in c) for c in cols]
    line = lambda r: '  '.join(str(x).ljust(w[i]) for i, x in enumerate(r))
    print('  ' + line(headers))
    print('  ' + '  '.join('-' * x for x in w))
    for r in rows:
        print('  ' + line(r))


def cmd_summary(args):
    """Print headline totals and breakdowns by drive, kind, and region."""
    con = duckdb.connect(args.db, read_only=True)
    nf, nr = con.execute(
        "SELECT count(*), coalesce(sum(n_rows),0) FROM files").fetchone()
    ad = con.execute("SELECT coalesce(sum(n_rows),0) FROM files "
                     "WHERE kind='all_data'").fetchone()[0]
    ga = con.execute("SELECT coalesce(sum(n_rows),0) FROM files "
                     "WHERE kind='ground_animals'").fetchone()[0]
    nregions = con.execute("SELECT count(DISTINCT region) FROM files"
                           ).fetchone()[0]
    print(f"\n{nf:,} parquet files · {nr:,} rows · {nregions} regions")
    print(f"  all_data rows (every image): {ad:,}")
    print(f"  ground_animals rows:         {ga:,}\n")

    fmt = lambda rows: [[f"{x:,}" if isinstance(x, int) else x for x in r]
                        for r in rows]

    print("by drive:")
    _table(fmt(con.execute(
        "WITH ds AS (SELECT drive, bool_or(online) online FROM drives "
        "GROUP BY drive) "
        "SELECT f.drive, ds.online, count(*) files, sum(f.n_rows) total_rows "
        "FROM files f LEFT JOIN ds USING(drive) "
        "GROUP BY f.drive, ds.online ORDER BY total_rows DESC NULLS LAST"
    ).fetchall()), ['drive', 'online', 'files', 'rows'])

    print("\nby kind / variant:")
    _table(fmt(con.execute(
        "SELECT kind, variant, count(*) files, sum(n_rows) total_rows FROM files "
        "GROUP BY 1,2 ORDER BY total_rows DESC NULLS LAST").fetchall()),
        ['kind', 'variant', 'files', 'rows'])

    print(f"\ntop {args.top} regions by ground-animal rows:")
    _table(fmt(con.execute(
        "SELECT region, sum(n_rows) FILTER (WHERE kind='ground_animals') ga, "
        "sum(n_rows) FILTER (WHERE kind='all_data') all_data, count(*) files "
        "FROM files GROUP BY region ORDER BY ga DESC NULLS LAST LIMIT ?",
        [args.top]).fetchall()),
        ['region', 'ground_animals', 'all_data', 'files'])

    img = con.execute("SELECT count(*), coalesce(sum(n_images),0), "
                      "coalesce(sum(bytes),0) FROM images").fetchone()
    if img[1]:
        sz = f" · {img[2]/1e12:.2f} TB" if img[2] else ""
        print(f"\ndownloaded images: {img[1]:,} jpgs across {img[0]:,} "
              f"cells{sz}")
        _table(fmt(con.execute(
            "SELECT drive, count(*) cells, sum(n_images) imgs FROM images "
            "GROUP BY 1 ORDER BY imgs DESC").fetchall()),
            ['drive', 'cells', 'images'])
    con.close()


def cmd_sql(args):
    """Run an arbitrary SQL query against the catalog and print the result."""
    con = duckdb.connect(args.db, read_only=True)
    con.sql(args.query).show(max_rows=args.limit)
    con.close()


def main():
    """Parse the CLI and dispatch to refresh / summary / sql."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db', default='data/catalog.duckdb',
                   help='Catalog DuckDB file (default: data/catalog.duckdb).')
    sub = p.add_subparsers(dest='cmd', required=True)

    def add_dirs(sp):
        """Add the shared --dirs / --dirs-file options to a subparser."""
        sp.add_argument('--dirs', nargs='+', default=None,
                        help='grid_runs roots to scan. Default: lines of '
                        '--dirs-file if it exists, else "grid_runs".')
        sp.add_argument('--dirs-file', default=DIRS_FILE,
                        help=f'File of grid_runs roots, one per line '
                        f'(gitignored local config; default {DIRS_FILE}).')

    r = sub.add_parser('refresh', help='Build/update the catalog.')
    add_dirs(r)
    r.set_defaults(func=cmd_refresh)

    im = sub.add_parser('images', help='Inventory downloaded jpgs per cell.')
    add_dirs(im)
    im.add_argument('--with-size', action='store_true',
                    help='Also sum bytes (a stat per file; slower).')
    im.set_defaults(func=cmd_images)

    s = sub.add_parser('summary', help='Print totals and breakdowns.')
    s.add_argument('--top', type=int, default=20)
    s.set_defaults(func=cmd_summary)

    q = sub.add_parser('sql', help='Run SQL against the catalog.')
    q.add_argument('query')
    q.add_argument('--limit', type=int, default=40)
    q.set_defaults(func=cmd_sql)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
