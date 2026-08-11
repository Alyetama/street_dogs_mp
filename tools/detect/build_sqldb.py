#!/usr/bin/env python3
"""Fold the parquet predictions store into one queryable DuckDB file.

The hive-partitioned parquet store stays AUTHORITATIVE. The sweep appends to
it, it survives a kill -9 mid-shard, and every guarantee in the design doc is
about it. This file is a DERIVED copy: convenient to query, to index, to hand
to another tool, to copy to another machine -- and stale the moment the sweep
writes another part.

That distinction is the whole risk here, so the database states it about
itself: `_meta` records what it was built from and when, `sweep_status()`
compares its counts against the live store, and every read path prints how far
behind it is. A derived table that looks authoritative is the failure this
project keeps meeting.

What it adds over reading the parquet directly:

  ONE FILE.        No 6,896-way glob, no union_by_name, no remembering which
                   column exists in which generation of the schema.
  REAL TIMESTAMPS. The store records ts_off -- seconds since its own run
                   began -- which is uninterpretable alone. Joined against the
                   run manifests it becomes an absolute time, so "what did the
                   sweep find on Tuesday afternoon" is a WHERE clause.
  PROVENANCE.      The manifests come in as a `runs` table, so a box can be
                   filtered by the model that drew it, and by whether that
                   model was MEASURED or merely attested.

Incremental by part file: a refresh reads only parts it has not seen, keyed on
(path, size, mtime), and every row records the part it came from in `src`.
A part's CONTENT is immutable once committed, but its existence is not --
store.compact() deletes a cell's parts once the merged pair verifies, and
store.tiling_resume(repair=True) deletes a torn or overlapping part for the
run to redo. So a refresh deletes by `src` before it re-reads a changed path,
and drops the rows of any part that is no longer on disk. Without that the
same images are folded in twice and `verify` reports the store has lost rows.

    python tools/detect/build_sqldb.py build      # create or refresh
    python tools/detect/build_sqldb.py verify     # counts vs the parquet store
    python tools/detect/build_sqldb.py info       # what is in it, how stale

READ-ONLY on the predictions store. Safe to run while the sweep is running.
"""

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

SCHEMA_VERSION = 2
BATCH = 400          # part files per INSERT; keeps the SQL text sane


def detect_root():
    import store
    return store.get_detect_root()


def default_db(root):
    """Beside the store it mirrors, not in the repo: it is data, and it is big
    enough and regenerable enough that it has no business in git."""
    return os.path.join(root, 'sweep.duckdb')


def part_files(root, kind):
    return sorted(glob.glob(os.path.join(
        root, 'shards', 'gen=*', 'region=*', 'cell=*', 'drive=*',
        f'*.{kind}.parquet')))


def _hive(path, root):
    """(gen, region, cell, drive) from the partition directories."""
    rel = os.path.relpath(path, os.path.join(root, 'shards'))
    got = {}
    for piece in rel.split(os.sep)[:-1]:
        if '=' in piece:
            k, _, v = piece.partition('=')
            got[k] = v
    return (got.get('gen'), got.get('region'), got.get('cell'),
            got.get('drive'))


# NOTE ON `drive`. The img parquet carries a numeric drive code AND its parts
# live under a drive=<name> hive partition. Same name, so with
# hive_partitioning=1 the PARTITION shadows the file column: `drive` reads as
# 'bobcat', never as the code, and the numeric column is unreachable through a
# hive read. The name is the more useful of the two and it is what every
# existing reader of this store already sees, so this mirrors that rather than
# giving one name two meanings.
#
# `src` IS NOT DECORATION. It is the part file each row came from, and it is
# the only thing that makes a refresh reversible: without it nothing can undo
# a load, so a part that is reloaded (same path, new bytes) or that vanishes
# (compaction, or tiling_resume dropping a torn range for the run to redo)
# leaves its old rows behind forever and the table double-counts. It is filled
# from duckdb's read_parquet(filename=1) and never from the row data.
SRC_COL = ('src', 'VARCHAR')

IMG_COLS = (('image_id', 'UBIGINT'), ('gen', 'VARCHAR'), ('region', 'VARCHAR'),
            ('cell', 'VARCHAR'), ('drive', 'VARCHAR'),
            ('run_id', 'USMALLINT'), ('model_sha8', 'VARCHAR'),
            ('status', 'UTINYINT'), ('n_det', 'USMALLINT'),
            ('max_conf', 'FLOAT'), ('orig_w', 'USMALLINT'),
            ('orig_h', 'USMALLINT'), ('reduce', 'UTINYINT'),
            ('guards', 'USMALLINT'), ('ts_off', 'UINTEGER'),
            ('shard_idx', 'UINTEGER'), SRC_COL)

DET_COLS = (('image_id', 'UBIGINT'), ('det_idx', 'UTINYINT'),
            ('gen', 'VARCHAR'), ('region', 'VARCHAR'), ('cell', 'VARCHAR'),
            ('drive', 'VARCHAR'), ('run_id', 'USMALLINT'),
            ('model_sha8', 'VARCHAR'), ('conf', 'FLOAT'), ('x1', 'FLOAT'),
            ('y1', 'FLOAT'), ('x2', 'FLOAT'), ('y2', 'FLOAT'),
            ('leash_class', 'UTINYINT'), ('leash_conf', 'FLOAT'),
            ('shard_idx', 'UINTEGER'), SRC_COL)


def ddl(con):
    con.execute('CREATE TABLE IF NOT EXISTS images ('
                + ', '.join(f'{n} {t}' for n, t in IMG_COLS) + ')')
    con.execute('CREATE TABLE IF NOT EXISTS detections ('
                + ', '.join(f'{n} {t}' for n, t in DET_COLS) + ')')
    # the incremental ledger: what has already been folded in
    con.execute('CREATE TABLE IF NOT EXISTS _files ('
                'path VARCHAR PRIMARY KEY, kind VARCHAR, size BIGINT, '
                'mtime DOUBLE, rows BIGINT, loaded_at TIMESTAMP)')
    con.execute('CREATE TABLE IF NOT EXISTS _meta ('
                'key VARCHAR PRIMARY KEY, value VARCHAR)')
    con.execute('CREATE TABLE IF NOT EXISTS runs ('
                'gen VARCHAR, run_id USMALLINT, provenance_class VARCHAR, '
                'model_sha8 VARCHAR, comet_run VARCHAR, comet_key VARCHAR, '
                'comet_project VARCHAR, run_started TIMESTAMP, '
                'run_started_epoch BIGINT, attested_by VARCHAR, '
                'corroborating_detections BIGINT, conflicting_detections BIGINT,'
                ' manifest VARCHAR)')


def meta_set(con, key, value):
    con.execute('INSERT INTO _meta VALUES (?, ?) ON CONFLICT (key) DO UPDATE '
                'SET value = excluded.value', [key, str(value)])


def meta_get(con, key, default=None):
    r = con.execute('SELECT value FROM _meta WHERE key = ?', [key]).fetchone()
    return r[0] if r else default


def load_runs(con, root):
    """Manifests -> the `runs` table. Rewritten whole: there are eleven of
    them, and a partial update is more code than a rebuild."""
    con.execute('DELETE FROM runs')
    n = 0
    for f in sorted(glob.glob(os.path.join(root, 'runs', 'gen=*',
                                           'run_*.json'))):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            print(f'WARNING: unreadable manifest {f}', file=sys.stderr)
            continue
        m = d.get('model') or {}
        am = d.get('attested_model') or {}
        at = d.get('attestation') or {}
        c = d.get('corroboration') or {}
        started = d.get('run_started_epoch')
        con.execute(
            'INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
            [f'{int(d["gen"]):04d}', int(d['run_id']),
             d.get('provenance_class'),
             m.get('sha8'),
             # the measured name if there is one, else the attested claim --
             # `provenance_class` beside it says which, so a query can filter
             m.get('comet_run') or am.get('comet_run'),
             m.get('comet_key') or am.get('comet_key'),
             m.get('comet_project') or am.get('comet_project'),
             (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started))
              if started else None),
             started, at.get('by'),
             c.get('identical'), c.get('differing'), os.path.basename(f)])
        n += 1
    return n


VIEWS = {
    # ts_off is seconds since its own run began; only the manifest knows when
    # that was, so an absolute time exists exactly where provenance does.
    'image_events': """
        SELECT i.*, r.run_started + INTERVAL (i.ts_off) SECOND AS ts,
               r.provenance_class, r.comet_run, r.comet_key,
               coalesce(i.model_sha8, r.model_sha8) AS model,
               (r.provenance_class = 'attested') AS model_is_attested
        FROM images i
        LEFT JOIN runs r ON r.gen = i.gen AND r.run_id = i.run_id""",
    # LEFT, not INNER. A detection whose image row is missing is an
    # incomplete record, and an inner join answers by pretending it does not
    # exist -- the row count looked right while 2,541 boxes were invisible.
    # They come through with NULL context and `has_image` says so.
    'detection_events': """
        SELECT d.*, i.ts, i.orig_w, i.orig_h,
               (d.x2 - d.x1) * (d.y2 - d.y1) AS box_area,
               i.provenance_class, i.comet_run, i.model, i.model_is_attested,
               (i.image_id IS NOT NULL) AS has_image
        FROM detections d
        LEFT JOIN image_events i ON i.image_id = d.image_id AND i.gen = d.gen
                                AND i.cell = d.cell AND i.run_id = d.run_id""",
}


def build(args):
    import duckdb
    root = detect_root()
    db = args.db or default_db(root)
    fresh = not os.path.exists(db)
    con = duckdb.connect(db)
    con.execute(f"SET memory_limit='{args.memory}'")
    con.execute(f"SET threads={args.threads}")
    ddl(con)

    # A table created by an earlier version of this script keeps its old shape
    # -- CREATE TABLE IF NOT EXISTS is a no-op, and the INSERT then fails on
    # arity or, worse, lines the wrong column up with the wrong value. Compare
    # and refuse rather than append into a shape this code does not know.
    for table, cols in (('images', IMG_COLS), ('detections', DET_COLS)):
        got = [r[0] for r in con.execute(f'DESCRIBE {table}').fetchall()]
        want = [n for n, _ in cols]
        if got != want:
            con.close()
            raise SystemExit(
                f'{db}\n  {table} was built with a different schema:\n'
                f'    on disk: {got}\n    expected: {want}\n'
                f'  Delete the file and run build again -- it is derived, so '
                f'nothing is lost.')

    if fresh:
        meta_set(con, 'schema_version', SCHEMA_VERSION)
        meta_set(con, 'created_at', time.strftime('%Y-%m-%d %H:%M:%S'))
    meta_set(con, 'source_root', root)
    meta_set(con, 'derived', 'yes -- the parquet store under source_root is '
                             'authoritative; this file is a materialised copy '
                             'and is stale as soon as the sweep writes again')

    _f = con.execute('SELECT path, kind, size, mtime FROM _files').fetchall()
    seen = {r[0]: (r[2], r[3]) for r in _f}
    seen_kind = {r[0]: r[1] for r in _f}
    # ONE snapshot of the directory for both kinds. Listing img, ingesting it,
    # then listing det let the running sweep commit in between -- det ran three
    # files ahead of img and the database ended up with 2,541 detections whose
    # image row did not exist. A shard's img and det parts are committed
    # together, so a single listing is internally consistent.
    listing = {k: part_files(root, k) for k in ('img', 'det')}
    total_new = total_gone = 0
    for kind, cols in (('img', IMG_COLS), ('det', DET_COLS)):
        table = 'images' if kind == 'img' else 'detections'
        todo = []
        for p in listing[kind]:
            try:
                st = os.stat(p)
            except OSError:
                continue
            was = seen.get(p)
            if was and was[0] == st.st_size and abs(was[1] - st.st_mtime) < 1e-6:
                continue
            todo.append((p, st.st_size, st.st_mtime))

        # PARTS THAT ARE GONE. "Immutable once committed" is true of a part's
        # CONTENT and not of its existence: store.compact() deletes a whole
        # cell's parts once the merged pair verifies, and store.tiling_resume
        # (repair=True, which sweep.lane_plan uses on every start) deletes a
        # torn, overlapping or out-of-range part for the run to redo. Their
        # rows are still in here, and the redone range or the compacted file
        # arrives as new work above -- so without this the same images are
        # counted twice and `verify` reports the store has LOST rows.
        on_disk = set(listing[kind])
        gone = [p for p in sorted(seen)
                if seen_kind.get(p) == kind and p not in on_disk]
        if gone:
            print(f'{kind}: {len(gone):,} part file(s) no longer on disk '
                  f'(compacted or redone) -- dropping their rows')
            for j in range(0, len(gone), BATCH):
                lst = ', '.join("'" + p.replace("'", "''") + "'"
                                for p in gone[j:j + BATCH])
                con.execute(f'DELETE FROM {table} WHERE src IN ({lst})')
                con.execute(f'DELETE FROM _files WHERE path IN ({lst})')
            total_gone += len(gone)

        if not todo:
            print(f'{kind}: nothing new')
            continue
        print(f'{kind}: {len(todo):,} new or changed part file(s)')
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            paths = ', '.join("'" + p.replace("'", "''") + "'"
                              for p, _, _ in chunk)
            # A part that WAS loaded and then changed must not double-count,
            # and this is where that is enforced: every row this database
            # already holds from one of these paths is deleted before the
            # path is read again. The comment used to say this and nothing
            # implemented it -- the INSERT simply ran a second time.
            again = [p for p, _, _ in chunk if p in seen]
            if again:
                lst = ', '.join("'" + p.replace("'", "''") + "'" for p in again)
                con.execute(f'DELETE FROM {table} WHERE src IN ({lst})')
            have = {c[0] for c in con.execute(
                f'DESCRIBE SELECT * FROM read_parquet([{paths}], '
                f'hive_partitioning=1, union_by_name=1)').fetchall()}
            sel = []
            for name, typ in cols:
                if name == SRC_COL[0]:
                    # duckdb's own per-row source path, not anything in the
                    # data: the point is that it cannot be wrong.
                    sel.append(f'filename AS {name}')
                    continue
                # model_sha8 exists in the SCHEMA but not in files written
                # before it was added; union_by_name can only union what is
                # there, so ask rather than assume.
                sel.append(f'{name} AS {name}' if name in have
                           else f'CAST(NULL AS {typ}) AS {name}')
            con.execute(
                f'INSERT INTO {table} SELECT {", ".join(sel)} '
                f'FROM read_parquet([{paths}], hive_partitioning=1, '
                f'union_by_name=1, filename=1)')
            now = time.strftime('%Y-%m-%d %H:%M:%S')
            for p, sz, mt in chunk:
                con.execute(
                    'INSERT INTO _files VALUES (?,?,?,?,?,?) ON CONFLICT '
                    '(path) DO UPDATE SET size=excluded.size, '
                    'mtime=excluded.mtime, loaded_at=excluded.loaded_at',
                    [p, kind, sz, mt, None, now])
            total_new += len(chunk)
            print(f'  {min(i + BATCH, len(todo)):,}/{len(todo):,}', end='\r')
        print(' ' * 40, end='\r')

    print(f'runs: {load_runs(con, root)} manifest(s)')
    for name, sql in VIEWS.items():
        con.execute(f'CREATE OR REPLACE VIEW {name} AS {sql}')

    # indexes last: building them once at the end beats maintaining them
    # through every insert
    for stmt in (
        'CREATE INDEX IF NOT EXISTS ix_img_id ON images (image_id)',
        'CREATE INDEX IF NOT EXISTS ix_img_run ON images (gen, run_id)',
        'CREATE INDEX IF NOT EXISTS ix_img_cell ON images (gen, cell)',
        'CREATE INDEX IF NOT EXISTS ix_det_id ON detections (image_id)',
        'CREATE INDEX IF NOT EXISTS ix_det_run ON detections (gen, run_id)',
    ):
        con.execute(stmt)

    meta_set(con, 'built_at', time.strftime('%Y-%m-%d %H:%M:%S'))
    ni = con.execute('SELECT count(*) FROM images').fetchone()[0]
    nd = con.execute('SELECT count(*) FROM detections').fetchone()[0]
    meta_set(con, 'images_at_build', ni)
    meta_set(con, 'detections_at_build', nd)
    con.close()
    print(f'\n{ni:,} images / {nd:,} detections in {db}')
    print(f'{total_new:,} part file(s) folded in this run'
          + (f'; {total_gone:,} vanished part file(s) dropped'
             if total_gone else ''))
    print('\nThis file is DERIVED. The parquet store is authoritative; run '
          '`verify` to see how far behind this copy has fallen.')
    return 0


def _live_counts(root):
    import duckdb
    import store
    con = duckdb.connect()
    out = {}
    for kind in ('img', 'det'):
        src = store._sql_src(store._store_globs(root, kind))
        out[kind] = con.execute(f'SELECT count(*) FROM {src}').fetchone()[0]
    con.close()
    return out


def verify(args):
    import duckdb
    root = detect_root()
    db = args.db or default_db(root)
    if not os.path.exists(db):
        raise SystemExit(f'no database at {db}; run build first')
    con = duckdb.connect(db, read_only=True)
    ni = con.execute('SELECT count(*) FROM images').fetchone()[0]
    nd = con.execute('SELECT count(*) FROM detections').fetchone()[0]
    built = meta_get(con, 'built_at')
    dangling = con.execute(
        'SELECT count(*) FROM detections d WHERE NOT EXISTS ('
        'SELECT 1 FROM images i WHERE i.image_id = d.image_id '
        'AND i.gen = d.gen AND i.cell = d.cell AND i.run_id = d.run_id)'
    ).fetchone()[0]
    con.close()
    live = _live_counts(root)
    print(f'built at {built}')
    print(f'{"":12}{"in this db":>14}{"in the store":>14}{"behind":>10}')
    bad = False
    for label, mine, theirs in (('images', ni, live['img']),
                                ('detections', nd, live['det'])):
        gap = theirs - mine
        print(f'{label:12}{mine:>14,}{theirs:>14,}{gap:>10,}')
        if gap < 0:
            bad = True
    print(f'{"detections without an image row":<12}{dangling:>26,}')
    if bad:
        print('\nThis database holds MORE rows than the store. That is not '
              'staleness -- it means rows were counted twice, or the store '
              'lost parts. Investigate before trusting it.')
        return 2
    if dangling:
        print(f'\n{dangling:,} detection row(s) have no image row in this '
              f'database. That is a torn snapshot, not staleness: re-run '
              f'`build` and it resolves once the matching image parts land.')
    if ni != live['img'] or nd != live['det']:
        print('\nBehind the store, as expected while the sweep runs. '
              'Re-run `build` to catch up -- it only reads new part files.')
        return 1
    print('\nExactly level with the parquet store.')
    return 0


def info(args):
    import duckdb
    root = detect_root()
    db = args.db or default_db(root)
    if not os.path.exists(db):
        raise SystemExit(f'no database at {db}; run build first')
    con = duckdb.connect(db, read_only=True)
    print(f'{db}  ({os.path.getsize(db) / 1e6:.0f} MB)')
    for k in ('schema_version', 'created_at', 'built_at', 'source_root',
              'derived'):
        print(f'  {k:20} {meta_get(con, k)}')
    print(f'\n{"table":16}{"rows":>14}')
    for t in ('images', 'detections', 'runs', '_files'):
        n = con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]
        print(f'  {t:14}{n:>14,}')
    print('\nby run:')
    rows = con.execute(
        'SELECT i.gen, i.run_id, count(*) n, any_value(r.comet_run), '
        'any_value(r.provenance_class), min(i.ts), max(i.ts) '
        'FROM image_events i LEFT JOIN runs r '
        'ON r.gen = i.gen AND r.run_id = i.run_id '
        'GROUP BY 1, 2 ORDER BY n DESC').fetchall()
    print(f'  {"gen":>5}{"run":>8}{"rows":>12}  {"basis":<24}{"model":<11}'
          f'first row           last row')
    for gen, rid, n, run, cls, t0, t1 in rows:
        print(f'  {gen:>5}{rid:>8}{n:>12,}  {(cls or "?"):<24}'
              f'{(run or "?"):<11}{str(t0 or "-"):<20}{t1 or "-"}')
    con.close()
    print('\nviews: ' + ', '.join(VIEWS))
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db', help='defaults to <detect_root>/sweep.duckdb')
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='create or incrementally refresh')
    b.add_argument('--memory', default='8GB',
                   help='duckdb memory_limit; the sweep is usually running '
                        'and needs the machine too')
    b.add_argument('--threads', type=int, default=4)
    b.set_defaults(func=build)
    v = sub.add_parser('verify', help='counts against the parquet store')
    v.set_defaults(func=verify)
    i = sub.add_parser('info', help='what is in it, and how stale')
    i.set_defaults(func=info)
    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
