#!/usr/bin/env python3
"""
Leashed / unleashed verdicts, in a local database you can query and undo.

    python tools/detect/leash_store.py                     # what is recorded
    python tools/detect/leash_store.py --list --label leashed
    python tools/detect/leash_store.py --remove <crop.jpg> # drop one verdict
    python tools/detect/leash_store.py --remove-label unleashed
    python tools/detect/leash_store.py --since 2026-08-06 --list
    python tools/detect/leash_store.py --export out.jsonl  # feed a dataset build

WHY A DATABASE AND NOT A JSONL. The dog/not-dog stores are append-only files
because their verdicts are permanent by design -- a false positive stays a
false positive. Leash verdicts are a newer, softer axis, and the point of
recording them separately is to be able to take them back: an annotator's
sense of "leashed" drifts, a whole afternoon's labels can turn out to have
been made at the wrong zoom, and a dataset built on them has to be rebuildable
without them. Deleting a row from an append-only file means rewriting the file
and hoping nothing else held an offset into it. sqlite3 is in the standard
library, so this costs a clone nothing.

WHAT A ROW MEANS. "This crop shows a dog, and it is/is not on a leash." The
dog part is implied by the verdict -- an unleashed cow is not a row here -- but
this store deliberately does NOT write to the dog/not-dog ledgers. Those two
axes answer different questions, they are consumed by different models, and
folding one into the other is how a store ends up meaning two things. In
particular hard_positives means "a dog the detector was unsure about", which
is not the same claim as "a dog", and filling it with easy leashed dogs would
quietly change what that set is for.

Crops are copied out at verdict time, exactly as the other stores do it: the
live pool rotates, and a label whose image has aged out is not trainable.
Removing a verdict removes its copies too, so `--remove` really does undo it.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time

# One spelling of "who judged this", shared with the flag ledgers and both
# audit ledgers -- fn_audit owns it, this store imports it. Same directory, no
# third-party imports, and the alternative is a second copy of the word
# `admin` that drifts the first time one of the two changes.
from fn_audit import AUTHOR_FIELD, author_of

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEASH_DIR = os.path.join(REPO, 'data', 'leash_labels')
DB_PATH = os.path.join(LEASH_DIR, 'leash.db')
CROPS_OUT = os.path.join(LEASH_DIR, 'crops')
FULL_OUT = os.path.join(LEASH_DIR, 'full')
LABELS = ('leashed', 'unleashed')
# same shape the rest of the pipeline uses: <ts>_<image_id>_<conf>.jpg
CROP_RE = re.compile(r'^(\d{10,})_([A-Za-z0-9_-]{1,64})_(\d{3})\.jpg$')

SCHEMA = """
CREATE TABLE IF NOT EXISTS leash (
    crop        TEXT PRIMARY KEY,
    image_id    TEXT NOT NULL,
    label       TEXT NOT NULL CHECK (label IN ('leashed', 'unleashed')),
    conf        REAL,
    ts          INTEGER,
    labelled_at INTEGER NOT NULL,
    source      TEXT NOT NULL DEFAULT 'review_page',
    note        TEXT,
    -- WHO said so. `source` was here first and answers a different question:
    -- it is the SURFACE a verdict came off ('review_page'), which was as near
    -- to an author as this table could get before the dashboard had accounts.
    -- NULL is allowed and means exactly what an absent `by` means in the
    -- jsonl ledgers -- see fn_audit.LEGACY_AUTHOR -- so the rows written
    -- before this column existed keep reading as the admin's without one of
    -- them being rewritten. Quoted because BY is a keyword to sqlite's
    -- parser; every statement below quotes it for the same reason.
    "by"        TEXT
);
CREATE INDEX IF NOT EXISTS leash_label ON leash(label);
CREATE INDEX IF NOT EXISTS leash_when  ON leash(labelled_at);
"""

# An annotation nobody can be named for. The review page is behind the login
# gate, so a leash POST that reaches record() is signed in by construction and
# a missing annotator is a PROGRAMMING error -- a caller that forgot to pass
# the session -- not a state a reviewer can get into. Refused rather than
# recorded as the admin: this row would then claim a person made a call they
# never made, and that is the one thing naming the annotator exists to stop.
# (A leash row CAN be taken back, unlike a jsonl line -- the refusal is not
# about recoverability, it is about not writing something untrue.)
NO_AUTHOR = 'no annotator — this write was not made by a signed-in account'


def connect(path=None):
    """Open the store, creating it if this is the first verdict.

    WAL so a reader (the dashboard rendering a page) never blocks the writer
    (the same dashboard recording a click) -- they are different threads of one
    ThreadingHTTPServer, and the default journal turns that into a lock error
    under exactly the double-click the buttons invite.
    """
    p = path or DB_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    con = sqlite3.connect(p, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA synchronous=FULL')   # a verdict must survive a crash
    con.executescript(SCHEMA)
    _migrate(con)
    return con


def _migrate(con):
    """Bring an existing store up to SCHEMA. Idempotent, and non-destructive.

    CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so
    a column added to SCHEMA after the first verdict was recorded reaches the
    live database only through here. ALTER TABLE ADD COLUMN appends a column
    of NULLs and rewrites no row: the thirteen verdicts on disk are the same
    thirteen bytes-for-bytes afterwards, and they read as the admin's because
    NULL is what an absent author means.

    Reading the table's own columns rather than a version number, because
    this store predates having one and a schema_version table introduced now
    would have to guess which version the rows on disk already are.

    IDEMPOTENT ACROSS THREADS, not just across runs. connect() is called once
    per request and the dashboard is a ThreadingHTTPServer, so the very first
    click on a store that has not been migrated yet can be racing the page
    reads beside it: every thread sees the column missing, every thread issues
    the ALTER, one wins and the rest get "duplicate column name: by". record()
    has no handler for that, so the loser's verdict came back to the reviewer
    as {'ok': false} and was never written. Losing the race is the expected
    outcome here and not an error -- the column is there either way, which is
    all the caller wanted -- so it is swallowed only after asking the table
    again. Anything else (a locked database, most likely) still raises: that
    one leaves the store WITHOUT the column, and a silent pass would turn it
    into "no such column" on the next insert.
    """
    have = {r['name'] for r in con.execute('PRAGMA table_info(leash)')}
    if AUTHOR_FIELD not in have:
        try:
            with con:
                con.execute('ALTER TABLE leash ADD COLUMN "by" TEXT')
        except sqlite3.OperationalError:
            have = {r['name'] for r in con.execute('PRAGMA table_info(leash)')}
            if AUTHOR_FIELD not in have:
                raise


def parse_crop(name):
    """(image_id, ts, conf) from a crop filename, or None if it is not one."""
    m = CROP_RE.match(name or '')
    if not m:
        return None
    return m.group(2), int(m.group(1)), round(int(m.group(3)) / 100.0, 2)


def record(name, label, copy_from=None, source='review_page', note=None,
           now=None, path=None, by=None):
    """Record one verdict, and who made it. Idempotent; re-deciding replaces.

    Returns (dict, http status) so the dashboard can hand it straight back.

    ``by`` is the signed-in username and is required -- see NO_AUTHOR. A
    re-decision takes the new annotator with it: THIS table holds one row per
    crop and that row is the verdict standing right now, so the person who
    owns it is the person whose call it now is. The flag ledgers answer the
    same question the other way (the first flag stands and a second press of
    it changes nothing, annotator included) because a flag is a line in an
    append-only file rather than a row that gets replaced -- see flag_crop's
    duplicate branch. Neither is the general rule; each follows the shape of
    the store it is written into.
    """
    got = parse_crop(name)
    if not got:
        return {'ok': False, 'error': 'malformed crop name'}, 400
    if label not in LABELS:
        return {'ok': False, 'error': 'unknown label %r' % (label,)}, 400
    if not by:
        return {'ok': False, 'error': NO_AUTHOR}, 400
    image_id, ts, conf = got
    copied = False
    if copy_from:
        copied = _copy_pair(name, copy_from)
    con = connect(path)
    try:
        with con:
            con.execute(
                'INSERT INTO leash (crop, image_id, label, conf, ts, '
                '                   labelled_at, source, note, "by") '
                'VALUES (?,?,?,?,?,?,?,?,?) '
                'ON CONFLICT(crop) DO UPDATE SET '
                '  label=excluded.label, labelled_at=excluded.labelled_at, '
                '  source=excluded.source, note=excluded.note, '
                '  "by"=excluded."by"',
                (name, image_id, label, conf, ts,
                 int(time.time() if now is None else now), source, note,
                 str(by)))
        return dict(_counts(con), ok=True, crop=name, label=label,
                    copied=copied), 200
    finally:
        con.close()


def remove(name, path=None):
    """Drop one verdict and the copies made for it. Missing is not an error."""
    con = connect(path)
    try:
        with con:
            cur = con.execute('DELETE FROM leash WHERE crop = ?', (name,))
        gone = cur.rowcount > 0
        if gone:
            _drop_pair(name)
        return dict(_counts(con), ok=True, removed=gone, crop=name), 200
    finally:
        con.close()


def remove_label(label, path=None):
    """Drop every verdict of one label. Returns how many went."""
    con = connect(path)
    try:
        names = [r['crop'] for r in
                 con.execute('SELECT crop FROM leash WHERE label = ?', (label,))]
        with con:
            con.execute('DELETE FROM leash WHERE label = ?', (label,))
        for n in names:
            _drop_pair(n)
        return dict(_counts(con), ok=True, removed=len(names)), 200
    finally:
        con.close()


def row_dict(row):
    """One stored verdict as a plain dict, with its annotator resolved.

    Every path OUT of this store goes through here, so nothing downstream --
    a dataset build reading the export, a person reading --list -- ever sees
    a verdict whose author is null. A row recorded before the column existed
    reads as fn_audit.LEGACY_AUTHOR, which is what it has always meant.
    """
    out = dict(row)
    out[AUTHOR_FIELD] = author_of(out.get(AUTHOR_FIELD))
    return out


def labels_for(names=None, path=None):
    """{crop: label} -- for the review page, which asks about one page of crops."""
    if not os.path.exists(path or DB_PATH):
        return {}
    con = connect(path)
    try:
        if names is None:
            rows = con.execute('SELECT crop, label FROM leash')
        else:
            names = list(names)
            if not names:
                return {}
            out = {}
            # chunked: sqlite caps a statement's variables (999 by default)
            for i in range(0, len(names), 500):
                part = names[i:i + 500]
                q = ('SELECT crop, label FROM leash WHERE crop IN (%s)'
                     % ','.join('?' * len(part)))
                out.update({r['crop']: r['label']
                            for r in con.execute(q, part)})
            return out
        return {r['crop']: r['label'] for r in rows}
    finally:
        con.close()


def _counts(con):
    rows = con.execute('SELECT label, COUNT(*) n FROM leash GROUP BY label')
    got = {r['label']: r['n'] for r in rows}
    return {'leashed': got.get('leashed', 0),
            'unleashed': got.get('unleashed', 0),
            'total': sum(got.values())}


def counts(path=None):
    if not os.path.exists(path or DB_PATH):
        return {'leashed': 0, 'unleashed': 0, 'total': 0}
    con = connect(path)
    try:
        return _counts(con)
    finally:
        con.close()


def _copy_pair(name, src_dirs):
    """Copy the crop and its full frame out of the rotating pool."""
    ok = False
    for src_dir, dst_dir in ((src_dirs.get('crop'), CROPS_OUT),
                             (src_dirs.get('full'), FULL_OUT)):
        if not src_dir:
            continue
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            continue
        try:
            os.makedirs(dst_dir, exist_ok=True)
            tmp = os.path.join(dst_dir, '.' + name + '.tmp')
            with open(src, 'rb') as r, open(tmp, 'wb') as w:
                w.write(r.read())
            os.replace(tmp, os.path.join(dst_dir, name))
            ok = True
        except OSError:
            pass
    return ok


def _drop_pair(name):
    for d in (CROPS_OUT, FULL_OUT):
        try:
            os.remove(os.path.join(d, name))
        except OSError:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--label', choices=LABELS)
    ap.add_argument('--since', help='YYYY-MM-DD; only verdicts on or after')
    ap.add_argument('--remove', metavar='CROP')
    ap.add_argument('--remove-label', choices=LABELS)
    ap.add_argument('--export', metavar='OUT.jsonl')
    ap.add_argument('--db', default=DB_PATH)
    a = ap.parse_args(argv)

    if a.remove:
        body, _ = remove(a.remove, path=a.db)
        print(('removed ' if body['removed'] else 'not recorded: ') + a.remove)
        print(f"now: {body['leashed']} leashed, {body['unleashed']} unleashed")
        return 0
    if a.remove_label:
        body, _ = remove_label(a.remove_label, path=a.db)
        print(f"removed {body['removed']} {a.remove_label} verdict(s)")
        print(f"now: {body['leashed']} leashed, {body['unleashed']} unleashed")
        return 0

    if not os.path.exists(a.db):
        print('nothing recorded yet')
        return 0
    con = connect(a.db)
    try:
        q = 'SELECT * FROM leash'
        args, where = [], []
        if a.label:
            where.append('label = ?')
            args.append(a.label)
        if a.since:
            try:
                import datetime as dt
                cut = int(dt.datetime.strptime(a.since, '%Y-%m-%d').timestamp())
            except ValueError:
                print('--since wants YYYY-MM-DD', file=sys.stderr)
                return 1
            where.append('labelled_at >= ?')
            args.append(cut)
        if where:
            q += ' WHERE ' + ' AND '.join(where)
        q += ' ORDER BY labelled_at DESC'
        rows = list(con.execute(q, args))

        if a.export:
            with open(a.export, 'w') as w:
                for r in rows:
                    w.write(json.dumps(row_dict(r)) + '\n')
            print(f'{len(rows)} verdict(s) -> {a.export}')
            return 0

        c = _counts(con)
        print(f"{c['total']} verdict(s): {c['leashed']} leashed, "
              f"{c['unleashed']} unleashed   [{a.db}]")
        if a.list:
            import datetime as dt
            for r in rows[:200]:
                when = dt.datetime.fromtimestamp(r['labelled_at'])
                print(f"  {when:%Y-%m-%d %H:%M}  {r['label']:9s} "
                      f"conf={r['conf']}  {r['crop']}"
                      f"  by {author_of(r[AUTHOR_FIELD])}")
            if len(rows) > 200:
                print(f'  ... and {len(rows) - 200} more')
        return 0
    finally:
        con.close()


if __name__ == '__main__':
    sys.exit(main())
