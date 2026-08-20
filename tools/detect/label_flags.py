#!/usr/bin/env python3
"""
Crops whose dataset label is wrong, so the next build can leave them out.

    python tools/detect/label_flags.py                    # what is flagged
    python tools/detect/label_flags.py --export drop.json # for --exclude-ids
    python tools/detect/label_flags.py --remove <file.jpg>
    python tools/detect/label_flags.py --clear

Looking at what a model got wrong is how you find these. Most of its mistakes
are the model's; some are the dataset's, and those are the expensive ones --
a crop labelled not_dog that is plainly a dog does not just cost the one
example, it teaches the opposite of the thing you want and it does so in
every epoch. dogbin_v5 had exactly one, found by hand, recorded in
manual_fixes.jsonl. This is that, as a button.

FLAGGED BY IMAGE ID, not by filename. Datasets get rebuilt and crops get
renamed -- dogbin_v5's `no_1097638894341354_0.jpg` is one build's name for an
image the harvest knows as 1097638894341354. The id is what survives, and it
is what rebuild_crop_dataset.py already excludes on, so a flag made today
still holds for a dataset built next month.

A DATABASE, so a flag can be taken back. This is a judgement about someone
else's judgement and it will sometimes be wrong; a store you cannot undo is
the wrong place to keep one.

AND IT SAYS WHOSE. A judgement about someone else's judgement is exactly the
kind that wants a name on it now that the dashboard has more than one person
making them -- and the row is an overrule that the next build acts on by
dropping the crop. Rows written before there were accounts carry no author
and read as the admin, the same way every other store in this repo reads one.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time

# One spelling of "who judged this", shared with the flag ledgers, both audit
# ledgers and the leash store -- fn_audit owns it, this store imports it.
# Same directory, no third-party imports, and the alternative is a second copy
# of the word `admin` that drifts the first time one of the two changes.
from fn_audit import AUTHOR_FIELD, author_of

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLAG_DIR = os.path.join(REPO, 'data', 'label_flags')
DB_PATH = os.path.join(FLAG_DIR, 'label_flags.db')
# The crop-name shapes this pipeline writes, all carrying the harvest's id:
#   dogbin      no_1490447559311326_0.jpg   <prefix>_<image_id>_<n>
#   leash_v2       1766261880545220_1.jpg            <image_id>_<n>
#   detector        413652014203972.jpg              <image_id>
# The prefix used to be required, so a flag raised on a leash run parsed to no
# id at all: the row went in, --export dropped it for having none, and the
# rebuild excluded nothing. The button reported success and did nothing.
CROP_RE = re.compile(
    r'^(?:[A-Za-z][A-Za-z0-9]*_)?(\d{6,})(?:_\d+)?\.(?:jpg|jpeg|png)$', re.I)

SCHEMA = """
CREATE TABLE IF NOT EXISTS flags (
    file       TEXT PRIMARY KEY,
    image_id   TEXT,
    dataset    TEXT,
    class_was  TEXT,
    should_be  TEXT,
    run        TEXT,
    note       TEXT,
    flagged_at INTEGER NOT NULL,
    -- WHO overruled the dataset. NULL is allowed and means what an absent
    -- `by` means in the jsonl ledgers -- see fn_audit.LEGACY_AUTHOR -- so the
    -- flags raised before the dashboard had accounts keep reading as the
    -- admin's without one of them being rewritten. Quoted because BY is a
    -- keyword to sqlite's parser; every statement below quotes it too.
    "by"       TEXT
);
CREATE INDEX IF NOT EXISTS flags_image ON flags(image_id);
"""

# An annotation nobody can be named for. The run panel is behind the login
# gate, so a relabel POST that reaches add() is signed in by construction and
# a missing annotator is a PROGRAMMING error -- a caller that forgot to pass
# the session -- not a state a reviewer can get into. Refused rather than
# recorded as the admin: the row would then claim a person called somebody
# else's label wrong when they never did.
NO_AUTHOR = 'no annotator — this write was not made by a signed-in account'


def connect(path=None):
    p = path or DB_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    con = sqlite3.connect(p, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA synchronous=FULL')
    con.executescript(SCHEMA)
    _migrate(con)
    return con


def _migrate(con):
    """Bring an existing store up to SCHEMA. Idempotent, and non-destructive.

    CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so
    a column added to SCHEMA after the first flag was raised reaches the live
    database only through here. ALTER TABLE ADD COLUMN appends a column of
    NULLs and rewrites no row: the flags on disk are unchanged afterwards and
    read as the admin's, because NULL is what an absent author means.

    IDEMPOTENT ACROSS THREADS, not just across runs. connect() is called once
    per request and the dashboard is a ThreadingHTTPServer, so the first click
    on a store that has not been migrated yet can be racing the panel reads
    beside it: every thread sees the column missing, every thread issues the
    ALTER, one wins and the rest get "duplicate column name: by". Losing that
    race is the expected outcome and not an error -- the column is there
    either way -- so it is swallowed only after asking the table again.
    Anything else (a locked database, most likely) still raises: that one
    leaves the store WITHOUT the column, and a silent pass would turn it into
    "no such column" on the next insert.
    """
    have = {r['name'] for r in con.execute('PRAGMA table_info(flags)')}
    if AUTHOR_FIELD not in have:
        try:
            with con:
                con.execute('ALTER TABLE flags ADD COLUMN "by" TEXT')
        except sqlite3.OperationalError:
            have = {r['name'] for r in con.execute('PRAGMA table_info(flags)')}
            if AUTHOR_FIELD not in have:
                raise


def image_id_of(name):
    """The harvest's id for a crop, or None if the name does not carry one."""
    m = CROP_RE.match(os.path.basename(name or ''))
    return m.group(1) if m else None


def add(file, dataset='', class_was='', should_be='', run='', note='',
        now=None, path=None, by=None):
    """Flag one crop, and say who did. Idempotent; re-flagging updates.

    ``by`` is the signed-in username and is required -- see NO_AUTHOR. A
    re-flag takes the new annotator with it, the same way it takes the new
    ``should_be``: this table holds one row per file and that row is the
    overrule standing right now, so the person who owns it is the person
    whose call it now is.
    """
    file = (file or '').strip()
    if not file:
        return {'ok': False, 'error': 'no file'}, 400
    if not by:
        return {'ok': False, 'error': NO_AUTHOR}, 400
    con = connect(path)
    try:
        with con:
            con.execute(
                'INSERT INTO flags (file, image_id, dataset, class_was, '
                '  should_be, run, note, flagged_at, "by") '
                'VALUES (?,?,?,?,?,?,?,?,?) '
                'ON CONFLICT(file) DO UPDATE SET '
                '  should_be=excluded.should_be, run=excluded.run, '
                '  note=excluded.note, flagged_at=excluded.flagged_at, '
                '  "by"=excluded."by"',
                (file, image_id_of(file), dataset, class_was, should_be, run,
                 note, int(time.time() if now is None else now), str(by)))
        return dict(counts(path=path, con=con), ok=True, file=file), 200
    finally:
        con.close()


def remove(file, path=None):
    con = connect(path)
    try:
        with con:
            cur = con.execute('DELETE FROM flags WHERE file = ?', (file,))
        return dict(counts(path=path, con=con), ok=True,
                    removed=cur.rowcount > 0, file=file), 200
    finally:
        con.close()


def row_dict(row):
    """One stored flag as a plain dict, with its annotator resolved.

    Every path OUT of this store goes through here, so nothing downstream ever
    sees a flag whose author is null. A row raised before the column existed
    reads as fn_audit.LEGACY_AUTHOR, which is what it has always meant.
    """
    out = dict(row)
    out[AUTHOR_FIELD] = author_of(out.get(AUTHOR_FIELD))
    return out


def flagged_files(path=None):
    """{file: row} -- what the dashboard needs to light a tile."""
    if not os.path.exists(path or DB_PATH):
        return {}
    con = connect(path)
    try:
        return {r['file']: row_dict(r)
                for r in con.execute('SELECT * FROM flags')}
    finally:
        con.close()


def counts(path=None, con=None):
    own = con is None
    if own:
        if not os.path.exists(path or DB_PATH):
            return {'total': 0, 'ids': 0}
        con = connect(path)
    try:
        n = con.execute('SELECT COUNT(*) c FROM flags').fetchone()['c']
        ids = con.execute('SELECT COUNT(DISTINCT image_id) c FROM flags '
                          'WHERE image_id IS NOT NULL').fetchone()['c']
        return {'total': n, 'ids': ids}
    finally:
        if own:
            con.close()


def export(out, path=None):
    """Write the ids in the shape rebuild_crop_dataset.py --exclude-ids reads."""
    rows = list(flagged_files(path).values())
    ids = sorted({r['image_id'] for r in rows if r.get('image_id')})
    doc = {'created': time.strftime('%Y-%m-%d'),
           'purpose': 'Crops whose dataset label a reviewer judged wrong, '
                      'flagged from the run panel. Feed to '
                      'rebuild_crop_dataset.py --exclude-ids so the next '
                      'build leaves them out.',
           'source': os.path.relpath(DB_PATH, REPO),
           'flags': len(rows), 'image_ids': ids}
    with open(out, 'w') as fh:
        json.dump(doc, fh, indent=1)
        fh.write('\n')
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--export', metavar='OUT.json')
    ap.add_argument('--remove', metavar='FILE')
    ap.add_argument('--clear', action='store_true')
    ap.add_argument('--db', default=DB_PATH)
    a = ap.parse_args(argv)

    if a.remove:
        body, _ = remove(a.remove, path=a.db)
        print(('removed ' if body['removed'] else 'not flagged: ') + a.remove)
        return 0
    if a.clear:
        con = connect(a.db)
        with con:
            con.execute('DELETE FROM flags')
        con.close()
        print('all flags cleared')
        return 0
    if a.export:
        doc = export(a.export, path=a.db)
        print(f"{doc['flags']} flag(s), {len(doc['image_ids'])} image id(s) "
              f"-> {a.export}")
        return 0

    rows = sorted(flagged_files(a.db).values(),
                  key=lambda r: -(r.get('flagged_at') or 0))
    c = counts(path=a.db)
    print(f"{c['total']} flagged crop(s), {c['ids']} distinct image id(s)")
    import datetime as dt
    for r in rows[:100]:
        when = dt.datetime.fromtimestamp(r['flagged_at'] or 0)
        print(f"  {when:%Y-%m-%d %H:%M}  {r.get('class_was') or '?':9s} "
              f"-> {r.get('should_be') or '?':9s} {r['file']}"
              f"  by {author_of(r.get(AUTHOR_FIELD))}")
    if len(rows) > 100:
        print(f'  ... and {len(rows) - 100} more')
    if rows:
        print('\nexport with --export drop.json, then rebuild with '
              '--exclude-ids drop.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
