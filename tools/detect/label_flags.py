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
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLAG_DIR = os.path.join(REPO, 'data', 'label_flags')
DB_PATH = os.path.join(FLAG_DIR, 'label_flags.db')
# <prefix>_<image_id>_<n>.jpg, the shape rebuild_crop_dataset.py writes
CROP_RE = re.compile(r'^[A-Za-z]+_(\d{6,})_\d+\.(?:jpg|jpeg|png)$', re.I)

SCHEMA = """
CREATE TABLE IF NOT EXISTS flags (
    file       TEXT PRIMARY KEY,
    image_id   TEXT,
    dataset    TEXT,
    class_was  TEXT,
    should_be  TEXT,
    run        TEXT,
    note       TEXT,
    flagged_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS flags_image ON flags(image_id);
"""


def connect(path=None):
    p = path or DB_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    con = sqlite3.connect(p, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA synchronous=FULL')
    con.executescript(SCHEMA)
    return con


def image_id_of(name):
    """The harvest's id for a crop, or None if the name does not carry one."""
    m = CROP_RE.match(os.path.basename(name or ''))
    return m.group(1) if m else None


def add(file, dataset='', class_was='', should_be='', run='', note='',
        now=None, path=None):
    """Flag one crop. Idempotent; re-flagging updates rather than duplicates."""
    file = (file or '').strip()
    if not file:
        return {'ok': False, 'error': 'no file'}, 400
    con = connect(path)
    try:
        with con:
            con.execute(
                'INSERT INTO flags (file, image_id, dataset, class_was, '
                '  should_be, run, note, flagged_at) VALUES (?,?,?,?,?,?,?,?) '
                'ON CONFLICT(file) DO UPDATE SET '
                '  should_be=excluded.should_be, run=excluded.run, '
                '  note=excluded.note, flagged_at=excluded.flagged_at',
                (file, image_id_of(file), dataset, class_was, should_be, run,
                 note, int(time.time() if now is None else now)))
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


def flagged_files(path=None):
    """{file: row} -- what the dashboard needs to light a tile."""
    if not os.path.exists(path or DB_PATH):
        return {}
    con = connect(path)
    try:
        return {r['file']: dict(r) for r in con.execute('SELECT * FROM flags')}
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
              f"-> {r.get('should_be') or '?':9s} {r['file']}")
    if len(rows) > 100:
        print(f'  ... and {len(rows) - 100} more')
    if rows:
        print('\nexport with --export drop.json, then rebuild with '
              '--exclude-ids drop.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
