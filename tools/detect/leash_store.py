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
    note        TEXT
);
CREATE INDEX IF NOT EXISTS leash_label ON leash(label);
CREATE INDEX IF NOT EXISTS leash_when  ON leash(labelled_at);
"""


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
    return con


def parse_crop(name):
    """(image_id, ts, conf) from a crop filename, or None if it is not one."""
    m = CROP_RE.match(name or '')
    if not m:
        return None
    return m.group(2), int(m.group(1)), round(int(m.group(3)) / 100.0, 2)


def record(name, label, copy_from=None, source='review_page', note=None,
           now=None, path=None):
    """Record one verdict. Idempotent, and re-deciding replaces the old one.

    Returns (dict, http status) so the dashboard can hand it straight back.
    """
    got = parse_crop(name)
    if not got:
        return {'ok': False, 'error': 'malformed crop name'}, 400
    if label not in LABELS:
        return {'ok': False, 'error': 'unknown label %r' % (label,)}, 400
    image_id, ts, conf = got
    copied = False
    if copy_from:
        copied = _copy_pair(name, copy_from)
    con = connect(path)
    try:
        with con:
            con.execute(
                'INSERT INTO leash (crop, image_id, label, conf, ts, '
                '                   labelled_at, source, note) '
                'VALUES (?,?,?,?,?,?,?,?) '
                'ON CONFLICT(crop) DO UPDATE SET '
                '  label=excluded.label, labelled_at=excluded.labelled_at, '
                '  source=excluded.source, note=excluded.note',
                (name, image_id, label, conf, ts,
                 int(time.time() if now is None else now), source, note))
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
                    w.write(json.dumps(dict(r)) + '\n')
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
                      f"conf={r['conf']}  {r['crop']}")
            if len(rows) > 200:
                print(f'  ... and {len(rows) - 200} more')
        return 0
    finally:
        con.close()


if __name__ == '__main__':
    sys.exit(main())
