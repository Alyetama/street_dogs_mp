#!/usr/bin/env python3
"""Set a whole detection sweep aside so it can be run again, and put one back.

    python tools/detect/sweep_archive.py restore              # list them
    python tools/detect/sweep_archive.py restore --archive NAME

The store is frozen by generation: a generation's worklist fixes which images
it covers, and a later generation only ever adds images (DETECTION_RUN_
STRATEGY.md 6.2). So judging the same 32.5M images again with a newly promoted
detector is not a second generation -- every reader would count each image
twice -- it is the first generation over again, with the boxes the old model
wrote moved out of the way.

MOVED, NEVER DELETED. The shards, their errors, the run manifests that say
which engine wrote them, the status file and the SQL snapshot built from them
go to <detect root>/archive/<when>/ beside a note naming the model that wrote
them and the one replacing them. The worklist and the bootstrap partition
stay: they describe the images, not anybody's boxes. Every reader globs
<detect root>/shards/, so an archive drops out of every count the moment it is
made, and `restore` puts it back -- after setting aside whatever is there,
so a restore is as undoable as the re-run it undoes.
"""

import argparse
import fcntl
import glob
import json
import os
import sys
import time

ARCHIVE = 'archive'
NOTE = 'ARCHIVED.json'
# what a sweep writes; the worklist and _bootstrap are what it reads
MOVES = ('shards', 'errors', 'runs', 'status.json', 'sweep.duckdb')


def _held(root):
    """Is a sweep holding its run lock? A held lock is a live sweep: flock is
    released by the kernel when the process dies, however it dies."""
    for lock in glob.glob(os.path.join(root, 'runs', 'gen=*', '.lock')):
        try:
            fh = open(lock, 'a')
        except OSError:
            continue
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh, fcntl.LOCK_UN)
        except OSError:
            return True
        finally:
            fh.close()
    return False


def current_model(root):
    """The model the newest run of this store loaded, from its manifest
    ({'engine', 'sha256', 'comet_run', ...}), or None."""
    best, at = None, None
    for f in glob.glob(os.path.join(root, 'runs', 'gen=*', 'run_*.json')):
        try:
            with open(f) as fh:
                doc = json.load(fh)
            started = float(doc.get('run_started_epoch') or 0)
        except (OSError, ValueError, TypeError):
            continue
        model = doc.get('model') or {}
        if not model.get('engine') and not model.get('sha256'):
            continue  # an attested run that never recorded its engine
        if at is None or started > at:
            best, at = model, started
    return best


def archive_store(root, models, best, reason='', now=None):
    """Move a sweep's output aside; returns the note, or None when there was
    nothing to move. Refuses while a sweep holds the store."""
    if _held(root):
        raise RuntimeError('a sweep is running on this store -- stop it first')
    moving = [m for m in MOVES if os.path.exists(os.path.join(root, m))]
    if not moving:
        return None
    at = time.localtime(now if now is not None else time.time())
    name = time.strftime('%Y%m%d-%H%M%S', at)
    dest = os.path.join(root, ARCHIVE, name)
    os.makedirs(dest)  # never into an archive that is already there
    note = {
        'archived_at':
        time.strftime('%Y-%m-%d %H:%M:%S', at),
        'models':
        dict(models or {}),
        'replaced_by':
        best,
        'reason':
        reason,
        'files':
        moving,
        'restore':
        f'python tools/detect/sweep_archive.py restore --archive '
        f'{name}',
    }
    # the note first: an archive interrupted mid-move still says what it is
    with open(os.path.join(dest, NOTE), 'w') as fh:
        json.dump(note, fh, indent=1)
    for m in moving:
        os.replace(os.path.join(root, m), os.path.join(dest, m))
    return dict(note, name=name, path=dest)


def archives(root):
    """Every archive's note, newest first, each with its `name`."""
    base = os.path.join(root, ARCHIVE)
    try:
        names = sorted(os.listdir(base), reverse=True)
    except OSError:
        return []
    out = []
    for n in names:
        try:
            with open(os.path.join(base, n, NOTE)) as fh:
                note = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(note, dict):
            out.append(dict(note, name=n))
    return out


def restore_store(root, name, now=None):
    """Put an archived sweep back, setting aside what is there first."""
    base = os.path.join(root, ARCHIVE)
    if (not isinstance(name, str) or not name or os.sep in name
            or name.startswith('.')
            or not os.path.isfile(os.path.join(base, name, NOTE))):
        raise ValueError(f'no archive {name!r} in this store')
    src = os.path.join(base, name)
    with open(os.path.join(src, NOTE)) as fh:
        note = json.load(fh)
    files = [
        m for m in note.get('files') or []
        if m in MOVES and os.path.exists(os.path.join(src, m))
    ]
    current = archive_store(root, {},
                            f'restore of {name}',
                            reason=f'set aside to restore {name}',
                            now=now)
    for m in files:
        os.replace(os.path.join(src, m), os.path.join(root, m))
    note['restored_at'] = time.strftime(
        '%Y-%m-%d %H:%M:%S',
        time.localtime(now if now is not None else time.time()))
    note['restored_files'] = files
    with open(os.path.join(src, NOTE), 'w') as fh:
        json.dump(note, fh, indent=1)
    return {'restored': files, 'set_aside': current and current['name']}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    rs = sub.add_parser('restore',
                        help='list the archived sweeps, or put '
                        'one back')
    rs.add_argument('--archive', default=None)
    rs.add_argument('--root',
                    default=None,
                    help='the detect root (default: the configured one)')
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if a.root:
        root = a.root
    else:
        import store
        root = store.get_detect_root()
    if not a.archive:
        got = archives(root)
        for n in got:
            print(f"{n['name']}  {', '.join(n.get('models') or {}) or '-':24s}"
                  f" {', '.join(n.get('files') or [])}" +
                  ('  (restored)' if n.get('restored_at') else ''))
        if not got:
            print('no archives')
        return 0
    got = restore_store(root, a.archive)
    print(f"restored {', '.join(got['restored'])} from {a.archive}" +
          (f"; what was there is in archive/{got['set_aside']}"
           if got['set_aside'] else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
