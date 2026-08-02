"""
Build and verify the frozen detection worklist (spec 2.1, 5.2, 6.1-6.2).

Enumerates every (cell, drive) image directory across the grid_runs roots,
records each directory's image ids in INODE order, dedups the 16 cross-drive
overlap cells, and freezes the result under::

    $DETECT_ROOT/worklist/gen=NNNN/
      _meta.json                 # sha256, built_at, n_images, n_shards, dedup
      _dirs.json                 # per-pair {cell, drive, region, root, dir_mtime}
      <cell>/<drive>.ids.npy     # uint64 image_ids, inode order, mmap-loadable

    python tools/detect/worklist.py build                 # next gen from catalog
    python tools/detect/worklist.py build --gen 3
    python tools/detect/worklist.py verify --gen 1        # re-hash frozen files

Design decisions (all measured in docs/DETECTION_RUN_STRATEGY.md, cited by
section -- do not re-litigate):

* 2.2  The enumerator reads ONLY ``e.name`` and ``e.inode()`` from getdents
  (3.8-4.1M entries/s). It never calls ``e.stat()`` (423-1,439 stat/s --
  capybara's 1.5M-file directory would be 1.0 h of stat alone).
* 2.2  Each directory's ids are sorted by inode before freezing: ext4 htree
  makes readdir order ~50% uncorrelated with inode order, and inode-ordered
  reads are +8% on the BOT drives and +140% on bobcat.
* 2.1  Exactly 16 cells overlap across drives (39,985 duplicate ids). They
  are deduped per cell by filename-set union -- never a global 32.5M set --
  and contested ids are assigned to bobcat (South_Asia) / lynx (Europe),
  the two lanes with the most idle capacity.
* 6.2  The catalog is opened read_only for < 1 s just to learn the (cell,
  drive, region) pairs; a naive os.walk would open ~2,900 empty cell dirs
  across three qd=1 drives. If the catalog is write-locked: 5 retries with
  backoff, then a depth-1 scandir fallback (assertions disabled).
* 6.1  Shard = 4,000 contiguous inode-ordered ids of one pair; ``n_shards``
  is the post-dedup ceil-sum (do not assert the pre-dedup 8,979).
* 6.2  The generation is FROZEN: the gen dir is written once via a temp dir
  rename and never mutated; the runner refuses to start if the on-disk hash
  differs, and ``verify`` re-checks every file against ``_meta.json``.
* 6.3  Resume needs from the worklist only: immutable per-pair ``ids.npy``
  (positional shard identity ``ids[i*4000:(i+1)*4000]``), mmap-loadable so
  lanes touch only the active shard's 32 KB slice.

No env-specific paths are hardcoded: roots come from the gitignored
``data/catalog_dirs.txt`` and DETECT_ROOT from ``data/detect_root.txt``.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

# Repo root = two levels above tools/detect/; only used to locate the
# gitignored default config files, never to hardcode drive layout.
REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CATALOG = os.path.join(REPO, 'data', 'catalog.duckdb')
DEFAULT_ROOTS_FILE = os.path.join(REPO, 'data', 'catalog_dirs.txt')
DEFAULT_DETECT_ROOT_FILE = os.path.join(REPO, 'data', 'detect_root.txt')

SHARD_SIZE = 4000  # 6.1 -- unit of work, positional shard identity
IMAGES_SUBDIR = 'ground_animal_images'  # <root>/<cell>/ground_animal_images/<id>.jpg
SENTINEL = '.detect_drive_id.json'  # 6.4 -- drive-identity authority at each root

# 2.1 -- contested ids go to bobcat (South_Asia) and lynx (Europe). Expressed
# as a preference order so an id contested only between two non-preferred
# drives still resolves deterministically (alphabetical fallback).
WINNER_PREFERENCE = ('bobcat', 'lynx')

# Same cell-name grammar as tools/catalog/catalog.py: Region_swlon_swlat_nelon_nelat.
CELL_RE = re.compile(
    r'^(?P<region>.+?)_(?P<sw_lon>-?\d+(?:\.\d+)?)_(?P<sw_lat>-?\d+(?:\.\d+)?)'
    r'_(?P<ne_lon>-?\d+(?:\.\d+)?)_(?P<ne_lat>-?\d+(?:\.\d+)?)$')

# ---------------------------------------------------------------------------
# config resolution (gitignored files, no hardcoded drive paths)
# ---------------------------------------------------------------------------


def read_lines_file(path):
    """Return non-empty, non-comment lines of a config file."""
    with open(path) as f:
        return [
            ln.strip() for ln in f if ln.strip() and not ln.startswith('#')
        ]


def resolve_detect_root(args):
    """DETECT_ROOT from --detect-root, else the gitignored data/detect_root.txt."""
    if args.detect_root:
        return args.detect_root
    if os.path.exists(DEFAULT_DETECT_ROOT_FILE):
        lines = read_lines_file(DEFAULT_DETECT_ROOT_FILE)
        if lines:
            return lines[0]
    sys.exit('no --detect-root and no %s' % DEFAULT_DETECT_ROOT_FILE)


def drive_label(root):
    """Return the drive label a grid_runs root lives on.

    The 6.4 sentinel (``.detect_drive_id.json`` at the root) is the
    authority when present -- it survives remounts and lets tests inject
    labels. Fallback: walk up to the mount point (device-id change) and use
    that directory's name, same convention as tools/catalog/catalog.py.
    """
    sentinel = os.path.join(root, SENTINEL)
    try:
        with open(sentinel) as f:
            label = json.load(f).get('drive')
        if label:
            return label
    except (OSError, ValueError):
        pass
    try:
        p = os.path.realpath(root)
        dev = os.stat(p).st_dev
        while True:
            parent = os.path.dirname(p)
            if parent == p or os.stat(parent).st_dev != dev:
                return os.path.basename(p) or '?'
            p = parent
    except OSError:
        return os.path.basename(os.path.normpath(root)) or '?'


def resolve_roots(args):
    """Map drive label -> grid_runs root from the gitignored roots file.

    Roots whose drive never appears in the catalog (crucial/weasel hold
    parquets, not JPGs) are simply never looked up. Two roots resolving to
    the same drive would make path reconstruction ambiguous -> hard fail.
    """
    paths = args.dirs or read_lines_file(args.roots_file)
    roots = {}
    for p in paths:
        label = drive_label(p)
        if label in roots and roots[label] != p:
            sys.exit('two roots resolve to drive %r: %s and %s' %
                     (label, roots[label], p))
        roots[label] = p
    return roots


# ---------------------------------------------------------------------------
# catalog access (6.2: read_only for < 1 s, retries, scandir fallback)
# ---------------------------------------------------------------------------

_CATALOG_SNIPPET = r'''
import json, sys
import duckdb
con = duckdb.connect(sys.argv[1], read_only=True)
pairs = con.sql("SELECT cell, drive, region, n_images FROM images "
                "WHERE n_images > 0").fetchall()
cells = con.sql("SELECT cell, region, n_unique FROM cell_images").fetchall()
con.close()
print(json.dumps({"pairs": pairs, "cells": cells}))
'''


def _catalog_query(path, python):
    """One attempt at reading the catalog; raises on lock/IO failure.

    Uses in-process duckdb when importable; otherwise shells out to an
    interpreter that has it (the dnd env deliberately doesn't -- catalog
    reads ride mp14). The connection lives < 1 s and is closed (6.2).
    """
    try:
        import duckdb
    except ImportError:
        proc = subprocess.run([python, '-c', _CATALOG_SNIPPET, path],
                              capture_output=True,
                              text=True)
        if proc.returncode != 0:
            raise OSError('catalog subprocess failed: %s' %
                          proc.stderr.strip()[-500:])
        return json.loads(proc.stdout)
    con = duckdb.connect(path, read_only=True)
    try:
        pairs = con.sql('SELECT cell, drive, region, n_images FROM images '
                        'WHERE n_images > 0').fetchall()
        cells = con.sql(
            'SELECT cell, region, n_unique FROM cell_images').fetchall()
    finally:
        con.close()
    return {'pairs': pairs, 'cells': cells}


def read_catalog(path, python, retries=5):
    """Catalog pairs + expectations, with 6.2's 5-retry backoff on locks.

    Returns ``(pairs, expected)`` where pairs is ``[(cell, drive, region,
    n_images), ...]`` and expected maps cell -> n_unique, or ``(None, None)``
    if the catalog stayed unreadable (caller falls back to scandir).
    """
    for attempt in range(retries):
        try:
            data = _catalog_query(path, python)
            pairs = [tuple(r) for r in data['pairs']]
            expected = {r[0]: int(r[2]) for r in data['cells']}
            return pairs, expected
        except Exception as exc:  # duckdb.IOException / OSError / json
            wait = 2 * (attempt + 1)
            print('catalog read failed (%s); retry %d/%d in %ds' %
                  (exc, attempt + 1, retries, wait),
                  file=sys.stderr)
            time.sleep(wait)
    return None, None


def scandir_pairs(roots):
    """6.2 fallback: depth-1 scandir of the roots + isdir on the images dir.

    ~5,396 isdir calls, 6-10 s. No image counts and no per-cell unique
    expectations are available on this path, so build() disables the
    catalog assertions and records ``asserted: false`` in _meta.json.
    """
    pairs = []
    for drive, root in sorted(roots.items()):
        try:
            entries = sorted(e.name for e in os.scandir(root) if e.is_dir())
        except OSError:
            continue  # offline root; catalog-less mode can't tell more
        for cell in entries:
            if not os.path.isdir(os.path.join(root, cell, IMAGES_SUBDIR)):
                continue
            m = CELL_RE.match(cell)
            region = m.group('region') if m else cell
            pairs.append((cell, drive, region, None))
    return pairs


# ---------------------------------------------------------------------------
# enumeration + dedup
# ---------------------------------------------------------------------------


def scan_ids_inode_order(dirpath):
    """Scan one images dir -> (uint64 ids in inode order, n_skipped).

    2.2: names + inodes only. ``e.inode()`` rides in the getdents dirent
    for free; ``e.stat()`` is never called (3.8M entries/s vs 423/s).
    Non-``<digits>.jpg`` entries are skipped and counted -- if the catalog
    counted them the per-cell assertion catches the drift downstream.
    """
    ids, inodes, skipped = [], [], 0
    with os.scandir(dirpath) as it:
        for e in it:
            name = e.name
            if name.endswith('.jpg'):
                stem = name[:-4]
                if stem.isdigit():
                    ids.append(int(stem))
                    inodes.append(e.inode())
                    continue
            skipped += 1
    arr = np.asarray(ids, dtype=np.uint64)
    order = np.argsort(np.asarray(inodes, dtype=np.uint64), kind='stable')
    return arr[order], skipped


def dedup_cell(per_drive):
    """Dedup one multi-drive cell by filename-set union (2.1).

    ``per_drive`` maps drive -> inode-ordered ids. Contested ids (present
    on >= 2 drives) are kept on exactly one drive -- the first in
    WINNER_PREFERENCE order that holds the id -- and dropped from the rest.
    The kept arrays remain in their original inode order (masking only).

    Returns ``(deduped, n_unique, stats)``; stats is None when the drives
    are a complementary split (zero overlap -- 484 of the 500 multi-drive
    cells, 2.1).
    """
    uniq, counts = np.unique(np.concatenate(list(per_drive.values())),
                             return_counts=True)
    contested = uniq[counts > 1]
    if contested.size == 0:
        return per_drive, int(uniq.size), None
    order = [d for d in WINNER_PREFERENCE if d in per_drive]
    order += sorted(d for d in per_drive if d not in WINNER_PREFERENCE)
    deduped, removed = {}, {}
    claimed = np.empty(0, dtype=np.uint64)  # contested ids already kept
    for drive in order:
        ids = per_drive[drive]
        drop = np.isin(ids, claimed) if claimed.size else np.zeros(ids.size,
                                                                   dtype=bool)
        deduped[drive] = ids[~drop]
        removed[drive] = int(drop.sum())
        newly = ids[np.isin(ids, contested) & ~drop]
        if newly.size:
            claimed = np.union1d(claimed, newly)
    stats = {
        'winner': order[0],
        'n_contested': int(contested.size),
        'removed': {
            d: n
            for d, n in removed.items() if n
        },
    }
    return deduped, int(uniq.size), stats


# ---------------------------------------------------------------------------
# freezing + hashing (6.2)
# ---------------------------------------------------------------------------


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def combined_hash(files):
    """One hash over the whole generation: sha256 of sorted relpath:hash lines.

    This is the value the runner pins in its manifest and refuses to start
    without (6.2), and what ``verify`` recomputes.
    """
    h = hashlib.sha256()
    for relpath in sorted(files):
        h.update(('%s:%s\n' % (relpath, files[relpath])).encode())
    return h.hexdigest()


def gen_dir(detect_root, gen):
    return os.path.join(detect_root, 'worklist', 'gen=%04d' % gen)


def next_gen(detect_root):
    base = os.path.join(detect_root, 'worklist')
    gens = []
    if os.path.isdir(base):
        for name in os.listdir(base):
            m = re.match(r'^gen=(\d{4})$', name)
            if m:
                gens.append(int(m.group(1)))
    return max(gens, default=0) + 1


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_build(args):
    detect_root = resolve_detect_root(args)
    roots = resolve_roots(args)
    if args.gen is not None and args.gen < 1:
        sys.exit('--gen must be >= 1 (got %d)' % args.gen)
    gen = args.gen if args.gen is not None else next_gen(detect_root)
    out = gen_dir(detect_root, gen)
    if os.path.exists(out):  # 6.2 -- generations are frozen, never rebuilt
        sys.exit('%s already exists; generations are frozen (pick another '
                 '--gen)' % out)

    pairs, expected = read_catalog(args.catalog, args.duckdb_python)
    asserted = pairs is not None
    if not asserted:
        print(
            'catalog unreadable after retries -> depth-1 scandir fallback '
            '(6.2); per-cell assertions DISABLED',
            file=sys.stderr)
        pairs = scandir_pairs(roots)
    if not pairs:
        sys.exit('no (cell, drive) pairs found')

    missing_drives = sorted({p[1] for p in pairs} - set(roots))
    if missing_drives:
        sys.exit('catalog references drives with no configured root '
                 '(offline?): %s' % ', '.join(missing_drives))

    # -- enumerate every pair (2.2: getdents names+inodes only) -------------
    t0 = time.time()
    scanned = {}  # (cell, drive) -> ids ndarray
    dir_meta = {}  # (cell, drive) -> {region, root, dir_mtime}
    raw_warn, total_skipped = [], 0
    for cell, drive, region, n_images in sorted(pairs):
        d = os.path.join(roots[drive], cell, IMAGES_SUBDIR)
        try:
            ids, skipped = scan_ids_inode_order(d)
            mtime = os.stat(
                d).st_mtime  # one stat per DIR (1,192), not per file
        except OSError as exc:
            sys.exit('cannot enumerate %s (catalog says it has images): %s' %
                     (d, exc))
        total_skipped += skipped
        if n_images is not None and ids.size != n_images:
            raw_warn.append((cell, drive, n_images, ids.size))
        scanned[(cell, drive)] = ids
        dir_meta[(cell, drive)] = {
            'region': region,
            'root': roots[drive],
            'dir_mtime': mtime
        }
    print('enumerated %d pairs in %.1fs (%d non-image entries skipped)' %
          (len(scanned), time.time() - t0, total_skipped))
    for cell, drive, want, got in raw_warn:
        # Raw per-pair drift is a warning only; the binding check is the
        # per-cell unique count below (2.1).
        print('warn: %s/%s raw count %d != catalog %d' %
              (cell, drive, got, want),
              file=sys.stderr)

    # -- dedup the overlap cells (2.1) --------------------------------------
    by_cell = {}
    for (cell, drive), ids in scanned.items():
        by_cell.setdefault(cell, {})[drive] = ids
    unique_per_cell, dedup_cells = {}, {}
    n_removed = n_contested = 0
    for cell, per_drive in by_cell.items():
        if len(per_drive) == 1:
            unique_per_cell[cell] = int(next(iter(per_drive.values())).size)
            continue
        deduped, n_unique, stats = dedup_cell(per_drive)
        unique_per_cell[cell] = n_unique
        if stats:
            by_cell[cell] = deduped
            dedup_cells[cell] = stats
            n_removed += sum(stats['removed'].values())
            n_contested += stats['n_contested']

    # -- assert against the catalog's unique counts; HARD FAIL with the
    #    per-cell delta (2.1: the authoritative figure is asserted here) ----
    total_unique = sum(unique_per_cell.values())
    if asserted:
        deltas = []
        for cell in sorted(set(unique_per_cell) | set(expected)):
            got = unique_per_cell.get(cell, 0)
            want = expected.get(cell, 0)
            if got != want:
                deltas.append((cell, want, got))
        if deltas:
            print('FATAL: worklist disagrees with catalog cell_images on '
                  '%d cell(s):' % len(deltas),
                  file=sys.stderr)
            for cell, want, got in deltas:
                print('  %-40s expected %9d  scanned %9d  delta %+d' %
                      (cell, want, got, got - want),
                      file=sys.stderr)
            sys.exit(1)
        exp_total = sum(expected.values())
        assert total_unique == exp_total, (total_unique, exp_total)

    # -- freeze (6.2): write into a temp dir, hash, rename once -------------
    tmp = out + '.tmp-%d' % os.getpid()
    os.makedirs(tmp)
    try:
        files = {}
        pair_rows = []
        for (cell, drive) in sorted(dir_meta):
            ids = by_cell[cell][drive]
            rel = os.path.join(cell, '%s.ids.npy' % drive)
            os.makedirs(os.path.join(tmp, cell), exist_ok=True)
            np.save(os.path.join(tmp, rel),
                    ids)  # uint64, mmap_mode='r' loadable
            files[rel] = sha256_file(os.path.join(tmp, rel))
            meta = dir_meta[(cell, drive)]
            pair_rows.append({
                'cell': cell,
                'drive': drive,
                'region': meta['region'],
                'root': meta['root'],
                'dir_mtime': meta['dir_mtime'],
                'n_ids': int(ids.size),
            })

        with open(os.path.join(tmp, '_dirs.json'), 'w') as f:
            json.dump(pair_rows, f, indent=1, sort_keys=True)
        files['_dirs.json'] = sha256_file(os.path.join(tmp, '_dirs.json'))

        # 6.1: post-dedup ceil-sum -- authoritative, do not assert 8,979.
        n_shards = sum(
            math.ceil(r['n_ids'] / SHARD_SIZE) for r in pair_rows
            if r['n_ids'])
        meta = {
            'gen': gen,
            'built_at': datetime.now(timezone.utc).isoformat(),
            'shard_size': SHARD_SIZE,
            'n_images': total_unique,
            'n_pairs': len(pair_rows),
            'n_cells': len(by_cell),
            'n_shards': n_shards,
            'dedup': {
                'n_overlap_cells': len(dedup_cells),
                'n_contested_ids': n_contested,
                'n_removed': n_removed,
                'cells': dedup_cells,
            },
            'catalog': {
                'path': args.catalog,
                'asserted': asserted,
            },
            'pairs':
            pair_rows,  # per-pair dir_mtime lives in _meta.json too (5.2)
            'files': files,
            'sha256': combined_hash(files),
        }
        with open(os.path.join(tmp, '_meta.json'), 'w') as f:
            json.dump(meta, f, indent=1, sort_keys=True)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        os.rename(tmp, out)  # single rename = the freeze point
    except BaseException:
        # A failed freeze must leave no debris beside the frozen gens.
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    print('frozen %s' % out)
    print('  n_images %d  n_pairs %d  n_cells %d  n_shards %d' %
          (total_unique, len(pair_rows), len(by_cell), n_shards))
    print('  dedup: %d overlap cells, %d contested ids, %d occurrences '
          'removed' % (len(dedup_cells), n_contested, n_removed))
    print('  sha256 %s' % meta['sha256'])
    return 0


def cmd_verify(args):
    """Re-hash a frozen generation against _meta.json; non-zero on any drift.

    This is the same check the runner performs before starting (6.2: it
    refuses to run if the on-disk worklist hash differs from its manifest).
    """
    detect_root = resolve_detect_root(args)
    out = gen_dir(detect_root, args.gen)
    meta_path = os.path.join(out, '_meta.json')
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError) as exc:
        sys.exit('cannot read %s: %s' % (meta_path, exc))

    errors = []
    for rel, want in sorted(meta['files'].items()):
        path = os.path.join(out, rel)
        try:
            got = sha256_file(path)
        except OSError as exc:
            errors.append('%s: unreadable (%s)' % (rel, exc))
            continue
        if got != want:
            errors.append('%s: sha256 mismatch' % rel)
    if combined_hash(meta['files']) != meta['sha256']:
        errors.append('_meta.json: combined sha256 mismatch')

    # Stray ids.npy not covered by the meta would silently change what a
    # listdir-based tool sees -- a frozen gen must contain exactly its files.
    for dirpath, _, names in os.walk(out):
        for name in names:
            rel = os.path.relpath(os.path.join(dirpath, name), out)
            if rel not in meta['files'] and rel != '_meta.json':
                errors.append('%s: not in _meta.json files' % rel)

    if errors:
        print('VERIFY FAILED for %s:' % out, file=sys.stderr)
        for e in errors:
            print('  ' + e, file=sys.stderr)
        sys.exit(1)
    print('verify OK: %s (%d files, n_images %d, sha256 %s)' %
          (out, len(meta['files']), meta['n_images'], meta['sha256']))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--detect-root',
                        help='override DETECT_ROOT (default: read the '
                        'gitignored data/detect_root.txt)')

    b = sub.add_parser('build',
                       parents=[common],
                       help='enumerate, dedup and freeze a new generation')
    b.add_argument('--gen', type=int, help='generation number (default: next)')
    b.add_argument('--catalog', default=DEFAULT_CATALOG)
    b.add_argument('--roots-file', default=DEFAULT_ROOTS_FILE)
    b.add_argument('--dirs',
                   nargs='*',
                   help='explicit grid_runs roots (overrides --roots-file)')
    b.add_argument('--duckdb-python',
                   default=os.environ.get('DETECT_DUCKDB_PYTHON', 'python'),
                   help='interpreter with duckdb, used only if this one '
                   'lacks it (catalog reads ride mp14)')
    b.set_defaults(func=cmd_build)

    v = sub.add_parser('verify',
                       parents=[common],
                       help='re-hash a frozen generation')
    v.add_argument('--gen', type=int, required=True)
    v.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
