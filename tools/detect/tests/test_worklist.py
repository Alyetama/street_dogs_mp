"""
Self-test for tools/detect/worklist.py (no pytest; exit non-zero on failure).

Builds a tiny fake tree -- 2 drives, 3 cells, one overlap cell with contested
ids -- plus a fake catalog with the expected unique counts, then checks:

* cross-drive dedup: contested ids assigned to bobcat (South_Asia winner,
  2.1), removed from capybara, non-contested ids untouched;
* per-dir inode ordering of the frozen ids.npy (2.2), via an independent
  os.scandir pass;
* mmap loadability (6.2: lanes open ids.npy with mmap_mode='r');
* _meta.json hash stability: combined sha256 recomputable from the per-file
  hashes, and a deterministic rebuild (gen 2) yields identical file hashes;
* verify passes on a pristine gen and detects a tampered ids.npy;
* build HARD FAILS printing the per-cell delta when the catalog's unique
  count disagrees (2.1).

Run with an interpreter that has numpy + duckdb (mp14):

    python tools/detect/tests/test_worklist.py
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import duckdb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKLIST = os.path.join(HERE, '..', 'worklist.py')

FAILURES = []


def check(cond, msg):
    status = 'ok' if cond else 'FAIL'
    print('%-4s %s' % (status, msg))
    if not cond:
        FAILURES.append(msg)


def run(args, **kw):
    return subprocess.run([sys.executable, WORKLIST] + args,
                          capture_output=True,
                          text=True,
                          **kw)


def make_tree(base):
    """Two fake drives, three cells; cell C overlaps with contested ids."""
    roots = {
        'capybara': os.path.join(base, 'rootA', 'grid_runs'),
        'bobcat': os.path.join(base, 'rootB', 'grid_runs'),
    }
    # Drive identity comes from the 6.4 sentinel so the test does not
    # depend on mount-point names under the temp dir.
    for drive, root in roots.items():
        os.makedirs(root)
        with open(os.path.join(root, '.detect_drive_id.json'), 'w') as f:
            json.dump({'drive': drive}, f)

    layout = {
        # cell A: capybara only. Created in shuffled order so readdir/inode
        # order differs from numeric order.
        ('South_Asia_70_20_75_25', 'capybara'): [905, 901, 909, 903, 907],
        # cell B: bobcat only.
        ('Europe_10_45_15_50', 'bobcat'): [104, 101, 103, 102],
        # cell C: overlap. Contested = {1004, 1005, 1006} -> bobcat wins
        # (South_Asia, 2.1); capybara must keep only 1001-1003.
        ('South_Asia_75_20_80_25', 'capybara'):
        [1004, 1001, 1006, 1002, 1005, 1003],
        ('South_Asia_75_20_80_25', 'bobcat'): [1007, 1004, 1006, 1005],
    }
    for (cell, drive), ids in layout.items():
        d = os.path.join(roots[drive], cell, 'ground_animal_images')
        os.makedirs(d)
        for i in ids:  # creation order != numeric order on purpose
            open(os.path.join(d, '%d.jpg' % i), 'wb').close()
        # A non-image and a non-numeric entry must be skipped, not crash.
        open(os.path.join(d, 'notes.txt'), 'wb').close()
        open(os.path.join(d, 'thumb_x.jpg'), 'wb').close()
    return roots, layout


def make_catalog(path, layout, uniques):
    con = duckdb.connect(path)
    con.sql('CREATE TABLE images (cell VARCHAR, drive VARCHAR, '
            'region VARCHAR, n_images BIGINT)')
    con.sql('CREATE TABLE cell_images (cell VARCHAR, region VARCHAR, '
            'n_unique BIGINT, n_drives INT)')
    for (cell, drive), ids in layout.items():
        region = cell.rsplit('_', 4)[0]
        con.execute('INSERT INTO images VALUES (?, ?, ?, ?)',
                    [cell, drive, region, len(ids)])
    for cell, n in uniques.items():
        con.execute('INSERT INTO cell_images VALUES (?, ?, ?, 1)',
                    [cell, cell.rsplit('_', 4)[0], n])
    con.close()


def inode_order_ids(dirpath):
    """Independent reference: (inode, id) via scandir, sorted by inode."""
    pairs = []
    with os.scandir(dirpath) as it:
        for e in it:
            if e.name.endswith('.jpg') and e.name[:-4].isdigit():
                pairs.append((e.inode(), int(e.name[:-4])))
    pairs.sort()
    return [i for _, i in pairs]


def combined_hash(files):
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(('%s:%s\n' % (rel, files[rel])).encode())
    return h.hexdigest()


def main():
    base = tempfile.mkdtemp(prefix='worklist_selftest_')
    try:
        roots, layout = make_tree(base)
        detect_root = os.path.join(base, 'detect')
        catalog = os.path.join(base, 'catalog.duckdb')
        uniques = {
            'South_Asia_70_20_75_25': 5,
            'Europe_10_45_15_50': 4,
            'South_Asia_75_20_80_25': 7,  # 6 + 4 raw, 3 contested
        }
        make_catalog(catalog, layout, uniques)
        dirs = list(roots.values())

        # ---- build gen 1 --------------------------------------------------
        p = run([
            'build', '--detect-root', detect_root, '--catalog', catalog,
            '--dirs'
        ] + dirs)
        check(
            p.returncode == 0, 'build gen 1 succeeds\n%s%s' %
            ('' if p.returncode == 0 else p.stdout,
             '' if p.returncode == 0 else p.stderr))
        gen1 = os.path.join(detect_root, 'worklist', 'gen=0001')
        meta = json.load(open(os.path.join(gen1, '_meta.json')))

        # ---- dedup assignment (2.1) --------------------------------------
        cellc = 'South_Asia_75_20_80_25'
        cap = np.load(os.path.join(gen1, cellc, 'capybara.ids.npy'))
        bob = np.load(os.path.join(gen1, cellc, 'bobcat.ids.npy'))
        check(
            set(cap.tolist()) == {1001, 1002, 1003},
            'contested ids removed from capybara (kept %s)' %
            sorted(cap.tolist()))
        check(
            set(bob.tolist()) == {1004, 1005, 1006, 1007},
            'contested ids kept on bobcat (winner)')
        check(cap.dtype == np.uint64 and bob.dtype == np.uint64,
              'ids.npy dtype is uint64')
        dd = meta['dedup']
        check(
            dd['n_overlap_cells'] == 1 and dd['n_contested_ids'] == 3
            and dd['n_removed'] == 3, 'dedup stats in _meta.json (%s)' % dd)
        check(dd['cells'][cellc]['winner'] == 'bobcat',
              'meta records bobcat as winner for the South_Asia cell')
        check(
            meta['n_images'] == 16 and meta['n_pairs'] == 4
            and meta['n_shards'] == 4,
            'totals: n_images=16 n_pairs=4 n_shards=4 (%d/%d/%d)' %
            (meta['n_images'], meta['n_pairs'], meta['n_shards']))

        # ---- inode order (2.2), against an independent scandir ------------
        for (cell, drive), _ in layout.items():
            d = os.path.join(roots[drive], cell, 'ground_animal_images')
            ref = inode_order_ids(d)
            got = np.load(os.path.join(gen1, cell, '%s.ids.npy' % drive),
                          mmap_mode='r')  # 6.2: must be mmap-loadable
            if cell == cellc and drive == 'capybara':
                # dedup masks contested ids but must preserve relative order
                ref = [i for i in ref if i in (1001, 1002, 1003)]
            check(
                list(got) == ref,
                'inode order preserved for %s/%s' % (cell, drive))
        # The shuffled creation order must actually exercise the sort (the
        # inode order should differ from numeric order somewhere).
        d = os.path.join(roots['capybara'], 'South_Asia_70_20_75_25',
                         'ground_animal_images')
        check(
            inode_order_ids(d) != sorted(inode_order_ids(d)),
            'fixture sanity: inode order differs from numeric order')

        # ---- meta hash stability -----------------------------------------
        check(
            combined_hash(meta['files']) == meta['sha256'],
            'combined sha256 recomputable from per-file hashes')
        p = run(['verify', '--detect-root', detect_root, '--gen', '1'])
        check(p.returncode == 0, 'verify passes on pristine gen 1')

        # Deterministic rebuild: same tree + catalog -> identical hashes.
        p = run([
            'build', '--detect-root', detect_root, '--catalog', catalog,
            '--dirs'
        ] + dirs)
        check(p.returncode == 0, 'build gen 2 (auto-numbered) succeeds')
        meta2 = json.load(
            open(
                os.path.join(detect_root, 'worklist', 'gen=0002',
                             '_meta.json')))
        ids_hashes = {
            k: v
            for k, v in meta['files'].items() if k.endswith('.ids.npy')
        }
        ids_hashes2 = {
            k: v
            for k, v in meta2['files'].items() if k.endswith('.ids.npy')
        }
        check(ids_hashes == ids_hashes2,
              'rebuild is deterministic (identical ids.npy hashes)')

        # Frozen: rebuilding an existing gen must be refused (6.2).
        p = run([
            'build', '--gen', '1', '--detect-root', detect_root, '--catalog',
            catalog, '--dirs'
        ] + dirs)
        check(p.returncode != 0 and 'frozen' in (p.stdout + p.stderr),
              'refuses to overwrite an existing (frozen) generation')

        # ---- verify detects tamper ---------------------------------------
        victim = os.path.join(gen1, cellc, 'bobcat.ids.npy')
        with open(victim, 'r+b') as f:
            f.seek(-1, os.SEEK_END)
            byte = f.read(1)
            f.seek(-1, os.SEEK_END)
            f.write(bytes([byte[0] ^ 0xFF]))
        p = run(['verify', '--detect-root', detect_root, '--gen', '1'])
        check(p.returncode != 0 and 'mismatch' in p.stderr,
              'verify detects a tampered ids.npy')
        with open(victim, 'r+b') as f:  # restore
            f.seek(-1, os.SEEK_END)
            f.write(byte)
        p = run(['verify', '--detect-root', detect_root, '--gen', '1'])
        check(p.returncode == 0, 'verify passes again after restore')
        # A stray file inside the frozen gen is also drift.
        stray = os.path.join(gen1, cellc, 'stray.ids.npy')
        open(stray, 'wb').close()
        p = run(['verify', '--detect-root', detect_root, '--gen', '1'])
        check(p.returncode != 0 and 'not in _meta.json' in p.stderr,
              'verify detects a stray file in the gen dir')
        os.unlink(stray)

        # ---- catalog mismatch HARD FAILS with per-cell delta (2.1) --------
        bad_catalog = os.path.join(base, 'catalog_bad.duckdb')
        bad = dict(uniques)
        bad['South_Asia_70_20_75_25'] = 6  # off by one
        make_catalog(bad_catalog, layout, bad)
        p = run([
            'build', '--gen', '9', '--detect-root', detect_root, '--catalog',
            bad_catalog, '--dirs'
        ] + dirs)
        check(p.returncode != 0, 'build hard-fails on catalog mismatch')
        check('South_Asia_70_20_75_25' in p.stderr and 'delta' in p.stderr,
              'mismatch failure prints the per-cell delta')
        check(
            not os.path.exists(
                os.path.join(detect_root, 'worklist', 'gen=0009')),
            'failed build leaves no frozen gen dir behind')
    finally:
        shutil.rmtree(base, ignore_errors=True)

    if FAILURES:
        print('\n%d FAILURE(S)' % len(FAILURES))
        return 1
    print('\nall checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
