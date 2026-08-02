"""
Adversarial test for tools/detect/worklist.py: dedup edges the self-test
does not reach (no pytest; exit non-zero on failure).

* a 3-drive South_Asia overlap cell (bobcat+capybara+jackal -- the real
  corpus shape, 2.1) where one contested id is NOT on the preferred drive:
  it must resolve deterministically to capybara (alphabetical fallback),
  never be double-assigned, never be dropped everywhere;
* a Europe cell where lynx is the winner (the second WINNER_PREFERENCE
  entry, unexercised by the self-test);
* an image_id at the top of the uint64 range (2**64 - 3) surviving dedup
  bit-exact (catches any float/int64 coercion in np.isin/union1d);
* a cell present in catalog cell_images but on no drive -> build must
  HARD FAIL and name that cell in the per-cell delta;
* verify must fail when a listed ids.npy is deleted (not just tampered);
* build --gen 0 must be rejected, not silently auto-numbered.

Run with an interpreter that has numpy + duckdb (mp14).
"""

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

BIG = 2**64 - 3  # top of uint64; int64/float64 would mangle it

FAILURES = []


def check(cond, msg):
    status = 'ok' if cond else 'FAIL'
    print('%-4s %s' % (status, msg))
    if not cond:
        FAILURES.append(msg)


def run(args):
    return subprocess.run([sys.executable, WORKLIST] + args,
                          capture_output=True,
                          text=True)


def make_tree(base):
    roots = {
        d: os.path.join(base, 'root_%s' % d, 'grid_runs')
        for d in ('bobcat', 'capybara', 'jackal', 'lynx')
    }
    for drive, root in roots.items():
        os.makedirs(root)
        with open(os.path.join(root, '.detect_drive_id.json'), 'w') as f:
            json.dump({'drive': drive}, f)
    layout = {
        # 3-drive SA cell: 12 contested bobcat+capybara -> bobcat wins;
        # 14 contested capybara+jackal ONLY -> capybara (alphabetical
        # fallback after the preference drives), jackal drops it.
        ('South_Asia_60_20_65_25', 'bobcat'): [11, 12, 13],
        ('South_Asia_60_20_65_25', 'capybara'): [15, 12, 14],
        ('South_Asia_60_20_65_25', 'jackal'): [16, 14],
        # EU cell: 22 and BIG contested capybara+lynx -> lynx wins (2.1).
        ('Europe_0_40_5_45', 'lynx'): [21, 22, BIG],
        ('Europe_0_40_5_45', 'capybara'): [23, BIG, 22],
    }
    for (cell, drive), ids in layout.items():
        d = os.path.join(roots[drive], cell, 'ground_animal_images')
        os.makedirs(d)
        for i in ids:
            open(os.path.join(d, '%d.jpg' % i), 'wb').close()
    return roots, layout


def make_catalog(path, layout, uniques):
    con = duckdb.connect(path)
    con.sql('CREATE TABLE images (cell VARCHAR, drive VARCHAR, '
            'region VARCHAR, n_images BIGINT)')
    con.sql('CREATE TABLE cell_images (cell VARCHAR, region VARCHAR, '
            'n_unique BIGINT, n_drives INT)')
    for (cell, drive), ids in layout.items():
        con.execute(
            'INSERT INTO images VALUES (?, ?, ?, ?)',
            [cell, drive, cell.rsplit('_', 4)[0],
             len(ids)])
    for cell, n in uniques.items():
        con.execute('INSERT INTO cell_images VALUES (?, ?, ?, 1)',
                    [cell, cell.rsplit('_', 4)[0], n])
    con.close()


def main():
    base = tempfile.mkdtemp(prefix='worklist_adv_dedup_')
    try:
        roots, layout = make_tree(base)
        detect_root = os.path.join(base, 'detect')
        catalog = os.path.join(base, 'catalog.duckdb')
        uniques = {'South_Asia_60_20_65_25': 6, 'Europe_0_40_5_45': 4}
        make_catalog(catalog, layout, uniques)
        dirs = list(roots.values())

        p = run([
            'build', '--detect-root', detect_root, '--catalog', catalog,
            '--dirs'
        ] + dirs)
        check(p.returncode == 0, 'build succeeds\n%s%s' % (p.stdout, p.stderr))
        gen1 = os.path.join(detect_root, 'worklist', 'gen=0001')
        meta = json.load(open(os.path.join(gen1, '_meta.json')))

        sa = 'South_Asia_60_20_65_25'
        eu = 'Europe_0_40_5_45'
        load = lambda c, d: np.load(os.path.join(gen1, c, '%s.ids.npy' % d))
        check(
            set(load(sa, 'bobcat').tolist()) == {11, 12, 13},
            'SA: bobcat keeps its ids incl. contested 12')
        check(
            set(load(sa, 'capybara').tolist()) == {14, 15},
            'SA: capybara drops 12 (bobcat won) but KEEPS 14 (contested '
            'only with jackal; alphabetical fallback)')
        check(
            set(load(sa, 'jackal').tolist()) == {16},
            'SA: jackal drops 14 (capybara claimed it)')
        total_sa = (load(sa, 'bobcat').size + load(sa, 'capybara').size +
                    load(sa, 'jackal').size)
        check(total_sa == 6, 'SA: no id lost or double-assigned (6 kept)')

        lynx_eu = load(eu, 'lynx')
        check(
            set(lynx_eu.tolist()) == {21, 22, BIG},
            'EU: lynx (2nd preference) keeps contested 22 and BIG')
        check(
            set(load(eu, 'capybara').tolist()) == {23},
            'EU: capybara keeps only uncontested 23')
        check(lynx_eu.dtype == np.uint64 and BIG in lynx_eu.tolist(),
              'uint64 id 2**64-3 survives dedup bit-exact')
        check(
            meta['dedup']['cells'][sa]['winner'] == 'bobcat'
            and meta['dedup']['cells'][eu]['winner'] == 'lynx',
            'meta winners: bobcat (SA), lynx (EU)')
        check(meta['n_images'] == 10 and meta['dedup']['n_removed'] == 4,
              'totals: n_images=10, 4 duplicate occurrences removed')

        # Determinism with the 3-drive resolution in play.
        p = run([
            'build', '--detect-root', detect_root, '--catalog', catalog,
            '--dirs'
        ] + dirs)
        meta2 = json.load(
            open(
                os.path.join(detect_root, 'worklist', 'gen=0002',
                             '_meta.json')))
        check({
            k: v
            for k, v in meta['files'].items() if k.endswith('.npy')
        } == {
            k: v
            for k, v in meta2['files'].items() if k.endswith('.npy')
        }, '3-drive dedup rebuild is deterministic')

        # --gen 0 must be rejected, not treated as "auto".
        p = run([
            'build', '--gen', '0', '--detect-root', detect_root, '--catalog',
            catalog, '--dirs'
        ] + dirs)
        check(p.returncode != 0 and '--gen must be >= 1' in p.stderr,
              'build --gen 0 rejected')
        check(
            not os.path.exists(
                os.path.join(detect_root, 'worklist', 'gen=0000')),
            'no gen=0000 dir created')

        # Phantom cell: in cell_images but on no drive -> named in delta.
        phantom_cat = os.path.join(base, 'catalog_phantom.duckdb')
        bad = dict(uniques)
        bad['South_Asia_0_0_5_5'] = 5
        make_catalog(phantom_cat, layout, bad)
        p = run([
            'build', '--gen', '7', '--detect-root', detect_root, '--catalog',
            phantom_cat, '--dirs'
        ] + dirs)
        check(p.returncode != 0 and 'South_Asia_0_0_5_5' in p.stderr,
              'phantom catalog cell hard-fails and is named in the delta')

        # verify must fail on a DELETED (not just tampered) ids.npy.
        os.unlink(os.path.join(gen1, sa, 'jackal.ids.npy'))
        p = run(['verify', '--detect-root', detect_root, '--gen', '1'])
        check(p.returncode != 0 and 'unreadable' in p.stderr,
              'verify fails on a deleted ids.npy')
    finally:
        shutil.rmtree(base, ignore_errors=True)

    if FAILURES:
        print('\n%d FAILURE(S)' % len(FAILURES))
        return 1
    print('\nall checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
