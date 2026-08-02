"""
Adversarial test for tools/detect/worklist.py: the 6.2 catalog-unreadable
fallback path, unexercised by the self-test (no pytest; exit non-zero on
failure). Takes ~30 s: the spec's 5-retry backoff sleeps are real.

* --catalog pointed at a garbage file -> 5 retries, then the depth-1
  scandir fallback must still build a correct generation;
* _meta.json must record catalog.asserted == false;
* dedup must still run on the fallback path (it needs no catalog);
* a root-level dir without ground_animal_images (lost+found) is skipped;
* an EMPTY ground_animal_images dir yields a 0-length uint64 ids.npy that
  contributes no shards;
* verify passes on the fallback-built generation.

Run with an interpreter that has numpy + duckdb (mp14) -- duckdb must be
importable so the in-process path (not the subprocess path) does the
failing/retrying, matching how a real locked-catalog build behaves.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKLIST = os.path.join(HERE, '..', 'worklist.py')

FAILURES = []


def check(cond, msg):
    status = 'ok' if cond else 'FAIL'
    print('%-4s %s' % (status, msg))
    if not cond:
        FAILURES.append(msg)


def main():
    base = tempfile.mkdtemp(prefix='worklist_adv_fallback_')
    try:
        roots = {
            'capybara': os.path.join(base, 'rootA', 'grid_runs'),
            'lynx': os.path.join(base, 'rootB', 'grid_runs'),
        }
        for drive, root in roots.items():
            os.makedirs(root)
            with open(os.path.join(root, '.detect_drive_id.json'), 'w') as f:
                json.dump({'drive': drive}, f)
        layout = {
            ('Europe_20_50_25_55', 'capybara'): [301, 303, 302],
            # overlap cell: 402 contested -> lynx wins (2.1)
            ('Europe_25_50_30_55', 'capybara'): [401, 402],
            ('Europe_25_50_30_55', 'lynx'): [403, 402],
        }
        for (cell, drive), ids in layout.items():
            d = os.path.join(roots[drive], cell, 'ground_animal_images')
            os.makedirs(d)
            for i in ids:
                open(os.path.join(d, '%d.jpg' % i), 'wb').close()
        # A root-level dir with no images subdir must be skipped ...
        os.makedirs(os.path.join(roots['capybara'], 'lost+found'))
        # ... and an EMPTY images dir must survive as a 0-shard pair.
        os.makedirs(
            os.path.join(roots['lynx'], 'Europe_30_50_35_55',
                         'ground_animal_images'))

        garbage = os.path.join(base, 'not_a_catalog.duckdb')
        with open(garbage, 'wb') as f:
            f.write(b'this is not a duckdb file')

        detect_root = os.path.join(base, 'detect')
        print('building via fallback (expect ~30 s of 6.2 retry backoff)')
        p = subprocess.run([
            sys.executable, WORKLIST, 'build', '--detect-root', detect_root,
            '--catalog', garbage, '--dirs'
        ] + list(roots.values()),
                           capture_output=True,
                           text=True)
        check(p.returncode == 0,
              'fallback build succeeds\n%s%s' % (p.stdout, p.stderr))
        check('fallback' in p.stderr and 'DISABLED' in p.stderr,
              'stderr announces the scandir fallback (assertions off)')

        gen1 = os.path.join(detect_root, 'worklist', 'gen=0001')
        meta = json.load(open(os.path.join(gen1, '_meta.json')))
        check(meta['catalog']['asserted'] is False,
              '_meta.json records catalog.asserted = false')
        check(meta['n_images'] == 6,
              'n_images correct without catalog (6, post-dedup)')
        check(meta['n_pairs'] == 4,
              'empty-images-dir pair is included (n_pairs=4)')
        check(meta['n_shards'] == 3,
              'empty pair contributes no shard (n_shards=3)')

        ov = 'Europe_25_50_30_55'
        lynx = np.load(os.path.join(gen1, ov, 'lynx.ids.npy'))
        cap = np.load(os.path.join(gen1, ov, 'capybara.ids.npy'))
        check(
            set(lynx.tolist()) == {402, 403} and set(cap.tolist()) == {401},
            'dedup still runs on the fallback path (lynx keeps 402)')

        empty = np.load(
            os.path.join(gen1, 'Europe_30_50_35_55', 'lynx.ids.npy'))
        check(empty.size == 0 and empty.dtype == np.uint64,
              'empty dir freezes as a 0-length uint64 ids.npy')

        cells = {r['cell'] for r in meta['pairs']}
        check('lost+found' not in cells,
              'root-level dir without images subdir is skipped')
        regions = {r['cell']: r['region'] for r in meta['pairs']}
        check(
            regions.get(ov) == 'Europe',
            'fallback parses region from the cell name')

        p = subprocess.run([
            sys.executable, WORKLIST, 'verify', '--detect-root', detect_root,
            '--gen', '1'
        ],
                           capture_output=True,
                           text=True)
        check(p.returncode == 0, 'verify passes on the fallback-built gen')
    finally:
        shutil.rmtree(base, ignore_errors=True)

    if FAILURES:
        print('\n%d FAILURE(S)' % len(FAILURES))
        return 1
    print('\nall checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
