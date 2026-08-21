#!/usr/bin/env python3
"""Composing a training set out of every annotation on record.

This does not run the real builders -- they read six drives and take minutes,
and a check that needs the corpus mounted is a check nobody runs. What it does
test is everything AROUND them, which is where the composer's own mistakes
live: which stores it reads, what it writes down about them, what it refuses,
and whether the numbers it reports are the numbers on disk.

Three things here have already been wrong once.

THE CLASS DIRECTORIES WERE READ AS FILES. A store laid out one directory per
class also holds a README and a manifest, and sorted() puts 'README.md' before
'dog' because capitals sort first -- so a check on the FIRST entry decided the
store was flat, and 344 crops were reported as 4.

LINES WERE COUNTED IN A DATABASE. leash.db is SQLite; counting newlines in it
produces a number that looks like a verdict count and is not one.

A BUILDER WAS HANDED A FLAG IT DOES NOT TAKE, and the one that cuts the crops
was run without --execute, which makes it print what it WOULD cut and write
nothing -- a build that then succeeds against an empty directory and produces
a dataset with none of the new annotations in it.

Run: python tools/detect/tests/adv_build_dataset.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DETECT = os.path.join(REPO, 'tools', 'detect')
sys.path.insert(0, DETECT)


def store_checks(bad, bd):
    """What a store looks like, including the two shapes it comes in."""
    tmp = tempfile.mkdtemp(prefix='adv_bd_store_')
    try:
        # a per-class store, with the README that broke the first version
        crops = os.path.join(tmp, 'finds')
        for klass, n in (('dog', 3), ('not_dog', 2)):
            os.makedirs(os.path.join(crops, klass))
            for i in range(n):
                open(os.path.join(crops, klass, '%s_%d.jpg' % (klass, i)),
                     'w').close()
        open(os.path.join(crops, 'README.md'), 'w').write('hi')
        ledger = os.path.join(tmp, 'verdicts.jsonl')
        with open(ledger, 'w') as fh:
            fh.write('{"a":1}\n\n{"a":2}\n')
        bd.STORES['_probe'] = {'ledger': ledger, 'crops': crops,
                               'for': ('dogbin',)}
        got = bd.store_state('_probe')
        if not isinstance(got['files'], dict):
            bad.append('a store laid out one directory per class read as a '
                       'flat list -- the README sorts before the classes')
        elif sorted(got['files']) != ['dog', 'not_dog']:
            bad.append('the classes found were %r' % (sorted(got['files']),))
        elif len(got['files']['dog']) != 3:
            bad.append('the crops in a class were miscounted: %r'
                       % (got['files'],))
        if got['lines'] != 2:
            bad.append('blank lines are counted as verdicts: %r'
                       % (got['lines'],))
        # THE DIGEST IS THE POINT. A ledger that changed since a build must
        # not produce the digest that build recorded.
        first = got['sha256']
        if not first:
            bad.append('a store has no digest, so "rebuilt from the same '
                       'annotations" cannot be checked against anything')
        with open(ledger, 'a') as fh:
            fh.write('{"a":3}\n')
        if bd.store_state('_probe')['sha256'] == first:
            bad.append('adding a verdict did not change the digest')

        # a flat store, and a store that is not there at all
        flat = os.path.join(tmp, 'flat')
        os.makedirs(flat)
        open(os.path.join(flat, 'a.jpg'), 'w').close()
        bd.STORES['_probe'] = {'ledger': os.path.join(tmp, 'nope.jsonl'),
                               'crops': flat, 'for': ('dogbin',)}
        got = bd.store_state('_probe')
        if got['files'] != ['a.jpg']:
            bad.append('a flat store read as %r' % (got['files'],))
        if got['sha256'] is not None or got['lines'] is not None:
            bad.append('a ledger that does not exist reported a digest')
        # a database is not a text file
        db = os.path.join(tmp, 'leash.db')
        open(db, 'wb').write(b'SQLite format 3\x00\n\n\n')
        bd.STORES['_probe'] = {'ledger': db, 'crops': None,
                               'for': ('dogbin',)}
        if bd.store_state('_probe')['lines'] is not None:
            bad.append('newlines in a SQLite file are reported as verdicts')
    finally:
        bd.STORES.pop('_probe', None)
        shutil.rmtree(tmp, ignore_errors=True)


def name_checks(bad, bd):
    """Every build is a new directory, and the name says which."""
    seen = {bd.new_name('dogbin') for _ in range(200)}
    if len(seen) != 200:
        bad.append('two builds in the same second collided: %d of 200 names'
                   % (len(seen),))
    one = bd.new_name('leash')
    if not one.startswith('leash_') or len(one.split('_')) != 3:
        bad.append('a dataset name does not carry the family, the day and an '
                   'id: %r' % (one,))
    if not one.split('_')[-1].isalnum():
        bad.append('the id is not something a directory can be called: %r'
                   % (one,))


def measure_checks(bad, bd):
    """The reported size is the size on disk, in the shape the model reads."""
    tmp = tempfile.mkdtemp(prefix='adv_bd_measure_')
    try:
        # a classify set
        for split, per in (('train', {'dog': 4, 'not_dog': 3}),
                           ('val', {'dog': 2, 'not_dog': 1})):
            for klass, n in per.items():
                d = os.path.join(tmp, split, klass)
                os.makedirs(d)
                for i in range(n):
                    open(os.path.join(d, '%d.jpg' % i), 'w').close()
        files, got = bd.inventory(tmp, 'dogbin')
        if got['total'] != 10:
            bad.append('a 10-image classify set measured %r' % (got['total'],))
        if got['splits']['train']['classes'] != {'dog': 4, 'not_dog': 3}:
            bad.append('the train split measured %r'
                       % (got['splits']['train'],))
        if got['classes'] != {'dog': 6, 'not_dog': 4}:
            bad.append('the per-class totals across splits are %r'
                       % (got['classes'],))
        if len(files['train']['dog']) != 4:
            bad.append('the file list does not hold every image: %r'
                       % (files['train']['dog'],))
        # EVERY listed image carries a digest. Two builds are the same
        # dataset when their file lists match, and a list of names without
        # digests cannot tell a re-cut crop from the same one.
        for row in files['train']['dog']:
            if not row.get('sha256') or not row.get('name'):
                bad.append('a listed crop has no digest or no name: %r'
                           % (row,))
                break
        if abs(got['splits']['val']['share'] - 0.3) > 0.001:
            bad.append('the val share reads %r, not 0.3'
                       % (got['splits']['val']['share'],))
        # a detect set, where an EMPTY label file is a background
        det = tempfile.mkdtemp(prefix='adv_bd_det_')
        for split, n, bg in (('train', 5, 2), ('val', 3, 0)):
            os.makedirs(os.path.join(det, 'images', split))
            os.makedirs(os.path.join(det, 'labels', split))
            for i in range(n):
                open(os.path.join(det, 'images', split, '%d.jpg' % i),
                     'w').close()
                with open(os.path.join(det, 'labels', split, '%d.txt' % i),
                          'w') as fh:
                    if i >= bg:
                        fh.write('0 0.5 0.5 0.1 0.1\n')
        dfiles, got = bd.inventory(det, 'dogdet')
        if got['total'] != 8:
            bad.append('an 8-image detect set measured %r' % (got['total'],))
        if got['splits']['train']['classes'].get('background') != 2:
            bad.append('backgrounds are not counted as the empty label files '
                       'they are: %r' % (got['splits']['train'],))
        rows = dfiles['train']['images']
        if len(rows) != 5 or any(r.get('boxes') is None for r in rows):
            bad.append('the detector listing does not say how many boxes '
                       'each frame carries: %r' % (rows[:2],))
        shutil.rmtree(det, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def refusal_checks(bad, bd):
    """What must never happen quietly."""
    tmp = tempfile.mkdtemp(prefix='adv_bd_refuse_')
    root = os.path.join(tmp, 'root')
    os.makedirs(os.path.join(root, 'dogbin_v5'))
    old = os.environ.get('TRAINING_ROOT')
    os.environ['TRAINING_ROOT'] = root
    # THE BUILDERS ARE STUBBED HERE TOO. Every check below expects a refusal
    # before any of them runs -- so with the refusal removed, an unstubbed
    # check does not fail, it starts reading six drives and hangs. A mutation
    # that hangs proves nothing.
    real_run = bd.Runner.run
    bd.Runner.run = lambda self, name, argv, cwd=None, env=None: (
        _ for _ in ()).throw(AssertionError('a builder ran past a refusal'))
    try:
        for fam in ('nope', '', None, 'dogdet; rm -rf /'):
            try:
                bd.build(fam)
                bad.append('a build of family %r was attempted' % (fam,))
            except SystemExit:
                pass
        # NEVER BUILT OVER. A dataset is the evidence for whatever trained on
        # it, and a name that gets reused is a run whose evidence is gone.
        there = os.path.join(root, 'already_here')
        os.makedirs(there)
        try:
            bd.build('dogbin', out=there)
            bad.append('a build wrote into a directory that already exists')
        except SystemExit as e:
            if 'exists' not in str(e):
                bad.append('building over a directory failed for the wrong '
                           'reason: %s' % (e,))
        # a missing base is a sentence, not a traceback halfway through
        try:
            bd.build('leash')
            bad.append('a build ran with no base dataset to derive from')
        except SystemExit as e:
            if 'base' not in str(e).lower():
                bad.append('a missing base said: %s' % (e,))
    except AssertionError as e:
        bad.append(str(e))
    finally:
        bd.Runner.run = real_run
        if old is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = old
        shutil.rmtree(tmp, ignore_errors=True)

    # ...and no training root at all is a sentence naming the key to set
    keep = os.environ.pop('TRAINING_ROOT', None)
    cfg = os.path.join(REPO, 'tools', 'dashboard', 'dashboard.config.json')
    hidden = cfg + '.adv_hidden'
    moved = False
    try:
        if os.path.exists(cfg):
            os.rename(cfg, hidden)
            moved = True
        try:
            bd.training_root()
            bad.append('an unconfigured training root was guessed at')
        except SystemExit as e:
            if 'TRAINING_ROOT' not in str(e):
                bad.append('the unconfigured message names no key: %s' % (e,))
    finally:
        if moved:
            os.rename(hidden, cfg)
        if keep is not None:
            os.environ['TRAINING_ROOT'] = keep


def composition_checks(bad, bd):
    """WHAT gets run, in what order, with which flags.

    The builders are stubbed: this is about the composer's decisions, and the
    builders have their own guards. Every argv is captured and read back.
    """
    tmp = tempfile.mkdtemp(prefix='adv_bd_compose_')
    root = os.path.join(tmp, 'root')
    for base in ('dogdet_v2', 'dogbin_v5', 'leash_v2'):
        os.makedirs(os.path.join(root, base))
    old_env = os.environ.get('TRAINING_ROOT')
    os.environ['TRAINING_ROOT'] = root
    real_run = bd.Runner.run
    real_stage = bd.stage_extras
    calls = []

    def fake_run(self, name, argv, cwd=None, env=None):
        calls.append({'name': name, 'argv': argv, 'env': env or {}})
        # leave behind whatever the real builder would have, so the composer's
        # own measuring and manifest work has something to read
        out = argv[argv.index('--out') + 1] if '--out' in argv else None
        if out:
            if name == 'build_dogdet_v3' or name == 'build_detector_negatives':
                for split in ('train', 'val'):
                    os.makedirs(os.path.join(out, 'images', split),
                                exist_ok=True)
                    os.makedirs(os.path.join(out, 'labels', split),
                                exist_ok=True)
                    open(os.path.join(out, 'images', split, 'a.jpg'),
                         'w').close()
                    open(os.path.join(out, 'labels', split, 'a.txt'),
                         'w').close()
                with open(os.path.join(out, 'manifest.json'), 'w') as fh:
                    json.dump({'holdout': ['x'], 'seed': 0}, fh)
            elif name == 'rebuild_crop_dataset':
                for split in ('train', 'val'):
                    for klass in ('a', 'b'):
                        os.makedirs(os.path.join(out, split, klass),
                                    exist_ok=True)
                        open(os.path.join(out, split, klass, '1.jpg'),
                             'w').close()
                with open(os.path.join(out, 'rebuild_manifest.json'),
                          'w') as fh:
                    json.dump({'kept': 7}, fh)
        self.steps.append({'name': name, 'argv': argv, 'exit_code': 0,
                           'seconds': 0.0})
        return 0

    try:
        bd.Runner.run = fake_run
        bd.stage_extras = lambda *a, **k: {'dog': 1, 'not_dog': 2}
        man = bd.build('dogdet', by='admin')
        names = [c['name'] for c in calls]
        if names != ['build_dogdet_v3', 'build_detector_negatives']:
            bad.append('the detector build ran %r -- the backgrounds pass '
                       'is what puts the false positives in, and it runs on '
                       'the OUTPUT of the first step' % (names,))
        else:
            first_out = calls[0]['argv'][calls[0]['argv'].index('--out') + 1]
            second_src = calls[1]['argv'][calls[1]['argv'].index('--src') + 1]
            if first_out != second_src:
                bad.append('the backgrounds pass does not read what the first '
                           'step wrote: %r vs %r' % (first_out, second_src))
            if '--execute' not in calls[1]['argv']:
                bad.append('build_detector_negatives ran without --execute, '
                           'so it printed what it would add and added '
                           'nothing')
        if not man['id'].startswith('dogdet_'):
            bad.append('the manifest is not named for the family: %r'
                       % (man['id'],))
        for key in ('stores', 'steps', 'counts', 'git', 'base', 'built_by'):
            if key not in man:
                bad.append('the manifest records no %r' % (key,))
        if man['built_by'] != 'admin':
            bad.append('the manifest does not say who asked for it')
        if not man['stores']:
            bad.append('the manifest digests no annotation store, so what it '
                       'was built from cannot be checked later')
        for name, st in man['stores'].items():
            if st['sha256'] is None and st['lines'] is None:
                continue
            if st['sha256'] is None:
                bad.append('%s is recorded without a digest' % (name,))
        # the builder's own record is kept rather than paraphrased
        if 'manifest.json' not in man.get('builder_manifests', {}):
            bad.append("the builder's own manifest is not kept -- it holds "
                       'the holdout ids and the sequences')
        # ── THE BUNDLE ──
        bundle = os.path.join(man['out'], 'bundle')
        for who in ('manifest.json', 'files.json', 'inputs.json',
                    'build_log.txt'):
            if not os.path.isfile(os.path.join(bundle, who)):
                bad.append('bundle/%s is not in the dataset' % (who,))
        with open(os.path.join(bundle, 'inputs.json')) as fh:
            got = json.load(fh)
        if not got.get('stores'):
            bad.append('bundle/inputs.json lists no store')
        with open(os.path.join(bundle, 'files.json')) as fh:
            listed = json.load(fh)['files']
        # EVERY image, by split, named. A count is not a manifest.
        named = sum(len(v) if isinstance(v, list)
                    else sum(len(x) for x in v.values())
                    for split in listed.values()
                    for v in [split.get('images', split)]
                    if True)
        flat = 0
        for split, per in listed.items():
            if 'images' in per:
                flat += len(per['images'])
            else:
                flat += sum(len(x) for x in per.values())
        if flat != man['counts']['total']:
            bad.append('the file list holds %d images and the counts say %d '
                       '-- one of them is wrong and neither says which'
                       % (flat, man['counts']['total']))
        for split, per in listed.items():
            for klass, rows in per.items():
                for row in rows:
                    if not row.get('sha256'):
                        bad.append('%s/%s/%s is listed without a digest, so '
                                   'two builds cannot be compared'
                                   % (split, klass, row.get('name')))
                        break
        # ...and the listing cannot drift from the manifest unnoticed
        if man.get('files_json', {}).get('sha256') != bd.sha256(
                os.path.join(bundle, 'files.json')):
            bad.append('the manifest does not carry the digest of the file '
                       'list, so the list can be edited without anything '
                       'noticing')
        for key in ('built_at_iso', 'timezone', 'command', 'versions'):
            if not man.get(key):
                bad.append('the manifest records no %r' % (key,))
        if man['command'].get('argv') is None:
            bad.append('the manifest does not record the command that ran')
        c = man['counts']
        if 'classes' not in c or 'splits' not in c or 'total' not in c:
            bad.append('the counts are not per class, per split and overall: '
                       '%r' % (sorted(c),))
        else:
            summed = sum(p['total'] for p in c['splits'].values())
            if summed != c['total']:
                bad.append('the splits add to %d and the total says %d'
                           % (summed, c['total']))
        # NOTHING IS LEFT BEHIND. The staging directory holds a whole second
        # copy of a detector set -- 1.7GB -- and leaving it doubles the disk
        # every build costs.
        if os.path.exists(man['out'] + '.stage'):
            bad.append('the staging directory was left behind')

        calls[:] = []
        man = bd.build('dogbin')
        argv = calls[0]['argv']
        if calls[0]['name'] != 'rebuild_crop_dataset':
            bad.append('a crop build ran %r' % (calls[0]['name'],))
        elif '--execute' not in argv:
            bad.append('rebuild_crop_dataset ran without --execute')
        else:
            if argv[argv.index('--pos-class') + 1] != 'dog' or \
                    argv[argv.index('--neg-class') + 1] != 'not_dog':
                bad.append('the dog-bin build names the wrong classes: %r'
                           % (argv,))
            if argv[argv.index('--src') + 1] != os.path.join(root,
                                                             'dogbin_v5'):
                bad.append('the dog-bin build derives from the wrong base')
        calls[:] = []
        man = bd.build('leash')
        argv = calls[0]['argv']
        if argv[argv.index('--pos-class') + 1] != 'leashed' or \
                argv[argv.index('--neg-class') + 1] != 'unleashed':
            bad.append('the leash build names the wrong classes: %r' % (argv,))
        if argv[argv.index('--src') + 1] != os.path.join(root, 'leash_v2'):
            bad.append('the leash build derives from the wrong base')
    except Exception as e:                # noqa: BLE001 - report, not die
        bad.append('composition threw %s: %s' % (type(e).__name__, e))
    finally:
        bd.Runner.run = real_run
        bd.stage_extras = real_stage
        if old_env is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = old_env
        shutil.rmtree(tmp, ignore_errors=True)


def harvest_checks(bad, bd):
    """How the crops get cut, which the composition check stubs past.

    Three flags decide whether this step does anything at all, and getting any
    of them wrong produces a build that SUCCEEDS with an empty extras
    directory -- a dataset with none of the new annotations in it, and nothing
    on screen to say so.
    """
    tmp = tempfile.mkdtemp(prefix='adv_bd_harvest_')
    calls = []

    class FakeRun:
        steps = []

        def say(self, text):
            pass

        def run(self, name, argv, cwd=None, env=None):
            calls.append({'name': name, 'argv': argv, 'env': env or {}})
            # leave the crops the real cutter would have written
            out = argv[argv.index('--out') + 1]
            for klass in ('dog', 'not_dog'):
                d = os.path.join(out, klass)
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, 'cut_%s.jpg' % (klass,)), 'w').close()
            return 0

    try:
        got = bd.stage_extras('dogbin', tmp, FakeRun(), '/duck/python',
                              '/crop/python')
        if len(calls) != 2:
            bad.append('the crop cutter ran %d times, not once per verdict '
                       'ledger' % (len(calls),))
        for call in calls:
            argv = call['argv']
            if '--execute' not in argv:
                bad.append('harvest_flagged ran without --execute -- it '
                           'prints what it WOULD cut and writes nothing, so '
                           'the build succeeds with no new crops in it')
            if '--duckdb-python' in argv:
                bad.append('harvest_flagged was handed --duckdb-python, '
                           'which it does not take: it exits 2 and the whole '
                           'build fails')
            if argv[0] != '/crop/python':
                bad.append('the crop cutter did not run in the interpreter '
                           'that has cv2: %r' % (argv[0],))
            if call['env'].get('DETECT_DUCKDB_PYTHON') != '/duck/python':
                bad.append('the cutter cannot reach the store: no helper '
                           'interpreter in its environment (%r)'
                           % (call['env'],))
        if not got or sum(got.values()) == 0:
            bad.append('nothing was staged from a cutter that wrote crops')
        for klass in ('dog', 'not_dog'):
            if not os.path.isdir(os.path.join(tmp, 'extras', klass)):
                bad.append('no staged directory for %r, so the builder is '
                           'given nothing to add' % (klass,))
        # ONE CROP IS ONE EXAMPLE. A crop flagged in the queue and found again
        # in the audit is the same picture, and staging it twice weights it
        # twice.
        seen = os.listdir(os.path.join(tmp, 'extras', 'dog'))
        if len(seen) != len(set(seen)):
            bad.append('a crop was staged twice')
    except Exception as e:                # noqa: BLE001
        bad.append('harvest staging threw %s: %s' % (type(e).__name__, e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cli_checks(bad):
    """The two read-only modes work against the real stores on this machine."""
    for args in (['--list'], ['--family', 'dogbin', '--dry-run']):
        got = subprocess.run([sys.executable,
                              os.path.join(DETECT, 'build_dataset.py')] + args,
                             capture_output=True, text=True, timeout=120)
        if got.returncode != 0:
            bad.append('build_dataset.py %s exited %d: %s'
                       % (' '.join(args), got.returncode,
                          got.stderr.strip()[-300:]))
        elif not got.stdout.strip():
            bad.append('build_dataset.py %s printed nothing'
                       % (' '.join(args),))
    # --dry-run must not create anything
    got = subprocess.run([sys.executable,
                          os.path.join(DETECT, 'build_dataset.py'),
                          '--family', 'leash', '--dry-run'],
                         capture_output=True, text=True, timeout=120)
    for line in (got.stdout or '').splitlines():
        if line.startswith('would build '):
            made = line.split()[2]
            if os.path.exists(os.path.join(line.split()[-1], made)):
                bad.append('--dry-run created the dataset it described')


def main():
    try:
        import build_dataset as bd
    except Exception as e:                # noqa: BLE001
        print('FAIL could not import build_dataset: %s: %s'
              % (type(e).__name__, e))
        return 1
    bad = []
    for fn, args in ((store_checks, (bd,)), (name_checks, (bd,)),
                     (measure_checks, (bd,)), (refusal_checks, (bd,)),
                     (composition_checks, (bd,)), (harvest_checks, (bd,)),
                     (cli_checks, ())):
        try:
            fn(bad, *args)
        except Exception as e:            # noqa: BLE001 - report, not die
            bad.append('%s threw %s: %s' % (fn.__name__, type(e).__name__, e))
    if bad:
        for b in bad:
            print('FAIL ' + b)
        return 1
    print('one command per model composes the builders in the right order '
          'with the flags that make them write, every store it read is '
          'digested into the dataset, and nothing is ever built over')
    return 0


if __name__ == '__main__':
    sys.exit(main())
