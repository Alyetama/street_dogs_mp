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


def remove_checks(bad, bd):
    """Deleting a dataset: gigabytes, one button, no undo."""
    tmp = tempfile.mkdtemp(prefix='adv_bd_rm_')
    old_env = os.environ.get('TRAINING_ROOT')
    os.environ['TRAINING_ROOT'] = tmp
    try:
        def make(name, bundle=True):
            d = os.path.join(tmp, name)
            for split in ('train', 'val'):
                os.makedirs(os.path.join(d, 'images', split))
                open(os.path.join(d, 'images', split, 'a.jpg'), 'w').write('x')
            open(os.path.join(d, 'dataset.yaml'), 'w').close()
            if bundle:
                os.makedirs(os.path.join(d, 'bundle'))
                with open(os.path.join(d, 'bundle', 'manifest.json'),
                          'w') as fh:
                    json.dump({'family': 'dogdet', 'kind': 'detect',
                               'built_at': 1, 'counts': {'total': 2}}, fh)
            return d

        make('dogdet_20260820_bbbbbb')
        base = make(bd.FAMILIES['dogdet']['base'])
        busy = make('dogdet_20260820_cccccc')
        outside = os.path.join(tempfile.mkdtemp(prefix='adv_bd_rm_out_'),
                               'elsewhere')
        os.makedirs(outside)
        # a directory in the training root that is NOT a dataset: notes,
        # scratch, somebody's export. The catalogue does not list it, and
        # that is the only thing standing between it and rmtree.
        notes = os.path.join(tmp, 'scratch_notes')
        os.makedirs(notes)
        open(os.path.join(notes, 'keep.txt'), 'w').write('mine')
        # ...and a dataset that is a symlink onto another drive, which this
        # machine has six of: following it deletes the original
        real = make('dogdet_20260820_dddddd')
        moved = os.path.join(os.path.dirname(outside), 'real_dataset')
        shutil.move(real, moved)
        linked = os.path.join(tmp, 'dogdet_20260820_dddddd')
        os.symlink(moved, linked)

        # THE BASE IS NOT DELETABLE. Every build starts from it.
        got = bd.remove(bd.FAMILIES['dogdet']['base'])
        if got['ok'] or not os.path.isdir(base):
            bad.append('A BASE DATASET WAS DELETED -- every future build for '
                       'that model starts from it')
        # nor is one a running job is reading
        got = bd.remove('dogdet_20260820_cccccc',
                        in_use=('dogdet_20260820_cccccc',))
        if got['ok'] or not os.path.isdir(busy):
            bad.append('a dataset was deleted out from under a running job')
        # nor a path, a traversal, or a name nobody built
        for bogus in ('../' + os.path.basename(tmp), '/etc', '', '.',
                      'never_built_this', outside, 'scratch_notes'):
            got = bd.remove(bogus)
            if got['ok']:
                bad.append('deleting accepted %r' % (bogus,))
        if not os.path.isfile(os.path.join(notes, 'keep.txt')):
            bad.append('A DIRECTORY THAT IS NOT A DATASET WAS DELETED out of '
                       'the training root')
        got = bd.remove('dogdet_20260820_dddddd')
        if got['ok'] or not os.path.isdir(moved):
            bad.append('a symlinked dataset was followed and the original on '
                       'the other drive was deleted')
        # ...and a link pointing at something INSIDE the root passed every
        # other check here: the name checked is the link's, and the resolved
        # path is still under the training root. It deleted the base.
        inside = os.path.join(tmp, 'dogdet_20260821_111111')
        os.symlink(base, inside)
        got = bd.remove('dogdet_20260821_111111')
        keeper = os.path.join(base, 'images', 'train', 'a.jpg')
        if got['ok'] or not os.path.isfile(keeper):
            bad.append('A LINK TO THE BASE DATASET WAS FOLLOWED AND THE BASE '
                       'WAS DELETED: %r' % (got,))
        # a finished dataset with a damaged record is not an unfinished build:
        # one is gigabytes that were fine, the other is wreckage
        hurt = make('dogdet_20260821_222222', bundle=False)
        os.makedirs(os.path.join(hurt, 'bundle'))
        with open(os.path.join(hurt, 'bundle', 'manifest.json'), 'w') as fh:
            fh.write('{"family": "dogd')
        row = [r for r in bd.catalogue()
               if r['id'] == 'dogdet_20260821_222222'][0]
        if row.get('unfinished'):
            bad.append('a finished dataset whose manifest is damaged is '
                       'offered as an unfinished build, safe to delete')
        if not row.get('damaged'):
            bad.append('a damaged manifest is not reported as damaged, so '
                       'nothing tells anybody the record is unreadable')
        # A BUILD IN PROGRESS looks exactly like a dead one from outside:
        # both are a generated name with no bundle. Its staging directory is
        # the difference, and it exists only while the build is running.
        live = make('dogdet_20260820_eeeeee', bundle=False)
        os.makedirs(live + '.stage')
        got = bd.remove('dogdet_20260820_eeeeee')
        if got['ok'] or not os.path.isdir(live):
            bad.append('A DATASET WAS DELETED OUT FROM UNDER THE BUILD THAT '
                       'WAS WRITING IT')
        # ...and the real one goes, with what it freed reported
        got = bd.remove('dogdet_20260820_bbbbbb')
        if not got['ok']:
            bad.append('a real dataset could not be deleted: %r' % (got,))
        elif os.path.isdir(os.path.join(tmp, 'dogdet_20260820_bbbbbb')):
            bad.append('a dataset reported as deleted is still on disk')
        elif not got.get('freed'):
            bad.append('the deletion does not say how much it freed')
    except Exception as e:                # noqa: BLE001
        bad.append('the remove checks threw %s: %s' % (type(e).__name__, e))
    finally:
        if old_env is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = old_env
        shutil.rmtree(tmp, ignore_errors=True)


def stop_checks(bad, bd):
    """A stopped build cleans up after itself.

    The page's stop button signals the whole process group, and the default
    action for a signal is to die where you stand: no unwinding, no finally,
    no cleanup. What is left is a multi-gigabyte staging copy and a half-built
    dataset directory. This drives the same signal at a process that has
    called the same setup, and checks the cleanup ran.
    """
    tmp = tempfile.mkdtemp(prefix='adv_bd_stop_')
    try:
        probe = os.path.join(tmp, 'probe.py')
        with open(probe, 'w') as fh:
            fh.write(
                'import os, signal, sys, time\n'
                'sys.path.insert(0, %r)\n'
                'import build_dataset as bd\n'
                'bd._dying_cleans_up(is_main=True)\n'
                'stage = %r\n'
                'os.makedirs(stage, exist_ok=True)\n'
                'try:\n'
                '    os.kill(os.getpid(), signal.SIGTERM)\n'
                '    time.sleep(5)\n'
                'finally:\n'
                '    import shutil; shutil.rmtree(stage, ignore_errors=True)\n'
                '    print("CLEANED")\n'
                % (DETECT, os.path.join(tmp, 'stage')))
        got = subprocess.run([sys.executable, probe], capture_output=True,
                             text=True, timeout=60)
        if 'CLEANED' not in (got.stdout or ''):
            bad.append('A STOPPED BUILD DIES WHERE IT STANDS -- its staging '
                       'copy and its half-built dataset stay on disk: %r %r'
                       % ((got.stdout or '')[-200:], (got.stderr or '')[-200:]))
        if os.path.isdir(os.path.join(tmp, 'stage')):
            bad.append('the staging directory survived the stop')
        # ...and importing this into the dashboard must not touch the
        # dashboard's own signal handlers
        import signal as _sig
        before = _sig.getsignal(_sig.SIGTERM)
        bd._dying_cleans_up()
        if _sig.getsignal(_sig.SIGTERM) is not before:
            bad.append('importing the builder rewired the signal handlers of '
                       'whatever is hosting it')
    except subprocess.TimeoutExpired:
        bad.append('the stop probe never returned, so the signal was ignored')
    except Exception as e:                # noqa: BLE001
        bad.append('the stop checks threw %s: %s' % (type(e).__name__, e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def wanted_checks(bad, bd):
    """Which drawn labels reach which model.

    One export, three readings. The leash model is asked leashed or unleashed
    ABOUT A DOG, so a goat somebody boxed as `other animal` is not a harder
    example for it -- it is not an example at all, and cutting one produces a
    crop of a goat filed under `leashed` or `unleashed`. The gate is asked dog
    or not, so the goat is exactly what it needs. The detector is asked where
    the dogs are, so the goat's box is dropped.
    """
    got = {f: bd.ls_wanted(f) for f in ('dogdet', 'dogbin', 'leash')}
    for name in bd.LS_NOT_DOG:
        if name in got['leash']:
            bad.append('THE LEASH MODEL TAKES %r -- it is asked leashed or '
                       'unleashed about a dog, so that crop would be filed as '
                       'one or the other' % (name,))
        if name in got['dogdet']:
            bad.append('the detector keeps %r boxes, so a one-class dog '
                       'detector is taught that a %s is a dog'
                       % (name, name))
        if got['dogbin'].get(name) != 'not_dog':
            bad.append('the gate does not read %r as not_dog: %r'
                       % (name, got['dogbin'].get(name)))
    for name in bd.LS_DOG:
        if got['leash'].get(name) != name.split()[0]:
            bad.append('%r does not become %r for the leash model: %r'
                       % (name, name.split()[0], got['leash'].get(name)))
        if got['dogbin'].get(name) != 'dog' or got['dogdet'].get(name) != 'dog':
            bad.append('%r is not a dog for the gate and the detector'
                       % (name,))
    # ...and the cutter must ask that question rather than keep its own copy
    tasks = [{'image': 'https://x/y/z.jpg', 'label': [
        {'x': 10, 'y': 10, 'width': 10, 'height': 10,
         'rectanglelabels': [name]}]}
        for name in list(bd.LS_DOG) + list(bd.LS_NOT_DOG)]
    kept = []
    real_frame = bd.ls_frame

    def fake_frame(task, index=None, s3=None, bucket=None):
        kept.append([r for b in (task.get(bd.LS_GROUP) or [])
                     for r in b.get('rectanglelabels') or []])
        return None                       # nothing to cut; we only want the ask

    tmp = tempfile.mkdtemp(prefix='adv_bd_want_')
    try:
        bd.ls_frame = fake_frame
        bd.ls_crops(tasks, bd.ls_wanted('leash'), os.path.join(tmp, 'out'))
        flat = [name for one in kept for name in one]
        for name in bd.LS_NOT_DOG:
            if name in flat:
                bad.append('the crop cutter reached for a %r frame while '
                           'cutting for the leash model' % (name,))
        if not any(n in flat for n in bd.LS_DOG):
            bad.append('the crop cutter skipped the dogs as well, so this '
                       'check proves nothing: %r' % (flat,))
    except Exception as e:                # noqa: BLE001
        bad.append('the wanted checks threw %s: %s' % (type(e).__name__, e))
    finally:
        bd.ls_frame = real_frame
        shutil.rmtree(tmp, ignore_errors=True)


def catalogue_checks(bad, bd):
    """Which datasets carry the hand-drawn boxes, and which quietly do not.

    Every build exports from Label Studio now, but the sets built before that
    are still on the selector and nothing about them says what is missing --
    and picking one trains the model on everything EXCEPT the boxes somebody
    drew by hand, which is a difference no number on the page would show.
    """
    tmp = tempfile.mkdtemp(prefix='adv_bd_cat_')
    old_env = os.environ.get('TRAINING_ROOT')
    os.environ['TRAINING_ROOT'] = tmp
    try:
        for name, ls in (('dogdet_withls_x', 5649), ('dogdet_without_y', None)):
            d = os.path.join(tmp, name)
            for split in ('train', 'val'):
                os.makedirs(os.path.join(d, 'images', split))
            open(os.path.join(d, 'dataset.yaml'), 'w').close()
            os.makedirs(os.path.join(d, 'bundle'))
            man = {'family': 'dogdet', 'kind': 'detect', 'built_at': 1,
                   'built_at_iso': 'probe', 'built_by': 'guard',
                   'counts': {'total': 0}}
            if ls is not None:
                man['label_studio'] = {'file': 'bundle/x.json',
                                       'counts': {'tasks': ls}}
            with open(os.path.join(d, 'bundle', 'manifest.json'), 'w') as fh:
                json.dump(man, fh)
        # a build that was killed outright: a generated name, images on disk,
        # and no bundle, because the bundle is the last thing a build writes
        half = os.path.join(tmp, 'dogdet_20260820_aaaaaa')
        for split in ('train', 'val'):
            os.makedirs(os.path.join(half, 'images', split))
        open(os.path.join(half, 'dataset.yaml'), 'w').close()
        rows = {r['id']: r for r in bd.catalogue('dogdet')}
        killed = rows.get('dogdet_20260820_aaaaaa')
        if killed is None:
            bad.append('a half-built dataset vanishes from the catalogue '
                       'entirely, so the gigabytes it is using are invisible')
        elif not killed.get('unfinished'):
            bad.append('A HALF-BUILT DATASET IS OFFERED LIKE ANY OTHER -- '
                       'training on it trains on however much was copied '
                       'before the build was stopped')
        for real in ('dogdet_withls_x', 'dogdet_without_y'):
            if rows.get(real, {}).get('unfinished'):
                bad.append('%s is a finished dataset reported as unfinished'
                           % (real,))
        for name in ('dogdet_withls_x', 'dogdet_without_y'):
            if name not in rows:
                bad.append('the catalogue lost %s' % (name,))
                return
        if rows['dogdet_withls_x'].get('label_studio') != 5649:
            bad.append('a dataset built from the hand-drawn boxes does not '
                       'say how many: %r'
                       % (rows['dogdet_withls_x'].get('label_studio'),))
        if rows['dogdet_without_y'].get('label_studio'):
            bad.append('a dataset built WITHOUT the hand-drawn boxes claims '
                       'to have them: %r'
                       % (rows['dogdet_without_y'].get('label_studio'),))
    except Exception as e:                # noqa: BLE001
        bad.append('the catalogue checks threw %s: %s' % (type(e).__name__, e))
    finally:
        if old_env is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = old_env
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
    real_head = bd.git_head
    calls = []
    # A detector build spends twenty-five minutes fetching frames, and work
    # carries on in the repo while it does. This stands in for a commit landing
    # mid-build: the manifest has to name the code that STARTED the build, or
    # it names a commit that had nothing to do with this dataset.
    head_now = ['the-commit-the-build-started-from']

    def fake_run(self, name, argv, cwd=None, env=None):
        head_now[0] = 'a-commit-somebody-made-during-the-build'
        calls.append({'name': name, 'argv': argv, 'env': env or {},
                      'cwd': cwd})
        if name == 'label studio export':
            # stand in for the real script: write the file it would have
            import subprocess as _sp
            _sp.run(['/bin/bash', argv[1]], cwd=cwd, capture_output=True)
            self.steps.append({'name': name, 'argv': argv, 'exit_code': 0,
                               'seconds': 0.0})
            return 0
        if name == 'prepare_detection_yolo_dataset':
            # it names its output after the export file, in the cwd
            stem = os.path.splitext(os.path.basename(
                argv[argv.index('-f') + 1]))[0]
            for split in ('train', 'val'):
                os.makedirs(os.path.join(cwd, stem, 'images', split),
                            exist_ok=True)
                open(os.path.join(cwd, stem, 'images', split, 'a.jpg'),
                     'w').close()
            self.steps.append({'name': name, 'argv': argv, 'exit_code': 0,
                               'seconds': 0.0})
            return 0
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

    # AN EXPORT SCRIPT THAT WORKS, so the chain under test is the whole one.
    # Without it ls_export finds nothing, the preparation step is skipped and
    # the ordering check grades a shorter chain than the real build runs.
    export = os.path.join(root, 'export_annotations.sh')
    with open(export, 'w') as fh:
        fh.write('#!/bin/bash\n'
                 # A LABEL NOBODY HAS TAUGHT THIS PROJECT ABOUT. Somebody
                 # adds a class in Label Studio and every build sees it the
                 # next time it exports.
                 'echo \'[{"image":"https://x/y/z.jpg","label":['
                 '{"x":10,"y":10,"width":10,"height":10,'
                 '"rectanglelabels":["unleashed dog"]},'
                 '{"x":20,"y":20,"width":10,"height":10,'
                 '"rectanglelabels":["horse"]}]}]\' > probe.json\n'
                 'echo probe.json\n')
    os.chmod(export, 0o755)
    try:
        bd.Runner.run = fake_run
        bd.git_head = lambda: {'commit': head_now[0], 'dirty': False}
        bd.stage_extras = lambda *a, **k: {'dog': 1, 'not_dog': 2}
        man = bd.build('dogdet', by='admin')
        got = (man.get('git') or {}).get('commit')
        if got != 'the-commit-the-build-started-from':
            bad.append('the manifest names %r as the code that built this '
                       'dataset -- read at the end, so a commit made while '
                       'the build ran renames what produced it' % (got,))
        names = [c['name'] for c in calls]
        # THE EXPORT IS FETCHED FIRST, AND PREPARED BEFORE ANYTHING ELSE
        if 'label studio export' not in names:
            bad.append('the build does not export the hand-drawn boxes: %r'
                       % (names,))
        # WHAT THE DETECTOR IS TOLD TO DROP. The preparer only takes a list
        # of classes to exclude, so a fixed pair meant a label added later
        # survived --single-class and became a dog: the detector taught that
        # a horse is one, and no count on the page would look any different.
        prep = [c for c in calls
                if c['name'] == 'prepare_detection_yolo_dataset']
        if prep:
            argv = prep[0]['argv']
            drop = (argv[argv.index('-e') + 1].split(',')
                    if '-e' in argv else [])
            if 'horse' not in drop:
                bad.append('a label this model never asked for (%r) is not '
                           'excluded, so --single-class turns it into a dog: '
                           'dropping %r' % ('horse', drop))
            for name in bd.LS_DOG:
                if name in drop:
                    bad.append('the detector drops %r, which is the only '
                               'thing it is being trained to find' % (name,))
        if 'prepare_detection_yolo_dataset' not in names:
            bad.append('the exported boxes are never prepared into a '
                       'detector set, so the detector is built from whatever '
                       'its base was cut from months ago: %r' % (names,))
        elif names.index('prepare_detection_yolo_dataset') > \
                names.index('build_dogdet_v3'):
            bad.append('the export is prepared AFTER the set is built from '
                       'it: %r' % (names,))
        else:
            prep = calls[names.index('prepare_detection_yolo_dataset')]
            argv = prep['argv']
            if '--single-class' not in argv:
                bad.append('the detector set is not built single-class -- it '
                           'finds dogs and nothing else')
            if '--background' not in argv:
                bad.append('the tasks marked background are dropped')
            # the exclusion is DERIVED: everything the export carried that
            # this model did not ask for, whatever it is called
            drop = set((argv[argv.index('-e') + 1].split(',')
                        if '-e' in argv else []))
            if 'horse' not in drop:
                bad.append('a label the detector never asked for survives '
                           'into its single class: dropping %r'
                           % (sorted(drop),))
            for name in bd.LS_DOG:
                if name in drop:
                    bad.append('the detector drops %r, the only thing it is '
                               'being trained to find' % (name,))
            if '--tracker-file' not in argv:
                bad.append('the split tracker is not passed, so train and '
                           'val move between rebuilds')
            if prep.get('cwd') != os.path.dirname(
                    calls[0].get('cwd') or '') and not str(
                    prep.get('cwd') or '').endswith('.stage'):
                bad.append('the preparation does not run in the staging '
                           'directory -- it names its output after the cwd '
                           'and writes dataset.yaml into it, so run anywhere '
                           'else it overwrites the one already there')
        names = [n for n in names
                 if n not in ('label studio export',
                              'prepare_detection_yolo_dataset')]
        if names != ['build_dogdet_v3', 'build_detector_negatives']:
            bad.append('the detector build ran %r -- the backgrounds pass '
                       'is what puts the false positives in, and it runs on '
                       'the OUTPUT of the first step' % (names,))
        else:
            # BY NAME, not by position. The export and the preparation run
            # first now, so calls[0] is no longer the first builder -- and
            # indexing by position quietly asked the export script for a
            # --out it does not have.
            def argv_of(want):
                for c in calls:
                    if c['name'] == want:
                        return c['argv']
                return []
            a1, a2 = argv_of('build_dogdet_v3'), \
                argv_of('build_detector_negatives')
            first_out = a1[a1.index('--out') + 1]
            second_src = a2[a2.index('--src') + 1]
            # ...and the set it builds from is the one prepared from the
            # export, not the base it would have used without one
            if '--src' not in a1 or '.stage' not in a1[a1.index('--src') + 1]:
                bad.append('build_dogdet_v3 was not pointed at the prepared '
                           'export: %r' % (a1,))
            if first_out != second_src:
                bad.append('the backgrounds pass does not read what the first '
                           'step wrote: %r vs %r' % (first_out, second_src))
            if '--execute' not in a2:
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
        # THE BUILDERS' OWN RECORDS ARE KEPT, under whichever names they
        # use. The detector's first step writes its into the staging
        # directory that is about to be deleted, and its second uses a name
        # of its own -- so a list naming only the crop builder's file kept
        # nothing at all for the detector, silently.
        kept = man.get('builder_manifests', {})
        if not kept:
            bad.append('no builder record was kept -- they hold the holdout '
                       'ids, the sequences and which frames were moved')
        elif 'split_manifest.json' not in kept:
            bad.append('the split record was lost with the staging '
                       'directory: kept %r' % (sorted(kept),))
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

        def argv_named(want):
            for c in calls:
                if c['name'] == want:
                    return c['argv']
            return []
        argv = argv_named('rebuild_crop_dataset')
        if not argv:
            bad.append('a crop build never ran the crop builder: %r'
                       % ([c['name'] for c in calls],))
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
        argv = argv_named('rebuild_crop_dataset')
        if not argv:
            bad.append('the leash build never ran the crop builder')
            argv = ['--pos-class', '?', '--neg-class', '?', '--src', '?']
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
        bd.git_head = real_head
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


def labelstudio_checks(bad, bd):
    """The hand-drawn boxes: read, cut, and kept.

    Two bugs here produced no error and no output -- which is the worst shape
    a bug can have in a build that takes four minutes.

    THE COORDINATES ARE PERCENTAGES OF THE IMAGE IN HAND. original_width is a
    note about the frame at annotation time; a frame found inside a built
    detector set has been resized to 1280, so scaling 4000-pixel coordinates
    onto it puts every box off the right edge. Clamped, that is a zero-width
    crop -- it cut nothing from every task whose frame was already here, and
    said so only by producing no files.

    AND THE FRAME HAS TO BE THE ORIGINAL. Cutting from the resized copy gives
    17x16 crops of a dog that was 79 pixels across, which is a classifier
    taught from thumbnails.
    """
    from PIL import Image
    tmp = tempfile.mkdtemp(prefix='adv_bd_ls_')
    try:
        # a frame, and a task whose box covers a known quarter of it
        os.makedirs(os.path.join(tmp, 'frames'))
        big = os.path.join(tmp, 'frames', 'probe.jpg')
        Image.new('RGB', (4000, 3000), (20, 30, 40)).save(big)
        task = {'image': 'https://example.invalid/bucket/probe.jpg',
                'label': [{'x': 25.0, 'y': 25.0, 'width': 50.0,
                           'height': 50.0, 'original_width': 4000,
                           'original_height': 3000,
                           'rectanglelabels': ['unleashed dog']}]}
        if bd.box_width_of(task) != 4000:
            bad.append('the recorded frame width is not read off a box')
        if bd.box_width_of({'label': []}) is not None:
            bad.append('a task with no box reports a frame width')
        # cut against the ORIGINAL: half of 4000 is 2000
        real_index = bd.ls_index
        real_s3 = bd._s3
        bd.ls_index = lambda: {'probe.jpg': big}
        bd._s3 = lambda: (None, None)
        try:
            out = os.path.join(tmp, 'out')
            got, missing = bd.ls_crops([task], {'unleashed dog': 'unleashed'},
                                       out)
            if got.get('unleashed') != 1:
                bad.append('a plain box was not cut: %r (%d without a frame)'
                           % (got, missing))
            else:
                cut = os.listdir(os.path.join(out, 'unleashed'))
                im = Image.open(os.path.join(out, 'unleashed', cut[0]))
                if abs(im.width - 2000) > 4 or abs(im.height - 1500) > 4:
                    bad.append('a box covering half the frame cut to %dx%d, '
                               'not 2000x1500' % (im.width, im.height))
            # NOW THE RESIZED COPY. Same task, a frame at 1280: the box still
            # covers half of it, so the crop is half of 1280 -- not a clamp to
            # nothing, which is what reading original_width produced.
            small = os.path.join(tmp, 'frames', 'small.jpg')
            Image.new('RGB', (1280, 960), (20, 30, 40)).save(small)
            task2 = json.loads(json.dumps(task))
            task2['image'] = 'https://example.invalid/bucket/small.jpg'
            bd.ls_index = lambda: {'small.jpg': small}
            out2 = os.path.join(tmp, 'out2')
            got, _ = bd.ls_crops([task2], {'unleashed dog': 'unleashed'}, out2)
            if got.get('unleashed') != 1:
                bad.append('a box on a resized frame cut nothing -- the '
                           'coordinates were scaled against a size the image '
                           'does not have')
            else:
                cut = os.listdir(os.path.join(out2, 'unleashed'))
                im = Image.open(os.path.join(out2, 'unleashed', cut[0]))
                if abs(im.width - 640) > 4:
                    bad.append('a half-frame box on a 1280 frame cut to %dpx, '
                               'not 640' % (im.width,))
            # a box too small to be a crop is skipped, not saved as a sliver.
            # The index has to hold THIS task's frame, or the check passes
            # because nothing was cut for an unrelated reason.
            bd.ls_index = lambda: {'probe.jpg': big, 'small.jpg': small}
            tiny = json.loads(json.dumps(task))
            tiny['label'][0].update(width=0.05, height=0.05)
            out3 = os.path.join(tmp, 'out3')
            got, _ = bd.ls_crops([tiny], {'unleashed dog': 'unleashed'}, out3)
            if got.get('unleashed'):
                bad.append('a box a few pixels across was cut anyway')
        finally:
            bd.ls_index = real_index
            bd._s3 = real_s3
        # ── the export is read for what it holds ──
        export = os.path.join(tmp, 'export.json')
        with open(export, 'w') as fh:
            json.dump([task, {'image': 'x/y.jpg', 'background': True},
                       {'image': 'x/z.jpg',
                        'label': [{'x': 1, 'y': 1, 'width': 9, 'height': 9,
                                   'rectanglelabels': ['other animal']}]}], fh)
        tasks, counts = bd.ls_read(export)
        if counts['tasks'] != 3 or counts['boxes'] != 2:
            bad.append('the export was counted as %r' % (counts,))
        if counts['background'] != 1:
            bad.append('tasks marked background are not counted')
        if counts['classes'].get('other animal') != 1:
            bad.append('the classes in the export are not counted: %r'
                       % (counts['classes'],))
        for junk in (b'not json', b'{"not": "a list"}'):
            bad_path = os.path.join(tmp, 'junk.json')
            open(bad_path, 'wb').write(junk)
            try:
                bd.ls_read(bad_path)
                bad.append('an export of %r was read as one' % (junk[:20],))
            except SystemExit:
                pass
        # ── the vocabulary is the one the project uses ──
        if 'other animal' not in bd.LS_NOT_DOG:
            bad.append('other animal is not a dog-bin negative -- 1,707 boxes '
                       'a person drew round something that is not a dog')
        if set(bd.LS_DOG) != {'leashed dog', 'unleashed dog'}:
            bad.append('the dog classes are %r' % (bd.LS_DOG,))
    except Exception as e:                # noqa: BLE001
        bad.append('the label studio checks threw %s: %s'
                   % (type(e).__name__, e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def progress_checks(bad, bd):
    """A progress bar is one line repeated, and it must not become the log.

    tqdm redraws with a carriage return, but a subprocess pipe is not a
    terminal, so every redraw arrives as its own line. Fetching three thousand
    frames wrote three thousand near-identical lines into the log and buried
    everything the build actually said -- and the page, which shows the tail,
    showed nothing but the bar.
    """
    for line, want in (
            ('Fetching Images:  10%|#  | 322/3078 [01:48<41:02, 1.12it/s]',
             ('322', '3078')),
            ('Creating Labels: 100%|##| 5649/5649 [00:03<00:00, 1600it/s]',
             ('5649', '5649')),
            ('wrote 2630 train / 480 val', None),
            ('PROGRESS 1 4 something', None),
            ('resolved 2,644/2,644 jpgs directly from the store', None)):
        got = bd._PROGRESS_RE.search(line)
        if want is None:
            if got:
                bad.append('a plain line was swallowed as a progress bar: %r'
                           % (line,))
        elif not got:
            bad.append('a tqdm line was not recognised: %r' % (line,))
        elif got.groups() != want:
            bad.append('a tqdm line read as %r, want %r'
                       % (got.groups(), want))

    # ...and the collapsing happens, with the count republished as this
    # step's own progress so a twenty-five minute step does not sit at "1 of 4"
    tmp = tempfile.mkdtemp(prefix='adv_bd_prog_')
    try:
        log = os.path.join(tmp, 'log.txt')
        run = bd.Runner(log, 4)
        run.progress(1, 'fetching frames')
        script = ('import sys\n'
                  'for i in range(200):\n'
                  '    sys.stdout.write("Fetching Images: %d%%|#| %d/200 '
                  '[00:01<00:02, 2.0it/s]\\n" % (i//2, i))\n'
                  'sys.stdout.write("done for real\\n")\n')
        path = os.path.join(tmp, 'noisy.py')
        open(path, 'w').write(script)
        run.run('noisy', [sys.executable, path])
        run.close()
        text = open(log).read()
        bars = [ln for ln in text.splitlines() if 'Fetching Images' in ln]
        if len(bars) > 20:
            bad.append('%d progress lines reached the log out of 200 -- the '
                       'bar is the log again' % (len(bars),))
        if 'done for real' not in text:
            bad.append('a real line was dropped along with the bar')
        subs = [ln for ln in text.splitlines()
                if ln.startswith('PROGRESS 1 4 fetching frames ')]
        if len(subs) > 20:
            bad.append('%d republished progress lines out of 200 -- the bar '
                       'moved into the log under a new name' % (len(subs),))
        if not subs:
            bad.append('the count inside the bar is not republished, so a '
                       'long step reports nothing while it runs')
        elif '/200' not in subs[-1]:
            bad.append('the republished progress carries no count: %r'
                       % (subs[-1],))
    except Exception as e:                # noqa: BLE001
        bad.append('the progress checks threw %s: %s' % (type(e).__name__, e))
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
                     (labelstudio_checks, (bd,)),
                     (progress_checks, (bd,)),
                     (catalogue_checks, (bd,)),
                     (wanted_checks, (bd,)),
                     (stop_checks, (bd,)),
                     (remove_checks, (bd,)),
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
