#!/usr/bin/env python3
"""Launching a training run: what it inherits, what it refuses, what it records.

Nothing here trains. A run is hours on a GPU somebody else may be using, and a
check that needs one is a check nobody runs -- so this drives everything up to
the moment ultralytics is handed the parameters, plus the record written
afterwards, which is where every mistake this file can make actually lives.

THE PARAMETERS ARE NOT A LIST IN THE SOURCE. They are read out of the newest
run's args.yaml, so the next run starts from whatever the last good one was
launched with. A hardcoded list would be right until somebody trains a better
run and silently wrong after that.

THE KEYS ARE ULTRALYTICS' OWN. Checked against DEFAULT_CFG_DICT in the
installed version rather than against the documentation page, which describes
several versions at once: 8.4.115 has no `label_smoothing`, and a parameter
this refuses is better than a parameter ultralytics ignores -- an ignored one
is a run that quietly trained something else.

AND THE VALUES COME FROM A WEB FORM, so a key that decides where the run
writes or what it reads is refused by name, not coerced.

Run: python tools/detect/tests/adv_train_model.py
"""
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


def training_python():
    """The interpreter that has ultralytics. Without one, half of this file
    cannot run -- and says so rather than passing."""
    got = os.environ.get('DOGBIN_PYTHON')
    if got and os.path.exists(got):
        return got
    try:
        with open(os.path.join(REPO, 'tools', 'dashboard',
                               'dashboard.config.json')) as fh:
            cfg = json.load(fh)
        for key in ('dogbin_python', 'sweep_python', 'confusion_python'):
            got = cfg.get(key)
            if got and os.path.exists(got):
                return got
    except (OSError, ValueError):
        pass
    return None


def _run(py, args, timeout=180):
    return subprocess.run([py, os.path.join(DETECT, 'train_model.py')] + args,
                          capture_output=True, text=True, timeout=timeout)


def inherit_checks(bad, tm):
    """What a run starts from, and what it must never start from."""
    tmp = tempfile.mkdtemp(prefix='adv_tm_inherit_')
    root = os.path.join(tmp, 'root')
    proj = os.path.join(root, 'runs', 'detect', 'dogdetection')
    os.makedirs(os.path.join(proj, 'older'))
    os.makedirs(os.path.join(proj, 'newer'))
    os.makedirs(os.path.join(proj, 'newest_but_died'))
    old = os.environ.get('TRAINING_ROOT')
    os.environ['TRAINING_ROOT'] = root
    try:
        with open(os.path.join(proj, 'older', 'args.yaml'), 'w') as fh:
            fh.write('epochs: 10\nimgsz: 320\n')
        with open(os.path.join(proj, 'newer', 'args.yaml'), 'w') as fh:
            fh.write('epochs: 500\nimgsz: 1280\noptimizer: SGD\nbatch: 2\n'
                     'single_cls: true\n'
                     # every one of these belongs to the RUN, not to how it
                     # trains: inherited, they point the next run at the last
                     # one's dataset and directory
                     'data: /somewhere/old/dataset.yaml\n'
                     'project: dogdetection\nname: older\n'
                     'save_dir: %s\nexist_ok: true\nresume: true\n' % (proj,))
        # BOTH OF THOSE ACTUALLY TRAINED. ultralytics writes args.yaml before
        # it does anything, so without this the fixture is two runs that never
        # started and the question being asked is not the real one.
        for name in ('older', 'newer'):
            with open(os.path.join(proj, name, 'results.csv'), 'w') as fh:
                fh.write('epoch,train/box_loss\n1,0.9\n')
        # ...and the NEWEST one died on its parameters, leaving a complete
        # args.yaml and nothing else. This is not hypothetical: a detector run
        # died on batch_size=2.0 and every run after it was offered that same
        # batch size as the parameters to start from.
        with open(os.path.join(proj, 'newest_but_died', 'args.yaml'),
                  'w') as fh:
            fh.write('epochs: 1\nimgsz: 64\nbatch: 2.0\n')
        os.utime(os.path.join(proj, 'older', 'args.yaml'), (1, 1))
        got, where = tm.last_args('dogdet')
        if where and 'newest_but_died' in where:
            bad.append('THE PARAMETERS COME FROM A RUN THAT CRASHED before it '
                       'trained a single epoch -- the page calls these the '
                       'best parameters and hands over the ones that failed')
        if not where or 'newer' not in where:
            bad.append('the parameters were inherited from %r, not the newest '
                       'run' % (where,))
        if got.get('epochs') != '500':
            bad.append('the newest run was not the one read: %r'
                       % (got.get('epochs'),))
        # a project where NOTHING ever trained falls back to ultralytics'
        # own defaults rather than to the wreckage of the last attempt
        empty = os.path.join(root, 'runs', 'classify', 'dog-bin', 'died')
        os.makedirs(empty)
        with open(os.path.join(empty, 'args.yaml'), 'w') as fh:
            fh.write('epochs: 1\nbatch: 2.0\n')
        got, where = tm.last_args('dogbin')
        if where is not None or got:
            bad.append('a project whose only run crashed still hands over its '
                       'parameters: %r %r' % (where, got))
        # ...and a results.csv with a header and no epoch is not a run either
        head = os.path.join(root, 'runs', 'classify', 'leash_models', 'head')
        os.makedirs(head)
        with open(os.path.join(head, 'args.yaml'), 'w') as fh:
            fh.write('epochs: 1\n')
        with open(os.path.join(head, 'results.csv'), 'w') as fh:
            fh.write('epoch,train/loss\n')
        got, where = tm.last_args('leash')
        if where is not None:
            bad.append('a run whose results.csv holds only a header reads as '
                       'having trained: %r' % (where,))
    except Exception as e:                # noqa: BLE001
        bad.append('inheriting threw %s: %s' % (type(e).__name__, e))
    finally:
        if old is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = old
        shutil.rmtree(tmp, ignore_errors=True)


def dataset_checks(bad, tm):
    """Which dataset, and whether it is one this model can read at all."""
    tmp = tempfile.mkdtemp(prefix='adv_tm_data_')
    root = os.path.join(tmp, 'root')
    old = os.environ.get('TRAINING_ROOT')
    os.environ['TRAINING_ROOT'] = root
    try:
        # a classify set, properly bundled
        crop = os.path.join(root, 'dogbin_x')
        os.makedirs(os.path.join(crop, 'train', 'dog'))
        os.makedirs(os.path.join(crop, 'bundle'))
        with open(os.path.join(crop, 'bundle', 'manifest.json'), 'w') as fh:
            json.dump({'family': 'dogbin', 'counts': {'total': 4},
                       'built_at_iso': 'x'}, fh)
        got = tm.find_dataset('dogbin', 'dogbin_x')
        if got['data'] != crop:
            bad.append('a classify run is pointed at %r, not the directory'
                       % (got['data'],))
        if not got['manifest_sha256']:
            bad.append('the dataset manifest is not digested, so a run '
                       'cannot say which build it trained on')
        # A DETECTOR MUST NOT BE POINTED AT CROPS. It fails deep inside
        # ultralytics with a message about channels, hours later.
        try:
            tm.find_dataset('dogdet', 'dogbin_x')
            bad.append('a detector was pointed at a classification dataset')
        except SystemExit as e:
            # The message has to name the MISMATCH. A refusal that happens to
            # mention the dataset because its path is in the message is a
            # different check passing by accident -- a crop set also has no
            # dataset.yaml, so that guard catches this one and hides it.
            if 'was built for' not in str(e):
                bad.append('a detector on a crop dataset is refused for the '
                           'wrong reason (%s) -- the family check is not the '
                           'one doing it' % (e,))
        # a detect set without its yaml is not a detect set
        det = os.path.join(root, 'dogdet_x')
        os.makedirs(os.path.join(det, 'images', 'train'))
        try:
            tm.find_dataset('dogdet', 'dogdet_x')
            bad.append('a detector dataset with no dataset.yaml was accepted')
        except SystemExit:
            pass
        for missing in ('nope', '../../etc', ''):
            try:
                tm.find_dataset('dogbin', missing)
                bad.append('a dataset of %r resolved' % (missing,))
            except SystemExit:
                pass
        # a run is named for what it trained on, and two runs never collide
        one = tm.run_name('dogbin', 'dogbin_x')
        if 'dogbin_x' not in one:
            bad.append('a run is not named for its dataset: %r' % (one,))
    except Exception as e:                # noqa: BLE001
        bad.append('dataset resolution threw %s: %s' % (type(e).__name__, e))
    finally:
        if old is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = old
        shutil.rmtree(tmp, ignore_errors=True)


def batch_checks(bad, py):
    """batch is a COUNT or a FRACTION, and torch will not take a float count.

    ultralytics lists batch as a float so 0 < batch < 1 can mean "use that
    share of the card" and -1 can mean "work it out". Coerced as a plain
    float, inheriting `batch: 2` from a previous run gave 2.0 -- and every
    detector run then died at the first dataloader with "batch_size should be
    a positive integer value, but got 2.0". Found by accident, when a mutation
    test launched a real run; pinned here so it is not found that way twice.
    """
    probe = (
        'import json,sys; sys.path.insert(0, %r); import train_model as t\n'
        'tb=t._cfg_tables()\n'
        'print(json.dumps([[repr(t._coerce("batch",v,tb)),\n'
        '                   type(t._coerce("batch",v,tb)).__name__]\n'
        '                  for v in ["2","2.0",2,"16","auto","-1","0.7"]]))\n'
        % (DETECT,))
    out = subprocess.run([py, '-c', probe], capture_output=True, text=True,
                         timeout=180)
    try:
        got = json.loads((out.stdout or '').strip().splitlines()[-1])
    except (ValueError, IndexError):
        bad.append('could not coerce batch at all: %s'
                   % ((out.stderr or '').strip()[-200:],))
        return
    want = [('2', 'int'), ('2', 'int'), ('2', 'int'), ('16', 'int'),
            ('-1', 'int'), ('-1', 'int'), ('0.7', 'float')]
    for (gv, gt), (wv, wt) in zip(got, want):
        if gt != wt or gv != wv:
            bad.append('batch coerced to %s %s, want %s %s -- a float count '
                       'is rejected by torch at the first dataloader'
                       % (gv, gt, wv, wt))


def form_checks(bad, py):
    """--show-defaults is what the dashboard draws, so it has to be usable."""
    got = _run(py, ['--family', 'dogdet', '--show-defaults'])
    if got.returncode != 0:
        bad.append('--show-defaults exited %d: %s'
                   % (got.returncode, got.stderr.strip()[-300:]))
        return
    try:
        doc = json.loads(got.stdout)
    except ValueError:
        bad.append('--show-defaults did not print JSON: %r'
                   % (got.stdout[:200],))
        return
    if not doc.get('fields'):
        bad.append('the form has no fields')
        return
    keys = {f['key'] for f in doc['fields']}
    for need in ('epochs', 'batch', 'imgsz', 'lr0', 'optimizer', 'patience'):
        if need not in keys:
            bad.append('%s is not offered, and it is the one people change'
                       % (need,))
    for f in doc['fields']:
        for k in ('key', 'value', 'default', 'from', 'type', 'why'):
            if k not in f:
                bad.append('the %s field carries no %r' % (f.get('key'), k))
                break
        if f.get('type') not in ('bool', 'int', 'float', 'fraction', 'text'):
            bad.append('%s has no type the form can draw: %r'
                       % (f.get('key'), f.get('type')))
    if not doc.get('ultralytics'):
        bad.append('the form does not say which ultralytics it describes -- '
                   'the parameters differ between versions')
    # EVERY OFFERED KEY IS A REAL ONE. A field the form draws and ultralytics
    # refuses is a run that dies on submit.
    probe = ('import json,sys; sys.path.insert(0, %r); import train_model as t;'
             'd=t._cfg_tables()["defaults"];'
             'print(json.dumps([k for k,_ in t.EDITABLE if k not in d]))'
             % (DETECT,))
    out = subprocess.run([py, '-c', probe], capture_output=True, text=True,
                         timeout=180)
    try:
        missing = json.loads((out.stdout or '[]').strip().splitlines()[-1])
    except (ValueError, IndexError):
        missing = None
    if missing is None:
        bad.append('could not check the offered keys against ultralytics: %s'
                   % (out.stderr.strip()[-200:],))
    elif missing:
        bad.append('the form offers %r, which ultralytics %s does not have'
                   % (missing, doc.get('ultralytics')))


def refusal_checks(bad, py, dataset):
    """What must never reach ultralytics."""
    base = ['--family', 'dogbin', '--dataset', dataset, '--dry-run']
    for setting, why in (
            ('label_smoothing=0.1', 'a key this version does not have'),
            ('nonsense=1', 'a key nothing has'),
            ('epochs=lots', 'a whole number that is not one'),
            ('lr0=abc', 'a number that is not one'),
            ('fliplr=5', 'a fraction outside 0..1'),
            ('data=/etc/passwd', 'the dataset, which the run decides'),
            ('project=/tmp/elsewhere', 'where the run writes'),
            ('save_dir=/tmp/elsewhere', 'where the run writes'),
            ('resume=true', 'whether it continues something else')):
        got = _run(py, base + ['--set', setting])
        if got.returncode == 0:
            bad.append('%s was accepted (%s)' % (setting, why))
        elif 'refused' not in (got.stderr or ''):
            bad.append('%s failed without saying why: %s'
                       % (setting, (got.stderr or '').strip()[-160:]))
    # ...and a refusal starts nothing
    got = _run(py, base + ['--set', 'nonsense=1'])
    if 'nothing was started' not in (got.stderr or ''):
        bad.append('a refused parameter does not say that nothing ran')
    # a good one resolves
    got = _run(py, base + ['--set', 'epochs=3', '--set', 'batch=8'])
    if got.returncode != 0:
        bad.append('a valid dry run failed: %s' % (got.stderr.strip()[-300:],))
    elif 'nothing was started' not in got.stdout:
        bad.append('--dry-run did not say it started nothing')
    for line in ('epochs', 'batch'):
        if line not in got.stdout:
            bad.append('--dry-run does not show %r, so nobody can check what '
                       'they are about to run' % (line,))


def unfinished_checks(bad, tm):
    """A build that was killed is not a dataset, wherever it is asked from.

    The page can grey it out, but the page is not the gate: the launcher takes
    a name over HTTP and has to refuse it itself. What is on disk is however
    many images had been copied when the build died -- a plausible-looking
    fraction of a dataset, with no bundle, which is the last thing a build
    writes.
    """
    tmp = tempfile.mkdtemp(prefix='adv_tm_unfin_')
    old = os.environ.get('TRAINING_ROOT')
    os.environ['TRAINING_ROOT'] = tmp
    try:
        half = os.path.join(tmp, 'dogdet_20260820_aaaaaa')
        for split in ('train', 'val'):
            os.makedirs(os.path.join(half, 'images', split))
        open(os.path.join(half, 'dataset.yaml'), 'w').close()
        try:
            tm.find_dataset('dogdet', 'dogdet_20260820_aaaaaa')
            bad.append('AN UNFINISHED BUILD WAS ACCEPTED FOR TRAINING -- '
                       'hours of GPU on whatever fraction of a dataset was '
                       'copied before the build was stopped')
        except SystemExit as e:
            if 'unfinished' not in str(e):
                bad.append('an unfinished build was refused for the wrong '
                           'reason: %s' % (e,))
        # ...and a dataset from before this tool existed still trains: it has
        # no bundle either, and it is not something this tool half-wrote
        legacy = os.path.join(tmp, 'dogdet_v3')
        for split in ('train', 'val'):
            os.makedirs(os.path.join(legacy, 'images', split))
        open(os.path.join(legacy, 'dataset.yaml'), 'w').close()
        try:
            tm.find_dataset('dogdet', 'dogdet_v3')
        except SystemExit as e:
            bad.append('a legacy dataset is refused as unfinished: %s' % (e,))
    except Exception as e:                # noqa: BLE001
        bad.append('the unfinished checks threw %s: %s'
                   % (type(e).__name__, e))
    finally:
        if old is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = old
        shutil.rmtree(tmp, ignore_errors=True)


def resume_checks(bad, tm, py):
    """Continuing a stopped run, and every way that must not be one."""
    tmp = tempfile.mkdtemp(prefix='adv_tm_res_')
    old = os.environ.get('TRAINING_ROOT')
    os.environ['TRAINING_ROOT'] = tmp
    try:
        proj = os.path.join(tmp, 'runs', 'detect', 'dogdetection')
        good = os.path.join(proj, 'stopped_halfway')
        os.makedirs(os.path.join(good, 'weights'))
        open(os.path.join(good, 'weights', 'last.pt'), 'w').close()
        with open(os.path.join(good, 'args.yaml'), 'w') as fh:
            fh.write('epochs: 100\n')
        if not tm.resumable('dogdet', 'stopped_halfway'):
            bad.append('a run with weights/last.pt beside its args.yaml is '
                       'reported as unresumable -- half a day of GPU thrown '
                       'away')
        # A RUN THAT REACHED ITS LAST EPOCH HAS NOTHING TO CONTINUE either.
        # ultralytics asserts on it, but only after a minute of scanning the
        # dataset, and only after a job has been recorded for it.
        with open(os.path.join(good, 'results.csv'), 'w') as fh:
            fh.write('epoch,x\n' + ''.join('%d,0.1\n' % (i + 1)
                                           for i in range(100)))
        if tm.resumable('dogdet', 'stopped_halfway'):
            bad.append('a run that reached its last epoch is offered as '
                       'resumable')
        # ...and one that stopped early is
        with open(os.path.join(good, 'results.csv'), 'w') as fh:
            fh.write('epoch,x\n' + ''.join('%d,0.1\n' % (i + 1)
                                           for i in range(40)))
        if not tm.resumable('dogdet', 'stopped_halfway'):
            bad.append('a run that stopped at 40 of 100 epochs is not offered '
                       'as resumable')
        os.remove(os.path.join(good, 'results.csv'))
        # a run that never finished an epoch has nothing to continue from
        nothing = os.path.join(proj, 'died_at_once')
        os.makedirs(nothing)
        with open(os.path.join(nothing, 'args.yaml'), 'w') as fh:
            fh.write('epochs: 100\n')
        if tm.resumable('dogdet', 'died_at_once'):
            bad.append('a run with no weights is offered as resumable')
        # a real run belonging to ANOTHER model, reachable only by walking
        # out of this family's project directory: resumed as a detector, a
        # classifier run is a different task on a different dataset shape
        other = os.path.join(tmp, 'runs', 'classify', 'dog-bin', 'someone_else')
        os.makedirs(os.path.join(other, 'weights'))
        open(os.path.join(other, 'weights', 'last.pt'), 'w').close()
        with open(os.path.join(other, 'args.yaml'), 'w') as fh:
            fh.write('epochs: 100\n')
        for bogus in ('../../etc', '', None, 'never_ran',
                      os.path.join('..', '..', 'classify', 'dog-bin',
                                   'someone_else')):
            if tm.resumable('dogdet', bogus):
                bad.append('resuming accepted %r -- a run belonging to '
                           'another model, or no run at all' % (bogus,))
        try:
            tm.resume('dogdet', 'died_at_once')
            bad.append('a run with nothing to continue from was resumed')
        except SystemExit:
            pass
        # THE DATASET IT WILL READ HAS TO STILL BE THERE. ultralytics answers
        # a checkpoint whose dataset has gone by quietly substituting its own
        # default -- coco8.yaml for a detector, imagenet10 for a classifier --
        # so a resume after the dataset was deleted trains on somebody else's
        # toy data and writes the result over this run's weights.
        with open(os.path.join(good, 'args.yaml'), 'w') as fh:
            fh.write('epochs: 100\ndata: %s\n'
                     % (os.path.join(tmp, 'gone', 'dataset.yaml'),))
        if tm.recorded_data(good) is None:
            bad.append('the dataset a run trained on is not readable from the '
                       'run, so a resume cannot pin it')
        try:
            tm.resume('dogdet', 'stopped_halfway')
            bad.append('A RESUME RAN WITH ITS DATASET MISSING -- ultralytics '
                       'substitutes its own default and the result lands on '
                       'top of this run')
        except SystemExit as e:
            if 'gone' not in str(e) and 'is gone' not in str(e):
                bad.append('a resume with a missing dataset failed for the '
                           'wrong reason: %s' % (e,))
        # A RESUME TAKES NOTHING. ultralytics continues from what the run
        # recorded, and epochs trained on two different settings would land
        # in one results.csv.
        for extra, why in ((['--set', 'epochs=5'], 'parameters'),
                           (['--weights', 'x.pt'], 'other weights'),
                           (['--dataset', 'other_v1'], 'another dataset')):
            got = subprocess.run(
                [py, os.path.join(DETECT, 'train_model.py'),
                 '--family', 'dogdet', '--resume', 'stopped_halfway'] + extra,
                capture_output=True, text=True, timeout=120,
                env=dict(os.environ, TRAINING_ROOT=tmp))
            # on the message, not the exit code: with the refusal gone this
            # still fails, just later and for an unrelated reason
            said = (got.stderr or '') + (got.stdout or '')
            if 'a resume takes no' not in said:
                bad.append('a resume accepted %s, so the epochs after the '
                           'interruption would train on different settings '
                           'than the ones before it: %r'
                           % (why, said[-160:]))
    except subprocess.TimeoutExpired:
        bad.append('the resume probe never returned')
    except Exception as e:                # noqa: BLE001
        bad.append('the resume checks threw %s: %s' % (type(e).__name__, e))
    finally:
        if old is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = old
        shutil.rmtree(tmp, ignore_errors=True)


def weights_record_checks(bad, tm):
    """A checkpoint identified by its content, not by what it was called."""
    tmp = tempfile.mkdtemp(prefix='adv_tm_w_')
    try:
        # a name that exists NOWHERE else: ultralytics resolves a bare name
        # against the working directory first, and the repo root really does
        # hold a yolo26x.pt, so using that name here would grade the wrong file
        real = os.path.join(tmp, 'adv_probe_checkpoint.pt')
        with open(real, 'wb') as fh:
            fh.write(b'not a checkpoint, but it is a file')
        got = tm.weights_record('adv_probe_checkpoint.pt', root=tmp)
        if got.get('path') != real:
            bad.append('a bare checkpoint name was not resolved against the '
                       'training root: %r' % (got,))
        if not got.get('sha256') or got.get('bytes') != 34:
            bad.append('the checkpoint was found but not digested: %r'
                       % (got,))
        by_path = tm.weights_record(real, root=tmp)
        if by_path.get('sha256') != got.get('sha256'):
            bad.append('the same checkpoint digests differently by path and '
                       'by name: %r %r' % (by_path, got))
        missing = tm.weights_record('nothing_like_this.pt', root=tmp)
        if missing.get('path') or missing.get('sha256'):
            bad.append('a checkpoint that is not there was reported as '
                       'found: %r' % (missing,))
        if missing.get('name') != 'nothing_like_this.pt':
            bad.append('an unresolvable checkpoint loses even its name')
    except Exception as e:                # noqa: BLE001
        bad.append('the weights checks threw %s: %s' % (type(e).__name__, e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def save_dir_checks(bad, py):
    """NOTHING may create the run directory before ultralytics does.

    ultralytics is handed exist_ok=False and increments the name when the
    directory is already there. A bundle copied in before the first epoch sent
    the entire run to `<name>-2` -- weights, results.csv, args.yaml -- while
    the record sat in `<name>`, and the page, which looks up the name the
    dashboard chose, showed a run with no score and nothing to resume while it
    was training normally.
    """
    tmp = tempfile.mkdtemp(prefix='adv_tm_dir_')
    root = os.path.join(tmp, 'root')
    ds = os.path.join(root, 'dogbin_probe')
    for split in ('train', 'val'):
        for klass in ('dog', 'not_dog'):
            os.makedirs(os.path.join(ds, split, klass))
    os.makedirs(os.path.join(ds, 'bundle'))
    with open(os.path.join(ds, 'bundle', 'manifest.json'), 'w') as fh:
        json.dump({'family': 'dogbin', 'counts': {'total': 0}}, fh)
    probe = (
        'import json,os,sys; sys.path.insert(0, %r)\n'
        'import train_model as t, ultralytics\n'
        'from ultralytics.utils.files import increment_path\n'
        'seen = {}\n'
        'class Boom:\n'
        '    def __init__(self, *a, **k):\n'
        '        pass\n'
        '    def train(self, **kw):\n'
        # exactly what ultralytics does with project/name/exist_ok
        '        p = str(increment_path(os.path.join(kw["project"], '
        'kw["name"]), exist_ok=kw["exist_ok"]))\n'
        '        seen["dir"] = os.path.basename(p)\n'
        '        os.makedirs(p, exist_ok=True)\n'
        '        raise RuntimeError("probe")\n'
        'ultralytics.YOLO = Boom\n'
        'try:\n'
        '    t.launch("dogbin", %r, overrides={"epochs": 1}, name="myrun",\n'
        '             weights="x.pt", by="guard")\n'
        'except BaseException:\n'
        '    pass\n'
        'print(json.dumps({"chose": seen.get("dir"), "dirs": sorted(\n'
        '    os.listdir(os.path.join(%r, "runs", "classify", "dog-bin")))}))\n'
        % (DETECT, ds, root))
    try:
        got = subprocess.run([py, '-c', probe], capture_output=True, text=True,
                             timeout=300, env=dict(os.environ,
                                                   TRAINING_ROOT=root))
        line = [x for x in (got.stdout or '').splitlines()
                if x.startswith('{')]
        if not line:
            bad.append('the run-directory probe said nothing: %r'
                       % ((got.stderr or '')[-300:],))
            return
        doc = json.loads(line[-1])
        if doc.get('chose') != 'myrun':
            bad.append('ULTRALYTICS PUT THE RUN IN %r, NOT THE DIRECTORY THE '
                       'DASHBOARD NAMED -- something created it first, so the '
                       'weights and the score land somewhere the page never '
                       'looks' % (doc.get('chose'),))
        if doc.get('dirs') != ['myrun']:
            bad.append('one run left %d directories behind: %r'
                       % (len(doc.get('dirs') or []), doc.get('dirs')))
    except subprocess.TimeoutExpired:
        bad.append('the run-directory probe never returned')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def git_stamp_checks(bad, py):
    """The bundle names the code that STARTED the run.

    Read at the end -- which is where the bundle is written -- this recorded
    whichever commit happened to be checked out hours later, so a run trained
    on Tuesday's code was filed under Thursday's. ultralytics' own YOLO is
    replaced here by one that makes a commit and then falls over, which is the
    whole event in one line. The attribute is patched rather than the module:
    the parameter schema is read out of ultralytics.cfg before a run starts,
    so a stand-in module kills the run before there is a bundle to grade.
    """
    tmp = tempfile.mkdtemp(prefix='adv_tm_git_')
    root = os.path.join(tmp, 'root')
    ds = os.path.join(root, 'dogbin_probe')
    for split in ('train', 'val'):
        for klass in ('dog', 'not_dog'):
            os.makedirs(os.path.join(ds, split, klass))
    os.makedirs(os.path.join(ds, 'bundle'))
    with open(os.path.join(ds, 'bundle', 'manifest.json'), 'w') as fh:
        json.dump({'family': 'dogbin', 'counts': {'total': 0},
                   'built_at_iso': 'probe'}, fh)
    probe = (
        'import json,sys; sys.path.insert(0, %r)\n'
        'import train_model as t\n'
        'import ultralytics\n'
        'holder = ["the-commit-the-run-started-from"]\n'
        't.bd.git_head = lambda: {"commit": holder[0], "dirty": False}\n'
        'class Boom:\n'
        '    def __init__(self, *a, **k):\n'
        '        holder[0] = "a-commit-made-while-it-trained"\n'
        '        raise RuntimeError("probe")\n'
        'ultralytics.YOLO = Boom\n'
        'try:\n'
        '    t.launch("dogbin", %r, overrides={"epochs": 1},\n'
        '             weights="/definitely/not/a/model.pt", by="guard")\n'
        'except BaseException as e:\n'
        '    print("THREW", type(e).__name__)\n'
        % (DETECT, ds))
    try:
        got = subprocess.run([py, '-c', probe], capture_output=True,
                             text=True, timeout=300,
                             env=dict(os.environ, TRAINING_ROOT=root))
        if 'THREW' not in (got.stdout or ''):
            bad.append('the git-stamp probe did not reach the bundle: %r %r'
                       % ((got.stdout or '')[-200:],
                          (got.stderr or '')[-300:]))
            return
        proj = os.path.join(root, 'runs', 'classify', 'dog-bin')
        made = sorted(os.listdir(proj)) if os.path.isdir(proj) else []
        if not made:
            bad.append('the git-stamp probe left no run directory')
            return
        with open(os.path.join(proj, made[0], 'bundle',
                               'manifest.json')) as fh:
            doc = json.load(fh)
        stamp = (doc.get('git') or {}).get('commit')
        if stamp != 'the-commit-the-run-started-from':
            bad.append('the run bundle names %r as the code that trained '
                       'these weights -- read when the run ended, so a commit '
                       'made meanwhile takes the credit' % (stamp,))
    except subprocess.TimeoutExpired:
        bad.append('the git-stamp probe never returned')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def bundle_checks(bad, py, _dataset):
    """The run's own record -- written even when the run falls over.

    A run that died after two hours is exactly the one somebody needs the
    parameters of, so the bundle is written from a finally block rather than
    after a success.

    IN A TEMPORARY TRAINING ROOT, and that is not tidiness. An earlier version
    ran against the real one and found the run it had just made by taking the
    newest directory by mtime -- so on any machine where the probe failed to
    create one, it would have deleted somebody's actual training run. A check
    that can destroy the thing it is checking is worse than no check.
    """
    tmp = tempfile.mkdtemp(prefix='adv_tm_bundle_')
    root = os.path.join(tmp, 'root')
    ds = os.path.join(root, 'dogbin_probe')
    for split in ('train', 'val'):
        for klass in ('dog', 'not_dog'):
            os.makedirs(os.path.join(ds, split, klass))
    os.makedirs(os.path.join(ds, 'bundle'))
    with open(os.path.join(ds, 'bundle', 'manifest.json'), 'w') as fh:
        json.dump({'family': 'dogbin', 'counts': {'total': 0},
                   'built_at_iso': 'probe'}, fh)
    # the rest of the dataset's record, which the run has to keep a copy of
    for name, doc in (('files.json', {'files': {'train': {'images': ['a']}}}),
                      ('inputs.json', {'stores': {'gate_audit': {}}}),
                      ('label_studio_export.json', [{'image': 'x'}])):
        with open(os.path.join(ds, 'bundle', name), 'w') as fh:
            json.dump(doc, fh)
    with open(os.path.join(ds, 'dataset.yaml'), 'w') as fh:
        fh.write('names:\n  0: dog\n')
    env = dict(os.environ, TRAINING_ROOT=root)
    probe = (
        'import json,sys; sys.path.insert(0, %r)\n'
        'import train_model as t\n'
        'try:\n'
        # weights that cannot load: YOLO() raises before a single epoch
        '    t.launch("dogbin", %r, overrides={"epochs": 1},\n'
        '             weights="/definitely/not/a/model.pt", by="guard")\n'
        'except BaseException as e:\n'
        '    print("THREW", type(e).__name__)\n'
        % (DETECT, ds))
    try:
        got = subprocess.run([py, '-c', probe], capture_output=True,
                             text=True, timeout=300, env=env)
        if 'THREW' not in (got.stdout or ''):
            bad.append('a run with impossible weights did not fail: %r %r'
                       % ((got.stdout or '')[-200:],
                          (got.stderr or '')[-200:]))
        proj = os.path.join(root, 'runs', 'classify', 'dog-bin')
        made = sorted(os.listdir(proj)) if os.path.isdir(proj) else []
        if not made:
            bad.append('no run directory was created, so nothing recorded '
                       'the attempt at all')
            return
        man = os.path.join(proj, made[0], 'bundle', 'manifest.json')
        if not os.path.isfile(man):
            bad.append('a run that failed left no bundle -- the parameters '
                       'of a run that died are the ones somebody needs')
            return
        with open(man) as fh:
            doc = json.load(fh)
        for key in ('params', 'params_from', 'dataset', 'command',
                    'versions', 'git', 'started_at_iso', 'weights', 'error'):
            if key not in doc:
                bad.append('the run bundle records no %r' % (key,))
        if not doc.get('error'):
            bad.append('a run that threw is recorded as having not')
        if (doc.get('dataset') or {}).get('id') != 'dogbin_probe':
            bad.append('the bundle does not say which dataset it trained on: '
                       '%r' % (doc.get('dataset'),))
        if not (doc.get('dataset') or {}).get('manifest_sha256'):
            bad.append('the bundle does not pin the dataset manifest, so '
                       'which build it used cannot be settled later')
        if not (doc.get('command') or {}).get('argv'):
            bad.append('the bundle does not record the command that ran')
        if doc.get('params', {}).get('epochs') != 1:
            bad.append('the bundle does not record the parameters actually '
                       'used: %r' % (doc.get('params'),))
        if not doc.get('versions', {}).get('ultralytics'):
            bad.append('the bundle does not say which ultralytics trained it')
        # THE DATASET IS DELETED ONE DAY. THE RUN IS THE RESULT.
        bundle = os.path.join(proj, made[0], 'bundle')
        kept = doc.get('dataset_record') or {}
        for name in ('dataset_manifest.json', 'dataset_files.json',
                     'dataset_inputs.json', 'dataset_label_studio_export.json',
                     'dataset.yaml'):
            if not os.path.isfile(os.path.join(bundle, name)):
                bad.append('the run keeps no copy of %s -- deleting the '
                           'dataset takes the answer to "which images" with '
                           'it, and the digest recorded here is then a hash '
                           'of a file nobody can produce' % (name,))
            elif name != 'dataset.yaml' and name not in kept:
                bad.append('%s was copied but not digested' % (name,))
        # ...and the checkpoint it started from is identified by content
        wr = doc.get('weights_record') or {}
        if 'name' not in wr:
            bad.append('the run does not record which checkpoint it started '
                       'from beyond a bare string')
        if wr.get('path') or wr.get('sha256'):
            bad.append('a checkpoint that does not exist was reported as '
                       'found: %r' % (wr,))
    except subprocess.TimeoutExpired:
        bad.append('the bundle probe never returned')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def promoted_fallback_checks(bad, tm):
    """A family with no run in its project directory inherits the PROMOTED
    recipe, not ultralytics' bare defaults.

    The promoted classifiers were trained by hand into top-level directories
    before the launcher existed, so a fresh family fell straight to 100
    epochs and no augmentation while the recipe that produced the shipping
    model sat readable in its run directory.
    """
    import tempfile
    tmp = tempfile.mkdtemp(prefix='adv_tm_promoted_')
    old_root = os.environ.get('TRAINING_ROOT')
    os.environ['TRAINING_ROOT'] = tmp
    try:
        # the promoted run, where best_models.json says dogbin's weights live
        run = os.path.join(tmp, 'dog-bin', 'dogbin_008')
        os.makedirs(os.path.join(run, 'weights'))
        open(os.path.join(run, 'weights', 'best.pt'), 'w').close()
        with open(os.path.join(run, 'results.csv'), 'w') as fh:
            fh.write('epoch,metrics/accuracy_top1\n1,0.9\n')
        with open(os.path.join(run, 'args.yaml'), 'w') as fh:
            fh.write('epochs: 500\npatience: 100\nerasing: 0.4\n'
                     'data: /gone/dogbin_v5\nname: dogbin_008\n')
        got, src = tm.last_args('dogbin')
        if not src or 'dogbin_008' not in src:
            bad.append('an empty project directory fell back to %r, not the '
                       'promoted run' % (src,))
        elif got.get('epochs') != '500':
            bad.append('the promoted recipe came back wrong: %r'
                       % (got.get('epochs'),))
        # ...and a run IN the project directory still wins over it
        newer = os.path.join(tmp, 'runs', 'classify', 'dog-bin', 'dogbin_x')
        os.makedirs(os.path.join(newer, 'weights'))
        open(os.path.join(newer, 'weights', 'best.pt'), 'w').close()
        with open(os.path.join(newer, 'results.csv'), 'w') as fh:
            fh.write('epoch,metrics/accuracy_top1\n1,0.5\n')
        with open(os.path.join(newer, 'args.yaml'), 'w') as fh:
            fh.write('epochs: 42\nname: dogbin_x\n')
        got, src = tm.last_args('dogbin')
        if not src or 'dogbin_x' not in src:
            bad.append('a real run in the project directory lost to the '
                       'promoted fallback: %r' % (src,))
    finally:
        if old_root is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = old_root
        shutil.rmtree(tmp, ignore_errors=True)


def comet_checks(bad, tm):
    """A run from the page reaches Comet, in the project everything tracks.

    The old runs logged because a person's shell had the key exported; the
    train page's job runner starts clean, comet_ml found no key, and the
    Comet callback declined SILENTLY -- a training that looked fine
    everywhere except the one place metrics are compared.
    """
    import tempfile
    tmp = tempfile.mkdtemp(prefix='adv_tm_comet_')
    fx = os.path.join(tmp, 'fixture.env')
    with open(fx, 'w') as fh:
        fh.write('# a comment\n'
                 'COMET_API_KEY="fixture-value"\n'
                 'COMET_WORKSPACE=fixture-ws\n'
                 'SECRET_ACCESS_KEY=must-not-load\n'
                 'BROKEN LINE\n')
    kept = {k: os.environ.pop(k, None)
            for k in ('COMET_API_KEY', 'COMET_WORKSPACE',
                      'SECRET_ACCESS_KEY')}
    try:
        got = tm.comet_env(fx)
        if sorted(got) != ['COMET_API_KEY', 'COMET_WORKSPACE']:
            bad.append('comet_env loaded %r -- only COMET_* variables '
                       'belong in the process, the same file holds S3 '
                       'credentials' % (got,))
        if os.environ.get('COMET_API_KEY') != 'fixture-value':
            bad.append('the key did not arrive, or kept its quotes: %r'
                       % (os.environ.get('COMET_API_KEY') is not None,))
        if os.environ.get('SECRET_ACCESS_KEY') == 'must-not-load':
            bad.append('comet_env leaked a NON-Comet secret into the '
                       'environment')
        os.environ['COMET_API_KEY'] = 'from-the-shell'
        again = tm.comet_env(fx)
        if os.environ['COMET_API_KEY'] != 'from-the-shell' \
                or 'COMET_API_KEY' in again:
            bad.append('comet_env overrides a key the shell exported')
        if tm.comet_env(os.path.join(tmp, 'nope.env')) != []:
            bad.append('a missing env file is not quietly empty')
    finally:
        for k, v in kept.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)

    # ...and launch actually calls it, before ultralytics loads. A helper
    # nobody calls is the bug wearing a fix's name.
    src = open(os.path.join(REPO, 'tools', 'detect',
                            'train_model.py')).read()
    body = src[src.index('def launch('):src.index('def main(')]
    if 'comet_env()' not in body:
        bad.append('launch() never loads the Comet key, so a run from the '
                   'page trains to completion with no experiment')
    elif body.index('comet_env()') > body.index('from ultralytics import'):
        bad.append('the key is loaded after ultralytics, whose Comet '
                   'callback reads the environment')

    # ONE SPELLING PER PROJECT. train_model alone said leash_models; the
    # promoted run, best_models.json and gate_store all say leash-models --
    # so a new leash run inherited from a three-class run and logged to a
    # Comet project nobody tracks.
    sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
    import gate_store
    with open(os.path.join(REPO, 'data', 'best_models.json')) as fh:
        tracked = set(json.load(fh)['projects'])
    for family, stage in (('dogbin', 'gate'), ('leash', 'leash')):
        ours = tm.PROJECTS[family]['project']
        theirs = gate_store.STAGES[stage]['project']
        if ours != theirs:
            bad.append('%s is %r to the trainer and %r to the sweep '
                       '-- two projects for one model' % (family, ours,
                                                          theirs))
        if ours not in tracked:
            bad.append('the trainer sends %s runs to %r, a project '
                       'best_models.json does not track' % (family, ours))


def main():
    bad = []
    try:
        import train_model as tm
    except Exception as e:                # noqa: BLE001
        print('FAIL could not import train_model: %s: %s'
              % (type(e).__name__, e))
        return 1
    for fn in (inherit_checks, dataset_checks, comet_checks,
               promoted_fallback_checks):
        try:
            fn(bad, tm)
        except Exception as e:            # noqa: BLE001
            bad.append('%s threw %s: %s' % (fn.__name__, type(e).__name__, e))
    py = training_python()
    if not py:
        bad.append('no interpreter with ultralytics is configured, so the '
                   'parameter checks did not run at all — set dogbin_python '
                   'in the dashboard config')
    else:
        # a real built dataset to point at, or one made here
        dataset, made = _a_dataset(tm)
        try:
            for fn in (form_checks, batch_checks):
                fn(bad, py)
            unfinished_checks(bad, tm)
            weights_record_checks(bad, tm)
            resume_checks(bad, tm, py)
            for fn in (refusal_checks, bundle_checks):
                fn(bad, py, dataset)
            save_dir_checks(bad, py)
            git_stamp_checks(bad, py)
        except Exception as e:            # noqa: BLE001
            bad.append('a parameter check threw %s: %s'
                       % (type(e).__name__, e))
        finally:
            if made:
                shutil.rmtree(made, ignore_errors=True)
    if bad:
        for b in bad:
            print('FAIL ' + b)
        return 1
    print('a run starts from what the last one used, every parameter is '
          'checked against ultralytics itself before anything is launched, '
          'and the run records what it trained on and what with — even when '
          'it falls over')
    return 0


def _a_dataset(tm):
    """A dogbin dataset to aim the checks at. Prefers a real built one, so
    the checks run against the shape the dashboard actually produces."""
    try:
        root = tm.bd.training_root()
        for name in sorted(os.listdir(root), reverse=True):
            path = os.path.join(root, name)
            man = os.path.join(path, 'bundle', 'manifest.json')
            if not os.path.isfile(man):
                continue
            with open(man) as fh:
                if json.load(fh).get('family') == 'dogbin':
                    return path, None
    except (OSError, ValueError, SystemExit):
        pass
    made = tempfile.mkdtemp(prefix='adv_tm_ds_')
    path = os.path.join(made, 'dogbin_probe')
    for split in ('train', 'val'):
        for klass in ('dog', 'not_dog'):
            os.makedirs(os.path.join(path, split, klass))
    os.makedirs(os.path.join(path, 'bundle'))
    with open(os.path.join(path, 'bundle', 'manifest.json'), 'w') as fh:
        json.dump({'family': 'dogbin', 'counts': {'total': 0}}, fh)
    return path, made


if __name__ == '__main__':
    sys.exit(main())
