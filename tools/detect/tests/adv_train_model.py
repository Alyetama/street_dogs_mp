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
    except subprocess.TimeoutExpired:
        bad.append('the bundle probe never returned')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    bad = []
    try:
        import train_model as tm
    except Exception as e:                # noqa: BLE001
        print('FAIL could not import train_model: %s: %s'
              % (type(e).__name__, e))
        return 1
    for fn in (inherit_checks, dataset_checks):
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
            for fn in (refusal_checks, bundle_checks):
                fn(bad, py, dataset)
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
