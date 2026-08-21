#!/usr/bin/env python3
"""Train one model on one built dataset, and record exactly what was done.

    python tools/detect/train_model.py --family dogdet --dataset dogdet_2026...
    python tools/detect/train_model.py --family dogbin --dataset <id> \\
        --set epochs=200 --set batch=8
    python tools/detect/train_model.py --family dogdet --show-defaults

WHERE THE PARAMETERS COME FROM. Not a list written down here -- a list here is
correct until somebody trains a better run and silently wrong afterwards. The
starting point is the args.yaml of the most recent run in this family's
project, which is ultralytics' own record of what that run was launched with.
Whatever was learned by getting a good run is therefore what the next one
starts from, and `--show-defaults` prints it so nobody has to guess what they
are about to inherit.

EVERY KEY IS CHECKED AGAINST ULTRALYTICS ITSELF. The settable keys are
DEFAULT_CFG_DICT in the installed version, not a list copied off the
documentation page: the docs describe several versions at once, and this
version does not have `label_smoothing` at all. A key ultralytics does not
know is refused before the run starts rather than ignored inside it, because
an ignored parameter is a training run that quietly did something else.

THESE VALUES ARRIVE FROM A WEB FORM. So they are coerced to the type
ultralytics declares for them, checked against a range where one exists, and
anything that is not a parameter is refused by name.

THE RUN CARRIES ITS OWN BUNDLE, in <save_dir>/bundle/:

    manifest.json   the full resolved parameters, where each one came from,
                    the dataset trained on and the digest of ITS manifest, the
                    argv, the interpreter, the library versions, the git
                    commit, the start and end times, and how it ended

Which is the other half of a reproducible run: the dataset bundle says what it
was trained ON, and this says what it was trained WITH. Together they are
enough to run it again.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import build_dataset as bd                                    # noqa: E402

# task and project per family. The project is a DIRECTORY under
# <training root>/runs/<task>/, and it is also the Comet project name -- see
# the note in launch() about why those are set separately.
PROJECTS = {
    'dogdet': {'task': 'detect', 'project': 'dogdetection',
               'weights': 'yolo26x.pt'},
    'dogbin': {'task': 'classify', 'project': 'dog-bin',
               'weights': 'yolo11x-cls.pt'},
    # 'leash-models', hyphen: the spelling best_models.json tracks,
    # gate_store.STAGES runs, and the promoted run's own args.yaml carries.
    # This file alone said leash_models -- so a new leash run inherited from
    # a THREE-CLASS run in the underscore directory, logged to a Comet
    # project nobody tracks, and landed in a project the dashboard config
    # hides. One spelling, and it is the one every other record uses.
    'leash': {'task': 'classify', 'project': 'leash-models',
              'weights': 'yolo11x-cls.pt'},
}

# Keys that belong to a RUN rather than to how it trains: inheriting them from
# a previous run would point the new one at the old dataset, or write it into
# the old directory.
PER_RUN = frozenset({
    'data', 'model', 'project', 'name', 'save_dir', 'exist_ok', 'resume',
    'task', 'mode', 'source', 'weights', 'cfg', 'tracker',
})

# What the dashboard offers to change, and why anybody would. Everything else
# in DEFAULT_CFG_DICT is still settable from the command line -- this is the
# curated surface, not the permitted one.
EDITABLE = (
    ('epochs', 'how many passes over the training set'),
    ('patience', 'stop when this many epochs pass without a better one'),
    ('batch', 'images per step. The detector trains at 1280 and a 16GB card '
              'holds very few of those'),
    ('imgsz', 'the size images are resized to'),
    ('optimizer', 'SGD, Adam, AdamW, NAdam, RAdam, RMSProp, or auto'),
    ('lr0', 'starting learning rate'),
    ('lrf', 'final learning rate, as a fraction of lr0'),
    ('momentum', 'SGD momentum, or Adam beta1'),
    ('weight_decay', 'L2 penalty'),
    ('warmup_epochs', 'epochs spent ramping the learning rate up'),
    ('cos_lr', 'cosine schedule instead of linear'),
    ('close_mosaic', 'turn mosaic off for the last N epochs'),
    ('dropout', 'classify only'),
    ('freeze', 'freeze this many leading layers'),
    ('single_cls', 'treat every box as one class — the detector finds dogs '
                   'and nothing else'),
    ('rect', 'rectangular batches, less padding'),
    ('cache', 'hold images in RAM. Fast, and a large detector set will not fit'),
    ('workers', 'dataloader processes'),
    ('seed', 'the random seed, so a run can be repeated'),
    ('fraction', 'train on this share of the dataset'),
    ('hsv_h', 'hue jitter'),
    ('hsv_s', 'saturation jitter'),
    ('hsv_v', 'brightness jitter'),
    ('degrees', 'rotation'),
    ('translate', 'shift'),
    ('scale', 'zoom'),
    ('fliplr', 'left-right flip probability'),
    ('flipud', 'up-down flip probability'),
    ('mosaic', 'four images stitched into one'),
    ('mixup', 'two images blended'),
    ('erasing', 'random erasing, classify only'),
)


def _cfg_tables():
    """Ultralytics' own view of what a parameter is, in this version."""
    from ultralytics.cfg import (DEFAULT_CFG_DICT, CFG_FLOAT_KEYS,
                                 CFG_FRACTION_KEYS, CFG_INT_KEYS,
                                 CFG_BOOL_KEYS)
    return {'defaults': dict(DEFAULT_CFG_DICT),
            'float': set(CFG_FLOAT_KEYS), 'fraction': set(CFG_FRACTION_KEYS),
            'int': set(CFG_INT_KEYS), 'bool': set(CFG_BOOL_KEYS)}


def runs_root(family):
    spec = PROJECTS[family]
    return os.path.join(bd.training_root(), 'runs', spec['task'],
                        spec['project'])


def trained(run_dir):
    """Did this run get anywhere, or did it fall over at the starting line?

    ultralytics writes args.yaml before it does anything else, so a run that
    died in its first second still leaves a complete-looking record of what it
    was launched with. An epoch in results.csv, or a weights file, is the
    cheapest evidence that those parameters actually ran.
    """
    csv = os.path.join(run_dir, 'results.csv')
    try:
        if os.path.isfile(csv):
            with open(csv) as fh:
                fh.readline()             # the header is not an epoch
                if fh.readline().strip():
                    return True
    except OSError:
        pass
    weights = os.path.join(run_dir, 'weights')
    try:
        return any(n.endswith('.pt') for n in os.listdir(weights))
    except OSError:
        return False


def last_args(family):
    """(params, where) from the newest run in this family's project THAT RAN.

    ultralytics writes args.yaml into every run directory, so the record of
    what a run was launched with is the run itself. Parsed by hand rather
    than with pyyaml: the file is flat `key: value` and the dashboard's
    environment has no yaml.

    A run that crashed on its parameters is the last run somebody wants to
    inherit from, and it is exactly the newest one: a detector run died on a
    bad batch size and every later run was then offered that same batch size
    as the parameters to start from. Runs that produced nothing are skipped,
    and if none produced anything the page falls back to ultralytics' own
    defaults rather than to the wreckage.
    """
    root = runs_root(family)
    best, best_at = None, -1
    try:
        for name in os.listdir(root):
            run = os.path.join(root, name)
            path = os.path.join(run, 'args.yaml')
            if not os.path.isfile(path) or not trained(run):
                continue
            at = os.path.getmtime(path)
            if at > best_at:
                best, best_at = path, at
    except OSError:
        pass
    if best is None:
        return {}, None
    out = {}
    try:
        with open(best) as fh:
            for line in fh:
                if ': ' not in line or line.startswith(' '):
                    continue
                key, _, raw = line.partition(': ')
                out[key.strip()] = raw.strip()
    except OSError:
        return {}, None
    return out, best


def _coerce(key, value, tables):
    """One value, as the type ultralytics declares for that key."""
    if key in tables['bool']:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ('true', '1', 'yes', 'on'):
            return True
        if text in ('false', '0', 'no', 'off'):
            return False
        raise ValueError('%s is true or false, not %r' % (key, value))
    if key == 'batch':
        # BATCH IS A COUNT OR A FRACTION, and the difference is not cosmetic.
        # ultralytics lists it as a float so that 0 < batch < 1 can mean "use
        # that share of the card", and -1 means work it out. Anything else is
        # a number of images -- and torch's BatchSampler raises on a float:
        # inheriting `batch: 2` from a previous run produced 2.0 and every
        # detector run died at the first dataloader with
        # "batch_size should be a positive integer value, but got 2.0".
        text = str(value).strip().lower()
        if text == 'auto':
            return -1
        try:
            got = float(text)
        except ValueError:
            raise ValueError('batch is a number, or "auto", not %r' % (value,))
        return got if 0.0 < got < 1.0 else int(got)
    if key in tables['int']:
        try:
            return int(float(str(value).strip()))
        except ValueError:
            # A refusal is read by whoever typed it, so it says what was
            # wanted rather than repeating float()'s complaint about a str.
            raise ValueError('%s is a whole number, not %r' % (key, value))
    if key in tables['float'] or key in tables['fraction']:
        try:
            got = float(str(value).strip())
        except ValueError:
            raise ValueError('%s is a number, not %r' % (key, value))
        if key in tables['fraction'] and not 0.0 <= got <= 1.0:
            raise ValueError('%s is a fraction between 0 and 1, not %r'
                             % (key, value))
        return got
    text = str(value).strip()
    if text.lower() in ('none', 'null', ''):
        return None
    default = tables['defaults'].get(key)
    if isinstance(default, bool):
        return text.lower() in ('true', '1', 'yes', 'on')
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(float(text))
        except ValueError:
            return text
    if isinstance(default, float):
        try:
            return float(text)
        except ValueError:
            return text
    return text


def resolve(family, overrides=None, tables=None):
    """The parameters this run will use, and where each one came from.

    Three layers, each one able to overrule the last: ultralytics' defaults,
    the last run in this project, and whatever was asked for now. The
    provenance is kept because "why is this training at 1280" is a question
    with an answer, and the answer is usually "the run before it did".
    """
    tables = tables or _cfg_tables()
    inherited, source = last_args(family)
    params, whence = {}, {}
    for key, raw in sorted(inherited.items()):
        if key in PER_RUN or key not in tables['defaults']:
            continue
        try:
            got = _coerce(key, raw, tables)
        except ValueError:
            continue
        if got != tables['defaults'].get(key):
            params[key] = got
            whence[key] = 'the last run'
    bad = []
    for key, raw in sorted((overrides or {}).items()):
        if key in PER_RUN:
            bad.append('%s is decided by the run, not set on it' % (key,))
            continue
        if key not in tables['defaults']:
            bad.append('%s is not a training parameter in ultralytics %s'
                       % (key, _version()))
            continue
        try:
            params[key] = _coerce(key, raw, tables)
        except ValueError as e:
            bad.append(str(e))
            continue
        whence[key] = 'you'
    return params, whence, source, bad


def _version():
    try:
        import ultralytics
        return ultralytics.__version__
    except Exception:                     # noqa: BLE001
        return '?'


def find_dataset(family, want):
    """A dataset by id, by name, or by path -- and it must be one of this
    family's, because training a detector on crops fails deep inside
    ultralytics with a message about channels."""
    root = bd.training_root()
    path = want if os.path.isabs(want) else os.path.join(root, want)
    if not os.path.isdir(path):
        raise SystemExit('no such dataset: %s' % (want,))
    man = os.path.join(path, 'bundle', 'manifest.json')
    got = None
    if os.path.isfile(man):
        try:
            with open(man) as fh:
                got = json.load(fh)
        except (OSError, ValueError):
            got = None
    if got and got.get('family') and got['family'] != family:
        raise SystemExit('%s was built for %s, not %s'
                         % (os.path.basename(path), got['family'], family))
    # A directory carrying a generated name and no bundle is a build that was
    # killed: the bundle is the last thing a build writes. However many images
    # had been copied by then look exactly like a dataset from here, and
    # training on it trains on an arbitrary fraction of one.
    if got is None and bd.BUILT_RE.match(os.path.basename(path)):
        raise SystemExit('%s is an unfinished build -- it has no bundle, so '
                         'it is however much of a dataset was copied before '
                         'the build was stopped. Delete it and build again.'
                         % (os.path.basename(path),))
    kind = PROJECTS[family]['task']
    if kind == 'detect':
        data = os.path.join(path, 'dataset.yaml')
        if not os.path.isfile(data):
            raise SystemExit('%s has no dataset.yaml, so it is not a '
                             'detector dataset' % (path,))
    else:
        data = path
        if not os.path.isdir(os.path.join(path, 'train')):
            raise SystemExit('%s has no train/ directory, so it is not a '
                             'classification dataset' % (path,))
    return {'path': path, 'data': data, 'id': os.path.basename(path),
            'manifest': got,
            'manifest_sha256': bd.sha256(man) if os.path.isfile(man) else None}


def run_name(family, dataset_id, now=None):
    """A run is named for what it trained on, plus the time, so the run list
    reads as a history rather than as train, train2, train-2, train3."""
    stamp = time.strftime('%Y%m%d-%H%M', time.localtime(now or time.time()))
    return '%s_%s' % (dataset_id, stamp)


def resumable(family, name):
    """The checkpoint a stopped run can be continued from, or None.

    ultralytics writes weights/last.pt every epoch and can pick a run up from
    it, reading back the arguments it was launched with -- but only while the
    run directory it wrote them into is still there. A run that never finished
    an epoch has nothing to continue from and is a fresh start, not a resume.
    """
    if not family or not name or os.sep in str(name):
        return None
    run = os.path.join(runs_root(family), str(name))
    last = os.path.join(run, 'weights', 'last.pt')
    if not os.path.isfile(last) or \
            not os.path.isfile(os.path.join(run, 'args.yaml')):
        return None
    # A RUN THAT REACHED ITS LAST EPOCH HAS NOTHING TO CONTINUE. This is
    # ultralytics' own condition (start_epoch < epochs, trainer.py) worked out
    # from the two files that are readable without torch -- otherwise a resume
    # spends a minute scanning the dataset and then aborts on an assertion.
    want = None
    try:
        with open(os.path.join(run, 'args.yaml')) as fh:
            for line in fh:
                key, _, raw = line.partition(': ')
                if key.strip() == 'epochs' and not line.startswith(' '):
                    want = int(float(raw.strip()))
                    break
    except (OSError, ValueError):
        want = None
    done = 0
    try:
        with open(os.path.join(run, 'results.csv')) as fh:
            fh.readline()
            done = sum(1 for line in fh if line.strip())
    except OSError:
        done = 0
    if want and done >= want:
        return None
    return last


def recorded_data(run_dir):
    """The dataset a run was launched against, out of its own args.yaml."""
    path = os.path.join(run_dir, 'args.yaml')
    try:
        with open(path) as fh:
            for line in fh:
                key, _, raw = line.partition(': ')
                if key.strip() == 'data' and not line.startswith(' '):
                    got = raw.strip().strip("'\"")
                    return got or None
    except OSError:
        pass
    return None


def resume(family, name, by=''):
    """Continue a run that was stopped, in the directory it already has.

    Deliberately takes no parameters. ultralytics resumes from the arguments
    recorded in the run itself, and a resume that quietly trained on different
    settings than the epochs before it would produce one results.csv covering
    two different runs.
    """
    last = resumable(family, name)
    if last is None:
        raise SystemExit('%s cannot be resumed: no weights/last.pt beside an '
                         'args.yaml, so there is nothing to continue from'
                         % (name,))
    save_dir = os.path.dirname(os.path.dirname(last))
    # THE DATASET IS PINNED, AND HAS TO STILL BE THERE. ultralytics quietly
    # swaps in its own default when the dataset a checkpoint names has gone
    # (trainer.py check_resume), and an unset default resolves to coco8.yaml
    # for a detector, imagenet10 for a classifier -- so a resume after the
    # dataset was deleted would train on somebody else's toy dataset and
    # write the result over this run's weights.
    data = recorded_data(save_dir)
    if not data:
        raise SystemExit('%s does not record which dataset it trained on, so '
                         'it cannot be safely resumed' % (name,))
    if not os.path.exists(data):
        raise SystemExit('the dataset %s trained on is gone (%s) -- resuming '
                         'now would train on ultralytics\' default dataset '
                         'and write it over this run. Build it again and '
                         'start a new run.' % (name, data))
    started = int(time.time())
    head = bd.git_head()
    print('resuming %s' % (save_dir,), flush=True)
    print('  from      %s' % (last,), flush=True)
    print('  data      %s' % (data,), flush=True)
    os.environ.setdefault('COMET_PROJECT_NAME', PROJECTS[family]['project'])
    comet_env()
    from ultralytics import YOLO
    err = None
    metrics = None
    try:
        results = YOLO(last).train(resume=True, data=data)
        metrics = _metrics_of(results)
    except BaseException as e:            # noqa: BLE001 - recorded, re-raised
        err = '%s: %s' % (type(e).__name__, e)
        raise
    finally:
        doc = {'bundle_version': 1, 'family': family, 'name': str(name),
               'run_dir': save_dir, 'resumed_at': started,
               'resumed_at_iso': time.strftime('%Y-%m-%dT%H:%M:%S',
                                               time.localtime(started)),
               'resumed_from': weights_record(last), 'data': data,
               'by': str(by or ''), 'error': err, 'metrics': metrics,
               'command': {'argv': list(sys.argv), 'cwd': os.getcwd(),
                           'python': sys.executable},
               'versions': {'ultralytics': _version(),
                            'torch': _torch_version(),
                            'python': sys.version.split()[0]},
               'git': head}
        # BESIDE the original record, never over it: the first bundle says
        # what this run was launched with, and that is what the epochs before
        # the interruption actually trained on.
        _write_bundle_as(save_dir, 'resume.json', doc)
    print('\ndone: %s' % (save_dir,), flush=True)
    return {'ok': True, 'save_dir': save_dir, 'metrics': metrics}


def _write_bundle_as(save_dir, name, doc):
    try:
        bundle = os.path.join(save_dir, 'bundle')
        os.makedirs(bundle, exist_ok=True)
        tmp = os.path.join(bundle, name + '.tmp')
        with open(tmp, 'w') as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        os.replace(tmp, os.path.join(bundle, name))
        src = os.path.join(save_dir, 'results.csv')
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(bundle, 'results.csv'))
    except OSError:
        pass


def comet_env(env_file=None):
    """Load the COMET_* variables from the training repo's own .env.

    The names that were loaded come back; the values go only into
    os.environ, and existing variables are never overridden -- a shell that
    exported its own key keeps it.

    WITHOUT THIS, A RUN FROM THE TRAIN PAGE NEVER REACHES COMET. The old
    runs logged because a person's shell had the key exported; the page's
    job runner starts from a clean environment, comet_ml finds no key, and
    ultralytics' Comet callback silently declines -- no error, no
    experiment, a training that looks fine everywhere except the one place
    metrics are compared. The key lives beside the datasets it describes
    (comet_env_file in the dashboard config, or <training_root>/.env) and
    is read at launch time, never copied anywhere.
    """
    path = (env_file or os.environ.get('COMET_ENV_FILE')
            or bd._cfg('comet_env_file')
            or os.path.join(bd.training_root(), '.env'))
    loaded = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, _, v = line.partition('=')
                k = k.strip()
                if not k.startswith('COMET_') or k in os.environ:
                    continue
                os.environ[k] = v.strip().strip('"').strip("'")
                loaded.append(k)
    except OSError:
        pass
    return loaded


def launch(family, dataset, overrides=None, weights=None, name=None,
           by='', dry_run=False):
    """Resolve everything, refuse anything wrong, then train."""
    if family not in PROJECTS:
        raise SystemExit('no such model: %s' % (family,))
    tables = _cfg_tables()
    ds = find_dataset(family, dataset)
    params, whence, source, bad = resolve(family, overrides, tables)
    if bad:
        for line in bad:
            print('refused: ' + line, file=sys.stderr)
        raise SystemExit('nothing was started')
    spec = PROJECTS[family]
    root = bd.training_root()
    project = os.path.join(root, 'runs', spec['task'], spec['project'])
    name = name or run_name(family, ds['id'])
    save_dir = os.path.join(project, name)
    if os.path.exists(save_dir):
        raise SystemExit('%s already exists' % (save_dir,))
    model = weights or _inherited_weights(family) or spec['weights']
    started = int(time.time())
    # Taken now, not when the run ends: a training run is hours long, and the
    # manifest is written at the end, so this recorded whatever was checked
    # out by then rather than the code that trained the weights.
    head = bd.git_head()

    print('training %s on %s' % (family, ds['id']), flush=True)
    print('  weights   %s' % (model,), flush=True)
    print('  data      %s' % (ds['data'],), flush=True)
    print('  run       %s' % (save_dir,), flush=True)
    print('  inherited %s' % (source or 'nothing — ultralytics defaults'),
          flush=True)
    for key in sorted(params):
        print('  %-14s %-10r %s' % (key, params[key],
                                    whence.get(key, '')), flush=True)
    if dry_run:
        print('\n--dry-run: nothing was started', flush=True)
        return {'ok': True, 'dry_run': True, 'params': params,
                'save_dir': save_dir, 'dataset': ds['id']}

    # THE COMET PROJECT IS SET BY ENVIRONMENT, not by `project=`. ultralytics
    # passes project= straight through as the Comet workspace name, and an
    # absolute path there creates a junk project named after a directory.
    os.environ.setdefault('COMET_PROJECT_NAME', spec['project'])
    got = comet_env()
    if got:
        print('  comet     %s loaded, project %s'
              % (', '.join(got), spec['project']), flush=True)
    elif 'COMET_API_KEY' not in os.environ:
        # Loud, because silence was the failure: the callback declines
        # quietly and the run trains to completion with no experiment.
        print('  comet     NO KEY -- this run will not reach Comet '
              '(set comet_env_file, or COMET_API_KEY)', flush=True)
    # NOTHING MAY CREATE save_dir BEFORE ULTRALYTICS DOES. It is handed
    # exist_ok=False and increments the name when the directory is already
    # there, so a bundle copied in early sent the whole run to `<name>-2`
    # while the record sat in `<name>` -- and the page, which looks up the
    # name the dashboard chose, then found a directory with no weights, no
    # score and nothing to resume, for a run that was training normally.
    from ultralytics import YOLO
    kept = {}
    err = None
    metrics = None
    try:
        model_obj = YOLO(model)
        results = model_obj.train(data=ds['data'], project=project, name=name,
                                  exist_ok=False, **params)
        metrics = _metrics_of(results)
    except BaseException as e:            # noqa: BLE001 - recorded, re-raised
        err = '%s: %s' % (type(e).__name__, e)
        raise
    finally:
        # ...so the dataset's record is copied here instead, once the run
        # directory exists. A run killed outright loses it, which is the same
        # deal the manifest itself has always had.
        try:
            kept = _keep_dataset_record(os.path.join(save_dir, 'bundle'), ds)
        except Exception:                 # noqa: BLE001 - provenance, not flow
            kept = {}
        _write_bundle(save_dir, {
            'bundle_version': 1,
            'family': family, 'task': spec['task'], 'project': spec['project'],
            'name': name, 'run_dir': save_dir,
            'started_at': started,
            'started_at_iso': time.strftime('%Y-%m-%dT%H:%M:%S',
                                            time.localtime(started)),
            'ended_at': int(time.time()),
            'seconds': int(time.time()) - started,
            'by': str(by or ''),
            'error': err,
            'metrics': metrics,
            'weights': model,
            'weights_record': weights_record(model),
            'dataset_record': kept,
            'dataset': {'id': ds['id'], 'path': ds['path'],
                        'data': ds['data'],
                        'manifest_sha256': ds['manifest_sha256'],
                        'counts': (ds['manifest'] or {}).get('counts'),
                        'built_at_iso': (ds['manifest']
                                         or {}).get('built_at_iso')},
            'params': params,
            'params_from': whence,
            'inherited_from': source,
            'command': {'argv': list(sys.argv), 'cwd': os.getcwd(),
                        'python': sys.executable},
            'versions': {'ultralytics': _version(),
                         'torch': _torch_version(),
                         'python': sys.version.split()[0]},
            'git': head,
        })
    print('\ndone: %s' % (save_dir,), flush=True)
    return {'ok': True, 'save_dir': save_dir, 'params': params,
            'dataset': ds['id'], 'metrics': metrics}


def weights_record(model, root=None):
    """The checkpoint this run started from, named so it can be found again.

    `yolo26x.pt` is not a file: ultralytics resolves a bare name against the
    working directory and its own cache, and a fine-tune from a previous run
    records a path inside a run directory that later gets cleared. The one
    input to a training run with no integrity record now has one.
    """
    out = {'name': str(model), 'path': None, 'sha256': None, 'bytes': None}
    tries = [str(model)]
    if not os.path.isabs(str(model)):
        tries.append(os.path.join(root or bd.training_root(), str(model)))
        tries.append(os.path.join(os.getcwd(), str(model)))
    for path in tries:
        if os.path.isfile(path):
            out['path'] = os.path.abspath(path)
            try:
                out['bytes'] = os.path.getsize(path)
                out['sha256'] = bd.sha256(path)
            except OSError:
                pass
            break
    return out


def _inherited_weights(family):
    got, _ = last_args(family)
    return got.get('model') or None


def _torch_version():
    try:
        import torch
        return torch.__version__
    except Exception:                     # noqa: BLE001
        return None


def _metrics_of(results):
    try:
        got = getattr(results, 'results_dict', None)
        return {str(k): float(v) for k, v in (got or {}).items()
                if isinstance(v, (int, float))}
    except Exception:                     # noqa: BLE001
        return None


def _write_bundle(save_dir, doc):
    """The run's own record. Written even when the run threw, because a run
    that fell over after two hours is exactly the one somebody needs the
    parameters of."""
    try:
        bundle = os.path.join(save_dir, 'bundle')
        os.makedirs(bundle, exist_ok=True)
        tmp = os.path.join(bundle, 'manifest.json.tmp')
        with open(tmp, 'w') as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        os.replace(tmp, os.path.join(bundle, 'manifest.json'))
        # ultralytics' own record beside it, so the run is self-contained
        for who in ('args.yaml', 'results.csv'):
            src = os.path.join(save_dir, who)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(bundle, who))
    except OSError:
        pass


def _keep_dataset_record(bundle, ds):
    """Copy the dataset's own record into the run.

    The run recorded which dataset it trained on as a path and a digest of a
    file inside that dataset -- so deleting the dataset, which is gigabytes
    somebody will eventually clear, takes the answer to "which images" with
    it, and the digest is then a hash of a file nobody can produce. A dataset
    is deleted; a run is a result. The record travels with the result.
    """
    try:
        os.makedirs(bundle, exist_ok=True)
    except OSError:
        return {}
    src = os.path.join(ds['path'], 'bundle')
    want = {'manifest.json': 'dataset_manifest.json',
            'files.json': 'dataset_files.json',
            'inputs.json': 'dataset_inputs.json',
            'label_studio_export.json': 'dataset_label_studio_export.json'}
    kept = {}
    for name, as_name in want.items():
        path = os.path.join(src, name)
        if not os.path.isfile(path):
            continue
        try:
            shutil.copy2(path, os.path.join(bundle, as_name))
            kept[as_name] = bd.sha256(os.path.join(bundle, as_name))
        except OSError:
            pass
    # ...and the class map, which for a detector lives nowhere else: the
    # dataset.yaml names the classes the label indices mean.
    yaml = os.path.join(ds['path'], 'dataset.yaml')
    if os.path.isfile(yaml):
        try:
            shutil.copy2(yaml, os.path.join(bundle, 'dataset.yaml'))
            kept['dataset.yaml'] = bd.sha256(os.path.join(bundle,
                                                          'dataset.yaml'))
        except OSError:
            pass
    return kept


def show_defaults(family):
    """What the dashboard needs to draw the form, without importing
    ultralytics: it runs this and reads the JSON."""
    tables = _cfg_tables()
    params, whence, source, _bad = resolve(family, None, tables)
    out = {'family': family, 'ultralytics': _version(),
           'inherited_from': source,
           'weights': _inherited_weights(family) or PROJECTS[family]['weights'],
           'fields': []}
    for key, why in EDITABLE:
        if key not in tables['defaults']:
            continue                      # not a parameter in this version
        out['fields'].append({
            'key': key, 'why': why,
            'value': params.get(key, tables['defaults'][key]),
            'default': tables['defaults'][key],
            'from': whence.get(key, 'the ultralytics default'),
            'type': ('bool' if key in tables['bool'] else
                     'int' if key in tables['int'] else
                     'fraction' if key in tables['fraction'] else
                     'float' if key in tables['float'] else 'text'),
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--family', required=True, choices=sorted(PROJECTS))
    ap.add_argument('--dataset', help='a built dataset: its id, or a path')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='override one parameter; repeatable')
    ap.add_argument('--params-json', help='every override at once, as JSON')
    ap.add_argument('--weights', help='starting weights (default: whatever '
                                      'the last run in this project used)')
    ap.add_argument('--name', help='the run directory name')
    ap.add_argument('--resume', metavar='RUN',
                    help='continue a run that was stopped, from its own '
                         'weights/last.pt and its own recorded arguments')
    ap.add_argument('--by', default='')
    ap.add_argument('--show-defaults', action='store_true',
                    help='print the form the dashboard draws, as JSON')
    ap.add_argument('--dry-run', action='store_true',
                    help='resolve and check everything, start nothing')
    a = ap.parse_args(argv)
    if a.show_defaults:
        print(json.dumps(show_defaults(a.family), indent=1, sort_keys=True))
        return 0
    if not a.dataset and not a.resume:
        ap.error('--dataset is required unless --show-defaults or --resume')
    over = {}
    if a.params_json:
        try:
            got = json.loads(a.params_json)
        except ValueError as e:
            raise SystemExit('--params-json is not JSON: %s' % (e,))
        if not isinstance(got, dict):
            raise SystemExit('--params-json must be an object')
        over.update(got)
    for item in a.set:
        key, sep, value = item.partition('=')
        if not sep:
            raise SystemExit('--set wants KEY=VALUE, got %r' % (item,))
        over[key.strip()] = value
    if a.resume:
        if over or a.weights or a.dataset:
            raise SystemExit('a resume takes no dataset, no weights and no '
                             'parameters: it continues from what the run '
                             'itself recorded, and epochs trained on two '
                             'different settings would land in one '
                             'results.csv')
        resume(a.family, a.resume, by=a.by)
        return 0
    launch(a.family, a.dataset, overrides=over, weights=a.weights,
           name=a.name, by=a.by, dry_run=a.dry_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
