#!/usr/bin/env python3
"""Build a training set for one model out of every annotation that exists.

    python tools/detect/build_dataset.py --family dogdet
    python tools/detect/build_dataset.py --family dogbin --dry-run
    python tools/detect/build_dataset.py --list

One command per model, because the answer to "rebuild the dataset" should not
be six commands in the right order with the right flags. This composes the
builders that already exist; it re-implements none of them, and every decision
about splitting, de-duplication and leakage stays where it was made.

    dogdet   build_dogdet_v3        v2 + every confirmed near-miss, audit
                                    verdict and hand-drawn box, re-split by
                                    sequence
             build_detector_negatives   the frames a reviewer called "not a
                                    dog" added as BACKGROUNDS -- an image with
                                    an empty label file, which is how YOLO
                                    learns a negative. Capped: past roughly a
                                    quarter of the train split they buy
                                    precision with recall, and recall is what
                                    this detector is selected on.
    dogbin   rebuild_crop_dataset   dogbin_v5 + the review queue's verdicts
                                    and the gate audit's finds
    leash    rebuild_crop_dataset   leash_v2 + the leash audit's finds and the
                                    review page's leash calls

EVERY BUILD IS A NEW DIRECTORY, named for the family, the day and six hex
characters. Nothing is ever built over: a dataset is the evidence for whatever
was trained on it, and a name that gets reused is a run whose evidence has
been overwritten. The id is in the directory name because that is what a
training run records.

EVERY BUILD CARRIES A BUNDLE, in <dataset>/bundle/:

    manifest.json   what this is, when, who asked, what it came from, the
                    exact command, the library versions, the git commit, and
                    the counts -- per class, per split, and overall
    files.json      EVERY image in the finished dataset: its name, its split,
                    its class, its sha256 and its size. For the detector, how
                    many boxes each frame carries, so a background is a fact
                    rather than an inference
    inputs.json     every file that was OFFERED to the build -- each
                    annotation store, its digest, and the name of every crop
    build_log.txt   everything the builders printed, in order

THE LIST IS THE POINT, not the counts. A count says a dataset had 3,247
images; a list says WHICH 3,247, in which split, under which class, with a
digest for each. Two builds are the same dataset when their file lists match,
and nothing weaker settles it -- so files.json's own digest is recorded in
manifest.json, and a listing edited afterwards stops matching.

The store digests answer the other half. "Rebuilt from the same annotations"
is a claim, and the only way to check it a month later is to compare the
ledger a build read against the ledger you have now: each sha256 is over the
bytes, so a verdict added, changed or withdrawn since shows up immediately.

A BUILD IS DERIVED, NEVER ACCUMULATED. Each one starts from the same
human-labelled base and re-applies every annotation on record, so building
twice from the same ledgers gives the same dataset, and a verdict somebody
undid is undone in the next build rather than baked in for ever.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DETECT = HERE


def training_root():
    """Where the datasets live. The training runs are a SEPARATE repo, so
    nothing here may assume a path."""
    got = os.environ.get('TRAINING_ROOT')
    if got:
        return got
    try:
        with open(os.path.join(REPO, 'tools', 'dashboard',
                               'dashboard.config.json')) as fh:
            cfg = json.load(fh)
        if cfg.get('training_root'):
            return cfg['training_root']
    except (OSError, ValueError):
        pass
    raise SystemExit(
        'no training root: set $TRAINING_ROOT, or training_root in\n'
        '  tools/dashboard/dashboard.config.json')


def _cfg(key):
    """One key out of the dashboard's config, or None. The environment wins,
    the same precedence every other tool here uses."""
    got = os.environ.get(key.upper())
    if got:
        return got
    try:
        with open(os.path.join(REPO, 'tools', 'dashboard',
                               'dashboard.config.json')) as fh:
            return json.load(fh).get(key) or None
    except (OSError, ValueError):
        return None


def D(*parts):
    return os.path.join(REPO, 'data', *parts)


# ── what each model is made of ──────────────────────────────────────────────
# `base` is the human-labelled dataset every build starts from. It is not the
# previous build: deriving from the last one compounds every decision it made
# and makes an undone verdict permanent, so each build goes back to the base
# and re-applies everything on record.
FAMILIES = {
    'dogdet': {
        'title': 'the detector',
        'kind': 'detect',
        'base': 'dogdet_v2',
        'classes': (),
        'what': 'boxes on full frames, one class',
    },
    'dogbin': {
        'title': 'the dog-bin gate',
        'kind': 'classify',
        'base': 'dogbin_v5',
        'classes': ('dog', 'not_dog'),
        'what': 'crops, dog against not a dog',
    },
    'leash': {
        'title': 'the leash model',
        'kind': 'classify',
        'base': 'leash_v2',
        'classes': ('leashed', 'unleashed'),
        'what': 'crops of dogs, leashed against unleashed',
    },
}

# Every store an annotation can be in, and which build reads it. Named here
# so the manifest can digest all of them without a second list going stale.
STORES = {
    'hard_negatives': {'ledger': D('hard_negatives', 'labels.jsonl'),
                       'crops': D('hard_negatives', 'crops'),
                       'for': ('dogdet', 'dogbin')},
    'hard_positives': {'ledger': D('hard_positives', 'labels.jsonl'),
                       'crops': D('hard_positives', 'crops'),
                       'for': ('dogdet', 'dogbin')},
    'box_corrections': {'ledger': D('box_corrections', 'boxes.jsonl'),
                        'crops': None, 'for': ('dogdet',)},
    'gate_audit': {'ledger': D('fn_audit', 'verdicts.jsonl'),
                   'crops': None, 'for': ('dogdet', 'dogbin')},
    'gate_audit_finds': {'ledger': D('audit_finds', 'manifest.jsonl'),
                         'crops': D('audit_finds'), 'for': ('dogbin',)},
    'leash_audit': {'ledger': D('leash_audit', 'verdicts.jsonl'),
                    'crops': None, 'for': ('leash',)},
    'leash_audit_finds': {'ledger': D('audit_finds_leash', 'manifest.jsonl'),
                          'crops': D('audit_finds_leash'), 'for': ('leash',)},
    'leash_calls': {'ledger': D('leash_labels', 'leash.db'),
                    'crops': D('leash_labels', 'crops'), 'for': ('leash',)},
}


# ── Label Studio ────────────────────────────────────────────────────────────
# The hand-drawn boxes: the only labels in this project a person sat down and
# drew, as opposed to a detector's guess that a person then agreed with. Every
# build fetches a fresh export and KEEPS IT, in the bundle -- so a build is
# reproducible from its own record even though the source is a live server
# that will have moved on by the time anybody looks.
#
# ONE EXPORT SERVES ALL THREE MODELS, because it is one project. Its boxes are
# labelled `leashed dog`, `unleashed dog`, `other animal` and `cow`:
#
#   the detector   every dog box, as one class; the other animals excluded and
#                  the tasks marked background kept as backgrounds
#   the leash      each dog box cropped, leashed against unleashed
#   the dog-bin    the same dog crops as `dog`, and every other-animal crop as
#                  `not_dog` -- which is where its negatives should come from:
#                  a person drew a box round a goat and said goat, rather than
#                  a reviewer catching the detector calling one a dog
LS_SCRIPT = 'export_annotations.sh'
LS_GROUP = 'label'                    # the field in a task that holds boxes
LS_DOG = ('leashed dog', 'unleashed dog')
LS_NOT_DOG = ('other animal', 'cow')
# Frames are fetched once and kept, because three builds of three models want
# the same pictures and the server is on the other side of the internet.
LS_FRAMES = os.path.join(REPO, 'data', 'labelstudio_frames')


def ls_export(stage, run, script_dir=None):
    """Fetch a fresh JSON_MIN export. Returns its path in the stage.

    The export script is the project's own -- it holds the token and the
    project number, and neither belongs in this repository. It prints the
    filename it wrote, which is what is read back.
    """
    root = script_dir or training_root()
    script = os.path.join(root, LS_SCRIPT)
    if not os.path.isfile(script):
        return None
    before = set(listing(root) or [])
    run.run('label studio export', ['/bin/bash', script], cwd=root)
    made = [f for f in (listing(root) or [])
            if f not in before and f.endswith('.json')]
    if not made:
        raise SystemExit('the export wrote no file — is the token still good?')
    src = os.path.join(root, sorted(made)[-1])
    dst = os.path.join(stage, 'label_studio_export.json')
    shutil.move(src, dst)
    return dst


def ls_read(path):
    """(tasks, counts) from a JSON_MIN export."""
    try:
        with open(path) as fh:
            tasks = json.load(fh)
    except (OSError, ValueError) as e:
        raise SystemExit('the export would not read: %s' % (e,))
    if not isinstance(tasks, list):
        raise SystemExit('the export is not a list of tasks')
    counts = {'tasks': len(tasks), 'boxes': 0, 'classes': {},
              'background': 0}
    for t in tasks:
        if t.get('background'):
            counts['background'] += 1
        for box in t.get(LS_GROUP) or []:
            for name in box.get('rectanglelabels') or []:
                counts['classes'][name] = counts['classes'].get(name, 0) + 1
                counts['boxes'] += 1
    return tasks, counts


def ls_search_dirs():
    """Where a frame might already be, before anything is fetched.

    dogdet_v2 and every detector set built from it hold export frames -- the
    same pictures, already on this machine. Searched rather than copied into a
    cache: 2,475 of them is 1.5GB, and a second copy of a file that is already
    there is a second copy to keep in step.
    """
    out = [LS_FRAMES]
    try:
        root = training_root()
    except SystemExit:
        return out
    for name in listing(root) or []:
        for split in ('train', 'val'):
            d = os.path.join(root, name, 'images', split)
            if os.path.isdir(d):
                out.append(d)
    return out


def _s3():
    """A client for the bucket the frames live in, from the project's .env.

    The credentials are the training repository's, read at call time and
    never copied anywhere -- they are not this repository's to hold.
    """
    import boto3
    from dotenv import load_dotenv
    load_dotenv(os.path.join(training_root(), '.env'))
    if not os.getenv('BUCKET_NAME'):
        return None, None
    return boto3.client(
        's3', endpoint_url=os.getenv('ENDPOINT_URL'),
        aws_access_key_id=os.getenv('ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('SECRET_ACCESS_KEY'),
        region_name=os.getenv('BUCKET_REGION')), os.getenv('BUCKET_NAME')


def ls_frame(task, index=None, s3=None, bucket=None):
    """The local jpg for one task, fetched from the bucket if it is not here.

    NOT over the https address in the export -- that answers 403. The picture
    lives in the project's S3 bucket and the address is how the key is
    spelled: the parts after the host are the key, which is what the
    project's own preparation script does.
    """
    url = str(task.get('image') or '')
    name = os.path.basename(url.split('?', 1)[0])
    if not name:
        return None
    if index is not None and name in index:
        return index[name]
    local = os.path.join(LS_FRAMES, name)
    if os.path.isfile(local) and os.path.getsize(local) > 0:
        return local
    if s3 is None or not bucket:
        return None
    os.makedirs(LS_FRAMES, exist_ok=True)
    key = '/'.join(url.split('://', 1)[-1].split('/')[1:])
    tmp = local + '.part'
    try:
        s3.download_file(bucket, key, tmp)
        os.replace(tmp, local)
    except Exception:                     # noqa: BLE001 - one frame, not the run
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None
    return local


def box_width_of(task):
    """The frame width Label Studio recorded, from any box on the task."""
    for box in task.get(LS_GROUP) or []:
        got = box.get('original_width')
        if got:
            return int(got)
    return None


def ls_index():
    """{filename: path} for every frame already on this machine."""
    index = {}
    for d in ls_search_dirs():
        for name in listing(d) or []:
            index.setdefault(name, os.path.join(d, name))
    return index


def ls_crops(tasks, want, out_dir, run=None, limit=None):
    """Cut every box of the wanted classes into <out_dir>/<class>/.

    `want` maps a Label Studio class to the class directory it becomes, so
    one export feeds the leash model (leashed/unleashed) and the dog-bin gate
    (dog/not_dog) without reading it twice.
    """
    from PIL import Image
    got = {v: 0 for v in want.values()}
    missing = small = failed = resized = 0
    index = ls_index()
    try:
        s3, bucket = _s3()
    except Exception as e:                # noqa: BLE001 - report and go on
        if run is not None:
            run.say('  no bucket (%s) — only frames already here can be cut'
                    % (type(e).__name__,))
        s3, bucket = None, None
    for i, task in enumerate(tasks):
        boxes = [b for b in (task.get(LS_GROUP) or [])
                 if any(r in want for r in (b.get('rectanglelabels') or []))]
        if not boxes:
            continue
        if limit is not None and sum(got.values()) >= limit:
            break
        frame = ls_frame(task, index, s3, bucket)
        if not frame:
            missing += 1
            continue
        try:
            img = Image.open(frame)
            img.load()
        except Exception:                 # noqa: BLE001
            missing += 1
            continue
        # ...AND IT HAS TO BE THE ORIGINAL. The frames already on this machine
        # live inside built detector sets, where they were resized to 1280 --
        # so a dog that was 79 pixels across in a 4000-pixel frame comes out
        # of one at 25, and a classifier trained on 17x16 crops is being
        # taught from thumbnails. When the local copy is not the size Label
        # Studio was shown, the original is fetched instead.
        want_w = box_width_of(task)
        if want_w and img.width != want_w:
            fresh = ls_frame(task, None, s3, bucket)
            if fresh and fresh != frame:
                try:
                    img = Image.open(fresh)
                    img.load()
                    frame = fresh
                except Exception:         # noqa: BLE001
                    pass
            if img.width != want_w:
                resized += 1
        stem = os.path.splitext(os.path.basename(frame))[0]
        for k, box in enumerate(boxes):
            name = next(r for r in box['rectanglelabels'] if r in want)
            klass = want[name]
            # AGAINST THE IMAGE IN HAND, not the one Label Studio was shown.
            # A box is stored as a percentage, and original_width is only a
            # note about the frame at annotation time -- a frame found inside
            # a built dataset has been resized to 1280, so scaling 4000-pixel
            # coordinates onto it puts every box off the right-hand edge.
            # Clamped, that is a zero-width crop: this cut nothing at all from
            # every task whose frame was already on the machine, and said so
            # only by producing no files.
            x1 = int(box['x'] / 100.0 * img.width)
            y1 = int(box['y'] / 100.0 * img.height)
            x2 = int((box['x'] + box['width']) / 100.0 * img.width)
            y2 = int((box['y'] + box['height']) / 100.0 * img.height)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                small += 1
                continue                  # a box too small to be a crop
            d = os.path.join(out_dir, klass)
            os.makedirs(d, exist_ok=True)
            try:
                img.crop((x1, y1, x2, y2)).save(
                    os.path.join(d, 'ls_%s_%d.jpg' % (stem, k)), quality=95)
                got[klass] += 1
            except Exception:             # noqa: BLE001
                failed += 1
                continue
        if run is not None and i % 250 == 0:
            run.say('  cut %d crops from %d/%d tasks'
                    % (sum(got.values()), i, len(tasks)))
    # SAID OUT LOUD. Every one of these three used to be a silent `continue`,
    # and a run that cut nothing looked exactly like a run with nothing to cut.
    if run is not None and (missing or small or failed):
        run.say('  skipped: %d without a frame, %d boxes too small, '
                '%d that would not save' % (missing, small, failed))
    if run is not None and resized:
        run.say('  %d frame(s) were only available resized — those crops are '
                'smaller than they should be' % (resized,))
    return got, missing


def sha256(path):
    """The bytes of one file. A ledger that has changed since a build has a
    different digest, which is the whole reason this is recorded."""
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def listing(path):
    """Sorted filenames one level down, or None if there is no such place."""
    try:
        return sorted(f for f in os.listdir(path)
                      if not f.startswith('.'))
    except OSError:
        return None


def store_state(name):
    """One annotation store, exactly as it is right now."""
    spec = STORES[name]
    out = {'ledger': spec['ledger'], 'sha256': sha256(spec['ledger']),
           'lines': None, 'crops': spec['crops'], 'files': None}
    # Lines only where lines mean something. leash.db is SQLite, and counting
    # newlines in it produces a number that looks like a verdict count and is
    # not one.
    if spec['ledger'].endswith('.jsonl'):
        try:
            with open(spec['ledger'], 'rb') as fh:
                out['lines'] = sum(1 for line in fh if line.strip())
        except OSError:
            pass
    if spec['crops']:
        names = listing(spec['crops'])
        if names is not None:
            # A class-per-directory store lists per class; a flat one lists
            # the crops themselves. Decided by whether ANY entry is a
            # directory, not the first: these directories also hold a README
            # and a manifest, and sorted() puts 'README.md' before 'dog'
            # because capitals sort first -- so the first entry was a file and
            # every per-class store read as flat.
            klasses = [c for c in names
                       if os.path.isdir(os.path.join(spec['crops'], c))]
            if klasses:
                out['files'] = {c: listing(os.path.join(spec['crops'], c))
                                for c in klasses}
            else:
                out['files'] = names
    return out


def _versions():
    """What the build ran against. A dataset rebuilt under a different duckdb
    is not obviously the same dataset, and this is where somebody looks."""
    out = {}
    for mod in ('duckdb', 'PIL', 'numpy'):
        try:
            got = __import__(mod)
            out[mod] = getattr(got, '__version__', None)
        except Exception:                 # noqa: BLE001 - provenance
            out[mod] = None
    return out


def git_head():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                             capture_output=True, text=True, timeout=20)
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO,
                               capture_output=True, text=True, timeout=30)
        return {'commit': (out.stdout or '').strip() or None,
                'dirty': bool((dirty.stdout or '').strip())}
    except Exception:                     # noqa: BLE001 - provenance, not flow
        return {'commit': None, 'dirty': None}


def new_name(family, now=None):
    """`<family>_<day>_<six hex>` -- sortable, readable, and never reused."""
    day = time.strftime('%Y%m%d', time.localtime(now or time.time()))
    return '%s_%s_%s' % (family, day, os.urandom(3).hex())


# ── running the builders ────────────────────────────────────────────────────

class Runner:
    """Runs each step, echoes it, and remembers what it did.

    Everything the builders print goes to this process's stdout as well as to
    build_log.txt, so the job runner's log is the whole story and the dataset
    carries its own copy.
    """

    def __init__(self, log_path, total):
        self.log = open(log_path, 'a', buffering=1)
        self.steps = []
        self.total = total

    def say(self, text):
        print(text, flush=True)
        self.log.write(text + '\n')

    def progress(self, n, what):
        # The job runner reads these. A convention, not an interface: a step
        # that never prints one is not broken, it just cannot be drawn.
        self.say('PROGRESS %d %d %s' % (n, self.total, what))

    def run(self, name, argv, cwd=None, env=None):
        self.say('\n$ ' + ' '.join(argv))
        t0 = time.time()
        run_env = dict(os.environ)
        run_env.update({str(k): str(v) for k, v in (env or {}).items()})
        proc = subprocess.Popen(argv, cwd=cwd or REPO, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, env=run_env)
        for line in proc.stdout:
            line = line.rstrip('\n')
            # a builder's own progress must not be mistaken for this build's
            print(line if not line.startswith('PROGRESS ')
                  else '| ' + line, flush=True)
            self.log.write(line + '\n')
        code = proc.wait()
        self.steps.append({'name': name, 'argv': argv, 'exit_code': code,
                           'seconds': round(time.time() - t0, 1)})
        if code != 0:
            raise SystemExit('%s failed (exit %d) — nothing was written to '
                             'the dataset directory' % (name, code))
        return code

    def close(self):
        try:
            self.log.close()
        except Exception:                 # noqa: BLE001
            pass


# ── the extras each crop model gets ─────────────────────────────────────────

def _copy_into(src_dir, dst_dir, seen):
    """Copy crops into a staged class directory, skipping repeats by name.

    A crop can be in two stores at once -- flagged in the review queue and
    found again in the audit -- and it is one example either way.
    """
    n = 0
    for name in listing(src_dir) or []:
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src) or name in seen:
            continue
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, name))
        seen.add(name)
        n += 1
    return n


def stage_extras(family, stage, run, duckdb_python, crop_python,
                 ls_tasks=None):
    """Every new crop for a classify family, staged one directory per class.

    rebuild_crop_dataset takes ONE directory per class, and the annotations
    arrive in several stores, so they are gathered here rather than by
    teaching that builder about each of them.
    """
    got = {c: 0 for c in FAMILIES[family]['classes']}
    seen = {c: set() for c in got}
    # THE HAND-DRAWN BOXES FIRST, because they are the best labels here: a
    # person drew them, rather than agreeing with a detector that had already
    # drawn one. For the gate that means the negatives are goats somebody
    # boxed as goats, not only the ones the detector mistook for dogs.
    if ls_tasks:
        want = ({'leashed dog': 'leashed', 'unleashed dog': 'unleashed'}
                if family == 'leash' else
                dict([(k, 'dog') for k in LS_DOG]
                     + [(k, 'not_dog') for k in LS_NOT_DOG]))
        cut, _missing = ls_crops(ls_tasks, want,
                                 os.path.join(stage, 'extras'), run=run)
        for klass, n in cut.items():
            got[klass] = got.get(klass, 0) + n
            for name in listing(os.path.join(stage, 'extras', klass)) or []:
                seen.setdefault(klass, set()).add(name)
        run.say('label studio crops: %s'
                % (', '.join('%s %d' % kv for kv in sorted(cut.items())),))
    if family == 'dogbin':
        # the review queue's own verdicts, cut from the original frames
        harvest = os.path.join(stage, 'harvest')
        for label, klass in (('false_positive', 'not_dog'),
                             ('true_positive', 'dog')):
            # --execute, or it prints what it WOULD cut and writes nothing:
            # the build then succeeds against an empty extras directory and
            # produces a dataset with none of the new annotations in it.
            # harvest_flagged finds duckdb through its own interpreter, so it
            # takes no --duckdb-python and must not be handed one.
            run.run('harvest %s' % (label,),
                    [crop_python, os.path.join(DETECT,
                                               'harvest_flagged.py'),
                     '--label', label, '--out', harvest, '--execute'],
                    # cv2 lives in the training env and duckdb does not, so
                    # the cutter runs there and reaches the store through the
                    # helper interpreter it already knows how to spawn.
                    env={'DETECT_DUCKDB_PYTHON': duckdb_python})
            got[klass] += _copy_into(os.path.join(harvest, klass),
                                     os.path.join(stage, 'extras', klass),
                                     seen[klass])
        for klass in got:
            got[klass] += _copy_into(D('audit_finds', klass),
                                     os.path.join(stage, 'extras', klass),
                                     seen[klass])
    elif family == 'leash':
        for klass in got:
            got[klass] += _copy_into(D('audit_finds_leash', klass),
                                     os.path.join(stage, 'extras', klass),
                                     seen[klass])
        # the review page's leash calls: the crop is already on disk, and the
        # verdict that names its class is in the database beside it
        try:
            sys.path.insert(0, DETECT)
            import leash_store
            con = leash_store.connect()
            try:
                rows = con.execute(
                    'SELECT crop, label FROM leash').fetchall()
            finally:
                con.close()
            for crop, label in rows:
                if label not in got:
                    continue
                src = os.path.join(leash_store.CROPS_OUT, str(crop))
                if not os.path.isfile(src) or crop in seen[label]:
                    continue
                dst = os.path.join(stage, 'extras', label)
                os.makedirs(dst, exist_ok=True)
                shutil.copy2(src, os.path.join(dst, str(crop)))
                seen[label].add(crop)
                got[label] += 1
        except Exception as e:            # noqa: BLE001 - a store, not the run
            run.say('leash calls unavailable (%s: %s) — building without them'
                    % (type(e).__name__, e))
    return got


# ── counting what came out ──────────────────────────────────────────────────

def inventory(out, family):
    """Every file in the finished dataset, with its digest and where it sits.

    THE POINT IS THE LIST, not the counts. A count says a dataset had 3,247
    images; a list says WHICH 3,247, in which split, under which class, and
    with a digest for each -- which is the difference between "we rebuilt it
    the same way" and "we rebuilt the same thing". Two builds are the same
    dataset when their file lists match, and nothing weaker settles it.

    Returns (files, counts). `files` is what goes in files.json; `counts` is
    the summary that goes on the front of the manifest.
    """
    kind = FAMILIES[family]['kind']
    files = {}
    counts = {'total': 0, 'splits': {}, 'classes': {}}
    for split in ('train', 'val'):
        files[split] = {}
        per = {'total': 0, 'classes': {}}
        if kind == 'detect':
            # ONE CLASS, and the interesting split is whether a frame carries
            # boxes or is a background -- an image with an empty label file,
            # which is how YOLO is taught what is not a dog.
            img_dir = os.path.join(out, 'images', split)
            lab_dir = os.path.join(out, 'labels', split)
            rows = []
            for name in listing(img_dir) or []:
                path = os.path.join(img_dir, name)
                stem = os.path.splitext(name)[0]
                lab = os.path.join(lab_dir, stem + '.txt')
                try:
                    boxes = sum(1 for line in open(lab) if line.strip())
                except OSError:
                    boxes = None
                rows.append({'name': name, 'sha256': sha256(path),
                             'bytes': _size(path), 'boxes': boxes,
                             'label_sha256': sha256(lab)})
            files[split] = {'images': rows}
            bg = sum(1 for r in rows if r['boxes'] == 0)
            per['classes'] = {'with_boxes': len(rows) - bg,
                              'background': bg}
            per['total'] = len(rows)
        else:
            for klass in listing(os.path.join(out, split)) or []:
                d = os.path.join(out, split, klass)
                if not os.path.isdir(d):
                    continue
                rows = [{'name': n, 'sha256': sha256(os.path.join(d, n)),
                         'bytes': _size(os.path.join(d, n))}
                        for n in listing(d) or []]
                files[split][klass] = rows
                per['classes'][klass] = len(rows)
                per['total'] += len(rows)
        counts['splits'][split] = per
        counts['total'] += per['total']
        for klass, n in per['classes'].items():
            counts['classes'][klass] = counts['classes'].get(klass, 0) + n
    # the share each split holds, because "is the val set big enough" is the
    # question anybody actually asks of these numbers
    for split, per in counts['splits'].items():
        per['share'] = (round(per['total'] / counts['total'], 4)
                        if counts['total'] else 0.0)
    return files, counts


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def measure(out, family):
    """The counts alone, for a caller that does not want the whole listing."""
    return inventory(out, family)[1]


def catalogue(family=None):
    """Every dataset on the training root, newest first.

    A SCAN, not a list. A dataset built five minutes ago has to show up
    without anybody editing a file, and one deleted by hand has to stop
    showing up -- so this walks the root and reads what it finds.

    A build made here carries bundle/manifest.json and gets its full record.
    The hand-built sets that came before -- dogbin_v5, dogdet_v2, leash_v2 --
    have no bundle, and they are still real datasets somebody may want to
    train on, so they are reported with what can be seen from their shape and
    marked as having no record rather than being hidden.
    """
    try:
        root = training_root()
    except SystemExit:
        return []
    out = []
    bases = {v['base']: k for k, v in FAMILIES.items()}
    for name in sorted(listing(root) or [], reverse=True):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or name.endswith('.stage'):
            continue
        man_path = os.path.join(path, 'bundle', 'manifest.json')
        man = None
        if os.path.isfile(man_path):
            try:
                with open(man_path) as fh:
                    man = json.load(fh)
            except (OSError, ValueError):
                man = None
        if man:
            row = {'id': name, 'path': path, 'family': man.get('family'),
                   'kind': man.get('kind'),
                   'built_at': man.get('built_at'),
                   'built_at_iso': man.get('built_at_iso'),
                   'built_by': man.get('built_by'),
                   'counts': man.get('counts'), 'bundle': True,
                   'base': os.path.basename(man.get('base') or ''),
                   'seconds': man.get('seconds')}
        else:
            kind, classes = _shape_of(path)
            if not kind:
                continue
            # THE CLASSES, NOT THE NAME. leash_binary_v2 is called leash and
            # holds dog/not_dog -- it was a dog-bin set renamed -- and a leash
            # model trained on it would learn the wrong question without one
            # error anywhere. A directory name is what somebody called it; the
            # class directories are what is in it.
            fam = None
            if kind == 'detect':
                fam = 'dogdet'
            elif classes:
                for key, spec in FAMILIES.items():
                    if spec['classes'] and set(spec['classes']) <= classes:
                        fam = key
                        break
            if fam is None:
                fam = bases.get(name)
            try:
                at = int(os.path.getmtime(path))
            except OSError:
                at = None
            row = {'id': name, 'path': path, 'family': fam, 'kind': kind,
                   'built_at': at,
                   'built_at_iso': (time.strftime('%Y-%m-%dT%H:%M:%S',
                                                  time.localtime(at))
                                    if at else None),
                   'built_by': '', 'counts': None, 'bundle': False,
                   'base': '', 'seconds': None}
        if family and row['family'] != family:
            continue
        out.append(row)
    out.sort(key=lambda r: r.get('built_at') or 0, reverse=True)
    return out


def _shape_of(path):
    """(kind, classes) from what is on disk. (None, set()) if it is not one."""
    if os.path.isfile(os.path.join(path, 'dataset.yaml')) and \
            os.path.isdir(os.path.join(path, 'images')):
        return 'detect', set()
    train = os.path.join(path, 'train')
    classes = {c for c in listing(train) or []
               if os.path.isdir(os.path.join(train, c))}
    return ('classify', classes) if classes else (None, set())


def build(family, out=None, by='', duckdb_python=None, crop_python=None,
          keep_stage=False, no_export=False, now=None):
    """Build one dataset. Returns the manifest it wrote."""
    if family not in FAMILIES:
        raise SystemExit('no such model: %s (try %s)'
                         % (family, ', '.join(sorted(FAMILIES))))
    root = training_root()
    spec = FAMILIES[family]
    base = os.path.join(root, spec['base'])
    if not os.path.isdir(base):
        raise SystemExit('the base dataset is missing: %s\n'
                         '  every build starts from it, so there is nothing '
                         'to derive from' % (base,))
    name = os.path.basename(out) if out else new_name(family, now)
    out = out or os.path.join(root, name)
    if os.path.exists(out):
        raise SystemExit('%s already exists — a dataset is the evidence for '
                         'what trained on it and is never built over' % (out,))
    duckdb_python = duckdb_python or sys.executable
    # The crops are cut with cv2, which lives in the training environment;
    # duckdb lives in the dashboard's. Neither has both, which is why the
    # builders take an interpreter for the half they cannot do themselves.
    crop_python = crop_python or _cfg('dogbin_python') or sys.executable
    stage = out + '.stage'
    os.makedirs(stage, exist_ok=True)
    started = int(time.time() if now is None else now)

    # the ledgers as they are RIGHT NOW, before anything reads them
    inputs = {k: store_state(k) for k, v in STORES.items()
              if family in v['for']}

    steps_total = 3 if family == 'dogdet' else 4
    run = Runner(os.path.join(stage, 'build_log.txt'), steps_total)
    extras = {}
    try:
        run.say('building %s (%s) from %s' % (name, spec['title'], base))
        run.progress(0, 'exporting the hand-drawn boxes')
        ls_path = ls_tasks = None
        ls_counts = None
        if not no_export:
            ls_path = ls_export(stage, run)
            if ls_path:
                ls_tasks, ls_counts = ls_read(ls_path)
                run.say('label studio: %d tasks, %d boxes (%s), %d background'
                        % (ls_counts['tasks'], ls_counts['boxes'],
                           ', '.join('%s %d' % kv for kv in
                                     sorted(ls_counts['classes'].items())),
                           ls_counts['background']))
            else:
                run.say('no export script at the training root — building '
                        'without the hand-drawn boxes')
        run.progress(0, 'reading the annotations')
        if family == 'dogdet':
            mid = os.path.join(stage, 'detect')
            run.progress(1, 'boxes, split by sequence')
            run.run('build_dogdet_v3',
                    [sys.executable, os.path.join(DETECT,
                                                  'build_dogdet_v3.py'),
                     '--out', mid])
            run.progress(2, 'backgrounds from the false positives')
            run.run('build_detector_negatives',
                    [sys.executable,
                     os.path.join(DETECT, 'build_detector_negatives.py'),
                     '--src', mid, '--out', out,
                     '--duckdb-python', duckdb_python, '--execute'])
            # The first step writes its record into the STAGE directory,
            # which is about to be deleted -- and that record holds the
            # holdout ids, the resolved sequences and which val frames were
            # moved for leaking. It is the answer to "why is this frame in
            # val", so it is carried across rather than thrown away.
            first = os.path.join(mid, 'manifest.json')
            if os.path.isfile(first):
                shutil.copy2(first, os.path.join(out, 'split_manifest.json'))
        else:
            run.progress(1, 'gathering new crops')
            extras = stage_extras(family, stage, run,
                                  duckdb_python, crop_python,
                                  ls_tasks=ls_tasks)
            run.say('staged extras: %s'
                    % (', '.join('%s %d' % (k, v)
                                 for k, v in sorted(extras.items())) or 'none'))
            argv = [sys.executable,
                    os.path.join(DETECT, 'rebuild_crop_dataset.py'),
                    '--src', base, '--out', out,
                    '--duckdb-python', duckdb_python,
                    '--pos-class', spec['classes'][0],
                    '--neg-class', spec['classes'][1]]
            pos = os.path.join(stage, 'extras', spec['classes'][0])
            neg = os.path.join(stage, 'extras', spec['classes'][1])
            if os.path.isdir(pos):
                argv += ['--extra-positives', pos]
            if os.path.isdir(neg):
                argv += ['--extra-negatives', neg]
            run.progress(2, 're-splitting by sequence')
            run.run('rebuild_crop_dataset', argv + ['--execute'])
        run.progress(steps_total - 1, 'listing what came out')
        files, counts = inventory(out, family)
        bundle = os.path.join(out, 'bundle')
        os.makedirs(bundle, exist_ok=True)

        # files.json first, so the manifest can carry its digest: a listing
        # that can be edited without the manifest noticing is a listing
        # nobody can rely on a month later.
        files_doc = {'id': name, 'family': family, 'kind': spec['kind'],
                     'built_at': started, 'files': files}
        files_path = os.path.join(bundle, 'files.json')
        with open(files_path, 'w') as fh:
            json.dump(files_doc, fh, indent=1, sort_keys=True)
        # THE EXPORT ITSELF, kept beside the record of it. The server it came
        # from is live and will have moved on; this file is what makes the
        # build reproducible rather than merely described.
        if ls_path and os.path.isfile(ls_path):
            shutil.copy2(ls_path,
                         os.path.join(bundle, 'label_studio_export.json'))
        inputs_path = os.path.join(bundle, 'inputs.json')
        with open(inputs_path, 'w') as fh:
            json.dump({'id': name, 'built_at': started, 'stores': inputs},
                      fh, indent=1, sort_keys=True)

        manifest = {
            'bundle_version': 1,
            'id': name, 'family': family, 'kind': spec['kind'],
            'title': spec['title'], 'what': spec['what'],
            'built_at': started,
            'built_at_iso': time.strftime('%Y-%m-%dT%H:%M:%S',
                                          time.localtime(started)),
            'timezone': time.strftime('%Z%z', time.localtime(started)),
            'built_by': str(by or ''),
            'seconds': int(time.time()) - started,
            'out': out, 'base': base,
            'counts': counts,
            'extras_staged': extras,
            'steps': run.steps,
            # WHAT WAS RUN, exactly. Reproducing a build means running the
            # same argv under the same interpreters against the same
            # annotations, and all three are recorded here.
            'command': {'argv': list(sys.argv), 'cwd': os.getcwd(),
                        'python': sys.executable,
                        'duckdb_python': duckdb_python,
                        'crop_python': crop_python},
            'label_studio': (None if not ls_path else {
                'file': 'bundle/label_studio_export.json',
                'sha256': sha256(ls_path),
                'counts': ls_counts}),
            'stores': {k: {'ledger': v['ledger'], 'sha256': v['sha256'],
                           'lines': v['lines'],
                           'files': (sum(len(x) for x in v['files'].values())
                                     if isinstance(v['files'], dict)
                                     else len(v['files'] or []))}
                       for k, v in inputs.items()},
            'files_json': {'path': 'bundle/files.json',
                           'sha256': sha256(files_path)},
            'inputs_json': {'path': 'bundle/inputs.json',
                            'sha256': sha256(inputs_path)},
            'git': git_head(),
            'python': sys.version.split()[0],
            'versions': _versions(),
            'builder_manifests': {},
        }
        # whatever the builders wrote about their own decisions, kept rather
        # than summarised: they record the holdout ids, the sequences and the
        # crops they dropped, and none of that is worth paraphrasing
        # Every record a builder leaves, under whichever name it uses. The
        # detector's two use three different ones between them, and a list
        # that named only the crop builder's quietly kept nothing at all for
        # the detector.
        for who in ('manifest.json', 'rebuild_manifest.json',
                    'negatives_manifest.json', 'split_manifest.json'):
            got = os.path.join(out, who)
            if os.path.isfile(got):
                try:
                    with open(got) as fh:
                        manifest['builder_manifests'][who] = json.load(fh)
                except (OSError, ValueError):
                    pass
        with open(os.path.join(bundle, 'manifest.json'), 'w') as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True)
        run.close()
        shutil.copy2(os.path.join(stage, 'build_log.txt'),
                     os.path.join(bundle, 'build_log.txt'))
        print('PROGRESS %d %d done' % (steps_total, steps_total), flush=True)
        print('\n%s: %d images total' % (name, counts['total']), flush=True)
        for split, per in sorted(counts['splits'].items()):
            print('  %-5s %6d  (%.0f%%)  %s'
                  % (split, per['total'], per['share'] * 100,
                     ', '.join('%s %d' % (k, v)
                               for k, v in sorted(per['classes'].items()))),
                  flush=True)
        print('  bundle: %s' % (os.path.join(out, 'bundle'),), flush=True)
        return manifest
    finally:
        run.close()
        if not keep_stage:
            shutil.rmtree(stage, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--family', choices=sorted(FAMILIES))
    ap.add_argument('--out', help='override the generated name (rare: the '
                                  'generated one is unique on purpose)')
    ap.add_argument('--by', default='', help='who asked for it')
    ap.add_argument('--duckdb-python',
                    default=os.environ.get('DUCKDB_PYTHON') or sys.executable,
                    help='interpreter with duckdb, for the sequence lookups')
    ap.add_argument('--crop-python',
                    default=os.environ.get('CROP_PYTHON'),
                    help='interpreter with cv2, for cutting crops out of the '
                         'original frames (default: the dogbin_python config '
                         'key, which is the training environment)')
    ap.add_argument('--keep-stage', action='store_true',
                    help='leave the working directory behind, to look at')
    ap.add_argument('--no-export', action='store_true',
                    help='skip the Label Studio export. For a rebuild that '
                         'must match an older one exactly, or for a machine '
                         'with no reach to the server -- and it is recorded, '
                         'so a dataset built without the hand-drawn boxes '
                         'says so')
    ap.add_argument('--list', action='store_true',
                    help='what can be built, and what it would read')
    ap.add_argument('--dry-run', action='store_true',
                    help='say what would be read and what it would be '
                         'called, and build nothing')
    a = ap.parse_args(argv)
    if a.list or not a.family:
        root = None
        try:
            root = training_root()
        except SystemExit as e:
            print(e)
        for fam in sorted(FAMILIES):
            spec = FAMILIES[fam]
            base = os.path.join(root, spec['base']) if root else spec['base']
            print('%-8s %-22s base %s%s' % (fam, spec['title'], spec['base'],
                                            '' if os.path.isdir(base)
                                            else '  (MISSING)'))
            for k, v in sorted(STORES.items()):
                if fam in v['for']:
                    st = store_state(k)
                    print('           %-18s %s lines  %s files' %
                          (k, st['lines'] if st['lines'] is not None else '-',
                           (sum(len(x) for x in st['files'].values())
                            if isinstance(st['files'], dict)
                            else len(st['files'] or []))))
        return 0
    if a.dry_run:
        print('would build %s in %s' % (new_name(a.family), training_root()))
        for k, v in sorted(STORES.items()):
            if a.family in v['for']:
                st = store_state(k)
                print('  %-18s sha %s  %s lines'
                      % (k, (st['sha256'] or '-')[:12], st['lines']))
        return 0
    build(a.family, out=a.out, by=a.by, duckdb_python=a.duckdb_python,
          crop_python=a.crop_python, keep_stage=a.keep_stage,
          no_export=a.no_export)
    return 0


if __name__ == '__main__':
    sys.exit(main())
