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
    dogbin   crop_sources           every human-labelled box -- the drawn
             + rebuild_crop_dataset boxes, the review queue's verdicts, every
                                    gate audit answer -- re-cut with
                                    production's own framing, de-duplicated
                                    across sources, split by sequence group,
                                    with the acceptance and test sets held out
    leash    crop_sources           the same for the leash model: drawn
             + rebuild_crop_dataset leashed/unleashed boxes, the leash audit
                                    and the review page's leash calls

THE CROP MODELS ARE NOT BUILT FROM A BASE ANY MORE. They used to be a copy of
dogbin_v5 / leash_v2 plus extra crops staged by filename, and the filenames
were the only identity a crop had -- which is how 2,933 crops of
dogbin_20261012 lost their sequence, how four framings ended up in one
dataset, and how the leash acceptance set would have gone back into training.
Each crop is now re-derived from a human ledger by crop_sources.py and carries
a manifest row (crops.jsonl, kept in the dataset and the bundle). The base is
read only to report how much of it the ledgers reproduce.

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

and a crop build adds crops.jsonl (the kept rows), collect_manifest.jsonl
(every row collected, before the split dropped any) and clusters.json (the
embedding near-duplicates) -- the stage is deleted, and without those two
nobody can say which rows a build dropped -- plus size_report.json, which
says how well crop size alone predicts the label.

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
import re
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
#
# For the crop models `base` is only what the reproduction report compares
# against (see the module docstring). `reserves` are the files whose image_ids
# AND sequences may never enter a split: FIRST the permanent acceptance set,
# THEN the frozen test set drawn from the audit verdicts (reserve_test_set.py).
# A missing one is fatal -- a build with no held-out set is how dogbin_v3 was
# accepted on crops it had trained on -- unless --no-reserve says so on the
# record. The acceptance set can never be replaced from the command line;
# --test-set replaces the second, --reserve adds more (family_reserves). `max_per_sequence` matches the dataset each family is compared with
# (dogbin_v5's 3, leash_v2's 5).
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
        'reserves': ('dogbin_acceptance_set.json', 'dogbin_test_v1.json'),
        'max_per_sequence': 3,
    },
    'leash': {
        'title': 'the leash model',
        'kind': 'classify',
        'base': 'leash_v2',
        'classes': ('leashed', 'unleashed'),
        'what': 'crops of dogs, leashed against unleashed',
        'reserves': ('leash_acceptance_set.json', 'leash_test_v1.json'),
        'max_per_sequence': 5,
    },
}

def _audit_name(stage, gen):
    """data/<audit dir> of generation `gen` of one audit stage --
    fn_audit.audit_name, imported late (fn_audit is stdlib-only, but this
    module is imported by the dashboard's training page)."""
    if DETECT not in sys.path:
        sys.path.insert(0, DETECT)
    import fn_audit
    return fn_audit.audit_name(fn_audit.spec(stage)['audit_dir'], gen)


def stores(gen=1):
    """Every store an annotation can be in, and which build reads it, for a
    build whose boxes are detector generation `gen`. Named here so the
    manifest can digest all of them without a second list going stale.

    The audit ledgers are per generation (fn_audit.paths(stage, gen=N)): a
    build reads the audit of its own detect store's generation and no other
    (crop_sources.audit_ctx_for), so the digests have to describe THOSE
    files -- a gen-2 build digesting data/fn_audit would record a ledger it
    never read. stores_for(detect_root) picks the generation.
    """
    ga, la = _audit_name('gate', gen), _audit_name('leash', gen)
    return {
        'hard_negatives': {'ledger': D('hard_negatives', 'labels.jsonl'),
                           'crops': None, 'for': ('dogdet', 'dogbin', 'leash')},
        'hard_positives': {'ledger': D('hard_positives', 'labels.jsonl'),
                           'crops': None, 'for': ('dogdet', 'dogbin')},
        # the review page's redraws AND the audit page's ("audit:<stage>"),
        # so every crop family reads it
        'box_corrections': {'ledger': D('box_corrections', 'boxes.jsonl'),
                            'crops': None,
                            'for': ('dogdet', 'dogbin', 'leash')},
        # the annotate page: dog boxes for the detector and the dog-bin, and
        # "no dog in this frame" as a veto for both crop families
        'frame_labels': {'ledger': D('frame_labels', 'labels.jsonl'),
                         'crops': None, 'for': ('dogdet', 'dogbin', 'leash')},
        # THE AUDIT CROPS ARE READ WHERE THE PAGE CUT THEM: full/ (the
        # detector's box, production framing) and edited/ (a person's
        # redraw). The audit_finds*/ exports are NOT inputs -- their
        # manifests had gone stale (62 lines over 35,810 files), and the leash
        # one had never been written. Each audit is read by BOTH crop
        # families: its own as labels, the other one's as vetoes on the same
        # boxes (crop_sources.other_family_vetoes)
        'gate_audit': {'ledger': D(ga, 'verdicts.jsonl'),
                       'crops': D(ga, 'full'), 'gen': gen,
                       'for': ('dogdet', 'dogbin', 'leash')},
        'gate_audit_edited': {'ledger': D(ga, 'pool.parquet'),
                              'crops': D(ga, 'edited'), 'gen': gen,
                              'for': ('dogbin',)},
        'leash_audit': {'ledger': D(la, 'verdicts.jsonl'),
                        'crops': D(la, 'full'), 'gen': gen,
                        'for': ('leash', 'dogbin')},
        'leash_audit_edited': {'ledger': D(la, 'pool.parquet'),
                               'crops': D(la, 'edited'), 'gen': gen,
                               'for': ('leash',)},
        'leash_calls': {'ledger': D('leash_labels', 'leash.db'),
                        'crops': None, 'for': ('leash', 'dogbin')},
        # what takes a label OUT, digested like what puts one in
        'label_flags': {'ledger': D('label_flags', 'label_flags.db'),
                        'crops': None, 'for': ('dogbin', 'leash')},
        'leash_label_conflicts': {'ledger': D('leash_label_conflicts.json'),
                                  'crops': None, 'for': ('leash',)},
        'dogbin_acceptance_set': {'ledger': D('dogbin_acceptance_set.json'),
                                  'crops': None, 'for': ('dogbin',)},
        'dogbin_test_set': {'ledger': D('dogbin_test_v1.json'),
                            'crops': None, 'for': ('dogbin',)},
        'leash_acceptance_set': {'ledger': D('leash_acceptance_set.json'),
                                 'crops': None, 'for': ('leash',)},
        'leash_test_set': {'ledger': D('leash_test_v1.json'),
                           'crops': None, 'for': ('leash',)},
    }


def stores_for(detect_root):
    """stores() of the generation `detect_root`'s boxes are
    (crop_sources.join_gen) -- the files a build on that store reads."""
    if DETECT not in sys.path:
        sys.path.insert(0, DETECT)
    import crop_sources
    return stores(crop_sources.join_gen(detect_root))


# The table for a build on the pinned ledger store -- the default build
# (detect_root defaults to data/ledger_detect_root.txt, gen 1), and what the
# training page lists. A build on another store reads stores_for(its root).
STORES = stores(1)


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


def ls_wanted(family):
    """Which drawn labels a model takes, and what it calls them.

    ONE EXPORT, THREE DIFFERENT READINGS. The leash model is asked leashed or
    unleashed about a dog, so a goat somebody boxed as `other animal` is not a
    harder example for it -- it is not an example at all. The gate is asked
    dog or not, so the goat is exactly what it needs. The detector is asked
    where the dogs are, so the goat's box is dropped and its frame stays as a
    picture with no dog in it.
    """
    if family == 'leash':
        return {'leashed dog': 'leashed', 'unleashed dog': 'unleashed'}
    if family == 'dogdet':
        return dict((k, 'dog') for k in LS_DOG)
    return dict([(k, 'dog') for k in LS_DOG]
                + [(k, 'not_dog') for k in LS_NOT_DOG])
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


def store_state(name, table=None):
    """One annotation store, exactly as it is right now. `table` is the
    stores() of the build's generation (default STORES, gen 1)."""
    spec = (table or STORES)[name]
    out = {'ledger': spec['ledger'], 'sha256': sha256(spec['ledger']),
           'lines': None, 'crops': spec['crops'], 'files': None}
    if 'gen' in spec:
        # which generation's audit dir this is: the same file names exist
        # under every generation, and the digest alone does not say which
        out['gen'] = spec['gen']
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


BUILT_RE = re.compile(r'^(%s)_\d{8}_[0-9a-f]{6}$'
                      % '|'.join(sorted(FAMILIES)))


def _dying_cleans_up(is_main=None):
    """Make a stop signal something Python can unwind from.

    Only in a process that owns its signals -- imported into a page, this must
    not touch the handlers of whatever is hosting it.
    """
    if not (__name__ == '__main__' if is_main is None else is_main):
        return
    import signal

    def stop(signum, _frame):
        raise SystemExit('stopped by signal %d' % (signum,))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, stop)
        except (OSError, ValueError):      # not ours to set
            pass


def new_name(family, now=None, variant=None):
    """`<family>_<day>_<six hex>` -- sortable, readable, and never reused.

    The six hex are random, so two builds -- a plain one and a rebalanced
    one of the same day, say -- never share a name. `variant` (the
    rebalance settings) is hashed in with the randomness so the id is
    derived from what was asked for as well; it cannot be read back off the
    name (BUILT_RE and train_model hold the name to three parts), which is
    why the rebuild manifest and the bundle say which variant it is.
    """
    day = time.strftime('%Y%m%d', time.localtime(now or time.time()))
    salt = json.dumps(variant, sort_keys=True).encode()
    return '%s_%s_%s' % (family, day, hashlib.sha256(
        os.urandom(16) + salt).hexdigest()[:6])


def rebalance_variant(max_per_creator=0, balance_size_bands=False):
    """{'variant': 'plain'|'rebalanced', ...settings} for a crop build, and
    the rebuild_crop_dataset flags that ask for it (the opt-in rebalance:
    rebuild_crop_dataset.rebalance_plan)."""
    flags = []
    if max_per_creator:
        flags += ['--max-per-creator', str(int(max_per_creator))]
    if balance_size_bands:
        flags += ['--balance-size-bands']
    return ({'variant': 'rebalanced' if flags else 'plain',
             'max_per_creator': int(max_per_creator) or None,
             'balance_size_bands': bool(balance_size_bands)}, flags)


# ── running the builders ────────────────────────────────────────────────────

# tqdm's "  10%|█  | 322/3078 [01:48<41:02,  1.12it/s]", however it is styled.
_PROGRESS_RE = re.compile(r'(\d+)/(\d+)\s*\[[\d:]+<')


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
        self._at = 0                      # which step we are on
        self._what = ''                   # and what it is doing
        self._tick = 0.0                  # when a progress line was last kept

    def say(self, text):
        print(text, flush=True)
        self.log.write(text + '\n')

    def progress(self, n, what):
        # The job runner reads these. A convention, not an interface: a step
        # that never prints one is not broken, it just cannot be drawn.
        self._at, self._what = n, what
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
            # A PROGRESS BAR IS ONE LINE REPEATED. tqdm redraws with a
            # carriage return, but nothing here is a terminal, so every redraw
            # arrives as its own line: fetching three thousand frames wrote
            # three thousand near-identical lines into the log and buried
            # everything the build actually said. One is kept every couple of
            # seconds, and the count in it is republished as this step's own
            # progress -- so a step that takes twenty-five minutes says how
            # far through it is instead of sitting at "1 of 4".
            sub = _PROGRESS_RE.search(line)
            if sub:
                now = time.time()
                if now - self._tick < 2.0:
                    continue
                self._tick = now
                self.say('PROGRESS %d %d %s (%s/%s)'
                         % (self._at, self.total, self._what,
                            sub.group(1), sub.group(2)))
                continue
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
        # A MANIFEST WE CANNOT READ IS NOT THE SAME AS NO MANIFEST. One is a
        # build that never got that far; the other is a finished dataset with
        # a damaged record, and calling that one 'unfinished, safe to delete'
        # invites somebody to throw away gigabytes that were fine.
        damaged = False
        if os.path.isfile(man_path):
            try:
                with open(man_path) as fh:
                    man = json.load(fh)
            except (OSError, ValueError):
                man, damaged = None, True
        if man:
            row = {'id': name, 'path': path, 'family': man.get('family'),
                   'kind': man.get('kind'),
                   'built_at': man.get('built_at'),
                   'built_at_iso': man.get('built_at_iso'),
                   'built_by': man.get('built_by'),
                   'counts': man.get('counts'), 'bundle': True,
                   'base': os.path.basename(man.get('base') or ''),
                   # Whether the hand-drawn boxes are in it. Every build makes
                   # its own export now, but the sets built before that do not
                   # say so anywhere -- and picking one trains a model on
                   # everything EXCEPT the boxes somebody drew by hand.
                   'label_studio': ((man.get('label_studio') or {})
                                    .get('counts') or {}).get('tasks'),
                   # ...and what those frames were labelled, because each
                   # model takes a different part of them: the leash model
                   # never sees a goat
                   'label_studio_classes': ((man.get('label_studio') or {})
                                            .get('counts') or {}).get(
                                                'classes') or {},
                   'seconds': man.get('seconds'), 'unfinished': False,
                   'damaged': False}
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
                   'base': '', 'label_studio': None,
                   'label_studio_classes': {}, 'seconds': None,
                   # This one carries a name only this tool generates, and no
                   # bundle -- the last thing a build writes. So it is a build
                   # that was killed outright: a directory of however many
                   # images had been copied when it died. Named rather than
                   # hidden, because it is also gigabytes somebody has to
                   # delete, but never offered as something to train on.
                   'damaged': damaged,
                   'unfinished': bool(BUILT_RE.match(name)) and not damaged}
        if family and row['family'] != family:
            continue
        out.append(row)
    out.sort(key=lambda r: r.get('built_at') or 0, reverse=True)
    return out


def remove(dataset_id, in_use=()):
    """Delete one dataset, and refuse every way that goes wrong.

    Gigabytes, one button, no undo -- so the id has to be one this catalogue
    already lists (not a path, not a name somebody typed), it must not be a
    family's base, which every future build derives from, and it must not be
    the dataset a job is reading right now.
    """
    name = str(dataset_id or '')
    if not name or os.sep in name or name.startswith('.') or name == '..':
        return {'ok': False, 'message': 'that is not a dataset built here'}
    bases = {spec['base'] for spec in FAMILIES.values()}
    if name in bases:
        return {'ok': False,
                'message': '%s is a base dataset -- every build starts from '
                           'it, so deleting it breaks all of them' % (name,)}
    if name in set(in_use or ()):
        return {'ok': False,
                'message': '%s is being used by something running right now'
                           % (name,)}
    row = None
    for got in catalogue():
        if got['id'] == name:
            row = got
            break
    if row is None:
        return {'ok': False, 'message': 'no such dataset'}
    root = os.path.realpath(training_root())
    # A LINK IS NOT THE THING IT POINTS AT. Following one deleted whatever was
    # on the other end -- and a link inside the training root pointing at a
    # base dataset passed every other check here, because the name checked is
    # the link's and the resolved path is still inside the root.
    if os.path.islink(row['path']):
        return {'ok': False,
                'message': '%s is a link to %s -- delete it where it actually '
                           'lives, or remove the link by hand'
                           % (name, os.path.realpath(row['path']))}
    path = os.path.realpath(row['path'])
    # belt and braces on top of the name check: whatever the catalogue said,
    # what is about to be removed has to live under the training root
    if not path.startswith(root + os.sep):
        return {'ok': False,
                'message': '%s points outside the training root (%s) -- '
                           'delete it where it actually lives'
                           % (name, path)}
    # A BUILD IN PROGRESS LOOKS EXACTLY LIKE A DEAD ONE from the outside:
    # both are a generated name with no bundle yet, and the page offers the
    # dead ones for deletion. The staging directory beside it is the
    # difference -- it exists only while a build is running, and is the last
    # thing that build removes.
    if os.path.isdir(path + '.stage'):
        return {'ok': False,
                'message': '%s is being built right now -- stop the build '
                           'first if you want it gone' % (name,)}
    freed = 0
    for here, _dirs, files in os.walk(path):
        for one in files:
            try:
                freed += os.path.getsize(os.path.join(here, one))
            except OSError:
                pass
    try:
        shutil.rmtree(path)
    except OSError as e:
        return {'ok': False, 'message': 'could not remove it: %s' % (e,)}
    return {'ok': True, 'message': '', 'freed': freed, 'path': path}


def _shape_of(path):
    """(kind, classes) from what is on disk. (None, set()) if it is not one."""
    if os.path.isfile(os.path.join(path, 'dataset.yaml')) and \
            os.path.isdir(os.path.join(path, 'images')):
        return 'detect', set()
    train = os.path.join(path, 'train')
    classes = {c for c in listing(train) or []
               if os.path.isdir(os.path.join(train, c))}
    return ('classify', classes) if classes else (None, set())


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


# ── the size shortcut ───────────────────────────────────────────────────────

# Short-side bands, (lo, hi] in px, the ones the leash audit measured in:
# rebuild_crop_dataset.SIZE_BANDS, the ONE definition, which
# --balance-size-bands balances in -- a diagnostic that cut the axis
# differently would not be measuring what the balance changed.
sys.path.insert(0, DETECT)
import rebuild_crop_dataset as _rcd               # noqa: E402
SIZE_BANDS = _rcd.SIZE_BANDS
# Size-only AUC past which a split is WARNED. leash_test_v1 (production
# framing, 628 crops) has size-only AUC 0.51 (0.53 with every geometry
# feature); leash_20261115_18b269 had 0.60 train / 0.65 val. 0.58 is clear
# of the test set's noise and well under what the build actually had.
SIZE_AUC_WARN = 0.58
# What each family's frozen production-framed test set itself carries,
# measured on its crops (pixel sizes of data/<audit>/full/<key>.jpg, the
# crops it is scored on). The gate's is NOT flat: dogbin_test_v1's dog rate
# runs 0.31 at <=64 px to 0.75 above 192 px, AUC 0.63 -- above
# dogbin_20261114_06c6b6's own 0.60 train / 0.60 val -- so for the gate a
# size-predicts-the-label build is what production looks like, and "the
# test set does not have it" would be false. A build against any other
# test set (--test-set) gets no baseline and a neutral message.
TEST_SIZE_AUC = {
    'leash': {'test_set': 'leash_test_v1.json', 'n': 628,
              'size_auc': 0.513, 'short_side_auc': 0.508},
    'dogbin': {'test_set': 'dogbin_test_v1.json', 'n': 1839,
               'size_auc': 0.629, 'short_side_auc': 0.678},
}


def rank_auc(scores, labels):
    """Mann-Whitney AUC of `scores` for labels==1, midranks for ties.

    None when either class is missing -- an AUC over one class is not 0.5,
    it is undefined, and a 0.5 would read as "no shortcut".
    """
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    n_pos = sum(1 for _s, y in pairs if y)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        mid = (i + j) / 2.0 + 1.0          # 1-based midrank of the tie block
        rank_sum += mid * sum(1 for k in range(i, j + 1) if pairs[k][1])
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _size_band(short):
    # this module's SIZE_BANDS, so a test can patch the bands here alone
    return _rcd.size_band(short, SIZE_BANDS)


def _size_slice(rows, pos):
    """One slice's band table and size-only AUCs, from (w, h, label) rows."""
    doc = {'n': len(rows), 'positives': sum(1 for r in rows if r[2] == pos),
           'bands': {}}
    doc['pos_rate'] = (round(doc['positives'] / doc['n'], 4)
                       if doc['n'] else None)
    for _lo, _hi, name in SIZE_BANDS:
        doc['bands'][name] = {'n': 0, 'classes': {}, 'pos_rate': None}
    for w, h, label in rows:
        b = doc['bands'][_size_band(min(w, h))]
        b['n'] += 1
        b['classes'][label] = b['classes'].get(label, 0) + 1
    for b in doc['bands'].values():
        if b['n']:
            b['pos_rate'] = round(b['classes'].get(pos, 0) / b['n'], 4)
    ys = [1 if r[2] == pos else 0 for r in rows]
    area = rank_auc([r[0] * r[1] for r in rows], ys)
    short = rank_auc([min(r[0], r[1]) for r in rows], ys)
    doc['size_auc'] = None if area is None else round(area, 4)
    doc['short_side_auc'] = None if short is None else round(short, 4)
    return doc


def size_shortcut(rows, pos, warn_at=SIZE_AUC_WARN, baseline=None):
    """How well crop size ALONE predicts the label, per split and source.

    leash_20261115_18b269 let it: leashed rate by short side ran from 0.17
    at <=64 px (n=242) to about 0.60 above 192 px, and crop area alone
    ranked leashed above unleashed at AUC 0.60 train / 0.65 val. In
    leash_test_v1, cut the way production cuts, every band sits at
    0.50-0.61 and the same AUC is 0.51 -- so a model that learned "small
    means unleashed" scores well on val and pushes production's small dogs
    (28% of boxes are under 64 px) toward unleashed, and val cannot show it.
    The causes are framing (Label Studio leashed boxes take in the leash)
    and the aimed audit draws (27% leashed overall, and 8% leashed among
    those under 64 px), so it is a diagnostic here, not a rebalance.

    `rows` are crops.jsonl rows (crop_wh, split, label, source); `pos` is
    the family's positive class; `baseline` is the family's TEST_SIZE_AUC
    entry, or None when the test set's own AUC is not known. A split past
    `warn_at` -- either way round: 0.40 is as much a shortcut as 0.60 -- is
    WARNED, and the message says what the test set does. When the test set
    is itself past the line (the gate, 0.63), size predicting the label is
    what production looks like: only a split further than the same margin
    (warn_at - 0.5) from the test set's AUC is WARNED, and one within it is
    a NOTE that says to score per size band. Sources are reported, not
    warned: a source is not what the model is scored on.
    """
    by_split, by_source = {}, {}
    for r in rows:
        wh = r.get('crop_wh')
        if not wh or len(wh) != 2 or not min(wh):
            continue
        one = (float(wh[0]), float(wh[1]), r.get('label'))
        by_split.setdefault(str(r.get('split')), []).append(one)
        by_source.setdefault(str(r.get('source')), []).append(one)
    rep = {'positive': pos, 'measure': 'rank AUC of crop area (w*h) for '
           'the positive class; short_side_auc is the same on min(w, h)',
           'bands_px': [name for _lo, _hi, name in SIZE_BANDS],
           'warn_at': warn_at,
           'splits': {k: _size_slice(v, pos)
                      for k, v in sorted(by_split.items())},
           'sources': {k: _size_slice(v, pos)
                       for k, v in sorted(by_source.items())},
           'baseline': baseline, 'warnings': [], 'notes': []}
    margin = warn_at - 0.5
    ref = (baseline or {}).get('size_auc')
    test = (os.path.splitext(baseline['test_set'])[0] if baseline
            else 'the test set')
    for split, doc in rep['splits'].items():
        auc = doc['size_auc']
        if auc is None or abs(auc - 0.5) <= margin:
            continue
        head = ('%s size-only AUC %.3f (crop area alone ranks %s %s the '
                'other class): crop size predicts the label in this set'
                % (split, auc, pos, 'above' if auc > 0.5 else 'below'))
        if ref is None:
            rep['warnings'].append(
                'WARNING: %s. Whether the production-framed test set shares '
                'it is not measured for this family: measure it, and score '
                'the model per size band on the test set before trusting '
                'val.' % head)
        elif abs(ref - 0.5) <= margin:
            rep['warnings'].append(
                'WARNING: %s -- a shortcut the production-framed test set '
                'does not have (%s: %.3f). Score the model per size band on '
                'the test set before trusting val.' % (head, test, ref))
        elif abs(auc - ref) > margin:
            rep['warnings'].append(
                'WARNING: %s, and %s does too (%.3f), but this set is %.3f '
                'away from it: the build\'s size mix is not production\'s. '
                'Score the model per size band on the test set.'
                % (head, test, ref, abs(auc - ref)))
        else:
            rep['notes'].append(
                'NOTE: %s, as the production-framed test set does (%s: '
                '%.3f), so it is not a build-versus-test mismatch. Score '
                'the model per size band on the test set.'
                % (head, test, ref))
    return rep


def print_size_shortcut(rep, say=print):
    cols = rep['bands_px']
    say('\nsize shortcut: %s rate per short-side band, and size-only AUC'
        % (rep['positive'],))
    say('  %-22s %5s  %s  %6s %6s'
        % ('', 'n', '  '.join('%9s' % c for c in cols), 'auc', 'short'))
    for kind in ('splits', 'sources'):
        for k, doc in rep[kind].items():
            cells = []
            for c in cols:
                b = doc['bands'][c]
                cells.append('%9s' % ('-' if b['pos_rate'] is None else
                                      '%.2f/%d' % (b['pos_rate'], b['n'])))
            say('  %-22s %5d  %s  %6s %6s'
                % ('%s %s' % (kind[:-1], k), doc['n'], '  '.join(cells),
                   '-' if doc['size_auc'] is None else '%.3f'
                   % doc['size_auc'],
                   '-' if doc['short_side_auc'] is None else '%.3f'
                   % doc['short_side_auc']))
    for w in rep['warnings'] + rep.get('notes', []):
        say(w)


def _read_jsonl(path):
    rows = []
    try:
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    except (OSError, ValueError):
        pass
    return rows


def _bundle_stage_records(stage, bundle, rm):
    """Keep what the stage alone holds: the pre-split manifest and clusters.

    leash_20261115_18b269 collected 5,410 rows and kept 3,744. The other
    1,666 (1,005 reserved, 508 over the sequence cap, 31 with no sequence,
    122 de-duplicated) were listed only in the stage's crops.jsonl and the
    embedding clusters only in its clusters.json -- and the stage is deleted
    when the build ends, so the bundle held counts, a manifest_sha256 of a
    file that no longer existed, and nothing that said WHICH rows went.
    Reconciling them took a two-minute re-collect and a twenty-minute
    re-embed, and against ledgers that have moved on it cannot be done at
    all. Copied byte for byte, so the recorded sha256 still checks: about
    5.8 MB for leash (5,410 rows), and about 49 MB for the gate --
    dogbin_20261114_06c6b6 collected 45,371 rows at ~1,070 bytes each,
    which takes its 54 MB bundle to ~103 MB, +7% on its 690 MB dataset.
    Left uncompressed so it opens and hashes like the file the rebuilder
    read. The collector's conflicts are already inlined in conflicts.json.

    Returns {name: {path (bundle-relative to the dataset), built_as (the
    stage path the build recorded), sha256, ...}} for what was copied.
    """
    got = {}
    for name, dest, recorded in (
            ('crops.jsonl', 'collect_manifest.jsonl', 'manifest'),
            ('clusters.json', 'clusters.json', 'dup_clusters_file')):
        src = os.path.join(stage, name)
        if not os.path.isfile(src):
            continue
        to = os.path.join(bundle, dest)
        shutil.copy2(src, to)
        doc = {'path': os.path.join(os.path.basename(bundle), dest),
               'built_as': rm.get(recorded) or src,
               'sha256': sha256(to), 'bytes': os.path.getsize(to)}
        if dest == 'collect_manifest.jsonl':
            with open(to) as fh:
                doc['rows'] = sum(1 for line in fh if line.strip())
            if rm.get('manifest_sha256'):
                doc['matches_rebuild_manifest'] = (
                    doc['sha256'] == rm['manifest_sha256'])
        got[dest.split('.')[0]] = doc
    return got


# Where a path inside a crop record resolves once the stage is gone. Every
# crop path written during the build points into <dataset>.stage/crops/,
# which the build deletes; the rows are copied as they were (the sha256s
# must still check), and this is how to read them.
STAGE_PATH_RULES = {
    'kept_crop': 'crops.jsonl rows: the crop is <dataset>/<split>/<label>/'
                 '<uid>; "path" is where it was cut, in the deleted stage',
    'dropped_crop': 'collect_manifest.jsonl rows not in crops.jsonl were '
                    'dropped (reserved, capped, no sequence, de-duplicated): '
                    'their crop is not kept -- re-cut it from frame.path and '
                    'box with gate_store.cut_crops',
    'clusters': 'clusters.json groups name stage crop paths; match them by '
                'basename, which is the uid',
}


def _point_at_bundle(rebuild_manifest, records):
    """Add the bundle copies beside the stage paths a rebuild manifest names.

    Both are kept: the stage path is what the build actually read, and the
    bundle path is the one that still opens.
    """
    if not isinstance(rebuild_manifest, dict):
        return rebuild_manifest
    for key, name in (('manifest', 'collect_manifest'),
                      ('dup_clusters_file', 'clusters')):
        if name in (records or {}):
            rebuild_manifest[key + '_in_bundle'] = records[name]['path']
    return rebuild_manifest


def _bundle_crop_record(out, stage, bundle, base, reserve_paths,
                        classes=None, say=print, family=None):
    """Copy a crop build's own records into the bundle; return their summary.

    crops.jsonl (one row per crop: what it is, where it came from), the
    pre-split manifest and the embedding clusters from the stage, the
    conflicts (both the collector's and the splitter's), the size report
    with its size-shortcut diagnostic, the reserves with their digests, and
    how much of the old base the ledgers reproduce -- informational, the
    base is no longer a source.
    """
    for name in ('crops.jsonl', 'reserves.json', 'reserved_crops.json'):
        src = os.path.join(out, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(bundle, name))
    for name in ('collect_counts.json', 'size_ref.json'):
        src = os.path.join(stage, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(bundle, name))
    rm = _read_json(os.path.join(out, 'rebuild_manifest.json'))
    records = _bundle_stage_records(stage, bundle, rm)
    kept = _read_jsonl(os.path.join(out, 'crops.jsonl'))
    resolved = sum(1 for r in kept if os.path.isfile(os.path.join(
        out, str(r.get('split')), str(r.get('label')), str(r.get('uid')))))
    records['crops'] = {'path': os.path.join(os.path.basename(bundle),
                                             'crops.jsonl'),
                        'rows': len(kept), 'rows_resolved': resolved}
    records['rules'] = STAGE_PATH_RULES
    split_conf = _read_json(os.path.join(out, 'conflicts.json'))
    conflicts = {'collect': _read_json(os.path.join(
                     stage, 'collect_conflicts.json')) or [],
                 'split': split_conf.get('conflicts') or [],
                 'reported_not_dropped': split_conf.get('reported') or []}
    with open(os.path.join(bundle, 'conflicts.json'), 'w') as fh:
        json.dump(conflicts, fh, indent=1, sort_keys=True)
    sizes = rm.get('sizes') or {}
    sizes['note'] = (
        'No gen-2-only box carries a human label, so the box distribution is '
        'gen-1\'s even though the framing is production\'s. Label Studio '
        '"other animal" boxes are much smaller than its dog boxes: a "small '
        'means not a dog" shortcut is available to the gate and should be '
        'checked per size band at evaluation.')
    pos = (classes or [None])[0]
    family = family or (kept[0].get('family') if kept else None)
    if pos is None:
        pos = ((FAMILIES.get(family) or {}).get('classes') or (None,))[0]
    # the test set's own size AUC applies only when that test set is the
    # one this build held out
    baseline = TEST_SIZE_AUC.get(family)
    if baseline and baseline['test_set'] not in {
            os.path.basename(p) for p in reserve_paths or ()}:
        baseline = None
    if pos and kept:
        shortcut = size_shortcut(kept, pos, baseline=baseline)
        sizes['shortcut'] = shortcut
        sizes['warnings'] = list(sizes.get('warnings') or []) + \
            shortcut['warnings']
        print_size_shortcut(shortcut, say=say)
    with open(os.path.join(bundle, 'size_report.json'), 'w') as fh:
        json.dump(sizes, fh, indent=1, sort_keys=True)
    # how much of the old base the ledgers re-derive, by image_id
    repro = {'base': base, 'present': os.path.isdir(base)}
    if repro['present']:
        sys.path.insert(0, DETECT)
        from crop_sources import parse_name
        have = set()
        try:
            with open(os.path.join(stage, 'crops.jsonl')) as fh:
                have = {json.loads(line)['image_id'] for line in fh
                        if line.strip()}
        except (OSError, ValueError):
            pass
        n = hit = 0
        unread = 0
        for split in ('train', 'val', 'test'):
            for klass in listing(os.path.join(base, split)) or []:
                for f in listing(os.path.join(base, split, klass)) or []:
                    n += 1
                    got = parse_name(f)
                    if got is None:
                        unread += 1
                    elif got[0] in have:
                        hit += 1
        repro.update({'crops': n, 'image_id_reproduced': hit,
                      'unparsed': unread,
                      'share': round(hit / n, 4) if n else None})
    with open(os.path.join(bundle, 'base_reproduction.json'), 'w') as fh:
        json.dump(repro, fh, indent=1, sort_keys=True)
    collect = _read_json(os.path.join(stage, 'collect_counts.json'))
    return {
        'framing': 'gate_store.cut_crops on full-resolution PIL-decoded '
                   'frames; audit crops reused from the audit page (same '
                   'constants), checked by re-cutting a sample',
        'roots': collect.get('roots'),
        'collect': {k: collect.get(k) for k in ('sources', 'dropped', 'rows',
                                                 'by_source_label', 'noseq',
                                                 'exclusion_digests',
                                                 'cross_family_verdicts',
                                                 'empty_frames',
                                                 'allowed_empty_sources')},
        'reserves': [{'path': p, 'sha256': sha256(p)} for p in reserve_paths],
        'conflicts': {k: len(v) for k, v in conflicts.items()},
        'size_warnings': sizes.get('warnings'),
        'size_notes': (sizes.get('shortcut') or {}).get('notes'),
        'size_auc': {k: v['size_auc'] for k, v in
                     ((sizes.get('shortcut') or {}).get('splits') or
                      {}).items()},
        # the stage's records, now in the bundle, and how to resolve every
        # path that still names the stage
        'records': records,
        'base_reproduction': repro,
        'rebuild': {k: rm.get(k) for k in ('dedup', 'over_sequence_cap',
                                           'kept', 'counts', 'groups',
                                           'groups_pinned_noseq',
                                           'noseq_crops', 'by_source',
                                           'max_per_sequence',
                                           'max_noseq_frac', 'keep_noseq',
                                           'noseq_rows', 'noseq_dropped',
                                           'reserved_rows_removed',
                                           'reserved_bytes_mates_removed',
                                           'reserved_dhash_mates_removed',
                                           'reserve_hamming', 'variant',
                                           'rebalance')},
    }


def family_reserves(family, extra=None, no_reserve=False, test_set=None):
    """The reserve files a crop build must exclude -- all of them, or a refusal.

    ALWAYS the family's acceptance set (the first of FAMILIES[family]
    ['reserves']); then its test set -- `test_set` (--test-set) in place of
    the default one, for a scratch build against a test set drawn somewhere
    else; then every `extra` file (--reserve), ADDED to those. --reserve used
    to REPLACE the list, so passing just a new test set dropped the
    acceptance set from the exclusion and put all 312 leash acceptance images
    back into train/val with every assertion passing -- defect D2 again, one
    flag away. Only --no-reserve builds with nothing held out.
    """
    if no_reserve:
        return []
    names = list(FAMILIES[family].get('reserves') or ())
    if not names:
        raise SystemExit('%s has no reserves configured' % (family,))
    paths = [D(names[0])]
    paths += [test_set] if test_set else [D(n) for n in names[1:]]
    for p in extra or ():
        if os.path.abspath(p) not in {os.path.abspath(q) for q in paths}:
            paths.append(p)
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            'reserve file(s) missing: %s\n  a crop build excludes its '
            'acceptance and test sets by image_id and sequence; draw the test '
            'set first (tools/detect/reserve_test_set.py --family %s --gen '
            '<the audit generation>), or '
            'pass --no-reserve to build with none, on the record'
            % (', '.join(missing), family))
    return paths


def build(family, out=None, by='', duckdb_python=None, crop_python=None,
          keep_stage=False, no_export=False, no_embed_dedup=False, now=None,
          ls_export_path=None, detect_root=None, s3_python=None, workers=4,
          allow_missing_frames=0, no_reserve=False, max_noseq_frac=0.02,
          reserves=None, size_ref_root=None, dedup_device='auto',
          test_set=None, allow_no_ls=False, keep_noseq=False,
          max_per_creator=0, balance_size_bands=False):
    """Build one dataset. Returns the manifest it wrote."""
    if family not in FAMILIES:
        raise SystemExit('no such model: %s (try %s)'
                         % (family, ', '.join(sorted(FAMILIES))))
    root = training_root()
    spec = FAMILIES[family]
    variant, rebalance_flags = rebalance_variant(max_per_creator,
                                                 balance_size_bands)
    if rebalance_flags and spec['kind'] != 'classify':
        raise SystemExit('--max-per-creator / --balance-size-bands are for '
                         'the crop models, not %s' % (family,))
    base = os.path.join(root, spec['base'])
    # The detector still derives from its base. A crop model does not -- it is
    # re-cut from the ledgers -- so its base is only read, if it is there, for
    # the reproduction report.
    if spec['kind'] == 'detect' and not os.path.isdir(base):
        raise SystemExit('the base dataset is missing: %s\n'
                         '  every build starts from it, so there is nothing '
                         'to derive from' % (base,))
    name = os.path.basename(out) if out else new_name(family, now, variant)
    out = out or os.path.join(root, name)
    if os.path.exists(out):
        raise SystemExit('%s already exists — a dataset is the evidence for '
                         'what trained on it and is never built over' % (out,))
    reserve_paths = []
    if spec['kind'] == 'classify':
        reserve_paths = family_reserves(family, reserves, no_reserve,
                                        test_set)
        if no_export and not ls_export_path and not allow_no_ls:
            raise SystemExit(
                '--no-export with no --ls-export: the hand-drawn boxes would '
                'not be in this build. Pass --ls-export PATH (a bundle\'s '
                'label_studio_export.json will do), or --allow-no-ls to '
                'build without them, on the record')
        if ls_export_path and not os.path.isfile(ls_export_path):
            raise SystemExit('no such export: %s' % (ls_export_path,))
    # EVERY family joins the human ledgers to a store -- the crop models by
    # conf, the detector by n_det and det 0 -- and those were recorded against
    # the gen-1 store. So both default to the pinned ledger root, never the
    # live one, and the manifest says which store it was.
    if detect_root is None:
        sys.path.insert(0, DETECT)
        import crop_sources
        detect_root = crop_sources.default_detect_root()
    duckdb_python = duckdb_python or sys.executable
    # The crops are cut with cv2, which lives in the training environment;
    # duckdb lives in the dashboard's. Neither has both, which is why the
    # builders take an interpreter for the half they cannot do themselves.
    crop_python = crop_python or _cfg('dogbin_python') or sys.executable
    stage = out + '.stage'
    os.makedirs(stage, exist_ok=True)
    # A stop from the page is a signal to the whole process group, and the
    # default action for one is to die where you stand -- no finally, no
    # cleanup. What is left behind is a multi-gigabyte staging copy and, once
    # a builder has started writing, a half-copied dataset directory with no
    # bundle in it. Turning the signal into an exception is what lets the
    # cleanup below happen at all.
    _dying_cleans_up()
    started = int(time.time() if now is None else now)
    # ...and the code as it is right now. Taken at the end, this recorded
    # whichever commit happened to be checked out when the build finished:
    # a detector build spends twenty-five minutes fetching frames, and a
    # commit made in that window renamed the code the manifest claims built
    # this dataset.
    head = git_head()

    # the ledgers as they are RIGHT NOW, before anything reads them -- the
    # ones of THIS build's detector generation (stores_for)
    table = stores_for(detect_root)
    inputs = {k: store_state(k, table) for k, v in table.items()
              if family in v['for']}

    # 5, not 4: the classify path gained the embedding-dedup step, and a
    # bar that reaches 4 of 4 with a whole cluster pass still to run tells
    # the page the build hung at "done".
    steps_total = 5
    run = Runner(os.path.join(stage, 'build_log.txt'), steps_total)
    extras = {}
    try:
        run.say('building %s (%s) from %s' % (name, spec['title'], base))
        run.progress(0, 'exporting the hand-drawn boxes')
        ls_path = ls_tasks = None
        ls_counts = None
        if ls_export_path:
            # AN EXPORT IN HAND -- a bundle's own, say -- read instead of a
            # fresh one, so a build can be re-run against exactly the boxes
            # an earlier one saw. Copied into the stage like a fresh export.
            ls_path = os.path.join(stage, 'label_studio_export.json')
            shutil.copy2(ls_export_path, ls_path)
            run.say('label studio: the export given, %s' % (ls_export_path,))
        elif not no_export:
            ls_path = ls_export(stage, run)
        if ls_path:
            ls_tasks, ls_counts = ls_read(ls_path)
            run.say('label studio: %d tasks, %d boxes (%s), %d background'
                    % (ls_counts['tasks'], ls_counts['boxes'],
                       ', '.join('%s %d' % kv for kv in
                                 sorted(ls_counts['classes'].items())),
                       ls_counts['background']))
        elif not no_export:
            run.say('no export script at the training root — building '
                    'without the hand-drawn boxes')
        run.progress(0, 'reading the annotations')
        if family == 'dogdet':
            mid = os.path.join(stage, 'detect')
            src = base
            if ls_path:
                # THE HAND-DRAWN BOXES BECOME THE BASE, rather than whatever
                # dogdet_v2 was cut from months ago. The project's own
                # preparation script does it: it knows the bucket, the label
                # group and the split tracker that keeps train and val stable
                # across rebuilds.
                #
                # Run with the STAGE as its working directory, because it
                # names its output after the export file relative to the cwd
                # AND writes dataset.yaml into the cwd -- run from the
                # training root it would overwrite the one that is there.
                run.progress(1, 'preparing the exported boxes')
                # THE EXCLUSION IS DERIVED FROM THE WHITELIST, not written
                # out beside it. The preparer only takes a list of classes to
                # drop, and a fixed list of two meant any label added in Label
                # Studio later -- horse, goat, sheep -- was not dropped: it
                # survived --single-class and became a dog, teaching the
                # detector that a horse is one. Everything this model does not
                # ask for is excluded, whatever it turns out to be called.
                keep = ls_wanted('dogdet')
                drop = sorted(set((ls_counts or {}).get('classes') or {})
                              - set(keep)) or list(LS_NOT_DOG)
                run.say('  keeping %s; dropping %s'
                        % (', '.join(sorted(keep)), ', '.join(drop)))
                run.run('prepare_detection_yolo_dataset',
                        [crop_python,
                         os.path.join(root, 'prepare_detection_yolo_dataset.py'),
                         '-f', ls_path, '-l', LS_GROUP,
                         '--single-class',
                         '-e', ','.join(drop),
                         '--background', '--compress',
                         '--tracker-file',
                         os.path.join(root, 'split_tracker.json')],
                        cwd=stage)
                made = os.path.join(stage,
                                    os.path.splitext(
                                        os.path.basename(ls_path))[0])
                if os.path.isdir(os.path.join(made, 'images', 'train')):
                    src = made
                    run.say('prepared %s from the export' % (made,))
                else:
                    run.say('the export prepared nothing usable — building '
                            'from %s instead' % (base,))
            run.progress(1, 'boxes, split by sequence')
            run.run('build_dogdet_v3',
                    [sys.executable, os.path.join(DETECT,
                                                  'build_dogdet_v3.py'),
                     '--src', src, '--out', mid,
                     '--detect-root', detect_root])
            run.progress(2, 'backgrounds from the false positives')
            run.run('build_detector_negatives',
                    [sys.executable,
                     os.path.join(DETECT, 'build_detector_negatives.py'),
                     '--src', mid, '--out', out,
                     '--detect-root', detect_root,
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
            run.progress(1, 'cutting every human-labelled crop')
            if ls_path is None:
                # A REFUSAL, not a warning: the base dataset is no longer a
                # source, so a build without an export is a build without
                # every hand-drawn box (48% of the leash rows) that would
                # otherwise call itself clean.
                if not allow_no_ls:
                    raise SystemExit(
                        'no Label Studio export (--no-export, or no export '
                        'script at the training root): the hand-drawn boxes '
                        'would not be in this build. Pass --ls-export PATH '
                        '(a bundle\'s label_studio_export.json will do), or '
                        '--allow-no-ls to build without them, on the record')
                run.say('WARNING: no Label Studio export -- the drawn boxes '
                        'are not in this build (--allow-no-ls)')
            collect = [sys.executable, os.path.join(DETECT, 'crop_sources.py'),
                       'collect', '--family', family, '--stage', stage,
                       '--detect-root', detect_root,
                       '--workers', str(int(workers)),
                       '--duckdb-python', duckdb_python]
            if ls_path:
                collect += ['--ls-export', ls_path]
            else:
                collect += ['--allow-empty-source', 'ls_box']
            # boto3 is in the training environment, not this one: the few
            # Label Studio frames no drive holds are fetched by a child
            # process there. The old path imported boto3 here, failed, and
            # cut 2,190 crops from resized copies while printing "no bucket".
            s3_python = s3_python or _cfg('dogbin_python')
            if s3_python:
                collect += ['--s3-python', s3_python]
            if allow_missing_frames:
                collect += ['--allow-missing-frames',
                            str(int(allow_missing_frames))]
            run.run('crop_sources', collect)
            manifest_rows = os.path.join(stage, 'crops.jsonl')
            # THE YARDSTICK, the store the model will be judging. Without a
            # configured one this is the gen-1 store, said out loud.
            size_root = size_ref_root or _cfg('size_ref_root') or detect_root
            if size_root == detect_root:
                run.say('size reference: the ledgers\' own store (%s) -- set '
                        'size_ref_root to the store production judges'
                        % (detect_root,))
            size_ref = os.path.join(stage, 'size_ref.json')
            run.run('size_reference',
                    [sys.executable, os.path.join(DETECT, 'crop_sources.py'),
                     'size-ref', '--detect-root', size_root,
                     '--out', size_ref])
            argv = [sys.executable,
                    os.path.join(DETECT, 'rebuild_crop_dataset.py'),
                    '--manifest', manifest_rows, '--out', out,
                    '--duckdb-python', duckdb_python,
                    '--pos-class', spec['classes'][0],
                    '--neg-class', spec['classes'][1],
                    '--max-per-sequence', str(spec['max_per_sequence']),
                    '--max-noseq-frac', str(max_noseq_frac),
                    '--size-ref', size_ref]
            if keep_noseq:
                argv += ['--keep-noseq']
            # the opt-in rebalance (off unless asked): recorded by the
            # rebuilder as 'variant' / 'rebalance' in its manifest
            argv += rebalance_flags
            for rp in reserve_paths:
                argv += ['--reserve', rp]
            if no_reserve:
                argv += ['--no-reserve']
            # THE DUPLICATES dHASH CANNOT SEE. The same dog turns up in TWO
            # sequences (two passes, two cameras), one frame later or three
            # metres closer, and dHash is blind to that by construction.
            # Those pairs are what leaks a subject across the train/val
            # boundary; the rebuilder joins every embedding cluster into one
            # split group. 0.93 is the calibrated threshold the promoted
            # datasets were built at, read off real pair sheets.
            clusters = os.path.join(stage, 'clusters.json')
            if no_embed_dedup:
                run.say('embedding dedup SKIPPED (--no-embed-dedup) -- '
                        'near-duplicates that dHash cannot see may cross '
                        'the split')
            else:
                run.progress(2, 'clustering near-duplicates')
                run.run('dedup_crops',
                        [crop_python, os.path.join(DETECT, 'dedup_crops.py'),
                         'cluster', os.path.join(stage, 'crops'),
                         '--threshold', '0.93', '--device', dedup_device,
                         '--out', clusters])
                argv += ['--dup-clusters', clusters]
            run.progress(3, 'de-duplicating and splitting by sequence group')
            run.run('rebuild_crop_dataset', argv + ['--execute'])
            # ...and once more from the outside, on what is actually on disk,
            # in a process that shares nothing with the one that wrote it
            run.run('verify_dataset',
                    [sys.executable,
                     os.path.join(DETECT, 'rebuild_crop_dataset.py'),
                     '--verify-only', out])
            extras = _read_json(os.path.join(stage, 'collect_counts.json'))
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
        crop_record = {}
        if spec['kind'] == 'classify':
            crop_record = _bundle_crop_record(out, stage, bundle, base,
                                              reserve_paths,
                                              classes=spec['classes'],
                                              say=run.say, family=family)
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
            # THE CROP MODELS' OWN RECORD: which store the ledgers' boxes were
            # matched against, which drives the frames came off, which
            # reserves were held out (with digests), and the framing used.
            'crops': crop_record or None,
            'detect_root': detect_root,
            'steps': run.steps,
            # WHAT WAS RUN, exactly. Reproducing a build means running the
            # same argv under the same interpreters against the same
            # annotations, and all three are recorded here.
            'command': {'argv': list(sys.argv), 'cwd': os.getcwd(),
                        'python': sys.executable,
                        'duckdb_python': duckdb_python,
                        'crop_python': crop_python},
            # A SKIPPED PASS IS A FACT ABOUT THE DATASET. A build without the
            # embedding dedup may hold the same subject on both sides of the
            # split, and whoever evaluates a model trained on it needs to
            # know that from the record, not from the shell history.
            'embed_dedup': (None if spec['kind'] != 'classify' else
                            'skipped' if no_embed_dedup else 'clustered'),
            # ...and so is a source left out, or a rule relaxed
            'allow_no_ls': bool(allow_no_ls) if spec['kind'] == 'classify'
            else None,
            'keep_noseq': bool(keep_noseq) if spec['kind'] == 'classify'
            else None,
            # plain, or rebalanced per creator / size band -- and with what
            'rebalance': variant if spec['kind'] == 'classify' else None,
            'test_set': test_set,
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
            'git': head,
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
        # Its 'manifest' and 'dup_clusters_file' name the stage, which is
        # deleted below; the bundle copies are recorded beside them.
        _point_at_bundle(manifest['builder_manifests'].get(
            'rebuild_manifest.json'), (crop_record or {}).get('records'))
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
        # A dataset without a bundle is one that did not finish: the bundle is
        # the last thing written. Left on disk it is offered in the selector
        # beside the real ones, with a plausible size, and training on it
        # trains on however much of it had been copied when the build died.
        if not os.path.isfile(os.path.join(out, 'bundle', 'manifest.json')):
            shutil.rmtree(out, ignore_errors=True)


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
    ap.add_argument('--no-embed-dedup', action='store_true',
                    help='skip the embedding near-duplicate pass (the dHash '
                         'pass still runs). The skip is recorded in the '
                         'manifest: a build without it may let the same '
                         'subject, seen from another distance or angle, land '
                         'on both sides of the train/val split.')
    ap.add_argument('--no-export', action='store_true',
                    help='skip the Label Studio export. For a rebuild that '
                         'must match an older one exactly, or for a machine '
                         'with no reach to the server -- and it is recorded, '
                         'so a dataset built without the hand-drawn boxes '
                         'says so')
    ap.add_argument('--ls-export', metavar='PATH',
                    help='use this Label Studio export (for instance a '
                         'bundle\'s label_studio_export.json) instead of '
                         'fetching a fresh one')
    ap.add_argument('--detect-root',
                    help='the store the ledgers\' det_idx and confidences '
                         'belong to, for every family (default: the pinned '
                         'ledger root, data/ledger_detect_root.txt; a '
                         'disagreeing $DETECT_ROOT is refused, and so is a '
                         'store whose shards are not only gen=0001)')
    ap.add_argument('--size-ref-root',
                    help='crop models: the store production judges, for the '
                         'size report (default: the size_ref_root config key, '
                         'else --detect-root)')
    ap.add_argument('--s3-python',
                    help='crop models: interpreter with boto3, for the few '
                         'Label Studio frames no drive holds (default: the '
                         'dogbin_python config key)')
    ap.add_argument('--workers', type=int, default=4,
                    help='crop models: frame decoders (run at low priority)')
    ap.add_argument('--allow-missing-frames', type=int, default=0,
                    metavar='N',
                    help='crop models: build even if up to N boxes have no '
                         'full-resolution frame (counted in the manifest)')
    ap.add_argument('--reserve', action='append', default=[],
                    help='crop models: another reserve file to hold out, ON '
                         'TOP of the acceptance and test sets (repeatable)')
    ap.add_argument('--test-set', metavar='PATH',
                    help='crop models: the frozen test set to hold out in '
                         'place of data/<family>_test_v1.json (the '
                         'acceptance set is held out regardless)')
    ap.add_argument('--allow-no-ls', action='store_true',
                    help='crop models: build without a Label Studio export '
                         '(refused otherwise: the drawn boxes would be '
                         'missing). Recorded.')
    ap.add_argument('--keep-noseq', action='store_true',
                    help='crop models: keep crops whose sequence no manifest '
                         'knows, pinned to train (default: dropped and '
                         'counted)')
    ap.add_argument('--no-reserve', action='store_true',
                    help='crop models: hold nothing out, on the record')
    ap.add_argument('--max-noseq-frac', type=float, default=0.02,
                    help='crop models: fail if more than this share of crops '
                         'has no sequence in any manifest')
    ap.add_argument('--dedup-device', default='auto',
                    help='crop models: device for the embedding dedup')
    ap.add_argument('--max-per-creator', type=int, default=0, metavar='N',
                    help='crop models, OPT-IN: keep at most N crops per '
                         '(Mapillary creator, class), after the split (%d '
                         'suggested). Off by default; recorded'
                         % (_rcd.MAX_PER_CREATOR_SUGGESTED,))
    ap.add_argument('--balance-size-bands', action='store_true',
                    help='crop models, OPT-IN: after the split and the '
                         'creator cap, balance each split\'s positive rate '
                         'per crop short-side band. Off by default; recorded')
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
    if a.max_per_creator < 0:
        ap.error('--max-per-creator must be 0 (off) or more')
    variant, flags = rebalance_variant(a.max_per_creator,
                                       a.balance_size_bands)
    if flags and FAMILIES[a.family]['kind'] != 'classify':
        ap.error('--max-per-creator / --balance-size-bands are for the crop '
                 'models')
    if a.dry_run:
        print('would build %s in %s' % (new_name(a.family, variant=variant),
                                        training_root()))
        if FAMILIES[a.family]['kind'] == 'classify':
            print('  rebalance: %s%s' % (variant['variant'],
                                         (' (' + ' '.join(flags) + ')')
                                         if flags else ' (no per-creator '
                                         'cap, no size-band balance)'))
        table = stores_for(a.detect_root) if a.detect_root else STORES
        for k, v in sorted(table.items()):
            if a.family in v['for']:
                st = store_state(k, table)
                print('  %-18s sha %s  %s lines'
                      % (k, (st['sha256'] or '-')[:12], st['lines']))
        return 0
    build(a.family, out=a.out, by=a.by, duckdb_python=a.duckdb_python,
          crop_python=a.crop_python, keep_stage=a.keep_stage,
          no_export=a.no_export, no_embed_dedup=a.no_embed_dedup,
          ls_export_path=a.ls_export, detect_root=a.detect_root,
          s3_python=a.s3_python, workers=a.workers,
          allow_missing_frames=a.allow_missing_frames,
          no_reserve=a.no_reserve, max_noseq_frac=a.max_noseq_frac,
          reserves=a.reserve or None, size_ref_root=a.size_ref_root,
          dedup_device=a.dedup_device, test_set=a.test_set,
          allow_no_ls=a.allow_no_ls, keep_noseq=a.keep_noseq,
          max_per_creator=a.max_per_creator,
          balance_size_bands=a.balance_size_bands)
    return 0


if __name__ == '__main__':
    sys.exit(main())
