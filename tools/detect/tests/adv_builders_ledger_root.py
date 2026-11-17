#!/usr/bin/env python3
"""The detector builders read the store the human ledgers were made on.

build_dogdet_v3 turns a confirmed positive into a label (n_det == 1, det 0 or
its gen-1 correction) and build_detector_negatives turns a flagged false
positive into an empty-label background (one detection, the flagged one).
Both are statements about the gen-1 boxes. Read off the live store after the
switch-over to gen 2 they go wrong silently: frames newly single in gen 2
take an unreviewed det 0 as the dog label, gen-1 corrections land on whatever
gen 2 numbered 0, and backgrounds go in on boxes nobody flagged. So:

  n1  build_detector_negatives reads the pinned ledger root
      (crop_sources.default_detect_root), not the live store pointer --
      detections_per_image and resolve_originals both
  n2  a lone detection must be the flagged one: conf within 0.006 of the
      flag's, and a flag with no conf vouches for nothing
  n3  main: the default store is the pinned one, --detect-root replaces it,
      and the manifest records which, with the not-the-flagged-box count;
      a --detect-root holding another generation is refused unless
      --any-generation
  v1  build_dogdet_v3.frame_meta reads the pinned ledger root
  v2  main: default pinned, --detect-root replaces it, manifest records it;
      a --detect-root holding another generation is refused unless
      --any-generation
  v3  leashed_ids reads only leash dirs that judged THIS store's boxes: a
      gen-2 data/leash is never read against gen-1 det 0, a renamed gen-1
      record is; today's shape (data/leash holding only an archive) gives
      an empty filter
  b1  build_dataset resolves the pinned root for a dogdet build, hands it to
      both builders as --detect-root and records it in the manifest; an
      explicit root is passed through

Every MUTANT puts a wrong behaviour back and must make its check fail. Writes
only under a temp dir.

    python tools/detect/tests/adv_builders_ledger_root.py
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(DETECT))
sys.path.insert(0, DETECT)

import build_dataset  # noqa: E402
import build_detector_negatives  # noqa: E402
import build_dogdet_v3  # noqa: E402
import crop_sources  # noqa: E402
import frame_labels  # noqa: E402
import harvest_flagged  # noqa: E402
import rebuild_crop_dataset  # noqa: E402
import store  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name +
          (f' -- {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


def mutant_module(mod, old, new):
    """A fresh copy of `mod` executed from its source with `old` -> `new`.

    The whole module, not one function: main() rebinds a module global that
    the helpers read, and a function compiled into a copied namespace would
    write one dict while they read another."""
    with open(mod.__file__) as fh:
        src = fh.read()
    if src.count(old) != 1:
        raise AssertionError('mutant anchor %r not found once in %s'
                             % (old, os.path.basename(mod.__file__)))
    m = types.ModuleType(mod.__name__ + '_mutant')
    m.__file__ = mod.__file__
    exec(compile(src.replace(old, new), mod.__file__, 'exec'), m.__dict__)
    return m


# ── fixtures ────────────────────────────────────────────────────────────────

def make_store(root, gen, frames, placements=(('r', 'c', 'd1'),
                                               ('r', 'c2', 'd1'))):
    """A detector store under shards/gen=<gen>/, every frame in every
    placement (cell twins, the shape the live store holds).
    frames: {image_id: [(det_idx, conf)]}."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    store.ensure_bootstrap(root)
    img_rows, det_rows = [], []
    for iid, dets in frames.items():
        img_rows.append((int(iid), len(dets)))
        det_rows += [(int(iid), di, c) for di, c in dets]
    img = pa.table({
        'image_id': pa.array([r[0] for r in img_rows], pa.uint64()),
        'drive': pa.array([1] * len(img_rows), pa.uint8()),
        'status': pa.array([1] * len(img_rows), pa.uint8()),
        'n_det': pa.array([r[1] for r in img_rows], pa.uint16()),
        'max_conf': pa.array([0.5] * len(img_rows), pa.float32()),
        'orig_w': pa.array([400] * len(img_rows), pa.uint16()),
        'orig_h': pa.array([300] * len(img_rows), pa.uint16()),
        'reduce': pa.array([0] * len(img_rows), pa.uint8()),
        'guards': pa.array([0] * len(img_rows), pa.uint16()),
        'ts_off': pa.array([0] * len(img_rows), pa.uint32()),
        'run_id': pa.array([1] * len(img_rows), pa.uint16()),
        'shard_idx': pa.array([0] * len(img_rows), pa.uint32()),
        'model_sha8': pa.array(['ab'] * len(img_rows))},
        schema=store.IMG_SCHEMA)
    n = len(det_rows)
    det = pa.table({
        'image_id': pa.array([r[0] for r in det_rows], pa.uint64()),
        'det_idx': pa.array([r[1] for r in det_rows], pa.uint8()),
        'conf': pa.array([r[2] for r in det_rows], pa.float32()),
        'x1': pa.array([10.0] * n, pa.float32()),
        'y1': pa.array([10.0] * n, pa.float32()),
        'x2': pa.array([90.0] * n, pa.float32()),
        'y2': pa.array([90.0] * n, pa.float32()),
        'run_id': pa.array([1] * n, pa.uint16()),
        'shard_idx': pa.array([0] * n, pa.uint32()),
        'leash_class': pa.array([None] * n, pa.uint8()),
        'leash_conf': pa.array([None] * n, pa.float32()),
        'model_sha8': pa.array(['ab'] * n)}, schema=store.DET_SCHEMA)
    for region, cell, drive in placements:
        d = os.path.join(root, 'shards', 'gen=%04d' % gen,
                         f'region={region}', f'cell={cell}',
                         f'drive={drive}')
        os.makedirs(d, exist_ok=True)
        pq.write_table(img, os.path.join(d, 's00000.p000000_000001'
                                            '.img.parquet'))
        pq.write_table(det, os.path.join(d, 's00000.p000000_000001'
                                            '.det.parquet'))
    return root


# Frames, and what each store makes of them.
#   1001  gen 1: one detection, conf 0.42 (flagged at 0.42)  gen 2: two
#   1002  gen 1: two detections                              gen 2: absent
#   1003  gen 1: one detection, conf 0.80 (flagged at 0.42)  gen 2: absent
#   1004  gen 1: one detection, conf 0.33 (flag has no conf) gen 2: absent
#   1005  gen 1: absent                                      gen 2: one, 0.55
G1_FRAMES = {'1001': [(0, 0.42)], '1002': [(0, 0.30), (1, 0.70)],
             '1003': [(0, 0.80)], '1004': [(0, 0.33)]}
G2_FRAMES = {'1001': [(0, 0.42), (1, 0.61)], '1005': [(0, 0.55)]}
FLAGS = {'1001': 0.42, '1002': 0.30, '1003': 0.42, '1004': None,
         '1005': 0.55}
# The flag on 1005 was made on the gen-2 detector's box, and says so: a
# reader keeps only the rows of the generation of the store it joins
# (untagged = gen 1), so --any-generation lets the gen-2 store be named and
# the row's own tag decides which flags it is joined with.
FLAG_GEN = {'1005': 2}


class World:
    """The fixture stores, a fake grid and the stand-ins every check needs:
    crop_sources.default_detect_root -> the gen-1 store (the pin), and
    store.get_detect_root -> the gen-2 store (the live pointer after the
    switch), so reading the live pointer is visible."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.g1 = make_store(os.path.join(tmp, 'detect'), 1, G1_FRAMES)
        self.g2 = make_store(os.path.join(tmp, 'detect_g2'), 2, G2_FRAMES)
        self.grid = os.path.join(tmp, 'grid')
        for cell in ('c', 'c2'):
            d = os.path.join(self.grid, cell, 'ground_animal_images')
            os.makedirs(d, exist_ok=True)
            for iid in set(G1_FRAMES) | set(G2_FRAMES):
                with open(os.path.join(d, iid + '.jpg'), 'wb') as fh:
                    fh.write(b'\xff\xd8' + iid.encode())
        self.ledger = os.path.join(tmp, 'labels.jsonl')
        with open(self.ledger, 'w') as fh:
            for iid, conf in FLAGS.items():
                row = {'image_id': iid, 'label': 'false_positive'}
                if conf is not None:
                    row['conf'] = conf
                if iid in FLAG_GEN:
                    row['gen'] = FLAG_GEN[iid]
                fh.write(json.dumps(row) + '\n')
        self.batches = os.path.join(tmp, 'no_batches')
        os.makedirs(self.batches, exist_ok=True)

    @contextlib.contextmanager
    def stood_in(self):
        saved = (crop_sources.default_detect_root, store.get_detect_root,
                 os.environ.get('DETECT_ROOT'), frame_labels.BATCHES)
        crop_sources.default_detect_root = lambda: self.g1
        store.get_detect_root = lambda *a, **k: self.g2
        os.environ.pop('DETECT_ROOT', None)
        frame_labels.BATCHES = self.batches
        try:
            yield
        finally:
            (crop_sources.default_detect_root, store.get_detect_root,
             env, frame_labels.BATCHES) = saved
            if env is not None:
                os.environ['DETECT_ROOT'] = env


# ── build_detector_negatives ────────────────────────────────────────────────

def check_negatives(w, bn):
    """n1-n3 against module `bn` (the real one, or a mutant)."""
    bad = []
    with w.stood_in():
        bn.DETECT_ROOT = None
        # n1: the pinned store, not the live pointer
        per = bn.detections_per_image({'1001', '1002', '1005'}, REPO)
        if per != {'1001': 1, '1002': 2}:
            bad.append('n1 detections_per_image did not read the pinned '
                       'gen-1 store: %r' % (per,))
        seen = []
        real_cells = harvest_flagged.cells_for
        harvest_flagged.cells_for = lambda ids, root: (
            seen.append(root) or {})
        try:
            roots = os.path.join(w.tmp, 'roots.txt')
            with open(roots, 'w') as fh:
                fh.write(w.grid + '\n')
            bn.DETECT_ROOT = None
            bn.resolve_originals({'1001'}, roots, REPO)
        finally:
            harvest_flagged.cells_for = real_cells
        if seen != [w.g1]:
            bad.append('n1 resolve_originals placed frames through %r, not '
                       'the pinned store' % (seen,))
        # n2: the lone detection must be the flagged one
        bn.DETECT_ROOT = None
        rows = bn.flagged_ids(w.ledger)
        per = bn.detections_per_image(
            {i: rows[i] for i in ('1001', '1002', '1003', '1004')}, REPO)
        if per != {'1001': 1, '1002': 2, '1003': None, '1004': None}:
            bad.append('n2 a lone detection that is not the flagged box '
                       '(conf 0.80 vs 0.42, or no conf on the flag) is not '
                       'refused: %r' % (per,))
        near = dict(rows['1001'], conf=0.425)
        per = bn.detections_per_image({'1001': near}, REPO)
        if per != {'1001': 1}:
            bad.append('n2 a conf within 0.006 (the ledger rounds to two '
                       'places) is not accepted: %r' % (per,))
        # n3: main, default and explicit
        man, err = run_negatives(w, bn, ['--detect-root', w.g2])
        if not err or '--detect-root refused' not in err:
            bad.append('n3 a gen-2 --detect-root without --any-generation '
                       'was not refused: %s' % (err or man.get('detect_root')))
        for tag, extra, want_root, want_ids, want_off in (
                ('default', [], w.g1, ['1001'], 2),
                ('--detect-root g1', ['--detect-root', w.g1], w.g1,
                 ['1001'], 2),
                ('--detect-root g2 --any-generation',
                 ['--detect-root', w.g2, '--any-generation'], w.g2,
                 ['1005'], 0)):
            man, err = run_negatives(w, bn, extra)
            if err:
                bad.append('n3 %s raised %s' % (tag, err))
                continue
            if man.get('detect_root') != want_root:
                bad.append('n3 %s: manifest detect_root %r, want %r'
                           % (tag, man.get('detect_root'), want_root))
            if man.get('added_image_ids') != want_ids:
                bad.append('n3 %s: backgrounds %r, want %r'
                           % (tag, man.get('added_image_ids'), want_ids))
            if man.get('lone_detection_not_flagged') != want_off:
                bad.append('n3 %s: lone_detection_not_flagged %r, want %r'
                           % (tag, man.get('lone_detection_not_flagged'),
                              want_off))
    return bad


def run_negatives(w, bn, extra):
    src = os.path.join(w.tmp, 'negsrc')
    if not os.path.isdir(src):
        for split in ('train', 'val'):
            for sub in ('images', 'labels'):
                os.makedirs(os.path.join(src, sub, split))
        for k in range(4):
            iid = str(3000000 + k)
            open(os.path.join(src, 'images', 'train', iid + '.jpg'),
                 'wb').close()
            with open(os.path.join(src, 'labels', 'train', iid + '.txt'),
                      'w') as fh:
                fh.write('0 0.5 0.5 0.1 0.1\n')
    out = tempfile.mkdtemp(prefix='negout_', dir=w.tmp)
    shutil.rmtree(out)
    real_ro, real_seq = bn.resolve_originals, rebuild_crop_dataset.\
        resolve_sequences
    jpg = {i: os.path.join(w.grid, 'c', 'ground_animal_images', i + '.jpg')
           for i in FLAGS}
    try:
        bn.resolve_originals = lambda ids, rf, repo: {i: jpg[i] for i in ids}
        rebuild_crop_dataset.resolve_sequences = (
            lambda ids, repo, py: {i: 'S' + i for i in ids})
        with contextlib.redirect_stdout(io.StringIO()):
            bn.main(['--src', src, '--out', out, '--ledger', w.ledger,
                     '--target-frac', '0.9', '--no-hidden-dogs',
                     '--execute'] + extra)
        with open(os.path.join(out, 'negatives_manifest.json')) as fh:
            return json.load(fh), None
    except BaseException as e:  # noqa: BLE001 - a raise is a failed build
        return {}, '%s: %s' % (type(e).__name__, e)
    finally:
        bn.resolve_originals = real_ro
        rebuild_crop_dataset.resolve_sequences = real_seq


# ── build_dogdet_v3 ─────────────────────────────────────────────────────────

def check_dogdet(w, v3):
    """v1-v3 against module `v3`."""
    import duckdb
    bad = []
    with w.stood_in():
        # v1: frame_meta on the pinned store
        v3.DETECT_ROOT = None
        meta, boxes = v3.frame_meta(['1001', '1005'], duckdb.connect())
        if {i: m[0] for i, m in meta.items()} != {'1001': 1} \
                or ('1001', 0) not in boxes:
            bad.append('v1 frame_meta did not read the pinned gen-1 store: '
                       '%r' % ({i: m[0] for i, m in meta.items()},))
        # v2: main, default and explicit
        man, err = run_dogdet(w, v3, ['--detect-root', w.g2])
        if not err or '--detect-root refused' not in err:
            bad.append('v2 a gen-2 --detect-root without --any-generation '
                       'was not refused: %s' % (err or man.get('detect_root')))
        for tag, extra, want_root, want_ids in (
                ('default', [], w.g1, ['1001']),
                ('--detect-root g1', ['--detect-root', w.g1], w.g1,
                 ['1001']),
                ('--detect-root g2 --any-generation',
                 ['--detect-root', w.g2, '--any-generation'], w.g2,
                 ['1005'])):
            man, err = run_dogdet(w, v3, extra)
            if err:
                bad.append('v2 %s raised %s' % (tag, err))
                continue
            if man.get('detect_root') != want_root:
                bad.append('v2 %s: manifest detect_root %r, want %r'
                           % (tag, man.get('detect_root'), want_root))
            if man.get('added_ids') != want_ids:
                bad.append('v2 %s: additions %r, want %r'
                           % (tag, man.get('added_ids'), want_ids))
        # v3: which leash calls are read
        bad += check_leash(w, v3)
    return bad


def run_dogdet(w, v3, extra):
    v2 = os.path.join(w.tmp, 'v2')
    if not os.path.isdir(v2):
        for split, iid in (('train', '4000001'), ('val', '4000002')):
            for sub in ('images', 'labels'):
                os.makedirs(os.path.join(v2, sub, split))
            open(os.path.join(v2, 'images', split, iid + '.jpg'),
                 'wb').close()
            with open(os.path.join(v2, 'labels', split, iid + '.txt'),
                      'w') as fh:
                fh.write('0 0.5 0.5 0.1 0.1\n')
    out = tempfile.mkdtemp(prefix='v3out_', dir=w.tmp)
    stood = {k: getattr(v3, k) for k in (
        'resolve_sequences', '_grid_roots', 'confirmed_positive_ids',
        'corrections')}
    real_read = frame_labels.read
    try:
        v3.resolve_sequences = lambda ids, con: {i: 'S' + i for i in ids}
        v3._grid_roots = lambda: {1: w.grid, '1': w.grid, 'd1': w.grid}
        v3.confirmed_positive_ids = lambda: {'1001', '1005'}
        v3.corrections = lambda: {}
        frame_labels.read = lambda path=None: {}
        with contextlib.redirect_stdout(io.StringIO()):
            v3.main(['--src', v2, '--out', out, '--val-target', '0',
                     '--no-hidden-dogs'] + extra)
        with open(os.path.join(out, 'manifest.json')) as fh:
            return json.load(fh), None
    except BaseException as e:  # noqa: BLE001
        return {}, '%s: %s' % (type(e).__name__, e)
    finally:
        for k, v in stood.items():
            setattr(v3, k, v)
        frame_labels.read = real_read


def check_leash(w, v3):
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq
    bad = []

    def shard(path, iid):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(pa.table({'image_id': [iid], 'det_idx': [0],
                                 'label': ['leashed']}), path)

    def plan(d, **kw):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'plan.json'), 'w') as fh:
            json.dump(kw, fh)

    def called(repo, root):
        saved = v3.REPO
        v3.REPO, v3.DETECT_ROOT = repo, root
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return v3.leashed_ids(['1001', '1005', '1009'],
                                      duckdb.connect())
        except BaseException as e:  # noqa: BLE001
            return '%s: %s' % (type(e).__name__, e)
        finally:
            v3.REPO, v3.DETECT_ROOT = saved, None

    # today: data/leash holds only the archived (invalid) gen-1 run
    today = os.path.join(w.tmp, 'leash_today')
    shard(os.path.join(today, 'data', 'leash', 'archive', '20261115',
                       'leash-00000.parquet'), '1001')
    got = called(today, w.g1)
    if got != set():
        bad.append('v3 today\'s data/leash (an archive only) is not an '
                   'empty filter: %r' % (got,))
    # after the switch: gen 2 in data/leash, gen 1 renamed beside it
    after = os.path.join(w.tmp, 'leash_after')
    live = os.path.join(after, 'data', 'leash')
    plan(live, gen=2, detect_root=w.g2)
    shard(os.path.join(live, 'leash-00000.parquet'), '1005')
    shard(os.path.join(after, 'data', 'leash_gen1_20261118',
                       'leash-00000.parquet'), '1001')
    got = called(after, w.g1)
    if got != {'1001'}:
        bad.append('v3 a gen-1 build after the switch read %r, want the '
                   'renamed gen-1 record only ({\'1001\'})' % (got,))
    got = called(after, w.g2)
    if got != {'1005'}:
        bad.append('v3 a gen-2 build read %r, want the gen-2 calls only '
                   '({\'1005\'})' % (got,))
    # a gen-2 plan without detect_root still says gen 2
    bare = os.path.join(w.tmp, 'leash_bare')
    plan(os.path.join(bare, 'data', 'leash'), gen=2)
    shard(os.path.join(bare, 'data', 'leash', 'leash-00000.parquet'), '1005')
    got = called(bare, w.g1)
    if got != set():
        bad.append('v3 a gen-2 plan (no detect_root) was read against the '
                   'gen-1 store: %r' % (got,))
    return bad


# ── build_dataset ───────────────────────────────────────────────────────────

def check_compose(w, bd):
    """b1 against module `bd`."""
    bad = []
    root = os.path.join(w.tmp, 'troot')
    os.makedirs(os.path.join(root, 'dogdet_v2'), exist_ok=True)
    saved_env = os.environ.get('TRAINING_ROOT')
    os.environ['TRAINING_ROOT'] = root
    calls = []

    def fake_run(self, name, argv, cwd=None, env=None):
        calls.append((name, list(argv)))
        if '--out' in argv:
            out = argv[argv.index('--out') + 1]
            for sub in ('images', 'labels'):
                for split in ('train', 'val'):
                    os.makedirs(os.path.join(out, sub, split), exist_ok=True)
        self.steps.append({'name': name, 'argv': argv, 'exit_code': 0,
                           'seconds': 0.0})
        return 0

    real_run, real_head = bd.Runner.run, bd.git_head
    try:
        bd.Runner.run = fake_run
        bd.git_head = lambda: {'commit': 'x', 'dirty': False}
        with w.stood_in():
            for tag, given, want in (('default', None, w.g1),
                                     ('explicit', w.g2, w.g2)):
                del calls[:]
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        man = bd.build('dogdet', by='t', no_export=True,
                                       detect_root=given,
                                       out=os.path.join(root, 'dd_%s_%d' % (
                                           tag, len(os.listdir(root)))))
                except BaseException as e:  # noqa: BLE001
                    bad.append('b1 %s raised %s: %s'
                               % (tag, type(e).__name__, e))
                    continue
                for step in ('build_dogdet_v3', 'build_detector_negatives'):
                    argv = next((a for n, a in calls if n == step), None)
                    got = (argv[argv.index('--detect-root') + 1]
                           if argv and '--detect-root' in argv else None)
                    if got != want:
                        bad.append('b1 %s: %s was handed --detect-root %r, '
                                   'want %r' % (tag, step, got, want))
                if man.get('detect_root') != want:
                    bad.append('b1 %s: manifest detect_root %r, want %r'
                               % (tag, man.get('detect_root'), want))
    finally:
        bd.Runner.run, bd.git_head = real_run, real_head
        if saved_env is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = saved_env
    return bad


# ── mutants ─────────────────────────────────────────────────────────────────

BN, V3, BD = build_detector_negatives, build_dogdet_v3, build_dataset
MUTANTS = [
    ('n1 detections_per_image reads the live pointer', check_negatives, BN,
     "    root = detect_root()\n    src = _store._sql_src",
     "    root = _store.get_detect_root()\n    src = _store._sql_src"),
    ('n1 resolve_originals reads the live pointer', check_negatives, BN,
     'placed = cells_for(list(ids), detect_root())',
     'placed = cells_for(list(ids), _store.get_detect_root())'),
    ('n2 no conf match', check_negatives, BN,
     '    if isinstance(ids, dict):', '    if False:'),
    ('n2 a flag without conf vouches', check_negatives, BN,
     'if n == 1 and (want is None or lo is None',
     'if n == 1 and want is not None and (lo is None'),
    ('n2 tolerance ignored', check_negatives, BN,
     'tol = _crop_sources().CONF_TOL', 'tol = 1.0'),
    ('n3 main passes bare ids', check_negatives, BN,
     'detections_per_image({i: flags[i] for i in cand}, REPO)',
     'detections_per_image(cand, REPO)'),
    ('n3 --detect-root ignored', check_negatives, BN,
     '        DETECT_ROOT = os.path.abspath(args.detect_root)\n',
     '        pass\n'),
    ('n3 another generation not refused', check_negatives, BN,
     '        if why and not args.any_generation:',
     '        if False:'),
    ('n3 --any-generation ignored', check_negatives, BN,
     '        if why and not args.any_generation:',
     '        if why:'),
    ('n3 manifest does not record the store', check_negatives, BN,
     "        'detect_root': DETECT_ROOT,", "        'detect_root': None,"),
    ('v1 frame_meta reads the live pointer', check_dogdet, V3,
     '    root = detect_root()\n    img = store._sql_src',
     '    root = store.get_detect_root()\n    img = store._sql_src'),
    ('v2 --detect-root ignored', check_dogdet, V3,
     '        DETECT_ROOT = os.path.abspath(a.detect_root)',
     '        pass'),
    ('v2 another generation not refused', check_dogdet, V3,
     '        if why and not a.any_generation:',
     '        if False:'),
    ('v2 --any-generation ignored', check_dogdet, V3,
     '        if why and not a.any_generation:',
     '        if why:'),
    ('v2 manifest does not record the store', check_dogdet, V3,
     "        'detect_root': DETECT_ROOT,", "        'detect_root': None,"),
    ('v3 data/leash read whatever its generation', check_dogdet, V3,
     '    dirs = leash_dirs(detect_root())',
     "    dirs = [os.path.join(REPO, 'data', 'leash')]"),
    ('v3 a plan\'s detect_root not compared', check_dogdet, V3,
     "            ok = os.path.realpath(plan['detect_root']) == want",
     '            ok = True'),
    ('v3 a plan\'s gen not compared', check_dogdet, V3,
     "            ok = gens == {int(plan.get('gen') or 1)}",
     '            ok = True'),
    ('v3 archived runs read too', check_dogdet, V3,
     "glob.glob(os.path.join(d, 'leash-*.parquet'))",
     "glob.glob(os.path.join(d, '**', 'leash-*.parquet'), recursive=True)"),
    ('b1 dogdet keeps the builders\' own default', check_compose, BD,
     "    if detect_root is None:\n        sys.path.insert(0, DETECT)",
     "    if detect_root is None and spec['kind'] == 'classify':\n"
     "        sys.path.insert(0, DETECT)"),
    ('b1 build_dogdet_v3 not handed the root', check_compose, BD,
     "'--src', src, '--out', mid,\n                     '--detect-root', "
     "detect_root])",
     "'--src', src, '--out', mid])"),
    ('b1 build_detector_negatives not handed the root', check_compose, BD,
     "                     '--detect-root', detect_root,\n"
     "                     '--duckdb-python', duckdb_python, '--execute'])",
     "                     '--duckdb-python', duckdb_python, '--execute'])"),
]


def main():
    tmp = tempfile.mkdtemp(prefix='adv_builders_root_')
    try:
        w = World(tmp)
        for fn, mod in ((check_negatives, BN), (check_dogdet, V3),
                        (check_compose, BD)):
            try:
                got = fn(w, mod)
            except Exception as e:  # noqa: BLE001 - report, not die
                got = ['threw %s: %s' % (type(e).__name__, e)]
            check(fn.__name__, not got, '; '.join(got))
        for name, fn, mod, old, new in MUTANTS:
            try:
                got = fn(w, mutant_module(mod, old, new))
            except AssertionError:
                raise
            except Exception as e:  # noqa: BLE001
                got = ['threw %s: %s' % (type(e).__name__, e)]
            check('mutant ' + name + ' is caught', bool(got),
                  '' if got else 'the check still passes with it -- it '
                  'proves nothing')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\n%d failed' % len(FAILS) if FAILS else '\nall passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
