#!/usr/bin/env python3
"""A dog found but boxed the other convention's way is counted once.

    python tools/detect/tests/adv_tolerant_match.py

The detector set carries two box conventions on purpose: a Label Studio
leashed dog is boxed with its leash, a detector-confirmed one without. Strict
IoU 0.5 scoring then counts one disagreement about the same dog twice -- an
invented box AND a missed dog. run_mistakes.match_boxes() keeps the strict
pass exactly as it was and pairs what is left over when one box sits mostly
inside the other; the dashboard shows the pair as its own kind and re-scores a
file written under the old rule.

Graded against boxes built for the purpose, never against a run on disk:

  m1  a strict hit is a hit and nothing else
  m2  a leash-inclusive box around a dog-only label is ONE pair
  m3  a tight box inside a loose label is one pair too
  m4  a huge box over a tiny label is not a pair (the IoU floor)
  m5  two offset boxes, neither inside the other, are not a pair
  m6  one box over two labelled dogs is a merge: never paired
  m7  pairing is one to one, the closer box wins
  m8  a label the strict pass took is not paired again
  m9  different classes never pair
  m10 fuzzed: every box lands in exactly one bucket, and the strict hits are
      the old algorithm's hits, box for box
  m11 a run's training imgsz is read off its args.yaml, junk is None
  m12 score_detect writes the pair once, with both boxes, ordered between
      inventions and misses
  m13 run_one writes both recalls, the rule, and the run's own imgsz --
      and only into the directory it was pointed at
  m14 --imgsz defaults to the run's own size, not 640
  d1  the dashboard calls a detector file without the rule stale
  d2  a stale file is re-scored with --force; a fresh one is left alone; a
      live scorer is never doubled
  d3  a scorer still alive is pending however long it has taken
  d4  the frame note says which way the boxes disagree
  d5  the frame viewer draws the pair's box and its label, the label once
  d6  the panel shows the new group and both recalls, and marks a stale file
  d7  the page's script draws and keys the two new box kinds
"""
import json
import os
import random
import shutil
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))

try:
    from PIL import Image
except ImportError:
    print('SKIP-AS-FAILURE: PIL is not importable here -- nothing was checked')
    sys.exit(1)

import dashboard as d  # noqa: E402
import run_mistakes as rm  # noqa: E402

_fails = []


class _Skip(Exception):
    """A check failed in a way that leaves nothing for the rest to read."""


def check(label, ok, detail=''):
    print(('ok   ' if ok else 'FAIL ') + label + ('' if ok else '  ' + detail))
    if not ok:
        _fails.append(label.split()[0])


def P(box, conf=0.9, cls=0):
    return (cls, conf, list(box))


def G(box, cls=0):
    return (cls, list(box))


def buckets(preds, gt):
    h, p, i, m = rm.match_boxes(preds, gt)
    return {'hits': h, 'pairs': p, 'invented': i, 'missed': m}


# ── m1..m9: the rule, case by case ──────────────────────────────────────────
DOG = [100, 100, 140, 150]  # a dog, boxed dog-only
LEASHED = [60, 90, 145, 180]  # the same dog with its leash
b = buckets([P(DOG)], [G([101, 101, 141, 151])])
check('m1 a strict hit is a hit and nothing else', b == {
    'hits': [(0, 0)],
    'pairs': [],
    'invented': [],
    'missed': []
}, f'got {b}')

b = buckets([P(LEASHED)], [G(DOG)])
check(
    'm2 a leash-inclusive box around a dog-only label is one pair, not an '
    'invention and a miss', b == {
        'hits': [],
        'pairs': [(0, 0)],
        'invented': [],
        'missed': []
    }, f'IoU {rm._iou(LEASHED, DOG):.2f}, got {b}')

b = buckets([P(DOG)], [G(LEASHED)])
check('m3 a tight box inside a loose label is one pair', b == {
    'hits': [],
    'pairs': [(0, 0)],
    'invented': [],
    'missed': []
}, f'got {b}')

HUGE = [0, 0, 400, 400]
b = buckets([P(HUGE)], [G(DOG)])
check('m4 a huge box over a tiny label is not a pair: under the IoU floor',
      b == {
          'hits': [],
          'pairs': [],
          'invented': [0],
          'missed': [0]
      }, f'IoU {rm._iou(HUGE, DOG):.3f}, got {b}')

SIDE = [120, 100, 160, 150]  # half on, half off the dog
b = buckets([P(SIDE)], [G(DOG)])
check('m5 two offset boxes, neither inside the other, are not a pair', b == {
    'hits': [],
    'pairs': [],
    'invented': [0],
    'missed': [0]
}, f'IoU {rm._iou(SIDE, DOG):.2f}, got {b}')

TWO = [[100, 100, 140, 150], [150, 100, 190, 150]]
OVER = [95, 95, 195, 155]
b = buckets([P(OVER)], [G(TWO[0]), G(TWO[1])])
check('m6 one box over two labelled dogs is a merge, never a pair', b == {
    'hits': [],
    'pairs': [],
    'invented': [0],
    'missed': [0, 1]
}, f'got {b}')

CLOSE = [95, 95, 142, 155]  # closer to DOG than LEASHED is
b = buckets([P(LEASHED, 0.95), P(CLOSE, 0.6)], [G(DOG)])
# CLOSE may clear IoU 0.5 outright; what matters is one dog, one credit
credited = len(b['hits']) + len(b['pairs'])
check(
    'm7 pairing is one to one: one dog is credited once, the other box is '
    'an invention', credited == 1 and b['invented'] and len(b['invented']) == 1
    and b['missed'] == [], f'got {b}')
b = buckets([P(LEASHED, 0.95), P([80, 95, 145, 165], 0.6)], [G(DOG)])
best = max((0, 1),
           key=lambda k: rm._iou([LEASHED, [80, 95, 145, 165]][k], DOG))
check('m7b of two boxes that both contain the dog, the closer one is paired',
      b['pairs'] == [(best, 0)] and b['invented'] == [1 - best], f'got {b}')

b = buckets([P(DOG, 0.95), P(LEASHED, 0.6)], [G(DOG)])
check(
    'm8 a label the strict pass took is not paired again: the second box '
    'is an invention', b == {
        'hits': [(0, 0)],
        'pairs': [],
        'invented': [1],
        'missed': []
    }, f'got {b}')

b = buckets([P(LEASHED, cls=1)], [G(DOG, cls=0)])
check('m9 different classes never pair', b == {
    'hits': [],
    'pairs': [],
    'invented': [0],
    'missed': [0]
}, f'got {b}')


# ── m10: fuzz against the old strict algorithm ─────────────────────────────
def old_strict(preds, gt, iou_hit=0.5):
    """run_mistakes' matching as it was before the tolerant pass."""
    used, hits = set(), []
    for pi, (pc, _c, pbox) in enumerate(preds):
        best, at = 0.0, -1
        for gi, (gc, gbox) in enumerate(gt):
            if gi in used or gc != pc:
                continue
            v = rm._iou(pbox, gbox)
            if v > best:
                best, at = v, gi
        if best >= iou_hit and at >= 0:
            used.add(at)
            hits.append((pi, at))
    return hits


rng = random.Random(7)
bad_part, bad_strict, n_pairs = [], [], 0
for t in range(3000):

    def rbox():
        x, y = rng.uniform(0, 300), rng.uniform(0, 300)
        return [x, y, x + rng.uniform(4, 90), y + rng.uniform(4, 90)]

    gt = [G(rbox(), rng.choice((0, 0, 1))) for _ in range(rng.randint(0, 5))]
    preds = []
    for g in gt:  # jittered copies, some loose
        if rng.random() < 0.8:
            x1, y1, x2, y2 = g[1]
            s = rng.uniform(0, 0.6)
            w, h = x2 - x1, y2 - y1
            preds.append(
                P([
                    x1 - s * w * rng.random(), y1 - s * h * rng.random(),
                    x2 + s * w * rng.random(), y2 + s * h * rng.random()
                ], rng.random(), g[0]))
    preds += [
        P(rbox(), rng.random(), rng.choice((0, 1)))
        for _ in range(rng.randint(0, 3))
    ]
    preds.sort(key=lambda x: -x[1])
    h, p, i, m = rm.match_boxes(preds, gt)
    n_pairs += len(p)
    pi = sorted([a for a, _ in h] + [a for a, _ in p] + i)
    gi = sorted([b_ for _, b_ in h] + [b_ for _, b_ in p] + m)
    if pi != list(range(len(preds))) or gi != list(range(len(gt))):
        bad_part.append(t)
    if h != old_strict(preds, gt):
        bad_strict.append(t)
check(
    f'm10 fuzzed ({n_pairs} pairs made): every box lands in exactly one '
    'bucket', not bad_part, f'{len(bad_part)} frames broke it, e.g. '
    f'{bad_part[:3]}')
check('m10b the strict hits are the old algorithm\'s hits, box for box',
      not bad_strict, f'{len(bad_strict)} frames differ, e.g. '
      f'{bad_strict[:3]}')

tmp = tempfile.mkdtemp(prefix='tolerant-')
try:
    # ── m11 ─────────────────────────────────────────────────────────────────
    def run_dir(name, args_text):
        rd = os.path.join(tmp, 'runs', name)
        os.makedirs(os.path.join(rd, 'weights'))
        open(os.path.join(rd, 'weights', 'best.pt'), 'w').close()
        if args_text is not None:
            with open(os.path.join(rd, 'args.yaml'), 'w') as fh:
                fh.write(args_text)
        return os.path.join(rd, 'weights', 'best.pt')

    w1280 = run_dir('r1280', 'task: detect\nimgsz: 1280\nsingle_cls: true\n')
    got = {
        'imgsz 1280': rm.run_imgsz(w1280),
        'no args.yaml': rm.run_imgsz(run_dir('rnone', None)),
        'imgsz big': rm.run_imgsz(run_dir('rjunk', 'imgsz: big\n')),
        'imgsz true': rm.run_imgsz(run_dir('rbool', 'imgsz: true\n')),
        'imgsz 0': rm.run_imgsz(run_dir('rzero', 'imgsz: 0\n')),
    }
    check(
        'm11 the training imgsz comes off args.yaml; missing or junk is '
        'None', got == {
            'imgsz 1280': 1280,
            'no args.yaml': None,
            'imgsz big': None,
            'imgsz true': None,
            'imgsz 0': None
        }, f'got {got}')

    # ── a dataset, a fake detector, and fake parsers ────────────────────────
    ds = os.path.join(tmp, 'ds')
    for sub in ('images/val', 'labels/val'):
        os.makedirs(os.path.join(ds, sub))
    W, H = 400, 300
    for f in ('a.jpg', 'b.jpg'):
        Image.new('RGB', (W, H),
                  (80, 90, 70)).save(os.path.join(ds, 'images/val', f))

    def yolo(box):
        x1, y1, x2, y2 = box
        return (f'0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} '
                f'{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n')

    MISS = [300, 200, 330, 240]
    with open(os.path.join(ds, 'labels/val/a.txt'), 'w') as fh:
        fh.write(yolo(DOG) + yolo(MISS))
    with open(os.path.join(ds, 'labels/val/b.txt'), 'w') as fh:
        fh.write(yolo(DOG))
    # JSON is YAML, so this reads the same whichever parser is installed
    yml = os.path.join(ds, 'dataset.yaml')
    with open(yml, 'w') as fh:
        json.dump({
            'path': ds,
            'val': 'images/val',
            'names': {
                0: 'target'
            }
        }, fh)
    if 'yaml' not in sys.modules:
        try:
            import yaml  # noqa: F401
        except ImportError:
            sys.modules['yaml'] = types.SimpleNamespace(
                safe_load=lambda fh: json.load(fh))

    PRED = {
        'a.jpg': [(0, 0.9, LEASHED), (0, 0.4, [10, 10, 30, 30])],
        'b.jpg': [(0, 0.8, [100, 100, 140, 151])]
    }
    SEEN = {}

    class _T:

        def __init__(self, v):
            self.v = v

        def item(self):
            return self.v

        def tolist(self):
            return list(self.v)

    class _Box:

        def __init__(self, c, p, box):
            self.cls, self.conf, self.xyxy = _T(c), _T(p), [_T(box)]

    class YOLO:

        def __init__(self, w):
            self.w = w

        def predict(self, ipath, imgsz=None, **kw):
            SEEN.setdefault('imgsz', set()).add(imgsz)
            res = types.SimpleNamespace(
                boxes=[_Box(*x) for x in PRED[os.path.basename(ipath)]])
            return [res]

    sys.modules['ultralytics'] = types.SimpleNamespace(YOLO=YOLO)

    # ── m12 ─────────────────────────────────────────────────────────────────
    classes, base, n_gt, n_hit, n_tol, wrong = rm.score_detect(w1280,
                                                               yml,
                                                               imgsz=1280)
    kinds = [x['kind'] for x in wrong]
    pair = [x for x in wrong if x['kind'] == 'boxed_differently']
    check(
        'm12 one dog boxed the other way is ONE record, with both boxes',
        len(pair) == 1 and pair[0].get('box') == LEASHED
        and pair[0].get('label') == [round(v, 1)
                                     for v in DOG] and pair[0].get('p') == 0.9
        and pair[0].get('file') == os.path.join('images', 'val', 'a.jpg'),
        f'pair records: {pair}')
    check(
        'm12b counts: 3 labelled, 1 strict, 1 tolerant; inventions, then '
        'pairs, then misses', (n_gt, n_hit, n_tol) == (3, 1, 1)
        and kinds == ['invented', 'boxed_differently', 'missed'],
        f'counts {(n_gt, n_hit, n_tol)}, order {kinds}')

    # ── m13 ─────────────────────────────────────────────────────────────────
    out = os.path.join(tmp, 'mistakes')
    real_out = rm.OUT_DIR
    rm.OUT_DIR = out
    SEEN.clear()
    err = None
    try:
        doc = rm.run_one('proj/r1280', 'detect', w1280, yml, 'val', 'cpu',
                         None)
        flagged = rm.run_one('proj/flag', 'detect', w1280, yml, 'val', 'cpu',
                             640)
    except Exception as e:
        err = f'{type(e).__name__}: {e}'
    finally:
        rm.OUT_DIR = real_out
    # a scorer that raises writes nothing, and every check below reads what it
    # wrote -- so the raise is the failure, named, and the rest is skipped
    check('m13 scoring a run to disk runs to the end', err is None, err or '')
    if err:
        raise _Skip
    with open(os.path.join(out, 'proj__r1280.json')) as fh:
        disk = json.load(fh)
    check(
        'm13 both recalls and the rule are written',
        disk.get('recall') == round(1 / 3, 4)
        and disk.get('recall_tolerant') == round(2 / 3, 4)
        and disk.get('hit_tolerant') == 2 and disk.get('match') == rm.MATCH,
        f'recall {disk.get("recall")} tolerant '
        f'{disk.get("recall_tolerant")} match {disk.get("match")}')
    check(
        'm13b scored at the run\'s own imgsz unless told otherwise',
        (doc['imgsz'], doc['imgsz_from'], flagged['imgsz'],
         flagged['imgsz_from']) == (1280, 'args.yaml', 640, 'flag')
        and SEEN.get('imgsz') == {1280, 640},
        f'own {doc["imgsz"]}/{doc["imgsz_from"]}, flag '
        f'{flagged["imgsz"]}/{flagged["imgsz_from"]}, model saw '
        f'{SEEN.get("imgsz")}')
    check(
        'm13c nothing was written outside the directory it was pointed at',
        sorted(os.listdir(out)) == ['proj__flag.json', 'proj__r1280.json']
        and not os.path.exists(rm.out_path('proj/r1280')),
        f'wrote {sorted(os.listdir(out))}')

    # ── m15 ─────────────────────────────────────────────────────────────────
    with open(os.path.join(out, 'proj__r1280.json'), 'rb') as fh:
        before = fh.read()
    rm.OUT_DIR = out
    err = None
    try:
        rm.run_one('proj/r1280', 'detect', w1280, yml, 'val', 'cpu', 640)
    except Exception as e:
        err = f'{type(e).__name__}: {e}'
    finally:
        rm.OUT_DIR = real_out
    check('m15 re-scoring a run that has a file runs to the end', err is None,
          err or '')
    sup = os.path.join(out, 'superseded')
    kept = sorted(os.listdir(sup)) if os.path.isdir(sup) else []
    same = False
    if len(kept) == 1:
        with open(os.path.join(sup, kept[0]), 'rb') as fh:
            same = fh.read() == before
    with open(os.path.join(out, 'proj__r1280.json')) as fh:
        now = json.load(fh)
    check(
        'm15 a re-score keeps the file it replaces, byte for byte, under '
        'superseded/',
        len(kept) == 1 and kept[0].startswith('proj__r1280.')
        and kept[0].endswith('.json') and same and now.get('imgsz') == 640,
        f'kept {kept}, identical {same}, new imgsz {now.get("imgsz")}')
    check('m15b the first scoring of a run supersedes nothing',
          not any(k.startswith('proj__flag.') for k in kept), f'kept {kept}')

    # ── m14 ─────────────────────────────────────────────────────────────────
    import argparse
    seen = {}
    real_parse = argparse.ArgumentParser.parse_args

    def spy(self, *a, **k):
        ns = real_parse(self, *a, **k)
        seen['imgsz'] = ns.imgsz
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = spy
    try:
        rm.main(['--run', 'x'])
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    check('m14 --imgsz has no fixed default: a detector gets its own size',
          seen.get('imgsz', 'unset') is None, f'default was {seen}')

    # ── d1 ──────────────────────────────────────────────────────────────────
    st = {
        'old detect':
        d.mistakes_stale({
            'task': 'detect',
            'items': []
        }),
        'new detect':
        d.mistakes_stale({
            'task': 'detect',
            'items': [],
            'match': dict(rm.MATCH)
        }),
        'classify':
        d.mistakes_stale({
            'task': 'classify',
            'items': []
        }),
        'junk match':
        d.mistakes_stale({
            'task': 'detect',
            'match': 'yes'
        }),
    }
    check(
        'd1 a detector file without the rule is stale; with it, or a '
        'classifier file, is not', st == {
            'old detect': True,
            'new detect': False,
            'classify': False,
            'junk match': True
        }, f'got {st}')

    # ── d2, d3: the re-score, with nothing actually spawned ─────────────────
    mdir = os.path.join(tmp, 'dash_mistakes')
    os.makedirs(mdir)
    saved = {
        k: getattr(d, k)
        for k in ('MISTAKE_DIR', 'scorable', 'mistakes_python',
                  'training_root')
    }
    real_popen = d.subprocess.Popen
    spawned = []

    class FakeProc:

        def __init__(self, argv, **kw):
            self.argv, self.rc = argv, None
            spawned.append(self)

        def poll(self):
            return self.rc

    d.MISTAKE_DIR = mdir
    d.scorable = lambda r: True
    d.mistakes_python = lambda: sys.executable
    d.training_root = lambda: tmp
    d.subprocess.Popen = FakeProc
    real_log = os.path.join(REPO, 'data', 'mistakes_run.log')
    had_log = os.path.exists(real_log)
    # the spawn opens REPO/data/mistakes_run.log for the child's output; a
    # test must not leave that behind, so its open is redirected too
    real_open = open

    def open_redirect(path, *a, **k):
        if str(path) == real_log:
            path = os.path.join(tmp, 'mistakes_run.log')
        return real_open(path, *a, **k)

    d.open = open_redirect
    try:

        def put(key, doc):
            with open(os.path.join(mdir,
                                   key.replace('/', '__') + '.json'),
                      'w') as fh:
                json.dump(doc, fh)
            d._MISS.pop(key, None)

        base_doc = {
            'task': 'detect',
            'dataset': ds,
            'n': 3,
            'hit': 1,
            'wrong': 0,
            'items': [],
            'imgsz': 640
        }
        r_old = {
            'project': 'p',
            'name': 'old',
            'task': 'detect',
            'status': 'finished',
            'dir': tmp
        }
        r_new = dict(r_old, name='new')
        r_none = dict(r_old, name='none')
        put('p/old', base_doc)
        put('p/new', dict(base_doc, match=dict(rm.MATCH)))
        d._MISS_TRIED.clear()
        a = d.mistakes_topup(r_old)
        b_ = d.mistakes_topup(r_new)
        c = d.mistakes_topup(r_none)
        argvs = {
            os.path.basename(p.argv[1]) + ':' + p.argv[3]: '--force' in p.argv
            for p in spawned
        }
        check(
            'd2 a stale file is re-scored WITH --force; a missing one '
            'without; a fresh one is left alone',
            (a, b_, c) == (True, False, True) and argvs == {
                'run_mistakes.py:p/old': True,
                'run_mistakes.py:p/none': False
            }, f'spawned {(a, b_, c)}, argv --force: {argvs}')
        # a scorer still running past the retry window is not doubled
        n0 = len(spawned)
        # started three retry windows ago, and still alive
        d._MISS_TRIED['p/old'] = (time.time() - 3 * d.MISS_RETRY_S, spawned[0])
        again = d.mistakes_topup(r_old)
        check(
            'd2b a scorer still running past the retry window is never '
            'started a second time', again is False and len(spawned) == n0,
            f'topup said {again}, spawned {len(spawned) - n0} more')
        # ── d3 ─────────────────────────────────────────────────────────────
        alive = d.mistakes_pending('p/old')
        spawned[0].rc = 1
        dead = d.mistakes_pending('p/old')
        check(
            'd3 a scorer still alive is pending however long it has taken; '
            'one that failed is not', (alive, dead) == (True, False),
            f'alive {alive}, failed {dead}')
    finally:
        for k, v in saved.items():
            setattr(d, k, v)
        d.subprocess.Popen = real_popen
        if hasattr(d, 'open'):
            del d.open
    check('d2c the test wrote no log into the repo',
          os.path.exists(real_log) == had_log)

    # ── d4 ──────────────────────────────────────────────────────────────────
    big = d._frame_note(
        {
            'kind': 'boxed_differently',
            'box': LEASHED,
            'label': DOG
        }, [])
    tight = d._frame_note(
        {
            'kind': 'boxed_differently',
            'box': DOG,
            'label': LEASHED
        }, [])
    check(
        'd4 the note says which way the boxes disagree, and that it is '
        'counted once', 'takes in more than the label' in big
        and 'tighter' in tight and 'tighter' not in big
        and 'counted once' in big, f'bigger: {big!r} / tighter: {tight!r}')

    # ── d5, d6 ──────────────────────────────────────────────────────────────
    items = [
        {
            'file': 'images/val/a.jpg',
            'kind': 'invented',
            'p': 0.4,
            'cls': 'target',
            'box': [10, 10, 30, 30]
        },
        {
            'file': 'images/val/a.jpg',
            'kind': 'boxed_differently',
            'p': 0.9,
            'cls': 'target',
            'box': LEASHED,
            'label': [round(v, 1) for v in DOG],
            'iou': 0.26
        },
        {
            'file': 'images/val/a.jpg',
            'kind': 'missed',
            'p': None,
            'cls': 'target',
            'box': MISS
        },
    ]
    d.MISTAKE_DIR = mdir
    try:
        fresh = dict(base_doc,
                     items=items,
                     wrong=3,
                     hit=1,
                     hit_tolerant=2,
                     match=dict(rm.MATCH),
                     imgsz=1280)
        put('p/fresh', fresh)
        fr = d.mistake_frame('p/fresh', 1) or {}
        drawn = sorted((x['kind'], tuple(round(v) for v in x['box']))
                       for x in fr.get('boxes', []))
        want = sorted([('invented', (10, 10, 30, 30)),
                       ('boxed_differently', tuple(LEASHED)),
                       ('paired', tuple(DOG)), ('missed', tuple(MISS))])
        check(
            'd5 the viewer draws the pair\'s box and its label, and the '
            'label once -- not again as "found"', drawn == want,
            f'drew {drawn}')
        sel = [x['kind'] for x in fr.get('boxes', []) if x.get('sel')]
        check('d5b clicking the pair selects both of its boxes',
              sorted(sel) == ['boxed_differently', 'paired'], f'sel {sel}')

        real_scor = d.scorable
        d.scorable = lambda r: False  # never spawn from a render
        try:
            html = d.render_mistakes({
                'project': 'p',
                'name': 'fresh',
                'task': 'detect',
                'status': 'finished',
                'dir': tmp
            })
            put('p/stale', dict(fresh, match=None, imgsz=640))
            old = d.render_mistakes({
                'project': 'p',
                'name': 'stale',
                'task': 'detect',
                'status': 'finished',
                'dir': tmp
            })
        finally:
            d.scorable = real_scor
        check(
            'd6 the panel has a "found the dog, box differs" group with '
            'its count', 'found the dog, box differs <em>1</em>' in html
            and 'class="wrbd"' in html, 'group or caption missing')
        check(
            'd6b the panel shows BOTH recalls and the imgsz it was scored at',
            'recall 33.3% at IoU 0.5' in html
            and '66.7% counting 1 found' in html and 'imgsz 1280' in html,
            'sub-line: ' + html[html.find('wrsub'):][:260])
        check(
            'd6c a file scored under the old rule says so; a fresh one '
            'does not', 'scored under the old rule, at imgsz 640' in old
            and 'old rule' not in html, 'stale marker wrong')
    finally:
        d.MISTAKE_DIR = saved['MISTAKE_DIR']

    # ── d7 ──────────────────────────────────────────────────────────────────
    page = d.TEMPLATE
    viewer = page[page.index('var wfv=(function(){'):]
    viewer = viewer[:viewer.index('})();')]
    check(
        'd7 the viewer gives the two new kinds their own strokes',
        "boxed_differently:'bdf'" in viewer and "paired:'bdl'" in viewer
        and '.wfvbox .bdf{' in page and '.wfvbox .bdl{' in page,
        'class map or stroke rules missing')
    check('d7b the viewer\'s key explains both new strokes',
          '<i class="bdf"></i>' in viewer and '<i class="bdl"></i>' in viewer)
    check('d7c the viewer titles a pair as found, boxed differently',
          "'found, boxed differently'" in viewer)
except _Skip:
    pass
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if _fails:
    print(f'\n{len(_fails)} tolerant-match check(s) FAILED: ' +
          '; '.join(_fails))
    sys.exit(1)
print('\nall clear: a dog found but boxed the other way is counted once, '
      'and the panel says so')
