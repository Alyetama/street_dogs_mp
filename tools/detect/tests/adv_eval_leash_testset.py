#!/usr/bin/env python3
"""eval_leash.py --test-set: the frozen leash test set, scored as frozen.

Until this mode existed nothing could score a model on leash_test_v1, so a
comparison with leash_v2_001/003 needed an ad-hoc script -- and an ad-hoc
script is how a model gets scored on edited/ instead of full/, on a re-cut
file, on every crop of an image instead of the one key, or with a label read
off a folder. What is held here:

  * a key '<image_id>#<det_idx>' resolves to <image_id>_<det_idx>.jpg under
    the repo's leash_audit/full, the stored crops_from only as a fallback;
  * a missing crop, one whose sha256 is not the frozen value (in full/ OR at
    crops_from), or an items[] that no longer holds the doc's keys[] REFUSES
    the run (a set that silently shrinks or changes is not the set the last
    number was measured on), and it refuses before any model is loaded;
  * a missing corrections ledger is said, and its items are in no redraw
    slice rather than all in 'no audit redraw';
  * the label is items[].label, not a class directory;
  * the slices -- audit redraw and redraw IoU < 0.5 against the pool box,
    crop short-side bands, region from the pool's grid cell, against the
    regional prior -- are computed right on a hand-built case, and an
    'audit:gate' redraw is not read as a leash one;
  * balanced accuracy, accuracy, per-class recall and AUC (ties sharing a
    rank) match a hand computation; McNemar is exact; a model's training
    data excludes the item from EVERY model so the paired test is paired;
    the bootstrap interval is on a - b and PAIRED (a 0.04 win from 8
    discordant items is wholly above 0; resampled unpaired it crosses 0);
    one class left (BA None) still prints;
  * through main() with a stand-in 'ultralytics': each checkpoint predicts
    at ITS train_args imgsz (640, not the gate-224 constant), on the crops
    as they are, with its own train_args.data in the training-data guard,
    and the JSON report survives a print that fails.

Synthetic crops and ledgers in a temp directory; no checkpoint is loaded.
The MUTANTS put an old or plausible wrong behaviour back -- by patching a
name, or for inline code by editing eval_leash's source in memory -- and
require the check to FAIL.

Run: python tools/detect/tests/adv_eval_leash_testset.py   (mp14 env)
"""
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT = os.path.dirname(HERE)
sys.path.insert(0, DETECT)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import crop_sources  # noqa: E402
import eval_leash as E  # noqa: E402

TMP = tempfile.mkdtemp(prefix='adv_eval_leash_')

# key, label, crop (w, h), cell, pool box, redraw box (None = no redraw)
CASES = [
    ('1000001#0', 'leashed', (64, 100), 'Europe_0_45_5_50',
     (100, 100, 200, 200), None),
    # redrawn, IoU 0.25 with the pool box: the leash is outside the crop
    ('1000002#3', 'leashed', (65, 90), 'South_Asia_80_25_85_30',
     (100, 100, 200, 200), (100, 100, 300, 300)),
    # redrawn, IoU 0.81: same dog, a nudge
    ('1000003#1', 'unleashed', (120, 300), 'Europe_0_45_5_50',
     (0, 0, 100, 100), (0, 0, 90, 90)),
    ('1000004#0', 'unleashed', (200, 193), 'Central_America_and_Caribbean'
     '_-80_15_-75_20', (10, 10, 50, 50), None),
    ('1000005#2', 'leashed', (40, 30), 'North_America_-75_40_-70_45',
     (10, 10, 50, 50), None),
    # in no pool row: no region, no redraw IoU though it has a redraw
    ('1000006#0', 'unleashed', (128, 128), None, None, (1, 1, 9, 9)),
]
# an 'audit:gate' redraw of a leash key -- must NOT count as a leash redraw
GATE_REDRAW = ('1000001', 0, (0, 0, 5, 5))


def _sha(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def make_world(name):
    """(doc_path, full_dir, alt_dir, pool_path, corrections_path)."""
    from PIL import Image
    import pyarrow as pa
    import pyarrow.parquet as pq
    # a fresh world per call: a mutant re-runs a check, and a check that
    # died on 'directory exists' would count as a kill it never made
    root = tempfile.mkdtemp(prefix=name + '_', dir=TMP)
    full = os.path.join(root, 'data', 'leash_audit', 'full')
    alt = os.path.join(root, 'old_place', 'full')
    os.makedirs(full)
    os.makedirs(alt)
    items, pool, corr = [], [], []
    for n, (key, label, (w, h), cell, box, redraw) in enumerate(CASES):
        # the last item exists only at the stored crops_from
        d = alt if n == len(CASES) - 1 else full
        p = os.path.join(d, key.replace('#', '_') + '.jpg')
        Image.new('RGB', (w, h), (n * 40, 90, 200 - n * 30)).save(p)
        items.append({'key': key, 'label': label, 'sha256': _sha(p),
                      # the legacy score: deliberately the WRONG way round,
                      # so a scorer that read it would be visibly wrong
                      'p_model': 0.1 if label == 'leashed' else 0.9,
                      'band': n, 'seq': 'seq%d' % n})
        iid, det = key.split('#')
        if cell:
            pool.append({'image_id': iid, 'det_idx': int(det),
                         'x1': float(box[0]), 'y1': float(box[1]),
                         'x2': float(box[2]), 'y2': float(box[3]),
                         'cell': cell})
        if redraw:
            corr.append({'crop': 'audit:leash', 'image_id': iid,
                         'det_idx': int(det), 'x1': redraw[0],
                         'y1': redraw[1], 'x2': redraw[2], 'y2': redraw[3]})
    iid, det, b = GATE_REDRAW
    corr.append({'crop': 'audit:gate', 'image_id': iid, 'det_idx': det,
                 'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3]})
    doc = {'family': 'leash', 'stage': 'leash', 'created': '2026-11-14',
           'crops_from': alt, 'items': items,
           'keys': [it['key'] for it in items]}
    doc_path = os.path.join(root, 'data', 'leash_test_v1.json')
    with open(doc_path, 'w') as fh:
        json.dump(doc, fh)
    pool_path = os.path.join(root, 'pool.parquet')
    pq.write_table(pa.Table.from_pylist(pool), pool_path)
    corr_path = os.path.join(root, 'boxes.jsonl')
    with open(corr_path, 'w') as fh:
        for r in corr:
            fh.write(json.dumps(r) + '\n')
    return doc_path, full, alt, pool_path, corr_path


def _refused(fn):
    """The SystemExit text, or None if fn() did not refuse."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except SystemExit as e:
        return str(e)
    return None


def check_resolves_keys():
    bad = []
    doc_path, full, alt, _, _ = make_world('resolve')
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, items = E.load_test_set(doc_path, full)
    except SystemExit as e:
        return ['resolve: a good set was refused: %s' % str(e)[:200]]
    got = {it['key']: it for it in items}
    if sorted(got) != sorted(c[0] for c in CASES):
        return ['resolve: scored keys %s, frozen %s'
                % (sorted(got), sorted(c[0] for c in CASES))]
    for key, label, *_ in CASES:
        it = got[key]
        want = key.replace('#', '_') + '.jpg'
        if os.path.basename(it['path']) != want:
            bad.append('resolve: %s -> %s, want %s' % (key, it['path'], want))
        if it['label'] != label:
            bad.append('resolve: %s labelled %s, items[] says %s'
                       % (key, it['label'], label))
        if (it['image_id'], it['det_idx']) != (key.split('#')[0],
                                               int(key.split('#')[1])):
            bad.append('resolve: %s parsed as %s#%s'
                       % (key, it['image_id'], it['det_idx']))
    last = got[CASES[-1][0]]
    if not last['fallback'] or os.path.dirname(last['path']) != alt:
        bad.append('resolve: a crop only at crops_from was not found there')
    if any(it['fallback'] for k, it in got.items() if k != CASES[-1][0]):
        bad.append('resolve: a repo-relative crop was read from crops_from')
    return bad


def check_refuses_sha_mismatch():
    """Both places a crop can come from: the repo's full/ and the stored
    crops_from fallback -- a fallback crop is no less the frozen bytes."""
    bad = []
    doc_path, full, alt, _, _ = make_world('sha')
    E.load_test_set(doc_path, full)
    for key, d in ((CASES[0][0], full), (CASES[-1][0], alt)):
        with open(os.path.join(d, key.replace('#', '_') + '.jpg'),
                  'ab') as fh:
            fh.write(b'\0')
    msg = _refused(lambda: E.load_test_set(doc_path, full))
    for key, where in ((CASES[0][0], 'full/'), (CASES[-1][0], 'crops_from')):
        if not msg or 'sha256' not in msg or key not in msg:
            bad.append('sha: a crop in %s changed since freezing was scored '
                       '(%r)' % (where, msg))
    return bad


def check_refuses_missing():
    doc_path, full, _, _, _ = make_world('missing')
    os.remove(os.path.join(full, CASES[1][0].replace('#', '_') + '.jpg'))
    msg = _refused(lambda: E.load_test_set(doc_path, full))
    if not msg or 'missing' not in msg or CASES[1][0] not in msg:
        return ['missing: a vanished crop shrank the set silently (%r)'
                % msg]
    return []


def check_refuses_bad_doc():
    bad = []
    doc_path, full, _, _, _ = make_world('baddoc')
    with open(doc_path) as fh:
        doc = json.load(fh)
    for what, mutate in (
            ('a dogbin doc', lambda d: d.update(family='dogbin',
                                                stage='dogbin')),
            ('a label that is neither class',
             lambda d: d['items'][0].update(label='dog')),
            ('a duplicate key',
             lambda d: d['items'].append(dict(d['items'][0]))),
            ('no frozen sha256', lambda d: d['items'][0].pop('sha256')),
            # every remaining sha still matches: only keys[] can tell
            ('an items[] entry cut, keys[] intact',
             lambda d: d['items'].pop(2))):
        d = json.loads(json.dumps(doc))
        mutate(d)
        p = doc_path + '.bad.json'
        with open(p, 'w') as fh:
            json.dump(d, fh)
        if not _refused(lambda: E.load_test_set(p, full)):
            bad.append('bad doc: %s was accepted' % what)
    return bad


def _annotated(name):
    doc_path, full, _, pool_path, corr_path = make_world(name)
    doc, items = E.load_test_set(doc_path, full)
    with contextlib.redirect_stdout(io.StringIO()):
        unknown = E.annotate_items(items, pool_path, corr_path)
    return doc, items, unknown


def check_slices():
    bad = []
    _, items, unknown = _annotated('slices')
    if unknown != {'region': 1, 'redrawn': 0, 'redraw_iou': 1}:
        bad.append('slices: unknown counts %s, want region 1, redrawn 0, '
                   'redraw_iou 1' % unknown)
    by = {it['key']: it for it in items}
    want_iou = {'1000002#3': 0.25, '1000003#1': 0.81}
    for k, v in want_iou.items():
        if by[k]['redraw_iou'] != v:
            bad.append('slices: %s redraw IoU %s, want %s'
                       % (k, by[k]['redraw_iou'], v))
    if by['1000001#0']['redrawn']:
        bad.append('slices: an audit:gate redraw was read as a leash one')
    got = {(g, n): sorted(items[i]['key'] for i in idx)
           for g, n, idx in E.slice_index(items)}
    low = [k for (g, k) in got if g == 'redraw' and k.startswith('redraw IoU')]
    want = {
        ('all', 'all'): sorted(c[0] for c in CASES),
        ('redraw', 'audit redraw'): ['1000002#3', '1000003#1', '1000006#0'],
        ('redraw', 'no audit redraw'): ['1000001#0', '1000004#0',
                                        '1000005#2'],
        ('redraw', low[0] if low else '?'): ['1000002#3'],
        # short side 64 is <=64; 65 is the next band; 128 closes 96-128
        ('short side', '<=64'): ['1000001#0', '1000005#2'],
        ('short side', '64-96'): ['1000002#3'],
        ('short side', '96-128'): ['1000003#1', '1000006#0'],
        ('short side', '128-192'): [],
        ('short side', '>192'): ['1000004#0'],
        # leashed in South Asia, unleashed in Europe
        ('prior', 'against regional prior'): ['1000002#3', '1000003#1'],
        ('prior', 'with regional prior'): ['1000001#0', '1000004#0',
                                           '1000005#2'],
        ('region', 'Europe'): ['1000001#0', '1000003#1'],
        ('region', 'South_Asia'): ['1000002#3'],
        ('region', 'Central_America_and_Caribbean'): ['1000004#0'],
        ('region', 'North_America'): ['1000005#2'],
    }
    for k, v in want.items():
        if got.get(k) != v:
            bad.append('slices: %s = %s, want %s' % (k, got.get(k), v))
    extra = set(got) - set(want)
    if extra:
        bad.append('slices: unexpected %s' % sorted(extra))
    return bad


def check_region_of_cell():
    want = {'Africa_-5_15_0_20': 'Africa',
            'Central_America_and_Caribbean_-80_15_-75_20':
                'Central_America_and_Caribbean',
            'Europe_2.5_45_5_47.5': 'Europe',
            'Europe': None, None: None, 'x_1_2_3': None}
    return ['region_of_cell(%r) = %r, want %r' % (c, E.region_of_cell(c), r)
            for c, r in want.items() if E.region_of_cell(c) != r]


def check_metrics_hand():
    bad = []
    # 3 leashed, 2 unleashed; 0.5 exactly counts as leashed (gate_store's >=)
    y = [1, 1, 1, 0, 0]
    p = [0.9, 0.5, 0.2, 0.4, 0.7]
    m = E.metrics(y, p)
    # tp 2 (0.9, 0.5), tn 1 (0.4): recall 2/3 and 1/2, BA 7/12, acc 3/5
    # AUC: (0.9 beats both) 2 + (0.5 beats 0.4) 1 + (0.2) 0 = 3 of 6
    want = {'balanced_accuracy': round(7 / 12, 4), 'accuracy': 0.6,
            'recall_leashed': round(2 / 3, 4), 'recall_unleashed': 0.5,
            'auc': 0.5, 'n': 5, 'n_leashed': 3, 'n_unleashed': 2,
            'confusion': {'tp': 2, 'fn': 1, 'tn': 1, 'fp': 1}}
    for k, v in want.items():
        if m[k] != v:
            bad.append('metrics: %s = %s, want %s' % (k, m[k], v))
    # tied scores share their rank: one pair, one tie -> 0.5, not 0 or 1
    for yy, pp, w in (([1, 0], [0.3, 0.3], 0.5),
                      ([1, 1, 0], [0.8, 0.3, 0.3], 0.75)):
        if E.rank_auc(yy, pp) != w:
            bad.append('auc: ties %s %s -> %s, want %s'
                       % (yy, pp, E.rank_auc(yy, pp), w))
    one = E.metrics([1, 1], [0.9, 0.1])
    if one['balanced_accuracy'] is not None or one['auc'] is not None \
            or one['recall_leashed'] != 0.5 or one['recall_unleashed']:
        bad.append('metrics: one class should give recall only, got %s'
                   % one)
    mc = E.mcnemar_exact([1] * 5 + [0, 1], [0] * 5 + [0, 1])
    if mc != {'a_only': 5, 'b_only': 0, 'p': 0.0625}:
        bad.append('mcnemar: 5 vs 0 discordant -> %s, want p 0.0625' % mc)
    if E.mcnemar_exact([1, 0], [0, 1])['p'] != 1.0:
        bad.append('mcnemar: 1 vs 1 discordant should be p 1')
    ci = E.bootstrap_ba_diff([1, 1, 0, 0], [0.9] * 2 + [0.1] * 2,
                             [0.9] * 2 + [0.1] * 2)
    if ci != [0.0, 0.0]:
        bad.append('bootstrap: identical models -> %s, want [0, 0]' % ci)
    return bad


def check_run_test_set():
    """Two stand-in models, one of which trained on an image in the set."""
    bad = []
    doc, items, _ = _annotated('run')
    # A always says leashed for leashed crops; B always says leashed
    truth = {it['path']: it['label'] == 'leashed' for it in items}

    def predict(m, paths):
        if m['name'] == 'A':
            return [0.9 if truth[p] else 0.1 for p in paths]
        return [0.8 for _ in paths]
    data = tempfile.mkdtemp(prefix='run_dataset_', dir=TMP)
    os.makedirs(os.path.join(data, 'train', 'leashed'))
    open(os.path.join(data, 'train', 'leashed', 'la_1000004_d0.jpg'),
         'wb').close()
    models = [{'name': 'A', 'weights': 'a.pt', 'imgsz': 640,
               'data_roots': []},
              {'name': 'B', 'weights': 'b.pt', 'imgsz': 640,
               'data_roots': [data]}]
    rep = E.run_test_set(doc, items, models, predict)
    if rep['test_set']['n_scored'] != len(CASES) - 1:
        bad.append('run: %s scored, want %d (one trained on)'
                   % (rep['test_set']['n_scored'], len(CASES) - 1))
    if any(i['key'] == '1000004#0' for i in rep['items']):
        bad.append('run: an item in B\'s training data was scored for A')
    if rep['models']['B']['excluded_trained_on'] != ['1000004_0.jpg']:
        bad.append('run: B exclusions %s'
                   % rep['models']['B']['excluded_trained_on'])
    ha, hb = rep['models']['A']['headline'], rep['models']['B']['headline']
    if (ha['balanced_accuracy'], ha['auc']) != (1.0, 1.0):
        bad.append('run: perfect model A got %s' % ha)
    if (hb['balanced_accuracy'], hb['recall_leashed'],
            hb['recall_unleashed']) != (0.5, 1.0, 0.0):
        bad.append('run: always-leashed model B got %s' % hb)
    pr = rep['paired'][0]
    # 5 items, 3 leashed: B is wrong exactly on the 2 unleashed
    if (pr['mcnemar_exact']['a_only'], pr['mcnemar_exact']['b_only'],
            pr['ba_diff']) != (2, 0, 0.5):
        bad.append('run: paired %s' % pr)
    sl = rep['models']['A']['slices']
    if sl['prior']['against regional prior']['n'] != 2:
        bad.append('run: against-prior n %s'
                   % sl['prior']['against regional prior']['n'])
    if 'p_model' in json.dumps(rep['items']):
        bad.append('run: the legacy p_model leaked into the report')
    return bad


def check_cli_refuses_before_model():
    """A changed crop stops the CLI before ultralytics is imported -- in this
    env there is none, so reaching the model load would be an ImportError."""
    doc_path, full, _, pool_path, corr_path = make_world('cli')
    p = os.path.join(full, CASES[0][0].replace('#', '_') + '.jpg')
    with open(p, 'ab') as fh:
        fh.write(b'\0')
    saved = E.AUDIT_FULL, E.AUDIT_POOL, E.CORRECTIONS
    E.AUDIT_FULL, E.AUDIT_POOL, E.CORRECTIONS = full, pool_path, corr_path
    try:
        msg = _refused(lambda: E.main(['--weights', 'none.pt',
                                       '--test-set', doc_path,
                                       '--device', 'cpu']))
    except ImportError as e:
        return ['cli: loaded a model before verifying the set (%s)' % e]
    finally:
        E.AUDIT_FULL, E.AUDIT_POOL, E.CORRECTIONS = saved
    if not msg or 'sha256' not in msg:
        return ['cli: --test-set with a changed crop did not refuse (%r)'
                % msg]
    return []


def check_missing_ledger():
    """boxes.jsonl is gitignored, so a fresh clone has none. read_audit_
    corrections then gives {} without a word, and every item would read as
    'no audit redraw'. It must be said, and the items left out of both
    redraw slices rather than put in the wrong one."""
    bad = []
    doc_path, full, _, pool_path, _ = make_world('noledger')
    _, items = E.load_test_set(doc_path, full)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        unknown = E.annotate_items(items, pool_path,
                                   os.path.join(TMP, 'no_such_ledger.jsonl'))
    if 'WARNING' not in buf.getvalue() or 'ledger' not in buf.getvalue():
        bad.append('ledger: a missing corrections ledger went unsaid (%r)'
                   % buf.getvalue())
    if unknown.get('redrawn') != len(CASES):
        bad.append('ledger: unknown %s, want redrawn %d' % (unknown,
                                                             len(CASES)))
    got = {n: idx for g, n, idx in E.slice_index(items) if g == 'redraw'}
    for name, idx in got.items():
        if idx:
            bad.append('ledger: with no ledger, %d item(s) in %r'
                       % (len(idx), name))
    return bad


def _one_class_report():
    doc, items, _ = _annotated('oneclass')
    keep = [it for it in items if it['label'] == 'leashed']
    models = [{'name': n, 'weights': n + '.pt', 'imgsz': 640,
               'data_roots': []} for n in ('A', 'B')]
    return E.run_test_set(doc, keep, models,
                          lambda m, paths: [0.9] * len(paths))


def check_print_one_class():
    """Trained-on exclusions or a subset can leave one class: BA, and so the
    BA difference, is then None, and the report must still print."""
    rep = _one_class_report()
    if rep['paired'][0]['ba_diff'] is not None:
        return ['one class: ba_diff %s, want None' % rep['paired'][0]]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            E.print_test_report(rep)
    except Exception as e:                   # noqa: BLE001
        return ['one class: the report died printing (%s: %s)'
                % (type(e).__name__, e)]
    if 'diff -' not in buf.getvalue():
        return ['one class: no "diff -" in the paired line']
    return []


def _ba_case(n_flip):
    """100 leashed + 100 unleashed; model a right on 70 of each, model b = a
    with n_flip more leashed wrong. BA 0.70 vs 0.70 - n_flip/200."""
    y = [1] * 100 + [0] * 100
    pa = ([0.9] * 70 + [0.1] * 30) + ([0.1] * 70 + [0.9] * 30)
    pb = list(pa)
    for i in range(n_flip):
        pb[i] = 0.1
    return y, pa, pb


def check_bootstrap():
    """The interval is on BA(a) - BA(b), and it is PAIRED. a = b + 8 errors
    on otherwise identical items: the paired interval sits wholly above 0
    (each resample's difference comes only from the 8 discordant items):
    measured [+0.015, +0.070]. Resampling the two models independently
    gives [-0.050, +0.127], 0.18 wide and across 0 -- the 0.04 gap would
    read as noise."""
    bad = []
    y, pa, pb = _ba_case(8)
    diff = E._ba(y, pa, range(200)) - E._ba(y, pb, range(200))   # 0.04
    ci = E.bootstrap_ba_diff(y, pa, pb)
    if not ci or not (ci[0] <= diff <= ci[1]):
        bad.append('bootstrap: CI %s does not bracket the difference %.4f'
                   % (ci, diff))
    elif not ci[0] > 0:
        bad.append('bootstrap: a 0.04 paired win has CI %s, want lo > 0 '
                   '(sign flipped, or not paired)' % ci)
    elif ci[1] - ci[0] > 0.1:
        bad.append('bootstrap: CI %s is %.3f wide for 8 discordant items '
                   '-- not paired' % (ci, ci[1] - ci[0]))
    rev = E.bootstrap_ba_diff(y, pb, pa)
    if not rev or not rev[1] < 0:
        bad.append('bootstrap: b - a for a 0.04 loss has CI %s, want hi < 0'
                   % rev)
    return bad


class _Probs:
    def __init__(self, p):
        self.data = [p, 1 - p]


class _Result:
    def __init__(self, p):
        self.probs = _Probs(p)


def _fake_ultralytics(trained, seen):
    """A stand-in 'ultralytics' with the parts eval_leash touches: names,
    ckpt.train_args (imgsz 640, data), predict() on PIL images. It records
    every call, and scores by crop size, which is unique per CASES item."""
    import types
    from PIL import Image
    leashed = {(w, h) for _, label, (w, h), *_ in CASES if label == 'leashed'}

    class YOLO:
        def __init__(self, w):
            self.w = w
            self.names = {0: 'leashed', 1: 'unleashed'}
            self.ckpt = {'train_args': {'imgsz': 640, 'data': trained[w]}}

        def predict(self, ims, imgsz=None, verbose=None, device=None):
            seen.append((self.w, imgsz, [im.size if isinstance(
                im, Image.Image) else type(im).__name__ for im in ims]))
            return [_Result(0.9 if (im.size in leashed or 'b.pt' in self.w)
                            else 0.1) for im in ims]
    mod = types.ModuleType('ultralytics')
    mod.YOLO = YOLO
    return mod


def check_cli_model_path():
    """The half of --test-set that loads checkpoints, end to end through
    main(): imgsz from the checkpoint's own train_args (the gate-224 bug was
    a constant here), the checkpoint's own train_args.data added to the
    training-data guard, crops handed over at their own size, and the JSON
    report written even if printing it fails."""
    bad = []
    doc_path, full, _, pool_path, corr_path = make_world('climodel')
    run = tempfile.mkdtemp(prefix='runs_', dir=TMP)
    wa = os.path.join(run, 'a', 'weights', 'a.pt')
    wb = os.path.join(run, 'b', 'weights', 'b.pt')
    empty = tempfile.mkdtemp(prefix='empty_', dir=TMP)
    own_b = tempfile.mkdtemp(prefix='own_b_', dir=TMP)
    os.makedirs(os.path.join(own_b, 'train', 'leashed'))
    open(os.path.join(own_b, 'train', 'leashed', 'la_1000004_d0.jpg'),
         'wb').close()
    seen = []
    out = os.path.join(run, 'report.json')
    saved = (E.AUDIT_FULL, E.AUDIT_POOL, E.CORRECTIONS,
             sys.modules.get('ultralytics'), E.print_test_report)
    E.AUDIT_FULL, E.AUDIT_POOL, E.CORRECTIONS = full, pool_path, corr_path
    sys.modules['ultralytics'] = _fake_ultralytics({wa: empty, wb: own_b},
                                                    seen)

    def dies(rep):
        raise RuntimeError('printing failed')
    E.print_test_report = dies
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            E.main(['--test-set', doc_path, '--weights', wa, '--compare', wb,
                    '--data', empty, '--out', out, '--device', 'cpu',
                    '--batch', '2'])
    except RuntimeError:
        pass
    finally:
        E.AUDIT_FULL, E.AUDIT_POOL, E.CORRECTIONS = saved[:3]
        if saved[3] is None:
            sys.modules.pop('ultralytics', None)
        else:
            sys.modules['ultralytics'] = saved[3]
        E.print_test_report = saved[4]
    sizes = {CASES[i][2] for i in range(len(CASES))} - {(200, 193)}
    if not seen:
        return ['cli model: predict() was never called']
    for w, imgsz, got in seen:
        if imgsz != 640:
            bad.append('cli model: %s predicted at %s, checkpoint says 640'
                       % (os.path.basename(w), imgsz))
        for g in got:
            if g not in sizes:
                bad.append('cli model: %s got %s, not a test crop as it is'
                           % (os.path.basename(w), g))
    for w in (wa, wb):
        got = sorted(g for ww, _, gs in seen if ww == w for g in gs)
        if got != sorted(sizes):
            bad.append('cli model: %s scored %s, want the %d crops not in '
                       "b's own training data" % (os.path.basename(w), got,
                                                  len(sizes)))
    if not os.path.isfile(out):
        return bad + ['cli model: no JSON report after the print failed']
    with open(out) as fh:
        rep = json.load(fh)
    if rep['test_set']['n_scored'] != len(CASES) - 1 or any(
            i['key'] == '1000004#0' for i in rep['items']):
        bad.append("cli model: an item in b's own train_args.data was scored "
                   '(n_scored %s)' % rep['test_set']['n_scored'])
    if any(v['imgsz'] != 640 for v in rep['models'].values()):
        bad.append('cli model: report imgsz %s'
                   % {k: v['imgsz'] for k, v in rep['models'].items()})
    return bad


def check_cli_out_needs_test_set():
    """--out without --test-set is refused like --compare, not ignored."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            E.main(['--weights', 'none.pt', '--out', 'x.json'])
    except SystemExit as e:
        if e.code == 2 and '--out' in err.getvalue():
            return []
        return ['cli: --out alone exited %r (%r)' % (e.code, err.getvalue())]
    except ImportError:
        return ['cli: --out alone went on to load a model']
    return ['cli: --out alone was accepted']


# ── mutants ────────────────────────────────────────────────────────────────


class AnchorGone(Exception):
    """A source mutant's text is no longer in eval_leash.py exactly once: the
    mutant would test nothing, so it is a FAIL, never a kill."""


_SRC_PATH = E.__file__
with open(_SRC_PATH) as _fh:
    _SRC = _fh.read()


def _src_mutant(old, new, check):
    """Run check against eval_leash with `old` replaced by `new` in its
    source. For behaviour that lives inline in a function (an imgsz lookup,
    a guard's append, a sort order) and cannot be swapped by patching a
    name."""
    def run():
        global E
        if _SRC.count(old) != 1:
            raise AnchorGone(old)
        import types
        mod = types.ModuleType('eval_leash')
        mod.__file__ = _SRC_PATH
        exec(compile(_SRC.replace(old, new), _SRC_PATH, 'exec'), mod.__dict__)
        real, E = E, mod
        try:
            return check()
        finally:
            E = real
    return run

def _patched(obj, name, value, fn):
    real = getattr(obj, name)
    setattr(obj, name, value)
    try:
        return fn()
    finally:
        setattr(obj, name, real)


_HASH_CACHE = {}
_REAL_SHA = E._sha256


def _cached_sha(path):
    # a hash remembered from the first look -- the bytes are never re-read
    if path not in _HASH_CACHE:
        _HASH_CACHE[path] = _REAL_SHA(path)
    return _HASH_CACHE[path]


def _skipping_loader(path, full_dir=None):
    # the readable()-style loader: score what is there, drop the rest
    with open(path) as fh:
        doc = json.load(fh)
    items = []
    for it in doc['items']:
        p = os.path.join(full_dir, E.key_file(it['key']))
        if os.path.isfile(p):
            iid, det = it['key'].split('#')
            items.append({'key': it['key'], 'image_id': iid,
                          'det_idx': int(det), 'label': it['label'],
                          'path': p, 'fallback': False})
    return doc, items


def _any_redraw(stage, path=None):
    # every record in the ledger, whichever page drew it
    out = {}
    for ln in open(path):
        r = json.loads(ln)
        out[(str(r['image_id']), int(r.get('det_idx') or 0))] = (
            r['x1'], r['y1'], r['x2'], r['y2'])
    return out


def _band_exclusive(s):
    for hi, name in E.SHORT_BANDS:
        if s < hi:
            return name
    return E.SHORT_BANDS[-1][1]


def _ordinal_auc(y, s):
    order = sorted(range(len(s)), key=lambda i: s[i])
    rank = {i: r + 1 for r, i in enumerate(order)}
    n1 = sum(y)
    n0 = len(y) - n1
    if not n1 or not n0:
        return None
    return (sum(rank[i] for i, t in enumerate(y) if t)
            - n1 * (n1 + 1) / 2) / (n1 * n0)


MUTANTS = [
    ("keys looked up with the '#' kept",
     lambda: _patched(E, 'key_file', lambda k: k + '.jpg',
                      check_resolves_keys)),
    ('the sha256 remembered instead of re-read',
     lambda: _patched(E, '_sha256', _cached_sha, check_refuses_sha_mismatch)),
    ('missing crops skipped, not refused',
     lambda: _patched(E, 'load_test_set', _skipping_loader,
                      check_refuses_missing)),
    ("an 'audit:gate' redraw read as a leash redraw",
     lambda: _patched(crop_sources, 'read_audit_corrections', _any_redraw,
                      check_slices)),
    ('short-side bands with an exclusive upper bound',
     lambda: _patched(E, 'short_band', _band_exclusive, check_slices)),
    ('against-prior by region alone, label ignored',
     lambda: _patched(E, 'against_prior',
                      lambda it: it.get('region') in E.PRIOR_UNLEASHED
                      or None, check_slices)),
    ('AUC with ordinal ranks for ties',
     lambda: _patched(E, 'rank_auc', _ordinal_auc, check_metrics_hand)),
    ('the training-data guard a no-op',
     lambda: _patched(E, 'drop_trained_on', lambda paths, root: (paths, []),
                      check_run_test_set)),
    # inline behaviour, put back in the source itself
    ('a crop found via crops_from not sha-checked',
     _src_mutant('        if got != want:',
                 '        if got != want and not fallback:',
                 check_refuses_sha_mismatch)),
    ('keys[] not compared with items[]',
     _src_mutant("    frozen = doc.get('keys')", '    frozen = None',
                 check_refuses_bad_doc)),
    ('a missing ledger read as no redraws, unsaid',
     _src_mutant('    if os.path.isfile(corrections_path):', '    if True:',
                 check_missing_ledger)),
    ("'no audit redraw' taking items whose redraw is unknown",
     _src_mutant("if it['redrawn'] is False]", "if not it['redrawn']]",
                 check_missing_ledger)),
    ('the BA difference formatted without a None guard',
     _src_mutant("d_s = '-' if d is None else f'{d:+.4f}'",
                 "d_s = f'{d:+.4f}'", check_print_one_class)),
    ('the bootstrap difference sign-flipped',
     _src_mutant('diffs.append(a - b)', 'diffs.append(b - a)',
                 check_bootstrap)),
    ('the bootstrap resampling each model independently',
     _src_mutant('a, b = _ba(y, pa, idx), _ba(y, pb, idx)',
                 'a, b = _ba(y, pa, idx), _ba(y, pb, [rng.randrange(n) '
                 'for _ in range(n)])', check_bootstrap)),
    ('the bootstrap interval a constant',
     _src_mutant('    return [round(lo, 4), round(hi, 4)]',
                 '    return [0.0, 0.0]', check_bootstrap)),
    ('the gate-224 bug: a constant imgsz, not the checkpoint\'s',
     _src_mutant('imgsz = args.imgsz or model_imgsz(model)',
                 'imgsz = args.imgsz or 224', check_cli_model_path)),
    ("the checkpoint's own train_args.data not guarded",
     _src_mutant('        roots.append(own)', '        pass',
                 check_cli_model_path)),
    ('crops resized before predict()',
     _src_mutant("ims.append(im.convert('RGB'))",
                 "ims.append(im.convert('RGB').resize((224, 224)))",
                 check_cli_model_path)),
    ('the report printed before the JSON is written',
     _src_mutant('    out = args.out or args.json or os.path.join(',
                 '    print_test_report(rep)\n'
                 '    out = args.out or args.json or os.path.join(',
                 check_cli_model_path)),
    ('--out without --test-set ignored',
     _src_mutant('    if args.out:\n', '    if False:\n',
                 check_cli_out_needs_test_set)),
]


def main():
    checks = [check_resolves_keys, check_refuses_sha_mismatch,
              check_refuses_missing, check_refuses_bad_doc, check_slices,
              check_region_of_cell, check_metrics_hand, check_run_test_set,
              check_cli_refuses_before_model, check_missing_ledger,
              check_print_one_class, check_bootstrap, check_cli_model_path,
              check_cli_out_needs_test_set]
    bad = []
    try:
        for fn in checks:
            try:
                got = fn()
            except (Exception, SystemExit) as e:   # noqa: BLE001
                got = ['threw %s: %s' % (type(e).__name__, e)]
            for g in got:
                bad.append('%s: %s' % (fn.__name__, g))
            print(('FAIL ' if got else 'ok   ') + fn.__name__)
        for name, fn in MUTANTS:
            _HASH_CACHE.clear()
            try:
                got = fn()
            except AnchorGone as e:
                bad.append('MUTANT %s: its source text is gone from '
                           'eval_leash.py (%r) -- update the mutant' % (name,
                                                                        str(e)))
                print('FAIL mutant: ' + name)
                continue
            except (Exception, SystemExit) as e:   # noqa: BLE001
                got = ['threw %s' % (type(e).__name__,)]
            if not got:
                bad.append('MUTANT %s: the check still passes with the old '
                           'behaviour put back -- it proves nothing' % name)
            print(('ok   ' if got else 'FAIL ') + 'mutant: ' + name)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    for b in bad:
        print('FAIL ' + b)
    if bad:
        return 1
    print('leash test set: keys resolved repo-relative, frozen bytes '
          'enforced, labels from items[], slices and paired tests right '
          '(%d checks, %d mutants)' % (len(checks), len(MUTANTS)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
