#!/usr/bin/env python3
"""A crop build's bundle keeps what its stage alone held, and says how well
crop size by itself predicts the label.

Both were measured on leash_20261115_18b269 first:

  B1  the build collected 5,410 rows and kept 3,744. The other 1,666
      (1,005 reserved, 508 over the sequence cap, 31 with no sequence, 122
      de-duplicated) were listed only in the stage's crops.jsonl, and the
      embedding clusters only in its clusters.json. The build deletes the
      stage, so the bundle held counts and a manifest_sha256 of a file that
      no longer existed. rebuild_manifest's 'manifest' and
      'dup_clusters_file' both named the deleted stage, and so did every
      row's 'path'.
  B2  crop area alone ranked leashed above unleashed at AUC 0.60 train /
      0.65 val, and leashed rate ran from 0.17 at <=64 px to about 0.60
      above 192 px. leash_test_v1, cut the way production cuts, is flat
      (0.51). Val carried the same skew, so nothing in the build said so.
      The gate is the other way: dogbin_test_v1 itself has AUC 0.63, more
      than its build (0.60), so a gate build must not be told the test set
      lacks the shortcut.

Synthetic stage and dataset directories in a temp directory; the builders
are stubbed. Each check returns a list of failures and prints 'ok   name' or
'FAIL name'. The MUTANTS put the old behaviour back and require the check to
FAIL: a check that passes either way proves nothing.

Run: python tools/detect/tests/adv_bundle_record.py
"""
import contextlib
import hashlib
import inspect
import io
import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DETECT = os.path.join(REPO, 'tools', 'detect')
sys.path.insert(0, DETECT)

import build_dataset as bd                # noqa: E402

TMP = tempfile.mkdtemp(prefix='adv_bundle_record_')


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        h.update(fh.read())
    return h.hexdigest()


def _row(uid, split, label, wh, source='leash_audit', stage=''):
    return {'uid': uid, 'split': split, 'label': label, 'crop_wh': list(wh),
            'source': source, 'family': 'leash', 'image_id': uid.split('_')[1],
            'path': os.path.join(stage, 'crops', label, uid)}


def _stage_and_out(where):
    """A stage of 5 collected rows and a dataset that kept 3 of them."""
    out = os.path.join(where, 'leash_x')
    stage = out + '.stage'
    collected = [_row('la_%d_d0.jpg' % i, None,
                      'leashed' if i % 2 else 'unleashed',
                      (40 + 50 * i, 60 + 50 * i), stage=stage)
                 for i in range(5)]
    os.makedirs(os.path.join(stage, 'crops'))
    with open(os.path.join(stage, 'crops.jsonl'), 'w') as fh:
        for r in collected:
            fh.write(json.dumps(r, sort_keys=True) + '\n')
    with open(os.path.join(stage, 'clusters.json'), 'w') as fh:
        json.dump({'threshold': 0.93, 'groups': [[
            os.path.join(stage, 'crops', 'leashed', 'la_1_d0.jpg'),
            os.path.join(stage, 'crops', 'leashed', 'la_3_d0.jpg')]]}, fh)
    with open(os.path.join(stage, 'collect_counts.json'), 'w') as fh:
        json.dump({'rows': 5}, fh)
    with open(os.path.join(stage, 'collect_conflicts.json'), 'w') as fh:
        json.dump([], fh)
    kept = []
    for r, split in zip(collected[:3], ('train', 'train', 'val')):
        r = dict(r, split=split)
        kept.append(r)
        d = os.path.join(out, split, r['label'])
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, r['uid']), 'w').close()
    with open(os.path.join(out, 'crops.jsonl'), 'w') as fh:
        for r in kept:
            fh.write(json.dumps(r, sort_keys=True) + '\n')
    rm = {'manifest': os.path.join(stage, 'crops.jsonl'),
          'manifest_sha256': _sha(os.path.join(stage, 'crops.jsonl')),
          'dup_clusters_file': os.path.join(stage, 'clusters.json'),
          'source_rows': 5, 'kept': 3, 'sizes': {'warnings': ['median']}}
    with open(os.path.join(out, 'rebuild_manifest.json'), 'w') as fh:
        json.dump(rm, fh)
    bundle = os.path.join(out, 'bundle')
    os.makedirs(bundle)
    return out, stage, bundle, rm


def check_bundle_keeps_stage_records():
    """B1, the bundle step alone: the copies land, and still open."""
    bad = []
    where = tempfile.mkdtemp(dir=TMP)
    out, stage, bundle, rm = _stage_and_out(where)
    said = []
    rec = bd._bundle_crop_record(out, stage, bundle,
                                 os.path.join(where, 'no_base'), [],
                                 classes=('leashed', 'unleashed'),
                                 say=said.append)
    stage_manifest = open(os.path.join(stage, 'crops.jsonl'), 'rb').read()
    shutil.rmtree(stage)                  # what build() does next
    got = os.path.join(bundle, 'collect_manifest.jsonl')
    if not os.path.isfile(got):
        return ['the bundle has no collect_manifest.jsonl: the rows the '
                'split dropped go with the stage']
    if open(got, 'rb').read() != stage_manifest:
        bad.append('collect_manifest.jsonl is not a byte copy of the stage '
                   'manifest, so manifest_sha256 no longer checks')
    if not os.path.isfile(os.path.join(bundle, 'clusters.json')):
        bad.append('the bundle has no clusters.json: the embedding '
                   'near-duplicates go with the stage')
    records = (rec or {}).get('records') or {}
    for name in ('collect_manifest', 'clusters', 'crops'):
        doc = records.get(name)
        if not doc or not os.path.isfile(os.path.join(out, doc['path'])):
            bad.append('the record of %s does not resolve under the dataset '
                       'once the stage is gone: %r' % (name, doc))
    cm = records.get('collect_manifest') or {}
    if cm.get('matches_rebuild_manifest') is not True or \
            cm.get('rows') != 5 or cm.get('built_as') != rm['manifest']:
        bad.append('the collect manifest record does not tie the copy to '
                   'what the build read: %r' % (cm,))
    crops = records.get('crops') or {}
    if crops.get('rows') != 3 or crops.get('rows_resolved') != 3:
        bad.append('the kept rows do not all resolve to <split>/<label>/'
                   '<uid>: %r' % (crops,))
    pointed = bd._point_at_bundle(dict(rm), records)
    for key in ('manifest_in_bundle', 'dup_clusters_file_in_bundle'):
        p = pointed.get(key)
        if not p or not os.path.isfile(os.path.join(out, p)):
            bad.append('rebuild_manifest\'s %s does not name a bundle copy '
                       'that opens: %r' % (key, p))
    if pointed.get('manifest') != rm['manifest']:
        bad.append('the stage path the build actually read was overwritten '
                   'rather than recorded beside')
    rep = bd._read_json(os.path.join(bundle, 'size_report.json'))
    if sorted(((rep.get('shortcut') or {}).get('splits') or {})) != \
            ['train', 'val']:
        bad.append('size_report.json has no per-split size-shortcut '
                   'diagnostic: %r' % (rep.get('shortcut'),))
    elif 'median' not in (rep.get('warnings') or []):
        bad.append('the rebuilder\'s own size warnings were dropped')
    conf = bd._read_json(os.path.join(bundle, 'conflicts.json'))
    if 'collect' not in conf:
        bad.append('the collector\'s conflicts are no longer in '
                   'conflicts.json')
    return bad


def check_record_fields_are_measured():
    """B1/B2: the record's fields are measured, not asserted -- a kept crop
    that is missing is counted missing, a stage manifest changed after the
    rebuild is not called a match, the leashed class is the positive one,
    the shortcut's warnings reach the record, and the test-set baseline is
    quoted only when that test set is the one held out."""
    bad = []
    where = tempfile.mkdtemp(dir=TMP)
    out, stage, bundle, rm = _stage_and_out(where)
    # one kept crop gone from the dataset; the stage manifest edited after
    # the rebuilder hashed it
    os.remove(os.path.join(out, 'train', 'leashed', 'la_1_d0.jpg'))
    with open(os.path.join(stage, 'crops.jsonl'), 'a') as fh:
        fh.write('\n')
    test_set = os.path.join(where, 'leash_test_v1.json')
    with open(test_set, 'w') as fh:
        json.dump({}, fh)
    rec = bd._bundle_crop_record(out, stage, bundle,
                                 os.path.join(where, 'no_base'), [test_set],
                                 classes=('leashed', 'unleashed'),
                                 say=lambda *a: None, family='leash')
    records = (rec or {}).get('records') or {}
    crops = records.get('crops') or {}
    if crops.get('rows') != 3 or crops.get('rows_resolved') != 2:
        bad.append('a kept crop missing from the dataset is not counted: '
                   'rows_resolved %r of %r, want 2 of 3'
                   % (crops.get('rows_resolved'), crops.get('rows')))
    cm = records.get('collect_manifest') or {}
    if cm.get('matches_rebuild_manifest') is not False:
        bad.append('a stage manifest changed after the rebuild still '
                   '"matches" its sha256: %r' % (cm.get('matches_rebuild_'
                                                        'manifest'),))
    sc = (bd._read_json(os.path.join(bundle, 'size_report.json'))
          .get('shortcut') or {})
    if sc.get('positive') != 'leashed':
        bad.append('the diagnostic\'s positive class is %r, not leashed'
                   % (sc.get('positive'),))
    # train kept 40x60 unleashed (<=64) and 90x110 leashed (64-96)
    bands = ((sc.get('splits') or {}).get('train') or {}).get('bands') or {}
    got = {b: (bands.get(b) or {}).get('pos_rate') for b in ('<=64', '64-96')}
    if got != {'<=64': 0.0, '64-96': 1.0}:
        bad.append('the per-band rate is not the LEASHED share: %r' % (got,))
    warned = [w for w in rec.get('size_warnings') or []
              if w.startswith('WARNING: train')]
    if not warned or 'median' not in (rec.get('size_warnings') or []):
        bad.append('the shortcut warnings did not reach the record beside '
                   'the rebuilder\'s own: %r' % (rec.get('size_warnings'),))
    elif 'leash_test_v1: 0.513' not in warned[0]:
        bad.append('the warning does not quote the held-out test set\'s own '
                   'AUC: %r' % (warned[0],))
    # the same build held out against some other test set: no baseline
    where = tempfile.mkdtemp(dir=TMP)
    out, stage, bundle, rm = _stage_and_out(where)
    other = os.path.join(where, 'leash_test_elsewhere.json')
    with open(other, 'w') as fh:
        json.dump({}, fh)
    rec = bd._bundle_crop_record(out, stage, bundle,
                                 os.path.join(where, 'no_base'), [other],
                                 classes=('leashed', 'unleashed'),
                                 say=lambda *a: None, family='leash')
    sc = (bd._read_json(os.path.join(bundle, 'size_report.json'))
          .get('shortcut') or {})
    if sc.get('baseline') is not None or \
            any('0.513' in w for w in rec.get('size_warnings') or []):
        bad.append('a build against another test set is given '
                   'leash_test_v1\'s baseline: %r' % (sc.get('baseline'),))
    return bad


def check_build_keeps_stage_records():
    """B1, through build(): the copy happens BEFORE the stage is deleted, and
    the bundle manifest's rebuild record names the copies."""
    bad = []
    where = tempfile.mkdtemp(dir=TMP)
    root = os.path.join(where, 'root')
    os.makedirs(os.path.join(root, 'leash_v2'))
    reserves = os.path.join(where, 'data')
    os.makedirs(reserves)
    for name in bd.FAMILIES['leash']['reserves']:
        with open(os.path.join(reserves, name), 'w') as fh:
            json.dump({'image_ids': ['1'], 'sequences': ['s']}, fh)
    export = os.path.join(where, 'export.json')
    with open(export, 'w') as fh:
        json.dump([], fh)
    real = (bd.Runner.run, bd.git_head, bd.D,
            os.environ.get('TRAINING_ROOT'))

    def fake_run(self, name, argv, cwd=None, env=None):
        def arg(flag):
            return argv[argv.index(flag) + 1]
        if name == 'crop_sources':
            stage = arg('--stage')
            os.makedirs(os.path.join(stage, 'crops'), exist_ok=True)
            with open(os.path.join(stage, 'crops.jsonl'), 'w') as fh:
                for i in range(4):
                    fh.write(json.dumps(_row(
                        'la_%d_d0.jpg' % i, None,
                        'leashed' if i % 2 else 'unleashed', (100, 100),
                        stage=stage)) + '\n')
            with open(os.path.join(stage, 'collect_counts.json'), 'w') as fh:
                json.dump({'rows': 4}, fh)
        elif name == 'size_reference':
            with open(arg('--out'), 'w') as fh:
                json.dump({}, fh)
        elif name == 'dedup_crops':
            with open(arg('--out'), 'w') as fh:
                json.dump({'groups': []}, fh)
        elif name == 'rebuild_crop_dataset':
            out, man = arg('--out'), arg('--manifest')
            rows = bd._read_jsonl(man)[:2]
            for r, split in zip(rows, ('train', 'val')):
                r['split'] = split
                d = os.path.join(out, split, r['label'])
                os.makedirs(d, exist_ok=True)
                open(os.path.join(d, r['uid']), 'w').close()
            for split in ('train', 'val'):
                for klass in ('leashed', 'unleashed'):
                    os.makedirs(os.path.join(out, split, klass),
                                exist_ok=True)
            with open(os.path.join(out, 'crops.jsonl'), 'w') as fh:
                for r in rows:
                    fh.write(json.dumps(r) + '\n')
            with open(os.path.join(out, 'rebuild_manifest.json'), 'w') as fh:
                json.dump({'manifest': man, 'manifest_sha256': _sha(man),
                           'dup_clusters_file': arg('--dup-clusters'),
                           'kept': 2}, fh)
        self.steps.append({'name': name, 'argv': argv, 'exit_code': 0,
                           'seconds': 0.0})
        return 0

    try:
        os.environ['TRAINING_ROOT'] = root
        bd.Runner.run = fake_run
        bd.git_head = lambda: {'commit': 'x', 'dirty': False}
        bd.D = lambda *p: os.path.join(reserves, *p)
        # the store names its detector generation: a build digests the audit
        # ledgers of that generation (build_dataset.stores_for)
        os.makedirs(os.path.join(where, 'gen1'), exist_ok=True)
        with open(os.path.join(where, 'gen1', 'status.json'), 'w') as fh:
            json.dump({'gen': 1}, fh)
        with contextlib.redirect_stdout(io.StringIO()):
            man = bd.build('leash', detect_root=os.path.join(where, 'gen1'),
                           ls_export_path=export)
    finally:
        bd.Runner.run, bd.git_head, bd.D = real[:3]
        if real[3] is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = real[3]
    out = man['out']
    if os.path.exists(out + '.stage'):
        bad.append('the stage was left behind')
    bundle = os.path.join(out, 'bundle')
    for name in ('collect_manifest.jsonl', 'clusters.json'):
        if not os.path.isfile(os.path.join(bundle, name)):
            bad.append('a finished build\'s bundle has no %s' % (name,))
    on_disk = bd._read_json(os.path.join(bundle, 'manifest.json'))
    rb = (on_disk.get('builder_manifests') or {}).get(
        'rebuild_manifest.json') or {}
    for key in ('manifest_in_bundle', 'dup_clusters_file_in_bundle'):
        p = rb.get(key)
        if not p or not os.path.isfile(os.path.join(out, p)):
            bad.append('the bundle manifest\'s rebuild record still names '
                       'only the deleted stage for %s: %r' % (key, p))
    cm = (((on_disk.get('crops') or {}).get('records') or {})
          .get('collect_manifest') or {})
    if cm.get('matches_rebuild_manifest') is not True:
        bad.append('the bundle copy of the manifest does not match the '
                   'sha256 the rebuilder recorded: %r' % (cm,))
    return bad


def _rows(spec):
    """[(split, label, short_side, source)] -> crops.jsonl-like rows."""
    return [{'split': sp, 'label': lab, 'crop_wh': [s, s], 'source': src}
            for sp, lab, s, src in spec]


def check_band_rates():
    """B2: the per-band table, at the band edges the audit measured in, by
    the SHORT side."""
    bad = []
    spec = ([('train', 'leashed', 30, 'a')] +
            [('train', 'unleashed', 50, 'a')] * 2 +
            [('train', 'unleashed', 64, 'a')] +        # edge: <=64
            [('train', 'leashed', 96, 'a')] +          # edge: 64-96
            [('train', 'leashed', 120, 'b')] * 3 +
            [('train', 'unleashed', 150, 'b')] +
            [('train', 'leashed', 300, 'b')] * 2)
    rep = bd.size_shortcut(_rows(spec), 'leashed')
    tr = rep['splits'].get('train') or {}
    want = {'<=64': (4, 0.25), '64-96': (1, 1.0), '96-128': (3, 1.0),
            '128-192': (1, 0.0), '>192': (2, 1.0)}
    for band, (n, rate) in want.items():
        got = (tr.get('bands') or {}).get(band) or {}
        if got.get('n') != n or got.get('pos_rate') != rate:
            bad.append('band %s: n %r rate %r, want n %d rate %.2f'
                       % (band, got.get('n'), got.get('pos_rate'), n, rate))
    lt = ((tr.get('bands') or {}).get('<=64') or {}).get('classes') or {}
    if lt != {'leashed': 1, 'unleashed': 3}:
        bad.append('the <=64 band does not count each class: %r' % (lt,))
    if sorted(rep['sources']) != ['a', 'b'] or \
            rep['sources']['b']['n'] != 6:
        bad.append('the per-source tables are wrong: %r'
                   % ({k: v['n'] for k, v in rep['sources'].items()},))
    # a 50x300 crop is a small dog in a tall box: <=64, not >192
    tall = [{'split': 'train', 'label': 'unleashed', 'crop_wh': [50, 300],
             'source': 'a'},
            {'split': 'train', 'label': 'leashed', 'crop_wh': [300, 100],
             'source': 'a'}]
    bands = bd.size_shortcut(tall, 'leashed')['splits']['train']['bands']
    got = {b: v['n'] for b, v in bands.items() if v['n']}
    if got != {'<=64': 1, '96-128': 1}:
        bad.append('non-square crops are banded by the wrong side: %r'
                   % (got,))
    return bad


def check_auc_and_warning():
    """B2: AUC 1.0 when size separates the classes perfectly, 0.5 when it is
    independent of them, ties at midrank, and a WARNING past the line."""
    bad = []
    sep = ([('val', 'leashed', s, 'x') for s in (150, 200, 250)] +
           [('val', 'unleashed', s, 'x') for s in (40, 60, 80)])
    said = []
    rep = bd.size_shortcut(_rows(sep), 'leashed')
    bd.print_size_shortcut(rep, say=said.append)
    if rep['splits']['val']['size_auc'] != 1.0:
        bad.append('perfect separation gives AUC %r, not 1.0'
                   % (rep['splits']['val']['size_auc'],))
    if not any('val' in w and 'WARNING' in w for w in rep['warnings']):
        bad.append('a split whose crop size predicts the label perfectly '
                   'is not warned: %r' % (rep['warnings'],))
    if not any(line.startswith('WARNING') for line in said):
        bad.append('the warning is recorded but never printed')
    # same sizes in both classes: every pair a tie, which is 0.5, not 0 or 1
    tie = ([('val', 'leashed', s, 'x') for s in (50, 100, 200)] +
           [('val', 'unleashed', s, 'x') for s in (50, 100, 200)])
    rep = bd.size_shortcut(_rows(tie), 'leashed')
    if rep['splits']['val']['size_auc'] != 0.5 or rep['warnings']:
        bad.append('size independent of the label (tied pairs) gives AUC %r '
                   'and warnings %r' % (rep['splits']['val']['size_auc'],
                                        rep['warnings']))
    # independent at random, n=4000: near 0.5, and no warning
    rng = random.Random(7)
    ind = [('train', rng.choice(('leashed', 'unleashed')),
            rng.randint(20, 400), 'x') for _ in range(4000)]
    rep = bd.size_shortcut(_rows(ind), 'leashed')
    auc = rep['splits']['train']['size_auc']
    if auc is None or abs(auc - 0.5) > 0.03 or rep['warnings']:
        bad.append('independent size and label give AUC %r, warnings %r'
                   % (auc, rep['warnings']))
    # the AUC against a brute-force count over every pair, ties included
    pos = [rng.randint(1, 30) for _ in range(60)]
    neg = [rng.randint(1, 25) for _ in range(70)]
    brute = (sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
             / (len(pos) * len(neg)))
    got = bd.rank_auc(pos + neg, [1] * len(pos) + [0] * len(neg))
    if got is None or abs(got - brute) > 1e-12:
        bad.append('rank_auc %r disagrees with the pairwise count %r'
                   % (got, brute))
    if bd.rank_auc([1, 2, 3], [1, 1, 1]) is not None:
        bad.append('an AUC over one class is reported as a number')
    # area and short side disagree: long thin leashed boxes are larger in
    # area and smaller on the short side, so the two AUCs must be 1 and 0
    thin = ([{'split': 'val', 'label': 'leashed', 'crop_wh': [w, 400],
              'source': 'x'} for w in (40, 42, 44)] +
            [{'split': 'val', 'label': 'unleashed', 'crop_wh': [w, 100],
              'source': 'x'} for w in (100, 102, 104)])
    doc = bd.size_shortcut(thin, 'leashed')['splits']['val']
    if (doc['size_auc'], doc['short_side_auc']) != (1.0, 0.0):
        bad.append('size_auc / short_side_auc are %r / %r on crops where '
                   'area and short side rank the classes opposite ways, '
                   'want 1.0 / 0.0'
                   % (doc['size_auc'], doc['short_side_auc']))
    # a strong inverse shortcut is a shortcut too
    inv = ([('val', 'leashed', s, 'x') for s in (40, 60, 80)] +
           [('val', 'unleashed', s, 'x') for s in (150, 200, 250)])
    if not bd.size_shortcut(_rows(inv), 'leashed')['warnings']:
        bad.append('small-means-positive (AUC 0.0) is not warned')
    return bad


def check_warning_names_the_test_set():
    """B2: what the warning says about the test set is what was measured
    for it. leash_test_v1 is flat (0.51), so a skewed leash build is told
    the test set lacks the shortcut; dogbin_test_v1 is at 0.63, so a gate
    build near that gets a NOTE, never "does not have"; an unmeasured test
    set gets neither claim."""
    bad = []
    leash, gate = bd.TEST_SIZE_AUC['leash'], bd.TEST_SIZE_AUC['dogbin']
    if not (abs(leash['size_auc'] - 0.5) <= bd.SIZE_AUC_WARN - 0.5 <
            abs(gate['size_auc'] - 0.5)):
        bad.append('the recorded baselines no longer straddle the line: '
                   'leash %r, gate %r' % (leash, gate))
    sep = _rows([('val', 'leashed', s, 'x') for s in (150, 200, 250)] +
                [('val', 'unleashed', s, 'x') for s in (40, 60, 80)])
    # AUC 0.625: 10 of the 16 pairs ranked the right way
    mild = _rows([('val', 'dog', s, 'x') for s in (40, 80, 120, 160)] +
                 [('val', 'not_dog', s, 'x') for s in (20, 60, 100, 140)])
    w = bd.size_shortcut(sep, 'leashed', baseline=leash)['warnings']
    if len(w) != 1 or 'does not have' not in w[0] or \
            'leash_test_v1' not in w[0]:
        bad.append('a skewed leash build is not told leash_test_v1 lacks '
                   'it: %r' % (w,))
    w = bd.size_shortcut(sep, 'leashed')['warnings']
    if len(w) != 1 or 'does not have' in w[0] or 'not measured' not in w[0]:
        bad.append('with no measured baseline the warning still claims '
                   'what the test set has: %r' % (w,))
    rep = bd.size_shortcut(mild, 'dog', baseline=gate)
    if rep['splits']['val']['size_auc'] != 0.625:
        bad.append('the mild set is not AUC 0.625: %r'
                   % (rep['splits']['val']['size_auc'],))
    said = rep['warnings'] + rep['notes']
    if rep['warnings'] or len(rep['notes']) != 1 or \
            any('does not have' in x for x in said):
        bad.append('a gate build as skewed as dogbin_test_v1 is warned, or '
                   'told the test set lacks it: %r' % (said,))
    printed = []
    bd.print_size_shortcut(rep, say=printed.append)
    if not any(x.startswith('NOTE') for x in printed):
        bad.append('the NOTE is recorded but never printed')
    far = _rows([('val', 'dog', s, 'x') for s in (150, 200, 250)] +
                [('val', 'not_dog', s, 'x') for s in (40, 60, 80)])
    w = bd.size_shortcut(far, 'dog', baseline=gate)['warnings']
    if len(w) != 1 or 'does not have' in w[0] or 'dogbin_test_v1' not in w[0]:
        bad.append('a gate build far past dogbin_test_v1 is not warned, or '
                   'is told the test set lacks it: %r' % (w,))
    return bad


CHECKS = [check_bundle_keeps_stage_records, check_record_fields_are_measured,
          check_build_keeps_stage_records, check_band_rates,
          check_auc_and_warning, check_warning_names_the_test_set]


def _mutated(fn, old, new):
    """fn re-compiled from its own source with `old` -> `new`, in a copy of
    its module's namespace (so the real function is left alone)."""
    src = inspect.getsource(fn)
    if src.count(old) != 1:
        raise AssertionError('mutant anchor %r not found once in %s'
                             % (old, fn.__name__))
    ns = dict(fn.__globals__)
    exec(compile(src.replace(old, new), fn.__code__.co_filename, 'exec'), ns)
    return ns[fn.__name__]


def _src(fn, old, new, check):
    """A mutant of one function's source, run against `check`."""
    return _mutant({fn.__name__: _mutated(fn, old, new)}, check)


def _mutant(patches, check):
    """Run `check` with module attributes patched; restore them after."""
    def run():
        saved = {k: getattr(bd, k) for k in patches}
        try:
            for k, v in patches.items():
                setattr(bd, k, v)
            return check()
        finally:
            for k, v in saved.items():
                setattr(bd, k, v)
    return run


def _no_tie_auc(scores, labels):
    """The obvious AUC that counts a tie as a loss."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    return sum(p > n for p in pos for n in neg) / (len(pos) * len(neg))


_real_shortcut = bd.size_shortcut


def _silent_shortcut(rows, pos, warn_at=bd.SIZE_AUC_WARN, baseline=None):
    rep = _real_shortcut(rows, pos, warn_at, baseline)
    rep['warnings'] = []
    return rep


MUTANTS = [
    ('the stage records are not copied (the old bundle)',
     _mutant({'_bundle_stage_records': lambda stage, bundle, rm: {}},
             check_bundle_keeps_stage_records)),
    ('the stage records are not copied, through build()',
     _mutant({'_bundle_stage_records': lambda stage, bundle, rm: {}},
             check_build_keeps_stage_records)),
    ('the rebuild record still names only the stage',
     _mutant({'_point_at_bundle': lambda rm, records: rm},
             check_build_keeps_stage_records)),
    ('the size diagnostic is not written to the bundle',
     _mutant({'size_shortcut': lambda rows, pos, **k: {
         'warnings': [], 'splits': {}, 'sources': {}, 'bands_px': [],
         'positive': pos}},
             check_bundle_keeps_stage_records)),
    ('64 px falls into the 64-96 band',
     _mutant({'SIZE_BANDS': ((None, 63, '<=64'), (63, 96, '64-96'),
                             (96, 128, '96-128'), (128, 192, '128-192'),
                             (192, None, '>192'))}, check_band_rates)),
    ('a tie counts as a loss',
     _mutant({'rank_auc': _no_tie_auc}, check_auc_and_warning)),
    ('no warning past the line',
     _mutant({'size_shortcut': _silent_shortcut}, check_auc_and_warning)),
    ('the bands use the long side',
     _src(bd._size_slice, '_size_band(min(w, h))', '_size_band(max(w, h))',
          check_band_rates)),
    ('size_auc is taken on the short side',
     _src(bd._size_slice, 'rank_auc([r[0] * r[1] for r in rows], ys)',
          'rank_auc([min(r[0], r[1]) for r in rows], ys)',
          check_auc_and_warning)),
    ('short_side_auc is taken on area',
     _src(bd._size_slice, 'rank_auc([min(r[0], r[1]) for r in rows], ys)',
          'rank_auc([r[0] * r[1] for r in rows], ys)',
          check_auc_and_warning)),
    ('the positive class is the last one',
     _src(bd._bundle_crop_record, 'pos = (classes or [None])[0]',
          'pos = (classes or [None])[-1]', check_record_fields_are_measured)),
    ('rows_resolved is just the row count',
     _src(bd._bundle_crop_record, "    records['crops'] = {",
          "    resolved = len(kept)\n    records['crops'] = {",
          check_record_fields_are_measured)),
    ('the copied manifest always "matches"',
     _src(bd._bundle_stage_records, "doc['sha256'] == rm['manifest_sha256']",
          'True', check_record_fields_are_measured)),
    ('the shortcut warnings are not merged into the record',
     _src(bd._bundle_crop_record, "+ \\\n            shortcut['warnings']",
          '+ []', check_record_fields_are_measured)),
    ('the baseline is quoted whatever test set was held out',
     _src(bd._bundle_crop_record,
          "if baseline and baseline['test_set'] not in {",
          "if False and baseline['test_set'] not in {",
          check_record_fields_are_measured)),
    ('every build is told the test set does not have it',
     _src(bd.size_shortcut, 'elif abs(ref - 0.5) <= margin:', 'elif True:',
          check_warning_names_the_test_set)),
]


def main():
    bad = []
    try:
        for fn in CHECKS:
            try:
                got = fn()
            except Exception as e:        # noqa: BLE001 - report, not die
                got = ['threw %s: %s' % (type(e).__name__, e)]
            print(('ok   ' if not got else 'FAIL ') + fn.__name__ +
                  ''.join('\n       ' + g for g in got))
            bad += ['%s: %s' % (fn.__name__, g) for g in got]
        for name, fn in MUTANTS:
            try:
                got = fn()
            except Exception as e:        # noqa: BLE001
                got = ['threw %s' % (type(e).__name__,)]
            print(('ok   ' if got else 'FAIL ') + 'MUTANT ' + name +
                  ('' if got else '  (the check still passes with the old '
                   'behaviour put back -- it proves nothing)'))
            if not got:
                bad.append('MUTANT %s survived' % (name,))
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    if bad:
        print('\n%d failure(s)' % (len(bad),))
        return 1
    print('\ncrop bundles keep the pre-split manifest and the clusters the '
          'stage held, every recorded path resolves, and the size shortcut '
          'is measured and warned (%d checks, %d mutants)'
          % (len(CHECKS), len(MUTANTS)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
