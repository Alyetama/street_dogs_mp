#!/usr/bin/env python3
"""Reserve a frozen, sequence-disjoint test set out of the audit verdicts.

    python tools/detect/reserve_test_set.py --family dogbin --dry-run
    python tools/detect/reserve_test_set.py --family dogbin
    python tools/detect/reserve_test_set.py --family leash

WHY. A model is only as honestly measured as the crops it never trained on,
and the two acceptance sets this project has are small (dogbin: 214
negatives, no positives; leash: 312 images) and cut in the OLD framing. The
audit sheets, meanwhile, hold tens of thousands of human answers on crops cut
exactly the way production cuts them, drawn evenly across the model's whole
score range. Some of them are worth more as a TEST than as training -- but
only if the decision is made ONCE, before any build reads them, and written
down where every later build has to read it back.

This does that, and nothing else. No new labelling: every crop it reserves is
a verdict somebody already gave.

WHAT IS ELIGIBLE. A verdict, and every reason one is not, is counted:

  * a real answer (dog / not_dog, leashed / unleashed -- never unsure);
  * from a SPREAD draw. A 'least'/'most confident' sheet hands out one end of
    a band on purpose (fn_audit.targeted); those answers are finds, not a
    sample, and stay trainable;
  * with a real sequence the catalog confirms -- not an `img:` stand-in,
    which could be any sequence's frame and so cannot be held disjoint;
  * whose image_id and sequence are in NONE of the comparison datasets'
    train/val. The default is the deployed model's own dataset (from
    data/best_models.json), so the deployed model and the next one are scored
    on the very same crops neither trained on. Their crop names are read
    with crop_sources.parse_name, so the Label Studio crops the old parser
    could not read are counted too;
  * whose image_id and sequence are not in the existing acceptance set;
  * not flagged as mislabelled, not in a known label conflict, and not a
    file whose exact bytes another verdict labelled the other way;
  * NOT THE SAME PICTURE as anything the comparison datasets or the
    acceptance set hold, under whatever name. The audit pool files one frame
    under several image_ids now and then (a re-upload, a cell twin), so a
    reserved crop can be the byte-identical twin of an image the deployed
    model trained on, under another image_id AND another sequence: 12 of the
    first draw's 1,852 were, 9 of them in dogbin_20261012's train. So every
    verdict's full/ crop is hashed, and so is every file of the comparison
    datasets; a candidate whose sha256 matches, or whose dHash is within
    TWIN_DHASH bits of, a crop of a compared or accepted image (or sequence)
    is out;
  * with its full/ crop on disk: the test crop is the DETECTOR's box as the
    reviewer saw it (full/), which is what production sees -- never edited/.

HOW IT IS DRAWN. Strata are band (0-9) x label. Within each, the units are
sorted, shuffled with a seeded generator, and round(frac * n) taken (at least
one when n >= 5). A unit is a sequence -- or, when two eligible crops are
twins (same bytes, or dHash within TWIN_DHASH) in different sequences, the
UNION of those sequences, so twins are reserved or left out together and a
test crop's copy can never be the one left to train on. Draws from
2026-11-15 on also join CAPTURE twins (same creator, within 60 s and 30 m:
rebuild_crop_dataset.capture_twins), the same moment filed under two
sequences; the frozen v1 sets were drawn before it and stay. A unit is reserved
WHOLE and lives in the stratum of its earliest eligible verdict -- whole
sequences go to test or to train, never both.

THE BAND POPULATIONS ARE RECORDED, because a rate read off a stratified sample
is only a rate after it is weighted by how many boxes each band really holds.
Those bands are the audited model's scores (dogbin_008 / leash_v2_001) on the
gen-1 detector's boxes, and the answer rate per sheet was 42-62%, so the class
mix inside a band is selection-biased: weighted rates are an upper bound, and
two models scored on the SAME reserved crops are comparable with each other.

Refuses to overwrite an existing reservation without --force: re-drawing one
lets a model be measured on crops an earlier build trained on.
"""

import argparse
import collections
import hashlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

FAMILY = {
    'dogbin': {'stage': 'gate', 'project': 'dog-bin', 'frac': 0.20,
               'acceptance': 'dogbin_acceptance_set.json',
               'out': 'dogbin_test_v1.json'},
    'leash': {'stage': 'leash', 'project': 'leash-models', 'frac': 0.30,
              'acceptance': 'leash_acceptance_set.json',
              'out': 'leash_test_v1.json'},
}
SEED = 20261114
# Two crops this close in dHash are taken as one picture. The re-encoded
# twins measured in the first draw sat at 0-1; at 3-4, about four in ten
# pairs are still the same object seen again (the same bracket under the same
# lamp, from another pass) and the rest are flat textures that merely share a
# gradient. 4 rules out 8 more dog-bin candidates than 2 does (of 9,200), and
# 6 would rule out 92 -- mostly the flat textures.
TWIN_DHASH = 4


def D(*parts):
    return os.path.join(REPO, 'data', *parts)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def deployed_dataset(project):
    """The dataset the promoted model of a Comet project was trained on."""
    try:
        with open(D('best_models.json')) as fh:
            doc = json.load(fh)
        return ((doc['projects'][project].get('best') or {})
                .get('dataset'))
    except (OSError, ValueError, KeyError):
        return None


def dataset_files(path):
    """Every train/val crop file of a classify dataset."""
    out = []
    for split in ('train', 'val'):
        d = os.path.join(path, split)
        if not os.path.isdir(d):
            continue
        for cls in sorted(os.listdir(d)):
            cd = os.path.join(d, cls)
            if os.path.isdir(cd):
                out += [os.path.join(cd, f) for f in sorted(os.listdir(cd))
                        if f.lower().endswith('.jpg')]
    return out


def hash_files(paths, workers=8):
    """{path: (sha256, dhash)} -- a file that cannot be read is left out."""
    from concurrent.futures import ThreadPoolExecutor
    from rebuild_crop_dataset import dhash

    def one(p):
        try:
            return p, (sha256(p), dhash(p))
        except OSError:
            return p, None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return {p: h for p, h in ex.map(one, paths) if h}


def _near(arr, h, dist):
    """True if dHash h is within dist bits of any hash in the uint64 array."""
    if h is None or arr is None or not len(arr):
        return False
    import numpy as np
    from rebuild_crop_dataset import _popcount
    return int(_popcount(arr ^ np.uint64(int(h))).min()) <= dist


def twin_units(cands, dist=TWIN_DHASH, captures=None):
    """Give every candidate a draw `unit`: its sequence, joined with the
    sequences of its twins (same sha256, or dHash within `dist`) -- and,
    when `captures` ({image_id: {creator, ts, lon, lat}}) is given, of its
    CAPTURE twins: rebuild_crop_dataset.capture_twins, the one rule the
    build splits and reserves by (same creator, within 60 s and 30 m). The
    same dog at the same moment in two sequences then goes to test or to
    train as one.

    FOR DRAWS FROM 2026-11-15 ON. leash_test_v1 and dogbin_test_v1 were drawn
    without it and stay as drawn (main() refuses to overwrite them); the
    build drops a capture twin of any of their images instead, and against
    leash_test_v1 there were none (0 at 60 s / 30 m, 0 at 300 s / 50 m).
    """
    from rebuild_crop_dataset import UnionFind, cluster, capture_twins
    uf = UnionFind()
    for c in cands:
        uf.find(c['seq'])
    by_sha = collections.defaultdict(list)
    for c in cands:
        by_sha[c['sha256']].append(c['seq'])
    for seqs in by_sha.values():
        for s in seqs[1:]:
            uf.union(seqs[0], s)
    for g in cluster([(c['seq'], c.get('dhash')) for c in
                      sorted(cands, key=lambda c: c['key'])], dist):
        for s in g[1:]:
            uf.union(g[0], s)
    if captures:
        seq_of = {}
        for c in sorted(cands, key=lambda c: c['key']):
            seq_of.setdefault(str(c['image_id']), c['seq'])
        for a, b in capture_twins([(i, captures.get(i))
                                   for i in sorted(seq_of)]):
            uf.union(seq_of[a], seq_of[b])
    for c in cands:
        c['unit'] = uf.find(c['seq'])
    return cands


def capture_draw_units(cands, fenced_ids, why, python_bin, repo=REPO):
    """(kept candidates, stats): the capture-twin rule applied to a draw.

    Two steps, one catalog lookup (candidates and fenced images together):
      1. a candidate that is a CAPTURE twin of a compared-dataset or
         acceptance image is not eligible. eligible() fences those images by
         image_id, sequence, bytes and dHash, but the same dog photographed
         by the same creator 3 s later in another sequence passes all four --
         and a test item the deployed model trained on the moment of inflates
         its score. Counted out of why['eligible'] into
         why['capture_twin_of_compare_or_acceptance'] (one per verdict).
      2. twin_units with the captures, so capture twins among the candidates
         share a draw unit.
    Against leash_test_v1 step 1 would have found none (0 at 300 s / 50 m
    against leash_v2); it is for future draws.
    """
    from rebuild_crop_dataset import (resolve_capture, capture_twins,
                                      capture_known)
    cand_ids = {str(c['image_id']) for c in cands}
    fenced = {str(i) for i in fenced_ids} - cand_ids
    caps = resolve_capture(cand_ids | fenced, repo, python_bin)
    nodes = ([('c:' + i, caps.get(i)) for i in sorted(cand_ids)]
             + [('x:' + i, caps.get(i)) for i in sorted(fenced)])
    hit = set()
    for a, b in capture_twins(nodes):
        if a[:2] != b[:2]:
            hit.add((a if a.startswith('c:') else b)[2:])
    kept = [c for c in cands if str(c['image_id']) not in hit]
    if len(kept) < len(cands):
        why['eligible'] -= len(cands) - len(kept)
        why['capture_twin_of_compare_or_acceptance'] += len(cands) - len(kept)
    n_before = len({c['unit'] for c in twin_units([dict(c) for c in kept])})
    twin_units(kept, captures=caps)
    kept_ids = {str(c['image_id']) for c in kept}
    return kept, {
        'draw_units_before': n_before,
        'draw_units_after': len({c['unit'] for c in kept}),
        'fenced_twin_images': sorted(hit),
        'images': len(kept_ids),
        'images_without_capture': sum(1 for i in kept_ids
                                      if not capture_known(caps.get(i))),
        'fenced_images_without_capture': sum(
            1 for i in fenced if not capture_known(caps.get(i))),
    }


def dataset_ids(path):
    """(image_ids, n crops) of a classify dataset's train/val, by parse_name.

    An unreadable name FAILS: a crop this cannot place is a crop the
    reservation would not know it is overlapping.
    """
    from crop_sources import parse_name
    ids, n, unread = set(), 0, []
    for split in ('train', 'val'):
        d = os.path.join(path, split)
        if not os.path.isdir(d):
            continue
        for cls in os.listdir(d):
            cd = os.path.join(d, cls)
            if not os.path.isdir(cd):
                continue
            for f in os.listdir(cd):
                if not f.lower().endswith('.jpg'):
                    continue
                n += 1
                got = parse_name(f)
                if got is None:
                    unread.append(f)
                else:
                    ids.add(got[0])
    if unread:
        raise SystemExit('%s: %d crop name(s) give no image_id, e.g. %s'
                         % (path, len(unread), sorted(unread)[:10]))
    if not n:
        raise SystemExit('%s holds no train/val crops' % (path,))
    return ids, n


def draw(cands, frac, seed=SEED):
    """{stratum: (eligible sequences, reserved sequences)} -- the whole draw.

    `cands` is [{key, label, band, seq, ts, unit?, ...}]. The unit drawn is
    `unit` (twin_units: a sequence joined with its twins' sequences) or, when
    absent, the sequence. Each unit goes to the stratum of its earliest
    verdict; within a stratum the units are sorted, shuffled by a generator
    seeded from (seed, stratum) and the first round(frac * n) taken, with at
    least one whenever n >= 5. Seeding per stratum keeps a stratum's draw
    fixed even if another stratum's population changes.
    """
    first = {}
    for c in sorted(cands, key=lambda c: (c.get('ts') or 0, c['key'])):
        first.setdefault(c.get('unit') or c['seq'], c)
    strata = collections.defaultdict(list)
    for s, c in first.items():
        strata[(int(c['band']) if c.get('band') is not None else -1,
                c['label'])].append(s)
    out = {}
    for st in sorted(strata):
        seqs = sorted(strata[st])
        rng = random.Random('%d:%d:%s' % (seed, st[0], st[1]))
        rng.shuffle(seqs)
        k = int(round(frac * len(seqs)))
        if len(seqs) >= 5:
            k = max(1, k)
        out[st] = (sorted(strata[st]), sorted(seqs[:k]))
    return out


def eligible(family, verdicts, aimed, crop_dir, excluded, compare_ids,
             compare_seqs, acc_ids, acc_seqs, catalog_seq, labels,
             ref_hashes=(), hashes=None):
    """(candidates, reason counts). Every verdict lands in exactly one bucket.

    `ref_hashes` is [(sha256, dhash)] of crops that are NOT verdicts but are
    in a comparison dataset (its own files). The full/ crop of every verdict
    whose image_id or sequence is compared or accepted is a reference too,
    whatever its label and whether or not it is eligible itself -- the twin
    that matters is the one the deployed model trained on. `hashes` is
    {path: (sha256, dhash)} if the caller already has them.
    """
    import numpy as np
    why = collections.Counter()
    cands = []
    paths = {v['key']: os.path.join(crop_dir,
                                    v['key'].replace('#', '_') + '.jpg')
             for v in verdicts}
    if hashes is None:
        hashes = hash_files(sorted({p for p in paths.values()
                                    if os.path.isfile(p)}))
    # bytes: a crop whose exact file another verdict labelled differently
    by_sha = collections.defaultdict(set)
    sha_of, dh_of = {}, {}
    for v in verdicts:
        h = hashes.get(paths[v['key']])
        if not h:
            continue
        sha_of[v['key']], dh_of[v['key']] = h
        if v['_label'] in labels:
            by_sha[h[0]].add(v['_label'])
    # what the compared and accepted images look like, under any name
    fenced_ids = set(compare_ids) | set(acc_ids)
    fenced_seqs = set(compare_seqs) | set(acc_seqs)
    ref_sha, ref_dh = set(), []
    for sha, dh in ref_hashes or ():
        ref_sha.add(sha)
        if dh is not None:
            ref_dh.append(int(dh))
    for v in verdicts:
        k = v['key']
        if k not in sha_of:
            continue
        iid = k.partition('#')[0]
        if iid in fenced_ids or v.get('seq') in fenced_seqs \
                or catalog_seq.get(iid) in fenced_seqs:
            ref_sha.add(sha_of[k])
            if dh_of[k] is not None:
                ref_dh.append(int(dh_of[k]))
    ref_arr = np.array(sorted(set(ref_dh)), dtype=np.uint64) \
        if ref_dh else None
    for v in sorted(verdicts, key=lambda v: v['key']):
        lab, key = v['_label'], v['key']
        iid = key.partition('#')[0]
        vs = v.get('seq')
        cs = catalog_seq.get(iid)
        if lab not in labels:
            why['not_a_label'] += 1
        elif key in aimed:
            why['aimed_draw'] += 1
        elif key not in sha_of:
            why['no_full_crop'] += 1
        elif iid in excluded:
            why['flagged_or_conflicting'] += 1
        elif not vs or str(vs).startswith('img:'):
            why['no_real_sequence'] += 1
        elif cs != vs:
            why['sequence_not_confirmed'] += 1
        elif iid in compare_ids:
            why['image_in_compare_dataset'] += 1
        elif vs in compare_seqs:
            why['sequence_in_compare_dataset'] += 1
        elif iid in acc_ids:
            why['image_in_acceptance_set'] += 1
        elif vs in acc_seqs:
            why['sequence_in_acceptance_set'] += 1
        elif sha_of[key] in ref_sha:
            why['same_bytes_as_compare_or_acceptance'] += 1
        elif _near(ref_arr, dh_of[key], TWIN_DHASH):
            why['near_copy_of_compare_or_acceptance'] += 1
        elif len(by_sha[sha_of[key]]) > 1:
            why['same_bytes_other_label'] += 1
        else:
            why['eligible'] += 1
            cands.append({'key': key, 'image_id': iid, 'label': lab,
                          'band': v.get('band'), 'p_model': v.get('p_dog'),
                          'seq': vs, 'ts': v.get('ts'),
                          'sha256': sha_of[key], 'dhash': dh_of[key]})
    return cands, why


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--family', required=True, choices=sorted(FAMILY))
    ap.add_argument('--frac', type=float,
                    help='share of each stratum\'s sequences to reserve '
                         '(default 0.20 dogbin, 0.30 leash)')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--out', help='default data/<family>_test_v1.json')
    ap.add_argument('--compare', action='append', default=[],
                    help='a dataset whose train/val the test set must be '
                         'disjoint from (default: the deployed model\'s, from '
                         'data/best_models.json)')
    ap.add_argument('--also', action='append', default=[],
                    help='another dataset to be disjoint from, ON TOP of the '
                         'default -- e.g. dogbin_v5, to compare dogbin_008 too')
    ap.add_argument('--acceptance',
                    help='the existing acceptance set (default '
                         'data/<family>_acceptance_set.json)')
    ap.add_argument('--python', default=os.environ.get('DUCKDB_PYTHON')
                    or sys.executable,
                    help='interpreter with duckdb, for the sequence lookup')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)

    import fn_audit as fa
    import crop_sources
    from rebuild_crop_dataset import (resolve_sequences, TWIN_SECONDS,
                                      TWIN_METRES, TWIN_SECONDS_NOGEO)
    fam = FAMILY[a.family]
    stage = fam['stage']
    frac = a.frac if a.frac is not None else fam['frac']
    out = a.out or D(fam['out'])
    if os.path.exists(out) and not a.force and not a.dry_run:
        with open(out) as fh:
            prev = json.load(fh)
        raise SystemExit(
            '%s already reserves %d keys (drawn %s). Re-drawing would let a '
            'model be measured on crops an earlier build trained on; --force '
            'only if no dataset has been built against it.'
            % (out, len(prev.get('keys') or []), prev.get('created_at')))

    # the comparison datasets
    import build_dataset
    root = build_dataset.training_root()
    compare = list(a.compare)
    if not compare:
        dep = deployed_dataset(fam['project'])
        if not dep:
            raise SystemExit('no deployed dataset for %s in best_models.json '
                             '-- pass --compare' % (fam['project'],))
        compare = [os.path.join(root, dep)]
    compare += a.also
    compare = [c if os.path.isabs(c) else os.path.join(root, c)
               for c in compare]
    comp_ids, comp_doc, comp_files = set(), [], []
    for c in compare:
        ids, n = dataset_ids(c)
        comp_ids |= ids
        comp_files += dataset_files(c)
        comp_doc.append({'path': c, 'crops': n, 'image_ids': len(ids)})
    acc_path = a.acceptance or D(fam['acceptance'])
    with open(acc_path) as fh:
        acc = json.load(fh)
    acc_ids = {str(i) for i in acc.get('image_ids') or []}

    P = fa.paths(stage)
    labels = fa.spec(stage)['answers'][:2]
    verdicts = []
    for v in fa.read_verdicts(stage=stage):
        v['_label'] = fa.verdict_of(v.get('verdict'), stage)
        verdicts.append(v)
    aimed = fa.targeted(stage)
    excluded, _digests = crop_sources.exclusions(a.family, root)

    want = (comp_ids | acc_ids
            | {v['key'].partition('#')[0] for v in verdicts})
    print('resolving sequences for %d images ...' % (len(want),))
    seq = resolve_sequences(want, REPO, a.python)
    comp_seqs = {seq[i] for i in comp_ids if i in seq}
    acc_seqs = set(acc.get('sequences') or ()) | \
        {seq[i] for i in acc_ids if i in seq}
    print('hashing %d compared-dataset files and the audit crops ...'
          % (len(comp_files),))
    comp_hashes = list(hash_files(comp_files).values())
    cands, why = eligible(a.family, verdicts, aimed, P['full'], excluded,
                          comp_ids, comp_seqs, acc_ids, acc_seqs, seq, labels,
                          ref_hashes=comp_hashes)
    print('capture lookup for the candidates and %d compared/accepted '
          'images ...' % (len(comp_ids | acc_ids),))
    cands, cap_units = capture_draw_units(cands, comp_ids | acc_ids, why,
                                          a.python)
    print('verdicts %d -> eligible %d' % (len(verdicts), len(cands)))
    for k, n in sorted(why.items(), key=lambda kv: -kv[1]):
        print('  %-36s %d' % (k, n))
    n_units = cap_units['draw_units_after']
    n_bytes_units = cap_units['draw_units_before']
    print('twins: %d eligible sequences in %d draw units (%d before capture '
          'twins; %d of %d images have no capture)'
          % (len({c['seq'] for c in cands}), n_units, n_bytes_units,
             cap_units['images_without_capture'], cap_units['images']))
    strata = draw(cands, frac, a.seed)
    held_units = set()
    print('\nband  label        eligible  reserved')
    for (band, lab), (el, rs) in sorted(strata.items()):
        print('  %2d  %-12s %8d  %8d' % (band, lab, len(el), len(rs)))
        held_units.update(rs)
    picked = sorted((c for c in cands if c['unit'] in held_units),
                    key=lambda c: c['key'])
    held = {c['seq'] for c in picked}
    print('reserved %d sequences, %d crops (%s)'
          % (len(held), len(picked), ', '.join(
              '%s %d' % kv for kv in sorted(collections.Counter(
                  c['label'] for c in picked).items()))))
    if a.dry_run:
        print('\n--dry-run: nothing written')
        return 0
    pool = fa.pool_info(stage)
    # the protocol the pool's scores were made under; a pool cut before the
    # key existed was cut from the legacy 224 verdicts (fn_audit.pool_info)
    pool_protocol = pool.get('protocol') or 'legacy'
    doc = {
        'created': time.strftime('%Y-%m-%d'),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'purpose': 'Frozen test set for %s: these keys, image_ids AND '
                   'sequences must never enter any training or validation '
                   'split. build_dataset.py passes this file to '
                   'rebuild_crop_dataset.py --reserve.' % (a.family,),
        'family': a.family, 'stage': stage, 'seed': a.seed, 'frac': frac,
        'ledger': {'path': P['verdicts'], 'sha256': sha256(P['verdicts'])},
        'crops_from': P['full'],
        'crop_framing': 'audit_full (gate_store.cut_crops constants, the '
                        "detector's box as the reviewer saw it)",
        'compared_datasets': comp_doc,
        'acceptance_set': {'path': acc_path, 'sha256': sha256(acc_path),
                           'image_ids': len(acc_ids),
                           'sequences': len(acc_seqs)},
        'eligibility': dict(why),
        'twin_dhash': TWIN_DHASH,
        'twin_capture': {'seconds': TWIN_SECONDS, 'metres': TWIN_METRES,
                         'seconds_without_geometry': TWIN_SECONDS_NOGEO,
                         'same_creator': True,
                         'draw_units_before': n_bytes_units,
                         'draw_units_after': n_units,
                         'fenced_twin_images':
                             cap_units['fenced_twin_images'],
                         'fenced_images_without_capture':
                             cap_units['fenced_images_without_capture']},
        'compared_files_hashed': len(comp_hashes),
        # p_model IS THE STRATIFYING SCORE, NOT A MODEL'S ANSWER. It is the
        # audit pool's stored score (fn_audit STAGES[stage].p_col) and it only
        # defines the sampling bands. On leash_test_v1 it is leash_v2_001's
        # legacy 224-px score (the pool predates the protocol key, so it was
        # cut from the gen-1 224 verdicts -- the gate-224 bug fixed in
        # 1d147bc0), and its AUC against the 628 human labels is 0.49. Read
        # as "what the old model says" it puts the old model at chance;
        # re-score the checkpoints at their trained imgsz instead. Band
        # weighting itself stays valid as a design weight.
        'p_model_protocol': pool_protocol,
        'band_populations': {
            'bands': [list(b) for b in fa.band_totals(stage)],
            'pool': pool,
            'note': 'bands are %s scores (pool protocol %r) on gen-1 '
                    'detector boxes. They define the sampling strata ONLY: '
                    'items[].p_model is that stratifying score, not a '
                    'measurement of the model, and must not be read as its '
                    'accuracy. Sheet answer rates were 42-62%%, so the '
                    'within-band class mix is selection-biased and weighted '
                    'rates are upper bounds'
                    % (fa.spec(stage)['model'], pool_protocol)},
        'strata': [{'band': b, 'label': lab, 'eligible_units': len(el),
                    'reserved_units': len(rs)}
                   for (b, lab), (el, rs) in sorted(strata.items())],
        'keys': [c['key'] for c in picked],
        'image_ids': sorted({c['image_id'] for c in picked}),
        'sequences': sorted(held),
        # sha256 AND dhash of each reserved crop, so a build can drop a copy
        # of one filed under another name (rebuild_crop_dataset reads both)
        'items': [{k: c[k] for k in ('key', 'label', 'band', 'p_model',
                                     'seq', 'sha256', 'dhash')}
                  | {'p_model_protocol': pool_protocol}
                  for c in picked],
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    os.replace(tmp, out)
    print('-> %s' % (out,))
    return 0


if __name__ == '__main__':
    sys.exit(main())
