#!/usr/bin/env python3
"""
Re-split and de-duplicate an existing crop-classifier dataset.

Two defects were measured in the first build of this dataset
(``leash_binary_v1``, since renamed ``dogbin_v1``) on 2026-08-02:

1. **Sequence leakage.** Mapillary images come in sequences -- consecutive
   frames from one camera pass, seconds apart, same animal, same street,
   different ``image_id``. The build split per image, so ``image_id`` overlap
   was zero and looked clean, while **70.8% of val images (363/513) sat in a
   sequence that also appeared in train**. Measured effect on dogbin_001:
   ROC AUC 0.9686 on the split as shipped vs **0.9213** on the
   sequence-clean remainder; not-dog rejection at full dog recall 77.9% vs
   58.3%.

2. **Near-duplicate inflation.** Those same frames are still redundant even
   after a clean split: six views of one dog carry roughly one example's
   worth of signal but get six times the training weight, and a val set
   built from them reports a sample size it does not have.

This tool fixes both without re-cutting a single crop -- the existing files
are already full-resolution, so it only needs to decide what to keep and
where each crop goes:

* resolve every crop's ``sequence`` from the ground_animals parquets;
* collapse near-duplicates by perceptual hash **within a class**, so a dog is
  never merged into a negative;
* cap the number of crops kept per (sequence, class);
* assign **whole sequences** to train/val, stratified per class;
* optionally fold in extra full-resolution crops from ``harvest_flagged.py``:
  negatives (reviewer-flagged false positives) into not_dog, and positives
  (low-confidence detections the reviewer confirmed are dogs) into dog.

Every count is written to ``rebuild_manifest.json``, including what was
dropped and why -- a dataset that silently shrinks is worse than one that
never changed.

    python tools/detect/rebuild_crop_dataset.py \\
        --src <datasets>/dogbin_v1 \\
        --out <datasets>/dogbin_v3 \\
        --extra-negatives <harvested full-res flagged crops> --execute

READ-ONLY on --src.
"""

import argparse
import collections
import json
import os
import re
import shutil
import sys

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# le_/un_ = leashed/unleashed (folded to dog), no_/nd_ = annotator negatives,
# flag_ = a reviewer-flagged false positive, pos_ = a reviewer-confirmed dog.
# A prefix missing from here does NOT fail loudly: image_id_of() returns None,
# the crop is treated as sequence-less and pinned to train, and it silently
# escapes the leak check this whole tool exists to enforce.
NAME_RE = re.compile(r'^(?:flag_|pos_)?(?:le|un|no|nd)?_?(\d{6,})[_.]')


def image_id_of(fname):
    m = NAME_RE.match(fname)
    return m.group(1) if m else None


def list_crops(root):
    """{split: {cls: [filenames]}} for an existing train/val dataset."""
    out = {}
    for split in ('train', 'val', 'test'):
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        for cls in sorted(os.listdir(d)):
            cd = os.path.join(d, cls)
            if not os.path.isdir(cd):
                continue
            fs = [f for f in os.listdir(cd) if f.lower().endswith('.jpg')]
            out.setdefault(split, {})[cls] = sorted(fs)
    return out


def resolve_sequences(image_ids, repo, python_bin):
    """{image_id: sequence} from the ground_animals parquets via duckdb."""
    if not image_ids:
        return {}
    # The result goes through a FILE, not stdout: duckdb prints progress bars
    # and the odd warning to stdout, which corrupts a JSON-on-stdout contract.
    prog = r'''
import duckdb, json, sys
spec = json.load(open(sys.argv[1]))
con = duckdb.connect(spec["db"], read_only=True)
paths = [r[0] for r in con.execute(
    "SELECT path FROM files WHERE path LIKE '%ground_animals%'").fetchall()]
con.execute("CREATE TEMP TABLE want(image_id VARCHAR)")
con.executemany("INSERT INTO want VALUES (?)", [(i,) for i in spec["ids"]])
src = "read_parquet([" + ",".join(
    "'" + p.replace("'", "''") + "'" for p in paths) + "])"
# min(), not the last row the JOIN happens to emit. The manifests carry
# duplicate rows (3.4% corpus-wide), so an image_id can come back more than
# once -- and at least one comes back with a real sequence AND a NULL. A dict
# comprehension over the raw rows keeps whichever landed last, which a
# parallel query does not fix in place: the same id flipped sequence between
# runs, moving its whole group across the train/val boundary and with it 353
# crops. NULLs are dropped rather than ranked; absence of a sequence is not a
# sequence, and the caller already pins unresolved ids to train.
rows = con.execute(
    "SELECT CAST(p.image_id AS VARCHAR) i, min(CAST(p.sequence AS VARCHAR)) s "
    "FROM " + src + " p JOIN want w ON CAST(p.image_id AS VARCHAR)=w.image_id "
    "WHERE p.sequence IS NOT NULL GROUP BY 1"
).fetchall()
json.dump({i: s for i, s in rows}, open(spec["out"], "w"))
'''
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        spec_p = os.path.join(tmp, 'spec.json')
        out_p = os.path.join(tmp, 'seq.json')
        with open(spec_p, 'w') as f:
            json.dump({'db': os.path.join(repo, 'data', 'catalog.duckdb'),
                       'ids': sorted(image_ids), 'out': out_p}, f)
        p = subprocess.run([python_bin, '-c', prog, spec_p],
                           capture_output=True, text=True)
        if p.returncode != 0 or not os.path.exists(out_p):
            raise SystemExit('sequence lookup failed:\n'
                             + (p.stderr or p.stdout or '')[-800:])
        with open(out_p) as f:
            return json.load(f)


def dhash(path, size=8):
    """64-bit difference hash. Robust to the small shifts between frames."""
    from PIL import Image
    try:
        im = Image.open(path).convert('L').resize((size + 1, size),
                                                  Image.BILINEAR)
    except Exception:
        return None
    px = list(im.getdata())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | (1 if row[c] < row[c + 1] else 0)
    return bits


def cluster(hashes, max_dist):
    """Greedy near-duplicate clustering. hashes: [(key, hash)] -> [[key,...]]"""
    if max_dist <= 0:
        return [[k] for k, _ in hashes]
    reps, groups = [], []
    for key, h in hashes:
        if h is None:
            groups.append([key])
            reps.append(None)
            continue
        placed = False
        for i, rh in enumerate(reps):
            if rh is not None and bin(rh ^ h).count('1') <= max_dist:
                groups[i].append(key)
                placed = True
                break
        if not placed:
            reps.append(h)
            groups.append([key])
    return groups


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    # derived from this file's location, not baked in
    ap.add_argument('--repo', default=REPO)
    ap.add_argument('--duckdb-python',
                    default=os.environ.get('DUCKDB_PYTHON') or sys.executable,
                    help='interpreter with duckdb, for the sequence lookup')
    ap.add_argument('--extra-negatives',
                    help='directory of additional FULL-RESOLUTION negative '
                         'crops (harvest_flagged.py --label false_positive)')
    ap.add_argument('--extra-positives',
                    help='directory of additional FULL-RESOLUTION positive '
                         'crops (harvest_flagged.py --label true_positive): '
                         'the low-confidence real dogs the reviewer '
                         'confirmed. Without this the "Is a dog" button '
                         'collects data nothing ever trains on.')
    ap.add_argument('--neg-class', default='not_dog')
    ap.add_argument('--pos-class', default='dog')
    ap.add_argument('--max-per-sequence', type=int, default=3,
                    help='cap crops kept per (sequence, class). Six frames of '
                         'one dog are about one example of signal.')
    ap.add_argument('--hamming', type=int, default=6,
                    help='dHash distance under which two crops of the SAME '
                         'class count as duplicates. 0 disables.')
    ap.add_argument('--exclude-ids',
                    default=os.path.join(REPO, 'data',
                                         'dogbin_acceptance_set.json'),
                    help='JSON with an "image_ids" list that must never enter '
                         'ANY split -- the permanent acceptance set written by '
                         'reserve_acceptance_set.py. On by default: dogbin_v3 '
                         'was accepted on a test it had trained on, and an '
                         'opt-in guard would not have prevented that. Pass "" '
                         'to disable.')
    ap.add_argument('--dup-clusters',
                    help='clusters.json from dedup_crops.py. dHash only sees '
                         'near-identical framings; the embedding clusters also '
                         'catch the same subject at another distance or angle, '
                         'which is what consecutive frames actually look like.')
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    src = list_crops(args.src)
    if not src:
        raise SystemExit(f'no train/val class dirs under {args.src}')

    # ---- gather every crop as (cls, path, image_id) ----------------------
    items = []
    for split, byc in src.items():
        for cls, fs in byc.items():
            for f in fs:
                items.append((cls, os.path.join(args.src, split, cls, f), f))
    for d, cls, tag, what in (
            (args.extra_negatives, args.neg_class, 'flag_', 'negatives'),
            (args.extra_positives, args.pos_class, 'pos_', 'positives')):
        if not d or not os.path.isdir(d):
            continue
        n0 = len(items)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith('.jpg'):
                items.append((cls, os.path.join(d, f), tag + f))
        print(f'extra {what} folded in: {len(items) - n0:,}')
    print(f'source crops: {len(items):,}')

    # ---- drop the permanent acceptance set -------------------------------
    # Before anything else: a crop reserved for accepting the model must not
    # reach the splitter at all. This is the guard whose absence let dogbin_v3
    # train on 297 of the crops it was later accepted on.
    held = set()
    if args.exclude_ids:
        try:
            with open(args.exclude_ids) as f:
                held = set(json.load(f).get('image_ids') or [])
        except OSError:
            print(f'no acceptance set at {args.exclude_ids} -- nothing held out')
        except ValueError as e:
            raise SystemExit(f'{args.exclude_ids} is not readable JSON: {e}')
    n_before_hold, n_held_removed = len(items), 0
    if held:
        n0 = len(items)
        items = [it for it in items if image_id_of(it[2]) not in held]
        n_held_removed = n0 - len(items)
        print(f'acceptance set: {len(held):,} reserved image_ids, '
              f'{n0 - len(items):,} crops removed -> {len(items):,} trainable')
        if n0 == len(items):
            print('  WARNING: none matched. If the reservation was drawn from '
                  'a different harvest, it is not protecting this build.')

    ids = {i for _, _, f in items if (i := image_id_of(f))}
    print(f'distinct source images: {len(ids):,}  -- resolving sequences ...')
    seq = resolve_sequences(ids, args.repo, args.duckdb_python)
    print(f'  resolved {len(seq):,} / {len(ids):,}'
          f'  ({len(ids) - len(seq):,} treated as their own sequence)')

    def seq_of(fname):
        iid = image_id_of(fname)
        if iid is None:
            return 'nofile:' + fname
        return seq.get(iid) or ('noseq:' + iid)

    # ---- collapse near-duplicates, within a class ------------------------
    print(f'\nhashing {len(items):,} crops (dHash, hamming<={args.hamming}) ...')
    by_cls = collections.defaultdict(list)
    for cls, path, fname in items:
        by_cls[cls].append((path, fname))
    # Embedding clusters, if supplied, join crops dHash cannot see are the
    # same: the same animal one frame later, closer, or cropped differently.
    # Keyed by ABSOLUTE path, because the same file reaches this function
    # under two names (a src crop keeps its filename, an extra negative gets a
    # flag_ prefix).
    emb_group = {}
    if args.dup_clusters:
        with open(args.dup_clusters) as f:
            cl = json.load(f)
        for gi, g in enumerate(cl.get('groups', [])):
            for p in g:
                emb_group[os.path.realpath(p)] = gi
        print(f'  embedding clusters loaded : {len(cl.get("groups", [])):,} '
              f'groups covering {len(emb_group):,} crops '
              f'(cosine >= {cl.get("threshold")})')

    keep, dropped_dup, dropped_cap = [], collections.Counter(), collections.Counter()
    for cls, lst in by_cls.items():
        hashes = [((path, fname), dhash(path)) for path, fname in lst]
        groups = cluster(hashes, args.hamming)
        if emb_group:
            # merge dHash groups that share an embedding cluster
            merged, by_eg = [], {}
            for g in groups:
                egs = {emb_group[k] for k in
                       (os.path.realpath(p) for p, _ in g)
                       if k in emb_group}
                tgt = next((by_eg[e] for e in egs if e in by_eg), None)
                if tgt is None:
                    merged.append(list(g))
                    tgt = len(merged) - 1
                else:
                    merged[tgt] += g
                for e in egs:
                    by_eg[e] = tgt
            groups = [g for g in merged if g]
        # one survivor per near-duplicate cluster: the largest file, i.e. the
        # sharpest crop rather than an arbitrary frame
        survivors = []
        for g in groups:
            if len(g) == 1:
                survivors.append(g[0])
                continue
            best = max(g, key=lambda pf: os.path.getsize(pf[0]))
            survivors.append(best)
            dropped_dup[cls] += len(g) - 1
        # then cap per sequence
        per_seq = collections.defaultdict(list)
        for path, fname in survivors:
            per_seq[seq_of(fname)].append((path, fname))
        # sorted for the same reason as the split below: which crops survive
        # the cap must not depend on the order duckdb happened to return rows
        for s in sorted(per_seq):
            lst2 = sorted(per_seq[s])
            rng.shuffle(lst2)
            if len(lst2) > args.max_per_sequence:
                dropped_cap[cls] += len(lst2) - args.max_per_sequence
                lst2 = lst2[:args.max_per_sequence]
            for path, fname in lst2:
                keep.append((cls, path, fname, s))
    print(f'  near-duplicates collapsed : {sum(dropped_dup.values()):,} '
          f'{dict(dropped_dup)}')
    print(f'  over per-sequence cap     : {sum(dropped_cap.values()):,} '
          f'{dict(dropped_cap)}')
    print(f'  kept                      : {len(keep):,}')

    # ---- split whole sequences, stratified per class ---------------------
    # a sequence is assigned once, for ALL classes it touches, or a dog frame
    # and a goat frame from one pass could still straddle the split
    seq_items = collections.defaultdict(list)
    for cls, path, fname, s in keep:
        seq_items[s].append((cls, path, fname))
    # order sequences by their dominant class so the per-class val fraction
    # lands close to the target even though assignment is whole-sequence
    # A crop whose sequence could not be resolved is a singleton here, but its
    # real sequence-mates may be resolved and sitting in train -- putting it in
    # val would leak exactly what this tool exists to prevent. Pin all of them
    # to train: unresolved ids are a handful, and a slightly larger train set
    # is harmless where a contaminated val set is not.
    # sorted() before shuffle, or --seed means nothing. seq_items is insertion
    # ordered from `keep`, which inherits the row order of a duckdb JOIN inside
    # resolve_sequences -- and a parallel query does not promise one. Two runs
    # with identical inputs and the same seed selected the identical 2,436
    # crops but moved 353 of them between train and val, because shuffling a
    # differently-ordered list with the same seed is a different shuffle.
    seqs = sorted(s for s in seq_items
                  if not s.startswith(('noseq:', 'nofile:')))
    pinned = sorted(s for s in seq_items
                    if s.startswith(('noseq:', 'nofile:')))
    if pinned:
        print(f'  {sum(len(seq_items[s]) for s in pinned):,} crops with no '
              f'resolvable sequence -> pinned to train (cannot verify they '
              f'are unseen)')
    rng.shuffle(seqs)
    want = {c: args.val_frac * sum(1 for k in keep if k[0] == c)
            for c in by_cls}
    got = collections.Counter()
    val_seqs = set()
    for s in seqs:
        cnt = collections.Counter(c for c, _, _ in seq_items[s])
        # take the sequence into val while any class it carries is short
        if any(got[c] < want[c] for c in cnt):
            val_seqs.add(s)
            got.update(cnt)

    counts = collections.Counter()
    plan = []
    for s, lst in seq_items.items():
        split = 'val' if s in val_seqs else 'train'
        for cls, path, fname in lst:
            plan.append((split, cls, path, fname))
            counts[f'{split}/{cls}'] += 1

    print('\nresulting split (whole sequences, never straddling):')
    for k in sorted(counts):
        print(f'  {k:<22} {counts[k]:,}')
    tr = sum(v for k, v in counts.items() if k.startswith('train/'))
    va = sum(v for k, v in counts.items() if k.startswith('val/'))
    print(f'  val fraction {va / max(tr + va, 1):.3f} (target {args.val_frac})')
    # assert rather than assert-in-prose: the whole point of the tool
    tr_seqs = {s for s in seq_items if s not in val_seqs}
    shared = tr_seqs & val_seqs
    print(f'  train sequences {len(tr_seqs):,} | val sequences '
          f'{len(val_seqs):,} | shared {len(shared)}')
    if shared:
        raise SystemExit(f'BUG: {len(shared)} sequences straddle the split')

    if not args.execute:
        print('\nnothing written. re-run with --execute')
        return 0

    for split, cls, path, fname in plan:
        d = os.path.join(args.out, split, cls)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(path, os.path.join(d, fname))
    man = {
        'src': args.src,
        'extra_negatives': args.extra_negatives,
        'extra_positives': args.extra_positives,
        'max_per_sequence': args.max_per_sequence,
        'hamming': args.hamming,
        # Which crops were withheld for accepting the model, and how the
        # duplicates were found. Without these two the manifest cannot answer
        # "was this model tested on data it trained on?", which is the exact
        # question dogbin_v3's manifest could not answer.
        'acceptance_set_file': args.exclude_ids or None,
        'acceptance_set_ids': len(held),
        'acceptance_set_crops_removed': n_held_removed,
        'dup_clusters_file': args.dup_clusters,
        'val_frac': args.val_frac,
        'seed': args.seed,
        'source_crops_before_acceptance_removal': n_before_hold,
        'source_crops': len(items),
        'near_duplicates_collapsed': dict(dropped_dup),
        'over_sequence_cap': dict(dropped_cap),
        'kept': len(keep),
        'counts': dict(counts),
        'train_sequences': len(seq_items) - len(val_seqs),
        'val_sequences': len(val_seqs),
        'sequences_resolved': len(seq),
        'sequences_unresolved': len(ids) - len(seq),
    }
    with open(os.path.join(args.out, 'rebuild_manifest.json'), 'w') as f:
        json.dump(man, f, indent=2)
    print(f'\nwrote {len(plan):,} crops -> {args.out}')
    print(f'manifest: {os.path.join(args.out, "rebuild_manifest.json")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
