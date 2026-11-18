#!/usr/bin/env python3
"""Build dogdet_v3: the detector dataset, re-split by sequence and fed the
confirmed near-misses.

    python tools/detect/build_dogdet_v3.py --out <dogs_detection>/dogdet_v3

WHY A REBUILD. Measured on dogdet_v2 before this was written: 63.1% of val
frames (309/490) sit in a sequence that also appears in train. Mapillary
frames come one second apart down one road, so those val frames are near
duplicates of training frames -- the same defect measured at 70.8% in
leash_binary_v1, now in the detector's own split. The retrain this feeds is
being selected on RECALL, and recall read off near-duplicates of the training
set is not a measurement.

WHAT GOES IN, beyond dogdet_v2:

  * hard_positives -- detections at conf as low as 0.05 that a person
    confirmed are real dogs. These are the recall frontier: dogs the current
    model very nearly missed, on the exact imagery it runs against.
  * audit verdicts of 'dog' -- the same thing found from the other end, boxes
    a person confirmed while auditing the gate. Human answers both; the
    model's own opinion is never a label.
  * box_corrections -- hand-drawn geometry overriding the detector's, applied
    wherever an added frame carries one.
  * frame labels -- whole frames a person boxed every dog in, or declared
    empty, on the dashboard's Annotate page (frame_labels.py). Any number of
    boxes, since the silence around them is the person's own; an empty one
    goes in as a background.

ONLY SINGLE-DETECTION FRAMES ARE ADDED, the same rule build_detector_negatives
applies to backgrounds, for the same reason read the other way: a frame gets
labels for the boxes a person confirmed and silence everywhere else, and
silence trains "no dog here". On a frame whose only detection is the confirmed
one, that silence is as true as the sweep could make it; on a multi-detection
frame it is a lie about the unconfirmed boxes.

THE HOLDOUT. Frames of old val whose sequences never appeared in old train
are the only frames the promoted model (train-30) provably never learned
from. They stay in val, listed in the manifest as `holdout`, so the old and
new model can be compared on ground neither trained on. Leaked old-val frames
move to TRAIN (their sequences live there already; val is the only place they
do harm).

Whole sequences move, never frames. Unresolved-sequence frames are pinned to
train -- absence of a sequence is not a sequence, and a frame that cannot
prove it is unrelated to train does not belong in val.
"""

import argparse
import glob
import json
import os
import random
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def _training_root():
    """The training repo. Env first, then the dashboard's own config -- the
    same answer gate_control resolves the promoted weights against -- so this
    works in a bare shell and in the sweep of guards alike."""
    got = os.environ.get('TRAINING_ROOT')
    if got:
        return got
    try:
        sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
        import dashboard as dash
        got = dash.training_root()
    except Exception:
        got = ''
    return got or os.path.dirname(os.path.dirname(REPO))

TRAINING_ROOT = _training_root()
# The set this build starts from. A default rather than a constant now: the
# composed build hands it a set prepared from a fresh Label Studio export, so
# the hand-drawn boxes are in every rebuild rather than only in whatever
# dogdet_v2 happened to be cut from.
V2 = os.path.join(TRAINING_ROOT, 'dogdet_v2')
HARD_POS = os.path.join(REPO, 'data', 'hard_positives', 'labels.jsonl')
CORRECTIONS = os.path.join(REPO, 'data', 'box_corrections', 'boxes.jsonl')
# The store n_det and the det-0 boxes are read from. main() sets it from
# --detect-root; unset, detect_root() resolves the pinned ledger root.
DETECT_ROOT = None


def detect_root():
    """The store the confirmations were made on: --detect-root, else the
    pinned ledger root (crop_sources.default_detect_root), NEVER the live one.

    A confirmed positive is an image_id a person called a dog on a gen-1
    crop, and its label box is that store's det 0 (or a correction recorded
    at gen-1 det 0). Read off the live store after the switch, a frame the
    gen-1 detector saw several boxes in can turn up single in gen 2, and an
    unreviewed gen-2 det 0 becomes the dog label with silence everywhere
    else; a gen-1 correction lands on whatever gen 2 numbered 0.
    """
    global DETECT_ROOT
    if not DETECT_ROOT:
        sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
        import crop_sources
        DETECT_ROOT = crop_sources.default_detect_root()
    return DETECT_ROOT


def _grid_roots():
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    import dashboard as dash
    return dash._grid_roots()


def resolve_sequences(ids, con):
    """{image_id: sequence} off the harvest manifests, NULLs dropped.

    min() per id, not any_value: a duplicated frame must resolve to the same
    sequence on every run or the split moves under the seed.
    """
    mans = []
    for root in _grid_roots().values():
        mans += glob.glob(os.path.join(root, '*', 'all_data_*.parquet'))
    con.execute("CREATE TEMP TABLE need(image_id VARCHAR)")
    con.executemany("INSERT INTO need VALUES (?)", [(i,) for i in ids])
    con.execute(f"""
        CREATE TEMP TABLE seqs AS
        SELECT image_id, min(seq) AS seq FROM (
            SELECT CAST(m.image_id AS VARCHAR) image_id,
                   CAST(m."sequence" AS VARCHAR) seq
            FROM read_parquet({mans!r}, union_by_name=true) m
            SEMI JOIN need n ON n.image_id = CAST(m.image_id AS VARCHAR)
            WHERE m."sequence" IS NOT NULL) GROUP BY 1""")
    return dict(con.execute("SELECT image_id, seq FROM seqs").fetchall())


def frame_meta(ids, con):
    """{image_id: (n_det, orig_w, orig_h, cell, drive)} from the sweep store."""
    sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
    import store
    root = detect_root()
    img = store._sql_src(store._store_globs(root, 'img'))
    det = store._sql_src(store._store_globs(root, 'det'))
    con.execute("CREATE TEMP TABLE want(image_id VARCHAR)")
    con.executemany("INSERT INTO want VALUES (?)", [(i,) for i in ids])
    meta = {}
    for iid, n, w, h, cell, drive in con.execute(f"""
            SELECT CAST(i.image_id AS VARCHAR), any_value(i.n_det),
                   any_value(i.orig_w), any_value(i.orig_h),
                   any_value(i.cell), any_value(i.drive)
            FROM {img} i SEMI JOIN want t
              ON t.image_id = CAST(i.image_id AS VARCHAR)
            GROUP BY 1""").fetchall():
        meta[iid] = (int(n or 0), int(w or 0), int(h or 0), cell, drive)
    boxes = {}
    for iid, di, x1, y1, x2, y2 in con.execute(f"""
            SELECT CAST(d.image_id AS VARCHAR), d.det_idx,
                   any_value(d.x1), any_value(d.y1),
                   any_value(d.x2), any_value(d.y2)
            FROM {det} d SEMI JOIN want t
              ON t.image_id = CAST(d.image_id AS VARCHAR)
            GROUP BY 1, 2""").fetchall():
        boxes[(iid, int(di))] = (float(x1), float(y1), float(x2), float(y2))
    return meta, boxes


def _store_gens(root):
    """{gen} of the shards a detector store holds (shards/gen=NNNN)."""
    try:
        names = os.listdir(os.path.join(root, 'shards'))
    except OSError:
        return set()
    return {int(n[4:]) for n in names
            if n.startswith('gen=') and n[4:].isdigit()}


def leash_dirs(root):
    """The leash stage dirs whose verdicts were made on `root`'s boxes.

    data/leash, and any gen-1 record renamed out of its way at the switch
    (data/leash_gen1_<when>). A dir counts when its plan.json names this
    store, or -- a plan without detect_root, or no plan at all, being a gen-1
    record -- when its generation is the store's only one. So a gen-1 build
    never reads gen-2 calls against gen-1 det 0, nor the other way round.
    """
    want = os.path.realpath(root)
    gens = _store_gens(root)
    out = []
    for d in ([os.path.join(REPO, 'data', 'leash')]
              + sorted(glob.glob(os.path.join(REPO, 'data', 'leash_gen*')))):
        if not os.path.isdir(d):
            continue
        try:
            with open(os.path.join(d, 'plan.json')) as fh:
                plan = json.load(fh)
        except FileNotFoundError:
            plan = {}
        except (OSError, ValueError) as e:
            raise SystemExit('%s/plan.json cannot be read (%s): which store '
                             'its leash calls were made on is unknown'
                             % (d, e))
        if plan.get('detect_root'):
            ok = os.path.realpath(plan['detect_root']) == want
        else:
            ok = gens == {int(plan.get('gen') or 1)}
        if ok:
            out.append(d)
    return out


def leashed_ids(ids, con):
    """The ids the leash stage calls leashed, off the shards of the leash
    dirs that judged this build's store (leash_dirs).

    Model output, used only to leave frames OUT: an addition carries the
    sweep's dog-only box, and a leashed dog is boxed with its leash
    everywhere a person drew one. Detection 0 only, because an addition is a
    single-detection frame -- of the store frame_meta read, which is why the
    calls must be that store's too: gen-2 det 0 is not gen-1 det 0.
    """
    # the stage's shards only: work.parquet sits beside them, and an archive
    # of a re-run sits below them
    dirs = leash_dirs(detect_root())
    shards = sorted(p for d in dirs
                    for p in glob.glob(os.path.join(d, 'leash-*.parquet')))
    print(f'leash calls for {detect_root()}: {len(shards)} shard(s) in '
          f'{[os.path.relpath(d, REPO) for d in dirs] or "no leash dir"}',
          flush=True)
    if not shards or not ids:
        return set()
    con.execute("CREATE OR REPLACE TEMP TABLE lwant(image_id VARCHAR)")
    con.executemany("INSERT INTO lwant VALUES (?)", [(i,) for i in ids])
    return {r[0] for r in con.execute(f"""
        SELECT DISTINCT CAST(l.image_id AS VARCHAR)
        FROM read_parquet({shards!r}, union_by_name=true) l
        SEMI JOIN lwant w ON w.image_id = CAST(l.image_id AS VARCHAR)
        WHERE l.det_idx = 0 AND l.label = 'leashed'""").fetchall()}


def ledger_gen():
    """The detector generation of detect_root() -- the only one whose human
    ledger rows this build reads (crop_sources.join_gen). A row's `gen`
    (absent: gen 1) names the detector its det_idx and conf belong to, and
    the det-0 label box and n_det are read off detect_root()."""
    sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
    import crop_sources
    return crop_sources.join_gen(detect_root())


def confirmed_positive_ids():
    """Frames a HUMAN said contain a real dog, from both ledgers -- of
    detect_root()'s detector generation only (ledger_gen).

    A hard positive tagged with another generation was confirmed on another
    detector's box, and is skipped, and said so. The gate audit is read from
    THIS generation's audit dir (crop_sources.audit_ctx_for: its pool cut
    from detect_root() itself), never "the stage's audit" -- so a build on
    the gen-1 store keeps every gen-1 confirmation after the audit page has
    moved on to gen 2, and the other generations' verdicts are counted and
    left out. Image-level ("this frame has a dog"), so a generation has no
    box to put on the wrong one here; it is read per generation anyway,
    because which frames a person looked at is a fact about one pool.
    """
    sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
    import fn_audit as fa
    import crop_sources
    gen = ledger_gen()
    out = set()
    other = 0
    try:
        for line in open(HARD_POS):
            if line.strip():
                d = json.loads(line)
                if d.get('label') == 'true_positive':
                    if fa.gen_of(d.get(fa.GEN_FIELD)) != gen:
                        other += 1
                        continue
                    out.add(str(d['image_id']))
    except OSError:
        pass
    if other:
        print(f'{other} hard positive(s) of another detector generation '
              f'skipped (the store is gen {gen})', flush=True)
    others = crop_sources.other_gen_verdicts('gate', gen)
    if others:
        print('gate audit verdicts of other detector generations left out '
              '(the store is gen %d): %s' % (gen, ', '.join(
                  f'gen {g}: {n if n >= 0 else "unreadable"}'
                  for g, n in sorted(others.items()))), flush=True)
    # A ledger that cannot be read, or holds a foreign row, stops the build
    # (SystemExit / GenError) -- it used to be swallowed here, which read as
    # "nobody confirmed anything".
    ac = crop_sources.audit_ctx_for('gate', detect_root())
    if ac is None:
        print(f'no gen-{gen} gate audit: no audit confirmations', flush=True)
        return out
    for v in fa.read_verdicts(stage='gate', gen=ac.gen):
        if fa.verdict_of(v.get('verdict'), 'gate') == 'dog':
            out.add(str(v['key']).split('#')[0])
    return out


def corrections():
    """{(image_id, det_idx): box} from the box-corrections ledger, last write
    wins -- the records of detect_root()'s generation only (ledger_gen): a
    redraw keyed by another detector's det_idx would move a box nobody
    redrew.

    An audit page's redraw (`crop` 'audit:<stage>') of gen >= 2 must also
    name that generation's pool (`pool` == fn_audit.pool_meta pool_id): a
    second pool of one generation would key the same det_idx, and a redraw
    is of one pool's box. And that pool must have been cut from detect_root()
    itself (realpath of its pool.json detect_root -- the rule
    crop_sources.audit_ctx_for applies to the confirmations): a generation
    number names a detector, not a store, and a redraw of another store's
    det_idx would move a box nobody redrew. Rows that do not are skipped and
    counted."""
    sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
    import fn_audit as fa
    gen = ledger_gen()
    out = {}
    other = other_pool = 0
    pools = {}

    def pool_of(stage):
        if stage not in pools:
            try:
                meta = fa.pool_meta(stage, gen)
                cut = meta.get('detect_root')
                pools[stage] = (meta['pool_id'] if cut and os.path.realpath(
                    cut) == os.path.realpath(detect_root()) else None)
            except (fa.PoolIdentityError, KeyError):
                pools[stage] = None
        return pools[stage]
    try:
        for line in open(CORRECTIONS):
            if line.strip():
                d = json.loads(line)
                if fa.gen_of(d.get(fa.GEN_FIELD)) != gen:
                    other += 1
                    continue
                crop = str(d.get('crop') or '')
                if gen >= 2 and crop.startswith('audit:'):
                    want = pool_of(crop[len('audit:'):])
                    if want is None or d.get(fa.POOL_FIELD) != want:
                        other_pool += 1
                        continue
                out[(str(d['image_id']), int(d.get('det_idx') or 0))] = (
                    float(d['x1']), float(d['y1']),
                    float(d['x2']), float(d['y2']))
    except OSError:
        pass
    if other:
        print(f'{other} box correction(s) of another detector generation '
              f'skipped (the store is gen {gen})', flush=True)
    if other_pool:
        print(f'{other_pool} gen-{gen} audit redraw(s) of no current gen-'
              f'{gen} pool cut from {detect_root()} skipped', flush=True)
    return out


def is_reset(fix, det):
    """Whether a "correction" is really the reviewer pressing Reset box.

    Reset writes the detector's own box back into the ledger, which is the
    right thing for it to do -- the ledger is append-only and last-wins, so
    withdrawing a hand-drawn box has to be another line rather than a deletion.
    But it means a record's existence is not evidence that anybody drew
    anything, and counting one as a correction overstates how much of this
    dataset carries human geometry. The training box is the same either way;
    only the count is wrong.

    Sub-pixel rather than exact: a box drawn within half a pixel of the
    detector's is not a correction anyone could see.
    """
    if fix is None or det is None:
        return False
    return all(abs(f - d) < 0.5 for f, d in zip(fix, det))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=os.path.join(TRAINING_ROOT,
                                                  'dogdet_v3'))
    ap.add_argument('--src', default=None,
                    help='the detector set to start from (default: '
                         'dogdet_v2 at the training root). The composed '
                         'build passes one prepared from a fresh Label '
                         'Studio export.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--val-target', type=int, default=480,
                    help='promote whole train sequences to val until val '
                         'reaches roughly this many frames')
    ap.add_argument('--memory', default='6GB')
    ap.add_argument('--hidden-dogs', default=None,
                    help='the hidden-dog store (default: '
                         'data/hidden_dogs/flagged.json). A confirmed '
                         'positive it flags is left out: an independent '
                         'model sees a second dog the one-box label would '
                         'teach as "not a dog"')
    ap.add_argument('--no-hidden-dogs', action='store_true',
                    help='ignore the hidden-dog store')
    ap.add_argument('--leashed-additions', choices=('keep', 'drop'),
                    default='keep',
                    help='drop: leave out every confirmed positive the leash '
                         'stage calls leashed. Their boxes are the sweep\'s, '
                         'drawn round the dog only, where a leashed dog in '
                         'Label Studio is boxed with its leash -- the mixed '
                         'convention that teaches the model both boxes')
    ap.add_argument('--detect-root', default=None,
                    help='the store n_det and the det-0 label boxes are read '
                         'from (default: the pinned ledger root, '
                         'data/ledger_detect_root.txt). The confirmations and '
                         'box corrections are gen-1, so another store reads '
                         'them against boxes nobody confirmed, and one whose '
                         'shards are not only gen=0001 is refused unless '
                         '--any-generation')
    ap.add_argument('--any-generation', action='store_true',
                    help='accept a --detect-root that holds another detector '
                         'generation. Only for confirmations and corrections '
                         'recorded against THAT store: every existing one '
                         'carries gen-1 det_idx. Either way only rows tagged '
                         'with the store\'s own generation (`gen`; untagged '
                         '= 1) are read.')
    a = ap.parse_args(argv)
    global DETECT_ROOT
    DETECT_ROOT = None
    if a.detect_root:
        if not os.path.isdir(os.path.join(a.detect_root, 'shards')):
            raise SystemExit('%s has no shards/ -- it is not a detector '
                             'store' % (a.detect_root,))
        sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
        import crop_sources
        why = crop_sources.ledger_store_refusal(a.detect_root)
        if why and not a.any_generation:
            raise SystemExit('--detect-root refused: %s. Pass '
                             '--any-generation only if the confirmations '
                             'were recorded against this store.' % why)
        DETECT_ROOT = os.path.abspath(a.detect_root)
    if a.src:
        # Module-level because the reading and writing below reach for it by
        # name in half a dozen places; rebinding it here is what makes one
        # source swappable without threading it through all of them.
        global V2
        V2 = a.src
    for split in ('train', 'val'):
        if not os.path.isdir(os.path.join(V2, 'images', split)):
            raise SystemExit('%s has no images/%s — it is not a detector '
                             'dataset to build from' % (V2, split))
    import duckdb
    rng = random.Random(a.seed)

    train_ids = sorted(f[:-4] for f in os.listdir(f'{V2}/images/train'))
    val_ids = sorted(f[:-4] for f in os.listdir(f'{V2}/images/val'))
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{a.memory}'")
    con.execute("SET preserve_insertion_order=false")

    print(f'{len(train_ids)} train + {len(val_ids)} val in dogdet_v2',
          flush=True)
    adds = confirmed_positive_ids()
    have = set(train_ids) | set(val_ids)
    adds -= have
    sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
    import frame_labels
    framed = frame_labels.read()
    # a test set's frames are for measuring, never for training; any other
    # frame of their sequences is held out below, once sequences are known
    test = frame_labels.test_sets()
    test_out = {'frame_labels': sorted(i for i in framed if i in test['ids']),
                'additions': sorted(i for i in adds if i in test['ids'])}
    framed = {i: r for i, r in framed.items() if i not in test['ids']}
    adds -= test['ids']
    # a whole-frame label says more than one confirmed box on the same frame
    adds -= set(framed)
    print(f'{len(adds)} confirmed-positive frames not yet in the dataset, '
          f'{len(framed)} whole-frame labels on record', flush=True)

    seq = resolve_sequences(sorted(have | adds | set(framed)), con)
    tr_seq = {seq[i] for i in train_ids if i in seq}

    # ── the split ───────────────────────────────────────────────────────────
    holdout = [i for i in val_ids
               if i in seq and seq[i] not in tr_seq]
    leaked = [i for i in val_ids if i in seq and seq[i] in tr_seq]
    unresolved_val = [i for i in val_ids if i not in seq]
    print(f'holdout (train-30-blind val): {len(holdout)}  ·  '
          f'leaked val -> train: {len(leaked)}  ·  '
          f'unresolved val -> train: {len(unresolved_val)}', flush=True)

    new_train = set(train_ids) | set(leaked) | set(unresolved_val)
    # a test-set frame already in the hand-labelled source is not trained on
    # either: it leaves train (val only measures, so it may stay there)
    base_in_test = sorted(new_train & test['ids'])
    new_train -= test['ids']
    new_val = set(holdout)
    val_seqs = {seq[i] for i in new_val}

    # promote whole sequences until val is a real benchmark again
    by_seq = {}
    for i in sorted(new_train):
        if i in seq:
            by_seq.setdefault(seq[i], []).append(i)
    candidates = sorted(by_seq)
    rng.shuffle(candidates)
    promoted = []
    for sq in candidates:
        if len(new_val) >= a.val_target:
            break
        if sq in val_seqs:
            continue
        members = by_seq[sq]
        new_val.update(members)
        new_train.difference_update(members)
        val_seqs.add(sq)
        promoted.append(sq)
    print(f'promoted {len(promoted)} sequences '
          f'({len(new_val) - len(holdout)} frames) from train to val',
          flush=True)

    # ── the additions ───────────────────────────────────────────────────────
    meta, boxes = frame_meta(sorted(adds | set(framed)), con)
    fixes = corrections()
    roots = _grid_roots()
    added, skipped = [], {'multi_detection': 0, 'val_sequence': 0,
                          'no_meta': 0, 'no_file': 0}
    for iid in sorted(adds):
        m = meta.get(iid)
        if not m or not m[1] or not m[2]:
            skipped['no_meta'] += 1
            continue
        n_det, w, h, cell, drive = m
        if n_det != 1:
            skipped['multi_detection'] += 1
            continue
        if seq.get(iid) in val_seqs:
            skipped['val_sequence'] += 1
            continue
        if seq.get(iid) in test['seqs']:
            skipped['test_sequence'] = skipped.get('test_sequence', 0) + 1
            continue
        src = os.path.join(roots.get(drive, ''), cell,
                           'ground_animal_images', f'{iid}.jpg')
        if not os.path.exists(src):
            skipped['no_file'] += 1
            continue
        box = fixes.get((iid, 0)) or boxes.get((iid, 0))
        if not box:
            skipped['no_meta'] += 1
            continue
        added.append((iid, src, box, (w, h)))
    # ── what the additions must not teach ───────────────────────────────────
    import hidden_dogs
    hidden_path = None if a.no_hidden_dogs else (a.hidden_dogs
                                                 or hidden_dogs.STORE)
    flagged = hidden_dogs.read(hidden_path) if hidden_path else {}
    added, hidden_out = hidden_dogs.screen(
        added,
        {i: r for i, r in flagged.items() if r.get('role') == 'addition'},
        key=lambda x: x[0])
    skipped['hidden_dog'] = len(hidden_out)
    leashed_out = []
    if a.leashed_additions == 'drop':
        leashed = leashed_ids([i for i, *_ in added], con)
        leashed_out = [x for x in added if x[0] in leashed]
        added = [x for x in added if x[0] not in leashed]
    skipped['leashed'] = len(leashed_out)
    print(f'adding {len(added)} frames to train, skipped {skipped}',
          flush=True)
    where = {}
    for iid in framed:
        m = meta.get(iid)
        if m and m[3] and m[4] in roots:
            where[iid] = os.path.join(roots[m[4]], m[3],
                                      'ground_animal_images', f'{iid}.jpg')
    in_test_seq = sorted(i for i in framed
                         if seq.get(i) is not None
                         and seq.get(i) in test['seqs'])
    framed = {i: r for i, r in framed.items() if i not in set(in_test_seq)}
    framed_added, framed_skipped = frame_labels.plan(
        framed, seq, val_seqs, have, where,
        size_of=frame_labels.image_size)
    framed_skipped['test_set'] = len(test_out['frame_labels'])
    framed_skipped['test_sequence'] = len(in_test_seq)
    print(f'adding {len(framed_added)} whole-frame labels to train '
          f'({sum(1 for *_, t in framed_added if not t)} empty), skipped '
          f'{framed_skipped}', flush=True)

    # ── write it ────────────────────────────────────────────────────────────
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    def put(src_img, src_lbl, split, iid):
        di = os.path.join(a.out, 'images', split, f'{iid}.jpg')
        dl = os.path.join(a.out, 'labels', split, f'{iid}.txt')
        for s, d in ((src_img, di), (src_lbl, dl)):
            if os.path.exists(d):
                os.remove(d)
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)

    old_split = {i: 'train' for i in train_ids}
    old_split.update({i: 'val' for i in val_ids})
    n_bg = {'train': 0, 'val': 0}
    for iid in sorted(new_train | new_val):
        split = 'train' if iid in new_train else 'val'
        was = old_split[iid]
        put(os.path.join(V2, 'images', was, f'{iid}.jpg'),
            os.path.join(V2, 'labels', was, f'{iid}.txt'), split, iid)
        if os.path.getsize(os.path.join(V2, 'labels', was,
                                        f'{iid}.txt')) == 0:
            n_bg[split] += 1

    for iid, src, (x1, y1, x2, y2), (w, h) in added:
        dst = os.path.join(a.out, 'images', 'train', f'{iid}.jpg')
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        with open(os.path.join(a.out, 'labels', 'train',
                               f'{iid}.txt'), 'w') as fh:
            fh.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')

    for iid, src, text in framed_added:
        dst = os.path.join(a.out, 'images', 'train', f'{iid}.jpg')
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        with open(os.path.join(a.out, 'labels', 'train',
                               f'{iid}.txt'), 'w') as fh:
            fh.write(text)

    with open(os.path.join(a.out, 'dataset.yaml'), 'w') as fh:
        fh.write(f'path: {a.out}\ntrain: images/train\nval: images/val\n\n'
                 f'names:\n  0: target\n')

    n_train = len(new_train) + len(added) + len(framed_added)
    n_bg['train'] += sum(1 for *_, t in framed_added if not t)
    manifest = {
        'src': V2, 'seed': a.seed,
        # the store n_det and the det-0 boxes came from
        'detect_root': DETECT_ROOT,
        'train': n_train, 'val': len(new_val),
        'backgrounds_train': n_bg['train'], 'backgrounds_val': n_bg['val'],
        'holdout': sorted(holdout),
        'leaked_val_moved_to_train': len(leaked),
        'unresolved_pinned_to_train': len(unresolved_val),
        'promoted_sequences': len(promoted),
        'added_confirmed_positives': len(added),
        'added_ids': sorted(i for i, *_ in added),
        'additions_skipped': skipped,
        'hidden_dogs': {
            'store': (os.path.relpath(hidden_path, REPO)
                      if hidden_path else None),
            'flagged_frames': len(flagged),
            'additions_left_out': sorted(x[0] for x in hidden_out),
        },
        'leashed_additions': {
            'mode': a.leashed_additions,
            'left_out': sorted(x[0] for x in leashed_out),
        },
        'corrections_applied': sum(
            1 for i, *_ in added
            if (i, 0) in fixes
            and not is_reset(fixes[(i, 0)], boxes.get((i, 0)))),
        'added_frame_labels': len(framed_added),
        'added_frame_label_ids': sorted(i for i, *_ in framed_added),
        'frame_labels_empty': sum(1 for *_, t in framed_added if not t),
        'frame_labels_skipped': framed_skipped,
        # What was held out for the test sets, and the hand-labelled frames
        # of the base set that share a sequence with one -- those were in
        # the source before the test set existed and stay; the scorer leaves
        # the test frames of those sequences out instead.
        'test_sets': {
            'batches': test['batches'],
            'frames': len(test['ids']),
            'frame_labels_left_out': test_out['frame_labels'],
            'additions_left_out': test_out['additions'],
            'additions_in_a_test_sequence': skipped.get('test_sequence', 0),
            'frame_labels_in_a_test_sequence': in_test_seq,
            'base_frames_in_a_test_set_left_out_of_train': base_in_test,
            'base_frames_sharing_a_test_sequence': sorted(
                i for i in new_train | new_val
                if seq.get(i) is not None and seq.get(i) in test['seqs']),
        },
        'sequences': {i: seq.get(i) for i in sorted(
            new_train | new_val | {x[0] for x in added}
            | {x[0] for x in framed_added})},
    }
    with open(os.path.join(a.out, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=1)
    print(f"dogdet_v3: {n_train} train ({n_bg['train']} bg) / "
          f"{len(new_val)} val ({n_bg['val']} bg), "
          f"holdout {len(holdout)}, +{len(added)} confirmed positives "
          f"({manifest['corrections_applied']} with corrected boxes), "
          f"+{len(framed_added)} whole-frame labels")
    return 0


if __name__ == '__main__':
    sys.exit(main())
