#!/usr/bin/env python3
"""
Split and de-duplicate a crop-classifier dataset, and prove the split is clean.

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

And a third, measured on dogbin_20261012 on 2026-11-14:

3. **Crops nobody could place.** The filename parser could not read a Label
   Studio crop's name, and a crop it could not read became a "sequence" of
   its own (``nofile:<name>``). 2,933 crops went that way: all pinned to
   train, all outside the per-sequence cap (one sequence held 114 crops of a
   class) and all invisible to the straddle check -- which read the same
   parser, so it could not see them straddle by construction. 6.77% of that
   dataset's val shared a sequence with its train.

TWO WAYS IN.

``--manifest crops.jsonl`` (what build_dataset.py uses). Every crop arrives as
a manifest row from crop_sources.py -- image_id, sequence, source, provenance
-- so nothing is inferred from a filename. Then, in this order:

* drop every row whose image_id OR sequence is reserved (``--reserve``), and
  every row that is a copy of a reserved crop under another name: the same
  bytes, or a dHash within max(2, --hamming) of one -- the reserved rows'
  own, and the crops a reserve file records. Their hashes are written to
  ``reserved_crops.json`` so the on-disk check can repeat it;
* drop every row whose sequence no manifest knows, unless ``--keep-noseq``
  pins them to train on the record: such a row cannot be held disjoint from
  a reserved sequence, so by default it is not trained on at all;
* de-duplicate ACROSS sources (see ``dedup_rows``): the same box, the same
  bytes, the same picture by dHash or by embedding -- conflicting labels for
  the same box drop the box, and are written down;
* cap the crops kept per (sequence, class);
* assign whole SPLIT GROUPS to train/val -- a group is every sequence joined
  by a shared image, a shared file or a near-duplicate, so a dog seen in two
  passes cannot sit on both sides;
* assert all of it on the plan (``check_rows``), write, and assert it again on
  what is on disk (``verify_dataset``). A failure removes the output.

``--src DIR`` (legacy). Re-split an existing dataset by filename, as before --
but a filename the parser cannot read now FAILS the run instead of quietly
becoming its own sequence.

Every count is written to ``rebuild_manifest.json``, including what was
dropped and why -- a dataset that silently shrinks is worse than one that
never changed.

    python tools/detect/rebuild_crop_dataset.py \\
        --manifest <stage>/crops.jsonl --out <datasets>/dogbin_x \\
        --reserve data/dogbin_acceptance_set.json \\
        --reserve data/dogbin_test_v1.json --execute
    python tools/detect/rebuild_crop_dataset.py --verify-only <datasets>/dogbin_x

READ-ONLY on its inputs.
"""

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import sys

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The parser is crop_sources.parse_name -- ONE parser for every tool. The copy
# below is a fallback for an interpreter that cannot import crop_sources, and
# tests/adv_crop_dataset_integrity.py holds the two to the same answers. The
# old NAME_RE here, '^(?:flag_|pos_)?(?:le|un|no|nd)?_?(\d{6,})[_.]', could not
# read 'pos_ls_<id>_<k>.jpg' and read '<13-digit ts>_<id>_<conf>.jpg' as an
# image_id equal to the timestamp.
_UID_RE = re.compile(r'^(ls|hn|hp|ga|la|lc|fl)_(\d{6,})_([db]\d+)\.jpg$')
_LEGACY_RE = re.compile(
    r'^(flag_|pos_)?(ls_|le_|un_|no_|nd_)?(\d{6,})_(?:[db])?(\d+)\.jpg$')
_TS_LED_RE = re.compile(r'^\d{13}_')


def _parse_name_fallback(fname):
    base = os.path.basename(str(fname))
    m = _UID_RE.match(base)
    if m:
        return m.group(2), m.group(3)
    if _TS_LED_RE.match(base):
        return None
    m = _LEGACY_RE.match(base)
    if not m:
        return None
    return m.group(3), (None if m.group(2) else 'd' + m.group(4))


try:
    from crop_sources import (parse_name, HUMAN_SOURCES, FORBIDDEN_PATH_RE,
                              FRAMINGS, DRAWN_SOURCES)
except Exception:                         # pragma: no cover - see above
    parse_name = _parse_name_fallback
    HUMAN_SOURCES = frozenset({'ls_box', 'hard_negative', 'hard_positive',
                               'gate_audit', 'leash_audit', 'leash_call',
                               'frame_label'})
    DRAWN_SOURCES = frozenset({'ls_box', 'frame_label'})
    FORBIDDEN_PATH_RE = re.compile(
        r'(?:^|/)data/(?:gate(?:_[^/]*)?|leash)(?:/|$)'
        r'|llm_[a]nnotations|tri[a]ge|guess|mistakes|hidden_dogs|confusion')
    FRAMINGS = frozenset({'cut_crops', 'audit_full', 'audit_edited'})


def image_id_of(fname):
    got = parse_name(fname)
    return got[0] if got else None


def straddling_sequences(plan, seq_of):
    """(train seqs, val seqs, both) read back off the PLAN that gets written.

    The leak this whole tool exists to prevent, asserted rather than asserted
    in prose. Both sides come from the emitted rows and each row's sequence is
    re-derived from the filename about to be written, so an edit that assigns
    a split per CROP rather than per sequence -- or a seq_of() that answers
    differently here than it did where the per-sequence cap was applied --
    puts one sequence on both sides and this returns it.

    What it replaces was `tr = {s for s in seq_items if s not in val_seqs}`
    intersected with val_seqs: the complement of a set, met with the set. That
    is empty for every possible input. It printed "shared 0" under every
    dataset it ever built and could not have printed anything else -- a green
    check beside a property nobody had measured.

    Args:
        plan: [(split, cls, path, fname)] -- exactly what main() writes.
        seq_of: fname -> sequence key, the same callable the cap used.
    """
    side = collections.defaultdict(set)
    for split, _cls, _path, fname in plan:
        side[seq_of(fname)].add(split)
    tr = {s for s, v in side.items() if 'train' in v}
    va = {s for s, v in side.items() if 'val' in v}
    return tr, va, tr & va


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
# The machine is shared with a long-running job: stay well under the RAM.
con.execute("SET memory_limit='12GB'")
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


def _popcount(a):
    import numpy as np
    if hasattr(np, 'bitwise_count'):
        return np.bitwise_count(a)
    return np.unpackbits(a.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def cluster(hashes, max_dist):
    """Greedy near-duplicate clustering. hashes: [(key, hash)] -> [[key,...]]

    Each hash joins the FIRST representative within max_dist, else becomes a
    representative itself -- the same greedy rule as always, vectorised: the
    pure-Python loop was quadratic in the class size, and with every audit
    verdict in the build a class is 25k crops, which is 300M popcounts.
    """
    if max_dist <= 0:
        return [[k] for k, _ in hashes]
    import numpy as np
    reps = np.zeros(max(1, len(hashes)), dtype=np.uint64)
    rep_group = []
    groups = []
    for key, h in hashes:
        if h is None:
            groups.append([key])
            continue
        n = len(rep_group)
        if n:
            d = _popcount(reps[:n] ^ np.uint64(h))
            hit = np.flatnonzero(d <= max_dist)
            if hit.size:
                groups[rep_group[int(hit[0])]].append(key)
                continue
        reps[n] = np.uint64(h)
        rep_group.append(len(groups))
        groups.append([key])
    return groups


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: the smaller key is the root
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ── reserves ────────────────────────────────────────────────────────────────

def load_reserves(paths, repo=REPO, python_bin=None):
    """[{path, sha256, image_ids, sequences}] -- what may never be trained on.

    BY SEQUENCE as well as image_id. A reservation drawn by image_id alone
    still lets frame 8 of a sequence train while frame 7 is in the test set:
    one gate train crop of dogbin_20261012 shares a sequence with the dogbin
    acceptance set. A file that lists its sequences is taken at its word;
    one that does not (the acceptance sets predate the field) has them
    resolved now.
    """
    out = []
    for p in paths or ():
        try:
            with open(p) as fh:
                doc = json.load(fh)
        except OSError as e:
            raise SystemExit('reserve %s is unreadable: %s' % (p, e))
        except ValueError as e:
            raise SystemExit('reserve %s is not JSON: %s' % (p, e))
        ids = {str(i) for i in doc.get('image_ids') or []}
        if not ids:
            raise SystemExit('reserve %s holds no image_ids -- a reservation '
                             'that reserves nothing protects nothing' % (p,))
        seqs = doc.get('sequences')
        resolved = False
        if seqs is None:
            seqs = set(resolve_sequences(ids, repo, python_bin
                                         or sys.executable).values())
            resolved = True
        # THE CROPS THEMSELVES, where the file records them (a test set
        # drawn by reserve_test_set.py does): a re-encoded copy of a reserved
        # crop filed under another image_id and another sequence is matched by
        # nothing above, and two of them reached val in the first scratch
        # build at dHash distance 1.
        items = doc.get('items') or []
        out.append({'path': os.path.abspath(p), 'sha256': sha256_file(p),
                    'image_ids': sorted(ids), 'sequences': sorted(set(seqs)),
                    'sequences_resolved_at_build': resolved,
                    'crop_sha256': sorted({str(i['sha256']) for i in items
                                           if i.get('sha256')}),
                    'crop_dhash': sorted({int(i['dhash']) for i in items
                                          if i.get('dhash') is not None})})
    return out


def reserved_sets(reserves):
    ids, seqs = set(), set()
    for r in reserves or ():
        ids.update(r['image_ids'])
        seqs.update(r['sequences'])
    return ids, seqs


def near_hashes(rows, ref, dist):
    """uids of rows whose dHash is within `dist` bits of any hash in `ref`."""
    ref = [int(h) for h in ref if h is not None]
    if not ref:
        return set()
    import numpy as np
    arr = np.array(ref, dtype=np.uint64)
    out = set()
    for r in rows:
        h = r.get('dhash')
        if h is None:
            continue
        if int(_popcount(arr ^ np.uint64(int(h))).min()) <= dist:
            out.add(r['uid'])
    return out


# ── de-duplication across sources ───────────────────────────────────────────

def _rank(row):
    """Which of several agreeing rows for ONE box survives.

    A redrawn box first (a person framed it), then a drawn box, then the
    detector's box as an audit reviewer saw it, then the review queue's, then
    a leash call from the review page.
    """
    if row.get('framing') == 'audit_edited':
        return 0
    s = row.get('source')
    if s in DRAWN_SOURCES:
        return 1
    if row.get('framing') == 'audit_full':
        return 2
    if s in ('hard_negative', 'hard_positive'):
        return 3
    return 4


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _member(r):
    return {'uid': r['uid'], 'label': r['label'], 'source': r['source'],
            'image_id': r['image_id'], 'det_key': r.get('det_key')}


def dedup_rows(rows, hamming=6, emb_groups=None, iou_thr=0.5):
    """(kept rows, sequence unions, conflicts, reported, counts).

    Before the split and ACROSS SOURCES -- the old pass collapsed duplicates
    within one class of one directory, so the same box arriving from the
    audit and from the review queue was two examples, and a box arriving
    under both labels taught both. In order:

    1. THE SAME BOX. Detector boxes are keyed (image_id, det_key). A drawn box
       (Label Studio, annotate page) joins the detector box on its frame it
       overlaps at IoU >= 0.5 -- the detector's own box (pool_box), not a
       redraw of it; failing that, another drawn box it overlaps as much. If
       two or more qualify, nobody can say which box it was and the whole
       image is dropped. Agreeing labels keep one row (_rank); disagreeing
       labels drop every member and are written to conflicts.
    2. THE SAME BYTES. sha256 equal: same label keeps one, and the
       sequences of all of them become one split group; different labels
       drop all.
    3. THE SAME PICTURE BY dHASH. Distance 0 across classes drops all of
       them, and so does distance <= CROSS_NEAR when the two crops are also
       the same size; within a class, distance <= `hamming` collapses to the
       largest file, and the sequences are joined for the split.
    4. EMBEDDING CLUSTERS (from dedup_crops.py). Every cluster joins its
       sequences into one split group; same-class members collapse; a
       cluster holding both labels is REPORTED, not dropped -- looking alike
       is not proof of being one box.

    `unions` is [(sequence a, sequence b)]: the split must keep each pair on
    one side.
    """
    counts = collections.Counter()
    conflicts, reported, unions = [], [], []
    rows = [r for r in rows]

    # 1. same box
    det = collections.defaultdict(list)
    for r in rows:
        if str(r.get('det_key') or '').startswith('d'):
            det[(r['image_id'], r['det_key'])].append(r)
    # The box a drawn box is joined against is the DETECTOR's, never a
    # person's redraw of it: 92% of the leash audit's redraws enlarge the box
    # to take in the leash, and a Label Studio box at IoU 0.97 with detector
    # box d0 sits at 0.41 with its redraw -- the join missed it, and both
    # rows trained (and would have, under two labels).
    det_boxes = collections.defaultdict(list)      # image -> [(key, box)]
    for (iid, dk), lst in det.items():
        box = next((x['pool_box'] for x in lst if x.get('pool_box')), None)
        if box is None:
            box = next((x['box'] for x in lst if x.get('box')
                        and x.get('framing') != 'audit_edited'), None)
        if box:
            det_boxes[iid].append(((iid, dk), box))
        else:
            counts['detector_box_unknown'] += 1
    groups = {k: list(v) for k, v in det.items()}
    ambiguous_images = set()
    drawn_boxes = collections.defaultdict(list)    # image -> [(key, box)]
    drawn = sorted((r for r in rows if r.get('source') in DRAWN_SOURCES),
                   key=lambda r: (_rank(r), r['uid']))
    for r in drawn:
        hits = [k for k, b in det_boxes.get(r['image_id'], ())
                if r.get('box') and _iou(r['box'], b) >= iou_thr]
        if len(hits) >= 2:
            ambiguous_images.add(r['image_id'])
        elif hits:
            groups[hits[0]].append(r)
            counts['drawn_joined_detector_box'] += 1
        else:
            # two DRAWN boxes of one dog -- Label Studio's and the annotate
            # page's -- are one box too. Keyed by uid, not det_key: 'b0' of
            # a Label Studio task and 'b0' of an annotate-page save are two
            # different boxes that happen to share an index.
            dh = [k for k, b in drawn_boxes.get(r['image_id'], ())
                  if r.get('box') and _iou(r['box'], b) >= iou_thr]
            if len(dh) >= 2:
                ambiguous_images.add(r['image_id'])
            elif dh:
                groups[dh[0]].append(r)
                counts['drawn_joined_drawn_box'] += 1
            else:
                key = (r['image_id'], r['uid'])
                groups[key] = [r]
                if r.get('box'):
                    drawn_boxes[r['image_id']].append((key, r['box']))
    for r in rows:
        if not r.get('det_key'):
            groups.setdefault((r['image_id'], r['uid']), []).append(r)
    if ambiguous_images:
        for iid in sorted(ambiguous_images):
            members = [r for r in rows if r['image_id'] == iid]
            conflicts.append({'kind': 'ambiguous_ls_box', 'image_id': iid,
                              'members': [_member(m) for m in members]})
            counts['dropped_ambiguous_image'] += len(members)
    kept = []
    for key in sorted(groups, key=str):
        lst = [r for r in groups[key] if r['image_id'] not in ambiguous_images]
        if not lst:
            continue
        labels = {r['label'] for r in lst}
        if len(labels) > 1:
            conflicts.append({'kind': 'same_box_two_labels',
                              'key': list(key),
                              'members': [_member(m) for m in lst]})
            counts['dropped_same_box_conflict'] += len(lst)
            continue
        lst.sort(key=lambda r: (_rank(r), r['uid']))
        kept.append(lst[0])
        counts['collapsed_same_box'] += len(lst) - 1
        for other in lst[1:]:
            unions.append((lst[0]['sequence'], other['sequence']))

    # 2. same bytes
    by_sha = collections.defaultdict(list)
    for r in kept:
        by_sha[r.get('sha256')].append(r)
    nxt = []
    for sha in sorted(by_sha, key=str):
        lst = by_sha[sha]
        if sha is None or len(lst) == 1:
            nxt += lst
            continue
        if len({r['label'] for r in lst}) > 1:
            conflicts.append({'kind': 'same_bytes_two_labels', 'sha256': sha,
                              'members': [_member(m) for m in lst]})
            counts['dropped_same_bytes_conflict'] += len(lst)
            continue
        lst.sort(key=lambda r: (_rank(r), r['uid']))
        nxt.append(lst[0])
        counts['collapsed_same_bytes'] += len(lst) - 1
        for other in lst[1:]:
            unions.append((lst[0]['sequence'], other['sequence']))
    kept = nxt

    # 3. dHash: identical across classes, near within a class
    by_dh = collections.defaultdict(list)
    for r in kept:
        if r.get('dhash') is not None:
            by_dh[r['dhash']].append(r)
    poison = set()
    for h, lst in by_dh.items():
        if len({r['label'] for r in lst}) > 1:
            conflicts.append({'kind': 'dhash_identical_two_labels',
                              'dhash': str(h),
                              'members': [_member(m) for m in lst]})
            poison.update(r['uid'] for r in lst)
    if poison:
        counts['dropped_dhash_cross_class'] += len(poison)
        kept = [r for r in kept if r['uid'] not in poison]
    # ...and NEAR-identical across classes when the crops are the same size
    # too: that is one picture re-encoded or re-cut a pixel differently, not
    # two subjects that share a gradient. The one pair this caught in the
    # first full build was a blurred figure, 188x181 'not_dog' in val and
    # 188x180 'dog' in train, dHash 3 apart, in two sequences.
    near_poison = set()
    by_lab = collections.defaultdict(list)
    for r in kept:
        if r.get('dhash') is not None:
            by_lab[r['label']].append(r)
    labs = sorted(by_lab)
    if len(labs) > 1:
        import numpy as np
        for i, la in enumerate(labs):
            for lb in labs[i + 1:]:
                other = by_lab[lb]
                arr = np.array([int(r['dhash']) for r in other],
                               dtype=np.uint64)
                for r in by_lab[la]:
                    d = _popcount(arr ^ np.uint64(int(r['dhash'])))
                    for j in np.flatnonzero(d <= CROSS_NEAR):
                        o = other[int(j)]
                        if _same_size(r, o):
                            conflicts.append({
                                'kind': 'near_copy_two_labels',
                                'dhash_distance': int(d[j]),
                                'members': [_member(r), _member(o)]})
                            near_poison.update((r['uid'], o['uid']))
    if near_poison:
        counts['dropped_near_copy_cross_class'] += len(near_poison)
        kept = [r for r in kept if r['uid'] not in near_poison]
    if hamming > 0:
        by_cls = collections.defaultdict(list)
        for r in kept:
            by_cls[r['label']].append(r)
        nxt = []
        for cls in sorted(by_cls):
            lst = sorted(by_cls[cls], key=lambda r: r['uid'])
            by_uid = {r['uid']: r for r in lst}
            for g in cluster([(r['uid'], r.get('dhash')) for r in lst],
                             hamming):
                if len(g) == 1:
                    nxt.append(by_uid[g[0]])
                    continue
                members = [by_uid[u] for u in g]
                best = max(members, key=lambda r: (_size_of(r), r['uid']))
                nxt.append(best)
                counts['collapsed_dhash'] += len(members) - 1
                for other in members:
                    if other is not best:
                        unions.append((best['sequence'], other['sequence']))
        kept = nxt

    # 4. embedding clusters
    if emb_groups:
        gid_of = {}
        for gi, g in enumerate(emb_groups):
            for p in g:
                gid_of[os.path.realpath(p)] = gi
        by_g = collections.defaultdict(list)
        rest = []
        for r in kept:
            gi = gid_of.get(os.path.realpath(r['path']))
            (by_g[gi] if gi is not None else rest).append(r)
        nxt = list(rest)
        for gi in sorted(by_g):
            lst = sorted(by_g[gi], key=lambda r: r['uid'])
            for other in lst[1:]:
                unions.append((lst[0]['sequence'], other['sequence']))
            if len({r['label'] for r in lst}) > 1:
                reported.append({'kind': 'embedding_cluster_two_labels',
                                 'members': [_member(m) for m in lst]})
            per = collections.defaultdict(list)
            for r in lst:
                per[r['label']].append(r)
            for cls in sorted(per):
                best = max(per[cls], key=lambda r: (_size_of(r), r['uid']))
                nxt.append(best)
                counts['collapsed_embedding'] += len(per[cls]) - 1
        kept = nxt
    kept.sort(key=lambda r: r['uid'])
    return kept, unions, conflicts, reported, counts


# A cross-class pair this close in dHash AND of the same size (within
# SAME_SIZE_PX on both sides) is one picture under two labels.
CROSS_NEAR = 4
SAME_SIZE_PX = 2


def _same_size(a, b):
    wa, wb = a.get('crop_wh'), b.get('crop_wh')
    if not wa or not wb:
        return False
    return (abs(int(wa[0]) - int(wb[0])) <= SAME_SIZE_PX
            and abs(int(wa[1]) - int(wb[1])) <= SAME_SIZE_PX)


def _size_of(r):
    try:
        return os.path.getsize(r['path'])
    except OSError:
        return 0


def cap_rows(rows, cap, rng):
    """At most `cap` rows per (sequence, class); deterministic for a seed."""
    per = collections.defaultdict(list)
    for r in rows:
        per[(r['sequence'], r['label'])].append(r)
    kept, dropped = [], collections.Counter()
    for k in sorted(per):
        lst = sorted(per[k], key=lambda r: r['uid'])
        rng.shuffle(lst)
        if len(lst) > cap:
            dropped[k[1]] += len(lst) - cap
            lst = lst[:cap]
        kept += lst
    kept.sort(key=lambda r: r['uid'])
    return kept, dropped


def split_groups(rows, unions):
    """{row uid: group id}. A group is every sequence joined by a shared image,
    a duplicate or a near-duplicate; its id is its smallest sequence key."""
    uf = UnionFind()
    for r in rows:
        uf.union('s:' + r['sequence'], 'i:' + r['image_id'])
    for a, b in unions:
        if a and b:
            uf.union('s:' + a, 's:' + b)
    canon = {}
    for node in list(uf.p):
        root = uf.find(node)
        if node.startswith('s:'):
            if root not in canon or node < canon[root]:
                canon[root] = node
    return {r['uid']: canon[uf.find('s:' + r['sequence'])][2:] for r in rows}


def assign_splits(rows, group_of, val_frac, rng):
    """{group id: 'train'|'val'}, whole groups, stratified per class.

    A group holding any sequence-less row is pinned to train: its real
    sequence-mates may be sitting in train under a resolved sequence, and in
    val it would leak exactly what this tool exists to prevent.
    """
    items = collections.defaultdict(list)
    for r in rows:
        items[group_of[r['uid']]].append(r)
    pinned = {g for g, lst in items.items()
              if any(r.get('seq_status') == 'noseq'
                     or str(r['sequence']).startswith('noseq:') for r in lst)}
    order = sorted(g for g in items if g not in pinned)
    rng.shuffle(order)
    want = collections.Counter()
    for r in rows:
        want[r['label']] += val_frac
    got = collections.Counter()
    split = {g: 'train' for g in pinned}
    for g in order:
        cnt = collections.Counter(r['label'] for r in items[g])
        if any(got[c] < want[c] for c in cnt):
            split[g] = 'val'
            got.update(cnt)
        else:
            split[g] = 'train'
    return split, pinned


# ── the assertions ──────────────────────────────────────────────────────────

def check_rows(rows, reserves=(), cap=None, max_noseq_frac=None,
               conflict_uids=(), framings=None, reserved_crops=None,
               allow_noseq=False):
    """Every property the dataset promises, as a list of failures.

    Run on the PLAN before anything is written and again by verify_dataset on
    what is on disk. Each one fails the build; a property asserted in prose
    is a property nobody measured (see straddling_sequences).

    rows: dicts with uid, split, label, image_id, det_key, sequence,
    seq_status, source, provenance, framing, sha256 and (optionally) group.
    """
    bad = []
    framings = framings or FRAMINGS

    def fail(what, ex):
        bad.append('%s (e.g. %s)' % (what, ex[:3]))

    # 1. nothing straddles: not a sequence, not a split group
    for field in ('sequence', 'group'):
        side = collections.defaultdict(set)
        for r in rows:
            if r.get(field) is not None:
                side[r[field]].add(r['split'])
        both = sorted(k for k, v in side.items() if len(v) > 1)
        if both:
            fail('%d %s(s) sit in more than one split' % (len(both), field),
                 both)
    # 2. nothing reserved
    ids, seqs = reserved_sets(reserves)
    hit = sorted(r['uid'] for r in rows
                 if r['image_id'] in ids or r['sequence'] in seqs)
    if hit:
        fail('%d crop(s) share an image_id or a sequence with a reserved set'
             % (len(hit),), hit)
    # 2b. ...nor its pixels under another name: the same bytes, or a dHash
    # within the distance the build dropped them at
    rc = reserved_crops or {}
    shas = set(rc.get('sha256') or ())
    for r in reserves or ():
        shas.update(r.get('crop_sha256') or ())
    dhs = list(rc.get('dhash') or ())
    for r in reserves or ():
        dhs += list(r.get('crop_dhash') or ())
    twin = sorted({r['uid'] for r in rows if r.get('sha256') in shas}
                  | near_hashes(rows, dhs, int(rc.get('hamming', 2))))
    if twin:
        fail('%d crop(s) are a copy or near-copy (dHash <= %d) of a reserved '
             'crop' % (len(twin), int(rc.get('hamming', 2))), twin)
    # 3. the cap
    if cap:
        per = collections.Counter((r['sequence'], r['label']) for r in rows)
        over = sorted(k for k, n in per.items() if n > cap)
        if over:
            fail('%d (sequence, class) group(s) hold more than %d crops'
                 % (len(over), cap), over)
    # 4. identity and provenance: human labels only
    for r in rows:
        if not r.get('image_id') or not r.get('sequence'):
            fail('a crop has no image_id or no sequence', [r.get('uid')])
            break
    nonhuman = sorted(r['uid'] for r in rows
                      if r.get('source') not in HUMAN_SOURCES)
    if nonhuman:
        fail('%d crop(s) come from a source that is not a human label'
             % (len(nonhuman),), nonhuman)
    forbidden = sorted(r['uid'] for r in rows
                       if FORBIDDEN_PATH_RE.search(str(
                           (r.get('provenance') or {}).get('ledger') or '')))
    if forbidden:
        fail("%d crop(s) carry provenance in a model's store"
             % (len(forbidden),), forbidden)
    noprov = sorted(r['uid'] for r in rows
                    if not (r.get('provenance') or {}).get('ledger'))
    if noprov:
        fail('%d crop(s) carry no provenance ledger' % (len(noprov),), noprov)
    # 5. sequence-less crops: none, unless the build said so -- then train
    # only, and few
    ns = [r for r in rows if r.get('seq_status') == 'noseq'
          or str(r.get('sequence') or '').startswith('noseq:')]
    if ns and not allow_noseq:
        fail('%d crop(s) have no resolved sequence and the build did not '
             'pass --keep-noseq' % (len(ns),), sorted(r['uid'] for r in ns))
    nv = sorted(r['uid'] for r in ns if r['split'] != 'train')
    if nv:
        fail('%d sequence-less crop(s) outside train' % (len(nv),), nv)
    if max_noseq_frac is not None and rows and \
            len(ns) / len(rows) > max_noseq_frac:
        bad.append('%.2f%% of crops have no sequence (limit %.2f%%)'
                   % (100 * len(ns) / len(rows), 100 * max_noseq_frac))
    # 6. no duplicate bytes, no box twice
    for field in ('sha256',):
        c = collections.Counter(r.get(field) for r in rows if r.get(field))
        dup = sorted(k for k, n in c.items() if n > 1)
        if dup:
            fail('%d %s value(s) appear more than once' % (len(dup), field),
                 dup)
    c = collections.Counter((r['image_id'], r.get('det_key')) for r in rows
                            if r.get('det_key'))
    dup = sorted(k for k, n in c.items() if n > 1)
    if dup:
        fail('%d (image_id, det_key) pair(s) appear more than once'
             % (len(dup),), dup)
    # 7. a conflict's members are gone
    cu = set(conflict_uids or ())
    survived = sorted(r['uid'] for r in rows if r['uid'] in cu)
    if survived:
        fail('%d member(s) of a label conflict survived' % (len(survived),),
             survived)
    # 9. framing
    wrong = sorted(r['uid'] for r in rows if r.get('framing') not in framings)
    if wrong:
        fail('%d crop(s) framed some other way than production'
             % (len(wrong),), wrong)
    return bad


def conflict_member_uids(conflicts):
    return {m['uid'] for c in conflicts or () for m in c.get('members', ())}


def verify_dataset(out, reserves=None, cap=None, max_noseq_frac=None):
    """Re-read a written dataset and assert check_rows on what is ON DISK.

    The plan can be right and the copy wrong -- a file written twice, a row
    with no file, a class directory edited by hand afterwards. So this reads
    the directory tree, recomputes every digest, and checks it against
    crops.jsonl both ways (assertion 8) before running every other check on
    the rows the files say exist. Returns a list of failures.
    """
    bad = []
    man = {}
    try:
        with open(os.path.join(out, 'rebuild_manifest.json')) as fh:
            man = json.load(fh)
    except (OSError, ValueError):
        bad.append('no readable rebuild_manifest.json in %s' % (out,))
    cap = cap if cap is not None else man.get('max_per_sequence')
    if max_noseq_frac is None:
        max_noseq_frac = man.get('max_noseq_frac')
    reserved_crops = None
    try:
        with open(os.path.join(out, 'reserved_crops.json')) as fh:
            reserved_crops = json.load(fh)
    except (OSError, ValueError):
        if man.get('mode') == 'manifest' and not man.get('no_reserve'):
            bad.append('no reserved_crops.json: the reserved crops\' own '
                       'hashes are not on record, so a copy of one under '
                       'another name cannot be checked for')
    if reserves is None:
        try:
            with open(os.path.join(out, 'reserves.json')) as fh:
                reserves = json.load(fh)
        except (OSError, ValueError):
            reserves = []
            if not man.get('no_reserve'):
                bad.append('no reserves.json, and the build did not record '
                           '--no-reserve')
    rows = {}
    try:
        with open(os.path.join(out, 'crops.jsonl')) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r['uid'] in rows:
                        bad.append('crops.jsonl lists %s twice' % (r['uid'],))
                    rows[r['uid']] = r
    except (OSError, ValueError) as e:
        return bad + ['crops.jsonl unreadable: %s' % (e,)]
    on_disk = {}
    for split in ('train', 'val', 'test'):
        d = os.path.join(out, split)
        if not os.path.isdir(d):
            continue
        for cls in sorted(os.listdir(d)):
            cd = os.path.join(d, cls)
            if not os.path.isdir(cd):
                continue
            for f in os.listdir(cd):
                if f in on_disk:
                    bad.append('%s is on disk twice (%s and %s/%s)'
                               % (f, on_disk[f][0], split, cls))
                on_disk[f] = (split, cls, os.path.join(cd, f))
    extra = sorted(set(on_disk) - set(rows))
    missing = sorted(set(rows) - set(on_disk))
    if extra:
        bad.append('%d file(s) in the dataset are in no manifest row (e.g. '
                   '%s)' % (len(extra), extra[:3]))
    if missing:
        bad.append('%d manifest row(s) have no file (e.g. %s)'
                   % (len(missing), missing[:3]))
    checked = []
    for uid, (split, cls, path) in sorted(on_disk.items()):
        r = rows.get(uid)
        if r is None:
            continue
        if r.get('split') != split or r.get('label') != cls:
            bad.append('%s is filed under %s/%s but its row says %s/%s'
                       % (uid, split, cls, r.get('split'), r.get('label')))
        sha = sha256_file(path)
        if r.get('sha256') and sha != r['sha256']:
            bad.append('%s on disk is not the file its row describes' % (uid,))
        checked.append(dict(r, split=split, label=cls, sha256=sha))
    conflicts = []
    try:
        with open(os.path.join(out, 'conflicts.json')) as fh:
            conflicts = json.load(fh).get('conflicts') or []
    except (OSError, ValueError):
        pass
    bad += check_rows(checked, reserves, cap, max_noseq_frac,
                      conflict_member_uids(conflicts),
                      reserved_crops=reserved_crops,
                      allow_noseq=bool(man.get('keep_noseq')))
    return bad


# ── the size report ─────────────────────────────────────────────────────────

def _quantile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def size_report(rows, ref=None):
    """Short side per split x class against production's (gen-2) boxes.

    Never a failure -- there is no gen-2 box a person has labelled, so the
    distributions cannot match, and a hard gate would only be switched off.
    A split whose median sits outside 0.5-2x production's is WARNED, and the
    warning travels in the bundle: a class that is systematically smaller
    than the other is a "small means not a dog" shortcut waiting to be
    learned.
    """
    by = collections.defaultdict(list)
    for r in rows:
        wh = r.get('crop_wh')
        if wh:
            by['%s/%s' % (r['split'], r['label'])].append(min(wh))
    out = {'splits': {}, 'reference': ref, 'warnings': []}
    rp50 = ((ref or {}).get('quantiles') or {}).get('p50')
    for k in sorted(by):
        xs = by[k]
        doc = {'n': len(xs)}
        for name, q in (('p10', .1), ('p25', .25), ('p50', .5), ('p75', .75),
                        ('p90', .9)):
            doc[name] = _quantile(xs, q)
        for t in (32, 48, 64):
            doc['lt%d' % t] = round(sum(1 for x in xs if x < t) / len(xs), 4)
        out['splits'][k] = doc
        if rp50 and doc['p50'] and not (0.5 * rp50 <= doc['p50'] <= 2 * rp50):
            out['warnings'].append(
                '%s median short side %s px is outside 0.5-2x production\'s '
                '%s px' % (k, doc['p50'], rp50))
    return out


def print_size_report(rep):
    ref = rep.get('reference') or {}
    q = ref.get('quantiles') or {}
    print('\ncrop short side (px)       n     p10   p25   p50   p75   p90  '
          '<32    <48    <64')
    if q:
        print('  %-20s %7s %5s %5s %5s %5s %5s %5.1f%% %5.1f%% %5.1f%%'
              % ('production (ref)', ref.get('boxes', ''), q.get('p10'),
                 q.get('p25'), q.get('p50'), q.get('p75'), q.get('p90'),
                 100 * ref.get('lt32', 0), 100 * ref.get('lt48', 0),
                 100 * ref.get('lt64', 0)))
    for k, d in sorted(rep['splits'].items()):
        print('  %-20s %7d %5s %5s %5s %5s %5s %5.1f%% %5.1f%% %5.1f%%'
              % (k, d['n'], d['p10'], d['p25'], d['p50'], d['p75'], d['p90'],
                 100 * d['lt32'], 100 * d['lt48'], 100 * d['lt64']))
    for w in rep.get('warnings') or ():
        print('  WARNING ' + w)


# ── manifest mode ───────────────────────────────────────────────────────────

def read_manifest(path):
    rows = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise SystemExit('%s:%d is not JSON: %s' % (path, n, e))
    return rows


def manifest_main(args, reserves):
    import random
    rng = random.Random(args.seed)
    rows = read_manifest(args.manifest)
    if not rows:
        raise SystemExit('%s holds no rows' % (args.manifest,))
    classes = (args.pos_class, args.neg_class)
    # every row has to say what it is BEFORE anything is decided about it
    bad = []
    for r in rows:
        if not r.get('image_id') or not r.get('sequence'):
            bad.append(r.get('uid'))
    if bad:
        raise SystemExit('%d row(s) have no image_id or no sequence, e.g. %s '
                         '-- a crop nobody can place cannot be split honestly'
                         % (len(bad), bad[:5]))
    stray = sorted({r['label'] for r in rows} - set(classes))
    if stray:
        raise SystemExit('rows carry labels %s outside %s' % (stray, classes))
    print('manifest rows: %d (%s)' % (len(rows), ', '.join(
        '%s %d' % kv for kv in sorted(collections.Counter(
            '%s/%s' % (r['source'], r['label']) for r in rows).items()))))
    pre = check_rows([dict(r, split='train') for r in rows], (), None, None,
                     allow_noseq=True)
    pre = [b for b in pre if 'appear more than once' not in b
           and 'no sequence' not in b]
    if pre:
        raise SystemExit('the manifest itself fails: ' + '; '.join(pre))

    # ---- reserves --------------------------------------------------------
    # A reserved crop is held out by image_id and sequence, and its PIXELS
    # are held out too: every other row that is the same bytes, or within
    # `rh` dHash bits, of a reserved crop -- the rows of this manifest that
    # are reserved, and the crops the reserve files themselves record -- is
    # dropped whatever its label, its image_id or its sequence. A re-encoded
    # copy filed under another image is the same picture, and the dedup
    # below never sees it: the reserved rows have left the manifest by then.
    ids, seqs = reserved_sets(reserves)
    rh = max(2, int(args.hamming))
    held = [r for r in rows if r['image_id'] in ids or r['sequence'] in seqs]
    held_sha = {r.get('sha256') for r in held if r.get('sha256')}
    held_dh = {int(r['dhash']) for r in held if r.get('dhash') is not None}
    for res in reserves:
        held_sha.update(res.get('crop_sha256') or ())
        held_dh.update(res.get('crop_dhash') or ())
    reserved_crops = {'hamming': rh, 'sha256': sorted(held_sha),
                      'dhash': sorted(held_dh)}
    n0 = len(rows)
    rows = [r for r in rows if not (r['image_id'] in ids
                                    or r['sequence'] in seqs)]
    n_held = n0 - len(rows)
    mates = [r for r in rows if r.get('sha256') in held_sha]
    rows = [r for r in rows if r.get('sha256') not in held_sha]
    near = near_hashes(rows, held_dh, rh)
    rows = [r for r in rows if r['uid'] not in near]
    print('reserved: %d image_ids / %d sequences in %d file(s): %d rows '
          'removed, %d more sharing a reserved crop\'s bytes, %d more '
          'within dHash %d of one'
          % (len(ids), len(seqs), len(reserves), n_held, len(mates),
             len(near), rh))

    # ---- sequence-less rows ------------------------------------------------
    # A row whose sequence no manifest knows cannot be held disjoint from a
    # reserved sequence -- its real sequence-mate may be in the test set and
    # nothing can tell. So it is DROPPED, and counted, unless --keep-noseq
    # pins it to train on the record (bounded by --max-noseq-frac).
    noseq = [r for r in rows if r.get('seq_status') == 'noseq'
             or str(r.get('sequence') or '').startswith('noseq:')]
    if noseq and not args.keep_noseq:
        drop_ns = {r['uid'] for r in noseq}
        rows = [r for r in rows if r['uid'] not in drop_ns]
        print('no sequence: %d row(s) dropped (%s) -- --keep-noseq pins them '
              'to train instead' % (len(noseq), ', '.join(
                  '%s %d' % kv for kv in sorted(collections.Counter(
                      r['label'] for r in noseq).items()))))
    elif noseq:
        print('no sequence: %d row(s) kept, pinned to train (--keep-noseq)'
              % (len(noseq),))

    # ---- de-duplicate ------------------------------------------------------
    emb = None
    if args.dup_clusters:
        with open(args.dup_clusters) as f:
            cl = json.load(f)
        emb = cl.get('groups') or []
        print('embedding clusters: %d (cosine >= %s)'
              % (len(emb), cl.get('threshold')))
    kept, unions, conflicts, reported, dc = dedup_rows(rows, args.hamming,
                                                      emb)
    for k, v in sorted(dc.items()):
        print('  %-32s %d' % (k, v))
    kept, capped = cap_rows(kept, args.max_per_sequence, rng)
    print('  over per-sequence cap (%d)      %d %s'
          % (args.max_per_sequence, sum(capped.values()), dict(capped)))

    # ---- split -------------------------------------------------------------
    group_of = split_groups(kept, unions)
    split_of, pinned = assign_splits(kept, group_of, args.val_frac, rng)
    plan = []
    for r in kept:
        g = group_of[r['uid']]
        plan.append(dict(r, split=split_of[g], group=g))
    counts = collections.Counter('%s/%s' % (r['split'], r['label'])
                                 for r in plan)
    print('\nresulting split (whole groups, never straddling):')
    for k in sorted(counts):
        print('  %-22s %d' % (k, counts[k]))
    va = sum(v for k, v in counts.items() if k.startswith('val/'))
    print('  val fraction %.3f (target %s); %d groups, %d pinned to train '
          '(no sequence)' % (va / max(len(plan), 1), args.val_frac,
                             len(split_of), len(pinned)))
    cu = conflict_member_uids(conflicts)
    fails = check_rows(plan, reserves, args.max_per_sequence,
                       args.max_noseq_frac, cu, reserved_crops=reserved_crops,
                       allow_noseq=args.keep_noseq)
    ref = None
    if args.size_ref:
        with open(args.size_ref) as fh:
            ref = json.load(fh)
    sizes = size_report(plan, ref)
    print_size_report(sizes)
    if fails:
        for f in fails:
            print('FAIL ' + f)
        raise SystemExit('%d assertion(s) failed on the plan -- nothing '
                         'written' % (len(fails),))
    print('\nassertions on the plan: all %d hold' % (ASSERTIONS,))
    if not args.execute:
        print('\nnothing written. re-run with --execute')
        return 0
    if os.path.exists(args.out):
        raise SystemExit('%s exists -- never built over' % (args.out,))
    try:
        for r in plan:
            d = os.path.join(args.out, r['split'], r['label'])
            os.makedirs(d, exist_ok=True)
            shutil.copy2(r['path'], os.path.join(d, r['uid']))
        with open(os.path.join(args.out, 'crops.jsonl'), 'w') as fh:
            for r in plan:
                fh.write(json.dumps(r, sort_keys=True) + '\n')
        with open(os.path.join(args.out, 'conflicts.json'), 'w') as fh:
            json.dump({'conflicts': conflicts, 'reported': reported}, fh,
                      indent=1, sort_keys=True)
        with open(os.path.join(args.out, 'reserves.json'), 'w') as fh:
            json.dump(reserves, fh, indent=1, sort_keys=True)
        with open(os.path.join(args.out, 'reserved_crops.json'), 'w') as fh:
            json.dump(reserved_crops, fh, indent=1, sort_keys=True)
        man = {
            'mode': 'manifest', 'manifest': os.path.abspath(args.manifest),
            'manifest_sha256': sha256_file(args.manifest),
            'max_per_sequence': args.max_per_sequence,
            'max_noseq_frac': args.max_noseq_frac,
            'hamming': args.hamming, 'dup_clusters_file': args.dup_clusters,
            'val_frac': args.val_frac, 'seed': args.seed,
            'no_reserve': bool(args.no_reserve),
            'reserves': [{k: r[k] for k in ('path', 'sha256',
                                             'sequences_resolved_at_build')}
                         | {'image_ids': len(r['image_ids']),
                            'sequences': len(r['sequences'])}
                         for r in reserves],
            'source_rows': n0, 'reserved_rows_removed': n_held,
            'reserved_bytes_mates_removed': len(mates),
            'reserved_dhash_mates_removed': len(near),
            # which ones, so a dropped near-copy can be looked at
            'reserved_mates': sorted([r['uid'] for r in mates] + list(near)),
            'reserve_hamming': rh,
            'keep_noseq': bool(args.keep_noseq),
            'noseq_rows': len(noseq),
            'noseq_dropped': 0 if args.keep_noseq else len(noseq),
            'dedup': dict(dc), 'over_sequence_cap': dict(capped),
            'conflicts': len(conflicts), 'conflict_members': len(cu),
            'reported_embedding_conflicts': len(reported),
            'kept': len(plan), 'counts': dict(counts),
            'groups': len(split_of), 'groups_pinned_noseq': len(pinned),
            'train_sequences': len({r['sequence'] for r in plan
                                    if r['split'] == 'train'}),
            'val_sequences': len({r['sequence'] for r in plan
                                  if r['split'] == 'val'}),
            'noseq_crops': sum(1 for r in plan
                               if r.get('seq_status') == 'noseq'),
            'by_source': dict(collections.Counter(
                '%s/%s/%s' % (r['split'], r['label'], r['source'])
                for r in plan)),
            'sizes': sizes,
        }
        with open(os.path.join(args.out, 'rebuild_manifest.json'), 'w') as f:
            json.dump(man, f, indent=2, sort_keys=True)
        fails = verify_dataset(args.out)
    except BaseException:
        shutil.rmtree(args.out, ignore_errors=True)
        raise
    if fails:
        for f in fails:
            print('FAIL ' + f)
        shutil.rmtree(args.out, ignore_errors=True)
        raise SystemExit('%d assertion(s) failed on the written dataset -- '
                         'removed' % (len(fails),))
    print('assertions on disk: all %d hold' % (ASSERTIONS,))
    print('\nwrote %d crops -> %s' % (len(plan), args.out))
    return 0


ASSERTIONS = 10


# ── legacy mode: an existing dataset, by filename ────────────────────────────

def legacy_main(args, reserves):
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
    # A name nobody can read is a FAILURE, not a sequence of its own. This is
    # where 2,933 Label Studio crops used to become 'nofile:<name>' -- each a
    # singleton the cap and the straddle check could not see.
    unread = sorted(f for _, _, f in items if image_id_of(f) is None)
    if unread:
        raise SystemExit(
            '%d crop name(s) give no image_id, e.g. %s -- a crop that cannot '
            'be placed in its sequence cannot be split honestly'
            % (len(unread), unread[:10]))

    # ---- drop crops a reviewer said are mislabelled -----------------------
    # Separate from the reserves and for a different reason: those are held
    # out to keep the measurement honest, these are thrown out because the
    # label is wrong. A crop labelled not_dog that is plainly a dog does not
    # cost one example, it teaches the opposite of the thing you want and
    # does so every epoch. See tools/detect/label_flags.py.
    n_flag_removed = 0
    if not args.keep_flagged:
        try:
            import label_flags
            bad = {r['image_id'] for r in label_flags.flagged_files().values()
                   if r.get('image_id')}
        except Exception:
            bad = set()
        if bad:
            n0 = len(items)
            items = [it for it in items if image_id_of(it[2]) not in bad]
            n_flag_removed = n0 - len(items)
            print(f'mislabelled: {len(bad):,} flagged image_ids, '
                  f'{n_flag_removed:,} crops removed -> {len(items):,} left')

    ids = {image_id_of(f) for _, _, f in items}
    print(f'distinct source images: {len(ids):,}  -- resolving sequences ...')
    seq = resolve_sequences(ids, args.repo, args.duckdb_python)
    print(f'  resolved {len(seq):,} / {len(ids):,}'
          f'  ({len(ids) - len(seq):,} treated as their own sequence)')

    def seq_of(fname):
        iid = image_id_of(fname)
        return seq.get(iid) or ('noseq:' + iid)

    # ---- drop the reserves, by image_id AND by sequence ---------------------
    held_ids, held_seqs = reserved_sets(reserves)
    n_before_hold, n_held_removed = len(items), 0
    if reserves:
        n0 = len(items)
        items = [it for it in items if image_id_of(it[2]) not in held_ids
                 and seq_of(it[2]) not in held_seqs]
        n_held_removed = n0 - len(items)
        print(f'reserved: {len(held_ids):,} image_ids / {len(held_seqs):,} '
              f'sequences, {n_held_removed:,} crops removed -> '
              f'{len(items):,} trainable')

    # ---- collapse near-duplicates, within a class ------------------------
    print(f'\nhashing {len(items):,} crops (dHash, hamming<={args.hamming}) ...')
    by_cls = collections.defaultdict(list)
    for cls, path, fname in items:
        by_cls[cls].append((path, fname))
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
            uf = UnionFind()
            touch = {}
            for gi, g in enumerate(groups):
                uf.find(gi)
                for p, _ in g:
                    e = emb_group.get(os.path.realpath(p))
                    if e is None:
                        continue
                    if e in touch:
                        uf.union(touch[e], gi)
                    else:
                        touch[e] = gi
            merged = collections.defaultdict(list)
            for gi, g in enumerate(groups):
                merged[uf.find(gi)] += g
            groups = [merged[r] for r in sorted(merged)]
        survivors = []
        for g in groups:
            if len(g) == 1:
                survivors.append(g[0])
                continue
            best = max(g, key=lambda pf: os.path.getsize(pf[0]))
            survivors.append(best)
            dropped_dup[cls] += len(g) - 1
        per_seq = collections.defaultdict(list)
        for path, fname in survivors:
            per_seq[seq_of(fname)].append((path, fname))
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

    seq_items = collections.defaultdict(list)
    for cls, path, fname, s in keep:
        seq_items[s].append((cls, path, fname))
    seqs = sorted(s for s in seq_items if not s.startswith('noseq:'))
    pinned = sorted(s for s in seq_items if s.startswith('noseq:'))
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
    tr_seqs, va_seqs, shared = straddling_sequences(plan, seq_of)
    print(f'  train sequences {len(tr_seqs):,} | val sequences '
          f'{len(va_seqs):,} | shared {len(shared)}')
    if shared:
        raise SystemExit(
            f'BUG: {len(shared)} sequences straddle the split, e.g. '
            f'{sorted(shared)[:3]} -- every recall number off this val set '
            f'would be part memorisation')
    left = sorted(f for _, _, _, f in plan
                  if image_id_of(f) in held_ids or seq_of(f) in held_seqs)
    if left:
        raise SystemExit(f'BUG: {len(left)} reserved crops survived, e.g. '
                         f'{left[:3]}')
    over = collections.Counter((seq_of(f), c) for _, c, _, f in plan)
    over = [k for k, n in over.items() if n > args.max_per_sequence]
    if over:
        raise SystemExit(f'BUG: {len(over)} (sequence, class) groups are over '
                         f'the cap, e.g. {over[:3]}')

    clash = collections.Counter((s, c, f) for s, c, _, f in plan)
    clash = {k: v for k, v in clash.items() if v > 1}
    if clash:
        for (s, c, f), v in sorted(clash.items())[:10]:
            print(f'  COLLISION {s}/{c}/{f} <- {v} source files')
        raise SystemExit(
            f'{len(clash)} output filename(s) map to multiple source crops. '
            'Rename or drop one side (usually: do not pass --extra-negatives '
            'that were already folded into --src).')

    if not args.execute:
        print('\nnothing written. re-run with --execute')
        return 0

    for split, cls, path, fname in plan:
        d = os.path.join(args.out, split, cls)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(path, os.path.join(d, fname))
    man = {
        'mode': 'legacy',
        'src': args.src,
        'extra_negatives': args.extra_negatives,
        'extra_positives': args.extra_positives,
        'max_per_sequence': args.max_per_sequence,
        'hamming': args.hamming,
        'reserves': [{k: r[k] for k in ('path', 'sha256')}
                     | {'image_ids': len(r['image_ids']),
                        'sequences': len(r['sequences'])}
                     for r in reserves],
        'no_reserve': bool(args.no_reserve),
        'acceptance_set_ids': len(held_ids),
        'mislabelled_removed': n_flag_removed,
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
        'images_with_sequence': len(seq),
        'images_without_sequence': len(ids) - len(seq),
        'distinct_sequences': len(set(seq.values())),
    }
    with open(os.path.join(args.out, 'rebuild_manifest.json'), 'w') as f:
        json.dump(man, f, indent=2)
    print(f'\nwrote {len(plan):,} crops -> {args.out}')
    print(f'manifest: {os.path.join(args.out, "rebuild_manifest.json")}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--manifest',
                     help='crops.jsonl from crop_sources.py collect')
    src.add_argument('--src', help='LEGACY: an existing train/val dataset')
    src.add_argument('--verify-only', metavar='DIR',
                     help='run the on-disk assertions over a built dataset')
    ap.add_argument('--out')
    ap.add_argument('--repo', default=REPO)
    ap.add_argument('--duckdb-python',
                    default=os.environ.get('DUCKDB_PYTHON') or sys.executable,
                    help='interpreter with duckdb, for the sequence lookup')
    ap.add_argument('--extra-negatives',
                    help='LEGACY: directory of additional FULL-RESOLUTION '
                         'negative crops')
    ap.add_argument('--extra-positives',
                    help='LEGACY: directory of additional FULL-RESOLUTION '
                         'positive crops')
    ap.add_argument('--neg-class', default='not_dog')
    ap.add_argument('--pos-class', default='dog')
    ap.add_argument('--max-per-sequence', type=int, default=3,
                    help='cap crops kept per (sequence, class). Six frames of '
                         'one dog are about one example of signal.')
    ap.add_argument('--max-noseq-frac', type=float, default=0.02,
                    help='with --keep-noseq: fail if more than this share of '
                         'the kept crops has no sequence in any manifest')
    ap.add_argument('--keep-noseq', action='store_true',
                    help='manifest mode: keep rows whose sequence no manifest '
                         'knows, pinned to train, instead of dropping them. '
                         'Recorded. They cannot be held disjoint from a '
                         'reserved sequence, which is why the default drops '
                         'them.')
    ap.add_argument('--hamming', type=int, default=6,
                    help='dHash distance under which two crops of the SAME '
                         'class count as duplicates. 0 disables.')
    ap.add_argument('--keep-flagged', action='store_true',
                    help='LEGACY: keep crops a reviewer flagged as '
                         'mislabelled (crop_sources applies the flags itself '
                         'in manifest mode)')
    ap.add_argument('--reserve', action='append', default=[],
                    help='JSON with "image_ids" (and optionally "sequences") '
                         'that must never enter ANY split -- an acceptance set '
                         'or a reserved test set. Repeatable. Excluded by '
                         'image_id AND by sequence.')
    ap.add_argument('--exclude-ids', action='append', default=[],
                    help='deprecated alias of --reserve')
    ap.add_argument('--no-reserve', action='store_true',
                    help='build with nothing reserved. Recorded: a dataset '
                         'with no reserved test set says so.')
    ap.add_argument('--dup-clusters',
                    help='clusters.json from dedup_crops.py. dHash only sees '
                         'near-identical framings; the embedding clusters also '
                         'catch the same subject at another distance or angle, '
                         'which is what consecutive frames actually look like.')
    ap.add_argument('--size-ref',
                    help='production short-side reference '
                         '(crop_sources.py size-ref)')
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args(argv)

    if args.verify_only:
        reserves = (load_reserves(args.reserve + args.exclude_ids, args.repo,
                                  args.duckdb_python)
                    if (args.reserve or args.exclude_ids) else None)
        fails = verify_dataset(args.verify_only, reserves)
        for f in fails:
            print('FAIL ' + f)
        if fails:
            raise SystemExit('%d assertion(s) failed on %s'
                             % (len(fails), args.verify_only))
        print('verify: all %d assertions hold on %s'
              % (ASSERTIONS, args.verify_only))
        return 0
    if not (args.manifest or args.src):
        ap.error('one of --manifest, --src or --verify-only is required')
    if not args.out:
        ap.error('--out is required')
    paths = [p for p in args.reserve + args.exclude_ids if p]
    if args.exclude_ids:
        print('note: --exclude-ids is deprecated; read as --reserve')
    if not paths and not args.no_reserve:
        raise SystemExit(
            'no --reserve given. A build with no held-out set is how dogbin_v3 '
            'was accepted on crops it trained on; pass the acceptance and test '
            'sets, or --no-reserve to say on the record that there are none.')
    reserves = load_reserves(paths, args.repo, args.duckdb_python)
    if args.manifest:
        return manifest_main(args, reserves)
    return legacy_main(args, reserves)


if __name__ == '__main__':
    sys.exit(main())
