#!/usr/bin/env python3
"""Every human-labelled crop a classifier can learn from, cut the way production cuts.

    python tools/detect/crop_sources.py collect --family dogbin \\
        --stage <stage dir> --ls-export <label_studio_export.json>
    python tools/detect/crop_sources.py size-ref --detect-root <gen-2 store> \\
        --out size_ref.json

WHY THIS EXISTS. The crop datasets used to be COPIES: a base dataset's files
plus whatever extra crops each store happened to have on disk, identified by
nothing but their filenames. Four measured defects came out of that:

  * the filename parser could not read a Label Studio crop (`pos_ls_<id>_<k>`),
    so 2,933 crops in dogbin_20261012 had no image_id and no sequence. Each
    became its own one-crop "sequence": the per-sequence cap never saw them
    (one sequence kept 114 crops of one class) and the leak check could not
    either -- 6.77% of that dataset's val sat in a sequence train also held;
  * the crops were framed four different ways. Production (gate_store.py)
    judges the box padded by PAD_FRAC of its long side plus PAD_PX, cut from
    the full-resolution frame. The base sets were cut with a wider margin and
    a 64-px floor,
    the Label Studio crops with no padding at all, and 2,190 of those from a
    1280-px resized copy of the frame because the interpreter that ran the
    build had no S3 client -- so the gate was trained on thumbnails and asked
    about full-resolution crops;
  * 15k gate audit verdicts and every leash audit verdict were in no dataset;
  * the same box turned up under two names, sometimes with opposite labels.

So nothing is copied by name any more. Every crop is RE-DERIVED from a human
ledger, carries a manifest row saying exactly what it is and where it came
from, and is cut by gate_store.cut_crops itself -- imported, never copied, so
a change to production framing is a change to the training framing on the
next build. A row's identity is its manifest row, not its filename.

WHAT COUNTS AS A LABEL. Only what a person decided (HUMAN_SOURCES):

    ls_box          a box somebody drew in Label Studio and named
    hard_negative   a detection the review page's reviewer called not-a-dog
    hard_positive   a low-confidence detection the reviewer confirmed is a dog
    gate_audit      an answer on the gate audit sheet ("is this a dog?")
    leash_audit     an answer on the leash audit sheet
    leash_call      a leashed/unleashed call made on the review page
    frame_label     a dog box somebody drew on the annotate page, which asks
                    for EVERY dog in a whole frame (or says there is none)

A model's verdict is never one -- not the gate's store, not the leash run's
shards, not any guesser that pre-sorts the review queue -- and the dataset
builder asserts that against FORBIDDEN_PATH_RE on every row's provenance.

WHAT THIS WRITES. Into the stage directory only:

    crops/<label>/<uid>.jpg   one file per row, uid = <src>_<image_id>_<d|b><n>
    crops.jsonl               one row per crop (the schema is in ROW_FIELDS)
    collect_counts.json       every source's offered / kept / dropped-and-why
    collect_conflicts.json    boxes two human sources labelled differently

READ-ONLY on every ledger, store and drive. Frames are decoded at full
resolution with PIL (convert('RGB'), as gate_store._cut does) at low CPU and
I/O priority, because the gen-2 gate run reads the same drives.
"""

import argparse
import collections
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

HUMAN_SOURCES = frozenset({'ls_box', 'hard_negative', 'hard_positive',
                           'gate_audit', 'leash_audit', 'leash_call',
                           'frame_label'})
# A box a person DREW, as opposed to a detector box a person judged. Its key
# is 'b<k>' and it is joined to the detector's boxes by overlap, never by key.
DRAWN_SOURCES = frozenset({'ls_box', 'frame_label'})
FRAMINGS = frozenset({'cut_crops', 'audit_full', 'audit_edited'})
SOURCE_PREFIX = {'ls_box': 'ls', 'hard_negative': 'hn',
                 'hard_positive': 'hp', 'gate_audit': 'ga',
                 'leash_audit': 'la', 'leash_call': 'lc',
                 'frame_label': 'fl'}

# Where a MODEL's opinion lives. A training row whose provenance points into
# any of these is a model teaching itself, and the builder fails on it. Two of
# the names are spelled as patterns rather than words: the isolation guards
# next door (tools/detect/tests/) fail any builder that spells the stores of
# the two suggestion tiers out, and this list exists to REFUSE those stores,
# not to read them -- `[a]` matches the same letter and keeps the literal out
# of the source.
FORBIDDEN_PATH_RE = re.compile(
    r'(?:^|/)data/(?:gate(?:_[^/]*)?|leash)(?:/|$)'
    r'|llm_[a]nnotations|tri[a]ge|guess|mistakes|hidden_dogs|confusion')

# The ONE filename form this module writes.
UID_RE = re.compile(r'^(ls|hn|hp|ga|la|lc|fl)_(\d{6,})_([db]\d+)\.jpg$')
# Every legacy form the older datasets use: le_/un_/no_/nd_ (Label Studio
# leashed/unleashed/negative, from the old build_crop_dataset.py), ls_ (the
# last build's Label Studio crops), flag_/pos_ (folded-in harvest crops) and
# the plain `<image_id>_<det>` of a harvest or an audit export. The trailing
# number of an ls_ crop is its index among the WANTED boxes of one build,
# which is not stable across families, so only the plain and flag_/pos_ forms
# -- which really are detector indices -- give back a det key.
LEGACY_RE = re.compile(
    r'^(flag_|pos_)?(ls_|le_|un_|no_|nd_)?(\d{6,})_(?:[db])?(\d+)\.jpg$')
# A leash-call crop: <13-digit ms timestamp>_<image_id>_<conf>.jpg. The old
# NAME_RE read the TIMESTAMP as the image_id -- a wrong answer rather than no
# answer, which is the worse of the two -- so this form is refused outright.
TS_LED_RE = re.compile(r'^\d{13}_')

# `box` is what the crop was cut from (a person's redraw, if there is one);
# `pool_box` is the DETECTOR's box for the same det_key, unedited, which is
# what a drawn box is joined against -- a redraw that takes in a leash
# overlaps the Label Studio box of the same dog at IoU 0.4, the detector box
# it replaced at 0.97 (rebuild_crop_dataset.dedup_rows, step 1).
ROW_FIELDS = ('uid', 'path', 'label', 'family', 'source', 'image_id',
              'det_key', 'sequence', 'seq_status', 'box', 'pool_box',
              'frame', 'framing',
              'crop_wh', 'sha256', 'dhash', 'band', 'p_model', 'aimed',
              'provenance')

# What one family takes from each source, and what it calls the class.
FAMILY_CLASSES = {'dogbin': ('dog', 'not_dog'),
                  'leash': ('leashed', 'unleashed')}
FAMILY_SOURCES = {
    'dogbin': ('ls_box', 'hard_negative', 'hard_positive', 'gate_audit',
               'frame_label'),
    'leash': ('ls_box', 'leash_audit', 'leash_call'),
}
CONF_TOL = 0.006          # the review ledgers round conf to two decimals
LS_IOU = 0.5              # a drawn box IS a detector box at this overlap
JPEG_Q = 95               # what audit.py writes its full/ crops at


def D(*parts):
    return os.path.join(REPO, 'data', *parts)


def _say(text):
    # flushed: the build runner reads this through a pipe, where an
    # unflushed print sits in a buffer until the whole collection is over
    print(text, flush=True)


# ── names ──────────────────────────────────────────────────────────────────

def uid_for(source, image_id, det_key):
    return '%s_%s_%s.jpg' % (SOURCE_PREFIX[source], image_id, det_key)


def parse_name(fname):
    """(image_id, det_key or None) for a crop filename, or None.

    THE ONE PARSER for crop names. rebuild_crop_dataset.py and
    reserve_acceptance_set.py both had their own regex, both missed the ls_
    form, and a crop that parsed to None was silently treated as a sequence
    of its own -- which is how 2,933 crops escaped the cap and the leak check
    at once. A caller that gets None back must FAIL, not carry on.
    """
    base = os.path.basename(str(fname))
    m = UID_RE.match(base)
    if m:
        return m.group(2), m.group(3)
    if TS_LED_RE.match(base):
        return None
    m = LEGACY_RE.match(base)
    if not m:
        return None
    tag = m.group(2)
    return m.group(3), (None if tag else 'd' + m.group(4))


def image_id_of(fname):
    got = parse_name(fname)
    return got[0] if got else None


# ── small helpers ──────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_or_none(path):
    try:
        return sha256_file(path)
    except OSError:
        return None


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def low_priority():
    """Nice 10 and idle I/O class, for this process and everything it forks.

    The gen-2 gate run decodes frames off the same drives around the clock; a
    build that competes with it at equal priority slows a twelve-hour job to
    save four minutes on a one-off.
    """
    try:
        os.nice(10)
    except OSError:
        pass
    if shutil.which('ionice'):
        subprocess.run(['ionice', '-c3', '-p', str(os.getpid())],
                       capture_output=True)


def default_detect_root():
    """The gen-1 store, from data/detect_root.txt -- NOT $DETECT_ROOT.

    Every det_idx and confidence a human ledger holds was recorded against
    the gen-1 detector. $DETECT_ROOT exists so a candidate detector can sweep
    into a side store; a build that followed it would match a reviewer's
    "0.82, not a dog" against another detector's boxes. So the file decides,
    and a disagreeing environment is refused unless --detect-root says which
    one is meant.
    """
    with open(D('detect_root.txt')) as fh:
        root = fh.read().strip()
    env = os.environ.get('DETECT_ROOT', '').strip().rstrip('/')
    if env and os.path.realpath(env) != os.path.realpath(root):
        raise SystemExit(
            '$DETECT_ROOT (%s) is not the store data/detect_root.txt names '
            '(%s). The human ledgers carry gen-1 det_idx and confidences; '
            'pass --detect-root explicitly if you really mean another store.'
            % (env, root))
    return root


def drive_roots():
    """{drive label: grid_runs root}: gate_store's own lookup, or the plan's.

    The planner's lookup imports the dashboard, which only parses under the
    dashboard's interpreter. Elsewhere the roots the gate run itself was
    planned with are read from its plan -- read, never written.
    """
    try:
        import gate_store
        got = gate_store._resolve_roots()
        if got:
            return dict(got)
    except Exception:                     # noqa: BLE001 - fall back below
        pass
    try:
        with open(D('gate', 'plan.json')) as fh:
            got = json.load(fh).get('roots') or {}
    except (OSError, ValueError):
        got = {}
    if not got:
        raise SystemExit('no drive roots: neither the dashboard lookup nor '
                         'data/gate/plan.json gave any')
    return got


def _duck(memory='12GB'):
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='%s'" % memory)
    con.execute('SET threads=8')
    con.execute('SET enable_progress_bar=false')
    return con


def frames_for(image_ids, detect_root, con=None):
    """{image_id: (cell, drive, orig_w, orig_h)} from the store's img table.

    One deterministic row per image: the store keeps cell twins (the same jpg
    filed under two cells), and any one of them is the same file.
    """
    ids = sorted({str(i) for i in image_ids})
    if not ids:
        return {}
    import store
    own = con is None
    con = con or _duck()
    try:
        con.execute('CREATE OR REPLACE TEMP TABLE want_f(i VARCHAR)')
        con.executemany('INSERT INTO want_f VALUES (?)', [(i,) for i in ids])
        src = store._sql_src(store._store_globs(detect_root, 'img'))
        rows = con.execute(
            'SELECT CAST(image_id AS VARCHAR) i, cell, drive, orig_w, orig_h '
            'FROM %s WHERE CAST(image_id AS VARCHAR) IN (SELECT i FROM want_f) '
            'QUALIFY row_number() OVER (PARTITION BY image_id '
            'ORDER BY cell, drive) = 1' % (src,)).fetchall()
    finally:
        if own:
            con.close()
    return {str(i): (c, d, w, h) for i, c, d, w, h in rows}


def frame_path(roots, cell, drive, image_id):
    r = roots.get(drive)
    if not r or not cell:
        return None
    return os.path.join(r, cell, 'ground_animal_images', '%s.jpg' % image_id)


# ── the ledgers ─────────────────────────────────────────────────────────────

def read_audit_corrections(stage, path=None):
    """{(image_id, det_idx): (x1, y1, x2, y2)} -- the audit page's redraws.

    Only `crop == "audit:<stage>"` records, last write wins: that is the box
    the page cut its edited/ crop from, which is what the framing check below
    re-cuts to prove the file on disk is what it claims to be.
    """
    want = 'audit:%s' % stage
    out = {}
    try:
        with open(path or D('box_corrections', 'boxes.jsonl')) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(r, dict) or r.get('crop') != want:
                    continue
                try:
                    out[(str(r['image_id']), int(r.get('det_idx') or 0))] = (
                        float(r['x1']), float(r['y1']),
                        float(r['x2']), float(r['y2']))
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    return out


def pool_rows(stage, keys, con=None):
    """{(image_id, det_idx): (x1, y1, x2, y2, cell, drive)} from the audit pool.

    The pool is what the audit page cut its full/ crops from, so its box is
    the detector box the reviewer was shown.
    """
    import fn_audit
    keys = sorted(set(keys))
    if not keys:
        return {}
    own = con is None
    con = con or _duck()
    try:
        con.execute('CREATE OR REPLACE TEMP TABLE want_p(i VARCHAR, d INTEGER)')
        con.executemany('INSERT INTO want_p VALUES (?, ?)', keys)
        rows = con.execute(
            "SELECT p.image_id, p.det_idx, p.x1, p.y1, p.x2, p.y2, p.cell, "
            "p.drive FROM read_parquet('%s') p JOIN want_p w "
            "ON p.image_id = w.i AND p.det_idx = w.d"
            % fn_audit.paths(stage)['pool'].replace("'", "''")).fetchall()
    finally:
        if own:
            con.close()
    return {(str(i), int(d)): (x1, y1, x2, y2, c, dr)
            for i, d, x1, y1, x2, y2, c, dr in rows}


def exclusions(family, training_root=None):
    """{image_id: reason} -- images whose label a person has said is wrong.

    Three records, each an input whose digest goes in the bundle:

      label_flags        a crop flagged as mislabelled from the run panel
      manual_fixes       a crop removed by hand from an old base dataset,
                         read out of every <training root>/*/manual_fixes.jsonl
                         whose class belongs to this family
      leash conflicts    (leash only) images whose leash label disagreed
                         between two Label Studio lineages

    WHOLE IMAGES, because the record names a crop of an old dataset whose box
    index cannot be mapped onto today's rows -- the safe reading of "this
    crop of this frame is mislabelled" is "do not learn from this frame".
    """
    out, digests = {}, {}
    try:
        import label_flags
        for r in label_flags.flagged_files().values():
            if r.get('image_id'):
                out.setdefault(str(r['image_id']), 'label_flags')
        digests['label_flags'] = sha256_or_none(label_flags.DB_PATH)
    except Exception as e:                # noqa: BLE001 - say so, go on
        digests['label_flags'] = 'unreadable: %s' % (type(e).__name__,)
    classes = set(FAMILY_CLASSES[family])
    root = training_root
    if root is None:
        try:
            import build_dataset
            root = build_dataset.training_root()
        except SystemExit:
            root = None
    for path in sorted(_manual_fix_files(root)):
        digests['manual_fixes:' + path] = sha256_or_none(path)
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(r, dict) or not r.get('removed'):
                    continue
                if r.get('class_was') and r['class_was'] not in classes:
                    continue
                iid = image_id_of(r['removed'])
                if iid is None:
                    raise SystemExit('%s names a crop no parser can read: %r'
                                     % (path, r['removed']))
                out.setdefault(iid, 'manual_fixes')
    if family == 'leash':
        p = D('leash_label_conflicts.json')
        digests['leash_label_conflicts'] = sha256_or_none(p)
        try:
            with open(p) as fh:
                for name in json.load(fh) or []:
                    iid = image_id_of(name)
                    if iid is None:
                        raise SystemExit('%s names a crop no parser can '
                                         'read: %r' % (p, name))
                    out.setdefault(iid, 'leash_label_conflicts')
        except OSError:
            pass
    return out, digests


def _manual_fix_files(root):
    if not root or not os.path.isdir(root):
        return []
    out = []
    for name in os.listdir(root):
        p = os.path.join(root, name, 'manual_fixes.jsonl')
        if os.path.isfile(p):
            out.append(p)
    return out


# ── candidates, before any pixel is touched ─────────────────────────────────

def ls_candidates(tasks, family, export_path):
    """(rows, vetoes, counts) from a Label Studio export.

    det_key is 'b<k>' with k the box's index in the task's FULL label list, so
    a box has the same key whichever family reads it -- the old ls_ names
    counted only the wanted boxes, so box 2 of a task was 'ls_<id>_1' in one
    build and 'ls_<id>_2' in the next.

    `vetoes` are the boxes this family does NOT learn from but that still say
    something about a box it does: for the leash model, a box a person called
    "other animal" is a statement that it is not a dog at all, and a leash
    verdict on the same box is a conflict, not a label.
    """
    import build_dataset as bd
    want = bd.ls_wanted(family)
    veto_names = (set(bd.LS_NOT_DOG) if family == 'leash' else set())
    rows, vetoes = [], []
    c = collections.Counter()
    for t in tasks:
        url = str(t.get('image') or '')
        name = os.path.basename(url.split('?', 1)[0])
        stem, ext = os.path.splitext(name)
        boxes = t.get(bd.LS_GROUP) or []
        if not boxes:
            continue
        if not stem.isdigit() or ext.lower() != '.jpg':
            c['ls_bad_frame_name'] += len(boxes)
            continue
        for k, b in enumerate(boxes):
            names = b.get('rectanglelabels') or []
            hit = [n for n in names if n in want]
            if not hit:
                if any(n in veto_names for n in names):
                    vetoes.append({'image_id': stem, 'det_key': 'b%d' % k,
                                   'pct': _pct(b), 'ls_label': names[0],
                                   'orig': (b.get('original_width'),
                                            b.get('original_height'))})
                else:
                    c['ls_not_wanted'] += 1
                continue
            c['ls_offered'] += 1
            if len({want[n] for n in hit}) > 1:
                c['ls_two_labels'] += 1
                continue
            rows.append({
                'source': 'ls_box', 'label': want[hit[0]],
                'image_id': stem, 'det_key': 'b%d' % k,
                'pct': _pct(b),
                'ls_orig': (b.get('original_width'), b.get('original_height')),
                'ls_url': url,
                'framing': 'cut_crops',
                'provenance': {'ledger': export_path,
                               'key_or_task_id': t.get('id'),
                               'by': t.get('annotator'),
                               'ts': t.get('updated_at')
                               or t.get('created_at')}})
    return rows, vetoes, c


def _pct(b):
    return (float(b['x']), float(b['y']), float(b['width']),
            float(b['height']))


def flag_candidates(label, detect_root, ledger=None, corrections=None):
    """(rows, counts) for one review-page ledger: hard negatives or positives.

    The matching is harvest_flagged's, imported: a flag records an image_id
    and the confidence of the box the reviewer judged, and is resolved to the
    one gen-1 detection within CONF_TOL of it. None or several -- skipped and
    counted, because a wrong guess is a dog filed under not-a-dog.
    """
    import harvest_flagged as hf
    source = 'hard_negative' if label == 'false_positive' else 'hard_positive'
    cls = 'not_dog' if label == 'false_positive' else 'dog'
    ledger = ledger or hf.LEDGERS[label]
    flags = hf.read_flags(ledger, label)
    c = collections.Counter({'offered': len(flags)})
    if not flags:
        return [], c
    boxes = hf.boxes_for(sorted(flags), detect_root)
    corr = hf.read_corrections(corrections or hf.CORRECTIONS)
    rows = []
    for iid in sorted(flags):
        rec = flags[iid]
        dets = boxes.get(iid)
        if not dets:
            c['not_in_store'] += 1
            continue
        want = rec.get('conf')
        if want is None:
            hit = dets if len(dets) == 1 else []
            if len(dets) > 1:
                c['ambiguous'] += 1
                continue
        else:
            hit = [d for d in dets if abs(d[5] - float(want)) <= CONF_TOL]
            if not hit:
                c['unmatched'] += 1
                continue
            if len(hit) > 1:
                c['ambiguous'] += 1
                continue
        di, x1, y1, x2, y2, _conf = hit[0]
        det_box = [x1, y1, x2, y2]
        fixed = corr.get((iid, di))
        if fixed:
            x1, y1, x2, y2 = fixed
            c['box_corrected'] += 1
        rows.append({'source': source, 'label': cls, 'image_id': iid,
                     'det_key': 'd%d' % di, 'box': [x1, y1, x2, y2],
                     'pool_box': det_box, 'framing': 'cut_crops',
                     'provenance': {'ledger': ledger, 'key_or_task_id': iid,
                                    'by': rec.get('by'),
                                    'ts': rec.get('flagged_at')
                                    or rec.get('ts')}})
    return rows, c


def audit_candidates(stage, con=None):
    """(rows, counts) for one audit ledger, reusing the page's own crops.

    full/<key>.jpg is cut by audit.py with gate_store's constants from the
    full-resolution frame -- production framing already, so it is reused, not
    re-cut. A person's redraw (edited/) wins for training, exactly as
    fn_audit.export decides. The framing check in collect() re-cuts a sample
    of both from the frame to prove that claim on today's files.
    """
    import fn_audit as fa
    sp, P = fa.spec(stage), fa.paths(stage)
    source = 'gate_audit' if stage == 'gate' else 'leash_audit'
    labels = sp['answers'][:2]
    aimed = fa.targeted(stage)
    edited_dir = os.path.join(P['out'], 'edited')
    corr = read_audit_corrections(stage)
    c = collections.Counter()
    picked = []
    for v in fa.read_verdicts(stage=stage):
        c['verdicts'] += 1
        cls = fa.verdict_of(v.get('verdict'), stage)
        if cls not in labels:
            c['not_a_label (%s)' % (cls,)] += 1
            continue
        key = str(v['key'])
        iid, _, di = key.partition('#')
        if not iid.isdigit() or not di.isdigit():
            c['bad_key'] += 1
            continue
        picked.append((v, cls, iid, int(di)))
    pool = pool_rows(stage, [(iid, di) for _, _, iid, di in picked], con)
    rows = []
    for v, cls, iid, di in picked:
        name = '%s_%d.jpg' % (iid, di)
        ed, fu = os.path.join(edited_dir, name), os.path.join(P['full'], name)
        prow = pool.get((iid, di))
        if os.path.isfile(ed):
            path, framing = ed, 'audit_edited'
            box = corr.get((iid, di))
            if box is None:
                c['edited_without_correction_record'] += 1
        elif os.path.isfile(fu):
            path, framing = fu, 'audit_full'
            box = prow[:4] if prow else None
        else:
            c['audit_missing_crop'] += 1
            continue
        if prow is None:
            c['no_pool_row'] += 1
        rows.append({
            'source': source, 'label': cls, 'image_id': iid,
            'det_key': 'd%d' % di,
            'box': [float(x) for x in box] if box else None,
            'pool_box': [float(x) for x in prow[:4]] if prow else None,
            'cell_drive': (prow[4], prow[5]) if prow else None,
            'src_path': path, 'framing': framing,
            'band': v.get('band'), 'p_model': v.get(sp['p_col'])
            if v.get(sp['p_col']) is not None else v.get('p_dog'),
            'aimed': str(v['key']) in aimed, 'verdict_seq': v.get('seq'),
            'provenance': {'ledger': P['verdicts'],
                           'key_or_task_id': str(v['key']),
                           'by': v.get('by'), 'ts': v.get('ts')}})
    c['rows'] = len(rows)
    return rows, c


def leash_call_candidates(detect_root, corrections=None):
    """(rows, counts, conflicts) for the review page's leash calls (leash.db).

    The image_id comes from the database COLUMN, never the crop's filename --
    that name leads with a 13-digit timestamp the old parser read as the id.
    The box is the one gen-1 detection within CONF_TOL of the stored conf; a
    call that matches none or several is dropped and counted.

    THE REVIEWER'S REDRAW WINS, as it does for the dog-bin flags: the review
    page writes one box correction per (image_id, det_idx) whichever question
    it was asking, and the hard positive cut from the same box already uses
    it. Cutting the leash crop from the raw detector box instead framed one
    box two ways in two datasets -- and 10 of the 11 calls that survive have
    a redraw, at IoU 0.61-0.93 with the detector's box.

    ONE CALL PER BOX. leash.db is keyed by the crop's FILE name, which leads
    with the time the crop was queued; the pool rotates, so one detection can
    be queued and called twice. Agreeing calls collapse to the latest;
    disagreeing ones are all dropped and returned as a conflict -- two people
    (or one, twice) said both, and neither answer is the label.
    """
    import harvest_flagged as hf
    import leash_store
    c = collections.Counter()
    if not os.path.isfile(leash_store.DB_PATH):
        c['offered'] = 0
        c['ledger_missing'] = 1
        return [], c, []
    try:
        con = leash_store.connect()
        try:
            got = con.execute('SELECT crop, image_id, label, conf, ts, '
                              'labelled_at, "by" FROM leash').fetchall()
        finally:
            con.close()
    except Exception as e:                # noqa: BLE001 - a store, say so
        raise SystemExit('leash.db unreadable (%s: %s)'
                         % (type(e).__name__, e))
    c['offered'] = len(got)
    boxes = hf.boxes_for(sorted({str(r[1]) for r in got}), detect_root)
    corr = hf.read_corrections(corrections or hf.CORRECTIONS)
    per_box = collections.defaultdict(list)
    for crop, iid, label, conf, _ts, at, by in got:
        iid = str(iid)
        dets = boxes.get(iid) or []
        hit = [d for d in dets if conf is not None
               and abs(d[5] - float(conf)) <= CONF_TOL]
        if len(hit) != 1:
            c['unmatched' if not hit else 'ambiguous'] += 1
            continue
        di, x1, y1, x2, y2, _ = hit[0]
        det_box = [x1, y1, x2, y2]
        box = list(det_box)
        fixed = corr.get((iid, di))
        if fixed:
            box = [float(v) for v in fixed]
        per_box[(iid, di)].append({
            'source': 'leash_call', 'label': label,
            'image_id': iid, 'det_key': 'd%d' % di,
            'box': box, 'pool_box': det_box, 'framing': 'cut_crops',
            '_corrected': bool(fixed),
            'provenance': {'ledger': leash_store.DB_PATH,
                           'key_or_task_id': crop, 'by': by,
                           'ts': at}})
    rows, conflicts = [], []
    for key in sorted(per_box):
        lst = per_box[key]
        if len({r['label'] for r in lst}) > 1:
            c['dropped_calls_disagree'] += len(lst)
            conflicts.append({
                'kind': 'leash_calls_disagree',
                'members': [{'uid': uid_for('leash_call', r['image_id'],
                                            r['det_key']),
                             'label': r['label'], 'source': 'leash_call',
                             'call': r['provenance']['key_or_task_id']}
                            for r in lst]})
            continue
        if len(lst) > 1:
            c['collapsed_repeat_calls'] += len(lst) - 1
        best = max(lst, key=lambda r: (r['provenance']['ts'] or 0,
                                       r['provenance']['key_or_task_id']))
        if best.pop('_corrected'):
            c['box_corrected'] += 1
        rows.append(best)
    return rows, c, conflicts


FRAME_LABELS = D('frame_labels', 'labels.jsonl')


def read_frame_labels(path=None):
    """{image_id: latest record} from the annotate page's ledger.

    Append-only, one line per save, and THE LATEST LINE FOR AN IMAGE IS ITS
    LABEL (annotate_page.py): a recheck is a second save, not an edit, so an
    earlier box a later save left out is not a label any more.
    """
    out = {}
    try:
        with open(path or FRAME_LABELS) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue          # a torn last line is not a label
                if not isinstance(r, dict) or \
                        not str(r.get('image_id') or '').isdigit():
                    continue
                prev = out.get(str(r['image_id']))
                if prev is None or (r.get('at') or 0) >= (prev.get('at') or 0):
                    out[str(r['image_id'])] = r
    except OSError:
        pass
    return out


def frame_label_candidates(path=None):
    """(rows, empty frames, counts) from the annotate page's whole-frame labels.

    A person was shown a WHOLE frame and asked to box every dog in it, however
    small, or to say there is none. So each drawn box is a `dog`, and a frame
    declared empty is a statement about every box on it: no detector box
    there is a dog. The second is returned as `empty` ({image_id: record})
    for collect() to hold every other source's dog rows against.

    Boxes are in the frame's own pixels at the size the page recorded; the
    cut refuses a frame decoded at any other size, as it does for a Label
    Studio box. det_key is 'b<k>', k the box's index in the latest save.

    The batch a frame came from may have been CHOSEN by a model (hidden_dogs:
    a second model suspected a dog nobody had labelled) -- the selection is
    a model's, the boxes are not, and the page never records the hint.
    """
    path = path or FRAME_LABELS
    c = collections.Counter()
    if not os.path.isfile(path):
        c['ledger_missing'] = 1
        return [], {}, c
    rows, empty = [], {}
    for iid, r in sorted(read_frame_labels(path).items()):
        c['frames'] += 1
        boxes = r.get('boxes') or []
        if r.get('empty') or not boxes:
            if r.get('empty') and boxes:
                c['empty_with_boxes'] += 1      # contradictory: skip both
                continue
            if r.get('empty'):
                empty[iid] = r
                c['frames_empty'] += 1
            continue
        for k, b in enumerate(boxes):
            try:
                x1, y1, x2, y2 = (float(v) for v in b[:4])
            except (TypeError, ValueError):
                c['bad_box'] += 1
                continue
            if x2 <= x1 or y2 <= y1:
                c['bad_box'] += 1
                continue
            rows.append({
                'source': 'frame_label', 'label': 'dog', 'image_id': iid,
                'det_key': 'b%d' % k, 'box': [x1, y1, x2, y2],
                'fl_orig': (r.get('w'), r.get('h')), 'framing': 'cut_crops',
                'provenance': {'ledger': path,
                               'key_or_task_id': '%s#%s' % (iid,
                                                            r.get('batch')),
                               'by': r.get('by'), 'ts': r.get('at')}})
    c['rows'] = len(rows)
    return rows, empty, c


def other_family_vetoes(family, detect_root, con=None):
    """(vetoes, counts): the OTHER family's human verdicts, as statements here.

    Each crop family reads only its own ledgers, so one box could be "not a
    dog" on the gate audit and "unleashed" on the leash audit -- five such
    boxes trained both datasets, each on a contradictory human label. The
    leash audit has no "not a dog" answer, so a non-dog that reaches it comes
    back "unleashed"; the gate audit's answer is the better one, and both
    families drop the box rather than pick:

      leash   any not_dog verdict on the box (gate audit, review-page hard
              negative) vetoes the leash row, as "other animal" in Label
              Studio already does;
      dogbin  any leash verdict (leash audit, leash call) says the box is a
              dog, so a not_dog row on it is a conflict.

    A veto is {image_id, det_key, box (the detector's), label, source, uid}.
    Matching is apply_box_vetoes'.
    """
    import fn_audit as fa
    out = []
    c = collections.Counter()

    def audit(stage, want):
        picked = []
        for v in fa.read_verdicts(stage=stage):
            cls = fa.verdict_of(v.get('verdict'), stage)
            if cls not in want:
                continue
            iid, _, di = str(v['key']).partition('#')
            if iid.isdigit() and di.isdigit():
                picked.append((iid, int(di), cls, v))
        pool = pool_rows(stage, [(i, d) for i, d, _, _ in picked], con)
        src = 'gate_audit' if stage == 'gate' else 'leash_audit'
        for iid, di, cls, v in picked:
            pr = pool.get((iid, di))
            out.append({'image_id': iid, 'det_key': 'd%d' % di,
                        'box': [float(x) for x in pr[:4]] if pr else None,
                        'label': cls, 'source': src,
                        'uid': uid_for(src, iid, 'd%d' % di),
                        'by': v.get('by'), 'ts': v.get('ts')})
            c[src] += 1

    if family == 'leash':
        audit('gate', ('not_dog',))
        r, _ = flag_candidates('false_positive', detect_root)
        for x in r:
            out.append({'image_id': x['image_id'], 'det_key': x['det_key'],
                        'box': x.get('pool_box') or x['box'],
                        'label': 'not_dog', 'source': 'hard_negative',
                        'uid': uid_for('hard_negative', x['image_id'],
                                       x['det_key']),
                        'by': x['provenance'].get('by'),
                        'ts': x['provenance'].get('ts')})
            c['hard_negative'] += 1
    else:
        audit('leash', ('leashed', 'unleashed'))
        r, _, _ = leash_call_candidates(detect_root)
        for x in r:
            out.append({'image_id': x['image_id'], 'det_key': x['det_key'],
                        'box': x.get('pool_box') or x['box'],
                        'label': x['label'], 'source': 'leash_call',
                        'uid': uid_for('leash_call', x['image_id'],
                                       x['det_key']),
                        'by': x['provenance'].get('by'),
                        'ts': x['provenance'].get('ts')})
            c['leash_call'] += 1
    return out, c


def apply_box_vetoes(rows, vetoes, drop_labels, kind):
    """(kept, conflicts): drop rows another human record says are not this.

    A row and a veto are ONE BOX when both carry a detector key and the keys
    are equal, or -- when either one is a drawn box -- when the row's
    detector box (pool_box, else its box) overlaps the veto's at IoU >=
    LS_IOU. Only rows whose label is in `drop_labels` are dropped; each drop
    is a conflict naming both records.
    """
    by_img = collections.defaultdict(list)
    for v in vetoes:
        by_img[v['image_id']].append(v)
    keep, conf = [], []
    for row in rows:
        hits = []
        if row['label'] in drop_labels:
            rk = str(row.get('det_key') or '')
            rb = row.get('pool_box') or row.get('box')
            for v in by_img.get(row['image_id'], ()):
                vk = str(v.get('det_key') or '')
                if rk.startswith('d') and vk.startswith('d'):
                    if rk == vk:
                        hits.append(v)
                elif rb and v.get('box') and iou(rb, v['box']) >= LS_IOU:
                    hits.append(v)
        if hits:
            conf.append({'kind': kind,
                         'members': [{'uid': row['uid'], 'label': row['label'],
                                      'source': row['source']}]
                         + [{'uid': v['uid'], 'label': v['label'],
                             'source': v['source'], 'by': v.get('by'),
                             'ts': v.get('ts')} for v in hits]})
            continue
        keep.append(row)
    return keep, conf


def apply_empty_frames(rows, empty, drop_labels):
    """(kept, conflicts): a frame a person declared dog-free has no dog box.

    Every row on such a frame whose label says dog (dog for dogbin; leashed
    or unleashed for leash, which both mean dog) is a conflict with that
    declaration and is dropped, whichever record is newer -- a later dog
    verdict and an earlier "no dog here" are two people disagreeing, and the
    build is not the place to decide which one looked harder.
    """
    keep, conf = [], []
    for row in rows:
        r = empty.get(row['image_id'])
        if r is not None and row['label'] in drop_labels:
            conf.append({'kind': 'dog_row_on_empty_frame',
                         'members': [{'uid': row['uid'], 'label': row['label'],
                                      'source': row['source']},
                                     {'uid': 'fl_%s_empty' % row['image_id'],
                                      'label': 'empty',
                                      'source': 'frame_label',
                                      'by': r.get('by'), 'ts': r.get('at')}]})
            continue
        keep.append(row)
    return keep, conf


# ── frames and cutting ──────────────────────────────────────────────────────

S3_PROG = r'''
import json, os, sys
spec = json.load(open(sys.argv[1]))
got = {}
try:
    import boto3
    from dotenv import load_dotenv
except Exception as e:
    json.dump({"__error__": "import " + type(e).__name__}, open(spec["out"], "w"))
    sys.exit(3)
load_dotenv(spec["env"])
bucket = os.getenv("BUCKET_NAME")
if not bucket:
    json.dump({"__error__": "no bucket configured"}, open(spec["out"], "w"))
    sys.exit(3)
s3 = boto3.client("s3", endpoint_url=os.getenv("ENDPOINT_URL"),
                  aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
                  aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
                  region_name=os.getenv("BUCKET_REGION"))
os.makedirs(spec["dir"], exist_ok=True)
for name, key in spec["want"]:
    local = os.path.join(spec["dir"], name)
    tmp = local + ".part"
    try:
        s3.download_file(bucket, key, tmp)
        os.replace(tmp, local)
        got[name] = "ok"
    except Exception as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        got[name] = type(e).__name__
json.dump(got, open(spec["out"], "w"))
'''


def fetch_ls_frames(urls, frames_dir, s3_python, env_file):
    """{name: 'ok' | error class} -- the Label Studio originals, from S3.

    In a SUBPROCESS, under the interpreter that has boto3. The old path
    imported boto3 inside the build's own interpreter, which did not have
    it, swallowed the ImportError as "no bucket" and cut 2,190 crops from
    resized copies instead. Here a failure to reach the bucket at all is
    `{'__error__': ...}` and the caller makes it fatal. Only the class of an
    error is ever reported -- the credentials are read by the child from
    the training repository's .env and never cross back.
    """
    want = []
    for url in urls:
        name = os.path.basename(url.split('?', 1)[0])
        key = '/'.join(url.split('://', 1)[-1].split('/')[1:])
        want.append((name, key))
    if not want:
        return {}
    if not s3_python:
        return {'__error__': 'no --s3-python'}
    with tempfile.TemporaryDirectory() as tmp:
        spec_p, out_p = os.path.join(tmp, 's.json'), os.path.join(tmp, 'o.json')
        with open(spec_p, 'w') as fh:
            json.dump({'want': want, 'dir': frames_dir, 'env': env_file,
                       'out': out_p}, fh)
        p = subprocess.run([s3_python, '-c', S3_PROG, spec_p],
                           capture_output=True, text=True)
        try:
            with open(out_p) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {'__error__': 'helper exited %d' % (p.returncode,)}


def _open_rgb(path):
    from PIL import Image
    im = Image.open(path)
    im.load()
    if im.mode != 'RGB':
        im = im.convert('RGB')
    return im


def ls_pixel_box(pct, w, h):
    """Label Studio percent box -> full-resolution pixel box, on the frame in hand."""
    x, y, bw, bh = pct
    return [x / 100.0 * w, y / 100.0 * h, (x + bw) / 100.0 * w,
            (y + bh) / 100.0 * h]


def cut_frame(job):
    """Decode one frame, cut every row's box with gate_store.cut_crops.

    job = (frame path, [(row index, row)], out_dir). Returns
    (frame path, (w, h) or None, [(row index, saved path or None, reason)]).

    EVERY BOX IS CHECKED AGAINST THE SIZE ITS COORDINATES BELONG TO before it
    is cut: a Label Studio box against the width Label Studio recorded, an
    annotate-page box against the size the page recorded, a detector box
    against the store's orig_w/orig_h (row['_expect_wh']). A frame decoded at
    any other size -- a 1280-px copy, a re-download, a cache gone stale --
    puts the box somewhere else on the picture, which is the D4 failure:
    2,190 crops cut from resized copies and nothing noticed.
    """
    import gate_store
    path, items, out_dir = job
    try:
        im = _open_rgb(path)
    except Exception:                     # noqa: BLE001 - counted by caller
        return path, None, [(i, None, 'unreadable_frame') for i, _ in items]
    w, h = im.size
    results, boxes, owners = [], [], {}
    for n, (i, row) in enumerate(items):
        if row['source'] == 'ls_box':
            ow, oh = row.get('ls_orig') or (None, None)
            if ow and (int(ow) != w or (oh and int(oh) != h)):
                results.append((i, None, 'ls_size_mismatch'))
                continue
            row['box'] = ls_pixel_box(row['pct'], w, h)
        elif row['source'] == 'frame_label':
            ow, oh = row.get('fl_orig') or (None, None)
            if not ow or int(ow) != w or (oh and int(oh) != h):
                results.append((i, None, 'fl_size_mismatch'))
                continue
        ew = row.get('_expect_wh')
        if ew and ew[0] and (int(ew[0]) != w or int(ew[1]) != h):
            results.append((i, None, 'frame_size_mismatch'))
            continue
        x1, y1, x2, y2 = row['box']
        boxes.append((n, x1, y1, x2, y2))
        owners[n] = i
    got = dict(gate_store.cut_crops(im, boxes))
    for n, i in owners.items():
        crop = got.get(n)
        row = dict(items)[i]
        if crop is None:
            results.append((i, None, 'below_min_side'))
            continue
        d = os.path.join(out_dir, row['label'])
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, row['uid'])
        tmp = dst + '.tmp'
        crop.save(tmp, 'JPEG', quality=JPEG_Q)
        os.replace(tmp, dst)
        results.append((i, dst, ''))
    im.close()
    return path, (w, h), results


def recut_size(path, box):
    """The (w, h) gate_store.cut_crops gives this box on this frame, or None."""
    import gate_store
    im = _open_rgb(path)
    try:
        got = gate_store.cut_crops(im, [(0,) + tuple(box)])
        return tuple(got[0][1].size) if got else None
    finally:
        im.close()


def place_file(src, dst):
    """Hard link when the two are on one filesystem, copy otherwise."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.unlink(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def describe(path):
    """(sha256, dhash, (w, h)) of one crop file on disk."""
    from PIL import Image
    import rebuild_crop_dataset as rcd
    with Image.open(path) as im:
        wh = list(im.size)
    return sha256_file(path), rcd.dhash(path), wh


# ── the whole collection ────────────────────────────────────────────────────

def collect(family, stage, tasks, export_path, detect_root=None, roots=None,
            workers=4, s3_python=None, env_file=None, frames_dir=None,
            allow_missing_frames=0, allow_ls_size_mismatch=False,
            duckdb_python=None, training_root=None, framing_check=True,
            allow_empty_sources=(), say=_say):
    """Every human-labelled box for `family`, cut into <stage>/crops/.

    Returns (rows, counts, conflicts). Writes crops.jsonl,
    collect_counts.json and collect_conflicts.json beside the crops. Fails --
    SystemExit -- on anything that would make a row a lie: a crop with no
    image_id, a verdict whose sequence disagrees with the catalog, an audit
    crop that is not framed the way production frames, a frame decoded at a
    size its boxes were not drawn on, a frame that cannot be found or read
    at full resolution beyond what --allow-missing-frames permits.

    AND ON A SOURCE THAT GAVE NOTHING. Every reader of a ledger swallows a
    missing file (fn_audit.read_verdicts and harvest_flagged.read_flags
    return empty), so a moved ledger used to mean a build that was 81%
    smaller and reported itself clean. Each source in FAMILY_SOURCES has to
    produce rows unless `allow_empty_sources` names it -- and the names
    allowed are recorded in collect_counts.json.
    """
    import rebuild_crop_dataset as rcd
    if family not in FAMILY_CLASSES:
        raise SystemExit('collect: no such family %r' % (family,))
    detect_root = detect_root or default_detect_root()
    roots = roots or drive_roots()
    frames_dir = frames_dir or D('labelstudio_frames')
    crops_dir = os.path.join(stage, 'crops')
    os.makedirs(crops_dir, exist_ok=True)
    counts = {'family': family, 'detect_root': detect_root,
              'roots': roots, 'sources': {}, 'dropped': {}}
    drop = collections.Counter()
    conflicts = []

    # ---- 1. candidates per source --------------------------------------
    cand, vetoes = [], []
    con = _duck()
    try:
        if 'ls_box' in FAMILY_SOURCES[family] and tasks:
            r, vetoes, c = ls_candidates(tasks, family, export_path)
            cand += r
            counts['sources']['ls_box'] = dict(c, rows=len(r),
                                               vetoes=len(vetoes))
        elif 'ls_box' in FAMILY_SOURCES[family]:
            counts['sources']['ls_box'] = {'rows': 0, 'note': 'no export'}
        if family == 'dogbin':
            for label in ('false_positive', 'true_positive'):
                r, c = flag_candidates(label, detect_root)
                cand += r
                counts['sources']['hard_negative' if label == 'false_positive'
                                  else 'hard_positive'] = dict(c, rows=len(r))
            r, c = audit_candidates('gate', con)
            cand += r
            counts['sources']['gate_audit'] = dict(c)
        else:
            r, c = audit_candidates('leash', con)
            cand += r
            counts['sources']['leash_audit'] = dict(c)
            r, c, lc_conf = leash_call_candidates(detect_root)
            cand += r
            conflicts += lc_conf
            counts['sources']['leash_call'] = dict(c, rows=len(r))
        # the annotate page: boxes for dogbin, "no dog here" for both
        fl_rows, empty_frames, c = frame_label_candidates()
        if 'frame_label' in FAMILY_SOURCES[family]:
            cand += fl_rows
            counts['sources']['frame_label'] = dict(c)
        counts['empty_frames'] = len(empty_frames)
        say('candidates: %s' % ', '.join(
            '%s %d' % (s, n) for s, n in sorted(collections.Counter(
                x['source'] for x in cand).items())))
        allowed = set(allow_empty_sources or ())
        unknown = sorted(allowed - set(FAMILY_SOURCES[family]))
        if unknown:
            raise SystemExit('--allow-empty-source names no source of %s: %s'
                             % (family, unknown))
        empty_src = [src for src in FAMILY_SOURCES[family]
                     if not (counts['sources'].get(src) or {}).get('rows')
                     and src not in allowed]
        if empty_src:
            raise SystemExit(
                'source(s) %s gave no rows for %s (%s). A ledger that is '
                'missing, moved or unreadable reads as empty, and a build '
                'without it would report itself clean; --allow-empty-source '
                'NAME builds without one, on the record'
                % (', '.join(empty_src), family,
                   '; '.join('%s: %s' % (src, counts['sources'].get(src))
                             for src in empty_src)))
        counts['allowed_empty_sources'] = sorted(allowed)

        # the other family's verdicts on the same boxes (read here, while
        # the duckdb connection is open, applied after the cut)
        cross, cc = other_family_vetoes(family, detect_root, con)
        counts['cross_family_verdicts'] = dict(cc)

        # ---- 2. exclusions -------------------------------------------------
        excl, digests = exclusions(family, training_root)
        counts['exclusion_digests'] = digests
        kept = []
        for row in cand:
            why = excl.get(row['image_id'])
            if why:
                drop['excluded:' + why] += 1
                continue
            kept.append(row)
        cand = kept
        say('exclusions: %d image_ids on record, %d rows removed'
            % (len(excl), sum(v for k, v in drop.items()
                               if k.startswith('excluded:'))))

        for row in cand:
            row['family'] = family
            row['uid'] = uid_for(row['source'], row['image_id'],
                                 row['det_key'])
        dup_uid = [u for u, n in collections.Counter(
            r['uid'] for r in cand).items() if n > 1]
        if dup_uid:
            raise SystemExit('two candidate rows share a uid, e.g. %s'
                             % (dup_uid[:3],))

        # ---- 3. frames -----------------------------------------------------
        need = sorted({r['image_id'] for r in cand
                       if r['framing'] == 'cut_crops'}
                      | {r['image_id'] for r in cand
                         if r['framing'] != 'cut_crops'
                         and r.get('cell_drive') is None})
        say('looking up %d frames in %s ...' % (len(need), detect_root))
        placed = frames_for(need, detect_root, con)
    finally:
        con.close()
    paths = {}
    for iid, (cell, drive, _w, _h) in placed.items():
        p = frame_path(roots, cell, drive, iid)
        if p and os.path.isfile(p):
            paths[iid] = p
    # the size every detector (and annotate-page) box was drawn on: the
    # store's, checked against the decoded frame in cut_frame
    for row in cand:
        if row['framing'] == 'cut_crops' and row['source'] != 'ls_box':
            got = placed.get(row['image_id'])
            row['_expect_wh'] = (got[2], got[3]) if got else None
    # Label Studio frames nowhere on the drives: the local cache, then S3,
    # and accepted only at the width Label Studio recorded.
    ls_missing = sorted({r['image_id'] for r in cand
                         if r['source'] == 'ls_box'
                         and r['image_id'] not in paths})
    if ls_missing:
        urls = {r['image_id']: r['ls_url'] for r in cand
                if r['source'] == 'ls_box'}
        orig = {r['image_id']: r['ls_orig'] for r in cand
                if r['source'] == 'ls_box'}
        fetch = []
        for iid in ls_missing:
            local = os.path.join(frames_dir, iid + '.jpg')
            if not _full_width(local, orig[iid]):
                fetch.append(urls[iid])
        if fetch:
            say('fetching %d Label Studio frame(s) from the bucket ...'
                % (len(fetch),))
            got = fetch_ls_frames(fetch, frames_dir, s3_python, env_file
                                  or _env_file(training_root))
            if '__error__' in got:
                if allow_missing_frames < len(fetch):
                    raise SystemExit(
                        'the bucket could not be reached (%s) and %d frame(s) '
                        'need it -- pass --allow-missing-frames %d to build '
                        'without them, on the record'
                        % (got['__error__'], len(fetch), len(fetch)))
                drop['ls_bucket_unreachable_frames'] += len(fetch)
        for iid in ls_missing:
            local = os.path.join(frames_dir, iid + '.jpg')
            if _full_width(local, orig[iid]):
                paths[iid] = local
    # ---- 4. cut ------------------------------------------------------------
    jobs = collections.defaultdict(list)
    for i, row in enumerate(cand):
        if row['framing'] != 'cut_crops':
            continue
        p = paths.get(row['image_id'])
        if not p:
            row['_drop'] = ('ls_no_fullres_frame' if row['source'] == 'ls_box'
                            else 'no_frame')
            continue
        row['frame'] = {'path': p, 'w': None, 'h': None}
        jobs[p].append((i, row))
    n_missing = sum(1 for r in cand if r.get('_drop') in
                    ('ls_no_fullres_frame', 'no_frame'))
    if n_missing > allow_missing_frames:
        ex = [r['uid'] for r in cand if r.get('_drop')][:5]
        raise SystemExit('%d row(s) have no full-resolution frame (e.g. %s); '
                         '--allow-missing-frames %d permits that, on the '
                         'record' % (n_missing, ex, n_missing))
    say('cutting %d crops from %d full-resolution frames (%d workers) ...'
        % (sum(len(v) for v in jobs.values()), len(jobs), workers))
    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for path, wh, results in ex.map(
                cut_frame, [(p, items, crops_dir)
                            for p, items in sorted(jobs.items())]):
            done += 1
            for i, dst, why in results:
                row = cand[i]
                if wh:
                    row['frame']['w'], row['frame']['h'] = wh
                if dst:
                    row['path'] = dst
                else:
                    row['_drop'] = why
            if done % 500 == 0:
                say('  %d/%d frames, %.0fs' % (done, len(jobs),
                                              time.time() - t0))
    n_mis = sum(1 for r in cand
                if r.get('_drop') in ('ls_size_mismatch', 'fl_size_mismatch'))
    if n_mis and not allow_ls_size_mismatch:
        raise SystemExit('%d drawn box(es) are on a frame whose decoded '
                         'size is not the one Label Studio or the annotate '
                         'page recorded -- the box would land somewhere else. '
                         '--allow-ls-size-mismatch drops them on the record'
                         % (n_mis,))
    n_size = [r['uid'] for r in cand if r.get('_drop') == 'frame_size_mismatch']
    if n_size:
        raise SystemExit('%d detector box(es) are on a frame whose decoded '
                         'size is not the store\'s orig_w x orig_h (e.g. %s) '
                         '-- the frame on disk is not the one the detector '
                         'boxed' % (len(n_size), n_size[:5]))
    # A frame that was on disk at lookup and cannot be decoded now is a
    # missing frame too: a drive (lynx has no fstab entry) dropping mid-build
    # turns every frame after it into OSError, and those labels must not
    # vanish from a build that calls itself clean.
    lost = [r['uid'] for r in cand if r.get('_drop') in
            ('ls_no_fullres_frame', 'no_frame', 'unreadable_frame')]
    if len(lost) > allow_missing_frames:
        raise SystemExit('%d row(s) have no readable full-resolution frame '
                         '(e.g. %s); --allow-missing-frames %d permits that, '
                         'on the record' % (len(lost), lost[:5], len(lost)))
    # audit crops are the page's own files, placed beside the cut ones
    for row in cand:
        if row['framing'] in ('audit_full', 'audit_edited'):
            dst = os.path.join(crops_dir, row['label'], row['uid'])
            place_file(row['src_path'], dst)
            row['path'] = dst
            cd = row.get('cell_drive')
            fp = frame_path(roots, cd[0], cd[1], row['image_id']) if cd \
                else paths.get(row['image_id'])
            row['frame'] = {'path': fp, 'w': None, 'h': None}

    for row in cand:
        if row.get('_drop'):
            drop[row['_drop'] + ':' + row['source']] += 1
    rows = [r for r in cand if not r.get('_drop')]

    # ---- 5. the audit framing check --------------------------------------
    if framing_check:
        framing_check_audit(rows, say=say)

    # ---- 6. vetoes: another human record says this box is not this --------
    if vetoes:
        rows, vc = apply_vetoes(rows, vetoes)
        conflicts += vc
        drop['vetoed_by_ls_not_dog'] += len(vc)
    dog_labels = (('dog',) if family == 'dogbin'
                  else FAMILY_CLASSES['leash'])
    if cross:
        if family == 'leash':
            rows, vc = apply_box_vetoes(rows, cross, dog_labels,
                                        'leash_vs_gate_not_dog')
        else:
            rows, vc = apply_box_vetoes(rows, cross, ('not_dog',),
                                        'not_dog_vs_leash_verdict')
        conflicts += vc
        drop['vetoed_by_other_family'] += len(vc)
    if empty_frames:
        rows, vc = apply_empty_frames(rows, empty_frames, dog_labels)
        conflicts += vc
        drop['vetoed_by_empty_frame'] += len(vc)
    say('vetoes: %d row(s) dropped against another human record (%s)'
        % (sum(drop[k] for k in ('vetoed_by_ls_not_dog',
                                 'vetoed_by_other_family',
                                 'vetoed_by_empty_frame')),
           ', '.join('%s %d' % (k, drop[k]) for k in
                     ('vetoed_by_ls_not_dog', 'vetoed_by_other_family',
                      'vetoed_by_empty_frame'))))

    # ---- 7. sequences --------------------------------------------------------
    ids = sorted({r['image_id'] for r in rows})
    say('resolving sequences for %d images ...' % (len(ids),))
    seq = rcd.resolve_sequences(set(ids), REPO,
                                duckdb_python or sys.executable)
    disagree = []
    for row in rows:
        if not row.get('image_id'):
            raise SystemExit('a row has no image_id: %r' % (row.get('uid'),))
        s = seq.get(row['image_id'])
        vs = row.get('verdict_seq')
        if s and vs and not str(vs).startswith('img:') and vs != s:
            disagree.append((row['uid'], vs, s))
        if s:
            row['sequence'], row['seq_status'] = s, 'manifest'
        else:
            row['sequence'], row['seq_status'] = ('noseq:' + row['image_id'],
                                                  'noseq')
    if disagree:
        raise SystemExit('%d verdict(s) carry a sequence the catalog '
                         'disagrees with, e.g. %s -- one of the two records '
                         'is wrong and the split would follow whichever was '
                         'read' % (len(disagree), disagree[:3]))

    # ---- 8. describe every file ---------------------------------------------
    say('hashing %d crops ...' % (len(rows),))
    with ThreadPoolExecutor(max_workers=8) as ex:
        for row, (sha, dh, wh) in zip(rows, ex.map(describe,
                                                   [r['path'] for r in rows])):
            row['sha256'], row['dhash'], row['crop_wh'] = sha, dh, wh
    out_rows = []
    for row in rows:
        out_rows.append({k: row.get(k) for k in ROW_FIELDS})
    counts['dropped'] = dict(drop)
    counts['rows'] = len(out_rows)
    counts['by_source_label'] = dict(collections.Counter(
        '%s/%s' % (r['source'], r['label']) for r in out_rows))
    counts['noseq'] = sum(1 for r in out_rows if r['seq_status'] == 'noseq')
    with open(os.path.join(stage, 'crops.jsonl'), 'w') as fh:
        for r in sorted(out_rows, key=lambda r: r['uid']):
            fh.write(json.dumps(r, sort_keys=True) + '\n')
    with open(os.path.join(stage, 'collect_counts.json'), 'w') as fh:
        json.dump(counts, fh, indent=1, sort_keys=True)
    with open(os.path.join(stage, 'collect_conflicts.json'), 'w') as fh:
        json.dump(conflicts, fh, indent=1, sort_keys=True)
    say('collected %d rows (%s); dropped %s'
        % (len(out_rows), ', '.join('%s %d' % kv for kv in
                                    sorted(counts['by_source_label'].items())),
           dict(drop) or 'nothing'))
    return out_rows, counts, conflicts


def _env_file(training_root=None):
    root = training_root
    if root is None:
        try:
            import build_dataset
            root = build_dataset.training_root()
        except SystemExit:
            return None
    return os.path.join(root, '.env')


def _full_width(path, orig):
    """True if `path` is a readable frame at the width Label Studio recorded."""
    if not os.path.isfile(path):
        return False
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
    except Exception:                     # noqa: BLE001
        return False
    ow, oh = orig or (None, None)
    return bool(ow) and int(ow) == w and (not oh or int(oh) == h)


def framing_check_audit(rows, frac=0.01, minimum=50, seed=20261114,
                        say=_say):
    """Prove the reused audit crops are framed the way production frames.

    A seeded sample (1%, at least 50) of the audit rows is re-cut from its
    frame with gate_store.cut_crops -- from the pool box for full/, from the
    page's last recorded redraw for edited/ -- and the size has to match the
    file exactly. A mismatch means the page cut with constants that are not
    production's, which would put every reused crop on the wrong framing; it
    fails the collection rather than shipping a set that only looks matched.
    """
    aud = sorted((r for r in rows if r['framing'] in
                  ('audit_full', 'audit_edited') and r.get('box')),
                 key=lambda r: r['uid'])
    if not aud:
        return {'checked': 0}
    k = min(len(aud), max(minimum, int(round(frac * len(aud)))))
    sample = random.Random(seed).sample(aud, k)
    from PIL import Image
    bad, skipped = [], 0
    from concurrent.futures import ThreadPoolExecutor

    def one(row):
        fp = (row.get('frame') or {}).get('path')
        if not fp or not os.path.isfile(fp):
            return row, None, 'no_frame'
        try:
            return row, recut_size(fp, row['box']), ''
        except Exception as e:            # noqa: BLE001
            return row, None, type(e).__name__

    with ThreadPoolExecutor(max_workers=4) as ex:
        for row, got, why in ex.map(one, sample):
            if why:
                skipped += 1
                continue
            with Image.open(row['path']) as im:
                have = tuple(im.size)
            if got != have:
                bad.append((row['uid'], have, got))
    say('framing check: %d audit crops re-cut from their frames, %d match, '
        '%d frames unavailable' % (len(sample) - skipped,
                                   len(sample) - skipped - len(bad), skipped))
    if bad:
        raise SystemExit('%d audit crop(s) are not framed the way '
                         'gate_store.cut_crops frames them, e.g. %s'
                         % (len(bad), bad[:3]))
    if skipped > len(sample) // 2:
        raise SystemExit('framing check: %d of %d sampled frames could not '
                         'be read -- the check proved nothing'
                         % (skipped, len(sample)))
    return {'checked': len(sample) - skipped, 'skipped': skipped}


def apply_vetoes(rows, vetoes):
    """Drop leash rows whose box a person ALSO called "other animal".

    The veto box is in percent of the frame Label Studio was shown; scaled by
    the size it recorded, which for every Label Studio frame is the
    full-resolution size (the store's orig_w matched it for all 2,971 that
    have a store row).
    """
    by_img = collections.defaultdict(list)
    for v in vetoes:
        ow, oh = v['orig']
        if not ow or not oh:
            continue
        by_img[v['image_id']].append((v, ls_pixel_box(v['pct'], ow, oh)))
    keep, conf = [], []
    for row in rows:
        hits = [v for v, vb in by_img.get(row['image_id'], ())
                if row.get('box') and iou(row['box'], vb) >= LS_IOU]
        if hits:
            conf.append({'kind': 'leash_vs_ls_not_dog',
                         'members': [{'uid': row['uid'], 'label': row['label'],
                                      'source': row['source']}]
                         + [{'uid': 'ls_%s_%s' % (v['image_id'], v['det_key']),
                             'label': v['ls_label'], 'source': 'ls_box'}
                            for v in hits]})
            continue
        keep.append(row)
    return keep, conf


# ── the production size reference ───────────────────────────────────────────

def size_reference(detect_root, out=None):
    """Short-side quantiles of every box in a store, padded the way production pads.

    The yardstick every split's size report is read against. Computed in SQL
    with gate_store's constants (imported), on the RAW boxes of the store the
    model will be judging -- the gen-2 store for a gen-2 gate.
    """
    import gate_store
    import store
    con = _duck()
    try:
        src = store._sql_src(store._store_globs(detect_root, 'det'))
        q = ('WITH b AS (SELECT (x2-x1) w, (y2-y1) h FROM %s), '
             'p AS (SELECT least(w,h) raw, CAST(%r*greatest(w,h) AS INT)+%d '
             'pad FROM b), s AS (SELECT raw, raw + 2*pad ss FROM p) '
             'SELECT count(*), quantile_cont(ss,[0.1,0.25,0.5,0.75,0.9]), '
             'avg((ss<32)::INT), avg((ss<48)::INT), avg((ss<64)::INT), '
             'quantile_cont(raw,[0.1,0.25,0.5,0.75,0.9]) FROM s'
             % (src, gate_store.PAD_FRAC, gate_store.PAD_PX))
        n, qs, l32, l48, l64, raw = con.execute(q).fetchone()
    finally:
        con.close()
    doc = {'detect_root': detect_root, 'boxes': int(n),
           'note': 'short side of the cut_crops-padded box, before clamping '
                   'to the frame (an upper bound by a few px at the edges)',
           'pad_frac': gate_store.PAD_FRAC, 'pad_px': gate_store.PAD_PX,
           'quantiles': dict(zip(('p10', 'p25', 'p50', 'p75', 'p90'),
                                 [round(x, 1) for x in qs])),
           'raw_quantiles': dict(zip(('p10', 'p25', 'p50', 'p75', 'p90'),
                                     [round(x, 1) for x in raw])),
           'lt32': round(l32, 4), 'lt48': round(l48, 4),
           'lt64': round(l64, 4), 'computed_at': int(time.time())}
    if out:
        with open(out, 'w') as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
    return doc


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('collect', help='cut every human-labelled crop')
    c.add_argument('--family', required=True, choices=sorted(FAMILY_CLASSES))
    c.add_argument('--stage', required=True)
    c.add_argument('--ls-export', help='Label Studio JSON_MIN export')
    c.add_argument('--detect-root',
                   help='the store whose det_idx the ledgers carry (default: '
                        'data/detect_root.txt; a disagreeing $DETECT_ROOT is '
                        'refused unless this is given)')
    c.add_argument('--workers', type=int, default=4)
    c.add_argument('--s3-python',
                   help='interpreter with boto3 for the few Label Studio '
                        'frames no drive holds')
    c.add_argument('--frames-dir', help='Label Studio frame cache '
                                        '(default data/labelstudio_frames)')
    c.add_argument('--allow-missing-frames', type=int, default=0)
    c.add_argument('--allow-ls-size-mismatch', action='store_true')
    c.add_argument('--duckdb-python', default=sys.executable)
    c.add_argument('--no-framing-check', action='store_true')
    c.add_argument('--allow-empty-source', action='append', default=[],
                   metavar='NAME',
                   help='build even though this source gave no rows '
                        '(repeatable; recorded in collect_counts.json). '
                        'Without it an empty source is fatal.')
    s = sub.add_parser('size-ref', help='the production short-side reference')
    s.add_argument('--detect-root', required=True)
    s.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    low_priority()
    if a.cmd == 'size-ref':
        doc = size_reference(a.detect_root, a.out)
        print(json.dumps(doc, indent=1, sort_keys=True))
        return 0
    tasks = None
    if a.ls_export:
        with open(a.ls_export) as fh:
            tasks = json.load(fh)
    root = a.detect_root or default_detect_root()
    collect(a.family, a.stage, tasks, a.ls_export, detect_root=root,
            workers=a.workers, s3_python=a.s3_python,
            frames_dir=a.frames_dir,
            allow_missing_frames=a.allow_missing_frames,
            allow_ls_size_mismatch=a.allow_ls_size_mismatch,
            duckdb_python=a.duckdb_python,
            framing_check=not a.no_framing_check,
            allow_empty_sources=a.allow_empty_source)
    return 0


if __name__ == '__main__':
    sys.exit(main())
