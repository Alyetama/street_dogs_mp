#!/usr/bin/env python3
"""Run a promoted classifier over the store, one stage at a time.

    python tools/detect/gate_store.py plan            # what there is to do
    python tools/detect/gate_store.py run             # do it, resumably
    python tools/detect/gate_store.py run --limit 5000
    python tools/detect/gate_store.py status
    python tools/detect/gate_store.py plan  --stage leash
    python tools/detect/gate_store.py run   --stage leash
    python tools/detect/gate_store.py restore --stage gate [--archive NAME]

TWO STAGES, ONE MACHINE. The pipeline narrows: the detector finds ground
animals, the gate decides which of them are dogs, and the leash model decides
which of those dogs are on a leash. Each stage reads the one before it, so the
leash stage judges the gate's dogs and nothing else -- 4.7M boxes become
~870K, which is most of why it is a separate pass rather than a second head.

Everything around the model is identical between them: the same locality
ordering, the same shards, the same resume, the same heartbeat. So it is the
same code with a stage table, not two files that drift.

WHAT THIS IS. The detector is single-class and deliberately loose: it finds
ground animals and calls them all "target". About one in five of its boxes is
a dog. The gate is the binary classifier trained on this project's own
reviewers' verdicts, and running it over the whole store turns 4.8M "some
animal" boxes into 4.8M dog / not-a-dog verdicts with a probability each.

WHAT THIS IS NOT. A verdict here is a MODEL'S opinion and never a label. It is
written to its own store under data/gate/, every row stamped unverified, and
nothing that builds a training set reads it. The reviewer ledgers are written
only by a human clicking a verdict, and this file neither opens nor names
them -- the guard in tools/detect/tests/ that enforces that separation reads
source, not intentions, which is why the rule is stated here without spelling
their paths.

WHY IT IS SHAPED LIKE THIS. Measured before it was written:

    read + decode one frame     ~116 ms      98% of the work
    classify one crop            ~2.4 ms      2%

The frames are 8000x4000 panoramas, so decoding is the whole cost and the GPU
would idle at 2% duty on one process. Sixteen decoder processes reach ~242
images/s, which at 1.45 boxes an image is ~350 crops/s against the gate's 417
-- the two sides balance, which is why the worker count defaults to the core
count and not higher.

Reduced-scale JPEG decode was measured and rejected: it doubles decode speed,
but the median box is 107px across an 8000px frame and only 3.4% are large
enough to survive even a half-scale decode with 224px left for the model. The
gate would have been judging a different picture from the one it was trained
on, everywhere.

Work is ordered by drive and cell so each worker walks one directory at a
time, and written in shards so a run that is interrupted resumes at the shard
boundary rather than the beginning.
"""

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# One row per stage. `feeds_on` is the whole difference between them: the gate
# judges every detection, the leash model judges only what the gate called a
# dog. `positive` names the class whose share the dashboard reports, and
# `p_col` the probability column written beside every verdict -- kept at five
# decimals so a threshold can be moved over the finished store without
# re-running anything.
STAGES = {
    'gate': {
        'dir': 'gate', 'project': 'dog-bin', 'title': 'dog-bin gate',
        'classes': ['dog', 'not_dog'], 'positive': 'dog', 'p_col': 'p_dog',
        'feeds_on': None,
    },
    'leash': {
        'dir': 'leash', 'project': 'leash-models', 'title': 'leash model',
        'classes': ['leashed', 'unleashed'], 'positive': 'leashed',
        'p_col': 'p_leashed', 'feeds_on': 'gate',
    },
}
STAGE = 'gate'


def stages_root(repo=None):
    """Where the stage directories live: data/ by default, or $GATE_ROOT.

    A separate root is how a store that is not live yet gets judged without
    touching the live record: the gen-2 gate writes under its own root while
    the dashboard keeps reading gen-1 detections beside gen-1 verdicts. Mixing
    them in one directory would join one detector's det_idx to the other's
    boxes. An explicit `repo` (the tests, the dashboard) always means
    <repo>/data, whatever the environment says.
    """
    env = os.environ.get('GATE_ROOT')
    if env and repo is None:
        return env if os.path.isabs(env) else os.path.join(REPO, env)
    return os.path.join(repo or REPO, 'data')


OUT_DIR = os.path.join(stages_root(), 'gate')
PLAN_FILE = os.path.join(OUT_DIR, 'plan.json')
# What the run is doing RIGHT NOW. Progress was read off the written shards
# alone, and a shard is 20,000 frames -- so for the first several minutes of a
# twelve-hour run the dashboard showed 0%, 0 judged, no rate and no share, an
# idle-looking panel above a machine at full tilt. The shards remain the
# record; this is the only thing that can speak before the first one lands.
BEAT_FILE = os.path.join(OUT_DIR, 'progress.json')
BEAT_EVERY_S = 1.0        # one small atomic write; the panel polls at 2 s

SHARD_ROWS = 20000
# The margin the gate was trained with. tools/detect/build_review_set.py and
# the live preview writer both use it; inference has to match or every verdict
# is made on a differently framed picture.
PAD_FRAC, PAD_PX = 0.12, 4
MIN_SIDE = 8
# images one drive contributes before the plan moves to the next. Sized
# so a worker's chunk falls inside one block and stays on one disk.
INTERLEAVE_BLOCK = 256
# Reader threads per drive (--readers overrides). Three of the four drives
# answer one request at a time (queue depth 1), and the old feed put 4-8
# decoders on each, which seeked between their files: capybara and jackal
# held ~15 MB/s at 100% busy while lynx sat idle 65% of the time, waiting its
# turn in a fixed order. Two readers keep one request queued behind the one
# being served; bobcat queues 30 deep and takes more, as in the sweep.
READERS = {'bobcat': 8}
READERS_DEFAULT = 2
# How many shards past the lowest unfinished one a drive may read. A fast
# drive runs ahead instead of waiting for the slowest at every shard, but a
# shard is only written whole and they fill side by side, so a stop throws
# away up to AHEAD + 1 partly-scored shards (5 x 20,000 frames by default),
# not the one shard the in-order feed used to lose. Lower it with --ahead
# when the run will be stopped often.
AHEAD = 4


def spec(stage=None):
    return STAGES[stage or STAGE]


def use(stage):
    """Point the module at one stage's directory.

    Module state rather than an argument threaded through nine functions: the
    process runs exactly one stage from `main` to exit, and the alternative
    was a parameter on every path helper that could only ever hold one value
    per process. `data/gate/` keeps its name so a job already part-written
    stays resumable.
    """
    global STAGE, OUT_DIR, PLAN_FILE, BEAT_FILE
    if stage not in STAGES:
        raise SystemExit(f'unknown stage {stage!r} -- '
                         f'one of {", ".join(sorted(STAGES))}')
    STAGE = stage
    OUT_DIR = os.path.join(stages_root(), STAGES[stage]['dir'])
    PLAN_FILE = os.path.join(OUT_DIR, 'plan.json')
    BEAT_FILE = os.path.join(OUT_DIR, 'progress.json')


def _roots():
    """{drive label: grid_runs root}, resolved by the PLANNER and carried in
    the plan.

    Not read here. The planner runs on the interpreter that has duckdb, which
    is also the one that can parse the dashboard module the roots come from;
    the runner runs on the one that has ultralytics, and importing the
    dashboard there is a SyntaxError -- it uses f-string syntax that only
    3.12 accepts. Whatever the runner needs, the plan carries.
    """
    try:
        with open(PLAN_FILE) as fh:
            got = (json.load(fh).get('roots') or {})
    except (OSError, ValueError):
        got = {}
    if not got:
        raise SystemExit('the plan carries no drive roots -- re-run `plan`')
    return got


def _resolve_roots():
    """The planner's own lookup, which may import the dashboard."""
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    import dashboard as dash
    return dash._grid_roots()


def gate_weights():
    """The promoted model for this stage, from data/best_models.json. Never
    hardcoded: a new one is promoted by editing that file, and this follows
    it."""
    project = spec()['project']
    with open(os.path.join(REPO, 'data', 'best_models.json')) as fh:
        best = (json.load(fh).get('projects') or {}) \
            .get(project, {}).get('best') or {}
    rel = str(best.get('weights') or '')
    if not rel:
        raise SystemExit(f'no {project} model promoted in '
                         f'data/best_models.json')
    root = os.environ.get('TRAINING_ROOT') or os.path.dirname(
        os.path.dirname(REPO))
    p = rel if os.path.isabs(rel) else os.path.join(root, rel)
    if not os.path.exists(p):
        raise SystemExit(f'promoted {project} model not on disk: {p}\n'
                         '  set TRAINING_ROOT to where the runs live')
    return p, best.get('run') or os.path.basename(os.path.dirname(
        os.path.dirname(p)))


def _beat(**kw):
    """Publish the in-flight figures. Never fatal: a run must not die because
    a status file could not be written."""
    try:
        tmp = BEAT_FILE + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(kw, fh)
        os.replace(tmp, BEAT_FILE)
    except OSError:
        pass


def _beat_clear():
    try:
        os.remove(BEAT_FILE)
    except OSError:
        pass


def _interleave(rows):
    """Cycle the drives, keeping each drive's own order.

    Sorting by drive alone put every shard on ONE disk, so five of the six sat
    idle while the sixth was read flat out -- measured at 48 images/s cold
    against 185 warm, and the gap is seek latency nobody else was covering.
    Cycling the drives keeps all of them reading at once while each still
    walks its own cells in order, which is what makes the sequential part
    sequential.
    """
    lanes, order = {}, []
    for r in rows:
        drive = r[8]
        if drive not in lanes:
            lanes[drive] = []
            order.append(drive)
        lanes[drive].append(r)
    # by image, not by row: the boxes of one frame must stay together or it is
    # decoded once per box
    per = {d: [] for d in order}
    for d in order:
        cur, key = [], None
        for r in lanes[d]:
            if r[0] != key:
                if key is not None:
                    per[d].append(cur)
                cur, key = [], r[0]
            cur.append(r)
        if key is not None:
            per[d].append(cur)
    # In BLOCKS, not one image at a time. Cycling per image put six drives in
    # every worker's chunk, so each one seeked between disks for every frame:
    # measured at 50 images/s with 48% of the machine sitting in iowait. A
    # block is long enough that a worker stays on one drive, walking its cells
    # in order, while the other drives are walked by other workers.
    out, idx = [], {d: 0 for d in order}
    while True:
        moved = False
        for d in order:
            i, lane = idx[d], per[d]
            if i < len(lane):
                for grp in lane[i:i + INTERLEAVE_BLOCK]:
                    out.extend(grp)
                idx[d] = i + INTERLEAVE_BLOCK
                moved = True
        if not moved:
            break
    return out


def upstream_state(stage=None):
    """(judged, planned, dir) for the stage this one feeds on, or None.

    A stage that reads another's output can only be planned once that output
    is complete: planning the leash model against a half-written gate would
    build a work list missing every dog the gate has not reached yet, and the
    shard numbering is fixed at plan time, so there is no adding to it later.
    """
    up = spec(stage)['feeds_on']
    if not up:
        return None
    d = os.path.join(stages_root(), STAGES[up]['dir'])
    planned = 0
    try:
        with open(os.path.join(d, 'plan.json')) as fh:
            planned = int(json.load(fh).get('rows') or 0)
    except (OSError, ValueError, TypeError):
        pass
    judged = 0
    try:
        import glob
        import pyarrow.parquet as pq
        judged = sum(pq.ParquetFile(f).metadata.num_rows for f in
                     glob.glob(os.path.join(d, f'{STAGES[up]["dir"]}-*.parquet')))
    except Exception:
        pass
    return judged, planned, d


def plan(args):
    """Every detection to judge, ordered for locality, split into shards."""
    sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
    import duckdb
    import store
    sp = spec()
    up = upstream_state()
    if up and not args.partial:
        judged, planned, updir = up
        if not planned or judged < planned:
            raise SystemExit(
                f'the {STAGES[sp["feeds_on"]]["title"]} has judged '
                f'{judged:,} of {planned:,} boxes -- the {sp["title"]} reads '
                f'its output, and a plan built now would permanently omit '
                f'every dog it has not reached yet (shard numbering is fixed '
                f'at plan time). Wait for it to finish, or pass --partial to '
                f'plan over what exists so far and accept that.')
    # Every box the upstream stage marked with its positive class, and only
    # those. The gate's verdicts live in their own store, joined back to the
    # detection rows for the coordinates -- the shards carry a verdict and a
    # probability, never a box.
    narrow, params = '', [f'{args.gen:04d}']
    if up:
        pre = STAGES[sp['feeds_on']]['dir']
        glob_sql = os.path.join(up[2], f'{pre}-*.parquet').replace("'", "''")
        narrow = (f"JOIN read_parquet('{glob_sql}') u "
                  f"ON u.image_id = CAST(d.image_id AS VARCHAR) "
                  f"AND u.det_idx = d.det_idx "
                  f"AND u.label = '{STAGES[sp['feeds_on']]['positive']}'")
    root = store.get_detect_root()
    img = store._sql_src(store._store_globs(root, 'img'))
    det = store._sql_src(store._store_globs(root, 'det'))
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{args.memory}'")
    rows = con.execute(f"""
        SELECT CAST(d.image_id AS VARCHAR), d.det_idx,
               d.x1, d.y1, d.x2, d.y2, d.conf, i.cell, i.drive
        FROM {det} d
        JOIN {img} i ON i.image_id = d.image_id AND i.gen = d.gen
                    AND i.cell = d.cell
        {narrow}
        WHERE d.gen = ?
        -- ONE row per box. A frame that was harvested into two cells on two
        -- drives has a detection row under each, same image_id and same
        -- det_idx: 47,320 boxes on this store, 97,380 rows. Left in, the
        -- output would carry duplicate keys and every join downstream would
        -- double-count them -- and the frame would be decoded twice for the
        -- same answer. Which copy is kept does not matter, only that it is
        -- always the same one.
        QUALIFY row_number() OVER (
            PARTITION BY d.image_id, d.det_idx
            ORDER BY i.drive, i.cell) = 1
        -- within a drive, one directory at a time: a worker walks a cell
        -- rather than seeking around it
        ORDER BY i.drive, i.cell, d.image_id, d.det_idx
    """, params).fetchall()
    con.close()
    rows = _interleave(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    doc = {'gen': args.gen, 'rows': len(rows),
           'images': len({r[0] for r in rows}),
           'stage': STAGE, 'model': sp['title'],
           # the runner cannot look these up for itself; see _roots()
           'roots': _resolve_roots(),
           'shard_rows': SHARD_ROWS, 'created': time.strftime('%F %T'),
           # which store's boxes this list names: det_idx means nothing
           # without it
           'detect_root': os.path.abspath(root)}
    with open(PLAN_FILE + '.tmp', 'w') as fh:
        json.dump(doc, fh)
        fh.write('\n')
    os.replace(PLAN_FILE + '.tmp', PLAN_FILE)
    # the work itself, once, so `run` does not re-query a 4.8M-row join
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = list(zip(*rows)) if rows else [[]] * 9
    pq.write_table(pa.table({
        'image_id': pa.array(cols[0], pa.string()),
        'det_idx': pa.array(cols[1], pa.int32()),
        'x1': pa.array(cols[2], pa.float32()),
        'y1': pa.array(cols[3], pa.float32()),
        'x2': pa.array(cols[4], pa.float32()),
        'y2': pa.array(cols[5], pa.float32()),
        'conf': pa.array(cols[6], pa.float32()),
        'cell': pa.array(cols[7], pa.string()),
        'drive': pa.array(cols[8], pa.string())}),
        os.path.join(OUT_DIR, 'work.parquet'))
    print(f"{doc['rows']:,} boxes across {doc['images']:,} images "
          f"-> {os.path.relpath(OUT_DIR, REPO)}/work.parquet")
    return 0


def _shard_size(want):
    """The shard size this job is committed to, pinned on first use."""
    mark = os.path.join(OUT_DIR, 'shard_size')
    try:
        with open(mark) as fh:
            pinned = int(fh.read().strip())
    except (OSError, ValueError):
        pinned = None
    if pinned is None:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(mark, 'w') as fh:
            fh.write(str(want))
        return want
    if pinned != want and done_shards():
        raise SystemExit(
            f'this job was sharded at {pinned} and {len(done_shards())} '
            f'shards are already written; --shard {want} would renumber them '
            f'and skip work that was never done. Use --shard {pinned}, or '
            f"delete data/{spec()['dir']}/ to start over.")
    return pinned


def shard_name(i):
    return f"{spec()['dir']}-{i:05d}.parquet"


def stop_after(shards, skip, limit):
    """Shard index a --limit run stops at, or None for the whole job.

    In WHOLE SHARDS, counted over the boxes this run would actually do. A
    shard is the smallest thing that can be written and its index is its
    identity, so "do exactly `limit` boxes" is not a thing a run can honour --
    the honest reading is "stop once that many boxes of NEW work are covered".
    Counted past `skip` so a trial run on a resumed job does some work instead
    of stopping before the first undone shard.

    Args:
        shards: [[job, ...], ...] -- jobs, in shard order, over the WHOLE job.
        skip: indices already on disk.
        limit: boxes, or 0/None for no limit.
    """
    if not limit:
        return None
    spent = 0
    for si, shard in enumerate(shards):
        if si in skip:
            continue
        spent += sum(len(j[3]) for j in shard)
        if spent >= limit:
            return si + 1
    return len(shards)


def check_job_count(n_jobs, plan_images):
    """'' or the message refusing a job list that is not the whole plan.

    The anchor the shard arithmetic cannot supply for itself. If the work rows
    are truncated before they are grouped -- which is exactly what `--limit`
    used to do -- then `jobs`, `shards` and "the last shard" are all short
    together and agree with each other perfectly; refuse_short_shard() sees a
    short shard that IS the last one it knows about and waves it through. Only
    a number recorded before any of that can tell: plan() counted the images
    when it wrote work.parquet, so a run whose grouping disagrees with it is
    working from a smaller job and must not write a shard at all.
    """
    if plan_images is None or n_jobs == plan_images:
        return ''
    return (f'the work list grouped into {n_jobs:,} images but the plan '
            f'records {plan_images:,}. Shards are cut over this list and a '
            f'shard index is permanent: a run over part of it writes short '
            f'shards under full indices, and done_shards() then skips them '
            f'forever. --limit is a stopping point, never a smaller job.')


def refuse_short_shard(si, n_jobs, n_shards, size):
    """Message refusing to commit a short shard under a full index, or ''.

    A shard's index IS its identity: once <stage>-000NN.parquet is on disk,
    done_shards() skips NN forever. So a shard cut over a truncated job list
    -- which is what `--limit` used to do, because the limit was applied to
    the work rows before the shards were cut -- is written short under the
    index a full run gives 20,000 images, and the rest of it is never judged
    by anything, ever, with nothing reporting a gap. Only the job's true last
    shard may be short.
    """
    if n_jobs == size or si == n_shards - 1:
        return ''
    return (f'refusing to write {shard_name(si)}: it holds {n_jobs:,} images '
            f'where shard {si} of this job is {size:,}. Committing it would '
            f'mark the index done and {size - n_jobs:,} images would never be '
            f'judged. The job was cut over a truncated work list -- rerun '
            f'`plan`, or delete data/{spec()["dir"]}/ and start over.')


def done_shards():
    pre = spec()['dir'] + '-'
    try:
        return {int(f[len(pre):].split('.')[0])
                for f in os.listdir(OUT_DIR)
                if f.startswith(pre) and f.endswith('.parquet')}
    except (OSError, ValueError):
        return set()


# ── the worker: decode one frame, cut every box on it ──────────────────────
_ROOTS = {}
# classify_spec() for the fast path; None hands back PIL crops for predict()
_PREP = None


def _init(roots, prep=None):
    global _ROOTS, _PREP
    _ROOTS = roots
    _PREP = prep


def cut_crops(im, boxes):
    """[(det_idx, x1, y1, x2, y2)] on a decoded RGB frame -> [(det_idx, crop)].

    Each crop is the padded box at FULL resolution -- never resized here.
    The models were trained and accepted on crop files that ultralytics
    resized itself (shortest side to the train imgsz, then centre-cropped), so
    handing predict() the same un-resized crop is what makes a store verdict
    the verdict the acceptance numbers describe. The old path squashed every
    crop to 224x224 and predicted at 224 against models trained at 640;
    measured on dogbin_008, that took v5-val dog recall@0.5 from 0.936 to
    0.330 and leash_v2_001 acceptance balanced accuracy from 0.895 to 0.718.
    """
    w, h = im.size
    out = []
    for di, x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        pad = int(PAD_FRAC * max(x2 - x1, y2 - y1)) + PAD_PX
        a, b = max(0, x1 - pad), max(0, y1 - pad)
        c, d = min(w, x2 + pad), min(h, y2 + pad)
        if c - a < MIN_SIDE or d - b < MIN_SIDE:
            continue
        # a crop, not the frame: a decoded 8000x4000 frame is 96 MB and must
        # never cross the process boundary. crop() is lazy, so load() it
        # before the frame is closed.
        cr = im.crop((a, b, c, d))
        cr.load()
        out.append((di, cr))
    return out


def _cut(task):
    """(shard, image_id, [(det_idx, box)], jpeg bytes, err) ->
    (shard, image_id, [(det_idx, crop)], err).

    The bytes were read by the drive's reader thread in the parent (see
    DriveLanes), so a decoder never waits on a disk. A crop is a PIL image
    for --legacy-predict and otherwise prep_crop()'s uint8 array: the resize
    to 640 was the largest piece of the parent's per-crop work, and here it
    runs on sixteen cores instead of one.
    """
    import io
    from PIL import Image
    si, iid, boxes, raw, err = task
    if err:
        return si, iid, [], err
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
        if im.mode != 'RGB':
            im = im.convert('RGB')
    except Exception:
        # its own kind: the bytes came off the disk, so the drive is fine --
        # a corrupt file, not a mount gone away (READ_ERRORS)
        return si, iid, [], 'undecodable'
    if _PREP is not None and _PREP.get('kind') == VIT_KIND:
        out = cut_vit(im, boxes, _PREP)
        im.close()
        return si, iid, out, ''
    out = cut_crops(im, boxes)
    im.close()
    if _PREP is not None:
        out = [(di, prep_crop(c, _PREP)) for di, c in out]
    return si, iid, out, ''


# ── the ViT leash model: wider crops, letterboxed in the decoder ───────────
# leash_vit.KIND, repeated so a decoder needs no import to dispatch on it
VIT_KIND = 'dinov3-leash'


def cut_vit(im, boxes, vspec):
    """[(det_idx, x1, y1, x2, y2)] -> [(det_idx, res x res x 3 uint8)] for
    the ViT leash model.

    The SAME boxes as cut_crops() judges -- a box whose ordinary padded crop
    is under MIN_SIDE is skipped here too, so both models answer for one set
    of boxes -- but each is cut in the model's own framing (leash_vit.rect_pad:
    the box grown by half its long side) and turned into the letterboxed
    picture the model sees, here in the decoder process.
    """
    import leash_vit
    w, h = im.size
    keep = {di for di, _ in cut_crops_rects(w, h, boxes)}
    out = []
    for di, x1, y1, x2, y2 in boxes:
        if di not in keep:
            continue
        a, b, c, d = leash_vit.rect_pad((x1, y1, x2, y2), w, h,
                                        vspec['frac'])
        if c <= a or d <= b:
            # a box under 1 px wide or tall: cut_crops' fixed PAD_PX keeps
            # it, the model's framing has nothing to cut (none in the gen-2
            # work list; skipped rather than crash the whole run)
            continue
        cr = im.crop((a, b, c, d))
        cr.load()
        out.append((di, leash_vit.prep_crop(cr, vspec['res'])))
    return out


def cut_crops_rects(w, h, boxes):
    """[(det_idx, (a, b, c, d))]: cut_crops()' rectangles and its MIN_SIDE
    rule, without cutting anything."""
    out = []
    for di, x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        pad = int(PAD_FRAC * max(x2 - x1, y2 - y1)) + PAD_PX
        a, b = max(0, x1 - pad), max(0, y1 - pad)
        c, d = min(w, x2 + pad), min(h, y2 + pad)
        if c - a < MIN_SIDE or d - b < MIN_SIDE:
            continue
        out.append((di, (a, b, c, d)))
    return out


def vit_cut_one(vspec, frame, box):
    """One box off one frame file through the run's own path (read_frame,
    the decoder's decode, cut_vit) -> uint8 array, or None if not cut. What
    the parity check re-cuts its crops with."""
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(read_frame(frame)))
    im.load()
    if im.mode != 'RGB':
        im = im.convert('RGB')
    got = cut_vit(im, [(0,) + tuple(box)], vspec)
    im.close()
    return got[0][1] if got else None


def frame_path(root, cell, iid):
    return os.path.join(root, cell, 'ground_animal_images', f'{iid}.jpg')


def read_frame(path):
    """A frame's bytes, dropped from the page cache once read: 7 TB of JPEG
    passes through here, and none of it is read twice."""
    with open(path, 'rb') as fh:
        raw = fh.read()
        try:
            os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    return raw


# ── the fast path: ultralytics' classify preprocessing, split in two ───────
# predict() on a list of PIL crops does, per crop, in the parent process:
# PIL -> BGR array -> RGB PIL, Resize(short side 640), CenterCrop(640),
# ToTensor, Normalize, then a Results object per crop. Measured at ~3.4 ms a
# crop on one core, in series with the GPU. The fast path keeps every pixel
# of that and moves it: the geometry (Resize + CenterCrop, which is PIL work)
# runs in the decoder that cut the crop, the value mapping (ToTensor +
# Normalize, which is a function of one 8-bit value and its channel) runs on
# the GPU through a 256-entry table computed BY those transforms, and the
# forward is the predictor's own fused AutoBackend. run() checks the result
# against predict() on the live device before it judges anything.

def classify_spec(transforms, imgsz):
    """The geometry half of the predictor's classify transform, as plain data
    a decoder can apply with PIL alone.

    Raises ValueError for anything but the Resize(short side) -> CenterCrop
    -> ToTensor -> Normalize chain ultralytics builds (classify_transforms),
    at `imgsz`, with a crop no larger than the short side -- the only shape
    prep_crop() reproduces. The caller refuses rather than approximates.
    """
    ts = list(getattr(transforms, 'transforms', None) or [])
    names = [type(t).__name__ for t in ts]
    if names != ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']:
        raise ValueError(f'the predictor transforms are {names}, not the '
                         f'Resize/CenterCrop/ToTensor/Normalize chain the '
                         f'fast path reproduces')
    rs, cc = ts[0], ts[1]
    size = rs.size
    if isinstance(size, (list, tuple)) and len(size) == 1:
        size = size[0]
    if not isinstance(size, int) or getattr(rs, 'max_size', None) is not None:
        raise ValueError(f'Resize(size={rs.size!r}, max_size='
                         f'{getattr(rs, "max_size", None)!r}) is not a '
                         f'short-side resize')
    if size != int(imgsz):
        raise ValueError(f'the predictor resizes to {size}, not imgsz {imgsz}')
    crop = cc.size
    if isinstance(crop, int):
        crop = (crop, crop)
    crop = tuple(int(v) for v in crop)
    if len(crop) != 2 or crop[0] > size or crop[1] > size:
        raise ValueError(f'CenterCrop{crop} would pad a {size}px short side')
    from torchvision.transforms.functional import pil_modes_mapping
    return {'short': size, 'crop': crop,
            'resample': int(pil_modes_mapping[rs.interpolation])}


def prep_crop(crop, spec):
    """PIL RGB crop -> HxWx3 uint8 array, pixel for pixel what torchvision's
    Resize + CenterCrop make of it (transforms/functional.py: resize via
    _compute_resized_output_size, then center_crop), with PIL alone so a
    decoder process never imports torch."""
    import numpy as np
    w, h = crop.size
    s = spec['short']
    short, long = (w, h) if w <= h else (h, w)
    new_short, new_long = s, int(s * long / short)
    nw, nh = (new_short, new_long) if w <= h else (new_long, new_short)
    if (nh, nw) != (h, w):
        crop = crop.resize((nw, nh), spec['resample'])
    ch, cw = spec['crop']
    if (ch, cw) != (nh, nw):
        top = int(round((nh - ch) / 2.0))
        left = int(round((nw - cw) / 2.0))
        crop = crop.crop((left, top, left + cw, top + ch))
    return np.array(crop, dtype=np.uint8)


def value_lut(transforms):
    """(3, 256) float tensor: what the predictor's ToTensor + Normalize make
    of each 8-bit value in each channel, computed by those transforms
    themselves on the CPU. Both are per-element, per-channel maps, so a
    lookup in this table is the transform -- on any device, bit for bit."""
    import numpy as np
    from PIL import Image
    ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
    x = Image.fromarray(ramp, 'RGB')
    for t in list(transforms.transforms)[2:]:
        x = t(x)
    return x[:, 0, :].contiguous()


def batch_input(arrs, lut, half=False):
    """[HxWx3 uint8] -> the (B, 3, H, W) tensor the predictor would build, on
    `lut`'s device. Eight bits a value cross the bus instead of 32."""
    import numpy as np
    import torch
    u8 = torch.from_numpy(np.stack(arrs))
    if lut.device.type == 'cuda':
        u8 = u8.pin_memory().to(lut.device, non_blocking=True)
    u8 = u8.permute(0, 3, 1, 2)
    x = torch.empty(u8.shape, dtype=lut.dtype, device=lut.device)
    for c in range(3):
        x[:, c] = lut[c][u8[:, c].long()]
    # ClassificationPredictor.preprocess: half when the backend is fp16
    return x.half() if half else x.float()


def forward_probs(backend, x):
    """The predictor's inference + postprocess without the Results objects:
    the AutoBackend forward under inference mode, first output (the softmax
    the Classify head returns in eval). Asynchronous on CUDA until read."""
    import torch
    with torch.inference_mode():
        y = backend(x)
    return y[0] if isinstance(y, (list, tuple)) else y


def probe_crops():
    """Deterministic synthetic crops in the shapes the store holds: small
    upscaled boxes, wide and tall ones, one already at the short side."""
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(20261115)
    out = []
    for w, h in ((93, 98), (166, 61), (40, 220), (640, 700), (13, 9),
                 (301, 300), (1200, 800), (97, 641)):
        yy, xx = np.mgrid[0:h, 0:w]
        a = np.stack([(xx * 7) % 256, (yy * 5) % 256, (xx + yy) % 256], -1)
        a = (a + rng.randint(0, 40, a.shape)) % 256
        out.append(Image.fromarray(a.astype(np.uint8), 'RGB'))
    return out


def fast_path_check(model, spec, lut, imgsz, device, batch, half=False,
                    tol=1e-5):
    """Max |p_fast - p_predict| over probe_crops(), in batches of `batch` on
    the run's own device. Raises ValueError when they disagree past `tol`
    or flip a verdict at 0.5 -- the run must not start on a path that judges
    differently from the one the acceptance numbers describe."""
    crops = probe_crops()
    worst = 0.0
    for i in range(0, len(crops), batch):
        part = crops[i:i + batch]
        ref = [r.probs.data.float().cpu() for r in model.predict(
            part, verbose=False, device=device, imgsz=imgsz)]
        got = forward_probs(model.predictor.model, batch_input(
            [prep_crop(c, spec) for c in part], lut, half)).float().cpu()
        for j, r in enumerate(ref):
            d = float((got[j] - r).abs().max())
            worst = max(worst, d)
            if d > tol or bool(((got[j] >= 0.5) != (r >= 0.5)).any()):
                raise ValueError(
                    f'probe crop {i + j} ({part[j].size[0]}x{part[j].size[1]})'
                    f': fast path {got[j].tolist()} vs predict() '
                    f'{r.tolist()}')
    return worst


def drive_map(pairs, flag):
    """['jackal=2', 'bobcat=8'] -> {'jackal': 2, 'bobcat': 8}; a malformed
    pair is an error, never a silent default."""
    out = {}
    for pr in pairs or ():
        name, sep, val = str(pr).partition('=')
        try:
            n = int(val) if sep and name else 0
        except ValueError:
            n = 0
        if n <= 0:
            raise SystemExit(f'{flag}: expected drive=N with N > 0, '
                             f'got {pr!r}')
        out[name] = n
    return out


def reader_counts(pairs, drives):
    """{drive: reader threads} from READERS and `--readers drive=N`."""
    want = dict(READERS, **drive_map(pairs, '--readers'))
    return {d: want.get(d, READERS_DEFAULT) for d in sorted(drives)}


class DriveLanes:
    """Per-drive reader threads in front of the decoder pool.

    The old feed was one imap over each shard's list, in its fixed
    drive-interleaved order, with the decoders reading their own frames: ~512
    images of look-ahead pushed every drive to the same image rate, so 4-8
    decoders queued on each slow disk while the fast one waited, and the pool
    drained to empty at every shard boundary. Here each drive is walked in
    plan order by its own readers, as fast as that drive goes, and the bytes
    go to whichever decoder is free. The stream runs across shards, so no
    boundary drains it; a drive may run up to `ahead` shards past the lowest
    unfinished one, and `slots` bounds the frames held between a reader and
    the parent's loop, so a GPU that falls behind slows the readers instead
    of filling memory.

    feed() yields exactly one task per job; consumed() must be called once
    per result.
    """

    def __init__(self, shards, todo, roots, readers, ahead=AHEAD, slots=64,
                 disk_order=None):
        import queue
        import threading
        self.shards, self.todo, self.roots = shards, list(todo), roots
        self.disk_order = dict(disk_order or {})
        self.rank = {si: k for k, si in enumerate(self.todo)}
        self.left = {si: len(shards[si]) for si in self.todo}
        self.total = sum(self.left.values())
        self.ahead, self.lo = max(0, int(ahead)), 0
        self.cond = threading.Condition()
        self.slots = threading.BoundedSemaphore(max(1, int(slots)))
        self.q = queue.Queue()
        self.stop = threading.Event()
        self.threads = []
        drives = {j[2] for si in self.todo for j in shards[si]}
        # frames each drive has delivered: the heartbeat carries them, so the
        # lane that holds the run back is visible without a profiler
        self.read = {d: 0 for d in sorted(drives)}
        self._count = threading.Lock()
        for d in sorted(drives):
            it, lock = self._lane(d), threading.Lock()
            for _ in range(max(1, int(readers.get(d, READERS_DEFAULT)))):
                t = threading.Thread(target=self._reader, args=(d, it, lock),
                                     daemon=True)
                self.threads.append(t)

    def _lane(self, drive):
        """(shard, job) for one drive, in plan order -- or, with a disk-order
        window, sorted by where the files sit on the platter. Measured on this
        plan, consecutive frames of a drive's lane are a median 78-1500 GiB
        apart on disk, so every read is a long seek; sorted, the median jump
        is 0.04-0.14 GiB. The sort never crosses a shard boundary: the lane
        stays monotone in shard, which is what keeps the look-ahead from
        deadlocking (a reader holding a far shard's frame can then never be
        what the lowest shard is waiting on)."""
        win = int(self.disk_order.get(drive) or 0)
        root = self.roots.get(drive)
        for si in self.todo:
            mine = [job for job in self.shards[si] if job[2] == drive]
            if win > 1 and root and mine:
                from sweep import disk_ordered
                items = (((si, j), frame_path(root, j[1], j[0]))
                         for j in mine)
                for meta, _ in disk_ordered(items, win, stop=self.stop):
                    yield meta
            else:
                for job in mine:
                    yield si, job

    def _reader(self, drive, it, lock):
        try:
            while not self.stop.is_set():
                with lock:
                    item = next(it, None)
                if item is None:
                    return
                si, (iid, cell, _, boxes) = item
                with self.cond:
                    while (not self.stop.is_set()
                           and self.rank[si] > self.lo + self.ahead):
                        self.cond.wait(1.0)
                while not self.slots.acquire(timeout=1.0):
                    if self.stop.is_set():
                        return
                root = self.roots.get(drive)
                raw, err = None, ''
                if not root:
                    err = 'no_root'
                else:
                    try:
                        raw = read_frame(frame_path(root, cell, iid))
                    except OSError:
                        err = 'unreadable'
                with self._count:
                    self.read[drive] += 1
                self.q.put((si, iid, boxes, raw, err))
        except BaseException as e:       # surfaced through feed(), never lost
            self.q.put(e)

    def feed(self):
        import queue
        for t in self.threads:
            t.start()
        for _ in range(self.total):
            while True:
                try:
                    item = self.q.get(timeout=1.0)
                    break
                except queue.Empty:
                    if self.stop.is_set():
                        return
            if isinstance(item, BaseException):
                raise item
            yield item

    def consumed(self, si):
        """One result is in the parent's hands: free its slot, and move the
        look-ahead on when it finished the lowest shard's reads."""
        self.slots.release()
        with self.cond:
            self.left[si] -= 1
            moved = False
            while (self.lo < len(self.todo)
                   and self.left[self.todo[self.lo]] == 0):
                self.lo += 1
                moved = True
            if moved:
                self.cond.notify_all()

    def lowest(self):
        """The lowest shard this run still has frames to take for."""
        return self.todo[min(self.lo, len(self.todo) - 1)] if self.todo else 0


def model_imgsz(model):
    """The imgsz a classifier was TRAINED at, from its own checkpoint.

    Never a constant here: a constant is what let the runner predict at 224
    for months against models trained at 640. None when the checkpoint does
    not say, and the caller refuses rather than guessing.
    """
    for src in ((getattr(model, 'ckpt', None) or {}).get('train_args'),
                getattr(model, 'overrides', None)):
        try:
            v = (src or {}).get('imgsz')
        except AttributeError:
            v = None
        if isinstance(v, (list, tuple)) and v:
            v = max(v)
        if isinstance(v, int) and v > 0:
            return v
    return None


def auto_batch(imgsz, at224=128):
    """Crops per predict call, holding pixels per batch to what 128 crops at
    224 used. Measured: --batch 128 at imgsz 640 asks for 12.8 GB and OOMs
    the 16 GB card."""
    v = int(at224 * (224 / imgsz) ** 2 + 0.5)
    return v // 8 * 8 if v >= 8 else max(1, v)


# How the verdicts in a stage directory were produced. Pinned on a job's first
# run, like the shard size: shards from two protocols in one directory would
# read as one model's opinion when they are two different measurements.
PROTOCOL_FILE = 'protocol.json'
CROP_MODE = 'full'


def check_protocol(want, out_dir=None, shards=None):
    """'' or the message refusing to mix protocols in one stage directory.

    `want` is {'crop': ..., 'imgsz': ...}. Shards already on disk with no
    protocol file were written by the legacy runner (224x224 squash, predict
    at 224), so they are refused too: resuming that job would append 640
    verdicts to 224 ones under one model name.
    """
    d = out_dir or OUT_DIR
    have_shards = done_shards() if shards is None else shards
    try:
        with open(os.path.join(d, PROTOCOL_FILE)) as fh:
            pinned = json.load(fh)
    except (OSError, ValueError):
        pinned = None
    if pinned is None and have_shards:
        pinned = {'crop': 'squash224', 'imgsz': 224, 'legacy': True}
    if pinned is None or not have_shards:
        return ''
    if {k: pinned.get(k) for k in want} == want:
        return ''
    return (f'{len(have_shards)} shards in {os.path.relpath(d, REPO)}/ were '
            f'written with crop={pinned.get("crop")} imgsz={pinned.get("imgsz")}'
            f'{" (the legacy runner)" if pinned.get("legacy") else ""}; this run '
            f'would add crop={want["crop"]} '
            f'imgsz={want["imgsz"] or "<as trained>"} to the same '
            f'record. Set the old verdicts aside first: '
            + (f'`gate_store.py archive --stage {STAGE} --replan`, then '
               f'`plan` again.' if STAGES[STAGE]['feeds_on'] else
               f'`gate_store.py archive --stage {STAGE}`.'))


def pin_protocol(want, out_dir=None):
    d = out_dir or OUT_DIR
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, PROTOCOL_FILE + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(want, fh)
        fh.write('\n')
    os.replace(tmp, os.path.join(d, PROTOCOL_FILE))


def protocol_tag(out_dir):
    """'<crop>@<imgsz>' for the verdicts in a stage directory: 'legacy' for a
    record written before protocol.json existed (224 squash), '' for none."""
    try:
        with open(os.path.join(out_dir, PROTOCOL_FILE)) as fh:
            p = json.load(fh)
        return f"{p.get('crop')}@{p.get('imgsz')}"
    except (OSError, ValueError, AttributeError):
        pre = os.path.basename(os.path.normpath(out_dir)) + '-'
        try:
            legacy = any(f.startswith(pre) and f.endswith('.parquet')
                         for f in os.listdir(out_dir))
        except OSError:
            legacy = False
        return 'legacy' if legacy else ''


def stale_plan(stage=None, repo=None):
    """'' or why this stage's work list is older than what it was built from.

    The leash list is the gate's dogs at plan time. Re-run the gate -- with a
    new model, or at the right imgsz -- and the list still names the OLD
    gate's dogs; a leash run over it writes a complete-looking record that
    never judges any dog only the new gate found, and nothing reports the
    gap. The dashboard's re-run replans on the same test (_gate_stale_plan);
    this is the one the CLI and the runner apply.
    """
    stage = stage or STAGE
    up = STAGES[stage]['feeds_on']
    if not up:
        return ''
    d = _stage_dir(stage, repo)
    try:
        made = os.path.getmtime(os.path.join(d, 'work.parquet'))
    except OSError:
        return ''
    ud = _stage_dir(up, repo)
    newest = 0.0
    for f in _verdict_files(up, repo):
        try:
            newest = max(newest, os.path.getmtime(os.path.join(ud, f)))
        except OSError:
            pass
    if newest > made:
        return (f'the {STAGES[stage]["title"]} work list was built '
                f'{time.strftime("%F %T", time.localtime(made))}, before the '
                f'{STAGES[up]["title"]} verdicts it reads were last written '
                f'({time.strftime("%F %T", time.localtime(newest))})')
    return ''


def live_runner(out_dir=None):
    """The pid of a runner still writing this stage, or None. Read off the
    heartbeat, and only believed if that pid is alive AND is a gate_store
    process -- a stale file names a pid the kernel may have reused."""
    d = out_dir or OUT_DIR
    try:
        with open(os.path.join(d, 'progress.json')) as fh:
            pid = int(json.load(fh).get('pid') or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    if pid <= 0 or pid == os.getpid():
        return None
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as fh:
            cmd = fh.read().replace(b'\0', b' ').decode('utf-8', 'replace')
    except OSError:
        return None
    return pid if 'gate_store' in cmd else None


# Frames a shard may lose to read errors before it is refused. The gen-2 gate
# read all 5,155,473 planned frames without one error, so a shard short of
# frames is a drive that went away, not a few bad files: bobcat dropped off
# USB mid-run on 2026-11-16 and every one of its frames came back
# 'unreadable' -- skipped, and the shards written as complete without them.
MAX_UNREADABLE = 20


def unreadable_roots(roots, drives):
    """'' or why the run cannot start: a drive this run reads whose root is
    missing or cannot be listed (a dead mount answers with EIO)."""
    bad = []
    for d in sorted(drives):
        root = roots.get(d)
        try:
            if not root:
                raise FileNotFoundError('no root in the plan')
            with os.scandir(root) as it:
                next(it, None)
        except OSError as e:
            bad.append(f'{d} ({root}): {e.strerror or e}\n    '
                       + mount_hint(root))
    if not bad:
        return ''
    return ('cannot read the frames of\n  ' + '\n  '.join(bad)
            + '\nmount the drive and run again')


def mount_point(path):
    """The mount point a path lives under (its nearest ancestor that is
    one), or None."""
    p = os.path.abspath(path or '')
    while p and p != os.path.dirname(p):
        try:
            if os.path.ismount(p):
                return p
        except OSError:
            pass
        p = os.path.dirname(p)
    return None


def mount_hint(root):
    """How to bring back the mount a drive root lives under. A USB drive
    that drops and comes back gets a new device node while the old mount
    stays behind dead (every read is EIO), so it is unmounted first. Without
    an /etc/fstab line `mount <point>` alone does not work."""
    mp = mount_point(root)
    if not mp or mp == '/':
        return f'(no mount point found for {root}: is the drive attached?)'
    try:
        with open('/etc/fstab') as fh:
            fstab = [ln.split() for ln in fh
                     if ln.strip() and not ln.lstrip().startswith('#')]
    except OSError:
        fstab = []
    if any(len(f) > 1 and f[1] == mp for f in fstab):
        again = f'sudo mount {mp}'
    else:
        again = (f'sudo mount /dev/disk/by-label/<label> {mp}   '
                 f'(no /etc/fstab entry for {mp})')
    return f'remount: sudo umount -l {mp} ; {again}'


# Error kinds the drive READERS report: the frame never came off the disk.
# A shard with any of these is written only once its drives answer again.
READ_ERRORS = ('unreadable', 'no_root')


def short_done_shards(skip, shards, limit):
    """[(shard, images in it, images planned)] for finished shards holding
    more than `limit` fewer images than their plan -- written while a drive
    was gone (2026-11-16), and skipped by every resume since done_shards()
    goes by index. Every planned image of the leash stage has a row (its
    boxes are the gate's dogs, all past MIN_SIDE); a gate image whose boxes
    are all under MIN_SIDE has none, which `limit` absorbs."""
    import pyarrow.parquet as pq
    out = []
    for si in sorted(skip):
        if si >= len(shards):
            continue
        try:
            col = pq.read_table(os.path.join(OUT_DIR, shard_name(si)),
                                columns=['image_id'])['image_id']
            have = len(set(col.to_pylist()))
        except Exception:
            have = -1
        if have < len(shards[si]) - limit:
            out.append((si, have, len(shards[si])))
    return out


def refuse_unreadable_shard(si, kinds, frames, limit):
    """'' or why shard `si` must not be written: more than `limit` of its
    frames could not be read. Its boxes would be missing from a shard that
    reads as complete, and done_shards() would never revisit them."""
    n = sum(kinds.values())
    if n <= limit:
        return ''
    return (f'shard {si}: {n:,} of {frames:,} frames unreadable '
            f'({", ".join(f"{k} {v:,}" for k, v in sorted(kinds.items()))}) '
            f'-- not written. A drive has most likely gone away: check the '
            f'mounts and run again (finished shards are kept; this one is '
            f'redone). --max-unreadable raises the limit ({limit}).')


def run(args):
    import multiprocessing as mp
    import pyarrow as pa
    import pyarrow.parquet as pq

    vit_dir = os.path.abspath(args.vit_model) if getattr(
        args, 'vit_model', None) else None
    if vit_dir and STAGE != 'leash':
        raise SystemExit('--vit-model is a leash model: --stage leash')
    if not os.path.exists(os.path.join(OUT_DIR, 'work.parquet')):
        raise SystemExit('no plan yet -- run `gate_store.py plan` first')
    # Both refusals come BEFORE the model and the 4.7M-row work list load: the
    # dashboard waits ~2.5 s after spawning a run to see whether it died, and
    # a refusal that lands after that is reported as a run that started.
    if done_shards() and protocol_tag(OUT_DIR) == 'legacy':
        raise SystemExit(check_protocol({'crop': CROP_MODE, 'imgsz': None}))
    _no = stale_plan()
    if _no:
        raise SystemExit(f'{_no}. Re-plan first: `gate_store.py archive '
                         f'--stage {STAGE} --replan`, then `plan`.')
    work = pq.read_table(os.path.join(OUT_DIR, 'work.parquet')).to_pydict()
    n = len(work['image_id'])

    sp = spec()
    pos = sp['positive']
    other = next(c for c in sp['classes'] if c != pos)
    model = vspec = dog_idx = None
    if vit_dir:
        # The fine-tuned DINOv3 leash model (leash_vit.py): its own framing
        # and resolution, read from the model directory, and its own batch --
        # the one it was validated at, because under bf16 the batch shape
        # moves p. Loaded below, after the shard bookkeeping, like YOLO.
        import leash_vit
        try:
            vcfg = leash_vit.read_config(vit_dir)
            vspec = leash_vit.spec(vit_dir)
        except (OSError, ValueError, KeyError) as e:
            raise SystemExit(f'--vit-model {vit_dir}: {e}')
        if not os.path.isfile(os.path.join(vit_dir, 'parity.json')):
            raise SystemExit(f'--vit-model {vit_dir}: no parity.json -- the '
                             f'run cannot prove it reproduces the model')
        run_name = os.path.basename(os.path.normpath(vit_dir))
        imgsz = vspec['res']
        if args.imgsz and args.imgsz != imgsz:
            raise SystemExit(f'--imgsz {args.imgsz}: {run_name} is fixed at '
                             f'{imgsz}')
        eb = leash_vit.eval_batch(vcfg)
        if args.batch and args.batch != eb:
            print(f'--batch {args.batch} ignored: {run_name} always scores '
                  f'in batches of {eb}, the shape it was validated at')
        args.batch = eb
        want = {'crop': vspec['framing'], 'imgsz': int(imgsz)}
        _no = check_protocol(want)
        if _no:
            raise SystemExit(_no)
        print(f'{sp["title"]}: {run_name} (DINOv3 ViT-L), {vspec["framing"]} '
              f'crops letterboxed to {imgsz}, batch {args.batch}')
    else:
        from ultralytics import YOLO
        weights, run_name = gate_weights()
        model = YOLO(weights)
        names = dict(model.names)
        got = sorted(str(v) for v in names.values())
        if got != sorted(sp['classes']):
            raise SystemExit(f"{weights} classifies {got} -- not a "
                             f"{sp['title']} (expected "
                             f"{sorted(sp['classes'])})")
        # index of the class whose probability is written beside every verdict
        dog_idx = [k for k, v in names.items() if v == pos][0]

        imgsz = args.imgsz or model_imgsz(model)
        if not imgsz:
            raise SystemExit(f'{weights} does not record the imgsz it was '
                             f'trained at -- pass --imgsz explicitly')
        want = {'crop': CROP_MODE, 'imgsz': int(imgsz)}
        if not args.batch:
            args.batch = auto_batch(imgsz)
        _no = check_protocol(want)
        if _no:
            raise SystemExit(_no)
        print(f'{sp["title"]}: {run_name}, full-resolution crops, '
              f'predict at imgsz={imgsz}, batch {args.batch}')

    # group into images, keeping the plan's order
    #
    # Over the WHOLE work table, always -- never over --limit. `jobs` is what
    # the shard boundaries are cut on, so grouping a truncated row list makes
    # the last shard of a limited run short, and it is still written under the
    # index a full run would give a 20,000-image shard. done_shards() then
    # skips that index forever and the rest of it is never judged: a single
    # `run --limit 5000` would have buried 16,593 images (~23,600 boxes) with
    # nothing anywhere reporting a gap. The limit is a stopping point below;
    # it must not be allowed to renumber anything.
    jobs, cur, cur_key = [], [], None
    for i in range(n):
        key = (work['image_id'][i], work['cell'][i], work['drive'][i])
        if key != cur_key:
            if cur_key:
                jobs.append((*cur_key, cur))
            cur_key, cur = key, []
        cur.append((work['det_idx'][i], work['x1'][i], work['y1'][i],
                    work['x2'][i], work['y2'][i]))
    if cur_key:
        jobs.append((*cur_key, cur))

    # Shards are cut over the WHOLE job, never over --limit, and always at
    # the size the first run used. Both matter: a shard's identity is its
    # index, so a second run with a different --shard or a different --limit
    # would number them differently and "already done" would skip work that
    # was never done. The size is pinned in the plan the first time it is
    # used, and a run that disagrees is refused rather than silently wrong.
    plan_images = None
    try:
        plan_images = json.load(open(PLAN_FILE))['images']
    except (OSError, ValueError, KeyError):
        print('WARNING: no image count in the plan file -- the job list '
              'cannot be checked against what was planned')
    _no = check_job_count(len(jobs), plan_images)
    if _no:
        raise SystemExit(_no)
    size = _shard_size(args.shard)
    shards = [jobs[i:i + size] for i in range(0, len(jobs), size)]
    skip = done_shards()
    short = short_done_shards(skip, shards, args.max_unreadable)
    if short:
        raise SystemExit(
            f'{len(short)} finished shard(s) hold fewer images than planned '
            f'(' + ', '.join(f'{shard_name(si)} {h:,}/{p:,}'
                             for si, h, p in short[:6])
            + (' ...' if len(short) > 6 else '') + ') -- written while a '
            f'drive was unreadable, and a resume would never revisit them. '
            f'Set the record aside: `gate_store.py archive --stage {STAGE}`'
            f', then run again.')
    stop = stop_after(shards, skip, args.limit)
    print(f'{len(jobs):,} images in {len(shards)} shards of {size}; '
          f'{len(skip)} already done'
          + (f'; stopping after shard {stop - 1}' if stop is not None else ''))

    roots = _roots()
    todo = [si for si in range(len(shards))
            if si not in skip and (stop is None or si < stop)]
    # every drive this run reads must answer BEFORE the model loads: a dead
    # mount is a remount, not a model problem (the ViT parity frames too)
    need = {j[2] for si in todo for j in shards[si]}
    _no = unreadable_roots(roots, need)
    if not _no and vit_dir:
        _no = leash_vit.unreadable_parity_frames(vit_dir)
    if _no:
        raise SystemExit(_no)
    prep = lut = fwd = None
    half = False
    if vit_dir:
        # Load, then prove the whole path on crops with known answers before
        # judging anything: parity.json names training crops by frame and
        # box with the p the model gave them in memory when it was fit.
        # They are re-cut from the frames by the decoder's own functions and
        # scored in batches of the validated shape; a framing, JPEG, resize,
        # normalisation or precision drift fails here, not in the record.
        dev = f'cuda:{args.device}' if str(args.device).isdigit() \
            else str(args.device)
        try:
            net, vcfg, vdev = leash_vit.load(vit_dir, dev)
        except (OSError, ValueError, KeyError) as e:
            raise SystemExit(f'--vit-model {vit_dir}: {e}')
        if vdev.type != 'cuda':
            raise SystemExit(f'{run_name} was validated on cuda (bf16); '
                             f'--device {args.device} is not cuda')
        fwd = leash_vit.Forward(net, vcfg, vdev)
        try:
            gap, n_par = leash_vit.parity_check(
                fwd, lambda f, b: vit_cut_one(vspec, f, b), vit_dir)
        except (OSError, ValueError, KeyError) as e:
            raise SystemExit(f'{run_name} does not reproduce its own '
                             f'predictions here: {e}')
        print(f'parity: {n_par} training crops re-cut from their frames '
              f'match the fit-time p to {gap:.1e}; {vdev} bf16')
        prep = vspec
    else:
        # One predict() before anything else: it is what picks the device,
        # fuses the model and chooses the transforms, and the fast path
        # reuses all three rather than rebuilding any of them.
        model.predict(probe_crops()[:1], verbose=False, device=args.device,
                      imgsz=imgsz)
        backend = model.predictor.model
        half = bool(getattr(backend, 'fp16', False))
    if vit_dir:
        pass                     # proven by the parity check above
    elif not args.legacy_predict:
        try:
            prep = classify_spec(model.predictor.transforms, imgsz)
            lut = value_lut(model.predictor.transforms).to(backend.device)
            gap = fast_path_check(model, prep, lut, imgsz, args.device,
                                  args.batch, half)
        except ValueError as e:
            raise SystemExit(f'the fast path does not reproduce predict() '
                             f'here: {e}\n  run with --legacy-predict')
        print(f'fast path: decoders resize, GPU normalises, '
              f'{backend.device} {"fp16" if half else "fp32"}; '
              f'matches predict() to {gap:.1e} on the probe crops')
    else:
        print('legacy predict(): PIL crops, preprocessed in this process')

    pin_protocol(dict(want, model=run_name, **(
        {'kind': vspec['kind'], 'model_dir': vit_dir} if vit_dir else {})))
    readers = reader_counts(args.readers,
                            {j[2] for si in todo for j in shards[si]})
    disk_order = drive_map(args.disk_order, '--disk-order')
    print(f'batch {args.batch}, readers {readers}, {args.workers} decoders, '
          f'{args.ahead} shards look-ahead, '
          f'disk order {disk_order or "off"}')
    t0, seen, bad = time.time(), 0, 0
    bad_in = {si: {} for si in todo}       # shard -> {error kind: frames}
    bad_drives = {}                        # shard -> drives a read failed on
    drive_of = {}                          # shard -> {image_id: drive}
    # frames the shards on disk already account for -- `seen` counts only what
    # THIS process walked, and a resumed run starts it at zero
    base = sum(len(shards[i]) for i in skip if i < len(shards))
    boxes_run = dogs_run = 0
    last_beat = 0.0
    # Shards fill side by side now: the drives move at their own pace, so a
    # shard is written when its last frame is in AND its last crop is scored,
    # in whatever order that happens.
    rows = {si: {'image_id': [], 'det_idx': [], 'label': [], 'p': []}
            for si in todo}
    pend = {si: 0 for si in todo}          # crops taken in, not yet scored
    closing = set()                        # every frame in, maybe not scored
    batch, meta, inflight = [], [], []

    def beat(si, flight, drives=None):
        el = max(1e-9, time.time() - t0)
        _beat(pid=os.getpid(), started=t0, updated=time.time(),
              drives_read=dict(drives or {}),
              shard=si, shards_total=len(shards),
              images=base + seen, images_total=len(jobs),
              rows_flight=flight, boxes=boxes_run, bad=bad,
              img_s=round(seen / el, 2), box_s=round(boxes_run / el, 2),
              dogs=dogs_run, imgsz=imgsz, batch=args.batch,
              dog_share=(dogs_run / boxes_run) if boxes_run else None)

    def launch():
        """Score what is batched. The fast path only queues the work on the
        GPU; settle() reads it, so the next batch is gathered meanwhile."""
        if not batch:
            return
        if fwd is not None:
            got = fwd(batch)        # P(leashed), queued on the device
        elif prep is None:
            got = [float(r.probs.data[dog_idx]) for r in model.predict(
                batch, verbose=False, device=args.device, imgsz=imgsz)]
        else:
            got = forward_probs(backend, batch_input(batch, lut, half))
            got = got[:, dog_idx]
        inflight.append((got, list(meta)))
        batch.clear()
        meta.clear()

    def settle(keep=0):
        nonlocal boxes_run, dogs_run
        while len(inflight) > keep:
            got, mt = inflight.pop(0)
            if not isinstance(got, list):
                got = got.float().cpu().tolist()
            for p, (si, iid, di) in zip(got, mt):
                r = rows[si]
                r['image_id'].append(iid)
                r['det_idx'].append(int(di))
                r['label'].append(pos if p >= 0.5 else other)
                r['p'].append(round(p, 5))
                pend[si] -= 1
                boxes_run += 1
                dogs_run += p >= 0.5

    def write_done(lanes):
        """Write every shard whose frames are all in and crops all scored."""
        nonlocal last_beat
        for si in sorted(closing):
            if lanes.left[si] or pend[si]:
                continue
            closing.discard(si)
            _no = refuse_short_shard(si, len(shards[si]), len(shards), size)
            if _no:
                raise SystemExit(_no)
            _no = refuse_unreadable_shard(si, bad_in.pop(si, {}),
                                          len(shards[si]),
                                          args.max_unreadable)
            # Under the limit is not enough when the failures were reads: a
            # drive that died with a few of this shard's frames left gives
            # only those few errors here, and the shard would be written
            # short for good. Its drives must answer now.
            if not _no and bad_drives.get(si):
                _no = unreadable_roots(roots, bad_drives.pop(si) - {None})
                if _no:
                    _no = (f'shard {si}: frames failed to read, and their '
                           f'drive is gone -- not written.\n' + _no)
            drive_of.pop(si, None)
            if _no:
                raise SystemExit(_no)
            r = rows.pop(si)
            tmp = os.path.join(OUT_DIR, f'.{shard_name(si)}.tmp')
            pq.write_table(pa.table({
                'image_id': pa.array(r['image_id'], pa.string()),
                'det_idx': pa.array(r['det_idx'], pa.int32()),
                'label': pa.array(r['label'], pa.string()),
                sp['p_col']: pa.array(r['p'], pa.float32()),
                # stamped on every row: this is a model's opinion, and nothing
                # that builds a training set may read it as a verdict
                'unverified': pa.array([True] * len(r['label']), pa.bool_()),
                'model': pa.array([run_name] * len(r['label']), pa.string()),
            }), tmp)
            os.replace(tmp, os.path.join(OUT_DIR, shard_name(si)))
            el = time.time() - t0
            print(f'  shard {si + 1}/{len(shards)}  {seen:,} images  '
                  f'{seen / el:.0f} img/s  '
                  f'eta {(len(jobs) - seen) / max(1e-9, seen / el) / 3600:.1f} h'
                  f'  read ' + ' '.join(f'{d} {v / el:.0f}/s'
                                        for d, v in lanes.read.items()),
                  flush=True)
            # the shard is on disk now, so its rows are no longer in flight --
            # counting them in both places would double them for one tick
            beat(lanes.lowest(), flying(), lanes.read)
            last_beat = time.time()

    def flying():
        return sum(len(r['label']) for r in rows.values())

    # The first one goes out BEFORE any frame is read: the model is already
    # loaded by here, so this is the moment the panel can stop saying nothing
    # is happening. Everything in it is zero, which is true and is not the
    # same as unknown.
    beat(todo[0] if todo else 0, 0)
    with mp.get_context('spawn').Pool(args.workers, _init,
                                      (roots, prep)) as pool:
        lanes = DriveLanes(shards, todo, roots, readers, ahead=args.ahead,
                           slots=max(32, 4 * args.workers),
                           disk_order=disk_order)
        try:
            for si, iid, crops, err in pool.imap_unordered(
                    _cut, lanes.feed(), chunksize=1):
                lanes.consumed(si)
                seen += 1
                if err:
                    bad += 1
                    k = bad_in.setdefault(si, {})
                    k[err] = k.get(err, 0) + 1
                    if err in READ_ERRORS:
                        if si not in drive_of:
                            drive_of[si] = {j[0]: j[2] for j in shards[si]}
                        bad_drives.setdefault(si, set()).add(
                            drive_of[si].get(iid))
                for di, crop in crops:
                    batch.append(crop)
                    meta.append((si, iid, di))
                    pend[si] += 1
                    if len(batch) >= args.batch:
                        launch()
                        settle(keep=1)
                if not lanes.left[si]:
                    closing.add(si)
                if closing:
                    write_done(lanes)
                if time.time() - last_beat >= BEAT_EVERY_S:
                    beat(lanes.lowest(), flying(), lanes.read)
                    last_beat = time.time()
            launch()
            settle()
            write_done(lanes)
        finally:
            lanes.stop.set()
        if rows:
            raise SystemExit(f'the feed ended with shards {sorted(rows)} '
                             f'unfinished -- nothing was written for them')
    # The record is the shards; a heartbeat outliving the process would be a
    # claim about a run that is over. A kill leaves it behind, which is why
    # the reader also ages it out.
    _beat_clear()
    print(f'done: {seen:,} images, {bad:,} unreadable, '
          f'{time.time() - t0:.0f}s')
    return 0


def status(args):
    if not os.path.exists(PLAN_FILE):
        print('no plan yet')
        return 0
    doc = json.load(open(PLAN_FILE))
    import glob
    fs = sorted(glob.glob(os.path.join(OUT_DIR, f"{spec()['dir']}-*.parquet")))
    rows = 0
    try:
        import pyarrow.parquet as pq
        rows = sum(pq.ParquetFile(f).metadata.num_rows for f in fs)
    except Exception:
        pass
    print(f"{spec()['title']} plan: {doc['rows']:,} boxes / "
          f"{doc['images']:,} images (gen {doc['gen']}, {doc['created']})")
    print(f'judged: {rows:,} boxes in {len(fs)} shards '
          f'({rows / max(1, doc["rows"]):.1%})')
    return 0


# ── re-running a stage with a newly promoted model ─────────────────────────
# A finished stage holds one model's verdicts, and the runner skips every
# shard already on disk -- so judging the store again with the model promoted
# since means moving the old record out of the runner's way first. MOVED, not
# deleted: into archive/<when>/ beside a note naming the model that wrote it,
# so a re-run that turns out worse is one `restore` away from undone. The
# readers glob <dir>-*.parquet at the top level only, so an archive is out of
# every count, map and pool the moment it is made.
ARCHIVE = 'archive'
ARCHIVE_NOTE = 'ARCHIVED.json'
# the planned work list and the shard size the job pinned on its first run;
# they move only when the work list itself is out of date
PLAN_FILES = ('plan.json', 'work.parquet', 'shard_size')


def _stage_dir(stage, repo=None):
    return os.path.join(stages_root(repo), STAGES[stage]['dir'])


def _verdict_files(stage, repo=None):
    d, pre = _stage_dir(stage, repo), STAGES[stage]['dir'] + '-'
    try:
        names = os.listdir(d)
    except OSError:
        return []
    return sorted(f for f in names
                  if f.startswith(pre) and f.endswith('.parquet'))


def archive_run(stage, models, best, replan=False, reason='', repo=None,
                now=None):
    """Move a stage's verdicts out of the runner's way; returns the note
    written beside them, or None when there was nothing to move.

    `models` is what wrote the verdicts ({run: rows}) and `best` what is about
    to replace them -- both go in the note, so the archive says what it is
    without anybody reading a shard. `replan` moves the work list as well,
    for a stage whose list was built from an upstream that has changed.
    """
    if stage not in STAGES:
        raise ValueError(f'unknown stage {stage!r}')
    d = _stage_dir(stage, repo)
    moving = _verdict_files(stage, repo)
    if os.path.exists(os.path.join(d, 'progress.json')):
        moving.append('progress.json')
    # the protocol describes the verdicts, so it travels with them -- left
    # behind, it would pin the next job to the old one's crop and imgsz
    if os.path.exists(os.path.join(d, PROTOCOL_FILE)):
        moving.append(PROTOCOL_FILE)
    if replan:
        moving += [f for f in PLAN_FILES if os.path.exists(os.path.join(d, f))]
    if not moving:
        return None
    at = time.localtime(now if now is not None else time.time())
    name = time.strftime('%Y%m%d-%H%M%S', at)
    dest = os.path.join(d, ARCHIVE, name)
    os.makedirs(dest)          # never into an archive that is already there
    note = {
        'stage': stage,
        'archived_at': time.strftime('%Y-%m-%d %H:%M:%S', at),
        'models': dict(models or {}),
        'replaced_by': best,
        'replanned': bool(replan),
        'reason': reason,
        'files': moving,
        'restore': f'python tools/detect/gate_store.py restore --stage {stage} '
                   f'--archive {name}',
    }
    # the note first: an archive interrupted mid-move still says what it is
    with open(os.path.join(dest, ARCHIVE_NOTE), 'w') as fh:
        json.dump(note, fh, indent=1)
    for f in moving:
        os.replace(os.path.join(d, f), os.path.join(dest, f))
    return dict(note, name=name, path=dest)


def restore_run(stage, name, repo=None, now=None):
    """Put an archived record back. Whatever is there now is archived first,
    so a restore is as undoable as the re-run it undoes."""
    if stage not in STAGES:
        raise ValueError(f'unknown stage {stage!r}')
    base = os.path.join(_stage_dir(stage, repo), ARCHIVE)
    if (not isinstance(name, str) or not name or os.sep in name
            or name.startswith('.') or not os.path.isfile(
                os.path.join(base, name, ARCHIVE_NOTE))):
        raise ValueError(f'no archive {name!r} for the {stage} stage')
    src = os.path.join(base, name)
    with open(os.path.join(src, ARCHIVE_NOTE)) as fh:
        note = json.load(fh)
    files = [f for f in note.get('files') or []
             if os.path.basename(f) == f and os.path.exists(
                 os.path.join(src, f))]
    replan = any(f in PLAN_FILES for f in files)
    current = archive_run(stage, {}, f'restore of {name}', replan=replan,
                          reason=f'set aside to restore {name}', repo=repo,
                          now=now)
    d = _stage_dir(stage, repo)
    for f in files:
        os.replace(os.path.join(src, f), os.path.join(d, f))
    note['restored_at'] = time.strftime(
        '%Y-%m-%d %H:%M:%S',
        time.localtime(now if now is not None else time.time()))
    note['restored_files'] = files
    with open(os.path.join(src, ARCHIVE_NOTE), 'w') as fh:
        json.dump(note, fh, indent=1)
    return {'restored': files, 'set_aside': current and current['name']}


def shard_models():
    """{model run: rows} over this stage's verdict shards, for the archive
    note -- so an archive says what wrote it without anybody opening one."""
    import pyarrow.parquet as pq
    got = {}
    for f in _verdict_files(STAGE):
        try:
            col = pq.read_table(os.path.join(OUT_DIR, f), columns=['model'])
        except Exception:
            continue
        for m in col['model'].to_pylist():
            got[m] = got.get(m, 0) + 1
    return got


def archive(args):
    """CLI: set a stage's verdicts aside so the next run starts clean."""
    best = None
    try:
        best = gate_weights()[1]
    except SystemExit:
        pass
    pid = live_runner()
    if pid:
        raise SystemExit(f'the {spec()["title"]} is running (pid {pid}) -- '
                         f'stop it first; archiving under a live run splits '
                         f'its record in two')
    replan = args.replan
    why = stale_plan()
    if why and not replan:
        print(f'{why} -- moving the work list too (--replan)')
        replan = True
    note = archive_run(STAGE, shard_models(), best, replan=replan,
                       reason=args.reason or 'archived from the CLI')
    if not note:
        print(f'nothing to archive in {os.path.relpath(OUT_DIR, REPO)}/')
        return 0
    print(f"archived {len(note['files'])} files to "
          f"{os.path.relpath(note['path'], REPO)}\n  undo: {note['restore']}")
    return 0


def restore(args):
    """CLI: list the archives, or put one back."""
    base = os.path.join(OUT_DIR, ARCHIVE)
    if not args.archive:
        try:
            names = sorted(os.listdir(base))
        except OSError:
            names = []
        for n in names:
            try:
                with open(os.path.join(base, n, ARCHIVE_NOTE)) as fh:
                    note = json.load(fh)
            except (OSError, ValueError):
                continue
            print(f"{n}  {', '.join(note.get('models') or {}) or '-':24s} "
                  f"{len(note.get('files') or [])} files"
                  + ('  (restored)' if note.get('restored_at') else ''))
        if not names:
            print('no archives')
        return 0
    got = restore_run(STAGE, args.archive)
    print(f"restored {len(got['restored'])} files from {args.archive}"
          + (f"; what was there is in archive/{got['set_aside']}"
             if got['set_aside'] else ''))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan'); p.set_defaults(fn=plan)
    p.add_argument('--gen', type=int, default=1)
    p.add_argument('--memory', default='8GB')
    p.add_argument('--partial', action='store_true',
                   help='plan a downstream stage over an unfinished upstream '
                        'one, permanently omitting whatever it has not '
                        'reached')
    r = sub.add_parser('run'); r.set_defaults(fn=run)
    r.add_argument('--workers', type=int, default=max(2, (os.cpu_count() or 4)))
    r.add_argument('--batch', type=int, default=0,
                   help='crops per predict call; default scales with imgsz '
                        '(128 at 224, 16 at 640 -- 128 at 640 OOMs a 16 GB '
                        'card). cuDNN picks its kernels per batch size, so p '
                        'can move by up to ~0.01 between sizes: resume a '
                        'stage at the batch it started with')
    r.add_argument('--shard', type=int, default=SHARD_ROWS)
    r.add_argument('--device', default=0)
    r.add_argument('--max-unreadable', type=int, default=MAX_UNREADABLE,
                   help='frames a shard may lose to read errors before it '
                        f'is refused, not written (default {MAX_UNREADABLE})')
    r.add_argument('--vit-model', default=None, metavar='DIR',
                   help='leash stage: judge with the fine-tuned DINOv3 leash '
                        'model in DIR (config.json, trainable.pt, '
                        'parity.json; see leash_vit.py) instead of the '
                        'promoted YOLO model. Needs torch + transformers + '
                        'pyarrow in the running interpreter')
    r.add_argument('--imgsz', type=int, default=0,
                   help='predict at this size instead of the one the '
                        'model was trained at (for experiments only)')
    r.add_argument('--limit', type=int, default=0,
                   help='stop after this many BOXES (for a trial run)')
    r.add_argument('--legacy-predict', action='store_true',
                   help='score PIL crops through model.predict() in this '
                        'process, as before the fast path (for comparison). '
                        'Only the scoring: frames still come through the '
                        'per-drive readers. The old feed too means the '
                        'gate_store.py from before the fast path, from git')
    r.add_argument('--readers', action='append', default=[],
                   metavar='DRIVE=N',
                   help=f'reader threads for one drive (repeatable); default '
                        f'{READERS_DEFAULT}, {READERS}')
    r.add_argument('--disk-order', action='append', default=[],
                   metavar='DRIVE=WINDOW',
                   help="read a drive's frames in on-disk (FIEMAP) order, "
                        'WINDOW at a time within each shard -- the sweep\'s '
                        'disk_ordered(); the lane pauses for each lookup')
    r.add_argument('--ahead', type=int, default=AHEAD,
                   help='shards a drive may read past the lowest unfinished '
                        'one; a stop throws away up to AHEAD+1 partly-scored '
                        'shards, so lower it if the run is stopped often')
    s = sub.add_parser('status'); s.set_defaults(fn=status)
    rs = sub.add_parser('restore', help='list the archived runs of a stage, '
                        'or put one back')
    rs.set_defaults(fn=restore)
    rs.add_argument('--archive', default=None,
                    help='the archive to put back (omit to list them)')
    ar = sub.add_parser('archive', help="move a stage's verdicts (and with "
                        '--replan its work list) into archive/<when>/')
    ar.set_defaults(fn=archive)
    ar.add_argument('--replan', action='store_true',
                    help='also move plan.json/work.parquet/shard_size, for a '
                         'store or upstream that has changed')
    ar.add_argument('--reason', default='')
    # every subcommand takes it, and it has to be applied before any of them
    # touches a path
    for sp_ in (p, r, s, rs, ar):
        sp_.add_argument('--stage', default='gate', choices=sorted(STAGES),
                         help='which pass over the store to work on')
    a = ap.parse_args(argv)
    use(a.stage)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
