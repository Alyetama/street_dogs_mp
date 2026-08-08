#!/usr/bin/env python3
"""Run the dog-bin gate over every detection the sweep committed.

    python tools/detect/gate_store.py plan            # what there is to do
    python tools/detect/gate_store.py run             # do it, resumably
    python tools/detect/gate_store.py run --limit 5000
    python tools/detect/gate_store.py status

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
OUT_DIR = os.path.join(REPO, 'data', 'gate')
PLAN_FILE = os.path.join(OUT_DIR, 'plan.json')
# What the run is doing RIGHT NOW. Progress was read off the written shards
# alone, and a shard is 20,000 frames -- so for the first several minutes of a
# twelve-hour run the dashboard showed 0%, 0 judged, no rate and no share, an
# idle-looking panel above a machine at full tilt. The shards remain the
# record; this is the only thing that can speak before the first one lands.
BEAT_FILE = os.path.join(OUT_DIR, 'progress.json')
BEAT_EVERY_S = 3.0
SHARD_ROWS = 20000
# The margin the gate was trained with. tools/detect/build_review_set.py and
# the live preview writer both use it; inference has to match or every verdict
# is made on a differently framed picture.
PAD_FRAC, PAD_PX = 0.12, 4
MIN_SIDE = 8
# images one drive contributes before the plan moves to the next. Sized
# so a worker's chunk falls inside one block and stays on one disk.
INTERLEAVE_BLOCK = 256


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
    """The promoted gate, from data/best_models.json. Never hardcoded: a new
    gate is promoted by editing that file, and this must follow it."""
    with open(os.path.join(REPO, 'data', 'best_models.json')) as fh:
        best = (json.load(fh).get('projects') or {}) \
            .get('dog-bin', {}).get('best') or {}
    rel = str(best.get('weights') or '')
    if not rel:
        raise SystemExit('no dog-bin model promoted in data/best_models.json')
    root = os.environ.get('TRAINING_ROOT') or os.path.dirname(
        os.path.dirname(REPO))
    p = rel if os.path.isabs(rel) else os.path.join(root, rel)
    if not os.path.exists(p):
        raise SystemExit(f'promoted gate not on disk: {p}\n'
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


def plan(args):
    """Every detection to judge, ordered for locality, split into shards."""
    sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
    import duckdb
    import store
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
    """, [f'{args.gen:04d}']).fetchall()
    con.close()
    rows = _interleave(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    doc = {'gen': args.gen, 'rows': len(rows),
           'images': len({r[0] for r in rows}),
           # the runner cannot look these up for itself; see _roots()
           'roots': _resolve_roots(),
           'shard_rows': SHARD_ROWS, 'created': time.strftime('%F %T')}
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
            f'delete data/gate/ to start over.')
    return pinned


def done_shards():
    try:
        return {int(f.split('-')[1].split('.')[0])
                for f in os.listdir(OUT_DIR)
                if f.startswith('gate-') and f.endswith('.parquet')}
    except OSError:
        return set()


# ── the worker: decode one frame, cut every box on it ──────────────────────
_ROOTS = {}


def _init(roots):
    global _ROOTS
    _ROOTS = roots


def _cut(job):
    """(image_id, cell, drive, [(det_idx, box)]) -> [(det_idx, crop)]."""
    from PIL import Image
    iid, cell, drive, boxes = job
    r = _ROOTS.get(drive)
    if not r:
        return iid, [], 'no_root'
    src = os.path.join(r, cell, 'ground_animal_images', f'{iid}.jpg')
    try:
        im = Image.open(src)
        im.load()
        if im.mode != 'RGB':
            im = im.convert('RGB')
    except Exception:
        return iid, [], 'unreadable'
    w, h = im.size
    out = []
    for di, x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        pad = int(PAD_FRAC * max(x2 - x1, y2 - y1)) + PAD_PX
        a, b = max(0, x1 - pad), max(0, y1 - pad)
        c, d = min(w, x2 + pad), min(h, y2 + pad)
        if c - a < MIN_SIDE or d - b < MIN_SIDE:
            continue
        # returned as bytes, not a PIL handle: a decoded 8000x4000 frame is
        # 96 MB and must never cross the process boundary
        out.append((di, im.crop((a, b, c, d)).resize((224, 224),
                                                     Image.BILINEAR)))
    im.close()
    return iid, out, ''


def run(args):
    import multiprocessing as mp
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ultralytics import YOLO

    if not os.path.exists(os.path.join(OUT_DIR, 'work.parquet')):
        raise SystemExit('no plan yet -- run `gate_store.py plan` first')
    work = pq.read_table(os.path.join(OUT_DIR, 'work.parquet')).to_pydict()
    n = len(work['image_id'])
    if args.limit:
        n = min(n, args.limit)

    weights, run_name = gate_weights()
    model = YOLO(weights)
    names = dict(model.names)
    got = sorted(str(v) for v in names.values())
    if got != ['dog', 'not_dog']:
        raise SystemExit(f'{weights} classifies {got} -- not a dog-bin gate')
    dog_idx = [k for k, v in names.items() if v == 'dog'][0]

    # group into images, keeping the plan's order
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
    size = _shard_size(args.shard)
    shards = [jobs[i:i + size] for i in range(0, len(jobs), size)]
    stop = None
    if args.limit:
        # limit is a stopping point, not a different job
        stop = min(len(shards), max(1, -(-args.limit // size)))
    skip = done_shards()
    print(f'{len(jobs):,} images in {len(shards)} shards of {size}; '
          f'{len(skip)} already done'
          + (f'; stopping after {stop}' if stop else ''))

    roots = _roots()
    t0, seen, bad = time.time(), 0, 0
    # frames the shards on disk already account for -- `seen` counts only what
    # THIS process walked, and a resumed run starts it at zero
    base = sum(len(shards[i]) for i in skip if i < len(shards))
    boxes_run = dogs_run = 0
    last_beat = 0.0

    def beat(si, flight):
        el = max(1e-9, time.time() - t0)
        _beat(pid=os.getpid(), started=t0, updated=time.time(),
              shard=si, shards_total=len(shards),
              images=base + seen, images_total=len(jobs),
              rows_flight=flight, boxes=boxes_run, bad=bad,
              img_s=round(seen / el, 2), box_s=round(boxes_run / el, 2),
              dog_share=(dogs_run / boxes_run) if boxes_run else None)

    # The first one goes out BEFORE any frame is read: the model is already
    # loaded by here, so this is the moment the panel can stop saying nothing
    # is happening. Everything in it is zero, which is true and is not the
    # same as unknown.
    beat(next((i for i in range(len(shards)) if i not in skip), 0), 0)
    with mp.get_context('spawn').Pool(args.workers, _init, (roots,)) as pool:
        for si, shard in enumerate(shards):
            if stop is not None and si >= stop:
                break
            if si in skip:
                continue
            rows = {'image_id': [], 'det_idx': [], 'label': [], 'p_dog': []}
            batch, meta = [], []

            def flush():
                nonlocal boxes_run, dogs_run
                if not batch:
                    return
                for res, (iid, di) in zip(
                        model.predict(batch, verbose=False,
                                      device=args.device, imgsz=224), meta):
                    p = float(res.probs.data[dog_idx])
                    rows['image_id'].append(iid)
                    rows['det_idx'].append(int(di))
                    rows['label'].append('dog' if p >= 0.5 else 'not_dog')
                    rows['p_dog'].append(round(p, 5))
                    boxes_run += 1
                    dogs_run += p >= 0.5
                batch.clear()
                meta.clear()

            for iid, crops, err in pool.imap_unordered(_cut, shard,
                                                       chunksize=32):
                seen += 1
                if err:
                    bad += 1
                for di, crop in crops:
                    batch.append(crop)
                    meta.append((iid, di))
                    if len(batch) >= args.batch:
                        flush()
                if time.time() - last_beat >= BEAT_EVERY_S:
                    beat(si, len(rows['label']))
                    last_beat = time.time()
            flush()
            tmp = os.path.join(OUT_DIR, f'.gate-{si:05d}.tmp')
            pq.write_table(pa.table({
                'image_id': pa.array(rows['image_id'], pa.string()),
                'det_idx': pa.array(rows['det_idx'], pa.int32()),
                'label': pa.array(rows['label'], pa.string()),
                'p_dog': pa.array(rows['p_dog'], pa.float32()),
                # stamped on every row: this is a model's opinion, and nothing
                # that builds a training set may read it as a verdict
                'unverified': pa.array([True] * len(rows['label']), pa.bool_()),
                'model': pa.array([run_name] * len(rows['label']), pa.string()),
            }), tmp)
            os.replace(tmp, os.path.join(OUT_DIR, f'gate-{si:05d}.parquet'))
            el = time.time() - t0
            print(f'  shard {si + 1}/{len(shards)}  {seen:,} images  '
                  f'{seen / el:.0f} img/s  '
                  f'eta {(len(jobs) - seen) / max(1e-9, seen / el) / 3600:.1f} h',
                  flush=True)
            # the shard is on disk now, so its rows are no longer in flight --
            # counting them in both places would double them for one tick
            beat(si, 0)
            last_beat = time.time()
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
    fs = sorted(glob.glob(os.path.join(OUT_DIR, 'gate-*.parquet')))
    rows = 0
    try:
        import pyarrow.parquet as pq
        rows = sum(pq.ParquetFile(f).metadata.num_rows for f in fs)
    except Exception:
        pass
    print(f"plan: {doc['rows']:,} boxes / {doc['images']:,} images "
          f"(gen {doc['gen']}, {doc['created']})")
    print(f'judged: {rows:,} boxes in {len(fs)} shards '
          f'({rows / max(1, doc["rows"]):.1%})')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan'); p.set_defaults(fn=plan)
    p.add_argument('--gen', type=int, default=1)
    p.add_argument('--memory', default='8GB')
    r = sub.add_parser('run'); r.set_defaults(fn=run)
    r.add_argument('--workers', type=int, default=max(2, (os.cpu_count() or 4)))
    r.add_argument('--batch', type=int, default=128)
    r.add_argument('--shard', type=int, default=SHARD_ROWS)
    r.add_argument('--device', default=0)
    r.add_argument('--limit', type=int, default=0,
                   help='stop after this many BOXES (for a trial run)')
    s = sub.add_parser('status'); s.set_defaults(fn=status)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    sys.exit(main())
