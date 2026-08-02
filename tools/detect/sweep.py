#!/usr/bin/env python3
"""
Sweep orchestrator: worklist gen -> per-drive lanes -> engine -> parquet store.
Spec: DETECTION_RUN_STRATEGY.md section 6 (+4.6 config, 5.2 layout).

    sweep.py run    --gen 1 [--max-images N] [--drives d1 d2] [--rate-cap x]
    sweep.py status
    sweep.py unit                # print a systemd user unit
    sweep.py verify | invariants | compact --gen N

Run under the yolo env. Single instance enforced with fcntl locks (released
by the kernel on SIGKILL, section 6.7). Ctrl+C = graceful: readers stop,
queues drain, the partial batch flushes through the ring's written==expected
gate, and every open shard commits its contiguous prefix -- the next run
resumes from the committed tiles (store.tiling_resume).
"""

import argparse
import fcntl
import json
import os
import queue
import signal
import sys
import threading
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine  # noqa: E402
import store  # noqa: E402
from status import StatusWriter, read_status  # noqa: E402

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Live preview crops land under the dashboard's static dir, so the existing
# file server serves them at /recent_crops/<name> with no new plumbing.
CROP_DIR = os.path.join(REPO, 'data', 'dashboard', 'recent_crops')
# The panel samples from the newest 100 crops, so retention is by COUNT, not
# time: a wall-clock TTL emptied the grid whenever positives were sparse.
# Retention is by COUNT, not time: a wall-clock TTL emptied the grid whenever
# positives were sparse. The pool doubles as the review queue for the flagging
# page, so it is sized for bulk review (3000 crops ~105 MB + full frames
# ~300 MB), not just the 24-tile live grid.
CROP_TTL = 86400.0  # upper bound only; CROP_MAX does the real work
CROP_MAX = 3000
SHARD = 4000  # images per shard (section 6.1)
# Measured per-drive read capacities img/s (section 2.2); pacing recomputes
# rates every 60 s from live remaining counts (closed-loop waterfill, §6.5).
CAP = {'lynx': 71.0, 'capybara': 71.1, 'jackal': 32.5, 'bobcat': 22.9}
READERS = {'lynx': 1, 'capybara': 1, 'jackal': 1, 'bobcat': 8}
RAW_BYTES = 4 << 30


def load_cfg():
    p = os.path.join(REPO, 'tools', 'detect', 'detect.config.json')
    cfg = {}
    if os.path.exists(p):
        cfg = json.load(open(p))
    cfg.setdefault(
        'engine',
        os.path.join(REPO, 'data', 'engines', 'yolo26x_train30.engine'))
    cfg.setdefault('conf', 0.05)
    cfg.setdefault('iou', 0.90)
    cfg.setdefault('max_det', 256)  # det_idx is UINT8 (store intake guard)
    return cfg


def lane_plan(detect_root, gen):
    """[(pair, shards)] per drive; pair = dict from _dirs.json, shards =
    [(shard_idx, start, end, remaining_ranges)] from the frozen ids + store.

    Returns ``(lanes, corpus, region_corpus)``. The region totals are built
    here for the same reason the drive totals are: they are the DENOMINATOR of
    the panel's per-region bars, and without them every region renders 0.0%
    forever no matter how much work lands.
    """
    gd = os.path.join(detect_root, 'worklist', 'gen=%04d' % gen)
    dirs = json.load(open(os.path.join(gd, '_dirs.json')))
    lanes = defaultdict(list)
    corpus = defaultdict(int)  # full per-drive totals, done or not
    region_corpus = defaultdict(int)  # same, keyed by region
    sidx = 0
    for pr in dirs:
        ids = np.load(os.path.join(gd, pr['cell'], pr['drive'] + '.ids.npy'),
                      mmap_mode='r')
        n = len(ids)
        corpus[pr['drive']] += n
        region_corpus[pr['region']] += n
        shards = []
        for st in range(0, n, SHARD):
            en = min(st + SHARD, n)
            sd = store.pair_dir(detect_root, gen, pr['region'], pr['cell'],
                                pr['drive'])
            committed, done = store.tiling_resume(sd, en - st, shard_idx=sidx) \
                if os.path.isdir(sd) else ([], False)
            if not done:
                # tiling_resume speaks SHARD-RELATIVE [0, len); everything in
                # this file is absolute positions into the pair's ids array.
                # Passing relative ranges through unconverted made resume
                # re-process every shard with st>0 (351 duplicate image_ids
                # in the first pilot -- caught by store.invariants).
                shards.append(
                    (sidx, st, en, [(st + a, st + b) for a, b in committed]))
            sidx += 1
        if shards:
            lanes[pr['drive']].append((pr, ids, shards))
    return lanes, dict(corpus), dict(region_corpus)


class ShardCollector:
    """Routes out-of-order per-image results back to their shard; commits a
    shard when all its positions have landed (or its contiguous prefix at
    graceful stop). Every enqueued position produces exactly one row, so a
    committed prefix can never contain a hole."""

    def __init__(self,
                 writer,
                 gen,
                 status,
                 base_done=None,
                 base_drive=None,
                 base_region=None):
        self.w = writer
        self.gen = gen
        self.status = status
        # Progress is GLOBAL: seeded with what previous runs already
        # committed, so restarts accumulate instead of appearing to reset.
        self.base_done = int(base_done or 0)
        self.base_drive = dict(base_drive or {})
        self.base_region = dict(base_region or {})
        self.lock = threading.Lock()
        self.sh = {}  # key -> dict(rows={pos:(img,dets)}, ...)
        self.done_imgs = 0
        self.drive_done = defaultdict(int)
        self.region_done = defaultdict(int)
        self.boxes = 0
        self.positives = 0
        self.errors = defaultdict(int)

    def open(self, key, pr, shard_idx, start, end, committed_upto=None):
        """``committed_upto`` (absolute) = end of the store's already-durable
        contiguous prefix from a previous run. Without it a resumed shard's
        prefix scan starts at a position this run never re-issues, and every
        row the run produced for that shard is silently discarded at stop
        (26% of issued work in the 3-run smoke)."""
        with self.lock:
            self.sh[key] = dict(pr=pr,
                                shard_idx=shard_idx,
                                start=start,
                                end=end,
                                rows={},
                                committed=committed_upto
                                if committed_upto is not None else start)

    def add(self, key, pos, img_row, det_rows):
        with self.lock:
            s = self.sh[key]
            s['rows'][pos] = (img_row, det_rows)
            self.done_imgs += 1
            self.drive_done[s['pr']['drive']] += 1
            self.region_done[s['pr']['region']] += 1
            n = img_row['n_det']
            self.boxes += n
            if img_row['status'] == 0 and n > 0:
                self.positives += 1
            if img_row['status'] != 0:
                k = {
                    1: 'read',
                    2: 'decode',
                    3: 'missing',
                    4: 'infer',
                    5: 'mount_lost'
                }[img_row['status']]
                self.errors[k] += 1
            full = s['committed'] - s['start'] + len(s['rows']) \
                == s['end'] - s['start']
            # Push live counters every 100 images, not only at shard commit:
            # a 4,000-image shard takes minutes on the slow drives, and the
            # dashboard's 30 s stalled badge fired falsely in the gaps.
            push = self.done_imgs % 100 == 0
        if push:
            self._publish()
        if full:
            self._commit(key, s['end'])

    def _commit(self, key, upto):
        with self.lock:
            s = self.sh[key]
            pr = s['pr']
            rows = s['rows']
            # Part bounds are SHARD-RELATIVE (store convention, section 5.2:
            # s00007.p000000_004000); collector state is absolute.
            base = s['start']
            sw = self.w.open_shard(self.gen, pr['region'], pr['cell'],
                                   pr['drive'], s['shard_idx'],
                                   s['committed'] - base, upto - base)
            for pos in range(s['committed'], upto):
                img, dets = rows[pos]
                sw.add_image(img)
                if dets:
                    sw.add_detections(dets)
            sw.commit(upto - base)
            s['committed'] = upto
            if upto == s['end']:
                del self.sh[key]
        self._publish()

    def _publish(self):
        drv = dict(self.base_drive)
        for k, v in self.drive_done.items():
            drv[k] = drv.get(k, 0) + v
        reg = dict(self.base_region)
        for k, v in self.region_done.items():
            reg[k] = reg.get(k, 0) + v
        self.status.update(imgs_done=self.base_done + self.done_imgs,
                           run_imgs_done=self.done_imgs,
                           boxes_total=self.boxes,
                           positives=self.positives,
                           drive_done=drv,
                           region_done=reg,
                           errors=dict(self.errors))

    def flush_prefixes(self):
        """Graceful stop: commit each open shard's contiguous prefix."""
        with self.lock:
            keys = list(self.sh)
        for key in keys:
            with self.lock:
                s = self.sh.get(key)
                if s is None:
                    continue
                pos = s['committed']
                while pos in s['rows'] or pos < s['committed']:
                    pos += 1
            if pos > s['committed']:
                self._commit(key, pos)


class PreviewWriter:
    """Rolling window of recent positive detections as small jpgs.

    Runs on the GPU consumer thread, so it must stay cheap: it crops the
    already-decoded letterboxed row (net space, no re-read, no re-decode),
    encodes at ~160px, and writes at most `per_sec` files a second. Old
    files are unlinked on a slow cadence, never inside the hot path.
    """

    def __init__(self,
                 out_dir=CROP_DIR,
                 per_sec=2.0,
                 ttl=CROP_TTL,
                 cap=CROP_MAX):
        import cv2
        self.cv2 = cv2
        self.dir = out_dir
        # full frames live in a SUBDIR so the panel's listdir of the crop dir
        # never mistakes them for crops
        self.full_dir = os.path.join(out_dir, 'full')
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(self.full_dir, exist_ok=True)
        self.min_gap = 1.0 / max(per_sec, 0.1)
        self.ttl = ttl
        self.cap = cap
        self._last = 0.0
        self._last_sweep = 0.0
        self._lock = threading.Lock()

    def __call__(self, m, row, net_box, conf):
        now = time.time()
        with self._lock:
            if now - self._last < self.min_gap:
                return
            self._last = now
        x1, y1, x2, y2 = (int(v) for v in net_box)
        pad = int(0.12 * max(x2 - x1, y2 - y1)) + 4
        h, w = row.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return
        crop = row[y1:y2, x1:x2].copy()  # copy: slot is released after
        side = max(crop.shape[:2])
        if side > 160:
            sc = 160.0 / side
            crop = self.cv2.resize(crop, (max(1, int(
                crop.shape[1] * sc)), max(1, int(crop.shape[0] * sc))),
                                   interpolation=self.cv2.INTER_AREA)
        name = '%d_%s_%03d.jpg' % (int(
            now * 1000), m['image_id'], int(round(conf * 100)))
        tmp = os.path.join(self.dir, '.' + name)
        if self.cv2.imwrite(tmp, crop,
                            [int(self.cv2.IMWRITE_JPEG_QUALITY), 80]):
            os.replace(tmp, os.path.join(self.dir, name))
        # Full frame with the box drawn, for the click-through. Rendered from
        # the letterboxed row ALREADY IN RAM -- re-reading the 1.6 MB original
        # would steal I/O from the drives, which are the sweep's bottleneck.
        try:
            frame = row.copy()
            self.cv2.rectangle(frame, (int(net_box[0]), int(net_box[1])),
                               (int(net_box[2]), int(net_box[3])),
                               (0, 200, 255), 3)
            nw = int(round(m['wd'] * m['s']))
            nh = int(round(m['hd'] * m['s']))
            t, l = m['top'], m['left']
            frame = frame[t:t + nh, l:l + nw]  # strip letterbox padding
            ftmp = os.path.join(self.full_dir, '.' + name)
            if self.cv2.imwrite(ftmp, frame,
                                [int(self.cv2.IMWRITE_JPEG_QUALITY), 78]):
                os.replace(ftmp, os.path.join(self.full_dir, name))
        except Exception:
            pass
        if now - self._last_sweep > 15:
            self._last_sweep = now
            self._prune(now)

    def _prune(self, now):
        try:
            names = sorted(n for n in os.listdir(self.dir)
                           if n.endswith('.jpg'))
        except OSError:
            return
        for n in names:
            try:
                ts = int(n.split('_', 1)[0]) / 1000.0
            except ValueError:
                continue
            if now - ts > self.ttl:
                try:
                    os.remove(os.path.join(self.dir, n))
                    os.remove(os.path.join(self.full_dir, n))
                except OSError:
                    pass
        if len(names) > self.cap:
            for n in names[:len(names) - self.cap]:
                try:
                    os.remove(os.path.join(self.dir, n))
                    os.remove(os.path.join(self.full_dir, n))
                except OSError:
                    pass


def cmd_run(args):
    cfg = load_cfg()
    import torch
    detect_root = store.get_detect_root()
    store.ensure_bootstrap(detect_root)
    run_dir = os.path.join(detect_root, 'runs', 'gen=%04d' % args.gen)
    os.makedirs(run_dir, exist_ok=True)
    lockf = open(os.path.join(run_dir, '.lock'), 'w')
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('another sweep holds the lock -- refusing (section 6.7)',
              file=sys.stderr)
        return 1

    lanes, corpus, region_corpus = lane_plan(detect_root, args.gen)
    if args.drives:
        lanes = {d: v for d, v in lanes.items() if d in args.drives}
    todo = {
        d:
        sum((sh[2] - sh[1]) - sum(e - s for s, e in sh[3])
            for _, _, shards in v for sh in shards)
        for d, v in lanes.items()
    }
    todo_region = defaultdict(int)
    for v in lanes.values():
        for pr, _, shards in v:
            todo_region[pr['region']] += sum(
                (sh[2] - sh[1]) - sum(e - s for s, e in sh[3]) for sh in shards)
    # GLOBAL denominators/numerators: the panel tracks the whole corpus, not
    # this invocation. base_* is what previous runs already committed.
    corpus_total = sum(corpus.values())
    base_drive = {d: corpus.get(d, 0) - todo.get(d, 0) for d in corpus}
    # Same shape for regions. With --drives the excluded lanes contribute no
    # todo, so their regions read as complete -- exactly how base_drive already
    # behaves, and the panel is approximate under --drives either way.
    base_region = {
        r: region_corpus.get(r, 0) - todo_region.get(r, 0)
        for r in region_corpus
    }
    base_done = corpus_total - sum(todo.values())
    total = sum(todo.values())
    if args.max_images:
        total = min(total, args.max_images)
    print('already committed: %s of %s (%.2f%%)' %
          (f'{base_done:,}', f'{corpus_total:,}',
           100.0 * base_done / max(corpus_total, 1)))
    print('lanes:', {
        d: f'{n:,}'
        for d, n in todo.items()
    }, f'-> {total:,} images this run')
    if not total:
        print('nothing to do')
        return 0

    run_id = int(time.time()) & 0xFFFF
    epoch = time.time()
    status = StatusWriter(run_id,
                          args.gen,
                          corpus_total,
                          drive_totals=corpus,
                          region_totals=region_corpus)
    status.start()
    writer = store.Writer(detect_root)
    coll = ShardCollector(writer,
                          args.gen,
                          status,
                          base_done=base_done,
                          base_drive=base_drive,
                          base_region=base_region)
    ring = engine.BatchRing(torch)
    stop_ev = threading.Event()
    signal.signal(signal.SIGINT, lambda *a: stop_ev.set())
    signal.signal(signal.SIGTERM, lambda *a: stop_ev.set())

    def sink(m, st, boxes, confs):
        n = len(confs)
        img = dict(image_id=m['image_id'],
                   drive=m['drive_code'],
                   status=st,
                   n_det=n,
                   max_conf=float(confs.max()) if n else None,
                   orig_w=m.get('w0', 0),
                   orig_h=m.get('h0', 0),
                   reduce=m.get('reduce', 0),
                   guards=0,
                   ts_off=int(time.time() - epoch),
                   run_id=run_id)
        dets = [
            dict(image_id=m['image_id'],
                 det_idx=i,
                 conf=float(confs[i]),
                 x1=float(boxes[i][0]),
                 y1=float(boxes[i][1]),
                 x2=float(boxes[i][2]),
                 y2=float(boxes[i][3]),
                 run_id=run_id) for i in range(n)
        ]
        coll.add(m['key'], m['pos'], img, dets)

    preview = None if args.no_preview else PreviewWriter()
    consumer = engine.Consumer(cfg['engine'],
                               ring,
                               sink,
                               conf=cfg['conf'],
                               iou=cfg['iou'],
                               max_det=cfg['max_det'],
                               preview_fn=preview)
    ct = threading.Thread(target=consumer.run, daemon=True)
    ct.start()

    rawqs = {d: queue.Queue() for d in lanes}
    budgets = {d: engine.ByteBudget(RAW_BYTES) for d in lanes}
    dts = [
        threading.Thread(target=engine.decoder_loop,
                         args=(ring, rawqs, budgets, stop_ev, sink),
                         daemon=True) for _ in range(4)
    ]
    for t in dts:
        t.start()

    drive_codes = {d: i for i, d in enumerate(sorted(CAP))}
    buckets = {d: engine.TokenBucket() for d in lanes}
    issued = [0]
    issued_lock = threading.Lock()

    def lane_work(drive, items):
        """Yield (meta, path) in strict positional order for this lane."""
        for pr, ids, shards in items:
            base = os.path.join(pr['root'], pr['cell'], 'ground_animal_images')
            for sidx, st, en, committed in shards:
                key = (pr['cell'], pr['drive'], sidx)
                have = set()
                pe = st  # contiguous committed prefix end
                for s0, e0 in sorted(committed):
                    have.update(range(s0, e0))
                    if s0 <= pe:
                        pe = max(pe, e0)
                coll.open(key, pr, sidx, st, en, committed_upto=pe)
                for pos in range(st, en):
                    if pos in have:  # already committed in a prior run
                        continue
                    with issued_lock:
                        if args.max_images and issued[0] >= args.max_images:
                            return
                        issued[0] += 1
                    iid = int(ids[pos])
                    yield (dict(image_id=iid,
                                key=key,
                                pos=pos,
                                drive=drive,
                                drive_code=drive_codes.get(drive, 255)),
                           os.path.join(base, f'{iid}.jpg'))

    rthreads = []
    for d, items in lanes.items():
        work = lane_work(d, items)
        wlock = threading.Lock()

        def locked(work=work, wlock=wlock):
            while True:
                with wlock:
                    try:
                        yield next(work)
                    except StopIteration:
                        return

        for _ in range(READERS.get(d, 1)):
            r = engine.DriveReader(d,
                                   locked(),
                                   rawqs[d],
                                   budgets[d],
                                   sink,
                                   stop_ev,
                                   pace=buckets[d])
            t = threading.Thread(target=r.run, daemon=True)
            t.start()
            rthreads.append(t)

    def pacer():
        while not stop_ev.is_set():
            rem = {
                d: max(1, todo[d] - coll.drive_done.get(d, 0))
                for d in lanes
            }
            t_fin = max(rem[d] / CAP.get(d, 50.0) for d in lanes)
            for d in lanes:
                buckets[d].set_rate(args.rate_cap or rem[d] / t_fin)
            time.sleep(60)

    threading.Thread(target=pacer, daemon=True).start()

    for t in rthreads:
        t.join()
    stop_ev.set()
    for t in dts:
        t.join()
    ring.flush_partial()
    # batq.join() (consumer calls task_done per batch) guarantees the final
    # batches' sink() calls have LANDED before prefixes are flushed -- an
    # empty-queue check races the consumer's in-flight NMS/sink work.
    ring.batq.join()
    consumer.stopping.set()
    ct.join(timeout=60)
    coll.flush_prefixes()
    status.stop(state='stopped' if stop_ev.is_set() else 'done')
    print(f'run finished: {coll.done_imgs:,} images, {coll.boxes:,} boxes, '
          f'{coll.positives:,} positives, errors={dict(coll.errors)}')
    return 0


UNIT = """[Unit]
Description=street_dogs detection sweep (gen %(gen)s)
[Service]
Type=simple
WorkingDirectory=%(repo)s
ExecStart=%(py)s tools/detect/sweep.py run --gen %(gen)s
Restart=on-failure
RestartSec=60
MemoryMax=40G
MemorySwapMax=0
[Install]
WantedBy=default.target
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--gen', type=int, default=1)
    r.add_argument('--max-images', type=int)
    r.add_argument('--drives', nargs='+')
    r.add_argument('--rate-cap', type=float)
    r.add_argument('--no-preview',
                   action='store_true',
                   help='disable the live detection-crop preview')
    r.set_defaults(func=cmd_run)
    s = sub.add_parser('status')
    s.set_defaults(func=lambda a: print(json.dumps(read_status(), indent=1)))
    u = sub.add_parser('unit')
    u.add_argument('--gen', type=int, default=1)
    u.set_defaults(func=lambda a: print(UNIT % dict(
        gen=a.gen,
        repo=REPO,
        py='<home>/miniforge3/envs/yolo/bin/python')))
    for name, fn in (('verify', store.verify), ('invariants',
                                                store.invariants)):
        c = sub.add_parser(name)
        c.set_defaults(func=lambda a, fn=fn: print(fn()))
    args = p.parse_args()
    rc = args.func(args)
    sys.exit(rc or 0)


if __name__ == '__main__':
    main()
