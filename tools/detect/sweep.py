#!/usr/bin/env python3
"""
Sweep orchestrator: worklist gen -> per-drive lanes -> engine -> parquet store.
Spec: DETECTION_RUN_STRATEGY.md section 6 (+4.6 config, 5.2 layout).

    sweep.py run    --gen 1 [--max-images N] [--drives d1 d2] [--rate-cap x]
    sweep.py status
    sweep.py unit                # print a systemd user unit
    sweep.py verify | invariants          # whole store, no arguments
    sweep.py compact --gen N --cell C [--execute]

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
import struct
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
DECODERS = 4  # cv2 decode threads (GIL released inside imdecode)


def parse_drive_map(pairs, cast, flag):
    """['jackal=8', 'bobcat=16'] -> {'jackal': 8, 'bobcat': 16}.

    The per-drive read capacities and reader counts above were measured once,
    in August; a drive measured again reads very differently (jackal 32.5
    then, ~95 img/s with 8 readers in November), so a run can override them
    without an edit to a tracked file. A malformed pair is an error, never a
    silent default."""
    out = {}
    for pr in pairs or ():
        name, sep, val = str(pr).partition('=')
        if not sep or not name:
            raise SystemExit(f'{flag}: expected drive=value, got {pr!r}')
        try:
            v = cast(val)
        except ValueError:
            raise SystemExit(f'{flag}: bad value in {pr!r}')
        if v <= 0:
            raise SystemExit(f'{flag}: {pr!r} must be positive')
        out[name] = v
    return out


def finish_time(rem, cap, gpu_cap=None):
    """Seconds until the slowest lane is done: each drive at its read cap,
    and all of them together no faster than the GPU can take. Without the
    GPU term, drives whose caps sum past what the engine can infer are each
    told to read faster than they will be served, and the slowest-to-finish
    lane gets no more of the GPU than the others."""
    t = max(rem[d] / cap.get(d, 50.0) for d in rem)
    if gpu_cap:
        t = max(t, sum(rem.values()) / gpu_cap)
    return t


def adapt_caps(eff, cap, planned, observed):
    """Read capacities as the drives actually deliver them, updated in place.

    A drive's speed changes from cell to cell (file sizes, where they sit on
    the platter): North America on capybara read at 44 img/s in one stretch
    and 26 in the next. A cap set once is wrong somewhere, and the planner
    trusts it: too high and it plans GPU work a disk cannot feed while the
    other lanes wait; too low and a disk with room is held back. So: a lane
    that delivered under 85% of its planned rate is disk-bound, and its cap
    moves halfway to what it delivered (+10%); a lane that kept up may be
    worth more, and its cap grows 25% a step, never past the configured one.
    Paused lanes say nothing about their drive and are left alone.
    """
    for d, p in planned.items():
        if not p or p <= 0 or d not in observed:
            continue
        top = cap.get(d, 50.0)
        cur = eff.get(d, top)
        if observed[d] < 0.85 * p:
            eff[d] = max(1.0, 0.5 * cur + 0.5 * observed[d] * 1.1)
        else:
            eff[d] = min(top, cur * 1.25)
    return eff


def gpu_ceiling(gpu, gpu_cap, planned, observed):
    """The GPU throughput to plan against, and whether this step was
    GPU-bound (so adapt_caps must not run on it).

    plan_rates trusts --gpu-cap, but the engine is not always ours alone:
    another job on the card cuts what we get. Then the decoder, serving the
    raw queues in turn, shares the shortfall over every lane that asked for
    more than its turn -- the slowest drive, the one that sets the finish,
    included -- and adapt_caps would read that as several disks being slow
    and cut them below what they can read. A disk falls short alone; the
    GPU makes two or more lanes fall short at once. Such a step plans
    against what was delivered and leaves the drive caps alone. Otherwise
    the ceiling probes back up 5% a step, never past --gpu-cap.
    """
    if not gpu_cap:
        return None, False
    lanes = [d for d, p in planned.items() if p and p > 0 and d in observed]
    tot_p = sum(planned[d] for d in lanes)
    tot_o = sum(observed[d] for d in lanes)
    short = [d for d in lanes if observed[d] < 0.85 * planned[d]]
    if tot_o < 0.92 * tot_p and (len(short) >= 2 or short == lanes):
        return max(1.0, min(gpu_cap, tot_o)), True
    return min(gpu_cap, (gpu or gpu_cap) * 1.05), False


def plan_rates(rem, prio_rem, cap, gpu_cap=None):
    """{drive: img/s} for the token buckets; 0 pauses a lane.

    The lanes still holding priority-region images are paced to finish
    THAT work together, and the GPU they cannot use (their read caps, not
    the engine, being the limit) is split over every lane's spare read
    capacity the same way. No priority work left: every lane is the
    priority, so each is paced to finish together and the GPU the slowest
    drive cannot use still goes to the others. Pacing them DOWN to the
    slowest drive's finish only idled the GPU: after North America,
    capybara at 23 img/s held jackal (43 -> 15) and bobcat with it. Without
    a GPU cap there is no measure of spare GPU, so lanes without priority
    work wait, and with none left at all it is the plain finish-together.
    """
    live = {d: r for d, r in prio_rem.items() if r > 0 and d in rem}
    if not live:
        live = {d: r for d, r in rem.items() if r > 0}
        if not live:
            return {d: 0.0 for d in rem}
    t = finish_time(live, cap, gpu_cap)
    rates = {d: 0.0 for d in rem}
    rates.update({d: live[d] / t for d in live})
    left = gpu_cap - sum(rates.values()) if gpu_cap else 0.0
    # Whatever GPU the priority work cannot use goes to EVERY lane with read
    # capacity to spare, priority lanes included: a lane holding only a few
    # priority images (lynx has 64 of North America's 13M) would otherwise
    # crawl through them at the priority pace and do nothing else for days.
    spare = {d: cap.get(d, 50.0) - rates[d] for d in rem}
    # a finished lane has nothing to read: no GPU goes to it
    spare = {d: c for d, c in spare.items() if c > 1e-9 and rem[d] > 0}
    if left < 1.0 or not spare:
        return rates
    # water-fill: split what is left in proportion to each lane's remaining
    # work, never past a lane's spare read capacity, and hand any excess on
    pool = dict(spare)
    while pool and left > 1e-9:
        tot = sum(rem[d] for d in pool)
        share = {d: left * rem[d] / tot for d in pool}
        full = [d for d in pool if share[d] >= pool[d]]
        if not full:
            for d in pool:
                rates[d] += share[d]
            break
        for d in full:
            rates[d] += pool.pop(d)
            left -= spare[d]
    return rates

RAW_BYTES = 4 << 30

FS_IOC_FIEMAP = 0xC020660B


def phys_offset(path):
    """Byte offset of a file's first extent on its block device (FIEMAP),
    or None when it cannot be had (missing file, no FIEMAP support)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        # struct fiemap (32 B) + one struct fiemap_extent (56 B)
        buf = bytearray(88)
        struct.pack_into('QQIIII', buf, 0, 0, (1 << 63) - 1, 0, 0, 1, 0)
        fcntl.ioctl(fd, FS_IOC_FIEMAP, buf)
        if struct.unpack_from('I', buf, 20)[0] < 1:  # no extent mapped
            return None
        return struct.unpack_from('Q', buf, 40)[0]
    except OSError:
        return None
    finally:
        os.close(fd)


class DirInodes:
    """name -> inode number from the directory entries alone (scandir's
    d_ino needs no inode read), for the last couple of directories."""

    def __init__(self, keep=2):
        self.keep = keep
        self.maps = {}

    def __call__(self, path):
        d, name = os.path.split(path)
        m = self.maps.get(d)
        if m is None:
            try:
                m = {e.name: e.inode() for e in os.scandir(d)}
            except OSError:
                m = {}
            while len(self.maps) >= self.keep:
                self.maps.pop(next(iter(self.maps)))
            self.maps[d] = m
        return m.get(name, 0)


def disk_ordered(work, window, phys=phys_offset, inode=None, stop=None):
    """Re-yield (meta, path) work so each run of ``window`` items is read in
    on-disk order.

    The worklist is in image-id order, and on capybara a cell's files are
    scattered over the whole 20 TB platter, so consecutive reads were
    full-stroke seeks: 16.6 img/s against 38.6 for the same files sorted by
    block (40,000 at a time, measured while the sweep ran). Offsets are
    looked up in inode order, which keeps that lookup near-sequential too,
    and a window's lookup (~2 min for 40,000 under load) runs in a
    background thread while the previous window is being read, so the drive
    never sits idle waiting for it -- an idle minute would read to the pacer
    as a dead disk. Every item is yielded exactly once; only the order
    changes, and the collector already takes a shard's results in any
    order -- but a graceful stop commits only each shard's contiguous
    prefix, so up to a window of read-but-scattered work is redone after a
    restart. A file whose offset cannot be had sorts first and reports itself
    when it is read. ``stop`` (an Event) ends the lookahead early.
    """
    if not window or window <= 1:
        yield from work
        return
    inode = inode or DirInodes()

    def ordered(buf):
        offs = {}
        for i in sorted(range(len(buf)), key=lambda i: inode(buf[i][1])):
            if stop is not None and stop.is_set():
                return buf  # unsorted is fine: nothing more will be read
            offs[i] = phys(buf[i][1])
        order = sorted(range(len(buf)),
                       key=lambda i: (offs[i] is not None, offs[i] or 0, i))
        return [buf[i] for i in order]

    # the reader's window, one sorted and queued, one being sorted: lane_work
    # counts an item issued when it is pulled here, so --max-images sees up
    # to three windows go at once (smoke runs: keep the window small)
    q = queue.Queue(maxsize=1)
    END = object()

    def ahead():
        try:
            buf = []
            for item in work:
                buf.append(item)
                if len(buf) >= window:
                    q.put(ordered(buf))
                    buf = []
                    if stop is not None and stop.is_set():
                        break
            if buf:
                q.put(ordered(buf))
        except BaseException as e:  # noqa: BLE001 -- surfaced to the reader
            q.put(e)
            return
        q.put(END)

    threading.Thread(target=ahead, daemon=True, name='disk-order').start()
    while True:
        got = q.get()
        if got is END:
            return
        if isinstance(got, BaseException):
            raise got
        yield from got


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


def cell_complete(detect_root, gen, cell):
    """(complete, why) -- does every shard of every drive of `cell` tile?

    store.compact() DELETES the parts it merges, and its docstring hands this
    check to sweep.py by name: "the caller (sweep.py) is responsible for the
    cell-is-done check against the frozen worklist". Compacting a cell the
    sweep has not finished would merge the committed prefix and delete the
    parts, and the shard-relative resume the next run does over that pair
    would then find nothing and redo ranges that are already in the merged
    file -- so the rule is post-run only, and this is where it is enforced.

    Read-only: tiling_resume is called with repair=False, so nothing on disk
    is touched by asking.
    """
    gd = os.path.join(detect_root, 'worklist', 'gen=%04d' % gen)
    dirs = json.load(open(os.path.join(gd, '_dirs.json')))
    # sidx is GLOBAL across every pair in _dirs.json order, exactly as
    # lane_plan numbers it; a per-cell counter would ask about the wrong shard
    sidx, seen, unfinished = 0, False, []
    for pr in dirs:
        ids = np.load(os.path.join(gd, pr['cell'], pr['drive'] + '.ids.npy'),
                      mmap_mode='r')
        n = len(ids)
        for st in range(0, n, SHARD):
            en = min(st + SHARD, n)
            if pr['cell'] == cell:
                seen = True
                sd = store.pair_dir(detect_root, gen, pr['region'],
                                    pr['cell'], pr['drive'])
                done = False
                if os.path.isdir(sd):
                    _, done = store.tiling_resume(sd, en - st,
                                                  shard_idx=sidx,
                                                  repair=False)
                if not done:
                    unfinished.append((pr['drive'], sidx, st, en))
            sidx += 1
    if not seen:
        return False, f'cell {cell} is not in the gen {gen} worklist'
    if unfinished:
        d, i, a, b = unfinished[0]
        return False, (f'{len(unfinished)} shard(s) of cell {cell} are not '
                       f'finished, e.g. drive {d} shard {i} [{a},{b})')
    return True, ''


def cmd_compact(args):
    """Rewrite one finished cell into a single img+det pair (store.compact)."""
    detect_root = store.get_detect_root()
    ok, why = cell_complete(detect_root, args.gen, args.cell)
    if not ok:
        print(f'refusing to compact: {why}', file=sys.stderr)
        return 2
    if not args.execute:
        print(f'cell {args.cell} (gen {args.gen}) is complete.\n'
              'Compaction rewrites it as one img+det pair and DELETES the '
              'part files once the merged pair verifies.\nre-run with '
              '--execute')
        return 0
    out = store.compact(args.gen, args.cell, detect_root)
    if not out:
        print(f'cell {args.cell} has no parts -- already compacted')
        return 0
    print('compacted -> %s\n            %s' % out)
    return 0


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
                 base_region=None,
                 base_positives=0,
                 base_boxes=0):
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
        # Global like base_done, and for the same reason: imgs_done is
        # all-time, so a per-process positives counter made positive_rate
        # (positives / imgs_done) collapse to ~0% after every restart.
        self.base_positives = int(base_positives or 0)
        self.base_boxes = int(base_boxes or 0)
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
                           boxes_total=self.base_boxes + self.boxes,
                           positives=self.base_positives + self.positives,
                           run_positives=self.positives,
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
    # callers that build the namespace by hand (tests, other tools) predate
    # the pacing overrides; their absence means the defaults, not a crash
    for k, v in dict(cap=None,
                     regions_first=None,
                     readers=None,
                     decoders=DECODERS,
                     consumers=1,
                     ring_slots=None,
                     gpu_cap=None,
                     disk_order=None).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    cfg = load_cfg()
    cap = dict(CAP, **parse_drive_map(args.cap, float, '--cap'))
    readers = dict(READERS, **parse_drive_map(args.readers, int, '--readers'))
    disk_window = parse_drive_map(args.disk_order, int, '--disk-order')
    print(f'pacing: caps {cap}, readers {readers}, decoders {args.decoders}, '
          f'consumers {args.consumers}, gpu cap {args.gpu_cap or "none"}, '
          f'disk order {disk_window or "none"}')
    if args.engine:
        # a re-run with a promoted detector, or the resume of one: the run
        # must keep loading the engine it started with, whatever the config
        # default is, or one store ends up holding two models' boxes
        cfg['engine'] = (args.engine if os.path.isabs(args.engine) else
                         os.path.join(REPO, args.engine))
        if not os.path.exists(cfg['engine']):
            print(f'engine not found: {cfg["engine"]}', file=sys.stderr)
            return 1
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
    first = set(args.regions_first or ())
    unknown = first - set(region_corpus)
    if unknown:
        print(f'--regions-first: no such region {sorted(unknown)}; known: '
              f'{sorted(region_corpus)}', file=sys.stderr)
        return 1
    if first:
        # stable: within each group the worklist order is kept, so a resume
        # walks the same shards in the same order
        lanes = {
            d: sorted(v, key=lambda it: it[0]['region'] not in first)
            for d, v in lanes.items()
        }
    todo = {
        d:
        sum((sh[2] - sh[1]) - sum(e - s for s, e in sh[3])
            for _, _, shards in v for sh in shards)
        for d, v in lanes.items()
    }
    prio_todo = {
        d: sum((sh[2] - sh[1]) - sum(e - s for s, e in sh[3])
               for pr, _, shards in v if pr['region'] in first
               for sh in shards)
        for d, v in lanes.items()
    }
    if first:
        print(f'regions first: {sorted(first)}, '
              f'{sum(prio_todo.values()):,} images: {prio_todo}')
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
    # All-time positives/boxes from the store, so the headline and the rate
    # survive a restart. One aggregate at startup; never on the 5 s publish
    # path. A failure here must not stop the sweep -- it only costs the
    # pre-restart history in the panel.
    base_positives = base_boxes = 0
    try:
        _img = store._sql_src(store._store_globs(detect_root, 'img'))
        _r = store._run_queries({
            'q': f'SELECT count(*) FILTER (WHERE status=0 AND n_det>0), '
                 f'coalesce(sum(n_det),0) FROM {_img}'
        })['q']
        base_positives, base_boxes = int(_r[0][0]), int(_r[0][1])
        print(f'already detected: {base_positives:,} positives / '
              f'{base_boxes:,} boxes (all-time)')
    except Exception as e:
        print(f'could not seed all-time positives ({e}); '
              f'panel will show this run only', file=sys.stderr)
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
    # Record which model is about to write boxes, BEFORE any are written.
    # run_id is time.time() & 0xFFFF -- it identifies this process, not this
    # model, and it wraps every ~18 hours. gen=0001 already held eleven of
    # them, all assumed to be train-30 because nothing said otherwise. Never
    # fatal: a sweep that cannot write its provenance should still sweep, it
    # just says so loudly.
    model_sha8 = None
    try:
        import run_manifest
        run_manifest.write_for_run(REPO, args.gen, run_id, cfg,
                                   started=epoch)
        model_sha8 = run_manifest.sha256_of(cfg['engine'])[:8]
    # BaseException, not Exception: run_manifest raises SystemExit for a
    # missing engine, which sailed straight through an `except Exception` and
    # killed a 32.5M-image sweep over a bookkeeping failure -- the exact
    # opposite of what the comment above promises.
    except BaseException as e:
        print(f'WARNING: could not write the run manifest ({e!r}); the boxes '
              f'from run_id={run_id} will not be attributable to a model',
              file=sys.stderr)
    status = StatusWriter(run_id,
                          args.gen,
                          corpus_total,
                          drive_totals=corpus,
                          region_totals=region_corpus)
    status.start()
    # the digest the manifest recorded, so each row says which model made it
    # max_det too: the writer's NMS-saturation guard compares against it, and
    # left at the module default it watches for 300 boxes on an engine that
    # can only ever return cfg['max_det'].
    writer = store.Writer(detect_root, max_det=int(cfg.get('max_det')
                                                   or store.MAX_DET),
                          model_sha8=model_sha8)
    coll = ShardCollector(writer,
                          args.gen,
                          status,
                          base_done=base_done,
                          base_drive=base_drive,
                          base_region=base_region,
                          base_positives=base_positives,
                          base_boxes=base_boxes)
    ring = engine.BatchRing(torch, slots=args.ring_slots
                            or engine.RING_SLOTS)
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
    # Several consumers, each with its own execution context and buffers,
    # take batches from the one ring: while one is in NMS and the sink (CPU
    # work, GPU idle) another's batch is on the GPU. Output is per image and
    # the collector is locked, so the order batches finish in does not matter.
    consumers = [
        engine.Consumer(cfg['engine'],
                        ring,
                        sink,
                        conf=cfg['conf'],
                        iou=cfg['iou'],
                        max_det=cfg['max_det'],
                        preview_fn=preview)
        for _ in range(max(1, args.consumers))
    ]
    cts = [threading.Thread(target=c.run, daemon=True) for c in consumers]
    for ct in cts:
        ct.start()

    rawqs = {d: queue.Queue() for d in lanes}
    budgets = {d: engine.ByteBudget(RAW_BYTES) for d in lanes}
    dts = [
        threading.Thread(target=engine.decoder_loop,
                         args=(ring, rawqs, budgets, stop_ev, sink),
                         daemon=True) for _ in range(max(1, args.decoders))
    ]
    for t in dts:
        t.start()

    drive_codes = {d: i for i, d in enumerate(sorted(CAP))}
    buckets = {d: engine.TokenBucket(stop=stop_ev) for d in lanes}
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

    eff = dict(cap)  # adaptive read capacities (adapt_caps)
    pace_state = {'t': None, 'done': {}, 'planned': {}, 'gpu': None}

    def repace():
        if args.max_images and issued[0] >= args.max_images:
            # every lane must reach lane_work to see the limit and return;
            # a paused lane never would, and the run would never end
            for d in lanes:
                buckets[d].set_rate(None)
            return
        rem = {d: max(0, todo[d] - coll.drive_done.get(d, 0)) for d in lanes}
        # lanes are ordered priority-first, so what a lane has done comes
        # off its priority work before anything else
        prio_rem = {
            d: max(0, prio_todo[d] - coll.drive_done.get(d, 0))
            for d in lanes
        }
        now = time.monotonic()
        done = {d: coll.drive_done.get(d, 0) for d in lanes}
        if pace_state['t'] is not None and now - pace_state['t'] > 5:
            dt = now - pace_state['t']
            observed = {
                d: (done[d] - pace_state['done'].get(d, 0)) / dt
                for d in lanes
            }
            gpu, gpu_bound = gpu_ceiling(pace_state['gpu'], args.gpu_cap,
                                         pace_state['planned'], observed)
            pace_state['gpu'] = gpu
            if not gpu_bound:
                adapt_caps(eff, cap, pace_state['planned'], observed)
        # a lane with nothing left is not planned (no GPU is set aside for
        # it) but is not paused either: a reader that still holds an item
        # (the done count lags the reads) must never wait on a 0 rate
        work_rem = {d: r for d, r in rem.items() if r > 0}
        rates = plan_rates(work_rem, prio_rem, eff,
                           pace_state['gpu'] or args.gpu_cap)
        rates.update({d: None for d in lanes if d not in work_rem})
        pace_state.update(t=now, done=done,
                          planned={d: r for d, r in rates.items()
                                   if r is not None})
        for d in lanes:
            buckets[d].set_rate(args.rate_cap or rates[d])

    def pacer():
        while not stop_ev.is_set():
            time.sleep(60 if not args.max_images else 2)
            repace()

    # rates are in place BEFORE the first reader starts: an unset bucket is
    # unlimited, and a lane meant to wait would read a burst first
    repace()
    rthreads = []
    for d, items in lanes.items():
        work = disk_ordered(lane_work(d, items), disk_window.get(d, 0),
                            stop=stop_ev)
        wlock = threading.Lock()

        def locked(work=work, wlock=wlock):
            # Take the next item under the lane lock, then yield it with the
            # lock RELEASED. Yielding inside the `with` held the lock for as
            # long as the reader spent reading that file, so a lane's readers
            # took turns instead of reading in parallel (bobcat's 8 were one),
            # and a reader that returned on stop could leave its suspended
            # generator holding the lock forever: the other readers of the
            # lane blocked, the join never returned, and a graceful stop hung.
            while True:
                with wlock:
                    try:
                        item = next(work)
                    except StopIteration:
                        return
                yield item

        for _ in range(readers.get(d, 1)):
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
    for c in consumers:
        c.stopping.set()
    for ct in cts:
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
    r.add_argument('--engine',
                   help='the TensorRT engine to load instead of the '
                        'configured one (absolute, or relative to the repo)')
    r.add_argument('--cap',
                   nargs='+',
                   metavar='DRIVE=IMG_S',
                   help='override a drive\'s measured read capacity')
    r.add_argument('--readers',
                   nargs='+',
                   metavar='DRIVE=N',
                   help='override a drive\'s reader thread count')
    r.add_argument('--disk-order',
                   nargs='+',
                   metavar='DRIVE=WINDOW',
                   help='read each run of WINDOW images on a drive in '
                        'on-disk order (FIEMAP) instead of image-id order')
    r.add_argument('--decoders', type=int, default=DECODERS)
    r.add_argument('--consumers',
                   type=int,
                   default=1,
                   help='GPU consumers, each its own TensorRT context')
    r.add_argument('--ring-slots', type=int)
    r.add_argument('--gpu-cap',
                   type=float,
                   help='img/s the engine sustains; the pacer never plans '
                   'past it')
    r.add_argument('--regions-first',
                   nargs='+',
                   metavar='REGION',
                   help='sweep these regions before any other: lanes '
                        'without them wait unless the GPU has room')
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
        py=os.environ.get('SWEEP_PYTHON') or sys.executable)))
    # store.compact() is the documented post-run step and had no entry point
    # at all: `sweep.py compact` answered "invalid choice". It stayed
    # reachable only by importing store by hand, which is exactly the move
    # this repo has been burned by.
    cp = sub.add_parser('compact',
                        help='rewrite one FINISHED cell as a single img+det '
                             'pair (deletes the parts once it verifies)')
    cp.add_argument('--gen', type=int, default=1)
    cp.add_argument('--cell', required=True)
    cp.add_argument('--execute', action='store_true')
    cp.set_defaults(func=cmd_compact)
    # These two take no arguments -- they read the whole store, not one
    # generation. The usage block used to advertise --gen on them and argparse
    # answered "unrecognized arguments".
    for name, fn in (('verify', store.verify), ('invariants',
                                                store.invariants)):
        c = sub.add_parser(name)
        c.set_defaults(func=lambda a, fn=fn: print(fn()))
    args = p.parse_args()
    rc = args.func(args)
    sys.exit(rc or 0)


if __name__ == '__main__':
    main()
