#!/usr/bin/env python3
"""
The sweep's GPU engine: readers -> decode pool -> pinned batch ring -> TensorRT
-> NMS -> coordinate transform -> store rows. Spec: DETECTION_RUN_STRATEGY.md
section 4 (shape, ring fix, decode, transform) and section 6.3 (commit order).

Design notes that are load-bearing, not decoration:

* Ring publish is on rows WRITTEN, not rows claimed (section 4.3). The
  prototype published a slot when its last row was *claimed* and fed 2.58% of
  images to the GPU as another image's stale pixels. Decoders copy pixels into
  the pinned slot first, then increment ``written[s]`` under the lock; the
  slot is queued only when written == expected. The graceful-stop partial
  flush goes through the identical path by shrinking ``expected[s]``.

* One persistent device tensor per ring slot (section 4.3, second hazard).
  Allocating the H2D destination on a side stream hands the caching allocator
  a block that can be re-issued while the previous batch's permute still reads
  it. ``dst[s].copy_(pin[s], non_blocking=True)`` on a dedicated copy stream,
  with pre-allocated ``dst``, removes the allocator from the picture.

* Batches mix drives and cells (four lanes feed one ring), so results are
  routed back to their shard by ``(pair, shard_idx, pos)`` and a shard commits
  when a contiguous prefix of its positions is complete -- out-of-order arrival
  within a shard is expected and buffered, holes are impossible because every
  enqueued position yields exactly one row (ok, missing, or error).

* NMS runs with ``max_time_img`` pinned huge (section 4.6): ultralytics'
  wall-clock break returns empty tensors for the rest of the batch --
  byte-identical to genuine negatives -- after 2.4 s. We fail loud instead.

Runs under the *yolo* env (python 3.11, torch cu128, ultralytics pinned
8.3.165 -- 8.4 switches yolo26 to NMS-free decode and ignores ``iou``,
invalidating every measured threshold curve; tensorrt pinned 10.12 -- the
TRT 11 pip default targets CUDA 13 and its Builder returns null here).
"""

import json
import os
import queue
import struct
import threading
import time

import numpy as np

# torch/cv2/tensorrt are imported lazily so worklist/store tooling can import
# this module's pure helpers (SOF parse, transform math) under any env.

BS = 8  # static TRT engine batch (section 4.6)
IMGSZ = 1280
RING_SLOTS = 6  # 6 x 8 x 1280^2 x 3 = 225 MiB pinned (section 4.1)
PAD_VALUE = 114

# ── pure helpers (no torch) ─────────────────────────────────────────────────


def sof_dims(buf):
    """(width, height) from JPEG SOFn, parsed from raw bytes (section 4.4).

    Microseconds, no decode, immune to the 1.53% parquet-metadata drift.
    Returns None if no SOF marker is found (caller treats as decode_error).
    """
    if len(buf) < 4 or buf[0] != 0xFF or buf[1] != 0xD8:
        return None
    i = 2
    n = len(buf)
    while i + 9 < n:
        if buf[i] != 0xFF:
            i += 1
            continue
        marker = buf[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA,
                      0xCB, 0xCD, 0xCE, 0xCF):
            h = (buf[i + 5] << 8) | buf[i + 6]
            w = (buf[i + 7] << 8) | buf[i + 8]
            return (w, h)
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg = (buf[i + 2] << 8) | buf[i + 3]
        i += 2 + seg
    return None


def choose_reduce(w0, h0, reduce_max=8, min_long=IMGSZ):
    """Largest r in {8,4,2,1} (capped) with long_side/r >= min_long (§4.4)."""
    long_side = max(w0, h0)
    for r in (8, 4, 2, 1):
        if r <= reduce_max and long_side / r >= min_long:
            return r
    return 1


def letterbox_params(wd, hd, imgsz=IMGSZ):
    """Exact ultralytics LetterBox(auto=False, scaleup=False) geometry (§4.5).

    Returns (s, nw, nh, left, top) -- verified byte-identical to
    ultralytics.data.augment.LetterBox on real images in the strategy work.
    """
    s = min(imgsz / wd, imgsz / hd, 1.0)
    nw, nh = round(wd * s), round(hd * s)
    dw, dh = (imgsz - nw) / 2, (imgsz - nh) / 2
    left = int(round(dw - 0.1))
    top = int(round(dh - 0.1))
    return s, nw, nh, left, top


def boxes_to_original(xyxy, m):
    """Net-space xyxy -> ORIGINAL full-res pixels (section 4.5).

    Uses W0/Wd rather than r_dec so ceil() on odd dimensions is exact, then
    clips. ``m`` is the per-image meta dict.
    """
    out = np.asarray(xyxy, dtype=np.float32).copy()
    out[:,
        [0, 2]] = (out[:, [0, 2]] - m['left']) / m['s'] * (m['w0'] / m['wd'])
    out[:, [1, 3]] = (out[:, [1, 3]] - m['top']) / m['s'] * (m['h0'] / m['hd'])
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, m['w0'])
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, m['h0'])
    return out


def load_trt_engine(path):
    """Deserialize an ultralytics-exported .engine (JSON header + blob)."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    with open(path, 'rb') as f:
        hlen = struct.unpack('<I', f.read(4))[0]
        meta = json.loads(f.read(hlen).decode('utf-8'))
        blob = f.read()
    engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
    if engine is None:
        raise RuntimeError(
            f'TRT engine failed to deserialize: {path} '
            '(TRT version mismatch? engine was built for 10.12)')
    return engine, meta


# ── the ring (section 4.3) ──────────────────────────────────────────────────


class BatchRing:
    """Pinned batch ring with written-count publish.

    Decoders: s, j = claim(meta); write pixels into pin_np[s][j]; done(s).
    Consumer:  s, n, metas = batq.get(); ... ; release(s).
    flush_partial() publishes any half-filled current slot (graceful stop),
    via the same written==expected gate as a full slot.
    """

    def __init__(self, torch, slots=RING_SLOTS, bs=BS, imgsz=IMGSZ):
        self.bs = bs
        self.pin = [
            torch.empty((bs, imgsz, imgsz, 3),
                        dtype=torch.uint8,
                        pin_memory=True) for _ in range(slots)
        ]
        self.pin_np = [p.numpy() for p in self.pin]
        self.lock = threading.Lock()
        self.freeq = queue.Queue()
        for s in range(slots):
            self.freeq.put(s)
        self.batq = queue.Queue()
        self.meta = [[None] * bs for _ in range(slots)]
        self.written = [0] * slots
        self.expected = [bs] * slots
        self.cur = None  # slot currently being filled
        self.cur_n = 0  # rows claimed in cur

    def claim(self, m):
        """Claim (slot, row) for one image; blocks when the GPU is behind.

        Row assignment happens entirely under the lock; the only blocking op
        (freeq.get) happens outside it, and losing the open-a-new-slot race
        returns the slot to the free list -- otherwise two decoders that both
        observed ``cur is None`` would cross-account rows between two slots.
        """
        while True:
            with self.lock:
                if self.cur is not None:
                    s = self.cur
                    j = self.cur_n
                    self.meta[s][j] = m
                    self.cur_n += 1
                    if self.cur_n == self.bs:
                        self.cur = None  # next claim opens a fresh slot
                    return s, j
            s = self.freeq.get()  # back-pressure lives here (§4.1)
            with self.lock:
                if self.cur is None:
                    self.cur = s
                    self.cur_n = 0
                    self.written[s] = 0
                    self.expected[s] = self.bs
                else:
                    self.freeq.put(s)  # lost the race; recycle

    def done(self, s):
        """Mark one row's pixels WRITTEN; publish iff the slot is complete."""
        with self.lock:
            self.written[s] += 1
            if self.written[s] == self.expected[s]:
                self.batq.put((s, self.expected[s], list(self.meta[s])))

    def flush_partial(self):
        """Publish the half-filled slot at shutdown (section 4.3: identical
        written==expected treatment -- shrink expected, publish only when the
        in-flight decodes have landed)."""
        with self.lock:
            s, n = self.cur, self.cur_n
            if s is None or n == 0:
                return
            self.cur = None
            self.expected[s] = n
            if self.written[s] == n:
                self.batq.put((s, n, list(self.meta[s])))
        # else: the last done() call publishes it.

    def release(self, s):
        self.freeq.put(s)


# ── decode one image into a ring slot (sections 4.4/4.5) ────────────────────


def decode_into(cv2, ring, raw, m, reduce_max=8):
    """SOF parse -> reduced imdecode -> letterbox straight into a pinned row.

    Fills m with the transform fields the consumer needs. Returns True on
    success; on failure the caller emits a decode_error row instead.
    """
    dims = sof_dims(raw)
    if dims is None:
        return False
    w0, h0 = dims
    r = choose_reduce(w0, h0, reduce_max)
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    flags |= {
        1: 0,
        2: cv2.IMREAD_REDUCED_COLOR_2,
        4: cv2.IMREAD_REDUCED_COLOR_4,
        8: cv2.IMREAD_REDUCED_COLOR_8
    }[r]
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), flags)
    if img is None:
        return False
    hd, wd = img.shape[:2]
    if max(wd, hd) < IMGSZ and r > 1:
        # reduced decode undershot (tiny source); redo at full res (§4.4)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8),
                           cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            return False
        r = 1
        hd, wd = img.shape[:2]
    s, nw, nh, left, top = letterbox_params(wd, hd)
    m.update(w0=w0, h0=h0, wd=wd, hd=hd, s=s, left=left, top=top, reduce=r)
    slot, j = ring.claim(m)
    row = ring.pin_np[slot][j]
    row[...] = PAD_VALUE
    if (nw, nh) != (wd, hd):
        cv2.resize(img, (nw, nh),
                   dst=row[top:top + nh, left:left + nw],
                   interpolation=cv2.INTER_LINEAR)
    else:
        row[top:top + nh, left:left + nw] = img
    ring.done(slot)
    return True


# ── the GPU consumer ────────────────────────────────────────────────────────


class Consumer:
    """One thread: ring -> H2D -> TRT -> NMS -> original-px rows -> sink.

    ``sink(meta, status, boxes, confs)`` receives one call per image; boxes
    are float32 (n,4) in ORIGINAL pixels, confs float32 (n,), both empty for
    negatives; status uses the section 5.3 codes (0 ok, 4 infer_error).
    """

    def __init__(self,
                 engine_path,
                 ring,
                 sink,
                 conf=0.05,
                 iou=0.90,
                 max_det=300):
        import torch
        from ultralytics.utils import ops
        self.torch = torch
        self.ops = ops
        self.ring = ring
        self.sink = sink
        self.conf, self.iou, self.max_det = conf, iou, max_det
        self.engine, self.trt_meta = load_trt_engine(engine_path)
        self.ctx = self.engine.create_execution_context()
        names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
        ]
        self.in_name, self.out_name = names[0], names[1]
        in_shape = tuple(self.ctx.get_tensor_shape(self.in_name))
        out_shape = tuple(self.ctx.get_tensor_shape(self.out_name))
        assert in_shape == (BS, 3, IMGSZ, IMGSZ), in_shape
        self.inp = torch.zeros(in_shape, dtype=torch.float32, device='cuda')
        self.out = torch.zeros(out_shape, dtype=torch.float32, device='cuda')
        self.ctx.set_tensor_address(self.in_name, self.inp.data_ptr())
        self.ctx.set_tensor_address(self.out_name, self.out.data_ptr())
        # persistent per-slot device tensors (section 4.3 second hazard)
        self.dst = [
            torch.empty((BS, IMGSZ, IMGSZ, 3),
                        dtype=torch.uint8,
                        device='cuda') for _ in range(len(ring.pin))
        ]
        self.copy_stream = torch.cuda.Stream()
        self.exec_stream = torch.cuda.Stream()
        self.stopping = threading.Event()
        self.batches = 0

    def run(self):
        torch, ops = self.torch, self.ops
        while True:
            try:
                s, n, metas = self.ring.batq.get(timeout=0.5)
            except queue.Empty:
                if self.stopping.is_set():
                    return
                continue
            if s is None:  # poison pill
                return
            with torch.cuda.stream(self.copy_stream):
                self.dst[s].copy_(self.ring.pin[s], non_blocking=True)
            self.exec_stream.wait_stream(self.copy_stream)
            with torch.cuda.stream(self.exec_stream):
                self.inp.copy_(self.dst[s].permute(0, 3, 1, 2).flip(
                    1)  # BGR (cv2) -> RGB
                               .to(torch.float32).div_(255.0))
                ok_exec = self.ctx.execute_async_v3(
                    self.exec_stream.cuda_stream)
                pred = self.out.clone()  # detach from TRT buffer
            self.exec_stream.synchronize()
            self.ring.release(s)  # slot reusable immediately
            if not ok_exec:
                for m in metas[:n]:
                    self.sink(m, 4, np.empty((0, 4), np.float32),
                              np.empty(0, np.float32))
                continue
            # section 4.6: max_time_img pinned huge; if ultralytics ever
            # prints its time-limit warning the run must die, not degrade.
            dets = ops.non_max_suppression(pred,
                                           conf_thres=self.conf,
                                           iou_thres=self.iou,
                                           max_det=self.max_det,
                                           max_time_img=1e9,
                                           agnostic=False)
            for j in range(n):
                m = metas[j]
                d = dets[j]
                if len(d) == 0:
                    self.sink(m, 0, np.empty((0, 4), np.float32),
                              np.empty(0, np.float32))
                    continue
                d = d.float().cpu().numpy()
                boxes = boxes_to_original(d[:, :4], m)
                self.sink(m, 0, boxes, d[:, 4].astype(np.float32))
            self.batches += 1


# ── readers (section 4.1: 1 thread/drive except bobcat 8) ───────────────────


class DriveReader:
    """Reads one lane's files in worklist order under a byte budget.

    ``work`` yields (meta, path) in strict positional order; every yielded
    item produces exactly one downstream row (raw bytes, or an ENOENT /
    read_error emitted straight to the sink -- holes are impossible, which is
    what makes contiguous-prefix commits sound).
    """

    def __init__(self,
                 drive,
                 work,
                 rawq,
                 byte_sem,
                 sink,
                 stop_ev,
                 pace=None,
                 fadvise=True):
        self.drive = drive
        self.work = work
        self.rawq = rawq
        self.byte_sem = byte_sem  # counted in bytes
        self.sink = sink
        self.stop_ev = stop_ev
        self.pace = pace  # TokenBucket or None
        self.fadvise = fadvise

    def run(self):
        for m, path in self.work:
            if self.stop_ev.is_set():
                return
            if self.pace is not None:
                self.pace.take()
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    raw = b''
                    while True:
                        chunk = os.read(fd, 1 << 20)
                        if not chunk:
                            break
                        raw += chunk
                    if self.fadvise:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except FileNotFoundError:
                self.sink(m, 3, np.empty((0, 4), np.float32),
                          np.empty(0, np.float32))
                continue
            except OSError:
                self.sink(m, 1, np.empty((0, 4), np.float32),
                          np.empty(0, np.float32))
                continue
            self.byte_sem.acquire_bytes(len(raw))
            self.rawq.put((m, raw))


class ByteBudget:
    """Semaphore counted in bytes (the 4 GiB/drive raw-queue budget, §4.6)."""

    def __init__(self, limit):
        self.limit = limit
        self.used = 0
        self.cv = threading.Condition()

    def acquire_bytes(self, n):
        with self.cv:
            while self.used + n > self.limit and self.used > 0:
                self.cv.wait()
            self.used += n

    def release_bytes(self, n):
        with self.cv:
            self.used -= n
            self.cv.notify_all()


class TokenBucket:
    """Closed-loop pacing (section 6.5): rate is set externally every 60 s."""

    def __init__(self, rate=None):
        self.lock = threading.Lock()
        self.rate = rate  # img/s or None = unlimited
        self._next = time.monotonic()

    def set_rate(self, rate):
        with self.lock:
            self.rate = rate

    def take(self):
        with self.lock:
            r = self.rate
            if not r:
                return
            wait = self._next - time.monotonic()
            self._next = max(self._next, time.monotonic()) + 1.0 / r
        if wait > 0:
            time.sleep(min(wait, 2.0))


def decoder_loop(ring, rawqs, budgets, stop_ev, sink, reduce_max=8):
    """One of 4 decode threads: drain the per-drive raw queues round-robin."""
    import cv2
    cv2.setNumThreads(1)  # section 4.1: default 16 would 4x
    names = list(rawqs)
    while True:
        got = False
        for d in names:
            try:
                m, raw = rawqs[d].get_nowait()
            except queue.Empty:
                continue
            got = True
            try:
                ok = decode_into(cv2, ring, raw, m, reduce_max)
            except Exception:
                ok = False
            budgets[d].release_bytes(len(raw))
            if not ok:
                sink(m, 2, np.empty((0, 4), np.float32),
                     np.empty(0, np.float32))
        if not got:
            if stop_ev.is_set() and all(q.empty() for q in rawqs.values()):
                return
            time.sleep(0.005)
