#!/usr/bin/env python3
"""Phase-0 gates for engine.py that need no real images.

Gate A  ring integrity under load: 16 producers hammer the ring while a
        consumer checks every row's pixel signature against its meta --
        the exact stale-pixel failure the prototype had at 2.58% must be 0.
Gate B  letterbox geometry == ultralytics LetterBox on adversarial sizes.
Gate C  coordinate transform round-trips to original pixels within 1e-3,
        including odd dimensions where W0/Wd != r_dec.
Gate D  SOF parser vs cv2 on real jpgs from the corpus.

Run under the yolo env.
"""
import glob
import os
import queue
import random
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import engine  # noqa: E402

FAILED = []


def check(name, ok, detail=''):
    print(('ok    ' if ok else 'FAIL  ') + name + ((' ' + detail) if detail and not ok else ''))
    if not ok:
        FAILED.append(name)


def gate_ring_integrity():
    import torch
    N = 8000
    ring = engine.BatchRing(torch, slots=6, bs=8, imgsz=64)
    bad = [0]
    done_ct = [0]
    lock = threading.Lock()

    def consumer():
        while done_ct[0] < N or not ring.batq.empty():
            try:
                s, n, metas = ring.batq.get(timeout=0.5)
            except queue.Empty:
                continue
            for j in range(n):
                m = metas[j]
                # signature: every byte of the row must equal image_id % 251
                row = ring.pin_np[s][j]
                if not (row == m['sig']).all():
                    with lock:
                        bad[0] += 1
            with lock:
                done_ct[0] += n
            ring.release(s)

    ct = threading.Thread(target=consumer)
    ct.start()

    ids = list(range(N))
    random.shuffle(ids)
    chunks = [ids[i::16] for i in range(16)]

    def producer(chunk):
        for iid in chunk:
            sig = iid % 251
            m = {'image_id': iid, 'sig': sig}
            s, j = ring.claim(m)
            # deliberately slow write path: memset AFTER claim, with a yield
            # in between, to widen any claim-vs-written race window
            row = ring.pin_np[s][j]
            row[:32] = sig
            if iid % 7 == 0:
                import time
                time.sleep(0)
            row[32:] = sig
            ring.done(s)

    ps = [threading.Thread(target=producer, args=(c,)) for c in chunks]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    ring.flush_partial()
    ct.join(timeout=30)
    check('ring integrity: 0 stale/crossed rows of %d' % N,
          bad[0] == 0 and done_ct[0] == N,
          f'bad={bad[0]} done={done_ct[0]}')


def gate_letterbox():
    from ultralytics.data.augment import LetterBox
    rng = np.random.default_rng(0)
    sizes = [(4080, 3072), (1920, 1080), (1281, 1279), (640, 481), (5279, 1057),
             (1280, 1280), (37, 4093), (2040, 1536), (963, 1280)]
    mism = 0
    for (w, h) in sizes:
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        ref = LetterBox((1280, 1280), auto=False, scaleup=False)(image=img)
        s, nw, nh, left, top = engine.letterbox_params(w, h)
        out = np.full((1280, 1280, 3), engine.PAD_VALUE, np.uint8)
        import cv2
        if (nw, nh) != (w, h):
            cv2.resize(img, (nw, nh), dst=out[top:top + nh, left:left + nw],
                       interpolation=cv2.INTER_LINEAR)
        else:
            out[top:top + nh, left:left + nw] = img
        if not (out == ref).all():
            mism += 1
    check('letterbox byte-identical to ultralytics on %d sizes' % len(sizes),
          mism == 0, f'{mism} mismatched')


def gate_transform_roundtrip():
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(500):
        w0 = int(rng.integers(200, 8000))
        h0 = int(rng.integers(200, 8000))
        r = engine.choose_reduce(w0, h0)
        wd = -(-w0 // r)          # ceil, like IMREAD_REDUCED
        hd = -(-h0 // r)
        s, nw, nh, left, top = engine.letterbox_params(wd, hd)
        m = dict(w0=w0, h0=h0, wd=wd, hd=hd, s=s, left=left, top=top)
        # a box in original pixels, forward to net space, back again
        x1, y1 = rng.uniform(0, w0 * 0.8), rng.uniform(0, h0 * 0.8)
        x2, y2 = x1 + rng.uniform(1, w0 - x1), y1 + rng.uniform(1, h0 - y1)
        fx = lambda x: (x * wd / w0) * s + left
        fy = lambda y: (y * hd / h0) * s + top
        net = np.array([[fx(x1), fy(y1), fx(x2), fy(y2)]], np.float32)
        back = engine.boxes_to_original(net, m)[0]
        err = np.abs(back - np.array([x1, y1, x2, y2])).max()
        worst = max(worst, err / max(w0, h0))
    check('transform roundtrip worst rel err < 1e-3', worst < 1e-3,
          f'worst={worst:.2e}')


def gate_sof():
    import cv2
    roots = [ln.strip() for ln in
             open(os.path.join(os.path.dirname(__file__), '..', '..', '..',
                               'data', 'catalog_dirs.txt'))
             if ln.strip() and not ln.startswith('#')]
    paths = []
    for root in roots:
        got_root = 0
        # walk until this root yields 40 jpgs -- the alphabetically first
        # cells (Antarctica...) are empty on most drives
        for d in sorted(glob.glob(os.path.join(root, '*',
                                               'ground_animal_images'))):
            js = sorted(glob.glob(os.path.join(d, '*.jpg')))[:15]
            paths += js
            got_root += len(js)
            if got_root >= 40:
                break
    bad = 0
    tested = 0
    for p in paths:
        try:
            raw = open(p, 'rb').read()
        except OSError:
            continue
        dims = engine.sof_dims(raw)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8),
                           cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            continue
        tested += 1
        if dims is None or dims != (img.shape[1], img.shape[0]):
            bad += 1
        if tested >= 150:
            break
    check(f'SOF dims == cv2 dims on {tested} real jpgs', tested >= 100 and bad == 0,
          f'bad={bad} tested={tested}')


def main():
    gate_ring_integrity()
    gate_letterbox()
    gate_transform_roundtrip()
    gate_sof()
    print()
    if FAILED:
        print('FAILED:', FAILED)
        return 1
    print('all engine gates passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
