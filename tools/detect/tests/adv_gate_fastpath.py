#!/usr/bin/env python3
"""The fast path must judge exactly what predict() judges.

gate_store used to hand model.predict() a list of PIL crops and let the
parent process do ultralytics' classify preprocessing for every one: Resize
(short side to imgsz), CenterCrop, ToTensor, Normalize -- ~3.4 ms a crop on
one core, in series with the GPU. The fast path splits that work without
changing a pixel: the decoders apply the geometry with PIL (prep_crop), the
GPU applies the value map through a table computed by the predictor's own
transforms (value_lut), and the forward is the predictor's AutoBackend.

What has to hold:

  f1  prep_crop + value_lut + batch_input build the tensor
      classify_transforms builds, bit for bit, over crop shapes the store
      holds (tiny boxes upscaled 50x, wide, tall, already at the short side)
  f2  classify_spec refuses every transform chain prep_crop does not
      reproduce, rather than approximating it
  f3  the f1 comparison catches the mistakes that would have shipped:
      ImageNet normalisation, BGR order, a squash instead of resize + centre
      crop, bicubic for bilinear, a rounded long side, /256
  f4  fast_path_check -- the guard run() applies on the live device before
      it judges anything -- passes the real path on a tiny model (CPU) with a
      tolerance of zero, and refuses a wrong normalisation
  f5  DriveLanes yields every job exactly once, never reads a shard past
      the look-ahead, and reports a missing root / unreadable frame as the
      old worker did; a decoded crop comes back as prep_crop's array
  f6  run() keeps --legacy-predict, checks the fast path before the pool
      opens, and its refusal says how to fall back

Runs in the yolo env (ultralytics + torch), CPU only:

    python tools/detect/tests/adv_gate_fastpath.py
"""

import os
import sys
import tempfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))

import gate_store as gs  # noqa: E402


def _shapes():
    import random
    rnd = random.Random(7)
    fixed = [(93, 98), (166, 61), (40, 220), (13, 9), (9, 13), (640, 640),
             (640, 900), (900, 640), (639, 641), (641, 639), (1, 1000),
             (1000, 1), (301, 300), (2000, 1100), (97, 641), (8, 8)]
    return fixed + [(rnd.randint(5, 1500), rnd.randint(5, 1500))
                    for _ in range(40)]


def _crop(w, h, seed):
    import numpy as np
    from PIL import Image
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 256, (h, w, 3), dtype=np.uint8),
                           'RGB')


def _reference(tf, crops):
    """What ClassificationPredictor.preprocess makes of PIL crops: the PIL
    loader's BGR array, back to RGB, through the transforms, stacked."""
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    ims = [np.ascontiguousarray(np.asarray(c)[:, :, ::-1]) for c in crops]
    return torch.stack([tf(Image.fromarray(cv2.cvtColor(im,
                                                        cv2.COLOR_BGR2RGB)))
                        for im in ims], 0).float()


def _fast(tf, crops, imgsz, prep=gs.prep_crop, lut=None):
    spec = gs.classify_spec(tf, imgsz)
    lut = gs.value_lut(tf) if lut is None else lut
    return gs.batch_input([prep(c, spec) for c in crops], lut)


def _mismatch(a, b):
    """'' when bit-identical, else where they first differ."""
    import torch
    if a.shape != b.shape:
        return f'shape {tuple(a.shape)} vs {tuple(b.shape)}'
    if a.dtype != b.dtype:
        return f'dtype {a.dtype} vs {b.dtype}'
    if torch.equal(a, b):
        return ''
    d = (a - b).abs()
    return f'max |diff| {float(d.max()):.3g} over {int((d > 0).sum())} values'


def f1(bad):
    from ultralytics.data.augment import classify_transforms
    for imgsz in (640, 224):
        tf = classify_transforms(imgsz)
        for k, (w, h) in enumerate(_shapes()):
            c = _crop(w, h, k)
            m = _mismatch(_fast(tf, [c], imgsz), _reference(tf, [c]))
            if m:
                bad.append(f'f1: {w}x{h} at imgsz {imgsz}: the fast path '
                           f'differs from classify_transforms -- {m}')
    # the probe crops run() checks on the device, and a batch of them
    tf = classify_transforms(640)
    pc = gs.probe_crops()
    m = _mismatch(_fast(tf, pc, 640), _reference(tf, pc))
    if m:
        bad.append(f'f1: the probe batch differs -- {m}')


def f2(bad):
    import torchvision.transforms as T
    from ultralytics.data.augment import classify_transforms
    good = classify_transforms(640)
    try:
        sp = gs.classify_spec(good, 640)
        if sp['short'] != 640 or sp['crop'] != (640, 640):
            bad.append(f'f2: classify_spec read {sp}')
    except ValueError as e:
        bad.append(f'f2: the stock transform was refused: {e}')
    tt = good.transforms
    refuse = {
        'imgsz disagrees': (good, 512),
        'extra transform': (T.Compose([*tt[:2], T.Grayscale(3), *tt[2:]]),
                            640),
        'no centre crop': (T.Compose([tt[0], *tt[2:]]), 640),
        'max_size': (T.Compose([T.Resize(640, max_size=700), *tt[1:]]), 640),
        '(h, w) resize': (T.Compose([T.Resize((640, 640)), *tt[1:]]), 640),
        'crop pads': (T.Compose([tt[0], T.CenterCrop(700), *tt[2:]]), 640),
        'not a Compose': (object(), 640),
    }
    for why, (tf, sz) in refuse.items():
        try:
            gs.classify_spec(tf, sz)
            bad.append(f'f2: classify_spec accepted a chain with {why}')
        except ValueError:
            pass


def f3(bad):
    import numpy as np
    from PIL import Image
    from ultralytics.data.augment import classify_transforms
    tf = classify_transforms(640)
    crops = [_crop(w, h, k) for k, (w, h) in enumerate(_shapes()[:24])]
    ref = _reference(tf, crops)

    def bgr(c, sp):
        return gs.prep_crop(c, sp)[:, :, ::-1].copy()

    def squash(c, sp):
        return np.array(c.resize(sp['crop'][::-1], sp['resample']))

    def bicubic(c, sp):
        return gs.prep_crop(c, dict(sp, resample=Image.BICUBIC))

    def rounded(c, sp):
        w, h = c.size
        s = sp['short']
        if w <= h:
            nw, nh = s, int(round(s * h / w))
        else:
            nw, nh = int(round(s * w / h)), s
        c = c.resize((nw, nh), sp['resample'])
        top, left = int(round((nh - s) / 2.0)), int(round((nw - s) / 2.0))
        return np.array(c.crop((left, top, left + s, top + s)))

    import torch
    imagenet = ((torch.arange(256).float()[None] / 255
                 - torch.tensor([0.485, 0.456, 0.406])[:, None])
                / torch.tensor([0.229, 0.224, 0.225])[:, None])
    by256 = (torch.arange(256).float() / 256)[None].repeat(3, 1)
    mutants = {
        'BGR order': dict(prep=bgr),
        'squash, no centre crop': dict(prep=squash),
        'bicubic resample': dict(prep=bicubic),
        'rounded long side': dict(prep=rounded),
        'ImageNet normalisation': dict(lut=imagenet),
        'x/256': dict(lut=by256),
    }
    for why, kw in mutants.items():
        if not _mismatch(_fast(tf, crops, 640, **kw), ref):
            bad.append(f'f3: a fast path with {why} passed the f1 '
                       f'comparison -- the test would not catch it')


def f4(bad):
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import classify_transforms
    imgsz = 64
    torch.manual_seed(0)
    model = YOLO('yolo11n-cls.yaml', verbose=False)
    model.model.transforms = classify_transforms(imgsz)
    model.predict(gs.probe_crops()[:1], verbose=False, device='cpu',
                  imgsz=imgsz)
    tf = model.predictor.transforms
    spec = gs.classify_spec(tf, imgsz)
    lut = gs.value_lut(tf)
    try:
        gap = gs.fast_path_check(model, spec, lut, imgsz, 'cpu', 3, tol=0.0)
        if gap != 0.0:
            bad.append(f'f4: fast path vs predict() gap {gap} on CPU')
    except ValueError as e:
        bad.append(f'f4: fast_path_check refused the real path: {e}')
    wrong = ((lut - torch.tensor([0.485, 0.456, 0.406])[:, None])
             / torch.tensor([0.229, 0.224, 0.225])[:, None])
    try:
        gs.fast_path_check(model, spec, wrong, imgsz, 'cpu', 3, tol=0.0)
        bad.append('f4: fast_path_check passed an ImageNet-normalised '
                   'table -- run() would judge on the wrong pixels')
    except ValueError:
        pass
    # the batch the run hands forward_probs is predict()'s batch
    crops = gs.probe_crops()[:4]
    ref = torch.stack([r.probs.data for r in model.predict(
        crops, verbose=False, device='cpu', imgsz=imgsz)])
    got = gs.forward_probs(model.predictor.model, gs.batch_input(
        [gs.prep_crop(c, spec) for c in crops], lut))
    if not torch.equal(got, ref):
        bad.append(f'f4: forward_probs != predict() probs, max diff '
                   f'{float((got - ref).abs().max()):.3g}')
    if abs(float(got.sum(1).mean()) - 1) > 1e-5:
        bad.append('f4: forward_probs is not a softmax')


def _lanes_once(bad, shards, todo, roots, order):
    """ahead=0: every frame read must belong to the lowest unfinished shard,
    which holds only if a lane -- disk-ordered or not -- is monotone in
    shard (one that is not would stall here, not pass)."""
    tag = f'f5 (disk order {order or "off"})'
    lanes = gs.DriveLanes(shards, todo, roots, {'a': 1, 'b': 3, 'c': 1},
                          ahead=0, slots=4, disk_order=order)
    seen, errs, ahead_bad = [], {}, 0
    for task in lanes.feed():
        si, iid, crops, err = gs._cut(task)
        if lanes.rank[si] > lanes.lo + lanes.ahead:
            ahead_bad += 1
        seen.append(iid)
        if err:
            errs[iid] = err
        else:
            for _, a in crops:
                if getattr(a, 'shape', None) != (64, 64, 3) or \
                        str(a.dtype) != 'uint8':
                    bad.append(f'{tag}: crop came back as {type(a)} '
                               f'{getattr(a, "shape", "")}')
        lanes.consumed(si)
    lanes.stop.set()
    want = sorted(j[0] for si in todo for j in shards[si])
    if sorted(seen) != want:
        bad.append(f'{tag}: the lanes yielded {len(seen)} jobs for '
                   f'{len(want)} '
                   f'({len(set(seen))} distinct)')
    if ahead_bad:
        bad.append(f'{tag}: {ahead_bad} frames read past the look-ahead')
    if errs.get('1_2') != 'unreadable':
        bad.append(f'{tag}: a missing frame came back as {errs.get("1_2")!r}')
    if sorted(k for k, v in errs.items() if v == 'no_root') != \
            sorted(f'3_{k}' for k in range(7)):
        bad.append(f'{tag}: frames on a drive with no root: {errs}')
    if any(lanes.left.values()):
        bad.append(f'{tag}: shards left unfinished: {lanes.left}')


def f5(bad):
    import io
    from PIL import Image
    from ultralytics.data.augment import classify_transforms
    tmp = tempfile.mkdtemp()
    roots = {'a': os.path.join(tmp, 'a'), 'b': os.path.join(tmp, 'b')}
    shards, n = [], 0
    for si in range(6):
        shard = []
        for k in range(7):
            drive = 'ab'[k % 2] if si != 3 else 'c'        # c: no root
            iid, cell = f'{si}_{k}', f'cell{si % 2}'
            d = os.path.join(roots.get(drive, tmp), cell,
                             'ground_animal_images')
            os.makedirs(d, exist_ok=True)
            if (si, k) != (1, 2):                          # one missing frame
                b = io.BytesIO()
                Image.new('RGB', (120, 90), (k * 30, si * 40, 9)).save(
                    b, 'JPEG')
                with open(os.path.join(d, f'{iid}.jpg'), 'wb') as fh:
                    fh.write(b.getvalue())
            shard.append((iid, cell, drive, [(0, 10, 10, 60, 50)]))
            n += 1
        shards.append(shard)
    spec = gs.classify_spec(classify_transforms(64), 64)
    gs._init(roots, spec)
    todo = [0, 1, 3, 4, 5]                               # 2 already on disk
    for order in ({}, {'a': 3, 'b': 100}):
        _lanes_once(bad, shards, todo, roots, order)
    gs._init({}, None)
    if gs.reader_counts([], {'x', 'bobcat'}) != {'bobcat': 8, 'x': 2} or \
            gs.reader_counts(['x=5'], {'x'}) != {'x': 5}:
        bad.append('f5: reader_counts does not apply the defaults/overrides')
    for hostile in (['x'], ['x=0'], ['=3'], ['x=y']):
        try:
            gs.reader_counts(hostile, {'x'})
            bad.append(f'f5: --readers {hostile} was accepted')
        except SystemExit:
            pass


def f6(bad):
    src = open(os.path.join(REPO, 'tools', 'detect', 'gate_store.py')).read()
    run = src[src.index('def run(args):'):src.index('def status(args):')]
    if "'--legacy-predict'" not in src or 'args.legacy_predict' not in run:
        bad.append('f6: --legacy-predict is gone')
    chk, pool = run.find('fast_path_check('), run.find('.Pool(')
    if chk < 0 or chk > pool:
        bad.append('f6: run() does not check the fast path before the pool')
    if 'run with --legacy-predict' not in run:
        bad.append('f6: the fast-path refusal does not say how to fall back')
    if 'prep_crop(' not in src[src.index('def _cut('):
                               src.index('def read_frame(')]:
        bad.append('f6: the decoders do not prepare the crops')


def main():
    bad = []
    for f in (f1, f2, f3, f4, f5, f6):
        f(bad)
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('the fast path builds predict()\'s tensor bit for bit, and its '
          'guard catches a wrong one')
    return 0


if __name__ == '__main__':
    sys.exit(main())
