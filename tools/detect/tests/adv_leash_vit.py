#!/usr/bin/env python3
"""The ViT leash path must judge the boxes the way the model was trained.

`gate_store.py run --stage leash --vit-model DIR` scores every gate-approved
dog box with the fine-tuned DINOv3 leash classifier (leash_vit.py) instead of
the YOLO model. Its numbers only mean anything if the pixels and the batch it
hands the network are the training code's, bit for bit: the crops were cut by
the leash90 framings script (rect('pad050'), saved as q95 JPEG files) and
scored by the fine-tune script's eval path (decode_cached -> letterbox ->
predict_arrays). Those scripts live outside the repo, so their logic is
re-typed below as the reference, never imported.

The same change added two guards that hold for EVERY run, YOLO too: a drive
that went away (bobcat dropped off USB mid-run, 2026-11-16) used to come back
as 'unreadable' frames, silently skipped, in shards written as complete.

What has to hold:

  v1  rect_pad == the framings script's rect('pad050') (and pad100), over
      edge, float, negative and degenerate boxes
  v2  read_config / spec / eval_batch: the framing, res and batch come from
      the model directory; a smoke fit, another backbone or framing refused
  v3  cut_vit judges exactly the boxes cut_crops judges (MIN_SIDE), and
      cut_crops_rects is cut_crops' rule, rectangle for rectangle
  v4  prep_crop == decode_cached(q95 file) + letterbox, as uint8 before
      normalisation: a crop past the 1.15 x res cap, at and one past the cap,
      thin, tiny, edge-clamped; and end to end through cut_vit on a frame
  v5  vit_cut_one (RGB and greyscale frame files) and _cut's ViT dispatch
      give the reference crop; a box under MIN_SIDE gives none
  v6  value_lut is the training normalisation in numpy float32, bit for bit
  v7  Forward hands the net a fixed-shape batch (a short one padded with the
      letterboxed mean canvas), channels-last memory under an NCHW view,
      values bit-equal to np.stack(letterbox(...)); returns only n outputs
  v8  parity_check passes a match, raises on a drift past tol, on a verdict
      flip inside tol, and on a crop that was not cut
  v9  unreadable_roots flags a missing / unlistable / non-directory root and
      a drive with no root, passes a good one, ignores drives not read
  v10 refuse_unreadable_shard: at the limit written, one past refused, error
      kinds summed
  v11 check_protocol refuses pad050@448 into a full@640 record and back, and
      a crop change at the same imgsz
  v12 run(): --vit-model refused on the gate stage before anything is read;
      --max-unreadable defaults to MAX_UNREADABLE; the guards sit before the
      pool / the shard write

Every guard is then broken on purpose in a copy of its source (a mutant) and
the check that holds it must fail.

Runs in the yolo env (torch on CPU, no transformers, no model, no drives):

    python tools/detect/tests/adv_leash_vit.py
"""

import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT = os.path.dirname(HERE)
sys.path.insert(0, DETECT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import gate_store  # noqa: E402
import leash_vit  # noqa: E402

GS, LV = gate_store, leash_vit          # swapped for a mutant while it runs
TMP = tempfile.mkdtemp(prefix='adv_leash_vit_')
FAILS = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name +
          (f' -- {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


# ── the reference: the training code's logic, re-typed ─────────────────────
REF_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
REF_STD = np.array([0.229, 0.224, 0.225], np.float32)
REF_FILL = tuple(int(round(v * 255)) for v in REF_MEAN)
REF_CACHE_SCALE = 1.15
REF_Q = 95


def ref_rect(framing, box, w, h):
    """cut_framings.rect: (rect, clamped)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    long_ = max(x2 - x1, y2 - y1)
    if framing in ('pad050', 'pad100'):
        p = int((0.5 if framing == 'pad050' else 1.0) * long_)
        a, b, c, d = x1 - p, y1 - p, x2 + p, y2 + p
    elif framing == 'ctx3x':
        cx, cy, half = (x1 + x2) / 2.0, (y1 + y2) / 2.0, 1.5 * long_
        a, b = int(round(cx - half)), int(round(cy - half))
        c, d = int(round(cx + half)), int(round(cy + half))
    else:
        raise ValueError(framing)
    raw = (a, b, c, d)
    a, b, c, d = max(0, a), max(0, b), min(w, c), min(h, d)
    return (a, b, c, d), (a, b, c, d) != raw


def ref_decode(raw):
    """cut_framings.decode: a frame file's bytes -> RGB."""
    im = Image.open(io.BytesIO(raw))
    im.load()
    if im.mode != 'RGB':
        im = im.convert('RGB')
    return im


def ref_crop_file(crop):
    """cut_framings.cut_all: the crop saved as a q95 JPEG file."""
    buf = io.BytesIO()
    crop.save(buf, 'JPEG', quality=REF_Q)
    return buf.getvalue()


def ref_decode_cached(b, res):
    """finetune.decode_cached."""
    im = Image.open(io.BytesIO(b))
    im.load()
    im = im.convert('RGB')
    cap = int(round(REF_CACHE_SCALE * res))
    w, h = im.size
    if max(w, h) > cap:
        s = cap / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                       Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8).copy()


def ref_canvas(im, res):
    """finetune.letterbox up to (not including) the normalisation."""
    w, h = im.size
    s = res / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new('RGB', (res, res), REF_FILL)
    canvas.paste(im, ((res - nw) // 2, (res - nh) // 2))
    return canvas


def ref_letterbox(im, res):
    """finetune.letterbox."""
    a = (np.asarray(ref_canvas(im, res), np.float32) / 255.0
         - REF_MEAN) / REF_STD
    return a.transpose(2, 0, 1)


def ref_from_crop(crop, res):
    """crop -> q95 file -> decode_cached -> letterbox canvas, uint8."""
    arr = ref_decode_cached(ref_crop_file(crop), res)
    return np.asarray(ref_canvas(Image.fromarray(arr), res), np.uint8)


def ref_from_frame(frame, box, res, framing='pad050'):
    rc, _ = ref_rect(framing, box, *frame.size)
    return ref_from_crop(frame.crop(rc), res)


# ── fixtures ───────────────────────────────────────────────────────────────
def _frame(w, h, seed, mode='RGB'):
    """Structured + noisy, so a wrong resize or JPEG step shows."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 255 // max(1, w - 1), yy * 255 // max(1, h - 1),
                     (xx + yy) % 256], -1)
    a = np.clip(base + rng.randint(-40, 41, (h, w, 3)), 0, 255)
    im = Image.fromarray(a.astype(np.uint8), 'RGB')
    return im if mode == 'RGB' else im.convert(mode)


FRAME = _frame(1600, 1000, 1)
W, H = FRAME.size


def _vspec(res):
    return {'kind': LV.KIND, 'framing': 'pad050', 'frac': 0.5, 'res': res}


def _boxes():
    rng = np.random.RandomState(3)
    out = [(0, 100, 100, 500, 400),        # pad050 crop 800 wide: past cap
           (1, 0, 0, 300, 200),            # clamped top-left
           (2, 1400, 850, 1590, 990),      # clamped bottom-right
           (3, 700, 500, 720, 560),        # small
           (4, 10.7, 20.2, 50.9, 80.99),   # float coordinates
           (5, 800, 400, 803, 403),        # 3x3: cut_crops keeps it
           (6, 0, 0, 1, 1),                # cut_crops: 5x5 < MIN_SIDE
           (7, 1599, 999, 1600, 1000),     # same at the far corner
           (8, 1597, 500, 1600, 504)]      # 7 wide after its padding
    # at least 1 px a side: a box narrower than that truncates to zero
    # width, and cut_vit cannot cut it at all (see the report)
    for k in range(9, 60):
        x1, y1 = rng.uniform(-20, W), rng.uniform(-20, H)
        s = rng.choice([2, 5, 20, 120, 400])
        out.append((k, x1, y1, x1 + rng.uniform(1.0, s),
                    y1 + rng.uniform(1.0, s)))
    return out


# ── the checks: each returns [] or what is wrong ───────────────────────────
def c_rect():
    bad = []
    if (LV.FRAMINGS.get('pad050'), LV.FRAMINGS.get('pad100')) != (0.5, 1.0):
        bad.append(f'FRAMINGS {LV.FRAMINGS}')
    boxes = [b[1:] for b in _boxes()] + [(5, 5, 5, 5), (-3, -3, 10, 10),
                                         (-0.9, 3.9, 7.99, 8.01),
                                         (1700, 1100, 1800, 1200)]
    n_clamped = 0
    for box in boxes:
        for framing, frac in (('pad050', 0.5), ('pad100', 1.0)):
            want, clamped = ref_rect(framing, box, W, H)
            n_clamped += clamped
            got = LV.rect_pad(box, W, H, frac)
            if tuple(got) != want:
                bad.append(f'{framing} {box}: {got} != {want}')
    if n_clamped < 4:
        bad.append(f'only {n_clamped} clamped rects -- the edges are untested')
    return bad


def c_config():
    bad = []
    d = tempfile.mkdtemp(dir=TMP)
    p = os.path.join(d, 'config.json')

    def put(**kw):
        cfg = {'model_id': LV.MODEL_ID, 'framing': 'pad050', 'res': 448,
               'args': {'batch': 16}}
        cfg.update(kw)
        json.dump(cfg, open(p, 'w'))

    put()
    sp = LV.spec(d)
    if sp != {'kind': LV.KIND, 'framing': 'pad050', 'frac': 0.5, 'res': 448}:
        bad.append(f'spec {sp}')
    if LV.eval_batch(LV.read_config(d)) != 32:
        bad.append('eval_batch is not 2 x batch without args.eval_batch')
    put(args={'batch': 16, 'eval_batch': 24})
    if LV.eval_batch(LV.read_config(d)) != 24:
        bad.append('args.eval_batch ignored')
    for kw in ({'smoke': True}, {'framing': 'ctx3x'},
               {'model_id': 'facebook/dinov2-large'}):
        put(**kw)
        try:
            LV.read_config(d)
            bad.append(f'read_config accepted {kw}')
        except ValueError:
            pass
    if GS.VIT_KIND != LV.KIND:
        bad.append(f'gate_store.VIT_KIND {GS.VIT_KIND!r} != leash_vit.KIND '
                   f'{LV.KIND!r} -- _cut would never dispatch to cut_vit')
    return bad


def c_same_boxes():
    bad = []
    im = Image.new('RGB', (W, H), (90, 120, 30))
    # boxes whose pad050 rect has pixels: one that cut_crops drops for lying
    # off the frame has none, and a cut_vit that kept it would throw rather
    # than show which boxes it judged
    boxes = [b for b in _boxes() if (lambda r: r[2] > r[0] and r[3] > r[1])(
        ref_rect('pad050', b[1:], W, H)[0])]
    kept = [di for di, _ in GS.cut_crops(im, boxes)]
    got = [di for di, _ in GS.cut_vit(im, boxes, _vspec(16))]
    if got != kept:
        bad.append(f'cut_vit judged {sorted(set(got) ^ set(kept))} '
                   f'differently from cut_crops')
    dropped = {b[0] for b in boxes} - set(kept)
    if not {6, 7, 8} <= dropped:
        bad.append(f'the MIN_SIDE fixtures were kept by cut_crops: {dropped}')
    sizes = [(di, c.size) for di, c in GS.cut_crops(im, boxes)]
    rects = [(di, (r[2] - r[0], r[3] - r[1]))
             for di, r in GS.cut_crops_rects(W, H, boxes)]
    if rects != sizes:
        bad.append('cut_crops_rects is not cut_crops\' rule')
    for di, a in GS.cut_vit(im, boxes[:3], _vspec(16)):
        if a.shape != (16, 16, 3) or a.dtype != np.uint8:
            bad.append(f'box {di}: cut_vit gave {a.shape} {a.dtype}')
    return bad


def c_prep():
    bad = []
    res = 448
    cap = int(round(1.15 * res))
    if (LV.CACHE_SCALE, LV.JPEG_Q, tuple(LV.FILL)) != (1.15, 95, REF_FILL):
        bad.append(f'constants {LV.CACHE_SCALE} {LV.JPEG_Q} {LV.FILL}')
    if (tuple(LV.MEAN), tuple(LV.STD)) != ((0.485, 0.456, 0.406),
                                           (0.229, 0.224, 0.225)):
        bad.append(f'MEAN/STD {LV.MEAN} {LV.STD}')
    edge, clamped = ref_rect('pad050', (0, 0, 300, 200), W, H)
    crops = {'900x700 (past the cap)': FRAME.crop((100, 100, 1000, 800)),
             f'{cap}x300 (at the cap)': FRAME.crop((0, 0, cap, 300)),
             f'{cap + 1}x200 (one past)': FRAME.crop((5, 5, cap + 6, 205)),
             '40x300 thin': FRAME.crop((600, 100, 640, 400)),
             '3x3 tiny': FRAME.crop((50, 50, 53, 53)),
             '448x448': FRAME.crop((300, 300, 748, 748)),
             'edge-clamped': FRAME.crop(edge)}
    if not clamped:
        bad.append('the edge fixture is not clamped')
    for name, c in crops.items():
        got = LV.prep_crop(c, res)
        want = ref_from_crop(c, res)
        if got.shape != (res, res, 3) or got.dtype != np.uint8:
            bad.append(f'{name}: {got.shape} {got.dtype}')
        elif not np.array_equal(got, want):
            bad.append(f'{name}: {int((got != want).sum())} values differ '
                       f'from decode_cached + letterbox')
    # end to end: frame -> cut_vit -> the crop the model was trained on
    boxes = _boxes()[:6]
    for di, a in GS.cut_vit(FRAME, boxes, _vspec(res)):
        want = ref_from_frame(FRAME, boxes[di][1:], res)
        if not np.array_equal(a, want):
            bad.append(f'cut_vit box {di}: {int((a != want).sum())} values '
                       f'differ from the framings script + decode_cached')
    return bad


def c_vit_cut_one():
    bad = []
    res = 448
    vs = _vspec(res)
    box = (200, 150, 520, 400)
    for mode in ('RGB', 'L'):
        p = os.path.join(TMP, f'frame_{mode}.jpg')
        _frame(900, 600, 5, mode).save(p, 'JPEG', quality=90)
        raw = open(p, 'rb').read()
        want = ref_from_frame(ref_decode(raw), box, res)
        got = GS.vit_cut_one(vs, p, box)
        if got is None or not np.array_equal(got, want):
            bad.append(f'vit_cut_one on a {mode} frame differs from the '
                       f'reference')
        if GS.vit_cut_one(vs, p, (0, 0, 1, 1)) is not None:
            bad.append(f'vit_cut_one cut a box under MIN_SIDE ({mode})')
    # the decoder process: _cut with the ViT spec installed
    raw = open(os.path.join(TMP, 'frame_RGB.jpg'), 'rb').read()
    try:
        GS._init({}, vs)
        si, iid, out, err = GS._cut((4, 'x', [(0,) + box, (1, 0, 0, 1, 1)],
                                     raw, ''))
    except Exception as e:   # noqa: BLE001
        return bad + [f'_cut with the ViT spec threw {type(e).__name__}: {e}'
                      f' -- it did not hand the boxes to cut_vit']
    finally:
        GS._init({}, None)
    want = ref_from_frame(ref_decode(raw), box, res)
    if err or [di for di, _ in out] != [0] or not \
            np.array_equal(out[0][1], want):
        bad.append(f'_cut with the ViT spec: err {err!r}, '
                   f'{[di for di, _ in out]} -- not cut_vit\'s crops')
    return bad


def c_lut():
    bad = []
    lut = LV.value_lut()
    v = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, 1)[None]
    canvas = Image.fromarray(v, 'RGB')
    want = np.ascontiguousarray(((np.asarray(canvas, np.float32) / 255.0
                                  - REF_MEAN) / REF_STD)[0].T)
    if lut.shape != (3, 256) or lut.dtype != np.float32:
        return [f'value_lut is {lut.shape} {lut.dtype}']
    lut = np.ascontiguousarray(lut)
    n = int((lut.view(np.uint32) != want.view(np.uint32)).sum())
    if n:
        bad.append(f'{n} of 768 entries differ from the numpy float32 '
                   f'normalisation')
    return bad


class _Rec(torch.nn.Module):
    """Records what it is handed; a logit per crop from its pixels."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):
        self.seen.append((tuple(x.shape), tuple(x.stride()), x.dtype,
                          x.is_contiguous(), x.detach().clone()))
        return x.float().mean((1, 2, 3)) * 7.0


def c_forward():
    bad = []
    R = 40
    cfg = {'res': R, 'args': {'batch': 16}}        # eval batch 32
    net = _Rec()
    fwd = LV.Forward(net, cfg, torch.device('cpu'))
    if fwd.batch != 32:
        bad.append(f'Forward.batch {fwd.batch}, want the eval batch 32')
    B = fwd.batch
    rng = np.random.RandomState(11)
    cap = int(round(1.15 * R))
    cached = [rng.randint(0, 256, (rng.randint(3, cap), rng.randint(3, cap),
                                   3)).astype(np.uint8) for _ in range(B)]
    fill = Image.new('RGB', (R, R), REF_FILL)
    for n in (5, 1, B):
        net.seen.clear()
        arrs = [np.asarray(ref_canvas(Image.fromarray(a), R), np.uint8)
                for a in cached[:n]]
        # predict_arrays' batch: np.stack(letterbox(...)), from_numpy
        ref = torch.from_numpy(np.stack(
            [ref_letterbox(Image.fromarray(a), R) for a in cached[:n]]
            + [ref_letterbox(fill, R)] * (B - n)))
        out = fwd(arrs)
        if len(net.seen) != 1:
            bad.append(f'n={n}: the net ran {len(net.seen)} times')
            continue
        shape, stride, dtype, contig, x = net.seen[0]
        if shape != (B, 3, R, R):
            bad.append(f'n={n}: the net got {shape}, not the fixed batch '
                       f'{(B, 3, R, R)}')
            continue
        if stride != tuple(ref.stride()) or contig:
            bad.append(f'n={n}: strides {stride} (contiguous {contig}), the '
                       f'training batch had {tuple(ref.stride())}')
        if dtype != torch.float32 or not torch.equal(x, ref):
            bad.append(f'n={n}: the input is not the training batch bit for '
                       f'bit ({dtype})')
        if tuple(out.shape) != (n,):
            bad.append(f'n={n}: {tuple(out.shape)} outputs')
        else:
            want = torch.sigmoid(ref.float().mean((1, 2, 3)) * 7.0)[:n]
            if not torch.allclose(out, want, atol=1e-6, rtol=0):
                bad.append(f'n={n}: outputs are not the crops\' own, in order')
    for n in (0, B + 1):
        try:
            fwd([np.zeros((R, R, 3), np.uint8)] * n)
            bad.append(f'Forward took {n} crops for a batch of {B}')
        except ValueError:
            pass
    return bad


class _FakeFwd:
    def __init__(self, batch, ps):
        self.batch, self.ps, self.calls = batch, ps, []

    def __call__(self, arrs):
        self.calls.append(len(arrs))
        return torch.tensor([self.ps[int(a[0, 0, 0])] for a in arrs],
                            dtype=torch.float32)


def c_parity():
    bad = []
    d = tempfile.mkdtemp(dir=TMP)
    want = [0.91, 0.12, 0.4995, 0.73, 0.05]
    items = [{'frame_path': f'f{i}', 'box': [0, 0, 9, 9], 'p': p}
             for i, p in enumerate(want)]
    json.dump({'items': items}, open(os.path.join(d, 'parity.json'), 'w'))

    def cut(fp, box):
        return np.full((2, 2, 3), int(fp[1:]), np.uint8)

    def run(ps, cutter=cut):
        f = _FakeFwd(2, ps)
        return LV.parity_check(f, cutter, d), f

    try:
        (gap, n), f = run([p + 1e-4 for p in want[:2]] + [0.4996]
                          + want[3:])
        if n != 5 or not 0 < gap < 2e-3:
            bad.append(f'a match returned ({gap}, {n})')
        if max(f.calls) > 2 or sum(f.calls) != 5:
            bad.append(f'batches {f.calls} for a batch of 2')
    except ValueError as e:
        bad.append(f'a match within tol was refused: {e}')
    cases = {'a drift of 3e-3': want[:3] + [want[3] + 3e-3] + want[4:],
             'a flip inside tol (0.4995 -> 0.5005)':
                 want[:2] + [0.5005] + want[3:]}
    for name, ps in cases.items():
        try:
            run(ps)
            bad.append(f'{name} passed the parity check')
        except ValueError as e:
            if 'max |dp|' not in str(e):
                bad.append(f'{name}: refused for another reason: {e}')
    try:
        run(want, lambda fp, b: None if fp == 'f3' else cut(fp, b))
        bad.append('a parity crop that was not cut passed')
    except ValueError as e:
        if 'f3: not cut' not in str(e):
            bad.append(f'an uncut crop refused only by accident: {e}')
    return bad


def c_unreadable_roots():
    bad = []
    base = tempfile.mkdtemp(dir=TMP)
    good = os.path.join(base, 'good')
    os.makedirs(os.path.join(good, 'cell0'))
    afile = os.path.join(base, 'afile')
    open(afile, 'w').write('x')
    locked = os.path.join(base, 'locked')
    os.makedirs(locked)
    os.chmod(locked, 0)
    roots = {'good': good, 'gone': os.path.join(base, 'gone'),
             'file': afile, 'locked': locked}
    try:
        if GS.unreadable_roots(roots, {'good'}):
            bad.append('a good root was refused')
        if GS.unreadable_roots(roots, set()):
            bad.append('no drives read, yet refused')
        cases = [('gone', 'a missing root'), ('file', 'a file for a root'),
                 ('nodrive', 'a drive with no root')]
        if os.geteuid() != 0:             # root lists a 000 directory anyway
            cases.append(('locked', 'an unlistable root'))
        for d, what in cases:
            msg = GS.unreadable_roots(roots, {'good', d})
            if not msg or f'{d} (' not in msg or 'good (' in msg:
                bad.append(f'{what}: {msg!r}')
    finally:
        os.chmod(locked, 0o700)
    return bad


def c_unreadable_shard():
    bad = []
    if GS.MAX_UNREADABLE != 20:
        bad.append(f'MAX_UNREADABLE {GS.MAX_UNREADABLE}')
    ok = [({}, 20), ({'unreadable': 20}, 20), ({'unreadable': 3,
                                                'no_root': 2}, 5)]
    no = [({'unreadable': 21}, 20), ({'unreadable': 15, 'no_root': 6}, 20),
          ({'unreadable': 1}, 0), ({'no_root': 20000}, 20)]
    for kinds, lim in ok:
        if GS.refuse_unreadable_shard(7, kinds, 20000, lim):
            bad.append(f'{kinds} at limit {lim} was refused')
    for kinds, lim in no:
        msg = GS.refuse_unreadable_shard(7, kinds, 20000, lim)
        if not msg:
            bad.append(f'{kinds} at limit {lim} was written')
        elif 'shard 7' not in msg or '--max-unreadable' not in msg:
            bad.append(f'the refusal does not name the shard / flag: {msg!r}')
    return bad


def c_protocol():
    bad = []
    full, vit = {'crop': 'full', 'imgsz': 640}, {'crop': 'pad050',
                                                 'imgsz': 448}
    for pinned, want in ((full, vit), (vit, full),
                         (full, {'crop': 'pad050', 'imgsz': 640})):
        d = tempfile.mkdtemp(dir=TMP)
        GS.pin_protocol(pinned, d)
        if not GS.check_protocol(want, d, shards={0}):
            bad.append(f'{want} let into a {pinned} record')
        if GS.check_protocol(pinned, d, shards={0}):
            bad.append(f'{pinned} refused its own record')
        if GS.check_protocol(want, d, shards=set()):
            bad.append(f'{want} refused with no shards written')
    d = tempfile.mkdtemp(dir=TMP)
    GS.pin_protocol(vit, d)
    if GS.protocol_tag(d) != 'pad050@448':
        bad.append(f'protocol_tag {GS.protocol_tag(d)!r}')
    return bad


def c_run():
    bad = []
    old = os.environ.get('GATE_ROOT')
    os.environ['GATE_ROOT'] = tempfile.mkdtemp(dir=TMP)   # never data/
    mdir = tempfile.mkdtemp(dir=TMP)                      # empty: unread
    real_run = GS.run
    try:
        for stage, want in (('gate', '--stage leash'),
                            ('leash', 'no plan yet')):
            try:
                GS.main(['run', '--stage', stage, '--vit-model', mdir])
                bad.append(f'{stage}: run() went ahead')
            except SystemExit as e:
                if want not in str(e.code):
                    bad.append(f'--vit-model on {stage}: {e.code!r}')
        GS.run = lambda a: a                      # what main() parsed
        a = GS.main(['run', '--stage', 'leash', '--vit-model', mdir])
        if a.max_unreadable != GS.MAX_UNREADABLE or a.vit_model != mdir:
            bad.append(f'parsed --max-unreadable {a.max_unreadable}, '
                       f'--vit-model {a.vit_model}')
    finally:
        GS.run = real_run
        if old is None:
            os.environ.pop('GATE_ROOT', None)
        else:
            os.environ['GATE_ROOT'] = old
        GS.use('gate')
    src = open(GS.__file__).read()
    body = src[src.index('def run(args):'):src.index('def status(args):')]
    at = {k: body.find(k) for k in (
        "STAGE != 'leash'", "'work.parquet')", 'unreadable_roots(',
        'pin_protocol(dict(want', 'parity_check(', '.Pool(',
        'refuse_unreadable_shard(', 'pq.write_table(', 'args.max_unreadable',
        "want = {'crop': vspec['framing']", 'k[err] = k.get(err, 0) + 1',
        'short_done_shards(', 'unreadable_parity_frames(',
        'leash_vit.load(', 'bad_drives.get(si)', 'if err in READ_ERRORS:')}
    missing = [k for k, v in at.items() if v < 0]
    if missing:
        return bad + [f'run() lost {missing}']
    for a, b in (("STAGE != 'leash'", "'work.parquet')"),
                 ('unreadable_roots(', 'pin_protocol(dict(want'),
                 ('unreadable_roots(', 'leash_vit.load('),
                 ('unreadable_parity_frames(', 'leash_vit.load('),
                 ('short_done_shards(', 'leash_vit.load('),
                 ('pin_protocol(dict(want', '.Pool('),
                 ('parity_check(', '.Pool('),
                 ('refuse_unreadable_shard(', 'pq.write_table('),
                 ('bad_drives.get(si)', 'pq.write_table(')):
        if at[a] > at[b]:
            bad.append(f'run(): {a} comes after {b}')
    return bad


def c_tiny_box():
    """A box under 1 px wide: cut_crops keeps it (PAD_PX), the ViT framing
    has nothing to cut -- skipped, not a crash of the whole run."""
    im = Image.new('RGB', (200, 100), (90, 90, 90))
    vspec = {'kind': GS.VIT_KIND, 'framing': 'pad050', 'frac': 0.5,
             'res': 64}
    boxes = [(0, 72.29, 50.31, 72.85, 51.10), (1, 20, 20, 60, 70)]
    try:
        got = GS.cut_vit(im, boxes, vspec)
    except Exception as e:   # noqa: BLE001
        return [f'cut_vit threw {type(e).__name__}: {e}']
    if [d for d, _ in got] != [1]:
        return [f'kept {[d for d, _ in got]}, want [1]']
    return []


def c_parity_frames():
    bad = []
    base = tempfile.mkdtemp(dir=TMP)
    okf = os.path.join(base, 'ok', 'a.jpg')
    os.makedirs(os.path.dirname(okf))
    open(okf, 'wb').write(b'x')
    gone = os.path.join(base, 'gone', 'b.jpg')
    for items, want_bad in (([okf], False), ([okf, gone], True)):
        mdir = tempfile.mkdtemp(dir=TMP)
        json.dump({'items': [{'frame_path': f, 'box': [0, 0, 1, 1], 'p': .5}
                             for f in items]},
                  open(os.path.join(mdir, 'parity.json'), 'w'))
        msg = LV.unreadable_parity_frames(mdir)
        if bool(msg) != want_bad or (want_bad and 'gone' not in msg):
            bad.append(f'{items}: {msg!r}')
    # an unreadable frame inside parity_check is a drive, not a mismatch
    mdir = tempfile.mkdtemp(dir=TMP)
    json.dump({'items': [{'frame_path': gone, 'box': [0, 0, 1, 1], 'p': .5}]},
              open(os.path.join(mdir, 'parity.json'), 'w'))

    def cut(f, b):
        open(f, 'rb')

    try:
        LV.parity_check(None, cut, mdir)
        bad.append('parity_check passed an unreadable frame')
    except OSError as e:
        if 'parity frame unreadable' not in str(e):
            bad.append(f'unreadable frame reported as {e}')
    except Exception as e:   # noqa: BLE001
        bad.append(f'unreadable frame raised {type(e).__name__}: {e}')
    return bad


def c_short_shards():
    """short_done_shards: a finished shard short of its planned images is
    found, a whole one and one inside the limit are not."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    bad = []
    base = tempfile.mkdtemp(dir=TMP)
    old = os.environ.get('GATE_ROOT')
    os.environ['GATE_ROOT'] = base
    try:
        GS.use('leash')
        os.makedirs(GS.OUT_DIR, exist_ok=True)
        shards = [[(f'{s}_{i}', 'c', 'd', []) for i in range(50)]
                  for s in range(3)]
        for si, n in ((0, 50), (1, 45), (2, 49)):
            pq.write_table(pa.table({'image_id': [f'{si}_{i}' for i in
                                                  range(n) for _ in (0, 1)]}),
                           os.path.join(GS.OUT_DIR, GS.shard_name(si)))
        got = GS.short_done_shards({0, 1, 2}, shards, 2)
        if [g[0] for g in got] != [1] or got[0][1:] != (45, 50):
            bad.append(f'limit 2: {got}')
        got = GS.short_done_shards({0, 1, 2}, shards, 0)
        if [g[0] for g in got] != [1, 2]:
            bad.append(f'limit 0: {got}')
    finally:
        if old is None:
            os.environ.pop('GATE_ROOT', None)
        else:
            os.environ['GATE_ROOT'] = old
        GS.use('gate')
    return bad


def c_mount_hint():
    bad = []
    h = GS.mount_hint(os.path.join(TMP, 'x', 'grid_runs'))
    if 'sudo umount -l' not in h or 'sudo mount' not in h:
        bad.append(f'hint: {h!r}')
    if GS.mount_point(TMP) is None:
        bad.append('no mount point found for a temp dir')
    return bad


CHECKS = [('v1 rect_pad == the framings script\'s rect', c_rect),
          ('v2 config / spec / eval batch', c_config),
          ('v3 cut_vit judges exactly cut_crops\' boxes', c_same_boxes),
          ('v4 prep_crop == decode_cached + letterbox', c_prep),
          ('v5 vit_cut_one and _cut\'s ViT dispatch', c_vit_cut_one),
          ('v6 value_lut == numpy float32 normalisation', c_lut),
          ('v7 Forward: fixed batch, channels-last, n outputs', c_forward),
          ('v8 parity_check refuses drift, flips, uncut crops', c_parity),
          ('v9 unreadable_roots', c_unreadable_roots),
          ('v10 refuse_unreadable_shard thresholds', c_unreadable_shard),
          ('v11 check_protocol keeps ViT and YOLO records apart', c_protocol),
          ('v12 run(): stage refusal, flag, guard order', c_run),
          ('v13 cut_vit skips a box under 1 px', c_tiny_box),
          ('v14 parity frames: a dead drive is a drive', c_parity_frames),
          ('v15 short finished shards are found', c_short_shards),
          ('v16 the remount hint names a mount point', c_mount_hint)]


# ── mutants: each guard broken in a copy of its source ─────────────────────
SRC = {'leash_vit': open(leash_vit.__file__).read(),
       'gate_store': open(gate_store.__file__).read()}


class AnchorGone(Exception):
    pass


def _load(name, old, new):
    src = SRC[name]
    if src.count(old) != 1:
        raise AnchorGone(f'{name}: {old!r} found {src.count(old)} times')
    # mirror tools/detect/ so the copy's REPO is a temp dir, never the repo
    d = os.path.join(tempfile.mkdtemp(dir=TMP), 'tools', 'detect')
    os.makedirs(d)
    path = os.path.join(d, name + '.py')
    with open(path, 'w') as fh:
        fh.write(src.replace(old, new))
    spec = importlib.util.spec_from_file_location(name + '_mutant', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mutant(name, old, new, fn):
    def run():
        global GS, LV
        mod = _load(name, old, new)
        saved = GS, LV, sys.modules['leash_vit']
        try:
            if name == 'leash_vit':
                LV = sys.modules['leash_vit'] = mod   # cut_vit imports it
            else:
                GS = mod
            return fn()
        finally:
            GS, LV, sys.modules['leash_vit'] = saved
    return run


MUTANTS = [
    ('.contiguous() reinstated after the permute',
     mutant('leash_vit', 'x = x.permute(0, 3, 1, 2)\n',
            'x = x.permute(0, 3, 1, 2).contiguous()\n', c_forward)),
    ('a short batch not padded to the eval batch',
     mutant('leash_vit', 'arrs = list(arrs) + [self.pad] * (self.batch - n)',
            'arrs = list(arrs)', c_forward)),
    ("the padding's outputs returned",
     mutant('leash_vit', '        return p[:n]', '        return p',
            c_forward)),
    ('normalisation by the reciprocal of 255 (the CUDA way)',
     mutant('leash_vit', 'out = (np.asarray(a, np.float32) / 255.0 - mean)',
            'out = (np.asarray(a, np.float32) * (1 / 255.0) - mean)',
            c_lut)),
    ('the MIN_SIDE skip removed from cut_vit',
     mutant('gate_store', '        if di not in keep:\n            continue\n',
            '', c_same_boxes)),
    ('cut_crops_rects without MIN_SIDE',
     mutant('gate_store', '            continue\n        out.append((di, (a, '
            'b, c, d)))', '            pass\n        out.append((di, (a, b, '
            'c, d)))', c_same_boxes)),
    ('rect_pad with the gate\'s PAD_PX added',
     mutant('leash_vit', '    p = int(frac * max(x2 - x1, y2 - y1))\n',
            '    p = int(frac * max(x2 - x1, y2 - y1)) + 4\n', c_rect)),
    ('rect_pad rounding instead of truncating the box',
     mutant('leash_vit', 'x1, y1, x2, y2 = (int(v) for v in box)',
            'x1, y1, x2, y2 = (int(round(v)) for v in box)', c_rect)),
    ('no q95 JPEG round trip',
     mutant('leash_vit', 'def prep_crop(crop, res, roundtrip=True):',
            'def prep_crop(crop, res, roundtrip=False):', c_prep)),
    ('the 1.15 x res cap resized BILINEAR',
     mutant('leash_vit', 'round(h * s))),\n                       '
            'Image.BICUBIC)', 'round(h * s))),\n                       '
            'Image.BILINEAR)', c_prep)),
    ('the cap at res, not 1.15 x res',
     mutant('leash_vit', 'cap = int(round(CACHE_SCALE * res))', 'cap = res',
            c_prep)),
    ('the letterbox pasted at the corner',
     mutant('leash_vit',
            'canvas.paste(im, ((res - nw) // 2, (res - nh) // 2))',
            'canvas.paste(im, (0, 0))', c_prep)),
    ('_cut not dispatching to cut_vit',
     mutant('gate_store',
            "    if _PREP is not None and _PREP.get('kind') == VIT_KIND:",
            '    if False:', c_vit_cut_one)),
    ('parity flips inside tol ignored',
     mutant('leash_vit', 'if d.max() > tol or flips:', 'if d.max() > tol:',
            c_parity)),
    ('an uncut parity crop skipped',
     mutant('leash_vit', "            raise ValueError(f'parity crop "
            "{it[\"frame_path\"]}: not cut')", '            continue',
            c_parity)),
    ('unreadable_roots swallowing the error',
     mutant('gate_store',
            "            bad.append(f'{d} ({root}): {e.strerror or e}\\n    '\n"
            "                       + mount_hint(root))",
            '            pass', c_unreadable_roots)),
    ('unreadable_roots checking existence, not a listing',
     mutant('gate_store', '            with os.scandir(root) as it:\n'
            '                next(it, None)',
            '            if not os.path.exists(root):\n'
            '                raise FileNotFoundError(2, "gone")',
            c_unreadable_roots)),
    ('the unreadable-shard threshold disabled',
     mutant('gate_store', "    if n <= limit:\n        return ''",
            "    if True:\n        return ''", c_unreadable_shard)),
    ('the unreadable-shard threshold off by one',
     mutant('gate_store', '    if n <= limit:', '    if n < limit:',
            c_unreadable_shard)),
    ('only the first error kind counted',
     mutant('gate_store', '    n = sum(kinds.values())',
            '    n = max(kinds.values(), default=0)', c_unreadable_shard)),
    ('check_protocol comparing imgsz only',
     mutant('gate_store',
            '    if {k: pinned.get(k) for k in want} == want:',
            "    if pinned.get('imgsz') == want.get('imgsz'):", c_protocol)),
    ('--vit-model allowed on the gate stage',
     mutant('gate_store', "    if vit_dir and STAGE != 'leash':",
            '    if False:', c_run)),
    ('the roots preflight after the protocol pin',
     mutant('gate_store', '    _no = unreadable_roots(roots, need)\n',
            "    _no = ''\n", c_run)),
    ('the tiny-box skip removed from cut_vit',
     mutant('gate_store', '        if c <= a or d <= b:\n',
            '        if False:\n', c_tiny_box)),
    ('the parity frames not checked before the model',
     mutant('leash_vit', "    if not bad:\n        return ''\n"
            "    return (f'{sum(1 for _ in bad)}",
            "    if True:\n        return ''\n"
            "    return (f'{sum(1 for _ in bad)}", c_parity_frames)),
    ('a short finished shard let through',
     mutant('gate_store', '        if have < len(shards[si]) - limit:',
            '        if have < 0:', c_short_shards)),
    ('the write-time drive re-probe removed',
     mutant('gate_store', '            if not _no and bad_drives.get(si):',
            '            if False:', c_run)),
]


def main():
    try:
        for name, fn in CHECKS:
            try:
                got = fn()
            except Exception as e:   # noqa: BLE001
                got = [f'threw {type(e).__name__}: {e}']
            check(name, not got, '; '.join(got[:4]))
        for name, fn in MUTANTS:
            try:
                got = fn()
            except AnchorGone as e:
                check(f'mutant: {name}', False,
                      f'its source text is gone ({e}) -- update the mutant')
                continue
            except Exception as e:   # noqa: BLE001
                got = [f'threw {type(e).__name__}']
            check(f'mutant: {name}', bool(got),
                  '' if got else 'the check still passes -- it proves nothing')
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    if FAILS:
        print(f'{len(FAILS)} failed')
        return 1
    print(f'the ViT leash path cuts, letterboxes and batches as it was '
          f'trained, and the drive guards hold ({len(CHECKS)} checks, '
          f'{len(MUTANTS)} mutants caught)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
