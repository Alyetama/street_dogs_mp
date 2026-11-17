"""The fine-tuned DINOv3 leash classifier, as gate_store runs it.

The leash model that cleared the frozen test set is not a YOLO classifier: it
is DINOv3 ViT-L/16 with its last four blocks and a small head fine-tuned on
wider crops (the detector box grown by half its long side on every side). This
module is everything gate_store needs to judge a box with it, and nothing it
does not:

  framing   rect_pad()        the crop rectangle, exactly as the crops it was
                              trained and tested on were cut
  pixels    prep_crop()       JPEG q95 round trip -> long side capped at
                              1.15 x res (BICUBIC) -> letterbox to res x res on
                              the ImageNet mean -> uint8 HxWx3. PIL + numpy
                              only, so a decoder process never imports torch.
  model     load()            base DINOv3 from the local HF cache + the saved
                              trainable tensors; refuses on any sha mismatch
  forward   Forward(...)      uint8 batch -> ImageNet normalise on the device
                              -> bf16 autocast -> sigmoid. Always at the eval
                              batch the model was validated at (a short batch
                              is padded): under bf16 the batch SHAPE changes
                              p by up to ~1e-2, at a fixed shape every crop's
                              p is reproducible bit for bit.

Every step reproduces the training code's eval path (decode_cached ->
letterbox -> predict_arrays). gate_store proves it on every run before it
judges anything: parity.json in the model directory names training crops by
frame and box with the p the model gave them in memory at fit time, and the
run re-cuts them from the frames through these functions and must match.

A model directory holds config.json, trainable.pt, parity.json (and whatever
else the fit wrote). It is passed to `gate_store.py run --stage leash
--vit-model <dir>`; nothing here names a path.
"""
import hashlib
import io
import json
import os

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

MODEL_ID = 'facebook/dinov3-vitl16-pretrain-lvd1689m'
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FILL = tuple(int(round(v * 255)) for v in MEAN)
CACHE_SCALE = 1.15      # long side kept in RAM at training: <= 1.15 x res
JPEG_Q = 95             # the training / test crops were JPEG files at q95
FRAMINGS = {'pad050': 0.5, 'pad100': 1.0}
KIND = 'dinov3-leash'


def rect_pad(box, w, h, frac):
    """(a, b, c, d): the box (truncated as gate_store.cut_crops truncates)
    grown by int(frac x its long side) on every side, clamped to the frame.
    The framing the model's training and test crops were cut with."""
    x1, y1, x2, y2 = (int(v) for v in box)
    p = int(frac * max(x2 - x1, y2 - y1))
    return max(0, x1 - p), max(0, y1 - p), min(w, x2 + p), min(h, y2 + p)


def jpeg_roundtrip(im, q=JPEG_Q):
    """What saving the crop as a q95 JPEG file and decoding it again gives:
    every crop the model ever saw went through exactly this."""
    from PIL import Image
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=q)
    out = Image.open(io.BytesIO(buf.getvalue()))
    out.load()
    return out.convert('RGB')


def prep_crop(crop, res, roundtrip=True):
    """PIL RGB crop -> res x res x 3 uint8: the letterboxed picture the model
    sees, before normalisation. decode_cached() then letterbox() of the
    training code, minus the float conversion (done on the device)."""
    import numpy as np
    from PIL import Image
    im = jpeg_roundtrip(crop) if roundtrip else crop.convert('RGB')
    cap = int(round(CACHE_SCALE * res))
    w, h = im.size
    if max(w, h) > cap:
        s = cap / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                       Image.BICUBIC)
    # the training code held the capped crop as a uint8 array and rebuilt the
    # image from it; the round trip through numpy is a no-op for RGB
    im = Image.fromarray(np.asarray(im, dtype=np.uint8))
    w, h = im.size
    s = res / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new('RGB', (res, res), FILL)
    canvas.paste(im, ((res - nw) // 2, (res - nh) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def read_config(model_dir):
    with open(os.path.join(model_dir, 'config.json')) as fh:
        cfg = json.load(fh)
    if cfg.get('model_id') != MODEL_ID:
        raise ValueError(f'{model_dir}: model_id {cfg.get("model_id")!r}, '
                         f'not {MODEL_ID}')
    if cfg.get('framing') not in FRAMINGS:
        raise ValueError(f'{model_dir}: framing {cfg.get("framing")!r} is '
                         f'not one of {sorted(FRAMINGS)}')
    if cfg.get('smoke'):
        raise ValueError(f'{model_dir} is a smoke-test fit')
    return cfg


def eval_batch(cfg):
    """The batch the model's CV and selfcheck predictions were made at."""
    a = cfg['args']
    return int(a.get('eval_batch') or 2 * a['batch'])


def spec(model_dir):
    """Plain data for a decoder process: framing and resolution."""
    cfg = read_config(model_dir)
    return {'kind': KIND, 'framing': cfg['framing'],
            'frac': FRAMINGS[cfg['framing']], 'res': int(cfg['res'])}


def _make_net(torch, backbone, head_dropout):
    nn = torch.nn
    nreg = int(getattr(backbone.config, 'num_register_tokens', 0) or 0)
    D = backbone.config.hidden_size

    class LeashNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.nreg = nreg
            self.head = nn.Sequential(nn.LayerNorm(2 * D),
                                      nn.Dropout(head_dropout),
                                      nn.Linear(2 * D, 1))

        def train(self, mode=True):
            super().train(mode)
            self.backbone.eval()   # train mode jitters DINOv3's RoPE coords
            return self

        def forward(self, x):
            o = self.backbone(pixel_values=x)
            hs = o.last_hidden_state
            feat = torch.cat([o.pooler_output, hs[:, 1 + self.nreg:].mean(1)],
                             1)
            with torch.autocast(device_type=x.device.type, enabled=False):
                return self.head(feat.float()).squeeze(1)

    return LeashNet()


def _freeze(backbone, n):
    for p in backbone.parameters():
        p.requires_grad_(False)
    L = len(backbone.model.layer)
    for i in range(L - n, L) if n > 0 else ():
        for p in backbone.model.layer[i].parameters():
            p.requires_grad_(True)
    if n > 0:
        for p in backbone.norm.parameters():
            p.requires_grad_(True)


def _frozen_sha256(backbone):
    import numpy as np
    h = hashlib.sha256()
    for n, p in backbone.named_parameters():
        if not p.requires_grad:
            h.update(n.encode())
            h.update(np.ascontiguousarray(p.detach().cpu().float().numpy()))
    return h.hexdigest()


def load(model_dir, device='cuda'):
    """-> (net, cfg, torch.device). Refuses when the saved tensors or the
    frozen backbone in the local HF cache are not the ones the model was fit
    with."""
    import torch
    from transformers import AutoModel
    cfg = read_config(model_dir)
    dev = torch.device(device)
    sp = os.path.join(model_dir, 'trainable.pt')
    with open(sp, 'rb') as fh:
        if hashlib.sha256(fh.read()).hexdigest() != cfg['state_sha256']:
            raise ValueError(f'{sp} does not match config.json')
    backbone = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32)
    net = _make_net(torch, backbone, cfg['head_dropout'])
    _freeze(backbone, cfg['unfreeze'])
    if _frozen_sha256(backbone) != cfg['backbone']['frozen_sha256']:
        raise ValueError('the frozen DINOv3 weights in the local HF cache are '
                         'not the ones this model was trained on')
    sd = torch.load(sp, map_location='cpu', weights_only=True)
    if sorted(sd) != sorted(cfg['trainable_keys']):
        raise ValueError('trainable.pt keys differ from config.json')
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if unexpected or set(missing) & set(sd) or not all(
            k.startswith('backbone.') for k in missing):
        raise ValueError(f'state dict mismatch: unexpected {unexpected[:3]}')
    net.to(dev).eval()
    return net, cfg, dev


def value_lut():
    """(3, 256) float32: what the training code's letterbox() makes of each
    8-bit value in each channel -- (np.float32(v) / 255.0 - MEAN) / STD in
    numpy float32, evaluated by that same expression. Per element and per
    channel, so a lookup in it IS the normalisation, bit for bit."""
    import numpy as np
    v = np.arange(256, dtype=np.uint8)
    a = np.repeat(v[:, None], 3, axis=1)[None]           # 1 x 256 x 3
    mean = np.array(MEAN, np.float32)
    std = np.array(STD, np.float32)
    out = (np.asarray(a, np.float32) / 255.0 - mean) / std
    return np.ascontiguousarray(out[0].T)                 # 3 x 256


class Forward:
    """uint8 crops -> P(leashed), at a FIXED batch shape.

    __call__ queues the batch on the device and returns the probabilities as
    a device tensor (read with .cpu() later), so the next batch is gathered
    meanwhile. A batch shorter than `batch` is padded with mean-grey canvases
    and the padding's outputs dropped: every crop is scored inside a batch of
    exactly the shape the model was validated at."""

    def __init__(self, net, cfg, dev):
        import torch
        self.torch = torch
        self.net, self.dev = net, dev
        self.res = int(cfg['res'])
        self.batch = eval_batch(cfg)
        self.amp = dev.type == 'cuda'
        # the training process ran with TF32 matmuls allowed; the reload
        # check matched bit for bit with it on
        torch.backends.cuda.matmul.allow_tf32 = True
        import numpy as np
        self.lut = torch.from_numpy(value_lut()).to(dev)
        self.pad = np.empty((self.res, self.res, 3), np.uint8)
        self.pad[:] = FILL

    def __call__(self, arrs):
        import numpy as np
        torch = self.torch
        n = len(arrs)
        if not 0 < n <= self.batch:
            raise ValueError(f'{n} crops for a batch of {self.batch}')
        arrs = list(arrs) + [self.pad] * (self.batch - n)
        u8 = torch.from_numpy(np.stack(arrs))
        if self.dev.type == 'cuda':
            u8 = u8.pin_memory().to(self.dev, non_blocking=True)
        # normalise by table lookup: the table is the training code's numpy
        # arithmetic, so the input is bit-identical to letterbox()'s. Doing
        # the arithmetic on the GPU is NOT -- CUDA divides by a scalar as a
        # multiply by its reciprocal, 1 ulp off, and bf16 turns that into
        # p moving by up to 1e-2 (caught by the parity check)
        u8 = u8.long()                                     # N x H x W x 3
        x = torch.empty(u8.shape, dtype=torch.float32, device=self.dev)
        for c in range(3):
            x[..., c] = self.lut[c][u8[..., c]]
        # Channels-last memory, NCHW view -- NOT .contiguous(). The training
        # code built its batch with np.stack(letterbox(...)), whose
        # transpose(2, 0, 1) leaves the pixels in HWC order under an NCHW
        # view, and torch.from_numpy kept those strides. The patch-embedding
        # conv picks its kernel by memory layout, and under bf16 the other
        # layout moves p by up to 1e-2 (found by the parity check: same
        # values, different strides).
        x = x.permute(0, 3, 1, 2)
        with torch.inference_mode(), torch.autocast(
                device_type=self.dev.type, dtype=torch.bfloat16,
                enabled=self.amp):
            p = torch.sigmoid(self.net(x).float())
        return p[:n]


def parity_check(fwd, cut, model_dir, tol=2e-3):
    """Re-cut parity.json's crops from their frames with `cut` (gate_store's
    own worker path) and score them with `fwd`. -> (max |dp|, n). Raises
    ValueError past `tol` or on any verdict flip at 0.5."""
    import numpy as np
    with open(os.path.join(model_dir, 'parity.json')) as fh:
        par = json.load(fh)
    items = par['items']
    arrs = []
    for it in items:
        try:
            got = cut(it['frame_path'], it['box'])
        except OSError as e:
            # a drive, not the model: unreadable_parity_frames() should have
            # caught it before the model loaded
            raise OSError(f'parity frame unreadable (drive not mounted?): '
                          f'{it["frame_path"]}: {e.strerror or e}') from e
        if got is None:
            raise ValueError(f'parity crop {it["frame_path"]}: not cut')
        arrs.append(got)
    p = np.concatenate([fwd(arrs[i:i + fwd.batch]).cpu().numpy()
                        for i in range(0, len(arrs), fwd.batch)])
    want = np.array([it['p'] for it in items], np.float32)
    d = np.abs(p - want)
    flips = int(((p >= 0.5) != (want >= 0.5)).sum())
    if d.max() > tol or flips:
        k = int(d.argmax())
        raise ValueError(f'{len(items)} parity crops: max |dp| {d.max():.2e} '
                         f'(tol {tol:g}), {flips} flip(s); worst '
                         f'{items[k]["frame_path"]}: {p[k]:.5f} vs '
                         f'{want[k]:.5f}')
    return float(d.max()), len(items)


def unreadable_parity_frames(model_dir):
    """'' or which of parity.json's frames cannot be opened, by directory --
    checked before the model loads, so a dead drive reads as one."""
    try:
        with open(os.path.join(model_dir, 'parity.json')) as fh:
            items = json.load(fh)['items']
    except (OSError, ValueError, KeyError) as e:
        return f'{model_dir}/parity.json: {e}'
    bad = {}
    for it in items:
        try:
            with open(it['frame_path'], 'rb') as fh:
                fh.read(1)
        except OSError as e:
            bad.setdefault(os.path.dirname(it['frame_path']),
                           e.strerror or str(e))
    if not bad:
        return ''
    return (f'{sum(1 for _ in bad)} folder(s) of the parity check\'s frames '
            f'cannot be read -- a drive is not mounted:\n  '
            + '\n  '.join(f'{d}: {e}' for d, e in sorted(bad.items())))
