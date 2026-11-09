#!/usr/bin/env python3
"""An end-to-end (8.4) engine must be decoded with NMS, and the production
one-to-many engine exactly as before.

ultralytics 8.3.165 -- the version the sweep runs -- takes a shortcut for an
output whose last dimension is 6: it filters on confidence and runs no NMS.
An 8.4 export of a yolo26 model produces exactly that output, so loading one
into the old engine.py would have stored every near-identical second box the
one-to-one head draws. engine.py now decides from the output shape which
decode applies and runs NMS itself for the end-to-end kind. Checked here,
CPU only, no TensorRT:

  * the shape rule: (BS, 4+nc, anchors) is one-to-many, (BS, K, 6) is
    end-to-end, and a header that disagrees with the graph is refused.
  * the header reader: the production engine reads as an 8.3 export with no
    end2end key; a file with no header reads as {} rather than raising.
  * e2e_nms: greedy NMS, strict at the threshold (IoU exactly 0.7 is kept,
    as torchvision and box_nms do), strict at conf, surest first, capped at
    max_det, per image, per class, non-finite rows dropped -- and equal to a
    brute-force greedy NMS on random frames.
  * the wiring: Consumer sends an end-to-end output through e2e_nms at the
    store's own ``iou`` (0.90 in the sweep) and never through
    non_max_suppression, and the one-to-many call is exactly main's. NOT
    0.7: the 0.7 is applied at count time within the gate's call
    (box_nms.py), and running it here, before the gate, loses dogs. The run
    manifest records which decode ran.

Run under the yolo env (torch + torchvision); without torch the tensor checks
report as skipped and the rest still run.

    python tools/detect/tests/adv_e2e_engine.py
"""

import ast
import json
import os
import random
import struct
import sys
import tempfile

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))

import engine  # noqa: E402

fails = []


def check(name, ok, detail=''):
    ok = bool(ok)
    extra = '' if ok or (isinstance(detail, str) and not detail) \
        else '  ' + str(detail)
    print(('ok   ' if ok else 'FAIL ') + name + extra)
    if not ok:
        fails.append(name)


def raises(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except Exception:
        return True
    return False


def write_engine(path, meta, blob=b'\0' * 16):
    js = json.dumps(meta).encode()
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<I', len(js)) + js + blob)


# ── the shape rule and the header ───────────────────────────────────────────


def shape_and_header():
    check('s1 a one-to-many output (8, 5, 33600) is not end-to-end',
          engine.is_end2end((8, 5, 33600)) is False)
    check('s2 an end-to-end output (8, 300, 6) is',
          engine.is_end2end((8, 300, 6)) is True)
    check('s3 a header saying end2end over a one-to-many graph is refused',
          raises(engine.is_end2end, (8, 5, 33600), {'end2end': True}))
    check('s4 a header saying not end2end over a (8, 300, 6) graph is refused',
          raises(engine.is_end2end, (8, 300, 6), {'end2end': False}))
    check(
        's5 an 8.3 header (no end2end key) leaves the shape to decide',
        engine.is_end2end((8, 5, 33600), {'version': '8.3.165'}) is False
        and engine.is_end2end((8, 300, 6), {'version': '8.3.165'}) is True)
    check('s6 a two-axis output is refused, not guessed',
          raises(engine.is_end2end, (8, 33600)))

    prod = os.path.join(REPO, 'data', 'engines', 'yolo26x_train30.engine')
    if os.path.exists(prod):
        m = engine.read_engine_meta(prod)
        check(
            'h1 the production engine reads as an 8.3.165 export with no '
            'end2end key (so it stays on the one-to-many path)',
            m.get('version') == '8.3.165' and 'end2end' not in m
            and m.get('batch') == engine.BS,
            {k: m.get(k)
             for k in ('version', 'batch')})
    else:
        print('skip h1 (no production engine in this checkout)')
    d = tempfile.mkdtemp(prefix='adv_e2e_')
    p = os.path.join(d, 'x.engine')
    write_engine(p, {'end2end': True, 'batch': 8})
    check('h2 an 8.4-style header is read back',
          engine.read_engine_meta(p).get('end2end') is True)
    open(p, 'wb').write(b'\xff\xff\xff\x7fgarbage')
    check('h3 a file with no header reads as {} instead of raising',
          engine.read_engine_meta(p) == {})
    open(p, 'wb').write(struct.pack('<I', 40) + b'{"end2')
    check('h4 a truncated header reads as {}',
          engine.read_engine_meta(p) == {})
    check('h5 a missing file reads as {}',
          engine.read_engine_meta(os.path.join(d, 'nope')) == {})
    return d


# ── e2e_nms ─────────────────────────────────────────────────────────────────


def ref_greedy(rows, conf, iou, max_det):
    """Brute force: rows [(x1,y1,x2,y2,c,cls)] -> kept rows, surest first."""
    rows = sorted([r for r in rows if r[4] > conf], key=lambda r: -r[4])
    kept = []
    for r in rows:
        if all(r[5] != k[5] or _iou(r, k) <= iou for k in kept):
            kept.append(r)
    return kept[:max_det]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms_checks():
    try:
        import torch
        import torchvision  # noqa: F401
    except Exception as e:
        print(f'skip n1-n9 (no torch/torchvision here: {e!r})')
        return
    f = torch.float32
    nan = float('nan')
    # image 0 of a batch of 2, K = 8 rows, surest first as the head emits
    img0 = [
        [0, 0, 10, 10, 0.90, 0],  # A
        [1, 0, 11, 10, 0.80, 0],  # IoU(A) = 0.818 -> dropped by A
        [0, 0, 10, 7, 0.75, 0],  # IoU(A) = 0.70 exactly -> kept
        [50, 50, 60, 60, 0.05, 0],  # conf == 0.05 -> dropped (strict)
        [80, 80, 90, 90, 0.06, 0],  # kept
        [0, 0, 10, 10, 0.70, 1],  # A's twin in another class -> kept
        [nan, 0, 10, 10, 0.99, 0],  # non-finite -> dropped
        [0, 0, 0, 0, 0.0, 0],  # padding row
    ]
    img1 = [[200, 200, 210, 210, 0.5, 0]] + [[0, 0, 0, 0, 0.0, 0]] * 7
    pred = torch.tensor([img0, img1], dtype=f)
    out = engine.e2e_nms(pred, 0.05, 0.70, 300)
    o0 = out[0].tolist()
    check(
        'n1 the near-identical second box is dropped and the box at IoU '
        'exactly 0.7 is kept (strict, as torchvision)',
        [r[4] for r in o0 if r[5] == 0][:2] == [
            torch.tensor(0.90, dtype=f).item(),
            torch.tensor(0.75, dtype=f).item()
        ], o0)
    check(
        'n2 conf exactly 0.05 is dropped, 0.06 kept (strict, as '
        'non_max_suppression)', not any(abs(r[0] - 50) < 1e-6 for r in o0)
        and any(abs(r[0] - 80) < 1e-6 for r in o0), o0)
    check('n3 a box is never dropped by a box of another class',
          any(r[5] == 1 for r in o0), o0)
    check('n4 a non-finite row never reaches the store',
          all(all(v == v for v in r) for r in o0), o0)
    check('n5 rows come back surest first (det_idx 0 = surest)',
          [r[4] for r in o0] == sorted([r[4] for r in o0], reverse=True), o0)
    check(
        'n6 each image is decoded on its own (image 1 keeps its one box)',
        len(out) == 2 and out[1].shape == (1, 6)
        and abs(out[1][0, 0].item() - 200) < 1e-6, out[1])
    many = torch.zeros((1, 300, 6), dtype=f)
    for k in range(300):
        many[0, k] = torch.tensor(
            [20 * k, 0, 20 * k + 10, 10, 0.99 - k * 0.003, 0])
    capped = engine.e2e_nms(many, 0.05, 0.70, 256)[0]
    check('n7 the output is capped at max_det after NMS (det_idx is uint8)',
          capped.shape[0] == 256 and abs(capped[0, 4].item() - 0.99) < 1e-6,
          capped.shape)
    empty = engine.e2e_nms(torch.zeros((1, 300, 6), dtype=f), 0.05, 0.7,
                           300)[0]
    check('n8 a frame with nothing above conf gives an empty (0, 6) result',
          tuple(empty.shape) == (0, 6), tuple(empty.shape))

    # n9: equal to brute-force greedy on random frames
    rng = random.Random(20261106)
    bad = []
    for trial in range(300):
        n = rng.randint(0, 60)
        rows = []
        for _ in range(n):
            x, y = rng.uniform(0, 300), rng.uniform(0, 300)
            w, h = rng.uniform(2, 60), rng.uniform(2, 60)
            rows.append([
                x, y, x + w, y + h,
                rng.uniform(0.0, 1.0),
                rng.choice([0, 0, 0, 1])
            ])
        # K = 64 with padding rows, sorted surest first like the head
        rows.sort(key=lambda r: -r[4])
        t = torch.tensor(rows + [[0, 0, 0, 0, 0.0, 0]] * (64 - n),
                         dtype=f).reshape(1, 64, 6)
        r32 = [[float(v) for v in r] for r in t[0].tolist()]
        # a pair within 1e-5 of the threshold could round either way in
        # float32 vs float64; such trials say nothing about the rule
        thr = (0.7, 0.9)[trial % 2]  # count-time 0.7 and the store's 0.90
        near = any(
            abs(_iou(a, b) - thr) < 1e-5 for i, a in enumerate(r32)
            for b in r32[i + 1:] if a[4] > 0.05 and b[4] > 0.05)
        if near:
            continue
        got = engine.e2e_nms(t, 0.05, thr, 40)[0].tolist()
        want = ref_greedy(r32, 0.05, thr, 40)
        if [[round(v, 4) for v in r] for r in got] != \
                [[round(v, 4) for v in r] for r in want]:
            bad.append(trial)
    check(
        'n9 e2e_nms equals brute-force greedy NMS on 300 random frames, '
        'at 0.7 and at 0.90', not bad, bad[:10])


# ── the wiring ──────────────────────────────────────────────────────────────

PROD_NMS = ('ops.non_max_suppression(pred, conf_thres=self.conf, '
            'iou_thres=self.iou, max_det=self.max_det, '
            'max_time_img=1000000000.0, agnostic=False)')
E2E_NMS = 'e2e_nms(pred[:n], self.conf, self.iou, self.max_det)'


def _method(tree, cls, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == name:
                    return fn
    return None


def wiring(tmp):
    src = open(os.path.join(REPO, 'tools', 'detect', 'engine.py')).read()
    tree = ast.parse(src)
    run = _method(tree, 'Consumer', 'run')
    body_calls, else_calls, other_nms = [], [], []
    branches = 0
    if run is not None:
        guarded = set()
        for node in ast.walk(run):
            if isinstance(node, ast.If) and ast.unparse(node.test) == \
                    'self.e2e':
                branches += 1
                for part, acc in ((node.body, body_calls), (node.orelse,
                                                            else_calls)):
                    for sub in part:
                        for c in ast.walk(sub):
                            if isinstance(c, ast.Call):
                                guarded.add(id(c))
                                acc.append(ast.unparse(c))
        other_nms = [
            ast.unparse(c)[:70] for c in ast.walk(run)
            if isinstance(c, ast.Call) and id(c) not in guarded and
            ast.unparse(c.func).endswith(('non_max_suppression', 'e2e_nms'))
        ]
    check(
        'w1 Consumer.run sends an end-to-end output through e2e_nms at the '
        "store's iou, and a one-to-many output through exactly main's "
        'non_max_suppression call', run is not None and branches == 1
        and E2E_NMS in body_calls and not any('non_max_suppression' in c
                                              for c in body_calls)
        and PROD_NMS in else_calls
        and not any('e2e_nms' in c for c in else_calls) and not other_nms,
        (branches, body_calls[:3], else_calls[:3], other_nms))

    init = _method(tree, 'Consumer', '__init__')
    init_src = ast.unparse(init) if init else ''
    defaults, params = {}, []
    if init:
        a = init.args
        params = [x.arg for x in a.args]
        for arg, dv in zip(a.args[-len(a.defaults):], a.defaults):
            defaults[arg.arg] = ast.unparse(dv)
    check(
        'w2 Consumer decides end-to-end from the engine output shape and '
        'header, keeps the old defaults (conf 0.05, iou 0.90) and has no '
        'second, end-to-end-only threshold',
        'self.e2e = is_end2end(out_shape, self.trt_meta)' in init_src
        and defaults.get('conf') == '0.05' and defaults.get('iou') == '0.9'
        and not [p for p in params if 'iou' in p and p != 'iou'],
        (defaults, params))

    sw = open(os.path.join(REPO, 'tools', 'detect', 'sweep.py')).read()
    kws = None
    for node in ast.walk(ast.parse(sw)):
        if isinstance(node, ast.Call) and \
                ast.unparse(node.func) == 'engine.Consumer':
            kws = {k.arg: ast.unparse(k.value) for k in node.keywords}
    check(
        "w3 the sweep hands the Consumer its one iou (cfg['iou']) and "
        'nothing else to decide NMS with', kws is not None
        and kws.get('iou') == "cfg['iou']"
        and not [k for k in kws if 'iou' in k and k != 'iou'], kws)
    check(
        'w4 the sweep config keeps iou at 0.90 (the store\'s NMS, which '
        'box_nms then refines to 0.7 within the gate call)',
        "cfg.setdefault('iou', 0.90)" in sw and 'e2e_iou' not in sw)

    import run_manifest as rm
    p84 = os.path.join(tmp, 'e2e.engine')
    p84o = os.path.join(tmp, 'o2m84.engine')
    p83 = os.path.join(tmp, 'o2m.engine')
    pnone = os.path.join(tmp, 'bare.engine')
    write_engine(p84, {'version': '8.4.115', 'end2end': True})
    write_engine(p84o, {'version': '8.4.115', 'end2end': False})
    write_engine(p83, {'version': '8.3.165'})
    open(pnone, 'wb').write(b'\xff\xff\xff\x7fplan')
    cfg = {'conf': 0.05, 'iou': 0.9, 'max_det': 256}
    got = {
        k: rm._inference_settings(cfg, p, REPO)
        for k, p in (('84', p84), ('84o', p84o), ('83', p83), ('none', pnone))
    }
    check(
        'w5 the run manifest records the decode that ran (end2end, '
        'one2many, or unknown when the header cannot say) beside the one '
        'iou', got['84'].get('decode') == 'end2end'
        and got['84o'].get('decode') == 'one2many'
        and got['83'].get('decode') == 'one2many'
        and got['none'].get('decode') == 'unknown'
        and all(v.get('iou') == 0.9 for v in got.values()), got)


def main():
    tmp = shape_and_header()
    nms_checks()
    wiring(tmp)
    print(f'\n{len(fails)} failed' if fails else '\nall passed')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
