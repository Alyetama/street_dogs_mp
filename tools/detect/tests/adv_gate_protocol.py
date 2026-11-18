#!/usr/bin/env python3
"""The store's verdicts must be made the way the models were measured.

gate_store used to squash every crop to 224x224 and predict at imgsz=224,
while dogbin_008 and leash_v2_001 were trained, evaluated and accepted at 640.
Measured on dogbin_008: v5-val dog recall@0.5 went from 0.936 at the eval
protocol to 0.330 in production; leash_v2_001 acceptance balanced accuracy
went from 0.895 to 0.718. Every number the project quoted described a model
the store never ran.

What has to hold:

  g1  crops are cut at full resolution: the padded box, never resized
  g2  the predict size is the one the checkpoint was trained at, from the
      checkpoint itself, with no constant anywhere in the runner; the default
      batch shrinks with it (128 crops at 640 OOMs the 16 GB card)
  g3  a stage directory holds ONE protocol: legacy shards (no protocol file)
      and shards from another imgsz are refused, not appended to
  g4  archive/restore carry the protocol file with the verdicts it describes
  g5  the eval scripts default to the same checkpoint lookup
  g7  a leash work list older than the gate verdicts it was cut from is
      refused by the runner, and the archive CLI moves it (--replan) on its
      own; the archive CLI refuses under a live runner
  g8  the FN-audit pool goes stale when the store's protocol changes, even
      with the same model name and shard count
  g9  the legacy refusal fires before the model or work list loads (the
      dashboard reads a spawn as started after ~2.5 s)
  g10 $GATE_ROOT moves the stage directories (a not-yet-live store is judged
      without touching data/gate); an explicit repo ignores it
  g6  (yolo env + weights) a PIL crop handed to predict() scores exactly what
      the same pixels score when read from a lossless file -- the eval path

Run under mp14 for g1-g5; g6 runs only where ultralytics imports and
TRAINING_ROOT (or the default training root) holds the promoted gate weights.
"""

import ast
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))

import gate_store as gs  # noqa: E402


def g1(bad):
    from PIL import Image
    im = Image.new('RGB', (1000, 600), (10, 20, 30))
    got = dict(gs.cut_crops(im, [
        (0, 100, 100, 200, 150),     # 100x50 box
        (1, 0, 0, 40, 40),           # clipped at the top-left edge
        (2, 990, 590, 1000, 600),    # near the corner, still >= MIN_SIDE
        (3, 500, 300, 501, 301),     # 1px box: padding makes it 9x9 -- kept
    ]))
    pad = int(gs.PAD_FRAC * 100) + gs.PAD_PX
    want0 = (100 + 2 * pad, 50 + 2 * pad)
    if got[0].size != want0:
        bad.append(f'g1: 100x50 box cut to {got[0].size}, want {want0} -- '
                   f'the crop is being resized')
    if any(c.size == (224, 224) for c in got.values()):
        bad.append('g1: a crop came out 224x224 -- the legacy squash is back')
    p1 = int(gs.PAD_FRAC * 40) + gs.PAD_PX
    if got[1].size != (40 + p1, 40 + p1):
        bad.append(f'g1: edge box {got[1].size}, want {(40 + p1,) * 2}')
    if 3 not in got:
        bad.append('g1: a 1px box padded past MIN_SIDE was dropped')
    tiny = gs.cut_crops(Image.new('RGB', (6, 6)), [(0, 0, 0, 6, 6)])
    if tiny:
        bad.append('g1: a crop under MIN_SIDE was kept')
    im.close()
    # the crops must survive the frame being closed (crop() is lazy)
    try:
        got[0].tobytes()
    except Exception as e:
        bad.append(f'g1: crop unusable after the frame closed: {e}')
    # one cut, one place: the audit page imports these, so must the runner
    src = open(os.path.join(REPO, 'tools', 'detect', 'gate_store.py')).read()
    body = src[src.index('def cut_crops('):src.index('def _cut(')]
    if '.resize(' in body:
        bad.append('g1: cut_crops resizes')


def g2(bad):
    class M:
        def __init__(self, ckpt=None, overrides=None):
            self.ckpt, self.overrides = ckpt, overrides
    cases = [
        (M({'train_args': {'imgsz': 640}}, {'imgsz': 224}), 640),
        (M({}, {'imgsz': 512}), 512),
        (M(None, None), None),
        (M({'train_args': {'imgsz': [480, 640]}}), 640),
        (M({'train_args': {'imgsz': 'x'}}, {}), None),
    ]
    for m, want in cases:
        if gs.model_imgsz(m) != want:
            bad.append(f'g2: model_imgsz({m.ckpt}, {m.overrides}) = '
                       f'{gs.model_imgsz(m)}, want {want}')
    # pixels per batch held to 128 crops at 224: measured, 128 at 640 OOMs
    for size, want in ((224, 128), (640, 16), (1024, 6), (4096, 1)):
        if gs.auto_batch(size) != want:
            bad.append(f'g2: auto_batch({size}) = {gs.auto_batch(size)}, '
                       f'want {want}')
    tree = ast.parse(open(os.path.join(REPO, 'tools', 'detect',
                                       'gate_store.py')).read())
    run = next(f for f in ast.walk(tree)
               if isinstance(f, ast.FunctionDef) and f.name == 'run')
    for n in ast.walk(run):
        if isinstance(n, ast.keyword) and n.arg == 'imgsz' and \
                isinstance(n.value, ast.Constant):
            bad.append(f'g2: run() predicts at a constant imgsz={n.value.value}')


def g3(bad):
    d = tempfile.mkdtemp()
    want = {'crop': 'full', 'imgsz': 640}
    if gs.check_protocol(want, d, shards=set()):
        bad.append('g3: an empty directory was refused')
    if not gs.check_protocol(want, d, shards={0, 1}):
        bad.append('g3: legacy shards (no protocol file) were not refused -- '
                   'a resumed job would append 640 verdicts to 224 ones')
    gs.pin_protocol(want, d)
    if gs.check_protocol(want, d, shards={0}):
        bad.append('g3: the same protocol was refused')
    if not gs.check_protocol({'crop': 'full', 'imgsz': 320}, d, shards={0}):
        bad.append('g3: a different imgsz was allowed into the same record')
    if gs.check_protocol({'crop': 'full', 'imgsz': 320}, d, shards=set()):
        bad.append('g3: a different protocol was refused with no shards '
                   'written -- nothing would be mixed')
    msg = gs.check_protocol(want, tempfile.mkdtemp(), shards={0})
    if 'archive' not in msg:
        bad.append(f'g3: the refusal does not say how to proceed: {msg!r}')


def g4(bad):
    repo = tempfile.mkdtemp()
    d = gs._stage_dir('gate', repo)
    os.makedirs(d)
    for f in ('gate-00000.parquet', gs.PROTOCOL_FILE, 'plan.json'):
        open(os.path.join(d, f), 'w').write('{}')
    note = gs.archive_run('gate', {'m': 1}, 'next', repo=repo, now=1e9)
    if gs.PROTOCOL_FILE not in note['files']:
        bad.append('g4: archive left protocol.json behind -- the next job '
                   'would be pinned to the old protocol')
    if not os.path.exists(os.path.join(d, 'plan.json')):
        bad.append('g4: a plain archive moved the plan')
    gs.restore_run('gate', note['name'], repo=repo, now=1e9 + 5)
    if not os.path.exists(os.path.join(d, gs.PROTOCOL_FILE)):
        bad.append('g4: restore did not bring protocol.json back')


def g5(bad):
    for f in ('eval_dogbin.py', 'eval_leash.py'):
        src = open(os.path.join(REPO, 'tools', 'detect', f)).read()
        if "'--imgsz', type=int, default=640" in src:
            bad.append(f'g5: {f} still defaults to imgsz 640 instead of the '
                       f'checkpoint')
        if 'model_imgsz' not in src:
            bad.append(f'g5: {f} does not use gate_store.model_imgsz')


def g7(bad):
    import time
    repo = tempfile.mkdtemp()
    gd, ld = gs._stage_dir('gate', repo), gs._stage_dir('leash', repo)
    os.makedirs(gd)
    os.makedirs(ld)
    open(os.path.join(ld, 'work.parquet'), 'w').write('x')
    open(os.path.join(gd, 'gate-00000.parquet'), 'w').write('x')
    old = time.time() - 3600
    os.utime(os.path.join(gd, 'gate-00000.parquet'), (old - 10, old - 10))
    os.utime(os.path.join(ld, 'work.parquet'), (old, old))
    if gs.stale_plan('leash', repo):
        bad.append('g7: a leash list built after the gate was called stale')
    if gs.stale_plan('gate', repo):
        bad.append('g7: the gate (no upstream) was called stale')
    open(os.path.join(gd, 'gate-00001.parquet'), 'w').write('x')   # new gate
    if not gs.stale_plan('leash', repo):
        bad.append('g7: a leash list older than the gate verdicts was not '
                   'called stale -- a leash run would skip every new dog')
    src = open(os.path.join(REPO, 'tools', 'detect', 'gate_store.py')).read()
    run_body = src[src.index('def run(args):'):src.index('def status(args):')]
    if 'stale_plan()' not in run_body:
        bad.append('g7: run() does not refuse a stale work list')
    ar = src[src.index('def archive(args):'):src.index('def restore(args):')]
    if 'stale_plan()' not in ar or 'replan = True' not in ar:
        bad.append('g7: the archive CLI does not replan a stale list')
    if 'live_runner()' not in ar:
        bad.append('g7: the archive CLI does not check for a live run')
    # live_runner: only a live gate_store pid counts
    d = tempfile.mkdtemp()
    json.dump({'pid': 1}, open(os.path.join(d, 'progress.json'), 'w'))
    if gs.live_runner(d):
        bad.append('g7: pid 1 (not gate_store) was taken for a live runner')
    json.dump({'pid': 2 ** 22 + 7}, open(os.path.join(d, 'progress.json'), 'w'))
    if gs.live_runner(d):
        bad.append('g7: a dead pid was taken for a live runner')
    import subprocess
    pr = subprocess.Popen([sys.executable, '-c',
                           'import time; time.sleep(30)  # gate_store'])
    try:
        time.sleep(0.3)
        json.dump({'pid': pr.pid}, open(os.path.join(d, 'progress.json'), 'w'))
        if gs.live_runner(d) != pr.pid:
            bad.append('g7: a live gate_store process was not detected')
    finally:
        pr.kill()


def g8(bad):
    if gs.protocol_tag(tempfile.mkdtemp()) != '':
        bad.append('g8: an empty store has a protocol')
    d = tempfile.mkdtemp()
    open(os.path.join(d, os.path.basename(d) + '-00000.parquet'), 'w')
    if gs.protocol_tag(d) != 'legacy':
        bad.append('g8: shards without protocol.json not read as legacy')
    gs.pin_protocol({'crop': 'full', 'imgsz': 640}, d)
    if gs.protocol_tag(d) != 'full@640':
        bad.append(f'g8: protocol_tag = {gs.protocol_tag(d)!r}')
    src = open(os.path.join(REPO, 'tools', 'detect', 'fn_audit.py')).read()
    body = src[src.index('def pool_info('):src.index('def band_totals(')]
    if 'protocol' not in body:
        bad.append('g8: the audit pool staleness ignores the protocol')
    try:
        import fn_audit as fa
    except Exception as e:
        print(f'SKIP g8 live check: {e}')
        return
    repo = tempfile.mkdtemp()
    fa.REPO = repo
    P = fa.paths('gate', gen=1)
    os.makedirs(P['out'])
    os.makedirs(P['store'])
    open(os.path.join(P['store'], 'gate-00000.parquet'), 'w')
    json.dump({'shards': 1}, open(os.path.join(P['out'], 'pool.json'), 'w'))
    # the gen-1 store is the LIVE one here (its plan names its gen-1 detect
    # store), so the gen-1 pool can be behind it at all -- a frozen
    # generation is never stale
    det = os.path.join(repo, 'det1')
    os.makedirs(det)
    json.dump({'gen': 1}, open(os.path.join(det, 'status.json'), 'w'))
    json.dump({'gen': 1, 'detect_root': det},
              open(os.path.join(P['store'], 'plan.json'), 'w'))
    if fa.pool_info('gate', gen=1)['stale']:
        bad.append('g8: a legacy pool over a legacy store was called stale')
    gs.pin_protocol({'crop': 'full', 'imgsz': 640}, P['store'])
    if not fa.pool_info('gate', gen=1)['stale']:
        bad.append('g8: a pool cut from 224 verdicts is not stale after a '
                   '640 re-run with the same shard count')


def g9(bad):
    src = open(os.path.join(REPO, 'tools', 'detect', 'gate_store.py')).read()
    run_body = src[src.index('def run(args):'):src.index('def status(args):')]
    early = run_body.find("protocol_tag(OUT_DIR) == 'legacy'")
    if early < 0:
        bad.append('g9: run() has no early legacy refusal')
    elif early > run_body.find('YOLO(weights)') or \
            early > run_body.find('to_pydict()'):
        bad.append('g9: the legacy refusal comes after the model / work list '
                   'load -- the dashboard reports the run as started')


def g10(bad):
    old = os.environ.get('GATE_ROOT')
    try:
        os.environ['GATE_ROOT'] = '/x/gen2'
        if gs.stages_root() != '/x/gen2':
            bad.append('g10: GATE_ROOT ignored')
        if gs._stage_dir('leash', '/r') != '/r/data/leash':
            bad.append('g10: an explicit repo followed GATE_ROOT')
        os.environ['GATE_ROOT'] = 'data/g2'
        if gs.stages_root() != os.path.join(gs.REPO, 'data/g2'):
            bad.append('g10: a relative GATE_ROOT is not anchored at the repo')
        gs.use('leash')
        if gs.OUT_DIR != os.path.join(gs.REPO, 'data/g2', 'leash'):
            bad.append(f'g10: use() put the leash stage at {gs.OUT_DIR}')
        del os.environ['GATE_ROOT']
        gs.use('gate')
        if gs.OUT_DIR != os.path.join(gs.REPO, 'data', 'gate'):
            bad.append(f'g10: default gate dir is {gs.OUT_DIR}')
    finally:
        if old is None:
            os.environ.pop('GATE_ROOT', None)
        else:
            os.environ['GATE_ROOT'] = old
        gs.use('gate')


def g6(bad):
    try:
        from ultralytics import YOLO
        from PIL import Image
    except ImportError:
        print('SKIP g6: no ultralytics in this env')
        return
    try:
        weights, _ = gs.gate_weights()
    except SystemExit as e:
        print(f'SKIP g6: {str(e).splitlines()[0]}')
        return
    model = YOLO(weights)
    imgsz = gs.model_imgsz(model)
    im = Image.new('RGB', (900, 500))
    px = im.load()
    for x in range(900):
        for y in range(0, 500, 3):
            px[x, y] = (x % 256, (x * y) % 256, y % 256)
    crops = [c for _, c in gs.cut_crops(im, [(0, 100, 50, 400, 300),
                                             (1, 600, 100, 640, 160)])]
    tmp = tempfile.mkdtemp()
    paths = []
    for i, c in enumerate(crops):
        p = os.path.join(tmp, f'{i}.png')
        c.save(p)
        paths.append(p)
    a = [r.probs.data.tolist() for r in model.predict(
        crops, imgsz=imgsz, verbose=False, device='cpu')]
    b = [r.probs.data.tolist() for r in model.predict(
        paths, imgsz=imgsz, verbose=False, device='cpu')]
    for i, (x, y) in enumerate(zip(a, b)):
        if max(abs(u - v) for u, v in zip(x, y)) > 1e-4:
            bad.append(f'g6: crop {i}: PIL {x} vs file {y} -- the runner and '
                       f'the eval path see different pixels')


def main():
    bad = []
    for g in (g1, g2, g3, g4, g5, g7, g8, g9, g10, g6):
        g(bad)
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('the store judges full-resolution crops at the trained imgsz, one '
          'protocol per record')
    return 0


if __name__ == '__main__':
    sys.exit(main())
