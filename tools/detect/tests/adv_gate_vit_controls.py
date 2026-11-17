#!/usr/bin/env python3
"""The gate panel drives a leash stage pinned to the ViT model.

`gate_store.py run --stage leash --vit-model DIR` judges the leash stage with
the fine-tuned DINOv3 model and pins protocol.json to it (kind
dinov3-leash, model, model_dir). The panel's controls only ever spawned the
YOLO run: Start/Resume on such a stage launched the wrong interpreter with no
--vit-model, and Re-run archived the ViT verdicts and judged everything again
with the promoted YOLO model.

  c1 Start on a ViT-pinned stage runs vit_python with --vit-model <the pinned
     dir>, and strips LD_LIBRARY_PATH from the child (the cuDNN clash)
  c2 Start on a YOLO stage is unchanged: dogbin_python, no --vit-model, the
     environment passed through
  c3 a ViT stage with no vit_python, or with its pinned dir gone, is refused
     with a message saying so -- and nothing is spawned
  c4 a promoted "vit_model" (relative to training_root) starts a stage with
     no verdicts, and names the promoted model; a YOLO record still resumes
     with YOLO; a ViT record resumes with its PINNED model whichever ViT
     model is promoted (the state after the switch-over)
  c5 Re-run with a promoted vit_model archives the YOLO record and starts the
     ViT model; with no vit_python it refuses BEFORE moving anything
  c6 Re-run on a ViT-pinned stage while the promoted model is YOLO is refused
     (naming both models and how to promote), and nothing is archived; with
     another ViT model promoted it goes ahead with that one
  c7 the /api/gate payload: can_run follows vit_python for a ViT stage and
     dogbin_python for a YOLO one; rerun_blocked carries the refusal; a
     live run stays stoppable (can_run, no run_error) whatever the start
     configuration says -- Stop needs no interpreter or model dir
  c8 the page hides Re-run when rerun_blocked is set, shows the reason, and
     enables Re-run on can_rerun
  c9 mutants: each guard removed is caught

Never spawns a run (subprocess is stubbed on the module) and never reads or
writes data/: every path is a temp directory.

    DASH_PY=<copy> checks an edited copy instead of the live file.

    python tools/detect/tests/adv_gate_vit_controls.py      (mp14 env)
"""

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DASH_PY = os.environ.get('DASH_PY') or os.path.join(
    REPO, 'tools', 'dashboard', 'dashboard.py')
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
os.environ.pop('GATE_ROOT', None)

import gate_store  # noqa: E402,F401  (the real one, before any temp REPO)

TMP = tempfile.mkdtemp(prefix='adv_gate_vit_')
FAILS = []
VIT_PY, YOLO_PY = '/x/vit/bin/python', '/x/dogbin/bin/python'
YOLO_RUN = 'leash_yolo_run'
FAKE_LD = '/elsewhere/envs/other/lib/cudnn'


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name +
          (f' -- {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


class FakePopen:
    calls = []

    def __init__(self, argv, **kw):
        FakePopen.calls.append({'argv': list(argv),
                                'env': dict(kw.get('env') or {})})
        self.pid = 999999

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired('fake', timeout)

    def poll(self):
        return None


def shard(path, labels, model):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({
        'image_id': [str(k) for k in range(len(labels))],
        'det_idx': list(range(len(labels))), 'label': labels,
        'p_dog': [0.9] * len(labels), 'model': [model] * len(labels)}), path)


def world(dash, protocol=None, verdicts_by=None, vit_model=None):
    """A temp repo with a planned leash stage; returns its paths."""
    root = tempfile.mkdtemp(dir=TMP)
    repo = os.path.join(root, 'repo')
    troot = os.path.join(root, 'training')
    ldir = os.path.join(repo, 'data', 'leash')
    os.makedirs(ldir)
    os.makedirs(os.path.join(repo, 'data', 'gate'))
    os.makedirs(os.path.join(repo, 'tools', 'detect'))
    open(os.path.join(repo, 'tools', 'detect', 'gate_store.py'), 'w').close()
    dirs = {}
    for name in ('leash_vit_A', 'leash_vit_B'):
        dirs[name] = os.path.join(troot, 'leash-models', name)
        os.makedirs(dirs[name])
    with open(os.path.join(ldir, 'plan.json'), 'w') as fh:
        json.dump({'rows': 6, 'images': 6}, fh)
    with open(os.path.join(ldir, 'work.parquet'), 'wb') as fh:
        fh.write(b'work')
    if verdicts_by:
        shard(os.path.join(ldir, 'leash-00000.parquet'),
              ['leashed', 'unleashed'], verdicts_by)
    if protocol == 'vit':
        pin = {'crop': 'pad050', 'imgsz': 448, 'model': 'leash_vit_A',
               'kind': 'dinov3-leash', 'model_dir': dirs['leash_vit_A']}
    elif protocol == 'yolo':
        pin = {'crop': 'full', 'imgsz': 640, 'model': YOLO_RUN}
    else:
        pin = None
    if pin:
        with open(os.path.join(ldir, 'protocol.json'), 'w') as fh:
            json.dump(pin, fh)
    best = {'run': YOLO_RUN, 'weights': 'runs/l/weights/best.pt'}
    if vit_model:
        best['vit_model'] = vit_model
    dash.REPO = repo
    dash.best_models = lambda: {'projects': {
        'dog-bin': {'best': {'run': 'gate_run', 'weights': 'g.pt'}},
        'leash-models': {'best': best}}}
    dash.training_root = lambda: troot
    dash.gate_pids = lambda *a, **k: []
    dash.gate_planning = lambda: False
    dash.sweep_pids = lambda: []
    dash.gate_upstream = lambda *a, **k: None
    dash._gate_stale_plan = lambda *a, **k: False
    dash.side_gate_root = lambda *a, **k: None
    dash.subprocess = types.SimpleNamespace(
        Popen=FakePopen, TimeoutExpired=subprocess.TimeoutExpired,
        DEVNULL=subprocess.DEVNULL)
    dash.DOGBIN_PYTHON, dash.VIT_PYTHON = YOLO_PY, VIT_PY
    dash._GATE.clear()
    dash._GATE_FILES.clear()
    del dash._SPAWNED[:]
    FakePopen.calls.clear()
    return {'repo': repo, 'ldir': ldir, 'troot': troot, 'dirs': dirs,
            'script': os.path.join(repo, 'tools', 'detect', 'gate_store.py')}


def listing(d):
    out = []
    for base, _, files in os.walk(d):
        out += [os.path.relpath(os.path.join(base, f), d) for f in files]
    return sorted(x for x in out if not x.endswith('_run.log'))


def payload(dash, stage='leash'):
    got = {}

    class Fake:
        path = f'/api/gate?stage={stage}'
        ADMIN_GET = ()
        session = None

        def _sane_path(self):
            return True

        def _gate(self):
            return False

        def _admin_only(self, *a, **k):
            return False

        def _json(self, obj, code=200):
            got.update(obj)

    dash.BoardHandler.do_GET(Fake())
    return got


def brief(doc):
    return {k: doc.get(k) for k in ('error', 'can_run', 'run_error',
                                    'can_rerun', 'rerun_error',
                                    'rerun_blocked', 'best_model')}


def run_checks(dash, tag=''):
    fails0 = len(FAILS)
    os.environ['LD_LIBRARY_PATH'] = FAKE_LD
    try:
        _run_checks(dash, tag)
    finally:
        os.environ.pop('LD_LIBRARY_PATH', None)
    return len(FAILS) == fails0


def _run_checks(dash, tag):
    # ── c1 ViT-pinned start ─────────────────────────────────────────────────
    w = world(dash, protocol='vit', verdicts_by='leash_vit_A')
    r = dash.gate_control('start', 'leash')
    call = FakePopen.calls[-1] if FakePopen.calls else {}
    check(f'{tag}c1 a ViT-pinned stage resumes with vit_python and '
          f'--vit-model <pinned dir>',
          r.get('ok') and call.get('argv') == [
              VIT_PY, w['script'], 'run', '--stage', 'leash',
              '--vit-model', w['dirs']['leash_vit_A']], (r, call.get('argv')))
    check(f'{tag}c1b the ViT child runs without LD_LIBRARY_PATH',
          call and 'LD_LIBRARY_PATH' not in call['env']
          and call['env'].get('TRAINING_ROOT') == w['troot'],
          call.get('env', {}).get('LD_LIBRARY_PATH'))

    # ── c2 YOLO start unchanged ─────────────────────────────────────────────
    w = world(dash, protocol='yolo', verdicts_by=YOLO_RUN)
    r = dash.gate_control('start', 'leash')
    call = FakePopen.calls[-1] if FakePopen.calls else {}
    check(f'{tag}c2 a YOLO stage starts as before',
          r.get('ok') and call.get('argv') == [
              YOLO_PY, w['script'], 'run', '--stage', 'leash']
          and call['env'].get('LD_LIBRARY_PATH') == FAKE_LD,
          (r, call.get('argv')))

    # ── c3 refusals ─────────────────────────────────────────────────────────
    w = world(dash, protocol='vit', verdicts_by='leash_vit_A')
    dash.VIT_PYTHON = ''
    r = dash.gate_control('start', 'leash')
    check(f'{tag}c3 no vit_python: refused, saying which key',
          r.get('ok') is False and 'vit_python' in (r.get('msg') or '')
          and not FakePopen.calls, r)
    w = world(dash, protocol='vit', verdicts_by='leash_vit_A')
    shutil.rmtree(w['dirs']['leash_vit_A'])
    r = dash.gate_control('start', 'leash')
    check(f'{tag}c3b the pinned model dir gone: refused, never YOLO',
          r.get('ok') is False and 'not on disk' in (r.get('msg') or '')
          and not FakePopen.calls, r)
    pin = dash.stage_vit('leash')
    check(f'{tag}c3c stage_vit reports the missing dir as an error',
          bool(pin) and pin.get('error') and pin['model'] == 'leash_vit_A',
          pin)

    # ── c4 promoted ViT model ───────────────────────────────────────────────
    w = world(dash, vit_model='leash-models/leash_vit_B')
    best = dash.stage_best('leash')
    bdir = dash.stage_best_vit('leash')
    r = dash.gate_control('start', 'leash')
    call = FakePopen.calls[-1] if FakePopen.calls else {}
    check(f'{tag}c4 stage_best / stage_best_vit read the promoted vit_model',
          best == 'leash_vit_B' and bdir == w['dirs']['leash_vit_B'],
          (best, bdir))
    check(f'{tag}c4b a stage with no verdicts starts with it',
          r.get('ok') and call.get('argv') == [
              VIT_PY, w['script'], 'run', '--stage', 'leash',
              '--vit-model', w['dirs']['leash_vit_B']]
          and 'LD_LIBRARY_PATH' not in call['env'], (r, call.get('argv')))
    check(f'{tag}c4c the gate stage never takes a ViT model',
          dash.stage_best_vit('gate') is None
          and dash.stage_best('gate') == 'gate_run')
    w = world(dash, protocol='yolo', verdicts_by=YOLO_RUN,
              vit_model='leash-models/leash_vit_B')
    r = dash.gate_control('start', 'leash')
    call = FakePopen.calls[-1] if FakePopen.calls else {}
    check(f'{tag}c4d a YOLO record still resumes with YOLO',
          r.get('ok') and call.get('argv') == [
              YOLO_PY, w['script'], 'run', '--stage', 'leash'],
          (r, call.get('argv')))

    # the state after the switch-over: a ViT record AND a ViT model promoted
    # -- the pin wins on Start/Resume, whichever ViT model is promoted
    for promoted in ('leash_vit_A', 'leash_vit_B'):
        w = world(dash, protocol='vit', verdicts_by='leash_vit_A',
                  vit_model=f'leash-models/{promoted}')
        r = dash.gate_control('start', 'leash')
        call = FakePopen.calls[-1] if FakePopen.calls else {}
        check(f'{tag}c4e a ViT record resumes with its pinned model while '
              f'{promoted} is promoted',
              r.get('ok') and call.get('argv') == [
                  VIT_PY, w['script'], 'run', '--stage', 'leash',
                  '--vit-model', w['dirs']['leash_vit_A']],
              (r, call.get('argv')))
        doc = payload(dash)
        check(f'{tag}c4f ...and the panel offers it ({promoted} promoted)',
              doc.get('can_run') is True and doc.get('rerun_blocked') is None
              and doc.get('run_model') == 'leash_vit_A', brief(doc))

    # ── c5 re-run with the promoted ViT model ───────────────────────────────
    cdir = os.path.join(TMP, 'leash_vit_C')      # an absolute vit_model
    os.makedirs(cdir, exist_ok=True)
    w = world(dash, protocol='yolo', verdicts_by=YOLO_RUN, vit_model=cdir)
    r = dash.gate_control('rerun', 'leash')
    call = FakePopen.calls[-1] if FakePopen.calls else {}
    arch = os.path.join(w['ldir'], 'archive')
    names = os.listdir(arch) if os.path.isdir(arch) else []
    note = {}
    if names:
        with open(os.path.join(arch, names[0], 'ARCHIVED.json')) as fh:
            note = json.load(fh)
    check(f'{tag}c5 re-run archives the YOLO record and judges with the '
          f'promoted ViT model',
          r.get('ok') and len(names) == 1
          and note.get('replaced_by') == 'leash_vit_C'
          and 'protocol.json' in note.get('files', [])
          and call.get('argv') == [
              VIT_PY, w['script'], 'run', '--stage', 'leash',
              '--vit-model', cdir],
          (r, note, call.get('argv')))
    shutil.rmtree(cdir)
    w = world(dash, protocol='yolo', verdicts_by=YOLO_RUN,
              vit_model='leash-models/leash_vit_B')
    dash.VIT_PYTHON = ''
    before = listing(w['ldir'])
    r = dash.gate_control('rerun', 'leash')
    check(f'{tag}c5b with no vit_python the re-run refuses before moving '
          f'anything',
          r.get('ok') is False and 'vit_python' in (r.get('msg') or '')
          and listing(w['ldir']) == before and not FakePopen.calls, r)

    # ── c6 re-run refused on a ViT record while YOLO is promoted ────────────
    w = world(dash, protocol='vit', verdicts_by='leash_vit_A')
    before = listing(w['ldir'])
    r = dash.gate_control('rerun', 'leash')
    msg = r.get('msg') or ''
    check(f'{tag}c6 re-run refused on a ViT record while YOLO is promoted',
          r.get('ok') is False and 'leash_vit_A' in msg and YOLO_RUN in msg
          and 'vit_model' in msg and 'worse' in msg, r)
    check(f'{tag}c6b and nothing is archived or spawned',
          listing(w['ldir']) == before and not FakePopen.calls,
          listing(w['ldir']))
    w = world(dash, protocol='vit', verdicts_by='leash_vit_A',
              vit_model='leash-models/leash_vit_B')
    r = dash.gate_control('rerun', 'leash')
    call = FakePopen.calls[-1] if FakePopen.calls else {}
    check(f'{tag}c6c with another ViT model promoted, the re-run uses it',
          r.get('ok') and call.get('argv', [])[-2:] == [
              '--vit-model', w['dirs']['leash_vit_B']]
          and os.path.isdir(os.path.join(w['ldir'], 'archive')),
          (r, call.get('argv')))

    # ── c7 the payload ──────────────────────────────────────────────────────
    w = world(dash, protocol='vit', verdicts_by='leash_vit_A')
    dash.DOGBIN_PYTHON = ''
    doc = payload(dash)
    check(f'{tag}c7 a ViT stage can run on vit_python alone',
          not doc.get('error') and doc.get('can_run') is True, brief(doc))
    check(f'{tag}c7b rerun_blocked explains the refusal',
          'leash_vit_A' in (doc.get('rerun_blocked') or '')
          and doc.get('vit_model') == 'leash_vit_A', doc.get('rerun_blocked'))
    dash.DOGBIN_PYTHON, dash.VIT_PYTHON = YOLO_PY, ''
    dash._GATE.clear()
    doc = payload(dash)
    check(f'{tag}c7c ...and not without it, whatever dogbin_python is',
          doc.get('can_run') is False
          and 'vit_python' in (doc.get('run_error') or ''), brief(doc))
    w = world(dash, protocol='yolo', verdicts_by=YOLO_RUN)
    dash.VIT_PYTHON = ''
    doc = payload(dash)
    check(f'{tag}c7d a YOLO stage runs on dogbin_python; nothing blocked',
          doc.get('can_run') is True and doc.get('can_rerun') is True
          and not doc.get('rerun_blocked'), brief(doc))
    dash.DOGBIN_PYTHON = ''
    dash._GATE.clear()
    doc = payload(dash)
    check(f'{tag}c7e ...and not without it',
          doc.get('can_run') is False, doc.get('can_run'))
    w = world(dash, protocol='yolo', verdicts_by=YOLO_RUN,
              vit_model='leash-models/leash_vit_B')
    dash.VIT_PYTHON = ''
    doc = payload(dash)
    check(f'{tag}c7f a re-run that needs vit_python is not offered without '
          f'it', doc.get('can_run') is True and doc.get('can_rerun') is False
          and 'vit_python' in (doc.get('rerun_error') or '')
          and doc.get('best_model') == 'leash_vit_B', brief(doc))

    # Stop needs no interpreter or model: a live run is stoppable whatever
    # the start configuration says
    w = world(dash, protocol='vit', verdicts_by='leash_vit_A')
    dash.VIT_PYTHON = ''
    dash.gate_pids = lambda *a, **k: [4242]
    doc = payload(dash)
    check(f'{tag}c7g a live ViT run is stoppable with no vit_python here',
          doc.get('running') is True and doc.get('can_run') is True
          and not doc.get('run_error'), brief(doc))
    w = world(dash, protocol='vit', verdicts_by='leash_vit_A')
    shutil.rmtree(w['dirs']['leash_vit_A'])
    dash.gate_pids = lambda *a, **k: [4242]
    doc = payload(dash)
    check(f'{tag}c7h ...or with its model dir unmounted',
          doc.get('can_run') is True and not doc.get('run_error'), brief(doc))
    w = world(dash, vit_model='leash-models/leash_vit_B')
    shutil.rmtree(w['dirs']['leash_vit_B'])
    dash.gate_pids = lambda *a, **k: [4242]
    doc = payload(dash)
    check(f'{tag}c7i a YOLO run in its first shard is stoppable while a '
          f'missing ViT model is promoted',
          doc.get('can_run') is True and not doc.get('run_error'), brief(doc))
    dash.gate_pids = lambda *a, **k: []
    dash._GATE.clear()
    doc = payload(dash)
    check(f'{tag}c7j ...while a Start there is refused, saying why',
          doc.get('can_run') is False
          and 'not on disk' in (doc.get('run_error') or ''), brief(doc))

    # ── c8 the page ─────────────────────────────────────────────────────────
    page = dash.TEMPLATE
    i = page.index("var rr=document.getElementById('gateRerun');\n"
                   "    if(rr&&rr.dataset.busy")
    paint = page[i:page.index('var ga=', i)]
    hid = paint[paint.index('rr.hidden='):]
    hid = hid[:hid.index(';')]
    check(f'{tag}c8 Re-run is hidden when rerun_blocked is set',
          '!!j.rerun_blocked' in hid, ' '.join(hid.split()))
    check(f'{tag}c8b it is enabled on can_rerun',
          'rr.disabled=!j.can_rerun;' in paint)
    gm = paint[paint.index("var gm=document.getElementById('gModel');"):]
    check(f'{tag}c8c the model line shows the reason, escaped',
          'j.rerun_blocked?' in gm and "'&lt;'" in gm[gm.index(
              'j.rerun_blocked?'):], 'model line rule changed')


def load(src_text=None):
    if src_text is None:
        if os.environ.get('DASH_PY'):
            src_text = open(DASH_PY).read()
        else:
            import dashboard
            return dashboard
    path = os.path.join(tempfile.mkdtemp(dir=TMP), 'dashboard.py')
    with open(path, 'w') as fh:
        fh.write(src_text)
    spec = importlib.util.spec_from_file_location('dashboard_vit_mut', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MUTANTS = [
    ('the re-run refusal on a ViT record removed',
     '    blocked = gate_rerun_blocked(stage)\n    if blocked:',
     '    blocked = None\n    if blocked:'),
    ('Start ignores the ViT pin',
     '    pin = None if fresh else stage_vit(stage)',
     '    pin = None'),
    ('LD_LIBRARY_PATH passed to the ViT child',
     "            env.pop('LD_LIBRARY_PATH', None)", '            pass'),
    ('--vit-model not passed',
     "            argv += ['--vit-model', launch['vit_dir']]",
     '            pass'),
    ('no vit_python check',
     '        if not VIT_PYTHON:\n', '        if False:\n'),
    ('stage_best ignores a promoted vit_model',
     '    if vit:\n        return os.path.basename(os.path.normpath(vit))\n',
     ''),
    ('the pre-archive ViT check removed',
     "    if launch['vit_dir'] and launch['error']:", '    if False:'),
    ('can_run on dogbin_python for every stage',
     "                    live or (not launch['error'] and ready))",
     "                    live or (bool(DOGBIN_PYTHON) and ready))"),
    ('a fresh stage ignores the promoted ViT model',
     '        if best_vit and (fresh or not _gate_has_verdicts(stage)):',
     '        if best_vit and fresh:'),
    ('a promoted ViT model resumes a YOLO record',
     '        if best_vit and (fresh or not _gate_has_verdicts(stage)):',
     '        if best_vit:'),
    ('the page shows Re-run despite rerun_blocked',
     '!!j.planning||!!waiting||\n        !!j.rerun_blocked;',
     '!!j.planning||!!waiting;'),
    ('Start ignores the ViT pin while a ViT model is promoted',
     '    pin = None if fresh else stage_vit(stage)',
     '    pin = None if (fresh or stage_best_vit(stage)) else '
     'stage_vit(stage)'),
    ('Stop gated on the start configuration',
     '                    live or (not launch[\'error\'] and ready))',
     '                    (not launch[\'error\'] and ready))'),
    ('the missing pinned dir not an error',
     '    elif not os.path.isdir(d):', '    elif False:'),
]


def main():
    try:
        dash = load()
        run_checks(dash)
        src = open(DASH_PY).read()
        for name, old, new in MUTANTS:
            if src.count(old) != 1:
                check(f'c9 mutant "{name}" applies', False,
                      f'target text found {src.count(old)} times')
                continue
            mod = load(src.replace(old, new))
            saved = list(FAILS)
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    caught = not run_checks(mod, 'mut: ')
                except Exception:   # noqa: BLE001
                    caught = True
            FAILS[:] = saved
            check(f'c9 mutant "{name}" is caught', caught)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    if FAILS:
        print(f'{len(FAILS)} failed')
        return 1
    print('the gate panel starts, resumes and re-runs a ViT leash stage with '
          'its own model and interpreter, and never re-judges ViT verdicts '
          'with a promoted YOLO run')
    return 0


if __name__ == '__main__':
    sys.exit(main())
