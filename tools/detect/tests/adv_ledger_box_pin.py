#!/usr/bin/env python3
"""The review page reads its crops' boxes from the pinned ledger store, and
the store's own generation drives every planner and sweep the dashboard
spawns.

At switch-over data/detect_root.txt is repointed at the gen-2 store. Every
crop on the review page was cut by the gen-1 detector and names its box only
by (image_id, conf); a redraw is saved under that store's det_idx. Read
through the live store, a gen-1 crop opened on gen-2's box 0, the HQ tile was
cut from it and cached for good, and the size floor lost 4,705 of 5,003
sizes. data/ledger_detect_root.txt pins the store they belong to.

  p1  box_for lists the pinned store's boxes, not the live store's
  p2  box_short_sides measures the pinned store's boxes
  p3  hq_crop cuts the one box whose conf matches the crop's name
  p4  two boxes at the crop's conf: no HQ cut, nothing cached
  p5  no box at the crop's conf: no HQ cut (never box 0), nothing cached
  p6  a cell twin's repeated rows are one box, not an ambiguity
  p7  no pin file: box_for refuses naming the file, hq_crop falls back,
      the size floor does not apply -- and detect_root.txt is never read
  p8  a pin naming a directory without shards is refused
  p9  an ambiguous crop is looked up once per pin: hq_crop and warm_hq skip
      it after that (no ~1 s store scans per tile), a new pin forgets it
  j1  the lightbox editor takes the single conf match (node)
  j2  ...and with zero or two matches says "box not editable", loads the
      burned-in frame and picks nothing -- never boxes[0]
  j3  ...and a harvested crop (no burned-in frame) keeps the original
      already loading, instead of a /recent_crops/full that 404s
  g1  the gate planner is spawned with --gen <the live store's gen>
  g2  ...and refused, spawning nothing, when the store names no gen
  g3  Resume/Re-run spawn the sweep with --gen <the live store's gen>
  g4  Re-run refuses before archive_store when worklist/gen=<g>/_dirs.json
      is missing, and goes ahead (archive, then spawn) when it is there --
      spawning the gen it checked, though archive_store moved status.json
      aside and a newer worklist/gen=* is on disk
  m   mutants: each guard removed or reverted is caught

Every path is a temp directory; no store under data/ is read.

    DASH_PY=<copy> checks an edited copy instead of the live file.

    python tools/detect/tests/adv_ledger_box_pin.py      (mp14 env)
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

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
except ImportError:
    print('SKIP: pyarrow/PIL not importable here -- nothing checked the pin')
    sys.exit(0)

import store  # noqa: E402

FAILS = []
IID = '1234567890'          # two boxes, conf 0.91 and 0.40
IID2 = '2234567890'         # two boxes at one conf: 0.550 and 0.552
IID3 = '3234567890'         # a cell twin: the same box under two cells
CELL, CELL_B = 'aa_1_2_3_4', 'bb_5_6_7_8'


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name +
          ('' if cond else f' -- {detail}'))
    if not cond:
        FAILS.append(name)


def crop(iid, c2):
    return '1786153241730_%s_%03d.jpg' % (iid, c2)


def write_store(root, gen, cells):
    """cells: {cell: {iid: [(det_idx, conf, x1, y1, x2, y2), ...]}}."""
    store.ensure_bootstrap(root)
    for cell, imgs in cells.items():
        part = os.path.join(root, 'shards', 'gen=%04d' % gen, 'region=r',
                            'cell=' + cell, 'drive=d1')
        os.makedirs(part, exist_ok=True)
        pq.write_table(pa.table({
            'image_id': pa.array([int(i) for i in imgs], pa.uint64()),
            'orig_w': pa.array([500] * len(imgs), pa.uint16()),
            'orig_h': pa.array([400] * len(imgs), pa.uint16())}),
            os.path.join(part, 's0.img.parquet'))
        dets = [(int(i),) + d for i, ds in imgs.items() for d in ds]
        pq.write_table(pa.table({
            'image_id': pa.array([d[0] for d in dets], pa.uint64()),
            'det_idx': pa.array([d[1] for d in dets], pa.uint8()),
            'conf': pa.array([d[2] for d in dets], pa.float32()),
            'x1': pa.array([d[3] for d in dets], pa.float32()),
            'y1': pa.array([d[4] for d in dets], pa.float32()),
            'x2': pa.array([d[5] for d in dets], pa.float32()),
            'y2': pa.array([d[6] for d in dets], pa.float32())}),
            os.path.join(part, 's0.det.parquet'))
    with open(os.path.join(root, 'status.json'), 'w') as fh:
        json.dump({'gen': gen}, fh)


def fixture():
    tmp = tempfile.mkdtemp(prefix='ledger-pin-')
    g1 = os.path.join(tmp, 'detect')
    g2 = os.path.join(tmp, 'detect_new')
    write_store(g1, 1, {
        CELL: {IID: [(0, 0.91, 10, 10, 110, 60), (1, 0.40, 200, 200, 230, 220)],
               IID2: [(0, 0.552, 10, 10, 60, 60), (1, 0.550, 100, 100, 200, 200)],
               IID3: [(0, 0.70, 20, 20, 120, 100)]},
        CELL_B: {IID3: [(0, 0.70, 20, 20, 120, 100)]}})
    # the live (gen-2) store knows the same frames with other boxes, one of
    # them at the crop's conf by coincidence -- the case nothing can tell apart
    write_store(g2, 2, {
        CELL: {IID: [(0, 0.77, 0, 0, 300, 300), (1, 0.91, 300, 250, 400, 390)],
               IID2: [(0, 0.66, 0, 0, 50, 50)],
               IID3: [(0, 0.30, 0, 0, 40, 40)]}})
    os.makedirs(os.path.join(g2, 'worklist', 'gen=0002'))
    repo = os.path.join(tmp, 'repo')
    os.makedirs(os.path.join(repo, 'data'))
    with open(os.path.join(repo, 'data', 'ledger_detect_root.txt'), 'w') as fh:
        fh.write(g1 + '\n')
    with open(os.path.join(repo, 'data', 'detect_root.txt'), 'w') as fh:
        fh.write(g2 + '\n')
    grid = os.path.join(tmp, 'grid')
    for cell in (CELL, CELL_B):
        os.makedirs(os.path.join(grid, cell, 'ground_animal_images'))
        for iid in (IID, IID2, IID3):
            Image.new('RGB', (500, 400), (90, 120, 30)).save(
                os.path.join(grid, cell, 'ground_animal_images', iid + '.jpg'))
    return {'tmp': tmp, 'g1': g1, 'g2': g2, 'repo': repo, 'grid': grid}


class AnyRoot(dict):
    def __init__(self, root):
        super().__init__()
        self.root = root

    def get(self, k, default=None):
        return self.root


class FakePopen:
    calls = []

    def __init__(self, argv, **kw):
        FakePopen.calls.append(list(argv))

    def wait(self, timeout=None):
        return 0


class FakeArchive:
    def __init__(self):
        self.archived = []

    def archive_store(self, *a, **k):
        # as sweep_archive does: status.json leaves, worklist/ stays
        self.archived.append(a)
        st = os.path.join(a[0], 'status.json')
        if os.path.exists(st):
            os.replace(st, st + '.archived')
        return {'name': 'x'}

    def current_model(self, root):
        return {}


def run_js(src, j, conf, has_full=True):
    """The lightbox's box pick, cut out of the page and run in node."""
    a = src.index('var want=+c.conf||0')
    b = src.index('boxMeta={name:c.name', a)
    snippet = src[a:b]
    prog = ('var out={src:null,txt:null};'
            'var els={lbi:{set src(v){out.src=v}},'
            'lbxy:{set textContent(v){out.txt=v}}};'
            'function $(id){return els[id]}'
            'function run(j,c){' + snippet + 'return pick}'
            'var p=run(%s,%s);'
            'console.log(JSON.stringify({pick:p===undefined?null:p,'
            'src:out.src,txt:out.txt}));'
            % (json.dumps(j), json.dumps({'name': 'n.jpg', 'conf': conf,
                                          'has_full': has_full})))
    r = subprocess.run(['node', '-e', prog], capture_output=True, text=True,
                       timeout=60)
    if r.returncode:
        return {'error': r.stderr.strip()[-300:]}
    return json.loads(r.stdout)


def run_checks(dash, src, fx, tag=''):
    fails0 = len(FAILS)
    os.environ['DETECT_ROOT'] = fx['g2']          # the LIVE store is gen 2
    dash.REPO = fx['repo']
    dash.HQ_DIR = tempfile.mkdtemp(dir=fx['tmp'], prefix='hq-')
    dash._grid_roots = lambda: AnyRoot(fx['grid'])
    dash._saved_box = lambda name: None
    dash._SZ['by_key'].clear()

    b = dash.box_for(crop(IID, 91))
    got = [(x['det_idx'], x['conf'], x['x1'], x['y2']) for x in
           b.get('boxes') or []]
    check(f'{tag}p1 box_for lists the pinned store\'s boxes',
          b.get('ok') and got == [(0, 0.91, 10.0, 60.0), (1, 0.4, 200.0, 220.0)],
          b)
    sides = dash.box_short_sides([(IID, 91), (IID, 40)])
    check(f'{tag}p2 the size floor measures the pinned store\'s boxes',
          sides.get((IID, 91)) == 50.0 and sides.get((IID, 40)) == 20.0,
          {k: v for k, v in sides.items() if k[0] == IID})
    p = dash.hq_crop(crop(IID, 91))
    size = Image.open(p).size if p else None
    # det 0 is 100x50 at (10,10); a 12-px pad clamped at the frame edge
    check(f'{tag}p3 hq_crop cuts the one conf-matched box',
          size == (122, 72), size)
    p = dash.hq_crop(crop(IID2, 55))
    check(f'{tag}p4 two boxes at the crop\'s conf: no cut, nothing cached',
          p is None and not os.path.exists(
              os.path.join(dash.HQ_DIR, crop(IID2, 55))), p)
    p = dash.hq_crop(crop(IID, 66))
    check(f'{tag}p5 no box at the crop\'s conf: no cut, never box 0',
          p is None and not os.path.exists(
              os.path.join(dash.HQ_DIR, crop(IID, 66))), p)
    b3 = dash.box_for(crop(IID3, 70))
    p = dash.hq_crop(crop(IID3, 70))
    check(f'{tag}p6 a cell twin\'s repeated rows are one box',
          len(b3.get('boxes') or []) == 1 and p is not None,
          (b3.get('boxes'), p))

    calls = []
    real_box_for = dash.box_for

    def counting(name):
        calls.append(name)
        return real_box_for(name)
    dash.box_for = counting
    amb = crop(IID2, 55)
    dash.hq_crop(amb)
    dash._hq_thread = None
    dash.warm_hq([amb])
    again = (len(calls), dash._hq_thread)
    pin = os.path.join(fx['repo'], 'data', 'ledger_detect_root.txt')
    g1b = os.path.join(fx['tmp'], 'detect_copy')
    shutil.copytree(fx['g1'], g1b)
    with open(pin, 'w') as fh:
        fh.write(g1b + '\n')
    try:
        dash.hq_crop(amb)
    finally:
        with open(pin, 'w') as fh:
            fh.write(fx['g1'] + '\n')
        shutil.rmtree(g1b, ignore_errors=True)
        dash.box_for = real_box_for
    check(f'{tag}p9 an ambiguous crop is looked up once per pin',
          again == (0, None) and calls == [amb], (again, calls))

    os.rename(pin, pin + '.off')
    try:
        dash._SZ['by_key'].clear()
        b = dash.box_for(crop(IID, 40))
        p = dash.hq_crop(crop(IID, 40))
        try:
            sides = dash.box_short_sides([(IID, 40)])
            raised = None
        except Exception as e:                        # noqa: BLE001
            sides, raised = {}, e
        check(f'{tag}p7 no pin: refused by name, preview, no floor -- and '
              f'never the live store',
              not b.get('ok') and 'ledger_detect_root.txt' in str(
                  b.get('error')) and p is None and raised is None
              and (IID, 40) not in sides, (b, p, sides, raised))
    finally:
        os.rename(pin + '.off', pin)
    empty = tempfile.mkdtemp(dir=fx['tmp'])
    with open(pin, 'w') as fh:
        fh.write(empty + '\n')
    try:
        b = dash.box_for(crop(IID, 40))
        check(f'{tag}p8 a pin naming no store is refused',
              not b.get('ok') and 'holds no shards' in str(b.get('error')), b)
    finally:
        with open(pin, 'w') as fh:
            fh.write(fx['g1'] + '\n')

    boxes = [{'det_idx': 0, 'conf': 0.91, 'x1': 1, 'y1': 1, 'x2': 9, 'y2': 9},
             {'det_idx': 1, 'conf': 0.55, 'x1': 2, 'y1': 2, 'x2': 8, 'y2': 8},
             {'det_idx': 2, 'conf': 0.553, 'x1': 3, 'y1': 3, 'x2': 7, 'y2': 7}]
    j = {'ok': True, 'has_file': True, 'w': 10, 'h': 10, 'boxes': boxes}
    one = run_js(src, j, 0.91)
    check(f'{tag}j1 the editor takes the single conf match',
          (one.get('pick') or {}).get('det_idx') == 0 and not one.get('txt'),
          one)
    two, none = run_js(src, j, 0.55), run_js(src, j, 0.30)
    check(f'{tag}j2 zero or two matches: "box not editable", the burned-in '
          f'frame, no pick',
          all(r.get('pick') is None and r.get('txt') == 'box not editable'
              and (r.get('src') or '').startswith('/recent_crops/full/')
              for r in (two, none)), (two, none))
    two, none = (run_js(src, j, 0.55, False), run_js(src, j, 0.30, False))
    check(f'{tag}j3 a harvested crop keeps the original, not a 404',
          all(r.get('pick') is None and r.get('txt') == 'box not editable'
              and r.get('src') is None and 'error' not in r
              for r in (two, none)), (two, none))

    # ── the store's generation drives what is spawned ───────────────────────
    dash.subprocess = types.SimpleNamespace(
        Popen=FakePopen, DEVNULL=subprocess.DEVNULL,
        TimeoutExpired=subprocess.TimeoutExpired)
    dash._detect_root = lambda: fx['g2']
    dash._script_pids = lambda *a, **k: []
    dash.training_root = lambda: fx['tmp']
    sp = {'title': 'dog-bin gate', 'dir': 'gate'}
    FakePopen.calls[:] = []
    r = dash._gate_plan('gate', sp)
    argv = FakePopen.calls[0] if FakePopen.calls else []
    check(f'{tag}g1 the gate planner is told the live store\'s gen',
          r.get('ok') and '--gen' in argv
          and argv[argv.index('--gen') + 1] == '2', (r, argv))
    nogen = tempfile.mkdtemp(dir=fx['tmp'])
    dash._detect_root = lambda: nogen
    FakePopen.calls[:] = []
    r = dash._gate_plan('gate', sp)
    check(f'{tag}g2 no gen in the store: the planner is not spawned',
          not r.get('ok') and not FakePopen.calls, (r, FakePopen.calls))
    dash._detect_root = lambda: fx['g2']

    FakePopen.calls[:] = []
    try:
        dash._sweep_spawn('e.engine')
    except Exception as e:                            # noqa: BLE001
        FakePopen.calls.append(['raised', str(e)])
    argv = FakePopen.calls[0] if FakePopen.calls else []
    check(f'{tag}g3 the sweep is spawned on the live store\'s gen',
          '--gen' in argv and argv[argv.index('--gen') + 1] == '2', argv)

    arch = FakeArchive()
    dash._sweep_archive = lambda: arch
    dash.sweep_pids = lambda: []
    dash.gate_pids = lambda *a, **k: []
    dash.gate_planning = lambda: False
    dash.detector_best = lambda: {'run': 'new', 'engine': 'e.engine',
                                  'engine_ok': True, 'sha256': 'S'}
    dash.sweep_model_state = lambda: {'best_side_store': None,
                                      'store_model': 'old', 'same': False}
    dirs = os.path.join(fx['g2'], 'worklist', 'gen=0002', '_dirs.json')
    newer = os.path.join(fx['g2'], 'worklist', 'gen=0003')
    st = os.path.join(fx['g2'], 'status.json')
    FakePopen.calls[:] = []
    refused = dash.sweep_control('rerun')
    ref_ok = (not refused.get('ok') and not arch.archived
              and not FakePopen.calls)
    with open(dirs, 'w') as fh:
        fh.write('[]')
    os.makedirs(newer)
    try:
        went = dash.sweep_control('rerun')
    finally:
        os.remove(dirs)
        os.rmdir(newer)
        if os.path.exists(st + '.archived'):
            os.replace(st + '.archived', st)
    argv = FakePopen.calls[0] if FakePopen.calls else []
    check(f'{tag}g4 re-run refuses before archiving with no work list, and '
          f'goes ahead with one',
          ref_ok and went.get('ok') and len(arch.archived) == 1
          and '--gen' in argv and argv[argv.index('--gen') + 1] == '2',
          (refused, went, arch.archived, argv))
    return len(FAILS) == fails0


def load(src_text):
    path = os.path.join(tempfile.mkdtemp(), 'dashboard.py')
    with open(path, 'w') as fh:
        fh.write(src_text)
    spec = importlib.util.spec_from_file_location('dashboard_pin_mut', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MUTANTS = [
    ('the live store read for the review boxes',
     [('    root = _ledger_detect_root()\n    return (_store._sql_src',
       '    root = _store.get_detect_root()\n    return (_store._sql_src')]),
    ('the pin read from detect_root.txt',
     [("LEDGER_ROOT_FILE = os.path.join('data', 'ledger_detect_root.txt')",
       "LEDGER_ROOT_FILE = os.path.join('data', 'detect_root.txt')")]),
    ('a missing pin falls back to the live store',
     [("    except OSError:\n        root = ''\n    if not root:",
       "    except OSError:\n        root = os.environ.get('DETECT_ROOT', '')"
       "\n    if not root:")]),
    ('a pin without shards accepted',
     [("    if not os.path.isdir(os.path.join(root, 'shards')):",
       "    if False:")]),
    ('hq_crop takes the first of two matches',
     [("    if len(hits) != 1:\n        _hq_ambig.add((pin, name))\n"
       "        return None\n    box = hits[0]",
       "    if not hits:\n        return None\n    box = hits[0]")]),
    ('hq_crop falls back to boxes[0]',
     [("    if len(hits) != 1:\n        _hq_ambig.add((pin, name))\n"
       "        return None\n    box = hits[0]",
       "    if len(hits) != 1:\n        hits = info['boxes']\n    box = hits[0]")]),
    ('hq_crop forgets an ambiguous crop',
     [("    if pin is None or (pin, name) in _hq_ambig:",
       "    if pin is None:")]),
    ('warm_hq re-queues an ambiguous crop',
     [("                and (pin, n) not in _hq_ambig][:120]",
       "                ][:120]")]),
    ('an ambiguous crop remembered across pins',
     [("    if pin is None or (pin, name) in _hq_ambig:",
       "    if pin is None or any(n == name for _, n in _hq_ambig):")]),
    ('the editor swaps a harvested crop to /recent_crops/full',
     [("        if(c.has_full)\n          $('lbi').src='/recent_crops/full/'",
       "        $('lbi').src='/recent_crops/full/'")]),
    ('the editor keeps the original even with a burned-in frame',
     [("        if(c.has_full)\n          $('lbi').src='/recent_crops/full/'",
       "        if(false)\n          $('lbi').src='/recent_crops/full/'")]),
    ('cell twins counted as two boxes',
     [('f"SELECT DISTINCT det_idx, x1, y1, x2, y2, conf FROM {det} "',
       'f"SELECT det_idx, x1, y1, x2, y2, conf FROM {det} "')]),
    ('the editor takes the first of two matches',
     [('if(hits.length!==1){', 'if(!hits.length){')]),
    ('the editor falls back to boxes[0]',
     [('if(hits.length!==1){', 'if(false){'),
      ('var pick=hits[0];', 'var pick=hits[0]||j.boxes[0];')]),
    ('the gate planner not told the gen',
     [("            [sys.executable, script, 'plan', '--stage', stage,\n"
       "             '--gen', str(gen)],",
       "            [sys.executable, script, 'plan', '--stage', stage],")]),
    ('the planner spawned with no gen',
     [("    if gen is None:\n        return {'ok': False, 'running': False,\n"
       "                'msg': 'the live store names no generation",
       "    if False:\n        return {'ok': False, 'running': False,\n"
       "                'msg': 'the live store names no generation")]),
    ('the sweep spawned on gen 1',
     [("'run', '--gen', str(gen)]", "'run', '--gen', '1']")]),
    ('re-run spawns a re-read gen, not the checked one',
     [("            _sweep_spawn(best['engine'], gen)",
       "            _sweep_spawn(best['engine'])")]),
    ('re-run archives with no work list',
     [('        if not dirs or not os.path.exists(dirs):',
       '        if not dirs:')]),
]


def main():
    src = open(DASH_PY).read()
    fx = fixture()
    env0 = os.environ.get('DETECT_ROOT')
    try:
        run_checks(load(src), src, fx)
        for name, edits in MUTANTS:
            msrc = src
            if any(msrc.count(old) != 1 for old, _ in edits):
                check(f'm  mutant "{name}" applies', False, 'target text gone')
                continue
            for old, new in edits:
                msrc = msrc.replace(old, new)
            saved = list(FAILS)
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                try:
                    caught = not run_checks(load(msrc), msrc, fx, 'mut: ')
                except Exception:                     # noqa: BLE001
                    caught = True
            FAILS[:] = saved
            check(f'm  mutant "{name}" is caught', caught)
    finally:
        if env0 is None:
            os.environ.pop('DETECT_ROOT', None)
        else:
            os.environ['DETECT_ROOT'] = env0
        shutil.rmtree(fx['tmp'], ignore_errors=True)
    if FAILS:
        print(f'{len(FAILS)} failed')
        return 1
    print('the review page reads its boxes from the pinned ledger store, and '
          'the dashboard spawns planners and sweeps on the live store\'s gen')
    return 0


if __name__ == '__main__':
    sys.exit(main())
