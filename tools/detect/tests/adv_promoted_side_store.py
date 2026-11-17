#!/usr/bin/env python3
"""A promoted detector that already swept into its own store is not re-run.

A detector is promoted before it goes live: dogdet_20261104 swept every
image into data/detect_dogdet20261104 while the live store stayed
train-30's, and goes live at switch-over. In that window the sweep panel
offered "Re-run with dogdet_20261104", which sets the live store aside and
spends days sweeping again what the side store already holds.

  s1 promoted_side_store finds the side store the promoted engine wrote
  s2 it ignores side stores of other models, and the live store itself
  s3 sweep_model_state reports it (and nothing when the live store is the
     promoted model's)
  s4 sweep_control('rerun') refuses while it exists, without touching the
     store; with no such store the refusal does not fire
  s5 the panel hides the re-run button when the payload names one
  s6 mutants: each guard removed is caught

    DASH_PY=<copy> checks an edited copy instead of the live file.

    python tools/detect/tests/adv_promoted_side_store.py      (mp14 env)
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DASH_PY = os.environ.get('DASH_PY') or os.path.join(
    REPO, 'tools', 'dashboard', 'dashboard.py')
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))

FAILS = []


def check(name, cond, detail=''):
    print(('ok   ' if cond else 'FAIL ') + name +
          (f' -- {detail}' if detail else ''))
    if not cond:
        FAILS.append(name)


class FakeArchive:
    def __init__(self, models):
        self.models = models
        self.archived = []

    def current_model(self, root):
        return self.models.get(os.path.basename(root))

    def archive_store(self, *a, **k):
        self.archived.append(a)
        return {'name': 'x'}

    def archives(self, root):
        return []


def setup(dash, tmp, models, best_engine):
    live = os.path.join(tmp, 'detect')
    side = os.path.join(tmp, 'detect_new')
    other = os.path.join(tmp, 'detect_other')
    for d in (live, side, other):
        os.makedirs(d, exist_ok=True)
    # a re-run refuses a live store with no work list for its gen
    os.makedirs(os.path.join(live, 'worklist', 'gen=0001'), exist_ok=True)
    open(os.path.join(live, 'worklist', 'gen=0001', '_dirs.json'), 'w').close()
    arch = FakeArchive(models)
    dash._sweep_archive = lambda: arch
    dash._detect_root = lambda: live
    dash._side_status_paths = lambda: [os.path.join(d, 'status.json')
                                       for d in (live, side, other)]
    dash.detector_best = lambda: {'run': 'new', 'engine': best_engine,
                                  'engine_ok': True, 'sha256': 'NEWSHA'}
    dash.sweep_pids = lambda: []
    dash.gate_pids = lambda *a, **k: []
    dash.gate_planning = lambda: False
    spawned = []
    dash._sweep_spawn = lambda engine=None, gen=None: spawned.append(engine)
    return arch, spawned


def run_checks(dash, tag=''):
    fails0 = len(FAILS)
    tmp = tempfile.mkdtemp()
    models = {'detect': {'sha256': 'OLDSHA', 'comet_run': 'old'},
              'detect_new': {'sha256': 'NEWSHA', 'comet_run': 'new'},
              'detect_other': {'sha256': 'OTHER', 'comet_run': 'other'}}
    arch, spawned = setup(dash, tmp, models, '/nonexistent/new.engine')
    got = dash.promoted_side_store()
    check(f'{tag}s1 the promoted engine\'s side store is found',
          got == 'detect_new', got)
    st = dash.sweep_model_state()
    check(f'{tag}s3 sweep_model_state names it',
          st.get('best_side_store') == 'detect_new' and not st['same'], st)
    r = dash.sweep_control('rerun')
    check(f'{tag}s4 re-run refuses while it exists',
          r.get('ok') is False and 'detect_new' in r.get('msg', '')
          and not arch.archived and not spawned, r)
    # no side store by the promoted model: nothing to refuse
    models2 = {'detect': {'sha256': 'OLDSHA'},
               'detect_new': {'sha256': 'ZZZ'},
               'detect_other': {'sha256': 'OTHER'}}
    arch, spawned = setup(dash, tmp, models2, '/nonexistent/new.engine')
    check(f'{tag}s2 other models\' side stores are not the promoted one',
          dash.promoted_side_store() is None)
    r = dash.sweep_control('rerun')
    check(f'{tag}s4b with none, re-run goes ahead',
          r.get('ok') is True and spawned, r)
    # the live store IS the promoted model's: same, no side store reported
    models3 = {'detect': {'sha256': 'NEWSHA'},
               'detect_new': {'sha256': 'NEWSHA'},
               'detect_other': {'sha256': 'OTHER'}}
    setup(dash, tmp, models3, '/nonexistent/new.engine')
    st = dash.sweep_model_state()
    check(f'{tag}s3b live store already the promoted model: none reported',
          st['same'] and not st.get('best_side_store'), st)
    return len(FAILS) == fails0


def load(src_text=None):
    import importlib.util
    if src_text is None:
        import dashboard
        return dashboard
    path = os.path.join(tempfile.mkdtemp(), 'dashboard.py')
    with open(path, 'w') as fh:
        fh.write(src_text)
    spec = importlib.util.spec_from_file_location('dashboard_mut', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.REPO = REPO
    return mod


def main():
    # DASH_PY: check an edited copy before it replaces the live file (an
    # edit to dashboard.py restarts the running dashboard)
    dash = load(open(DASH_PY).read() if os.environ.get('DASH_PY') else None)
    run_checks(dash)
    src = open(DASH_PY).read()
    check('s5 the panel hides the re-run button when one is named',
          "rr.hidden=!best||!!j.running||!!j.best_side_store;" in src)

    mutants = [
        ('no refusal in rerun', "        if st.get('best_side_store'):\n",
         "        if False and st.get('best_side_store'):\n"),
        ('side store never looked up',
         "'best_side_store': None if same else promoted_side_store(best)}",
         "'best_side_store': None}"),
    ]
    for name, old, new in mutants:
        if src.count(old) != 1:
            check(f's6 mutant "{name}" applies', False, 'target text gone')
            continue
        mod = load(src.replace(old, new))
        saved = list(FAILS)
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            caught = not run_checks(mod, 'mut: ')
        FAILS[:] = saved
        check(f's6 mutant "{name}" is caught', caught)
    if FAILS:
        print(f'{len(FAILS)} failed')
        return 1
    print('a promoted detector that already swept into its own store is '
          'switched over, never re-swept into the live store')
    return 0


if __name__ == '__main__':
    sys.exit(main())
