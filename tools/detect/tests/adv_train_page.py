#!/usr/bin/env python3
"""The training page: who may reach it, and what a button is allowed to start.

Every control on this page starts work on shared hardware -- minutes of six
drives, or hours of the one GPU -- so the interesting checks are not about
what it draws. They are about who can press it and what the press turns into.

ADMIN ONLY, AND ANSWERED AS A DEAD ADDRESS. A member gets the same empty 404
from /train and from every /api/train/* route that an address which does not
exist gets. A 403 would tell them the page is real and worth asking about
again, and the read-only routes matter as much as the buttons: the overview
names every dataset and every job on the machine.

THE ARGV IS BUILT FROM A LIST. The dataset name comes off a form and ends up
in a command line, so it is checked for being a name rather than a path -- and
the whole command is a list, never a string, so a semicolon is an argument.

WHAT IS DELIBERATELY *NOT* CHECKED HERE. The training parameters. They are
handed to train_model.py as JSON and validated there against ultralytics' own
table, in the environment that has ultralytics. A second list of valid keys in
the dashboard would drift from the first, and the drift would be silent.

Run: python tools/detect/tests/adv_train_page.py
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DASH = os.path.join(REPO, 'tools', 'dashboard')
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
sys.path.insert(0, DASH)


def load_dashboard():
    spec = importlib.util.spec_from_file_location(
        'dashboard', os.path.join(DASH, 'dashboard.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['dashboard'] = mod
    spec.loader.exec_module(mod)
    return mod


def page_checks(bad):
    """What the page module answers, without a server in the way."""
    import train_page as tp
    got = tp.overview()
    if [f['key'] for f in got.get('families', [])] != list(tp.ORDER):
        bad.append('the page does not offer the three models: %r'
                   % ([f['key'] for f in got.get('families', [])],))
    for fam in got['families']:
        for key in ('title', 'what', 'kind', 'base', 'stores'):
            if key not in fam:
                bad.append('%s carries no %r' % (fam.get('key'), key))
        if not fam['stores']:
            bad.append('%s says it reads no annotation store, so the build '
                       'button is a leap of faith' % (fam['key'],))
    if 'datasets' not in got or 'jobs' not in got or 'lanes' not in got:
        bad.append('the overview is missing a section: %r' % (sorted(got),))
    # NOTHING SECRET IN A JOB ROW. The record holds an argv and could hold an
    # environment; a page shows what it is given.
    blob = json.dumps(got)
    for leak in ('PASSWORD', 'COMET_API_KEY', 'session.key', 'pw_hash'):
        if leak in blob:
            bad.append('the overview carries %r' % (leak,))
    for row in got['jobs']:
        if 'env' in row:
            bad.append('a job row carries its environment')
    html = tp.page_html(account=('', ''))
    for need in ('id="models"', 'id="dataset"', 'id="params"', 'id="jobs"',
                 'id="build"', 'id="train"', '/api/train/overview'):
        if need not in html:
            bad.append('the page has no %r' % (need,))
    if '__ACCT' in html:
        bad.append('the identity strip was never substituted')
    _script_checks(bad, html)
    _overrides_checks(bad, html)


def _script_checks(bad, html):
    """The page's script parses.

    One syntax error anywhere in the block kills every handler on the page at
    once -- the model picker, the build button, the job panel -- and the page
    still renders, so it looks fine and does nothing. The trap this catches:
    the script is inside a NON-raw Python string, so a lone backslash escape
    written into a JS literal is eaten by Python and what ships is broken.
    """
    if shutil.which('node') is None:
        print('note: node is not on PATH, so the page script was not parsed')
        return
    script = html[html.rindex('<script>') + 8:html.rindex('</script>')]
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        got = subprocess.run(['node', '--check', path], capture_output=True,
                             text=True, timeout=60)
    finally:
        os.unlink(path)
    if got.returncode:
        bad.append('THE TRAINING PAGE SCRIPT DOES NOT PARSE -- every control '
                   'on it is dead while the page still draws: %s'
                   % ((got.stderr or '').strip()[:400],))


def _overrides_checks(bad, html):
    """What the parameter form actually sends.

    ultralytics settles a hundred keys and the form curates thirty-one, so
    the box for the rest is the only way to reach device, save_period, amp and
    the loss weights from this page -- and a box whose lines are dropped on
    the floor is worse than no box, because the run then trains with settings
    the person believes they set.
    """
    if shutil.which('node') is None:
        return
    script = html[html.rindex('<script>') + 8:html.rindex('</script>')]
    m = re.search(r'^function overrides\(', script, re.M)
    if not m:
        bad.append('overrides() is not a top-level function of the page')
        return
    end = script.index('\n}', m.start()) + 2
    drive = r"""
'use strict';
var FIELDS = [{key: 'epochs', value: '100'}, {key: 'imgsz', value: '640'}];
var TYPED = ['device=0', 'save_period=10', '', 'not a parameter line',
             'lr0: 0.002', '=orphan value'].join(String.fromCharCode(10));
global.document = {};
function $(id) {
  if (id === 'params') return {querySelector: function (sel) {
    if (sel.indexOf('epochs') >= 0) return {value: ' 50 '};   // changed
    if (sel.indexOf('imgsz') >= 0) return {value: '640'};     // left alone
    return null }};
  if (id === 'pextra') return {value: TYPED};
  return null;
}
__FN__
var got = overrides();
var bad = [];
function want(key, value) {
  if (got[key] !== value)
    bad.push(key + ' was sent as ' + JSON.stringify(got[key]) + ', want '
             + JSON.stringify(value));
}
want('epochs', '50');          // a curated field that was changed
want('device', '0');           // typed into the box
want('save_period', '10');
want('lr0', '0.002');          // colon reads the same as equals
if ('imgsz' in got) bad.push('an untouched field was sent as an override');
if ('not a parameter line' in got) bad.push('a line with no key became one');
if ('' in got) bad.push('a line starting with = became a nameless parameter');
if (bad.length) { bad.forEach(function (b) { console.log('FAIL ' + b) });
                  process.exit(1) }
console.log('ok');
""".replace('__FN__', script[m.start():end])
    with tempfile.TemporaryDirectory() as tmp:
        js = os.path.join(tmp, 'ov.js')
        with open(js, 'w', encoding='utf-8') as fh:
            fh.write(drive)
        got = subprocess.run(['node', js], capture_output=True, text=True)
    if got.returncode:
        for line in (got.stdout or '').splitlines():
            if line.startswith('FAIL '):
                bad.append('the parameter form: ' + line[5:])
        if not (got.stdout or '').strip():
            bad.append('the override probe died: %s'
                       % ((got.stderr or '').strip()[:300],))


def outcome_checks(bad):
    """A finished run is reachable from the job that produced it.

    A training job that somebody waited hours for reported 'done, exit 0' and
    an argv line: the score was in ultralytics' last screenful of log and the
    weights were wherever the reader could work out they had gone. The run is
    named by the dashboard at submit time so the job knows from the start
    which directory it is producing.
    """
    import train_page as tp
    tmp = tempfile.mkdtemp(prefix='adv_tp_run_')
    old = os.environ.get('TRAINING_ROOT')
    os.environ['TRAINING_ROOT'] = tmp
    try:
        run = os.path.join(tmp, 'runs', 'detect', 'dogdetection',
                           'dogdet_20260820_b58be7_20260821-0100')
        os.makedirs(os.path.join(run, 'bundle'))
        os.makedirs(os.path.join(run, 'weights'))
        with open(os.path.join(run, 'bundle', 'manifest.json'), 'w') as fh:
            json.dump({'metrics': {'metrics/mAP50-95(B)': 0.612},
                       'error': None}, fh)
        with open(os.path.join(run, 'results.csv'), 'w') as fh:
            fh.write('epoch,metrics/mAP50-95(B)\n1,0.4\n2,0.61\n')
        open(os.path.join(run, 'weights', 'best.pt'), 'w').close()

        got = tp._run_state('dogdet', os.path.basename(run))
        if not got:
            bad.append('the run a job produced cannot be found from the job')
            return
        if (got.get('metrics') or {}).get('metrics/mAP50-95(B)') != 0.612:
            bad.append('the score is not reported: %r' % (got.get('metrics'),))
        if got.get('epochs') != 2:
            bad.append('how far the run got is not reported: %r'
                       % (got.get('epochs'),))
        if not (got.get('weights') or '').endswith('best.pt'):
            bad.append('the weights the run produced are not named: %r'
                       % (got.get('weights'),))
        # a run still going has no bundle metrics, and the last written epoch
        # is the only honest answer to how it is doing
        going = os.path.join(tmp, 'runs', 'detect', 'dogdetection', 'live')
        os.makedirs(going)
        with open(os.path.join(going, 'results.csv'), 'w') as fh:
            fh.write('epoch,metrics/mAP50-95(B)\n1,0.22\n')
        live = tp._run_state('dogdet', 'live')
        if (live.get('metrics') or {}).get('metrics/mAP50-95(B)') != 0.22:
            bad.append('a run in progress reports nothing about itself: %r'
                       % (live,))
        if tp._run_state('dogdet', 'never_existed') is not None:
            bad.append('a run directory that is not there reports as a run')
        if tp._run_state('dogdet', None) is not None:
            bad.append('a job with no run recorded invents one')
    except Exception as e:                # noqa: BLE001
        bad.append('the outcome checks threw %s: %s' % (type(e).__name__, e))
    finally:
        if old is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = old
        shutil.rmtree(tmp, ignore_errors=True)


def route_checks(bad, d):
    """Who may reach it, over real HTTP, as two real people."""
    import accounts as A
    import auth
    tmp = tempfile.mkdtemp(prefix='adv_tp_')
    srv = None
    try:
        got = auth.bootstrap(
            db_path=os.path.join(tmp, 'a.db'), key_path=os.path.join(tmp, 'k'),
            env={'DASHBOARD_USER': 'boss',
                 'DASHBOARD_PASSWORD': 'a-password-long-enough'})
        if not got.get('ok'):
            bad.append('the gate would not start: %r' % (got,))
            return
        p, key = d._accounts_db(), os.path.join(tmp, 'k')
        A.create_user('sam', 'another-good-password', path=p)

        def cookie(who):
            user = A.get_user(who, path=p)
            return {'Cookie': auth.COOKIE + '='
                    + auth.mint(user, key_path=key)[0]}

        class Quiet(d.BoardHandler):
            def log_message(self, *a):
                pass
        srv = ThreadingHTTPServer(('127.0.0.1', 0), Quiet)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = 'http://127.0.0.1:%d' % srv.server_port

        def hit(path, who, body=None, timeout=120):
            req = urllib.request.Request(base + path, headers=cookie(who))
            if body is not None:
                req.data = json.dumps(body).encode()
                req.add_header('Content-Type', 'application/json')
            r = urllib.request.urlopen(req, timeout=timeout)
            return r.status, r.read()

        READS = ('/train', '/api/train/overview',
                 '/api/train/params?family=dogdet',
                 '/api/train/log?job=x')
        WRITES = (('/api/train/build', {'family': 'dogdet'}),
                  ('/api/train/start', {'family': 'dogdet',
                                        'dataset': 'dogdet_v3'}),
                  ('/api/train/cancel', {'job': 'x'}),
                  ('/api/train/forget', {'job': 'x'}),
                  ('/api/train/dataset-delete', {'dataset': 'x'}))
        # ── the admin ──
        st, body = hit('/train', 'boss')
        if st != 200 or b'id="models"' not in body:
            bad.append('an admin did not get the page: %s' % (st,))
        st, body = hit('/api/train/overview', 'boss')
        doc = json.loads(body)
        if 'families' not in doc:
            bad.append('the overview route answered %r' % (doc,))
        # ── THE MEMBER ──
        for path in READS:
            try:
                st, body = hit(path, 'sam')
                bad.append('a member read %s (%s)' % (path, st))
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    bad.append('a member got %d from %s, not the same empty '
                               '404 a dead address gets' % (e.code, path))
        for path, payload in WRITES:
            try:
                st, body = hit(path, 'sam', body=payload)
                bad.append('A MEMBER STARTED WORK at %s (%s)' % (path, st))
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    bad.append('a member got %d from %s, not 404'
                               % (e.code, path))
        # ── signed out ──
        for path in READS:
            try:
                r = urllib.request.urlopen(base + path, timeout=60)
                if r.status == 200 and b'id="models"' in r.read():
                    bad.append('%s served the page with no cookie' % (path,))
            except urllib.error.HTTPError as e:
                if e.code not in (302, 401, 403, 404):
                    bad.append('an unauthenticated %s answered %d'
                               % (path, e.code))

        # ── FROM HERE ON, THE JOB RUNNER IS A STUB ──
        # Everything below presses a button, and a button starts minutes of
        # six drives or hours of the GPU. Left on the real runner these checks
        # do exactly that -- and under a mutation that opens the admin gate,
        # so does the member probe above: one run of this file launched a real
        # detector build and a real training run before anybody noticed. A
        # check that starts the work it is checking is not a check.
        real_jobs = dict(d._JOBS)
        submitted = []

        class FakeJobs:
            LANES = ('build', 'train')

            def submit(self, kind, argv, lane, label='', by='', meta=None,
                       **kw):
                submitted.append({'kind': kind, 'argv': argv, 'lane': lane})
                return {'ok': True, 'job': {'id': 'stub'}, 'message': ''}

            def cancel(self, job_id, **kw):
                return {'ok': True, 'job': None, 'message': ''}

            def read(self, job_id, **kw):
                return None

            def tail(self, job_id, n=0):
                return ''

            def progress(self, job_id):
                return None

            def listing(self, **kw):
                return []

            def lane_holder(self, lane):
                return None

        d._JOBS.update(mod=FakeJobs(), tried=True)
        try:
            _button_checks(bad, hit, submitted)
        finally:
            d._JOBS.clear()
            d._JOBS.update(real_jobs)

    except Exception as e:                # noqa: BLE001 - report, not die
        bad.append('the route checks threw %s: %s' % (type(e).__name__, e))
    finally:
        if srv is not None:
            srv.shutdown()
            srv.server_close()
        shutil.rmtree(tmp, ignore_errors=True)


def _button_checks(bad, hit, submitted):
    """What a press turns into, against a job runner that starts nothing."""
    if True:
        for payload, why in (
                ({'family': 'nope'}, 'a model that does not exist'),
                ({'family': None}, 'no model at all'),
                ({}, 'an empty request')):
            st, body = hit('/api/train/build', 'boss', body=payload)
            if not json.loads(body).get('error'):
                bad.append('a build was accepted for %s' % (why,))
        for payload, why in (
                ({'family': 'dogdet', 'dataset': '../../etc'},
                 'a path instead of a dataset'),
                ({'family': 'dogdet', 'dataset': '/etc/passwd'},
                 'an absolute path'),
                ({'family': 'dogdet', 'dataset': ''}, 'no dataset'),
                ({'family': 'dogdet', 'dataset': 'x', 'params': 'nope'},
                 'parameters that are not an object')):
            st, body = hit('/api/train/start', 'boss', body=payload)
            if not json.loads(body).get('error'):
                bad.append('a training run was accepted with %s' % (why,))
        if submitted:
            bad.append('%d job(s) were submitted for requests that should '
                       'have been refused: %r'
                       % (len(submitted), [s['argv'][:3] for s in submitted]))
        # ...and a good one does submit, exactly once
        st, body = hit('/api/train/build', 'boss', body={'family': 'dogdet'})
        if json.loads(body).get('error') or len(submitted) != 1:
            bad.append('a valid build was not submitted: %r %r'
                       % (json.loads(body), submitted))


def argv_checks(bad, d):
    """The command a button submits is a list, and it is the one shown."""
    made = []

    class FakeJobs:
        LANES = ('build', 'train')

        def submit(self, kind, argv, lane, label='', by='', meta=None,
                   **kw):
            made.append({'kind': kind, 'argv': argv, 'lane': lane,
                         'label': label, 'by': by, 'meta': meta})
            return {'ok': True, 'job': {'id': 'probe'}, 'message': ''}

    j = FakeJobs()
    got = d._train_build(j, 'dogdet', 'boss')
    if not got.get('ok') or not made:
        bad.append('a build was not submitted: %r' % (got,))
    else:
        argv = made[0]['argv']
        if not isinstance(argv, list) or not all(isinstance(a, str)
                                                 for a in argv):
            bad.append('the build command is not a list of strings: %r'
                       % (argv,))
        if 'build_dataset.py' not in ' '.join(argv):
            bad.append('the build does not run the builder: %r' % (argv,))
        if '--family' not in argv or argv[argv.index('--family') + 1] != \
                'dogdet':
            bad.append('the build does not name the model: %r' % (argv,))
        if made[0]['lane'] != 'build':
            bad.append('a build is not in the build lane')
    made[:] = []
    got = d._train_start(j, {'family': 'dogbin', 'dataset': 'dogbin_x',
                             'params': {'epochs': '5; rm -rf /'}}, 'boss')
    if not got.get('ok') or not made:
        bad.append('a training run was not submitted: %r' % (got,))
    else:
        argv = made[0]['argv']
        if made[0]['lane'] != 'train':
            bad.append('a training run is not in the train lane, so two could '
                       'share the GPU')
        if 'train_model.py' not in ' '.join(argv):
            bad.append('the run does not launch the trainer: %r' % (argv,))
        # THE PARAMETERS TRAVEL AS ONE JSON ARGUMENT. Spliced in as separate
        # words they would be a place to put a space and then a flag.
        if '--params-json' not in argv:
            bad.append('the parameters are not passed as JSON: %r' % (argv,))
        else:
            blob = argv[argv.index('--params-json') + 1]
            if json.loads(blob) != {'epochs': '5; rm -rf /'}:
                bad.append('the parameters were rewritten on the way: %r'
                           % (blob,))
            if any(a.startswith('rm') for a in argv):
                bad.append('a parameter value became its own argument')
        if not any('dogbin_x' == a for a in argv):
            bad.append('the run does not name the dataset: %r' % (argv,))
        # THE RUN IS NAMED HERE, NOT BY THE LAUNCHER. Left to the launcher the
        # only place the name ever appeared was a line in the log, so a job
        # that finished could not reach the run it had just produced.
        if '--name' not in argv:
            bad.append('the run is not named at submit time, so the job has '
                       'no way to reach what it produced: %r' % (argv,))
        else:
            name = argv[argv.index('--name') + 1]
            if not name.startswith('dogbin_x'):
                bad.append('the run name does not say what it trained on: %r'
                           % (name,))
            if made[0]['meta'].get('run') != name:
                bad.append('the job does not record the run it produced: %r'
                           % (made[0]['meta'],))


def main():
    bad = []
    try:
        d = load_dashboard()
    except Exception as e:                # noqa: BLE001
        print('FAIL could not load dashboard.py: %s: %s'
              % (type(e).__name__, e))
        return 1
    for fn, args in ((page_checks, ()), (outcome_checks, ()),
                     (route_checks, (d,)), (argv_checks, (d,))):
        try:
            fn(bad, *args)
        except Exception as e:            # noqa: BLE001
            bad.append('%s threw %s: %s' % (fn.__name__, type(e).__name__, e))
    if bad:
        for b in bad:
            print('FAIL ' + b)
        return 1
    print('the training page and every one of its routes answer a member '
          'with the same empty 404 a dead address gets, and a button turns '
          'into a list of arguments rather than a command line')
    return 0


if __name__ == '__main__':
    sys.exit(main())
