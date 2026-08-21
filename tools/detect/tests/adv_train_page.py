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
    # THE HAND-DRAWN BOXES ARE PART OF WHAT A BUILD READS. They are fetched
    # rather than sitting in data/, so they were missing from the line that
    # lists the inputs -- the only labels in this project somebody drew from
    # scratch, invisible next to four ledgers of corrections.
    for fam in got['families']:
        if 'label_studio' not in fam:
            bad.append('%s does not say anything about the Label Studio '
                       'export, so the hand-drawn boxes are invisible in the '
                       'list of what a build reads' % (fam.get('key'),))
    _labelstudio_checks(bad)
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
    _selector_checks(bad, html)


def _labelstudio_checks(bad):
    """How much of the export the page reports, in a training root we own.

    One Label Studio project feeds all three models, so a build of any of
    them says how big the last export was -- and the two models that have not
    been built since the export chain landed must not read as having no
    hand-drawn boxes at all, which is the opposite of the truth.
    """
    import train_page as tp
    tmp = tempfile.mkdtemp(prefix='adv_tp_ls_')
    old_root = os.environ.get('TRAINING_ROOT')
    os.environ['TRAINING_ROOT'] = tmp
    try:
        for base in ('dogdet_v2', 'dogbin_v5', 'leash_v2'):
            os.makedirs(os.path.join(tmp, base))
        # A DATASET THAT PREDATES THE EXPORT, and nothing else. The page says
        # the export happens at build time and never invents a number from a
        # build that never had one.
        older = os.path.join(tmp, 'dogdet_v3')
        for split in ('train', 'val'):
            os.makedirs(os.path.join(older, 'images', split))
        open(os.path.join(older, 'dataset.yaml'), 'w').close()
        os.makedirs(os.path.join(older, 'bundle'))
        with open(os.path.join(older, 'bundle', 'manifest.json'), 'w') as fh:
            json.dump({'family': 'dogdet', 'kind': 'detect', 'built_at': 1,
                       'built_at_iso': 'probe', 'counts': {'total': 1}}, fh)
        blank = {f['key']: f.get('label_studio')
                 for f in tp.overview()['families']}
        if any(blank.values()):
            bad.append('a training root with no builds still claims a Label '
                       'Studio count: %r' % (blank,))
        # one detector build, carrying an export of 5,649 frames
        made = os.path.join(tmp, 'dogdet_20260820_aaaaaa')
        for split in ('train', 'val'):
            os.makedirs(os.path.join(made, 'images', split))
        open(os.path.join(made, 'dataset.yaml'), 'w').close()
        os.makedirs(os.path.join(made, 'bundle'))
        with open(os.path.join(made, 'bundle', 'manifest.json'), 'w') as fh:
            json.dump({'family': 'dogdet', 'kind': 'detect', 'built_at': 1,
                       'built_at_iso': 'probe', 'counts': {'total': 1},
                       'label_studio': {'counts': {
                           'tasks': 5649, 'boxes': 4322,
                           'classes': {'leashed dog': 1224,
                                       'unleashed dog': 1384,
                                       'other animal': 1707, 'cow': 7}}}}, fh)
        got = {f['key']: f.get('label_studio')
               for f in tp.overview()['families']}
        # EACH MODEL TAKES A DIFFERENT PART OF ONE EXPORT. Reporting the
        # export's own size against all three promised the leash model twice
        # the data it gets: it never sees the goats and cows.
        if (got.get('leash') or {}).get('takes') != 1224 + 1384:
            bad.append('the leash model is credited with %r of the export, '
                       'but it only takes the dogs (%d)'
                       % ((got.get('leash') or {}).get('takes'), 1224 + 1384))
        if (got.get('dogbin') or {}).get('takes') != 4322:
            bad.append('the gate is credited with %r, though every drawn box '
                       'is a dog or a not-dog for it'
                       % ((got.get('dogbin') or {}).get('takes'),))
        if (got.get('dogdet') or {}).get('takes') != 1224 + 1384:
            bad.append('the detector is credited with %r boxes, though the '
                       'non-dog ones are dropped'
                       % ((got.get('dogdet') or {}).get('takes'),))
        # AN EXPORT THIS MODEL TAKES NOTHING FROM IS NOT NO EXPORT. Somebody
        # adds a class in Label Studio, every model drops it, and reporting
        # that as 'fetched at build' -- the same words a training root with no
        # export shows -- is how nobody finds out.
        with open(os.path.join(made, 'bundle', 'manifest.json'), 'w') as fh:
            json.dump({'family': 'dogdet', 'kind': 'detect', 'built_at': 1,
                       'built_at_iso': 'probe', 'counts': {'total': 1},
                       'label_studio': {'counts': {
                           'tasks': 12, 'boxes': 42,
                           'classes': {'horse': 40, 'tractor': 2}}}}, fh)
        none = {f['key']: f.get('label_studio')
                for f in tp.overview()['families']}
        for key, row in none.items():
            if row is None or row.get('takes') != 0:
                bad.append('%s reports %r for an export whose labels no model '
                           'takes -- it has to say none of it, not nothing '
                           'at all' % (key, row and row.get('takes')))
            elif row.get('skipped') != 42:
                bad.append('%s does not say how much of that export it '
                           'skipped: %r' % (key, row.get('skipped')))
        # A RECORD THAT IS NOT SHAPED LIKE A RECORD must not take the page
        # down: a bundle is a file on disk that somebody can edit, truncate or
        # write with an older tool, and /train reads every one of them.
        for junk in ('not a dict', ['a'], 42, None):
            with open(os.path.join(made, 'bundle', 'manifest.json'), 'w') as fh:
                json.dump({'family': 'dogdet', 'kind': 'detect',
                           'built_at': 1, 'counts': {'total': 1},
                           'label_studio': {'counts': {'tasks': 5,
                                                       'classes': junk}}}, fh)
            try:
                tp.overview()
            except Exception as e:          # noqa: BLE001
                bad.append('a bundle whose classes field is %r takes the '
                           'whole training page down: %s: %s'
                           % (junk, type(e).__name__, e))
        # ...and back to the real fixture for the rest of the checks
        with open(os.path.join(made, 'bundle', 'manifest.json'), 'w') as fh:
            json.dump({'family': 'dogdet', 'kind': 'detect', 'built_at': 1,
                       'built_at_iso': 'probe', 'counts': {'total': 1},
                       'label_studio': {'counts': {
                           'tasks': 5649, 'boxes': 4322,
                           'classes': {'leashed dog': 1224,
                                       'unleashed dog': 1384,
                                       'other animal': 1707, 'cow': 7}}}}, fh)
        got = {f['key']: f.get('label_studio')
               for f in tp.overview()['families']}
        if (got.get('leash') or {}).get('skipped') != 1707 + 7:
            bad.append('nothing says how much of the export the leash model '
                       'has no use for: %r' % (got.get('leash'),))
        if (got.get('dogdet') or {}).get('tasks') != 5649:
            bad.append('the detector does not report the export its own last '
                       'build used: %r' % (got.get('dogdet'),))
        elif not got['dogdet'].get('mine'):
            bad.append('a build of this very model is reported as somebody '
                       "else's export")
        for other in ('dogbin', 'leash'):
            if (got.get(other) or {}).get('tasks') != 5649:
                bad.append('%s reports no hand-drawn boxes, though one Label '
                           'Studio project feeds all three models: %r'
                           % (other, got.get(other)))
            elif got[other].get('mine'):
                bad.append('%s claims the export as its own build' % (other,))
    except Exception as e:                # noqa: BLE001
        bad.append('the Label Studio checks threw %s: %s'
                   % (type(e).__name__, e))
    finally:
        if old_root is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = old_root
        shutil.rmtree(tmp, ignore_errors=True)


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
    fns = []
    for name in ('harvest', 'overrides'):
        m = re.search(r'^function %s\(' % name, script, re.M)
        if not m:
            bad.append('%s() is not a top-level function of the page'
                       % (name,))
            return
        fns.append(script[m.start():script.index('\n}', m.start()) + 2])
    drive = r"""
'use strict';
var FIELDS = [{key: 'epochs', value: '100'}, {key: 'imgsz', value: '640'}];
var EDITS = {};                 // the page's own store of what was typed
var TYPED = ['device=0', 'save_period=10', '', 'not a parameter line',
             'lr0: 0.002', '=orphan value'].join(String.fromCharCode(10));
global.document = {};
// one object per id, because the code under test assigns to what it gets back
var NODES = {
  params: {querySelector: function (sel) {
    if (sel.indexOf('epochs') >= 0) return {value: ' 50 '};   // changed
    if (sel.indexOf('imgsz') >= 0) return {value: '640'};     // left alone
    return null }},
  pextra: {get value() { return TYPED }},
};
function $(id) { return NODES[id] || null; }
__FN__
// TYPED VALUES SURVIVE THE FORM BEING REBUILT. The list is rebuilt whenever
// it is expanded, and rebuilding it from the inherited values alone threw the
// edit away -- silently, so the run started with the inherited number.
FIELDS = [{key: 'epochs', value: '100', from: 'the last run'},
          {key: 'imgsz', value: '640', from: 'the last run'}];
var TYPED_IN = {epochs: ' 50 ', imgsz: '640'};
NODES.params.querySelector = function (sel) {
  var k = /data-k="([^"]+)"/.exec(sel);
  return (k && k[1] in TYPED_IN) ? {value: TYPED_IN[k[1]]} : null;
};
harvest();                                   // what a repaint does first
TYPED_IN = {};                               // ...and the controls are gone
if (EDITS.epochs !== '50')
  console.log('FAIL a typed parameter did not survive the form being rebuilt: '
              + JSON.stringify(EDITS));
if ('imgsz' in EDITS)
  console.log('FAIL an untouched field was remembered as an edit');
EDITS = {};
NODES.params.querySelector = function (sel) {
  return sel.indexOf('epochs') >= 0 ? {value: ' 50 '}
       : sel.indexOf('imgsz') >= 0 ? {value: '640'} : null };
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
""".replace('__FN__', '\n'.join(fns))
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


def _selector_checks(bad, html):
    """What the dataset selector does to a choice somebody made.

    Nothing may be auto-selected onto an unfinished build -- step 4 would post
    a dataset that was never finished. But an explicit pick has to survive the
    poll: the page labels an unfinished build 'safe to delete', and snapping
    the selection away every few seconds put the delete button permanently out
    of reach of the only thing it is meant for.
    """
    if shutil.which('node') is None:
        return
    script = html[html.rindex('<script>') + 8:html.rindex('</script>')]
    drive = r"""
'use strict';
var SEEN = {};
function mkEl(id) { return SEEN[id] || (SEEN[id] = {value: '', textContent: '',
  innerHTML: '', disabled: false, options: [], selectedIndex: 0, hidden: false,
  className: '', title: '', addEventListener: function () {},
  querySelector: function () { return null },
  querySelectorAll: function () { return [] },
  getAttribute: function () { return null }, setAttribute: function () {} }); }
global.window = {confirm: function () { return true },
                 addEventListener: function () {},
                 location: {reload: function () {}}};
global.document = {getElementById: mkEl, addEventListener: function () {},
  createElement: function () { return {textContent: '',
    get innerHTML() { return String(this.textContent).replace(/&/g, '&amp;')
      .replace(/</g, '&lt;').replace(/>/g, '&gt;') }} },
  querySelector: function () { return mkEl('q') },
  querySelectorAll: function () { return [] }};
global.fetch = function () { return Promise.resolve({ok: true, status: 200,
  json: function () { return Promise.resolve({families: [{key: 'dogdet',
    title: 'D', what: '', kind: 'detect', base: 'b', stores: {}}],
    datasets: [], jobs: [], lanes: {}}) }}) };
global.setTimeout = function () { return 0 };
global.clearTimeout = function () {};
global.setInterval = function () { return 0 };
__SCRIPT__
var bad = [];
STATE = {datasets: [
  {id: 'good', family: 'dogdet', bundle: true, label_studio: 5,
   built_at_iso: '2026-08-20T10:00:00',
   counts: {total: 10, splits: {train: {total: 8, share: 0.8, classes: {d: 8}},
                                val: {total: 2, share: 0.2, classes: {d: 2}}}}},
  {id: 'half', family: 'dogdet', counts: null, bundle: false,
   unfinished: true}], jobs: [], lanes: {},
  families: [{key: 'dogdet', title: 'D'}]};
FAM = 'dogdet';
paintDatasets();
if (SEEN['dataset'].value !== 'good')
  bad.push('nothing usable was selected on the first paint: '
           + JSON.stringify(SEEN['dataset'].value));
SEEN['dataset'].value = 'half';           // the person picks it to delete it
paintDatasets();                          // ...and the next poll repaints
if (SEEN['dataset'].value !== 'half')
  bad.push('the poll snapped the selection off the unfinished build to '
           + JSON.stringify(SEEN['dataset'].value)
           + ', so the delete button can never reach it');
// an unfinished build must still never be the automatic choice
STATE.datasets = [{id: 'half', family: 'dogdet', counts: null, bundle: false,
                   unfinished: true}];
SEEN['dataset'].value = '';
paintDatasets();
if (SEEN['dataset'].value === 'half')
  bad.push('an unfinished build was selected automatically');
// resume is offered on a run that stopped, never on one that finished
function rowFor(state) {
  STATE.jobs = [{id: 'j', state: state, meta: {family: 'dogdet'}, argv: [],
                 run: {name: 'r', resumable: true, metrics: null, epochs: 3}}];
  paintJobs();
  return SEEN['jobs'].innerHTML || '';
}
if (rowFor('failed').indexOf('data-resume') < 0)
  bad.push('a run that fell over is not offered a resume');
if (rowFor('done').indexOf('data-resume') >= 0)
  bad.push('a run that finished cleanly is offered a resume -- ultralytics '
           + 'answers that with "nothing to resume" after a job is recorded');
if (bad.length) { bad.forEach(function (b) { console.log('FAIL ' + b) });
                  process.exit(1) }
console.log('ok');
""".replace('__SCRIPT__', script)
    with tempfile.TemporaryDirectory() as tmp:
        js = os.path.join(tmp, 'sel.js')
        with open(js, 'w', encoding='utf-8') as fh:
            fh.write(drive)
        got = subprocess.run(['node', js], capture_output=True, text=True,
                             timeout=90)
    if got.returncode:
        for line in (got.stdout or '').splitlines():
            if line.startswith('FAIL '):
                bad.append('the dataset selector: ' + line[5:])
        if not (got.stdout or '').strip():
            bad.append('the selector probe died: %s'
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
        # A SUCCESSFUL RESUME IS THE RUN'S LATEST WORD ABOUT ITSELF. The
        # first manifest still holds the error that interrupted it, and
        # reading only that reported a finished run as still broken.
        again = os.path.join(tmp, 'runs', 'detect', 'dogdetection', 'again')
        os.makedirs(os.path.join(again, 'bundle'))
        with open(os.path.join(again, 'bundle', 'manifest.json'), 'w') as fh:
            json.dump({'error': 'KeyboardInterrupt: stopped',
                       'metrics': None}, fh)
        with open(os.path.join(again, 'bundle', 'resume.json'), 'w') as fh:
            json.dump({'error': None,
                       'metrics': {'metrics/mAP50-95(B)': 0.71}}, fh)
        back = tp._run_state('dogdet', 'again')
        if back.get('error'):
            bad.append('a run that was resumed and finished still reports the '
                       'error that interrupted it: %r' % (back.get('error'),))
        if (back.get('metrics') or {}).get('metrics/mAP50-95(B)') != 0.71:
            bad.append('the score the resume produced is never read: %r'
                       % (back.get('metrics'),))
        if not back.get('resumed'):
            bad.append('nothing says the run was resumed')
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
                  ('/api/train/dataset-delete', {'dataset': 'x'}),
                  ('/api/train/resume', {'family': 'dogdet', 'run': 'x'}))
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
        # ── OPERATING THE MACHINE IS NOT ANNOTATING ──
        # These start, stop or reset something shared: a six-drive rescan, the
        # sweep, the guessers, the ledger of what everybody has been shown.
        # The dashboard is public now and its annotators are volunteers; one
        # of them pressing a button they were never meant to see costs the
        # operator a rescan of sixty-eight thousand files, or hands the whole
        # queue back to everyone. NOT probed as the admin, deliberately: that
        # would start the very work this is about.
        for path in sorted(d.BoardHandler.ADMIN_POST):
            try:
                st, body = hit(path, 'sam', body={})
                bad.append('A MEMBER REACHED %s (%s) -- it operates the '
                           'machine' % (path, st))
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    bad.append('a member got %d from %s, not the same empty '
                               '404 a dead address gets' % (e.code, path))
        for want in ('/api/refresh', '/api/sweep', '/api/gate', '/api/triage'):
            if want not in d.BoardHandler.ADMIN_POST:
                bad.append('%s is not on the admin-only list, so any '
                           'annotator can call it' % (want,))
        # ...and the one route that carries both kinds of thing: marking what
        # YOU have seen is annotating, handing the queue back to everybody is
        # not, and they arrive on the same path.
        #
        # NOTE TO WHOEVER MUTATION-TESTS THIS LINE: the refusal is what keeps
        # the probe harmless. Break the check and this call reaches the real
        # seen ledger under data/, which it resets -- it leaves a .bak beside
        # it, and restoring that is on you.
        try:
            st, body = hit('/api/review/seen', 'sam', body={'reset': True})
            if json.loads(body).get('ok'):
                bad.append('A MEMBER RESET THE SHARED SEEN LEDGER, restoring '
                           'every judged crop into everybody else\'s queue')
        except urllib.error.HTTPError as e:
            if e.code != 403:
                bad.append('a member resetting the queue got %d, which is '
                           'neither a refusal nor a 404' % (e.code,))
        st, body = hit('/api/review/seen', 'sam', body={'names': []})
        if not json.loads(body).get('ok'):
            bad.append('a member can no longer mark what they have seen: %r'
                       % (body[:120],))

        # THE OPERATOR'S OWN READING IS NOT ANNOTATING EITHER. /datasets
        # walks the training tree: it names every root by absolute path --
        # the account and the drives -- and hands out every image in every
        # dataset on the box. An annotator does not need it to judge a crop.
        for path in sorted(d.BoardHandler.ADMIN_GET):
            probe = path + ('?key=x' if path.endswith(('tree', 'files',
                                                       'thumb')) else '')
            try:
                st, body = hit(probe, 'sam')
                bad.append('A MEMBER READ %s (%s)' % (path, st))
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    bad.append('a member got %d from %s, not the same empty '
                               '404 a dead address gets' % (e.code, path))
        for want in ('/datasets', '/api/datasets'):
            if want not in d.BoardHandler.ADMIN_GET:
                bad.append('%s is not admin-only, so every annotator can '
                           'read the training tree' % (want,))

        # A REQUEST LINE THAT IS NOT A PATH. A NUL reaches open() as an
        # embedded null and raises out of the standard library's own static
        # handler, taking the connection down with a traceback instead of an
        # answer -- one request at a time, a way to fill the log.
        import socket as _sk
        tok2 = auth.mint(A.get_user('sam', path=p), key_path=key)[0]

        def raw(line):
            c = _sk.create_connection(('127.0.0.1', srv.server_port),
                                      timeout=20)
            try:
                c.sendall(line + b'\r\nHost: x\r\nCookie: '
                          + auth.COOKIE.encode() + b'=' + tok2.encode()
                          + b'\r\n\r\n')
                return c.recv(120).split(b'\r\n')[0].decode()
            except Exception as e:        # noqa: BLE001
                return 'closed (%s)' % (type(e).__name__,)
            finally:
                c.close()

        for line, what in ((b'GET /recent_crops/a\x00b.jpg HTTP/1.1', 'a NUL'),
                           (b'GET /recent_crops/a%00b.jpg HTTP/1.1',
                            'an encoded NUL'),
                           (b'POST /api/detect/flag\x00 HTTP/1.1',
                            'a NUL on a POST')):
            got = raw(line)
            if '400' not in got:
                bad.append('%s in the request line answered %r -- the '
                           'standard handler raises on it and takes the '
                           'connection down' % (what, got.strip()))
        # HEAD IS A VERB TOO, and the one nobody types was the one that
        # skipped the check: do_GET and do_POST both refused a NUL and
        # do_HEAD went straight into the standard library with it.
        for line, what in ((b'HEAD /recent_crops/a\x00b.jpg HTTP/1.1',
                            'a NUL on a HEAD'),
                           (b'HEAD /recent_crops/a%00b.jpg HTTP/1.1',
                            'an encoded NUL on a HEAD')):
            got = raw(line)
            if '400' not in got:
                bad.append('%s answered %r -- the standard handler raises on '
                           'it and takes the connection down'
                           % (what, got.strip()))
        if '200' not in raw(b'GET / HTTP/1.1'):
            bad.append('an ordinary request stopped working')

        # A JUDGEMENT IS A FEW HUNDRED BYTES. Every POST used to hand
        # Content-Length straight to json.loads, so anything holding a session
        # could claim a gigabyte and the server would try to hold all of it.
        # Refused on the header, before a byte of the body is read.
        import socket as _sock
        port = srv.server_port
        tok = auth.mint(A.get_user('sam', path=p), key_path=key)[0]

        def claim(path, length, body=b'{}'):
            c = _sock.create_connection(('127.0.0.1', port), timeout=20)
            try:
                c.sendall(('POST %s HTTP/1.1\r\nHost: x\r\nCookie: %s=%s\r\n'
                           'Content-Type: application/json\r\n'
                           'Content-Length: %d\r\n\r\n'
                           % (path, auth.COOKIE, tok, length)).encode() + body)
                return c.recv(120).split(b'\r\n')[0].decode()
            except Exception as e:            # noqa: BLE001
                return 'closed (%s)' % (type(e).__name__,)
            finally:
                c.close()

        # `null` IS VALID JSON, and None is _body()'s "I already answered".
        # A body of null (or a list, or a bare number) parsed to None, every
        # route read that as "the reply is written, return", and the request
        # ended with no status line at all: a hung fetch and an empty curl.
        for body in (b'null', b'[]', b'3', b'"a string"'):
            got = claim('/api/detect/flag', len(body), body)
            if '400' not in got:
                bad.append('a body of %s answered %r rather than refusing it '
                           '-- the caller is left holding a request that was '
                           'never answered' % (body.decode(), got.strip()))

        for path in ('/api/detect/flag', '/api/review/box', '/api/audit/box',
                     '/api/review/seen'):
            got = claim(path, 2_000_000_000)
            if '413' not in got:
                bad.append('%s accepted a two-gigabyte body claim (%s) -- one '
                           'annotator can take the box down with it'
                           % (path, got.strip()))
        if '200' not in claim('/api/review/seen', 2):
            bad.append('an ordinary judgement no longer gets through')

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
                submitted.append({'kind': kind, 'argv': argv, 'lane': lane,
                                  'meta': meta})
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
            _nan_checks(bad, hit, FakeJobs)
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


def _nan_checks(bad, hit, FakeJobs):
    """A metric that is not a number must not take the page down with it.

    ultralytics produces NaN -- a metric for a class with no instances in val
    is one -- and Python writes NaN and Infinity into JSON as bare words that
    no browser will parse. One of them anywhere in the payload and the page's
    poll dies on JSON.parse, which stops everything, not just that row.
    """
    tmp = tempfile.mkdtemp(prefix='adv_tp_nan_')
    old_root = os.environ.get('TRAINING_ROOT')
    os.environ['TRAINING_ROOT'] = tmp
    try:
        run = os.path.join(tmp, 'runs', 'detect', 'dogdetection', 'nanrun')
        os.makedirs(os.path.join(run, 'bundle'))
        with open(os.path.join(run, 'bundle', 'manifest.json'), 'w') as fh:
            fh.write('{"metrics": {"metrics/mAP50-95(B)": NaN, '
                     '"fitness": Infinity}, "error": null}')

        # THE PAGE HAS ITS OWN HANDLE ON THE JOB RUNNER. Stubbing the
        # dashboard's does nothing for /api/train/overview, which asks
        # train_page, which asks the jobs module it imported itself.
        import train_page as tp
        real_jobs = tp.jobs

        class WithRun(object):
            LANES = ('build', 'train')

            def listing(self, **kw):
                return [{'id': '20260821-000000-train-aaaaaa', 'kind': 'train',
                         'lane': 'train', 'label': 'probe', 'by': 'guard',
                         'state': 'done', 'created_at': 1, 'started_at': 1,
                         'ended_at': 2, 'exit_code': 0, 'argv': [],
                         'meta': {'family': 'dogdet', 'run': 'nanrun'}}]

            def lane_holder(self, lane):
                return None

            def progress(self, job_id):
                return None

        tp.jobs = WithRun()
        try:
            st, body = hit('/api/train/overview', 'boss')
        finally:
            tp.jobs = real_jobs
        if b'nanrun' not in body:
            bad.append('the NaN probe never reached the payload, so it '
                       'proves nothing: %r' % (body[:200],))

        def strict(text):
            def boom(word):
                raise ValueError(word)
            return json.loads(text, parse_constant=boom)

        try:
            strict(body.decode())
        except ValueError as e:
            bad.append('THE OVERVIEW ANSWERED WITH %s, WHICH NO BROWSER WILL '
                       'PARSE -- one metric that is not a number and the whole '
                       'page stops updating' % (e,))
    except Exception as e:                # noqa: BLE001
        bad.append('the NaN checks threw %s: %s' % (type(e).__name__, e))
    finally:
        if old_root is None:
            os.environ.pop('TRAINING_ROOT', None)
        else:
            os.environ['TRAINING_ROOT'] = old_root
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
                       % (len(submitted), [x['argv'][:3] for x in submitted]))
        # A RESUME LOCKS THE DATASET IT WILL READ, the same way a fresh run
        # does. Without it the delete button removes the dataset out from
        # under a live resume -- and ultralytics answers a missing dataset by
        # quietly substituting its own, so the run would carry on training on
        # coco8 and write that over these weights.
        tmp = tempfile.mkdtemp(prefix='adv_tp_res_')
        old_root = os.environ.get('TRAINING_ROOT')
        os.environ['TRAINING_ROOT'] = tmp
        try:
            run = os.path.join(tmp, 'runs', 'detect', 'dogdetection', 'halted')
            os.makedirs(os.path.join(run, 'weights'))
            open(os.path.join(run, 'weights', 'last.pt'), 'w').close()
            data = os.path.join(tmp, 'dogdet_20260821_aaaaaa', 'dataset.yaml')
            os.makedirs(os.path.dirname(data))
            open(data, 'w').close()
            with open(os.path.join(run, 'args.yaml'), 'w') as fh:
                fh.write('epochs: 100\ndata: %s\n' % (data,))
            was = len(submitted)
            st, body = hit('/api/train/resume', 'boss',
                           body={'family': 'dogdet', 'run': 'halted'})
            got = json.loads(body)
            if got.get('error') or len(submitted) != was + 1:
                bad.append('a run that can be continued was not resumed: %r'
                           % (got,))
            else:
                meta = submitted[-1].get('meta') or {}
                if meta.get('dataset') != 'dogdet_20260821_aaaaaa':
                    bad.append('A RESUMED RUN DOES NOT LOCK ITS DATASET (%r) '
                               '-- the delete button will take it out from '
                               'under the run' % (meta,))
        finally:
            if old_root is None:
                os.environ.pop('TRAINING_ROOT', None)
            else:
                os.environ['TRAINING_ROOT'] = old_root
            shutil.rmtree(tmp, ignore_errors=True)
        # A RUN THAT CANNOT BE CONTINUED NEVER BECOMES A JOB. Left to the
        # launcher the answer is the same one second later, as a failed job
        # on the page that somebody has to read and then clear.
        before = len(submitted)
        st, body = hit('/api/train/resume',
                       'boss', body={'family': 'dogdet', 'run': 'no_such_run'})
        got = json.loads(body)
        if not got.get('error'):
            bad.append('a resume was accepted for a run that does not exist: '
                       '%r' % (got,))
        if len(submitted) != before:
            bad.append('a job was recorded for a run that cannot be resumed')
        # ...and a good one does submit, exactly once
        was = len(submitted)
        st, body = hit('/api/train/build', 'boss', body={'family': 'dogdet'})
        if json.loads(body).get('error') or len(submitted) != was + 1:
            bad.append('a valid build was not submitted: %r %r'
                       % (json.loads(body), submitted[was:]))
        elif submitted[-1]['kind'] != 'build':
            bad.append('the build button submitted a %r'
                       % (submitted[-1]['kind'],))


def argv_checks(bad, d):
    """The command a button submits is a list, and it is the one shown.

    In a temporary training root holding one real dogbin dataset: the start
    route resolves the dataset before it records a job, so a made-up name
    here would grade the refusal rather than the command.
    """
    made = []
    tmp = tempfile.mkdtemp(prefix='adv_tp_argv_')
    old_root = os.environ.get('TRAINING_ROOT')
    os.environ['TRAINING_ROOT'] = tmp
    ds = os.path.join(tmp, 'dogbin_x')
    for split in ('train', 'val'):
        for klass in ('dog', 'not_dog'):
            os.makedirs(os.path.join(ds, split, klass))
    os.makedirs(os.path.join(ds, 'bundle'))
    with open(os.path.join(ds, 'bundle', 'manifest.json'), 'w') as fh:
        json.dump({'family': 'dogbin', 'counts': {'total': 0}}, fh)

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
        if 'S' not in ''.join(argv[argv.index('--name') + 1:][:1]) and \
                len(argv[argv.index('--name') + 1].split('-')[-1]) != 6:
            bad.append('the run name is only minute-resolution, so two runs '
                       'on one dataset in the same minute collide: %r'
                       % (argv[argv.index('--name') + 1],))
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
    if old_root is None:
        os.environ.pop('TRAINING_ROOT', None)
    else:
        os.environ['TRAINING_ROOT'] = old_root
    shutil.rmtree(tmp, ignore_errors=True)


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
