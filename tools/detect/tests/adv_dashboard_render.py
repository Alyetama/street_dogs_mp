"""
Adversarial test for the dashboard's detection-sweep client JS (§7.4 / A.5).

Written by the verification pass. node --check only proves the script parses;
this test EXECUTES the actual `f-detect` IIFE extracted from the freshly built
data/dashboard/index.html under node with a stub DOM, and drives render()
with:

  1. a REAL payload produced by StatusWriter (the exact wire format),
  2. {'running': false} (sweep absent),
  3. a stale frame (running false + stale/age_s/state),
  4. a degenerate running frame with every optional field missing,
  5. a frame full of nulls (eta, gpu, rates, boxes_per_img, not_a_dog).

The "Live detections" grid is a second, independent IIFE on its own 60 s
loop, so it is extracted and driven separately with /api/detect/crops
payloads: a normal sample, the empty pre-sweep state, a null (fetch failed)
response, and a hostile one whose image_id/name carry quote + tag injection.

A ReferenceError (helper defined in another scope), a TypeError
(null.toFixed) or any other throw fails the test. Also asserts the on/off
panels toggle correctly and that innerHTML actually gets populated.

Requires node on PATH; skips (exit 0, loud message) if absent.
"""

import glob
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
INDEX = os.path.join(REPO, 'data', 'dashboard', 'index.html')
sys.path.insert(0, os.path.dirname(HERE))
import status as st  # noqa: E402


def real_payload(tmp):
    """Produce a genuine on-the-wire payload via StatusWriter + read_status."""
    path = os.path.join(tmp, 's.json')
    w = st.StatusWriter(
        'r-render', 7, 32_542_334, status_path=path,
        drive_totals={'lynx': 9e6, 'bobcat': 8e6, 'capybara': 9e6,
                      'jackal': 6.5e6},
        region_totals={'South_Asia': 7e6, 'Europe': 5e6, 'Africa': 4e6,
                       'Oceania': 3e6},   # untouched -> must render muted 0%
        gpu_fn=lambda: {'util': 97, 'mem_used_mb': 23888,
                        'mem_total_mb': 24564, 'temp': 83})
    w.update(imgs_done=1_234_567, boxes_total=120_000, positives=99_000,
             crops_classified=118_000,
             class_counts={'leashed': 20_000, 'unleashed': 88_000,
                           'not_a_dog': 10_000},
             drive_done={'lynx': 400_000, 'bobcat': 300_000,
                         'capybara': 350_000, 'jackal': 184_567},
             drive_queue={'lynx': 4096, 'bobcat': 0, 'capybara': 512,
                          'jackal': 9},
             region_done={'South_Asia': 600_000, 'Europe': 400_000,
                          'Africa': 234_567},
             errors={'decode': 3, 'mount_lost': 1},
             last_error='decode: <img src=x onerror=alert(1)>.jpg')
    assert w.publish_now()
    return st.read_status(path, stale_after=120)


def _take_fn(src, start):
    """Return the whole `function ...{...}` beginning at ``start``.

    Brace-counts from the declaration's first `{`, skipping over string
    literals and comments so a `{` inside either cannot end the function
    early. Enough for hand-written page JS; not a real parser.
    """
    i = src.index('{', start)
    depth, j, n = 0, i, len(src)
    while j < n:
        c = src[j]
        if c in '"\'`':
            q, j = c, j + 1
            while j < n and src[j] != q:
                j += 2 if src[j] == '\\' else 1
        elif c == '/' and j + 1 < n and src[j + 1] == '/':
            j = src.find('\n', j)
            if j < 0:
                break
        elif c == '/' and j + 1 < n and src[j + 1] == '*':
            j = src.find('*/', j) + 1
            if j < 1:
                break
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return src[start:j + 1]
        j += 1
    raise SystemExit('could not brace-match the function at offset %d' % start)


def extract_snippets(html):
    """Pull the helper fns + the detect and crops IIFEs out of the built page."""
    script = html[html.rindex('<script>') + 8:html.rindex('</script>')]
    helpers = []
    # Brace-matched, not line-matched. These were taken with `.*$`, which
    # carries a helper only while it happens to fit on one line: adding a
    # second line to pctColor() left a dangling half-function in the harness
    # and every panel check died on a SyntaxError, reporting nothing about
    # the page it was meant to be grading.
    for name in ('fmt', 'pctColor', 'esc'):
        m = re.search(r'^function %s\(' % name, script, re.M)
        if not m:
            raise SystemExit(f'helper {name}() not found at top level '
                             f'of the built script — detect IIFE would '
                             f'throw ReferenceError')
        helpers.append(_take_fn(script, m.start()))
    # makeLightbox() is a multi-line top-level helper (LB_JS) that the crops
    # IIFE calls at construction time, taken the same way.
    # mkSpark() is the shared sparkline the detect and gate panels both call,
    # and the SPARK_* colours go with it. Without them the gate IIFE throws
    # ReferenceError before the first card is painted -- which is exactly what
    # would happen in a browser if the helper were ever moved into an IIFE.
    m = re.search(r'^function mkSpark\(', script, re.M)
    if not m:
        raise SystemExit('helper mkSpark() not found at top level of the '
                         'built script — both KPI panels call it')
    helpers.append(_take_fn(script, m.start()))
    m = re.search(r'^var SPARK_ACC=', script, re.M)
    if not m:
        raise SystemExit('the SPARK_* palette is not at top level')
    helpers.append(script[m.start():script.index(';', m.start()) + 1])

    m = re.search(r'^function makeLightbox\(', script, re.M)
    if not m:
        raise SystemExit('helper makeLightbox() not found at top level of the '
                         'built script — the crops IIFE would throw '
                         'ReferenceError (is __LB_JS__ being substituted?)')
    helpers.append(_take_fn(script, m.start()))

    def iife(marker):
        s = script.index(marker)
        return script[s:script.index('})();', s) + 5]

    return ('\n'.join(helpers), iife('/* ── detection sweep panel'),
            iife('/* ── live detection crops'),
            iife("/* ── the gate's progress"))


def check_whole_script(html):
    """Parse the ENTIRE inline script, not just the IIFEs we drive.

    Driving IIFEs one at a time cannot see a syntax error somewhere else in
    the block -- and one syntax error anywhere kills every handler on the
    page at once. The specific trap this guards: TEMPLATE is a NON-raw Python
    string, so a lone ``\\n`` written inside a JS string literal is consumed
    by Python and emitted as a real newline, producing an unterminated
    string. That shipped once and took the whole dashboard down.
    """
    script = html[html.rindex('<script>') + 8:html.rindex('</script>')]
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(script)
        path = f.name
    try:
        p = subprocess.run(['node', '--check', path],
                           capture_output=True, text=True)
    finally:
        os.unlink(path)
    if p.returncode:
        raise SystemExit('BUILT PAGE SCRIPT DOES NOT PARSE — every handler on '
                         'the dashboard is dead:\n' + p.stderr.strip()[:900])
    print('ok   whole inline script parses (%d chars)' % len(script))


def check_markup(html):
    """Static assertions on the panel's hand-written HTML/CSS.

    The node test drives the client functions, but it stubs the DOM — it
    cannot see a KPI card or a button that was never emitted in the first
    place, so those are checked here against the built page.
    """
    bad = []
    for frag, why in (
            # ── the machine ──
            ('id="f-sys"', 'machine section'),
            ('id="syCpu"', 'cpu readout'),
            ('id="syMem"', 'memory readout'),
            ('id="syGpu"', 'gpu readout'),
            ('id="syIo"', 'io-stall readout — the figure that explains the '
             'other three on a box that is storage bound'),
            ('id="sySwap"', 'swap line — the target is to sit just under the '
             'RAM ceiling, and swap is the first sign of overshooting it'),
            ('/api/sys', 'machine endpoint call'),
            ('id="gateSpark"', 'gate throughput sparkline'),
            ('data-stage="leash"', 'leash stage tab'),
            ('id="gDogLbl"', 'positive-class label — it is the one label that '
             'is about the model rather than the run, so it has to follow '
             'the stage'),
            ("'/api/gate?stage='", 'stage-scoped gate endpoint'),
            ('function mkSpark(', 'shared sparkline helper'),
            # one number per panel leads; six at one weight is six headlines
            ('class="kpi lead"', 'lead KPI card'),
            ('.kpi.lead{grid-column:span 2}', 'lead card spans two tracks'),
            ('--red:#ef5350', 'a failure colour distinguishable from the '
             'amber accent — rust was 12.5 OKLab units from it, under the 15 '
             'an average reader needs'),
            ('id="dhDone"', 'Processed KPI value slot'),
            ('>Processed<', 'Processed KPI label'),
            ('id="dcropGrid"', 'live detection grid container'),
            ('id="dcropSub"', 'live detection subtitle'),
            ('id="dcropShuffle"', 'Shuffle button'),
            ('>Live detections<', 'Live detections section title'),
            ('/api/detect/crops', 'crops endpoint call'),
            ('minmax(110px,1fr)', 'crop grid track sizing'),
            ('object-fit:cover', 'crop thumbnail fit'),
            # ── lightbox (click a tile -> full frame, box already drawn) ──
            ('id="cropLb"', 'lightbox overlay'),
            ('id="cropLbImg"', 'lightbox image slot'),
            ('id="cropLbCap"', 'lightbox caption slot'),
            ('id="cropLbClose"', 'lightbox close button'),
            ('id="cropLbPrev"', 'lightbox previous button'),
            ('id="cropLbNext"', 'lightbox next button'),
            ('/recent_crops/full/', 'full-frame URL prefix'),
            ('rgba(0,0,0,.85)', 'lightbox scrim'),
            ('max-width:92vw', 'lightbox image width cap'),
            ('max-height:88vh', 'lightbox image height cap'),
            ('object-fit:contain', 'lightbox image fit'),
            ('.dcrop.cx{cursor:pointer', 'clickable-tile affordance'),
            # ── false-positive flag -> hard negatives ──
            ('/api/detect/flag', 'flag endpoint call'),
            ('id="cropLbFlag"', 'lightbox flag button'),
            ('id="dcropFlagged"', 'flagged-count line'),
            ('class="fx"', 'per-tile flag button'),
            ('.dcrop .fx{position:absolute;left:4px;top:4px', 'flag button '
             'must sit top-LEFT, clear of the bottom-right conf badge'),
            ('@media(hover:none){.dcrop .fx{opacity:1}}', 'flag button always '
             'visible without hover (touch)'),
            # colour is a design choice, not a contract -- assert the RULE
            # exists so a flagged tile stays visually distinct, and let the
            # palette move without breaking the suite
            ('.dcrop.fl{border-color:', 'flagged tile border'),
            ('flagged as false positive', 'flagged title/undo copy'),
            ('stopPropagation', 'flag click must not open the lightbox')):
        if frag not in html:
            bad.append(f'{why} missing ({frag!r})')
    # the buttons must reuse the dashboard's own button style, not a new one
    for bid in ('cropLbClose', 'cropLbPrev', 'cropLbNext'):
        m = re.search(r'<button[^>]*id="%s"' % bid, html)
        if m and 'rbtn' not in m.group(0):
            bad.append(f'#{bid} does not use the shared .rbtn style')
    # the grid must sit inside #detOn, or an idle sweep would still show it
    on = html.find('id="detOn"')
    if on < 0 or not (on < html.find('id="dcropGrid"') < html.find('id="f-board"')):
        bad.append('crop grid is not nested inside the #detOn panel')
    # ...but the lightbox must NOT: #detOn is display:none whenever the sweep
    # is idle, which would blank an open overlay out from under the user.
    if not (html.find('id="f-board"') < html.find('id="cropLb"')):
        bad.append('lightbox is nested inside the #detOn panel (must be at '
                   'body level so an idle sweep cannot hide it)')
    if bad:
        raise SystemExit('MARKUP FAILURES: ' + ' | '.join(bad))
    print('ok   markup (Processed card + live detection grid + lightbox)')


def check_flag_api():
    """Drive dashboard.flag_crop() directly against a throwaway data dir.

    The node test can only prove the button calls the endpoint; this proves
    the contract behind it — that a flag outlives the rolling crop window,
    that the ledger is append-only and idempotent, that a lost race with the
    pruner still records the image_id, and that undo really deletes.
    """
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    import dashboard as dash  # noqa: E402

    bad = []

    def want(cond, why):
        if not cond:
            bad.append(why)

    with tempfile.TemporaryDirectory() as tmp:
        crops = os.path.join(tmp, 'recent_crops')
        os.makedirs(os.path.join(crops, 'full'))
        hn = os.path.join(tmp, 'hard_negatives')
        keep = (dash.CROPS, dash.HN_DIR, dash.HN_CROPS, dash.HN_FULL,
                dash.HN_LABELS, dash._flagged)
        dash.CROPS = crops
        dash.HN_DIR = hn
        dash.HN_CROPS = os.path.join(hn, 'crops')
        dash.HN_FULL = os.path.join(hn, 'full')
        dash.HN_LABELS = os.path.join(hn, 'labels.jsonl')
        dash._flagged = None
        try:
            name = '1785663300000_1606751523958968_073.jpg'
            with open(os.path.join(crops, name), 'wb') as f:
                f.write(b'\xff\xd8crop')
            with open(os.path.join(crops, 'full', name), 'wb') as f:
                f.write(b'\xff\xd8fullframe')

            def lines():
                try:
                    with open(dash.HN_LABELS) as f:
                        return [json.loads(x) for x in f if x.strip()]
                except OSError:
                    return []

            b, c = dash.flag_crop('../../etc/passwd')
            want(c == 400 and not b['ok'], 'malformed name did not 400')
            b, c = dash.flag_crop('')
            want(c == 400, 'empty name did not 400')

            b, c = dash.flag_crop(name)
            want(c == 200 and b['ok'] and b['copied'] is True,
                 f'flag of a live crop: {b}')
            want(b['flagged_total'] == 1, 'flagged_total should be 1')
            # the whole point: the pixels are OUT of the rolling window now
            want(os.path.exists(os.path.join(dash.HN_CROPS, name)),
                 'crop was not copied to hard_negatives/crops')
            want(os.path.exists(os.path.join(dash.HN_FULL, name)),
                 'full frame was not copied to hard_negatives/full')
            want(not glob.glob(os.path.join(dash.HN_CROPS, '*.part')),
                 'a .part temp file was left behind')
            rows = lines()
            want(len(rows) == 1, f'expected 1 ledger line, got {len(rows)}')
            if rows:
                r = rows[0]
                want(r['image_id'] == '1606751523958968', 'wrong image_id')
                want(r['conf'] == 0.73, f"wrong conf: {r.get('conf')}")
                want(r['crop'] == name, 'wrong crop name')
                want(r['label'] == 'false_positive', 'wrong label')
                want(isinstance(r['flagged_at'], int), 'flagged_at not a ts')

            # simulate the pruner: the source vanishes, the copy must not
            os.remove(os.path.join(crops, name))
            want(os.path.exists(os.path.join(dash.HN_CROPS, name)),
                 'the copy did not survive the source being pruned')

            b, _ = dash.flag_crop(name)          # idempotent
            want(b['ok'] and b.get('duplicate') is True, f're-flag: {b}')
            want(len(lines()) == 1, 'a duplicate flag appended a second line')

            # lost the race with the pruner entirely: record it anyway
            gone = '1785663399999_777777777777777_042.jpg'
            b, c = dash.flag_crop(gone)
            want(c == 200 and b['ok'] and b['copied'] is False,
                 f'pruned crop should be ok/copied:false, got {b}')
            want(len(lines()) == 2, 'pruned crop did not get a ledger line')
            want([r for r in lines() if r['image_id'] == '777777777777777'],
                 'pruned crop lost its image_id')

            # 12 threads racing on one name -> exactly one line
            race = '1785663311111_888888888888888_055.jpg'
            with open(os.path.join(crops, race), 'wb') as f:
                f.write(b'\xff\xd8x')
            ts = [threading.Thread(target=dash.flag_crop, args=(race,))
                  for _ in range(12)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            want(len([r for r in lines() if r['crop'] == race]) == 1,
                 'concurrent flags of one crop duplicated the ledger line')

            b, _ = dash.flag_crop(name, undo=True)
            want(b['ok'] and b['undone'], f'undo: {b}')
            want(not os.path.exists(os.path.join(dash.HN_CROPS, name)),
                 'undo left the copied crop behind')
            want(not os.path.exists(os.path.join(dash.HN_FULL, name)),
                 'undo left the copied full frame behind')
            want([r['crop'] for r in lines()] == [gone, race],
                 f'undo mangled the ledger: {[r["crop"] for r in lines()]}')
            b, _ = dash.flag_crop(name, undo=True)   # no-op success
            want(b['ok'], 'undo of an unflagged crop should still be ok')

            # a torn final line (crash mid-append) must not poison the reload
            with open(dash.HN_LABELS, 'a') as f:
                f.write('{"crop": "half-writ')
            dash._flagged = None
            want(dash._load_flags() == {gone, race},
                 f'reload after a torn line: {dash._load_flags()}')

            # Durability is not observable from a live process (the page cache
            # answers reads either way) and a lost flag is unrecoverable, so
            # assert the calls are in the write paths by inspection.
            want('os.fsync' in inspect.getsource(dash.flag_crop),
                 'flag_crop no longer fsyncs the ledger append')
            want('os.fsync' in inspect.getsource(dash._rewrite_labels),
                 '_rewrite_labels no longer fsyncs the rewritten ledger')
            for fn in (dash._copy_out, dash._rewrite_labels):
                want('os.replace' in inspect.getsource(fn),
                     f'{fn.__name__} no longer publishes through os.replace '
                     f'(a partial write would be visible)')

            # the crops payload must advertise what is flagged
            dash.CROPS = crops
            with open(os.path.join(crops, race), 'wb') as f:
                f.write(b'\xff\xd8x')
            p = dash.crops_payload(now_ms=1785663400000)
            want(p.get('flagged') == [race],
                 f"payload flagged list: {p.get('flagged')}")
            want(p.get('flagged_total') == 2,
                 f"payload flagged_total: {p.get('flagged_total')}")
        finally:
            (dash.CROPS, dash.HN_DIR, dash.HN_CROPS, dash.HN_FULL,
             dash.HN_LABELS, dash._flagged) = keep

    if bad:
        raise SystemExit('FLAG API FAILURES: ' + ' | '.join(bad))
    print('ok   flag api (copy-out, idempotent, pruned-race, undo, reload)')


STUB = r"""
'use strict';
const payloads = JSON.parse(process.argv[2]);
const cropPayloads = JSON.parse(process.argv[4]);
let failures = [];
const DASH = '—';          // the idle placeholder the cards must show

// The crop grid slices itself to whole rows, so it measures the resolved
// auto-fill tracks. COLS drives that measurement; 0 makes getComputedStyle
// answer 'none' so the clientWidth arithmetic fallback gets exercised too.
let COLS = 8;
global.getComputedStyle = () => ({
  gridTemplateColumns: COLS ? ('111px '.repeat(COLS)).trim() : 'none',
});

function makeEl(id) {
  return {
    id, style: {}, dataset: {}, open: true, hidden: false, clientWidth: 0,
    _innerHTML: '', textContent: '',
    set innerHTML(v) { this._innerHTML = v; },
    get innerHTML() { return this._innerHTML; },
    classList: { add(){}, remove(){} },
    addEventListener(ev, fn) { (this._h ||= {})[ev] = fn; },
    querySelectorAll() { return []; },
    querySelector() { return null; },
    setAttribute(){}, appendChild(){},
    removeAttribute(n) { delete this[n]; },
    focus(){},
  };
}
const els = {};
global.document = {
  getElementById(id) { return els[id] ||= makeEl(id); },
  querySelectorAll() { return []; },
  createElement() { return makeEl('_dyn'); },
  addEventListener(){},
  // .style matters: the lightbox freezes/restores body scroll through it
  body: { style: {}, appendChild(){}, removeChild(){} },
  hidden: false,
};
global.window = { addEventListener(){} };
global.localStorage = { getItem(){ return null; }, setItem(){} };
global.echarts = {
  init() { return { setOption(){}, resize(){} }; },
  getInstanceByDom() { return null; },
};
// fetch is never exercised: we call render() directly per payload.
// fetch is never exercised for render(); the FLAG button does POST, so it is
// captured and its answer is scripted per case.
let fetchCalls = [], fetchReply = null;   // null -> hang forever
global.fetch = (url, opt) => {
  fetchCalls.push({ url, opt });
  return fetchReply === null ? new Promise(() => {})
    : Promise.resolve({ json: () => Promise.resolve(fetchReply) });
};
const settle = () => new Promise(r => setImmediate(r));

// Minimal element with real .closest(), for driving delegated click handlers.
function fakeEl(cls, attrs, parent) {
  return {
    className: cls, _a: attrs || {}, parent: parent || null,
    getAttribute(k) { return k in this._a ? this._a[k] : null; },
    closest(sel) {
      const need = sel.split('.').filter(Boolean);
      for (let el = this; el; el = el.parent) {
        const have = String(el.className || '').split(/\s+/);
        if (need.every(c => have.includes(c))) return el;
      }
      return null;
    },
  };
}
function clickEvt(target) {
  const e = { target, _stopped: false, _prevented: false, key: undefined };
  e.stopPropagation = () => { e._stopped = true; };
  e.preventDefault = () => { e._prevented = true; };
  return e;
}

require(process.argv[3]);          // helpers + IIFE; IIFE may call tick()

// The IIFE keeps render private; re-evaluate its body with render exposed.
const fs = require('fs');
let src = fs.readFileSync(process.argv[3], 'utf8');
src = src.replace(/\(function\(\)\{/, '').replace(/\}\)\(\);?\s*$/, '');
let render;
try {
  render = new Function(src + '\nreturn render;')();
} catch (e) {
  console.log('FAIL: could not evaluate detect IIFE body: ' + e);
  process.exit(1);
}

for (const [name, p] of Object.entries(payloads)) {
  for (const id of ['detDrives', 'detRegions', 'detHealth', 'detErrs', 'detMeta'])
    els[id] && (els[id]._innerHTML = '');
  for (const id of ['dhPct', 'dhDone', 'dhEta', 'dhNow', 'dhSus', 'dhCount', 'dhRun'])
    els[id] && (els[id].textContent = '');
  try {
    render(p);
    const on = els['detOn'], off = els['detOff'];
    // The cards are ALWAYS on screen now: an idle sweep dashes the numbers,
    // it does not collapse the layout (that made the panel jump on start).
    if (on.style.display === 'none')
      failures.push(name + ': card layout hidden — it must never collapse');
    if (p && p.running) {
      const head = ['dhPct', 'dhDone', 'dhEta', 'dhNow', 'dhSus', 'dhCount']
        .map(i => String(els[i].textContent)).join(' | ');
      if (!head.replace(/[ |]/g, '') || /undefined|NaN|null/.test(head))
        failures.push(name + ': junk in headline: ' + head);
      if (!els['detDrives']._innerHTML)
        failures.push(name + ': drives empty');
      if ((p.crops_classified || 0) > 0) {
        if (!els['detHealth']._innerHTML.includes('dband'))
          failures.push(name + ': health gauge missing');
      } else if (!els['detHealth']._innerHTML.includes('classifier not wired'))
        failures.push(name + ': classifier-absent line missing');
      if (!els['detErrs']._innerHTML)
        failures.push(name + ': errors line empty');
      // the headline % / bar track the GLOBAL imgs_done, never the per-process
      // run_imgs_done (which is a secondary line and may be far smaller)
      if (p.imgs_total && p.imgs_done != null) {
        const want = 100 * p.imgs_done / p.imgs_total,
              got = parseFloat(String(els['dhPct'].textContent)),
              w = parseFloat(String(els['dhFill'].style.width));
        if (!(Math.abs(got - want) < 0.02))
          failures.push(name + ': % complete is ' + got + ', expected '
            + want.toFixed(2) + ' — must use the global imgs_done');
        if (!(Math.abs(w - Math.min(want, 100)) < 0.02))
          failures.push(name + ': progress bar (' + w + '%) does not track the '
            + 'global % (' + want.toFixed(2) + ')');
      }
      if (p.run_imgs_done != null) {
        const run = String(els['dhRun'].textContent);
        if (!run.includes(p.run_imgs_done.toLocaleString('en-US')))
          failures.push(name + ': per-run line missing run_imgs_done: ' + run);
        if (String(els['dhDone'].textContent) ===
            p.run_imgs_done.toLocaleString('en-US'))
          failures.push(name + ': Processed headline shows the per-run count');
      }
      const h = els['detHealth']._innerHTML + els['detErrs']._innerHTML
        + els['detMeta']._innerHTML;
      if (h.includes('<img'))
        failures.push(name + ': last_error not HTML-escaped (XSS)');
      if (/undefined%|NaN%|null%/.test(els['detRegions']._innerHTML))
        failures.push(name + ': junk value rendered in regions');
      if (p.regions && Object.keys(p.regions).length) {
        const rows = (els['detRegions']._innerHTML.match(/class="drow/g) || []).length;
        if (rows !== Object.keys(p.regions).length)
          failures.push(name + ': region list not complete (' + rows + ' of '
            + Object.keys(p.regions).length + ' rendered)');
        if (Object.values(p.regions).some(v => !v)
            && !els['detRegions']._innerHTML.includes('dmut'))
          failures.push(name + ': untouched region not muted');
      }
    } else if (p && p.imgs_total) {
      // IDLE, BUT THE WORK HAPPENED. The panel used to treat "no process" as
      // "no data" and dashed a finished 32.5M-image sweep into six em-dashes,
      // reporting 0.00% complete under a full store. What was DONE survives
      // the process; only rates and an ETA describe one.
      if (off.style.display !== '')
        failures.push(name + ': idle status line not shown');
      const want = (100 * p.imgs_done / p.imgs_total).toFixed(2) + '%';
      if (String(els['dhPct'].textContent) !== want)
        failures.push(name + ': % complete is '
          + JSON.stringify(els['dhPct'].textContent) + ' with ' + p.imgs_done
          + ' of ' + p.imgs_total + ' images done — expected ' + want);
      if (els['dhFill'].style.width !== want)
        failures.push(name + ': progress bar is ' + els['dhFill'].style.width
          + ', expected ' + want + ' — the work does not un-happen');
      if (String(els['dhDone'].textContent) === DASH)
        failures.push(name + ': Processed dashed away a real image count');
      // an ETA is a claim about a process; a finished run says so instead
      if (String(els['dhEta'].textContent) !== (p.finished ? 'complete' : DASH))
        failures.push(name + ': ETA is '
          + JSON.stringify(els['dhEta'].textContent) + ', expected '
          + (p.finished ? 'complete' : 'a dash'));
      for (const id of ['dhNow', 'dhSus'])
        if (String(els[id].textContent) !== DASH)
          failures.push(name + ': ' + id + ' is a throughput — it must dash '
            + 'when nothing is reading, got '
            + JSON.stringify(els[id].textContent));
      const reg = els['detRegions']._innerHTML;
      const rows = (reg.match(/class="drow/g) || []).length;
      if (rows !== Object.keys(p.regions || {}).length)
        failures.push(name + ': region list not complete (' + rows + ' of '
          + Object.keys(p.regions || {}).length + ' rendered)');
      for (const [rn, rv] of Object.entries(p.regions || {}))
        if (!reg.includes('>' + rv.toFixed(1) + '%<'))
          failures.push(name + ': region ' + rn + ' lost its ' + rv
            + '% — completion is work done, not work happening');
      const doneN = Object.values(p.regions || {}).filter(v => v >= 100).length;
      if (Object.keys(p.regions || {}).length
          && !String(els['detRegHead'].textContent).includes(
              doneN + ' of ' + Object.keys(p.regions).length + ' complete'))
        failures.push(name + ': region heading does not count the finished '
          + 'ones: ' + JSON.stringify(els['detRegHead'].textContent));
      const drv = els['detDrives']._innerHTML;
      for (const [dn, d] of Object.entries(p.drives || {})) {
        const dp = d.total ? (100 * d.done / d.total).toFixed(0) + '%' : null;
        if (dp && !drv.includes(dp))
          failures.push(name + ': drive ' + dn + ' lost its ' + dp
            + ' of work done');
      }
      if (Object.keys(p.drives || {}).length && !drv.includes(DASH + ' img/s'))
        failures.push(name + ': per-drive img/s must dash when idle: ' + drv);
    } else {
      if (off.style.display !== '')
        failures.push(name + ': idle status line not shown');
      if (!off.textContent.includes('sweep idle'))
        failures.push(name + ': missing idle text: ' + off.textContent);
      // every headline slot reads as a dash — never 0, blank or undefined
      for (const id of ['dhPct', 'dhDone', 'dhEta', 'dhNow', 'dhSus'])
        if (String(els[id].textContent) !== DASH)
          failures.push(name + ': ' + id + ' should be a dash when idle, got '
            + JSON.stringify(els[id].textContent));
      for (const id of ['dhCount', 'dhRun'])
        if (!String(els[id].textContent).includes(DASH))
          failures.push(name + ': ' + id + ' lost its dash placeholder: '
            + JSON.stringify(els[id].textContent));
      if (els['dhFill'].style.width !== '0.00%')
        failures.push(name + ': progress bar not zeroed when idle');
      // the per-drive / per-region rows stay put (dashed), they do not vanish
      for (const id of ['detDrives', 'detRegions', 'detHealth', 'detErrs'])
        if (!els[id]._innerHTML)
          failures.push(name + ': ' + id + ' emptied out when idle');
      if (!els['detDrives']._innerHTML.includes('lynx'))
        failures.push(name + ': drive roster not carried over from the last '
          + 'live frame (rows disappeared)');
      for (const id of ['detDrives', 'detRegions'])
        if (!els[id]._innerHTML.includes(DASH))
          failures.push(name + ': idle ' + id + ' rows show values, not dashes');
      if (/\d+(\.\d+)?%<\/span>/.test(els['detRegions']._innerHTML))
        failures.push(name + ': idle region rows still show a percentage');
    }
    console.log('ok   ' + name);
  } catch (e) {
    failures.push(name + ': THREW ' + e.constructor.name + ': ' + e.message);
    console.log('FAIL ' + name + ' — ' + e);
  }
}
// ── the "Live detections" grid + lightbox: a second, independent IIFE ──
let csrc = fs.readFileSync(process.argv[5], 'utf8');
csrc = csrc.replace(/\(function\(\)\{/, '').replace(/\}\)\(\);?\s*$/, '');
let C;
try {
  // the open-crop snapshot now lives inside the shared makeLightbox()
  // component, so read it back through LB rather than an IIFE-local
  C = new Function(csrc +
    '\nreturn {renderCrops, openLb, closeLb, step, onGridClick, toggleFlag,'
    + ' list: () => LB.list(), flags: () => flagged, total: () => flagTotal};')();
} catch (e) {
  console.log('FAIL: could not evaluate crops IIFE body: ' + e);
  process.exit(1);
}
const lb = els['cropLb'], lbImg = els['cropLbImg'], lbCap = els['cropLbCap'],
      cgrid = els['dcropGrid'];

// Grid widths to fit against. {c:0} forces getComputedStyle to answer 'none'
// so the clientWidth arithmetic fallback is exercised: floor((1000+8)/118)=8.
const COLCASES = [{c: 8}, {c: 6}, {c: 4}, {c: 1}, {c: 0, w: 1000, eff: 8}];
const ROWS = 2;

for (const [name, p] of Object.entries(cropPayloads)) {
for (const cc of COLCASES) {
  const eff = cc.eff || cc.c, tag = 'crops/' + name + '@' + eff + 'col'
    + (cc.c ? '' : '(width-fallback)');
  COLS = cc.c;
  cgrid.clientWidth = cc.w || 0;
  cgrid._innerHTML = '';
  try {
    C.closeLb();                       // never leak state between cases
    document.body.style.overflow = 'auto';   // a page value worth restoring
    C.renderCrops(p);
    const all = (p && p.crops) || [], html = cgrid._innerHTML;
    // whole rows only: floor(have/cols)*cols capped at ROWS, and if we cannot
    // fill even one row we show what we have (a single short row is fine — a
    // ragged row *trailing a full one* is what looked broken)
    const want = Math.min(Math.floor(all.length / eff), ROWS) * eff || all.length;
    const cs = all.slice(0, want);
    if (!html) { failures.push(tag + ': grid left empty'); }
    const tiles = (html.match(/class="dcrop/g) || []).length;
    if (tiles !== want)
      failures.push(tag + ': ' + tiles + ' tiles of ' + all.length
        + ' available, expected ' + want);
    if (tiles > eff && tiles % eff !== 0)
      failures.push(tag + ': ragged trailing row — ' + tiles + ' tiles in a '
        + eff + '-wide grid');
    if (tiles > ROWS * eff)
      failures.push(tag + ': ' + tiles + ' tiles exceeds ' + ROWS + ' rows');
    if (!cs.length) {
      if (!/no detections/.test(html))
        failures.push(tag + ': empty state text missing');
    } else {
      // exactly the has_full crops are clickable — no more, no fewer
      const want = cs.filter(c => c.has_full).length,
            got = (html.match(/class="dcrop cx"/g) || []).length;
      if (got !== want)
        failures.push(tag + ': ' + got + ' clickable tiles, expected ' + want);
      if ((html.match(/data-i="/g) || []).length !== want)
        failures.push(tag + ': data-i index missing on a clickable tile');
      // exactly one <img> per crop: a hostile name/image_id that escaped
      // encodeURIComponent/esc() would show up as an extra tag here
      const imgs = (html.match(/<img/g) || []).length;
      if (imgs !== cs.length)
        failures.push(tag + ': ' + imgs + ' <img> tags for ' + cs.length
          + ' crops — markup injection');
      if (/<script/i.test(html))
        failures.push(tag + ': <script> injected into the grid');
      if (/undefined|NaN/.test(html))
        failures.push(tag + ': junk rendered into a tile');
    }
    // ── lightbox ──
    const clickable = cs.map((c, i) => [c, i]).filter(([c]) => c.has_full);
    if (!clickable.length) {
      C.openLb(0);                     // nothing to show, must not throw
      C.closeLb();
    } else {
      const [c0, i0] = clickable[0];
      C.openLb(i0);
      if (lb.hidden !== false) failures.push(tag + ': lightbox did not open');
      if (!String(lbImg.src).startsWith('/recent_crops/full/'))
        failures.push(tag + ': lightbox src not the full frame: ' + lbImg.src);
      if (String(lbImg.src).includes('<') || String(lbImg.src).includes('"'))
        failures.push(tag + ': lightbox src not URL-encoded: ' + lbImg.src);
      const cp = String(lbCap.textContent);
      if (!/^image_id .* · conf \d\.\d\d · \d+s ago$/.test(cp))
        failures.push(tag + ': bad caption: ' + cp);
      if (!cp.includes(String(c0.image_id)))
        failures.push(tag + ': caption lost the image_id');
      if (document.body.style.overflow !== 'hidden')
        failures.push(tag + ': body scroll not frozen');
      // arrows must stay inside the snapshot and wrap, never run off the end
      const seen = new Set();
      for (let k = 0; k < clickable.length + 3; k++) {
        C.step(1);
        if (!String(lbImg.src).startsWith('/recent_crops/full/'))
          failures.push(tag + ': step(1) produced a bad src: ' + lbImg.src);
        seen.add(String(lbImg.src));
      }
      C.step(-1); C.step(-1);
      if (!String(lbImg.src).startsWith('/recent_crops/full/'))
        failures.push(tag + ': step(-1) produced a bad src: ' + lbImg.src);
      if (seen.size !== C.list().length)
        failures.push(tag + ': stepped through ' + seen.size + ' of '
          + C.list().length + ' snapshotted crops');
      // a 60 s refresh landing mid-view must NOT close or blank the lightbox
      const held = String(lbImg.src), heldN = C.list().length;
      C.renderCrops({crops: [], total_last_min: 0});
      if (lb.hidden !== false)
        failures.push(tag + ': refresh auto-closed the lightbox');
      if (String(lbImg.src) !== held)
        failures.push(tag + ': refresh swapped the lightbox image');
      C.step(1);
      if (C.list().length !== heldN)
        failures.push(tag + ': refresh shrank the lightbox snapshot');
      C.closeLb();
      if (lb.hidden !== true) failures.push(tag + ': lightbox did not close');
      if (document.body.style.overflow !== 'auto')
        failures.push(tag + ': body scroll not restored (got '
          + document.body.style.overflow + ')');
      C.closeLb();                     // idempotent: a double Esc must be safe
    }
  } catch (e) {
    failures.push(tag + ': THREW ' + e.constructor.name + ': ' + e.message);
    console.log('FAIL ' + tag + ' — ' + e);
  }
}
console.log('ok   crops/' + name + ' (all ' + COLCASES.length + ' grid widths)');
}

// ── false-positive flagging (hard negatives) ──────────────────────────────
(async () => {
  const fail = m => failures.push('flag: ' + m);
  const P = cropPayloads.mixed_has_full, NAMES = P.crops.map(c => c.name);
  const lbF = els['cropLbFlag'], fLine = els['dcropFlagged'];
  const frame = (flagged, total) => ({ crops: P.crops, total_last_min: 6,
    flagged, flagged_total: total });
  const fxEvt = i => clickEvt(fakeEl('fx', { 'data-fx': NAMES[i] },
    fakeEl('dcrop cx', { 'data-i': String(i) })));
  try {
    COLS = 8; cgrid.clientWidth = 0; fetchReply = null; C.closeLb();

    // seeded from the payload, so a flag survives refresh / Shuffle / reload
    C.renderCrops(frame([NAMES[0]], 3));
    if (!C.flags().has(NAMES[0])) fail('Set not seeded from payload.flagged');
    if (!cgrid.innerHTML.includes(' fl"'))
      fail('seeded flag did not render the flagged tile state');
    if (!cgrid.innerHTML.includes('flagged as false positive'))
      fail('flagged tile is missing the undo title');
    if (String(fLine.textContent) !== '3 flagged as false positive')
      fail('count line: ' + JSON.stringify(fLine.textContent));
    C.renderCrops(frame([], 0));
    if (String(fLine.textContent) !== '')
      fail('count line must say nothing at 0: '
        + JSON.stringify(fLine.textContent));

    // clicking the ✗ flags, swallows the event, and does NOT open the lightbox
    fetchCalls = []; C.closeLb();
    const e1 = fxEvt(0);
    C.onGridClick(e1);
    if (!e1._stopped) fail('the ✗ click did not stopPropagation');
    if (lb.hidden === false) fail('the ✗ click opened the lightbox');
    if (!C.flags().has(NAMES[0])) fail('the ✗ click did not flag');
    if (!cgrid.innerHTML.includes(' fl"')) fail('tile did not turn flagged');
    if (!fetchCalls.length || !String(fetchCalls[0].url).includes('/api/detect/flag'))
      fail('no POST to /api/detect/flag');
    else {
      const o = fetchCalls[0].opt || {}, b = JSON.parse(o.body || '{}');
      if (o.method !== 'POST') fail('flag request is not a POST');
      if (b.name !== NAMES[0] || b.label !== 'false_positive' || b.undo !== false)
        fail('wrong flag body: ' + o.body);
    }
    // ...but a click on the tile itself still opens it
    C.onGridClick(clickEvt(fakeEl('dcrop cx', { 'data-i': '0' })));
    if (lb.hidden !== false) fail('a plain tile click stopped opening the lightbox');
    C.closeLb();

    // a refresh landing while the POST is still in flight must not revert it
    C.renderCrops(frame([], 0));
    if (!C.flags().has(NAMES[0]))
      fail('an in-flight flag was reverted by a refresh');

    // clicking again undoes
    fetchCalls = [];
    C.onGridClick(fxEvt(0));
    if (C.flags().has(NAMES[0])) fail('the second ✗ click did not undo');
    if (JSON.parse((fetchCalls[0].opt || {}).body || '{}').undo !== true)
      fail('undo:true not sent');

    // a server refusal rolls the optimistic tile back
    fetchReply = { ok: false, error: 'nope' };
    C.renderCrops(frame([], 0));
    C.toggleFlag(NAMES[2]);
    if (!C.flags().has(NAMES[2])) fail('the tile did not flip optimistically');
    await settle(); await settle();
    if (C.flags().has(NAMES[2])) fail('a refused flag was not rolled back');

    // an accepted flag takes flagged_total from the response
    fetchReply = { ok: true, flagged_total: 41 };
    C.toggleFlag(NAMES[2]);
    await settle(); await settle();
    if (C.total() !== 41) fail('flagged_total not taken from the response');
    if (String(fLine.textContent) !== '41 flagged as false positive')
      fail('count line not updated from the response: '
        + JSON.stringify(fLine.textContent));

    // the lightbox button tracks the shown image and syncs back to the grid
    fetchReply = { ok: true, flagged_total: 1 };
    C.renderCrops(frame([], 0));
    C.closeLb(); C.openLb(0);
    if (String(lbF.className).includes('on'))
      fail('lightbox flag button starts in the flagged state');
    C.toggleFlag(P.crops[0].name);
    await settle(); await settle();
    if (!String(lbF.className).includes('on'))
      fail('lightbox flag button did not turn on');
    if (!String(lbF.title).includes('click to undo'))
      fail('lightbox flag title not updated: ' + lbF.title);
    if (!cgrid.innerHTML.includes(' fl"'))
      fail('the grid tile did not sync from the lightbox flag');
    C.closeLb();
    console.log('ok   flag ui (seed, ✗ click, no-lightbox, undo, rollback, lightbox sync)');
  } catch (e) {
    fail('THREW ' + e.constructor.name + ': ' + e.message);
    console.log('FAIL flag ui — ' + e);
  }

  if (failures.length) { console.log('FAILURES: ' + failures.join(' | ')); process.exit(1); }
  console.log('all render cases passed');
})();
"""


def check_key_metrics():
    """Every project's key_metric must name a metric its best model reports.

    An unmatched key accents nothing, which is indistinguishable from a
    project that simply has no headline. leash-models sat like that after its
    metrics were renamed, while dog-bin accented accuracy_top1 -- the one
    metric its own entry says it was NOT promoted on. A silent accent is worse
    than a wrong one: nobody looks twice at a panel that renders cleanly.
    """
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    path = os.path.join(repo, 'data', 'best_models.json')
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise SystemExit(f'{path}: {e}')
    bad = []
    for name, proj in (doc.get('projects') or {}).items():
        key = proj.get('key_metric')
        best = proj.get('best')
        if not best:
            continue                      # nothing promoted: nothing to accent
        metrics = best.get('metrics') or {}
        if not key:
            bad.append(f'{name}: best model but no key_metric')
        elif key not in metrics:
            bad.append(f'{name}: key_metric {key!r} not in '
                       f'{sorted(metrics)}')
    if bad:
        raise SystemExit('key_metric does not match the reported metrics:\n  '
                         + '\n  '.join(bad))
    n = sum(1 for p in (doc.get('projects') or {}).values() if p.get('best'))
    print(f'ok   key_metric resolves for all {n} promoted model(s)')


def check_training_tracker():
    """The tracker's failure modes all render cleanly.

    t1  A live training must be claimed by AT MOST ONE run directory. Three
        directories here share one project, one dataset and one command line
        from an afternoon of restarts; a matcher without exclusivity paints
        all three "running" and the page looks fine.

    t2  Ultralytics breaks a fitness tie toward the FIRST epoch and Python's
        max() toward the LAST. On a saturated metric that moves the reported
        best epoch by tens of epochs -- and the wrong answer is a plausible
        number in the right range, which nothing downstream can catch.

    t3  results.csv headers are padded in older ultralytics versions. An
        unstripped parser returns zero columns, which is indistinguishable
        from a run that has not started.

    t4  A run that never wrote an epoch must not be reported as finished.

    t5  The detect loss heads are not stable across ultralytics versions --
        this project has runs with train/dfl_loss and runs with train/l1_loss.
        A hardcoded triple drops the unknown term, still draws a plausible
        curve, and captions it with a loss that is not in the sum.

    t6  A run directory is wherever ultralytics' runs_dir puts it. A fixed
        <root>/<project>/<run> walk found a stale same-named directory and
        reported "no epoch finished" while the real run was eight epochs in.
    """
    import importlib.util
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    mod = os.path.join(repo, 'tools', 'dashboard', 'training_tracker.py')
    spec = importlib.util.spec_from_file_location('training_tracker', mod)
    tt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tt)
    bad = []

    # t2 -- a tie at the top, deliberately not at the last epoch
    rows = [{'metrics/accuracy_top1': v, 'metrics/accuracy_top5': 1.0}
            for v in (0.80, 0.92, 0.91, 0.92, 0.90)]
    if tt.best_index(rows, 'classify') != 1:
        bad.append(f't2 fitness tie broke toward epoch '
                   f'{tt.best_index(rows, "classify") + 1}, not the first')

    with tempfile.TemporaryDirectory() as tmp:
        # t3 -- the padded header an older ultralytics writes
        pad = os.path.join(tmp, 'padded.csv')
        with open(pad, 'w') as fh:
            fh.write('                  epoch,             train/loss,'
                     '  metrics/accuracy_top1\n1,0.6,0.75\n2,0.5,0.80\n')
        got = tt.read_results(pad)
        if len(got) != 2 or 'metrics/accuracy_top1' not in (got[0] if got
                                                           else {}):
            bad.append(f't3 padded header parsed to {got}')

        # t5 -- the same idea, two different loss vocabularies
        for heads, want in ((('box', 'cls', 'dfl'), 'box + cls + dfl'),
                            (('box', 'cls', 'l1'), 'box + cls + l1')):
            f = os.path.join(tmp, 'loss_' + heads[-1] + '.csv')
            cols = ','.join(['epoch']
                            + [f'train/{h}_loss' for h in heads]
                            + [f'val/{h}_loss' for h in heads])
            with open(f, 'w') as fh:
                fh.write(cols + '\n1,1,2,3,1,2,3\n2,1,1,1,1,1,1\n')
            r = tt.read_results(f)
            tr, va = tt.loss_series(r)
            if tr[0] != 6 or va[0] != 6:
                bad.append(f't5 {heads} summed to train={tr[0]} val={va[0]}, '
                           f'not 6 -- a head was dropped')
            if not tt.loss_label(r).startswith(want):
                bad.append(f't5 {heads} labelled {tt.loss_label(r)!r}')

        # t6 -- the real run lives under runs_dir, a stale twin sits at the
        # two-level path, and both carry the same project and name
        deep = os.path.join(tmp, 'deep')
        real = os.path.join(deep, 'runs', 'detect', 'proj', 'r1')
        stale = os.path.join(deep, 'proj', 'r1')
        for d, sd in ((real, real), (stale, 'proj/r1')):
            os.makedirs(d)
            with open(os.path.join(d, 'args.yaml'), 'w') as fh:
                fh.write(f'name: r1\nproject: proj\ndata: /d.yaml\n'
                         f'epochs: 100\npatience: 10\nsave_dir: {sd}\n')
        with open(os.path.join(real, 'results.csv'), 'w') as fh:
            fh.write('epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n'
                     '1,0.5,0.3\n2,0.6,0.4\n')
        found = {r['dir'] for r in tt.discover(deep)}
        if real not in found:
            bad.append('t6 the run under runs_dir was not discovered')
        claimed = tt.attach_live(
            tt.discover(deep),
            [{'pid': -3, 'argv': [], 'project': 'proj', 'name': 'r1',
              'data': '/d.yaml', 'started': None}])
        if list(claimed) != [real]:
            bad.append(f't6 the live process claimed {list(claimed)}, '
                       f'not the directory its own save_dir names')

        # t1 / t4 -- three lookalike run dirs, one live process
        root = os.path.join(tmp, 'root', 'proj')
        os.makedirs(root)
        for name in ('run_a', 'run_a2', 'older'):
            d = os.path.join(root, name)
            os.makedirs(d)
            with open(os.path.join(d, 'args.yaml'), 'w') as fh:
                fh.write(f'name: {name}\nproject: proj\n'
                         f'data: /same/dataset.yaml\nepochs: 100\n'
                         f'patience: 10\n')
        runs = tt.discover(os.path.join(tmp, 'root'))

        def proc(pid, name):
            return {'pid': pid, 'argv': [], 'project': 'proj', 'name': name,
                    'data': '/same/dataset.yaml', 'started': None}

        claims = tt.attach_live(runs, [proc(-1, 'run_a')])
        if len(claims) != 1:
            bad.append(f't1 one live process claimed {len(claims)} run dirs')
        elif os.path.basename(next(iter(claims))) != 'run_a':
            bad.append(f't1 claimed {next(iter(claims))!r}, not run_a')

        # Two trainings started WITHOUT name= -- how ultralytics' default
        # train/train2 runs happen. Neither command line names a directory, so
        # both score every candidate identically and both pick the same one.
        # Only exclusivity separates them; without it the two live runs
        # collapse to one and the other vanishes from the page.
        two = tt.attach_live(runs, [proc(-1, None), proc(-2, None)])
        if len(two) != 2:
            bad.append(f't1 two unnamed live processes resolved to '
                       f'{sorted(os.path.basename(d) for d in two)}')

        # t4 -- none of them wrote an epoch
        states = {r['name']: r['status']
                  for r in tt.collect(os.path.join(tmp, 'root'))}
        wrong = {k: v for k, v in states.items()
                 if v not in ('never_started', 'running')}
        if wrong:
            bad.append(f't4 epoch-less runs reported as {wrong}')

        # t7 the detect fitness formula CHANGED in ultralytics 8.4. Keyed on
        # the wrong one, train-22 was reported early-stopped at best@248 when
        # it ran its full 300-epoch budget and peaked at 262.
        det = [{'epoch': i + 1, 'metrics/mAP50(B)': m50,
                'metrics/mAP50-95(B)': m95}
               for i, (m50, m95) in enumerate([(0.90, 0.40), (0.50, 0.41)])]
        if tt.best_index(det, 'detect', tt.DET_W_84) != 1:
            bad.append('t7 the 8.4 fitness (mAP50-95 alone) picked the wrong '
                       'epoch')
        if tt.best_index(det, 'detect', tt.DET_W_LEGACY) != 0:
            bad.append('t7 the <=8.3 fitness (0.1/0.9) picked the wrong epoch')
        if tt.det_weights('8.4.115') != tt.DET_W_84 or \
                tt.det_weights('8.3.165') != tt.DET_W_LEGACY:
            bad.append(f't7 version->weights mapping is wrong: '
                       f'8.4.115->{tt.det_weights("8.4.115")} '
                       f'8.3.165->{tt.det_weights("8.3.165")}')
        # ultralytics keeps replacing the best while fitness is still 0, so a
        # run that never leaves zero has its best at the LAST epoch
        zero = [{'epoch': i + 1, 'metrics/mAP50(B)': 0.0,
                 'metrics/mAP50-95(B)': 0.0} for i in range(5)]
        if tt.best_index(zero, 'detect') != 4:
            bad.append('t7 the best_fitness==0 clause is not reproduced')

        # t9 each latest-epoch tile shows the peak of ITS OWN metric. Taking
        # the value at the best-FITNESS epoch instead would attribute one
        # metric's peak to another metric's epoch -- a wrong number that sits
        # in the right range and matches on any run where they coincide.
        rows9 = [{'epoch': 1, 'metrics/recall(B)': 0.90,
                  'metrics/mAP50-95(B)': 0.30},
                 {'epoch': 2, 'metrics/recall(B)': 0.40,
                  'metrics/mAP50-95(B)': 0.50}]
        got9 = {m['key']: m for m in tt.latest_metrics(rows9, 'detect')}
        # fitness peaks at epoch 2, but recall peaked at epoch 1
        if (round(got9['recall']['latest'], 4),
                round(got9['recall']['peak'], 4)) != (0.40, 0.90):
            bad.append(f't9 recall latest/peak wrong: {got9["recall"]}')
        if got9['recall']['peak_epoch'] != 1:
            bad.append(f't9 recall peak_epoch wrong: '
                       f'{got9["recall"]["peak_epoch"]}')
        if (round(got9['mAP50-95']['latest'], 4),
                round(got9['mAP50-95']['peak'], 4)) != (0.50, 0.50):
            bad.append(f't9 mAP50-95 latest/peak wrong: {got9["mAP50-95"]}')
        # the card draws the metric's own history, so it must come back with it
        if got9['recall']['series'] != [0.90, 0.40]:
            bad.append(f't9 series not carried: {got9["recall"]["series"]}')
        # and the list itself is the short one, on purpose
        if list(got9) != ['mAP50-95', 'recall']:
            bad.append(f't9 unexpected latest metric set: {list(got9)}')

        # t8 a run directory called "train" is ultralytics' DEFAULT name, and
        # "train" is also a dataset split -- pruning by name alone hid it
        d8 = os.path.join(tmp, 'named')
        for name in ('train', 'val', 'my_run'):
            os.makedirs(os.path.join(d8, 'runs', 'detect', name))
            with open(os.path.join(d8, 'runs', 'detect', name,
                                   'args.yaml'), 'w') as fh:
                fh.write(f'name: {name}\nproject: p\n')
        os.makedirs(os.path.join(d8, 'dataset', 'images', 'train'))
        got8 = sorted(r['name'] for r in tt.discover(d8))
        if got8 != ['my_run', 'train', 'val']:
            bad.append(f't8 discovery dropped default-named runs: {got8}')

        # t10 one project written two ways is one project. Left alone,
        # DogDetection and dogdetection split 42 runs across two headings.
        # through collect(), not by calling canon_projects directly: testing
        # the function alone leaves the WIRING uncovered, and this check
        # stayed green with the fold removed from collect().
        d10 = os.path.join(tmp, 'cased')
        for proj, name in (('DogDetection', 'a'), ('dogdetection', 'b'),
                           ('dogdetection', 'c')):
            rd = os.path.join(d10, proj, name)
            os.makedirs(rd, exist_ok=True)
            with open(os.path.join(rd, 'args.yaml'), 'w') as fh:
                fh.write(f'name: {name}\nproject: {proj}\n')
        got10 = {r['project'] for r in
                 tt.collect(d10, registry={'projects': {'dogdetection': {}}})}
        if got10 != {'dogdetection'}:
            bad.append(f't10 case variants not folded by collect(): {got10}')
        # with no registry the more-used spelling still wins
        got10b = {r['project'] for r in tt.collect(d10)}
        if got10b != {'dogdetection'}:
            bad.append(f't10 majority spelling not chosen: {got10b}')

        # t11 <runs_dir>/detect/<name> means NO project was set; "detect" is
        # the task, and printing it as a project invents one
        d11 = os.path.join(tmp, 'noproj', 'runs', 'detect', 'train')
        os.makedirs(d11)
        with open(os.path.join(d11, 'args.yaml'), 'w') as fh:
            fh.write('name: train\n')
        got11 = [r['project'] for r in
                 tt.discover(os.path.join(tmp, 'noproj'))]
        if got11 != ['(no project)']:
            bad.append(f't11 task folder read as a project: {got11}')

    if bad:
        raise SystemExit('training tracker:\n  ' + '\n  '.join(bad))
    print('ok   training tracker: live claim is exclusive and save_dir-aware, '
          'fitness matches the run ultralytics version, default-named runs '
          'are found')


# Classes every section is entitled to wear: the section wrapper idiom, the
# shared components, and the state modifiers. Everything else a section uses
# should belong to it alone.
SHARED_CLASSES = {
    'panel', 'sect', 'fold', 'sec', 'phead', 'phint', 'chart', 'cards',
    'ok', 'bad', 'warn', 'dim', 'mnone', 'cnt', 'hint', 'bar', 'fill',
    'rbtn', 'quiet', 'danger', 'bico', 'sp', 'num', 'pill', 'tag',
    # the KPI card and its sparkline underlay: every panel that reports a
    # number uses them, which is the point -- one readout shape, page-wide
    'kpis', 'kpi', 'kpi-label', 'kpi-val', 'spk', 'dspark', 'hot', 'lead',
}
# A section whose markup is expected to keep to its own prefix. The whole page
# shares one stylesheet, so a section that invents a class already in use
# silently restyles somebody else's markup: `.dv` was the sweep panel's
# percentage span, and a drive-card rule of the same name put a border and a
# panel background around every percentage on the dashboard. Nothing failed --
# the page was valid, rendered, and wrong.
#
# Naming the prefix is the cheap half of the fix. Add a section, add a line.
SECTION_PREFIX = {'f-drives': 'dh', 'f-sys': 'sy'}


def css_collisions(index_path):
    """[(class, section)] the section wears that are neither its own nor shared."""
    import re as _re
    h = open(index_path).read()
    body = h[h.index('</style>'):]
    out = []
    for sid, prefix in SECTION_PREFIX.items():
        i = body.find(f'id="{sid}"')
        if i < 0:
            continue
        j = body.find('</details>', i)
        seg = body[i:j if j > 0 else len(body)]
        mine = {w for a in _re.findall(r'class="([^"]+)"', seg)
                for w in a.split()}
        for c in sorted(mine - SHARED_CLASSES):
            if not c.startswith(prefix):
                out.append((c, sid))
    return out


GATE_STUB = r'''
var els = {};
function E(id){
  return els[id] || (els[id] = {
    id: id, textContent: '', title: '', className: '', disabled: false,
    dataset: {}, style: {}, hidden: false,
    addEventListener: function(){}, closest: function(){ return null },
  });
}
global.document = {
  getElementById: E, hidden: false,
  addEventListener: function(){},
  createElement: function(){ return {style:{}, setAttribute:function(){},
    appendChild:function(){}, select:function(){},
    setSelectionRange:function(){}} },
};
global.window = { addEventListener: function(){}, __stage: 'gate' };
global.setInterval = function(){ return 1 };
global.clearInterval = function(){};
global.setTimeout = function(){ return 1 };
global.clearTimeout = function(){};
global.fetch = function(){ return {then:function(){return {catch:function(){
  return {then:function(){}} }} }} };
global.confirm = function(){ return true };
'''


def check_gate_panel(html, snips):
    """Drive the classifier panel over every state it can be handed.

    It serves two stages now -- the gate and the leash model -- and the second
    spends its whole life before it is planned in a state the first never had:
    no plan, no totals, no shards. fmt(undefined) is the string "NaN", so that
    state rendered six NaNs across the cards, which reads as a broken panel
    rather than one with nothing to do yet.
    """
    helpers, _, _, gate = snips
    if 'gateSpark' not in gate or 'api/gate' not in gate:
        print('FAIL the gate IIFE was not extracted — this check ran on '
              'nothing')
        return 1
    payloads = {
        # the leash stage before the gate has finished: no plan at all
        'leash_waiting': {
            'ever': False, 'stage': 'leash', 'planned': False,
            'running': False, 'can_run': False,
            'upstream': {'stage': 'gate', 'title': 'dog-bin gate',
                         'rows': 1515214, 'total': 4688510, 'ready': False}},
        # planned, never started
        'leash_ready': {
            'ever': True, 'stage': 'leash', 'planned': True, 'running': False,
            'can_run': True, 'rows': 0, 'total': 863000, 'shards': 0,
            'pct': 0, 'dog_share': None, 'dogs': None, 'dogs_of': None,
            'rate': 0, 'sustained': 0, 'eta_s': None, 'model': 'leash model',
            'images': 700000, 'images_done': None, 'img_s': None,
            'created': '2026-08-08 06:00:00',
            'upstream': {'stage': 'gate', 'title': 'dog-bin gate',
                         'rows': 4688510, 'total': 4688510, 'ready': True}},
        # upstream done, but no work list yet: this is a DIFFERENT button
        # doing a different thing, and saying Run while quietly planning
        # would misdescribe the click
        'leash_needs_plan': {
            'ever': False, 'stage': 'leash', 'planned': False,
            'running': False, 'planning': False, 'can_run': True,
            'upstream': {'stage': 'gate', 'title': 'dog-bin gate',
                         'rows': 4688510, 'total': 4688510, 'ready': True}},
        'leash_planning': {
            'ever': False, 'stage': 'leash', 'planned': False,
            'running': False, 'planning': True, 'can_run': True,
            'upstream': {'stage': 'gate', 'title': 'dog-bin gate',
                         'rows': 4688510, 'total': 4688510, 'ready': True}},
        'gate_running': {
            'ever': True, 'stage': 'gate', 'planned': True, 'running': True,
            'can_run': True, 'rows': 1530446, 'total': 4688510, 'shards': 52,
            'pct': 0.3264, 'dog_share': 0.184, 'dogs': 278874,
            'dogs_of': 1515214, 'rate': 66.4, 'sustained': 66.2,
            'eta_s': 47798.5, 'model': 'dog-bin gate', 'images': 3292062,
            'images_done': 1045505, 'img_s': 45.6,
            'created': '2026-08-07 22:28:52', 'upstream': None},
        'null_response': None,
        # what the route returns when gate_progress throws: not a progress
        # document, so nothing on the cards may be read as a measurement
        'server_error': {'ever': False, 'error': 'disk went away'},
        'empty': {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        js = os.path.join(tmp, 'gate.js')
        with open(js, 'w', encoding='utf-8') as f:
            # the IIFE keeps paint() private, so its body is re-evaluated
            # with paint returned -- the same trick the detect panel's check
            # uses, and the reason both are checked as bodies rather than
            # reimplemented
            body = re.sub(r'\(function\(\)\{', '', gate, count=1)
            body = re.sub(r'\}\)\(\);?\s*$', '', body)
            f.write(GATE_STUB + helpers + '\n'
                    + 'var paint = (function(){' + body
                    + '\nreturn paint})();\n' + r'''
var P = JSON.parse(process.argv[2]), bad = [];
for (var name in P) {
  for (var k in els) { els[k].textContent = ''; els[k].title = '' }
  window.__stage = (P[name] && P[name].stage) || 'gate';
  try { paint(P[name]) }
  catch (e) { bad.push(name + ': THREW ' + e.message); continue }
  var seen = [];
  for (var k in els)
    seen.push(String(els[k].textContent) + ' ' + String(els[k].title));
  var all = seen.join(' | ');
  if (/NaN|undefined|\[object/.test(all))
    bad.push(name + ': junk on the cards -> ' + all.slice(0, 300));
}
// the states that carry no numbers must SAY so, not sit blank
var w = {}; for (var k in els) w[k] = 0;
for (var k in els) { els[k].textContent = ''; els[k].title = '' }
window.__stage = 'leash'; paint(P.leash_waiting);
// The panel's own fmt() maps an absent number to 0, not to NaN -- so a
// stage that has never been planned rendered "0 judged, 0 of 0, 0%", which
// is indistinguishable from a planned stage that has done nothing yet. It
// has to read as UNKNOWN, which is a dash.
['gDone', 'gPct', 'gNow', 'gSus', 'gDog'].forEach(function (id) {
  if (String(els[id].textContent) !== '\u2014')
    bad.push('unplanned stage shows ' + id + ' = '
      + JSON.stringify(els[id].textContent) + ', expected a dash — a zero '
      + 'here is a claim that the stage has judged nothing, not that it has '
      + 'not been asked to');
});
if (/\b0 of 0\b/.test(String(els.gCount.textContent)))
  bad.push('unplanned stage counts against a total it does not have: '
    + JSON.stringify(els.gCount.textContent));
if (!/waiting on the dog-bin gate/.test(String(els.gMeta.textContent)))
  bad.push('a stage waiting on another does not say so: '
    + JSON.stringify(els.gMeta.textContent));
if (String(els.gateState.textContent) !== 'waiting')
  bad.push('pill reads ' + JSON.stringify(els.gateState.textContent)
    + ', expected "waiting"');
if (!els.gateBtn.disabled)
  bad.push('Run is offered for a stage that cannot be planned yet');
for (var k in els) { els[k].textContent = ''; els[k].title = '' }
paint(P.leash_ready);
if (els.gateBtn.disabled || !/Run leash/.test(String(els.gateBtn.textContent)))
  bad.push('a planned leash stage is not offered a Run: '
    + JSON.stringify(els.gateBtn.textContent));
for (var k in els) { els[k].textContent = ''; els[k].title = '' }
paint(P.server_error);
if (/\b0 of 0\b/.test(String(els.gCount.textContent))
    || String(els.gDone.textContent) !== '\u2014')
  bad.push('an error response renders as a measured zero: '
    + JSON.stringify(els.gCount.textContent) + ' / '
    + JSON.stringify(els.gDone.textContent));
if (!/disk went away/.test(String(els.gMeta.textContent)))
  bad.push('the error itself is not shown: '
    + JSON.stringify(els.gMeta.textContent));
for (var k in els) { els[k].textContent = ''; els[k].title = '' }
paint(P.leash_needs_plan);
if (els.gateBtn.disabled || !/Plan leash/.test(String(els.gateBtn.textContent)))
  bad.push('an unplanned stage whose upstream IS done offers '
    + JSON.stringify(els.gateBtn.textContent) + ' — it needs a work list '
    + 'built before it can run, and the button has to say which it is doing');
for (var k in els) { els[k].textContent = ''; els[k].title = '' }
paint(P.leash_planning);
if (!els.gateBtn.disabled
    || !/Planning/.test(String(els.gateBtn.textContent)))
  bad.push('a planner already running is still offered a button: '
    + JSON.stringify(els.gateBtn.textContent));
if (String(els.gateState.textContent) !== 'planning')
  bad.push('pill during planning reads '
    + JSON.stringify(els.gateState.textContent));
if (bad.length) { bad.forEach(function(b){ console.log('FAIL ' + b) });
  process.exit(1) }
console.log('ok   gate panel: two stages, and an unplanned one says so');
''')
        r = subprocess.run(['node', js, json.dumps(payloads)],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        return r.returncode


def check_run_diff():
    """A comparison must subtract in the direction it is written.

    _delta computed B minus A while the table is headed "A vs B", so a run
    scoring 0.3699 against a promoted 0.4968 reported +0.1270 in the colour of
    a win. Every regression read as an improvement, on the one screen someone
    uses to decide what to promote.

    And it must show the metrics the model is SHIPPED on. This detector is
    selected on mAP and lives or dies on recall; a diff that omits recall
    cannot answer the question a retrain was run to answer.
    """
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    try:
        import dashboard as d
    except ImportError as e:
        print(f'SKIP: cannot import the dashboard ({e})')
        return 0
    bad = []
    # direction: A worse than B must render as a loss, in the losing colour
    worse = d._delta(0.3699, 0.4968, 4, True, lambda v: f'{v:.4f}')
    if 'ddn' not in worse or '&minus;' not in worse:
        bad.append(f'A scoring below B rendered as {worse!r} — a regression '
                   f'must not read as a gain')
    better = d._delta(0.60, 0.50, 4, True, lambda v: f'{v:.4f}')
    if 'dup' not in better or '+' not in better:
        bad.append(f'A scoring above B rendered as {better!r}')
    # lower-is-better rows invert the colour, not the sign
    slower = d._delta(300.0, 180.0, 1, False, None)
    if 'ddn' not in slower or '+' not in slower:
        bad.append(f'a slower run rendered as {slower!r} — the sign says '
                   f'which way it moved, the colour says whether that is good')
    # the rows that matter are present
    runs = d.training_runs()
    det = [r for r in runs if r.get('task') == 'detect'
           and (r.get('latest') or [])]
    if len(det) < 2:
        print('SKIP: fewer than two detector runs on disk')
    else:
        html = d.render_run_diff(f"{det[0]['project']}/{det[0]['name']}",
                                 f"{det[1]['project']}/{det[1]['name']}")
        for want in ('recall at best epoch', 'precision at best epoch',
                     'mAP50 at best epoch'):
            if want not in html:
                bad.append(f'the comparison omits "{want}" — a detector is '
                           f'promoted on mAP and shipped on recall')
        # and those rows carry the value at the promoted checkpoint, not the
        # peak of a metric at some epoch nobody would ship
        for r in det[:2]:
            for m in r['latest']:
                if m['key'] == 'recall' and m.get('at_best') is None:
                    bad.append(f"{r['name']}: recall has no value at the best "
                               f"epoch, so the diff would fall back to a peak "
                               f"belonging to a different checkpoint")
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('ok   run comparison: subtracts A from B, and shows what the model '
          'is shipped on')
    return 0


def check_progress_ramp():
    """One quantity, one ramp -- in both languages that draw it.

    A completion bar is rendered twice: once into the static page by
    bar_color() in Python, and once live by pctColor() in the page script.
    They had already drifted -- same thresholds, but one finished in #3fb27f
    and the other in #43b581 -- which is invisible until the two appear on
    one screen, and then reads as two different kinds of "done".

    The ramp itself has one property that has to hold: lightness climbing
    monotonically. That is what makes a magnitude readable at all, and it is
    the channel every kind of colour blindness keeps, which is why this ramp
    replaced seven scattered hues.
    """
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    try:
        import dashboard as d
    except ImportError as e:
        print(f'SKIP: cannot import the dashboard ({e})')
        return 0
    bad = []
    src = inspect.getsource(d)
    m = re.search(r"function pctColor\(p\)\{(.*?)\n(?=[a-zA-Z/])", src, re.S)
    js = re.findall(r"#[0-9a-fA-F]{6}", m.group(1)) if m else []
    if not js:
        bad.append('pctColor() not found in the page script')
    else:
        # js is [done, ...ramp descending by threshold]; compare as sets of
        # the ramp steps plus the terminal colour
        py = list(d.PROGRESS_RAMP) + [d.bar_color(100)]
        if sorted(set(x.lower() for x in js)) != sorted(set(x.lower() for x in py)):
            bad.append(f'the two ramps disagree: page script uses {js}, '
                       f'bar_color() uses {py} — one bar, two scales')
    # every 5% step must be non-decreasing in lightness, and 100 must differ
    def lum(h):
        def c(v):
            v = int(h[v:v + 2], 16) / 255
            return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
        r, g, b = c(1), c(3), c(5)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    steps = [lum(d.bar_color(p)) for p in range(0, 100, 5)]
    if any(b < a - 1e-9 for a, b in zip(steps, steps[1:])):
        bad.append(f'the ramp is not monotonic in lightness: '
                   f'{[round(x, 3) for x in steps]} — a magnitude that gets '
                   f'darker as it grows cannot be read')
    if d.bar_color(100) == d.bar_color(99):
        bad.append('finished is painted the same as nearly-finished')
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('ok   progress ramp: one scale in both languages, rising in '
          'lightness')
    return 0


def check_machine_stats():
    """The machine panel reports live figures, so every one of them is a way
    to be confidently wrong."""
    import subprocess as _sp
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    try:
        import dashboard as d
    except ImportError as e:
        print(f'SKIP: cannot import the dashboard ({e})')
        return 0
    bad = []

    # /proc/stat holds counters since BOOT. The first read has nothing to
    # subtract from, and the delta against zero is the machine's lifetime
    # average -- a small, plausible number that is not about now.
    d._CPU.update(t=0.0, idle=0.0, total=0.0, pct=None)
    if d._cpu_pct() is not None:
        bad.append('the first cpu reading is the since-boot average, not a '
                   'measurement of now — it must report unknown')
    if any(d._cpu_pct() is not None for _ in range(3)):
        bad.append('cpu re-sampled within the second — the window would be '
                   'microseconds wide and one scheduler tick reads as 100%')
    time.sleep(1.1)
    v = d._cpu_pct()
    if v is None or not 0.0 <= v <= 100.0:
        bad.append(f'cpu reading {v} is not a percentage')
    # and the cadence is the server's, not the audience's: two tabs polling
    # must not each consume the delta
    held = [d._cpu_pct() for _ in range(4)]
    if len(set(held)) != 1:
        bad.append(f'four reads inside one second gave {held} — each client '
                   f'is consuming the delta')

    m = d._meminfo()
    if not m or not m.get('MemTotal'):
        bad.append('meminfo unreadable')
    elif not 0 < m['MemAvailable'] <= m['MemTotal']:
        bad.append(f'available {m["MemAvailable"]} vs total {m["MemTotal"]}')

    # ── the card ────────────────────────────────────────────────────────────
    real = _sp.Popen

    def reset():
        d._GPU.update(proc=None, samples=None, meta={}, retry_at=0.0)

    class FakeSmi:
        """One nvidia-smi -l 1, as a finite stream of rows."""

        def __init__(self, rows):
            self.stdout = iter(rows)

    def feed(rows):
        reset()
        d.subprocess.Popen = lambda *a, **k: FakeSmi(rows)
        try:
            d._gpu()                            # spawns and drains the reader
            for _ in range(100):                # the reader is a thread
                if d._GPU['samples'] and len(d._GPU['samples']) >= len(rows):
                    break
                time.sleep(0.01)
            return d._gpu()
        finally:
            d.subprocess.Popen = real

    # A single reading is not a measurement of a bursty workload. Measured on
    # a real run: forty zeroes and four bursts in twenty-two seconds, so a
    # glance taken when a browser asks lands on zero nine times in ten and the
    # card reads 0% forever. The headline has to be the window.
    burst = ['NVIDIA X, 0, 4103, 16303, 33, 40, 360'] * 9 + \
            ['NVIDIA X, 63, 4103, 16303, 35, 190, 360']
    g = feed(burst) or {}
    if g.get('util') is None or abs(g['util'] - 6.3) > 0.01:
        bad.append(f'gpu utilisation is {g.get("util")}, expected the 6.3% '
                   f'mean of the window — a point sample of this workload is '
                   f'0% nine times in ten')
    if g.get('util_peak') != 63:
        bad.append(f'peak is {g.get("util_peak")}, expected 63 — a mean of 6% '
                   f'from bursts of 63% is a different machine from a flat 6%')

    # No card, no driver, a binary that is not there: one unknown figure.
    for boom in (FileNotFoundError('nvidia-smi'), OSError('no such device'),
                 ValueError('bad argv')):
        reset()
        d.subprocess.Popen = lambda *a, **k: (_ for _ in ()).throw(boom)
        try:
            if d._gpu() is not None:
                bad.append(f'{type(boom).__name__} still produced a card')
            got = d.sys_stats()
            if got['gpu'] is not None or got['mem_total'] <= 0:
                bad.append(f'{type(boom).__name__} broke the rest of the panel')
            # and it must not respawn on every request for the rest of time
            if d._GPU['retry_at'] <= time.time():
                bad.append(f'{type(boom).__name__} left no backoff — a box '
                           f'with no card would fork nvidia-smi every 2 s')
        except Exception as e:                 # noqa: BLE001 - that is the test
            bad.append(f'{type(boom).__name__} propagated: {e}')
        finally:
            d.subprocess.Popen = real
    reset()

    # Rows that are not measurements: short, empty, or "[N/A]" throughout.
    g = feed(['n/a, [N/A], , not supported', 'garbage', '', 'a,b'])
    if g and g.get('util') is not None:
        bad.append(f'"[N/A]" read as a measurement: {g}')
    # One field the driver will not answer for (power on a laptop card, temp
    # in a container) must cost that field, not the whole readout.
    g = feed(['Card X, 74, 4103, 16303, 51, [N/A], [N/A]']) or {}
    if g.get('util') != 74 or g.get('mem_total') != 16303:
        bad.append(f'an unsupported power reading took the whole card down '
                   f'with it: {g}')
    if g.get('power') is not None:
        bad.append(f'"[N/A]" power read as a number: {g.get("power")}')
    # A card that goes away mid-run (driver reset, eGPU unplugged) closes the
    # pipe; the panel must notice rather than serve the last frame forever.
    reset()
    if d._GPU['proc'] is not None:
        bad.append('a closed nvidia-smi stream left a live process behind')
    d.subprocess.Popen = real

    s = d.sys_stats()
    for k in ('cpu', 'cores', 'mem_used', 'mem_total', 'swap_used',
              'io_stall', 'gpu'):
        if k not in s:
            bad.append(f'sys_stats() has no {k}')
    if s.get('mem_used', 0) > s.get('mem_total', 0):
        bad.append('memory used exceeds total — MemFree read as MemAvailable?')
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('ok   machine stats: measured windows, and a missing card is one '
          'unknown figure')
    return 0


def hidden_that_still_shows(html):
    """[(selector, element)] the page hides in markup but styles into view.

    The UA's ``[hidden]{display:none}`` is the weakest rule there is: any
    author rule that names a display wins, so ``.swctl{display:flex}`` made
    ``<span class="swctl" hidden>`` fully visible. That put four buttons in
    the section header -- Resume sweep AND Run gate -- when only one stage was
    on screen. This file already carries a dozen hand-written
    ``.x[hidden]{display:none}`` rules for exactly that, which is the tell: it
    is a rule everyone must remember, so nobody does. Checked, not remembered.
    """
    import re as _re
    css = '\n'.join(_re.findall(r'<style[^>]*>(.*?)</style>', html, _re.S))
    css = _re.sub(r'/\*.*?\*/', ' ', css, flags=_re.S)
    shows, hides = set(), set()
    for sel, decls in _re.findall(r'([^{}]+)\{([^{}]*)\}', css):
        disp = _re.search(r'(?:^|;)\s*display\s*:\s*([a-z-]+)', decls)
        for one in sel.split(','):
            one = one.strip()
            if not one:
                continue
            # the last simple-selector sequence is what the element itself
            # must match; ".a .b" styles .b, not .a
            leaf = _re.split(r'[\s>+~]+', one)[-1]
            for tok in _re.findall(r'[.#][A-Za-z0-9_-]+', leaf):
                if '[hidden]' in leaf and (disp and disp.group(1) == 'none'):
                    hides.add(tok)
                elif disp and disp.group(1) != 'none':
                    shows.add(tok)
    if _re.search(r'(?:^|[,}])\s*\[hidden\]\s*\{[^{}]*display\s*:\s*none',
                  css):
        return []                       # a global rule covers everything
    body = html[html.index('</style>'):]
    out = []
    for tag in _re.findall(r'<[a-z][^>]*\shidden(?:=[^>]*)?>', body):
        toks = ['#' + m for m in _re.findall(r'\bid="([^"]+)"', tag)]
        toks += ['.' + c for a in _re.findall(r'\bclass="([^"]+)"', tag)
                 for c in a.split()]
        if any(t in hides for t in toks):
            continue
        for t in toks:
            if t in shows:
                out.append((t, tag[:70]))
                break
    return out


def main():
    if shutil.which('node') is None:
        print('SKIP: node not on PATH — client render test not run')
        return 0
    if not os.path.exists(INDEX):
        raise SystemExit(f'{INDEX} missing — run dashboard.py build first')
    # This whole file judges a BUILT page. Read a stale one and it will
    # cheerfully pass while the source it is supposed to be guarding is
    # broken: that is exactly what happened when an apostrophe escaped as
    # \\' in the non-raw template emitted a bare quote, killed every handler
    # on the page, and this test -- run before the rebuild -- said ok.
    stray = css_collisions(INDEX)
    if stray:
        for c, sid in stray[:8]:
            print(f'FAIL {sid} wears class "{c}", which is neither its own '
                  f'prefix nor a shared component — one stylesheet, so it is '
                  f'styling or being styled by another section')
        return 1
    print('ok   every section keeps to its own class prefix')

    src = os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py')
    if os.path.exists(src) and os.path.getmtime(src) > os.path.getmtime(INDEX):
        raise SystemExit(
            'STALE PAGE: tools/dashboard/dashboard.py is newer than\n'
            f'  {INDEX}\n'
            'so this run would grade the previous build. Rebuild first:\n'
            '  python tools/dashboard/dashboard.py build --no-refresh')
    html = open(INDEX, encoding='utf-8').read()
    showing = hidden_that_still_shows(html)
    if showing:
        for sel, tag in showing[:8]:
            print(f'FAIL {sel} names a display, so [hidden] does not hide it: '
                  f'{tag}\n     add {sel}[hidden]{{display:none}}')
        return 1
    print('ok   every hidden element is actually hidden')
    if check_machine_stats():
        return 1
    if check_progress_ramp():
        return 1
    if check_run_diff():
        return 1
    if check_gate_panel(html, extract_snippets(html)):
        return 1
    check_whole_script(html)
    check_markup(html)
    check_key_metrics()
    check_training_tracker()
    check_flag_api()
    helpers, iife, crops_iife, gate_iife = extract_snippets(html)

    with tempfile.TemporaryDirectory() as tmp:
        payloads = {
            'real_running': real_payload(tmp),
            'not_running': {'running': False},
            'stale': {'running': False, 'stale': True, 'age_s': 500,
                      'state': 'running', 'run_id': 'r', 'pid': 1},
            'terminal': {'running': False, 'state': 'done'},
            # The sweep that FINISHED. Not running, so no rate and no ETA --
            # but 32.5M images were swept and every region is done, and the
            # panel showed six em-dashes for it.
            'finished_idle': {
                'running': False, 'finished': True, 'state': 'stopped',
                'age_s': 4269, 'run_id': 56381, 'gen': 1,
                'imgs_done': 32_542_334, 'imgs_total': 32_542_334,
                'run_imgs_done': 28_022_101,
                # the writer keeps publishing its last window; the client must
                # not believe it once the process is gone
                'img_per_sec': {'w60': 71.8, 'w900': 69.0},
                'drives': {'bobcat': {'done': 4043431, 'total': 4043431},
                           'lynx': {'done': 3654093, 'total': 3654093}},
                'regions': {'Africa': 100.0, 'Europe': 100.0,
                            'South_Asia': 100.0},
                'positives': 3_368_223, 'positive_rate': 10.4,
                'boxes_total': 4_785_890, 'crops_classified': 0,
                'class_split': {}, 'errors': {},
                'started_at': '2026-08-02 20:09:33'},
            # Stopped PART WAY. Same rule, different number: show what was
            # done, and no ETA because nothing is running to have one.
            'paused_idle': {
                'running': False, 'finished': False, 'state': 'stopped',
                'age_s': 900, 'run_id': 7, 'gen': 1,
                'imgs_done': 13_016_933, 'imgs_total': 32_542_334,
                'run_imgs_done': 1_000_000,
                'drives': {'lynx': {'done': 1827046, 'total': 3654093}},
                'regions': {'Africa': 100.0, 'Europe': 40.0,
                            'South_Asia': 0.0},
                'positives': 1_200_000, 'positive_rate': 9.2,
                'boxes_total': 1_500_000, 'crops_classified': 0,
                'class_split': {}, 'errors': {},
                'started_at': '2026-08-05 11:00:00'},
            'degenerate_min': {'running': True, 'state': 'running'},
            # imgs_done is GLOBAL (all-time); run_imgs_done is this process
            # only. The %, the bar and the ETA must come from the former.
            'global_vs_run': {
                'running': True, 'state': 'running', 'run_id': 4799, 'gen': 1,
                'imgs_done': 16_271_167, 'run_imgs_done': 369_333,
                'imgs_total': 32_542_334, 'eta_s': 670791,
                'img_per_sec': {'w60': 50.6, 'w900': 48.0},
                'drives': {'lynx': {'done': 50464, 'total': 3654093,
                                    'rate': 12.5, 'queue_depth': 0,
                                    'stalled': False}},
                'regions': {'Africa': 0.0, 'Australia': 12.5},
                'positives': 353, 'positive_rate': 0.1, 'boxes_total': 513,
                'crops_classified': 0, 'class_split': {}, 'errors': {},
                'started_at': '2026-08-02 05:49:51'},
            'all_nulls': {
                'running': True, 'state': 'running', 'run_id': 'r', 'gen': 1,
                'imgs_done': 0, 'imgs_total': 0,
                'img_per_sec': {'w60': None, 'w900': None}, 'eta_s': None,
                'drives': {'lynx': {'done': 0, 'total': 0, 'rate': None,
                                    'queue_depth': 0, 'stalled': False}},
                'regions': {}, 'positives': 0, 'positive_rate': None,
                'boxes_total': 0, 'boxes_per_img': None,
                'crops_classified': 0, 'class_split': {},
                'not_a_dog_rate': None,
                'not_a_dog_band': {'lo': 7.0, 'hi': 16.0, 'in_band': None},
                'gpu': None,
                'errors': {}, 'last_error': None, 'publish_errors': 0},
            'null_response': None,
        }
        # /api/detect/crops payloads. `has_full` is the new field: it says a
        # full frame (box already drawn) exists in recent_crops/full/, i.e.
        # the tile is clickable. Mixed, all-off and hostile cases below.
        crop_payloads = {
            # a full CROP_CAP payload: an 8-wide grid must show 16, not 24
            'sample': {'crops': [
                {'name': f'17856633{i:05d}_10492200492028{i:02d}_0{70 + i:02d}.jpg',
                 'image_id': f'10492200492028{i:02d}', 'ts': 1785663300000 + i,
                 'conf': round(0.20 + i / 100.0, 2), 'age_s': i,
                 'has_full': True} for i in range(24)],
                'total_last_min': 97, 'pool_n': 100},
            # the reported defect: 12 tiles in an 8-wide grid left a ragged
            # second row of 4 with dead space beside it -> must show 8
            'twelve': {'crops': [
                {'name': f'1785663301{i:03d}_t{i}_09{i}.jpg', 'image_id': f't{i}',
                 'ts': 1785663301000 + i, 'conf': 0.9, 'age_s': i,
                 'has_full': True} for i in range(12)],
                'total_last_min': 12, 'pool_n': 100},
            # fewer crops than one row is wide: show all 5, no dead row
            'short_row': {'crops': [
                {'name': f'1785663302{i:03d}_s{i}_04{i}.jpg', 'image_id': f's{i}',
                 'ts': 1785663302000 + i, 'conf': 0.4, 'age_s': i,
                 'has_full': True} for i in range(5)],
                'total_last_min': 5, 'pool_n': 100},
            'mixed_has_full': {'crops': [
                {'name': f'1785663300{i:03d}_img{i}_08{i}.jpg',
                 'image_id': f'img{i}', 'ts': 1785663300000 + i,
                 'conf': 0.8, 'age_s': 3 * i,
                 'has_full': i % 2 == 0} for i in range(6)],
                'total_last_min': 6, 'pool_n': 100},
            'no_full_at_all': {'crops': [        # writer predates full frames
                {'name': '1785663300000_a_077.jpg', 'image_id': 'a',
                 'ts': 1785663300000, 'conf': 0.77, 'age_s': 4,
                 'has_full': False}],
                'total_last_min': 1, 'pool_n': 100},
            'field_absent': {'crops': [          # older server, no has_full key
                {'name': '1785663300000_b_066.jpg', 'image_id': 'b',
                 'ts': 1785663300000, 'conf': 0.66, 'age_s': 9}],
                'total_last_min': 1, 'pool_n': 100},
            'single_clickable': {'crops': [      # arrows must hide, not wrap-crash
                {'name': '1785663300000_c_099.jpg', 'image_id': 'c',
                 'ts': 1785663300000, 'conf': 0.99, 'age_s': 0,
                 'has_full': True}],
                'total_last_min': 1, 'pool_n': 100},
            'empty': {'crops': [], 'total_last_min': 0, 'pool_n': 100},
            'null_response': None,
            'hostile': {'crops': [
                {'name': '1785663300000_"><script>alert(1)</script>_090.jpg',
                 'image_id': '"><img src=x onerror=alert(1)>',
                 'ts': 1785663300000, 'conf': 0.9, 'age_s': 2,
                 'has_full': True}],
                'total_last_min': 1, 'pool_n': 100},
            'degenerate': {'crops': [            # every value the wrong shape
                {'name': '1785663300000_d_050.jpg', 'image_id': 'd',
                 'conf': None, 'age_s': None, 'has_full': True}],
                'total_last_min': None},
        }
        js = os.path.join(tmp, 'detect_iife.js')
        with open(js, 'w', encoding='utf-8') as f:
            f.write(helpers + '\n' + iife + '\n')
        cjs = os.path.join(tmp, 'crops_iife.js')
        with open(cjs, 'w', encoding='utf-8') as f:
            f.write(helpers + '\n' + crops_iife + '\n')
        runner = os.path.join(tmp, 'runner.js')
        with open(runner, 'w', encoding='utf-8') as f:
            f.write(STUB)
        r = subprocess.run(['node', runner, json.dumps(payloads), js,
                            json.dumps(crop_payloads), cjs],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        return r.returncode


if __name__ == '__main__':
    sys.exit(main())
