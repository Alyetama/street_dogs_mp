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
    for name in ('fmt', 'pctColor', 'esc'):
        m = re.search(r'^function %s\(.*$' % name, script, re.M)
        if not m:
            raise SystemExit(f'helper {name}() not found at top level '
                             f'of the built script — detect IIFE would '
                             f'throw ReferenceError')
        helpers.append(m.group(0))
    # makeLightbox() is a multi-line top-level helper (LB_JS) that the crops
    # IIFE calls at construction time. Single-line extraction cannot carry it,
    # so take the whole function body by brace-matching from its declaration.
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
            iife('/* ── live detection crops'))


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


def main():
    if shutil.which('node') is None:
        print('SKIP: node not on PATH — client render test not run')
        return 0
    if not os.path.exists(INDEX):
        raise SystemExit(f'{INDEX} missing — run dashboard.py build first')
    html = open(INDEX, encoding='utf-8').read()
    check_whole_script(html)
    check_markup(html)
    check_flag_api()
    helpers, iife, crops_iife = extract_snippets(html)

    with tempfile.TemporaryDirectory() as tmp:
        payloads = {
            'real_running': real_payload(tmp),
            'not_running': {'running': False},
            'stale': {'running': False, 'stale': True, 'age_s': 500,
                      'state': 'running', 'run_id': 'r', 'pid': 1},
            'terminal': {'running': False, 'state': 'done'},
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
