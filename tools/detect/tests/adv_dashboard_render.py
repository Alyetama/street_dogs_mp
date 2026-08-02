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

A ReferenceError (helper defined in another scope), a TypeError
(null.toFixed) or any other throw fails the test. Also asserts the on/off
panels toggle correctly and that innerHTML actually gets populated.

Requires node on PATH; skips (exit 0, loud message) if absent.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

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


def extract_snippets(html):
    """Pull the helper fns + the detect IIFE out of the built page."""
    script = html[html.rindex('<script>') + 8:html.rindex('</script>')]
    helpers = []
    for name in ('fmt', 'pctColor', 'esc'):
        m = re.search(r'^function %s\(.*$' % name, script, re.M)
        if not m:
            raise SystemExit(f'helper {name}() not found at top level '
                             f'of the built script — detect IIFE would '
                             f'throw ReferenceError')
        helpers.append(m.group(0))
    start = script.index('/* ── detection sweep panel')
    end = script.index('})();', start) + 5
    return '\n'.join(helpers), script[start:end]


STUB = r"""
'use strict';
const payloads = JSON.parse(process.argv[2]);
let failures = [];

function makeEl(id) {
  return {
    id, style: {}, dataset: {}, open: true, hidden: false,
    _innerHTML: '', textContent: '',
    set innerHTML(v) { this._innerHTML = v; },
    get innerHTML() { return this._innerHTML; },
    classList: { add(){}, remove(){} },
    addEventListener(ev, fn) { (this._h ||= {})[ev] = fn; },
    querySelectorAll() { return []; },
    querySelector() { return null; },
    setAttribute(){}, appendChild(){},
  };
}
const els = {};
global.document = {
  getElementById(id) { return els[id] ||= makeEl(id); },
  querySelectorAll() { return []; },
  createElement() { return makeEl('_dyn'); },
  addEventListener(){},
  body: { appendChild(){}, removeChild(){} },
  hidden: false,
};
global.window = { addEventListener(){} };
global.localStorage = { getItem(){ return null; }, setItem(){} };
global.echarts = {
  init() { return { setOption(){}, resize(){} }; },
  getInstanceByDom() { return null; },
};
// fetch is never exercised: we call render() directly per payload.
global.fetch = () => new Promise(() => {});

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
  try {
    render(p);
    const on = els['detOn'], off = els['detOff'];
    if (p && p.running) {
      if (on.style.display !== '') failures.push(name + ': on-panel hidden');
      const head = ['dhPct', 'dhEta', 'dhNow', 'dhSus', 'dhCount']
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
        failures.push(name + ': off-panel not shown');
      if (!off.textContent.includes('sweep idle'))
        failures.push(name + ': missing idle text: ' + off.textContent);
    }
    console.log('ok   ' + name);
  } catch (e) {
    failures.push(name + ': THREW ' + e.constructor.name + ': ' + e.message);
    console.log('FAIL ' + name + ' — ' + e);
  }
}
if (failures.length) { console.log('FAILURES: ' + failures.join(' | ')); process.exit(1); }
console.log('all render cases passed');
"""


def main():
    if shutil.which('node') is None:
        print('SKIP: node not on PATH — client render test not run')
        return 0
    if not os.path.exists(INDEX):
        raise SystemExit(f'{INDEX} missing — run dashboard.py build first')
    html = open(INDEX, encoding='utf-8').read()
    helpers, iife = extract_snippets(html)

    with tempfile.TemporaryDirectory() as tmp:
        payloads = {
            'real_running': real_payload(tmp),
            'not_running': {'running': False},
            'stale': {'running': False, 'stale': True, 'age_s': 500,
                      'state': 'running', 'run_id': 'r', 'pid': 1},
            'terminal': {'running': False, 'state': 'done'},
            'degenerate_min': {'running': True, 'state': 'running'},
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
        js = os.path.join(tmp, 'detect_iife.js')
        with open(js, 'w', encoding='utf-8') as f:
            f.write(helpers + '\n' + iife + '\n')
        runner = os.path.join(tmp, 'runner.js')
        with open(runner, 'w', encoding='utf-8') as f:
            f.write(STUB)
        r = subprocess.run(['node', runner, json.dumps(payloads), js],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        return r.returncode


if __name__ == '__main__':
    sys.exit(main())
